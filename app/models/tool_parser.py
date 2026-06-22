"""
Parse local-HF generated text into app.schema.ToolCall objects.

Supported output formats (the local backend prompts the model to emit the first one):

  {"function_call": {"name": "<tool>", "arguments": {...}}}
  {"name": "<tool>", "arguments": {...}}                       # Qwen-native-ish
  <tool_call> {"name": "...", "arguments": {...}} </tool_call> # Qwen tool tag

Rules:
  * function.arguments is ALWAYS returned as a JSON string (never a dict).
  * <think>...</think> content is stripped from the visible text and from tool parsing,
    but the caller counts it in output tokens.
"""

import json
import re
from typing import List, Optional, Tuple

from app.schema import Function, ToolCall


THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
TOOLTAG_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE)


def split_think(text: str) -> Tuple[str, str]:
    """Return (think_text, visible_text)."""
    if not text:
        return "", ""

    thinks = [t.strip() for t in THINK_RE.findall(text)]
    visible = THINK_RE.sub("", text)

    # Handle an unbalanced/open <think> with no closing tag.
    if "<think>" in visible.lower() and "</think>" not in text.lower():
        idx = visible.lower().find("<think>")
        thinks.append(visible[idx + len("<think>") :].strip())
        visible = visible[:idx]

    think_text = "\n".join(t for t in thinks if t)
    return think_text.strip(), visible.strip()


def _iter_json_candidates(text: str) -> List[str]:
    """Yield top-level balanced {...} substrings, plus anything inside <tool_call> tags."""
    candidates: List[str] = []

    # 1) explicit <tool_call> blocks
    for inner in TOOLTAG_RE.findall(text):
        candidates.append(inner.strip())

    # 2) balanced-brace scan over the remaining text
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    candidates.append(text[start : i + 1])
                    start = -1
    return candidates


def _normalize_arguments(arguments) -> str:
    """Ensure arguments come back as a JSON string."""
    if arguments is None:
        return "{}"
    if isinstance(arguments, str):
        # validate it is JSON; if it is plain text, wrap it
        try:
            json.loads(arguments)
            return arguments
        except json.JSONDecodeError:
            return json.dumps({"input": arguments})
    try:
        return json.dumps(arguments)
    except (TypeError, ValueError):
        return json.dumps({"input": str(arguments)})


def _interpret(obj: dict) -> Optional[Tuple[str, str]]:
    """Map a parsed JSON object to (name, arguments_json_string)."""
    if not isinstance(obj, dict):
        return None

    if "function_call" in obj and isinstance(obj["function_call"], dict):
        fc = obj["function_call"]
        name = fc.get("name")
        args = fc.get("arguments", fc.get("parameters", {}))
    elif "name" in obj and ("arguments" in obj or "parameters" in obj):
        name = obj.get("name")
        args = obj.get("arguments", obj.get("parameters", {}))
    else:
        return None

    if not name:
        return None
    return str(name), _normalize_arguments(args)


def parse_tool_calls(text: str) -> List[ToolCall]:
    """Parse generated text (already think-stripped) into schema ToolCall objects."""
    calls: List[ToolCall] = []
    if not text:
        return calls

    seen = set()
    for raw in _iter_json_candidates(text):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        interpreted = _interpret(obj)
        if not interpreted:
            continue
        name, args = interpreted
        sig = (name, args)
        if sig in seen:
            continue
        seen.add(sig)
        calls.append(
            ToolCall(
                id=f"call_{len(calls)}",
                type="function",
                function=Function(name=name, arguments=args),
            )
        )
    return calls


def strip_tool_json(visible_text: str, calls: List[ToolCall]) -> str:
    """Best-effort removal of the tool-call JSON from the visible content."""
    cleaned = TOOLTAG_RE.sub("", visible_text)
    # Drop balanced JSON objects that were turned into tool calls.
    if calls:
        for raw in _iter_json_candidates(cleaned):
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if _interpret(obj):
                cleaned = cleaned.replace(raw, "")
    return cleaned.strip()


def parse_generated_output(text: str) -> Tuple[str, str, List[ToolCall]]:
    """
    Full pipeline for one generation.

    Returns (think_text, content_text, tool_calls) where:
      think_text   : raw <think> reasoning (counted in output tokens by the caller)
      content_text : visible assistant text with think + tool JSON removed
      tool_calls   : list[app.schema.ToolCall]
    """
    think_text, visible = split_think(text)
    calls = parse_tool_calls(visible)
    content = strip_tool_json(visible, calls)
    return think_text, content, calls
