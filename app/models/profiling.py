"""
Profiling recorder for the API-main / local-HF-sub OpenManus execution path.

One ProfilingRecorder instance is created per full user request (per PlanningFlow.execute).
It is registered as the process-wide "active recorder" so that:

  - app/llm.py        -> records main-LLM and sub-LLM latency / token metrics
  - app/agent/toolcall.py -> records per-tool latency
  - app/models/hf_local.py (indirectly, via the message._profiling payload) -> V-cache /
    influence analysis summaries

Outputs (under {profile_dir}/...):
  record/response/{run_id}.txt          (input + output log; human-readable)
  record/profiling/{run_id}.csv         (per-call token / latency metrics)
  record/tool_compare/{run_id}.csv      (T1 pairwise V-cache L2/cosine)
  record/sensitivity/{run_id}.csv       (T2/T3/T5 per layer×head metrics)
  record/sensitivity/{run_id}.meta.json (model / analysis metadata)

Notes
-----
* T1 "head" = KV head index. T2/T3/T5 "head" = query head index (GQA: num_q_heads >= num_kv_heads).
* kv_group = query_head // (num_attention_heads // num_key_value_heads).
* Full tensors are never stored; only scalar summaries.
"""

import csv
import json
import math
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except Exception:
    np = None

from app.logger import logger


# --------------------------------------------------------------------------------------
# Active-recorder registry
# --------------------------------------------------------------------------------------
_active_recorder: Optional["ProfilingRecorder"] = None
_registry_lock = threading.Lock()


def set_active_recorder(recorder: Optional["ProfilingRecorder"]) -> None:
    global _active_recorder
    with _registry_lock:
        _active_recorder = recorder


def get_active_recorder() -> Optional["ProfilingRecorder"]:
    return _active_recorder


def clear_active_recorder() -> None:
    set_active_recorder(None)


# --------------------------------------------------------------------------------------
# Event model
# --------------------------------------------------------------------------------------
@dataclass
class ProfilingEvent:
    run_id: str
    step_index: Optional[int]
    call_index: Optional[int]
    agent_name: str
    llm_type: str  # "main" or "sub"
    model: str
    selected_tool: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    output_tokens_including_think: int = 0
    llm_latency_sec: float = 0.0
    tool_latency_sec: Optional[float] = None
    tool_success: Optional[bool] = None
    prompt_text: str = ""
    generated_text: str = ""
    # main-call diagnostics (why a plan call produced nothing, etc.)
    finish_reason: Optional[str] = None
    tool_names: Optional[List[str]] = field(default=None)
    # raw input token ids (for input-overlap analysis between compared calls)
    input_token_ids: Optional[List[int]] = field(default=None)
    # {(layer:int, kv_head:int): [float, ...]} mean-pooled V vector per layer/head
    v_cache_summary: Optional[Dict[Tuple[int, int], List[float]]] = field(default=None)


# --------------------------------------------------------------------------------------
# Recorder
# --------------------------------------------------------------------------------------
class ProfilingRecorder:
    def __init__(self, profile_dir: str = "record"):
        self.profile_dir = profile_dir or "record"
        self.run_id: Optional[str] = None
        self.request_text: str = ""
        self.events: List[ProfilingEvent] = []
        self.step_prompts: List[Tuple[Optional[int], str]] = []

        self.current_step_index: Optional[int] = None
        self.current_step_text: str = ""

        self._sub_call_counter: int = 0
        self.current_sub_event: Optional[ProfilingEvent] = None

        # ---- influence analysis (T2/T3/T5) ----------------------------------------
        # Each row: run_id, step_index, call_index, selected_tool, decision_token_pos,
        #           decision_token_id, layer, head, head_type, kv_group, metric_type, value
        self._sensitivity_rows: List[Dict[str, Any]] = []
        # call_index -> {pos, token_id, token_str, selected_tool, step_index}
        self._decision_info: Dict[int, Dict[str, Any]] = {}
        # call_index -> {messages: [...], tool_schema_summary: str}
        self._sub_input_log: Dict[int, Dict[str, Any]] = {}
        # model / analysis metadata for meta.json (set once from hf_local)
        self._hf_model_meta: Optional[Dict[str, Any]] = None

    # ---- lifecycle ----------------------------------------------------------------
    def start_run(self, request_text: str) -> str:
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.request_text = request_text or ""
        logger.info(f"[profiling] started run {self.run_id}")
        return self.run_id

    def start_step(self, step_index: Optional[int], step_text: str) -> None:
        self.current_step_index = step_index
        self.current_step_text = step_text or ""
        self.step_prompts.append((step_index, self.current_step_text))

    # ---- sub-LLM input pre-recording (called before the sub-LLM completes) --------
    def peek_next_call_index(self) -> int:
        """Return the call_index that will be assigned to the next sub-LLM call."""
        return self._sub_call_counter

    def record_sub_llm_input(
        self,
        call_index: int,
        messages: List[dict],
        tool_schema_summary: str = "",
    ) -> None:
        """Store input messages and tool schema for inclusion in response.txt."""
        self._sub_input_log[call_index] = {
            "messages": self._sanitize_messages_for_log(messages),
            "tool_schema_summary": tool_schema_summary or "",
        }

    # ---- model metadata -----------------------------------------------------------
    def set_hf_model_meta(self, meta: Dict[str, Any]) -> None:
        if self._hf_model_meta is None:
            self._hf_model_meta = dict(meta)

    # ---- main / sub LLM events ----------------------------------------------------
    def record_main_llm(
        self,
        agent_name: str,
        model: str,
        latency: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        prompt_text: str = "",
        generated_text: str = "",
        finish_reason: Optional[str] = None,
        tool_names: Optional[List[str]] = None,
    ) -> ProfilingEvent:
        ev = ProfilingEvent(
            run_id=self.run_id,
            step_index=self.current_step_index,
            call_index=None,
            agent_name=agent_name,
            llm_type="main",
            model=model,
            selected_tool=None,
            input_tokens=int(input_tokens or 0),
            output_tokens=int(output_tokens or 0),
            output_tokens_including_think=int(output_tokens or 0),
            llm_latency_sec=float(latency),
            prompt_text=prompt_text or "",
            generated_text=generated_text or "",
            finish_reason=finish_reason,
            tool_names=list(tool_names) if tool_names else None,
        )
        self.events.append(ev)
        return ev

    def record_sub_llm(
        self,
        agent_name: str,
        model: str,
        latency: float,
        profiling: dict,
    ) -> ProfilingEvent:
        profiling = profiling or {}
        call_index = self._sub_call_counter
        self._sub_call_counter += 1

        ev = ProfilingEvent(
            run_id=self.run_id,
            step_index=self.current_step_index,
            call_index=call_index,
            agent_name=agent_name,
            llm_type="sub",
            model=model,
            selected_tool=profiling.get("selected_tool"),
            input_tokens=int(profiling.get("input_tokens", 0) or 0),
            output_tokens=int(profiling.get("output_tokens", 0) or 0),
            output_tokens_including_think=int(
                profiling.get("output_tokens_including_think", 0) or 0
            ),
            llm_latency_sec=float(latency),
            prompt_text=profiling.get("prompt_text", "") or "",
            generated_text=profiling.get("generated_text", "") or "",
            v_cache_summary=profiling.get("v_cache_summary"),
            input_token_ids=profiling.get("input_token_ids"),
        )
        self.events.append(ev)
        self.current_sub_event = ev

        # ---- process influence analysis payload ------------------------------------
        # Model meta (sent once on first hf_local call that runs analysis)
        if profiling.get("hf_model_meta"):
            self.set_hf_model_meta(profiling["hf_model_meta"])

        # Decision token info
        dec_pos = profiling.get("decision_token_pos")
        dec_id = profiling.get("decision_token_id")
        dec_str = profiling.get("decision_token_str", "")
        if dec_pos is not None:
            self._decision_info[call_index] = {
                "pos": dec_pos,
                "token_id": dec_id,
                "token_str": dec_str,
                "selected_tool": profiling.get("selected_tool"),
                "step_index": self.current_step_index,
            }

        # Per-head metric rows (T2/T3/T5)
        for row in (profiling.get("head_metrics") or []):
            self._sensitivity_rows.append(
                {
                    "run_id": self.run_id,
                    "step_index": (
                        self.current_step_index
                        if self.current_step_index is not None
                        else ""
                    ),
                    "call_index": call_index,
                    "selected_tool": profiling.get("selected_tool") or "",
                    "decision_token_pos": dec_pos if dec_pos is not None else "",
                    "decision_token_id": dec_id if dec_id is not None else "",
                    "layer": row.get("layer", ""),
                    "head": row.get("head", ""),
                    "head_type": row.get("head_type", "query"),
                    "kv_group": row.get("kv_group", ""),
                    "metric_type": row.get("metric_type", ""),
                    "value": row.get("value", ""),
                }
            )

        return ev

    # ---- tool latency -------------------------------------------------------------
    def record_tool_latency(
        self, tool_name: str, latency: float, success: bool = True
    ) -> None:
        ev = self.current_sub_event
        if ev is not None and ev.tool_latency_sec is None:
            ev.tool_latency_sec = float(latency)
            ev.tool_success = bool(success)
            if not ev.selected_tool:
                ev.selected_tool = tool_name
            return

        extra = ProfilingEvent(
            run_id=self.run_id,
            step_index=self.current_step_index,
            call_index=ev.call_index if ev else None,
            agent_name=ev.agent_name if ev else "sub",
            llm_type="sub",
            model=ev.model if ev else "",
            selected_tool=tool_name,
            llm_latency_sec=0.0,
            tool_latency_sec=float(latency),
            tool_success=bool(success),
        )
        self.events.append(extra)

    # ---- finalization -------------------------------------------------------------
    def finalize(self) -> None:
        if not self.run_id:
            logger.warning("[profiling] finalize() called without an active run")
            return
        self._write_response_txt()
        self._write_profiling_csv()
        self._write_tool_compare_csv()
        self._write_sensitivity_csv()
        self._write_sensitivity_meta_json()
        logger.info(f"[profiling] finalized run {self.run_id} -> {self.profile_dir}/")

    # ---- writers ------------------------------------------------------------------
    def _path(self, sub: str, ext: str) -> str:
        d = os.path.join(self.profile_dir, sub)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{self.run_id}.{ext}")

    @staticmethod
    def _sanitize_messages_for_log(messages: List[dict]) -> List[dict]:
        """Replace base64/binary content with length annotations; keep all text."""
        out = []
        b64_re = re.compile(r"^data:[^;]+;base64,", re.IGNORECASE)
        for msg in messages:
            if not isinstance(msg, dict):
                out.append(msg)
                continue
            role = msg.get("role", "?")
            content = msg.get("content", "")
            if isinstance(content, list):
                safe = []
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "image_url":
                        url = (c.get("image_url") or {}).get("url", "")
                        safe.append(
                            {"type": "image_url", "[base64_length]": len(url)}
                        )
                    else:
                        safe.append(c)
                out.append({"role": role, "content": safe})
            elif isinstance(content, str) and b64_re.match(content):
                out.append({"role": role, "content": f"[base64 length={len(content)}]"})
            else:
                out.append({"role": role, "content": content})
        return out

    def _write_response_txt(self) -> None:
        path = self._path("response", "txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"=== RUN {self.run_id} ===\n\n")

            # ---- user request --------------------------------------------------------
            f.write("--- USER REQUEST ---\n")
            f.write(self.request_text.strip() + "\n\n")

            # ---- step tasks (indexed) ------------------------------------------------
            if self.step_prompts:
                for step_idx, step_text in self.step_prompts:
                    f.write(f"--- STEP {step_idx} TASK ---\n")
                    f.write(step_text.strip() + "\n\n")

            # ---- LLM calls in order --------------------------------------------------
            written_sub_inputs: set = set()
            for ev in self.events:
                # skip tool-only auxiliary events
                if (
                    ev.llm_type == "sub"
                    and ev.llm_latency_sec == 0.0
                    and not ev.generated_text
                ):
                    continue

                if ev.llm_type == "sub":
                    call_idx = ev.call_index

                    # sub-LLM INPUT
                    if call_idx not in written_sub_inputs:
                        written_sub_inputs.add(call_idx)
                        f.write(f"--- SUB-LLM INPUT (call {call_idx}) ---\n")
                        inp = self._sub_input_log.get(call_idx)
                        if inp:
                            for msg in inp.get("messages", []):
                                role = msg.get("role", "?")
                                content = msg.get("content", "")
                                f.write(f"[{role}]\n")
                                if isinstance(content, list):
                                    for c in content:
                                        if isinstance(c, dict):
                                            if c.get("type") == "image_url":
                                                f.write(
                                                    f"  [image base64 length={c.get('[base64_length]', '?')}]\n"
                                                )
                                            else:
                                                for line in str(
                                                    c.get("text", str(c))
                                                ).splitlines():
                                                    f.write("  " + line + "\n")
                                else:
                                    for line in str(content).splitlines():
                                        f.write("  " + line + "\n")
                            ts = inp.get("tool_schema_summary", "")
                            if ts:
                                f.write(f"[tools] {ts}\n")
                        f.write("\n")

                    # sub-LLM OUTPUT
                    f.write(f"--- SUB-LLM OUTPUT (call {call_idx}) ---\n")
                    f.write(f"selected_tool: {ev.selected_tool or '(none)'}\n")
                    dec = self._decision_info.get(call_idx)
                    if dec:
                        f.write(
                            f"decision_token_pos={dec.get('pos')}  "
                            f"token_id={dec.get('token_id')}  "
                            f"token_str={dec.get('token_str', '')!r}\n"
                        )
                    if ev.generated_text:
                        f.write("generated (raw, incl. <think>):\n")
                        for line in ev.generated_text.splitlines():
                            f.write("  " + line + "\n")
                    f.write(
                        f"[latency={ev.llm_latency_sec:.3f}s "
                        f"in={ev.input_tokens} out={ev.output_tokens} "
                        f"out+think={ev.output_tokens_including_think} "
                        f"tool_lat={ev.tool_latency_sec}]\n\n"
                    )

                else:  # main LLM
                    f.write(
                        f"--- MAIN-LLM (step={ev.step_index} call={ev.call_index}) ---\n"
                    )
                    f.write(
                        f"model={ev.model} in={ev.input_tokens} out={ev.output_tokens} "
                        f"llm={ev.llm_latency_sec:.3f}s\n"
                        f"finish_reason={ev.finish_reason} tool_calls={ev.tool_names or []}\n"
                    )
                    if ev.finish_reason == "length" and not ev.tool_names:
                        f.write(
                            "** WARNING: hit output-token ceiling before emitting a tool call. "
                            "Increase max_tokens. **\n"
                        )
                    if ev.generated_text:
                        f.write("generated:\n")
                        for line in ev.generated_text.splitlines():
                            f.write("  " + line + "\n")
                    f.write("\n")

    def _write_profiling_csv(self) -> None:
        path = self._path("profiling", "csv")
        columns = [
            "run_id",
            "step_index",
            "call_index",
            "agent_name",
            "llm_type",
            "model",
            "selected_tool",
            "input_tokens",
            "output_tokens",
            "output_tokens_including_think",
            "llm_latency_sec",
            "tool_latency_sec",
            "finish_reason",
        ]
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(columns)
            for ev in self.events:
                w.writerow(
                    [
                        ev.run_id,
                        ev.step_index if ev.step_index is not None else "",
                        ev.call_index if ev.call_index is not None else "",
                        ev.agent_name,
                        ev.llm_type,
                        ev.model,
                        ev.selected_tool or "",
                        ev.input_tokens,
                        ev.output_tokens,
                        ev.output_tokens_including_think,
                        f"{ev.llm_latency_sec:.6f}",
                        ""
                        if ev.tool_latency_sec is None
                        else f"{ev.tool_latency_sec:.6f}",
                        ev.finish_reason or "",
                    ]
                )

    def _write_tool_compare_csv(self) -> None:
        path = self._path("tool_compare", "csv")
        cache_events = [
            ev for ev in self.events if ev.llm_type == "sub" and ev.v_cache_summary
        ]
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "run_id",
                    "tool_a",
                    "call_a",
                    "tool_b",
                    "call_b",
                    "layer",
                    "head",  # kv_head index (see meta for head type)
                    "l2",
                    "cosine",
                    "input_tokens_a",
                    "input_tokens_b",
                    "overlap_tokens",
                    "overlap_pct_a",
                    "overlap_pct_b",
                ]
            )
            for i in range(len(cache_events)):
                for j in range(i + 1, len(cache_events)):
                    a = cache_events[i]
                    b = cache_events[j]

                    na = len(a.input_token_ids) if a.input_token_ids else 0
                    nb = len(b.input_token_ids) if b.input_token_ids else 0
                    if na and nb:
                        overlap = self._contiguous_overlap(
                            a.input_token_ids, b.input_token_ids, min_run=3
                        )
                        pct_a = f"{100.0 * overlap / na:.2f}"
                        pct_b = f"{100.0 * overlap / nb:.2f}"
                        overlap_s = str(overlap)
                    else:
                        overlap_s = pct_a = pct_b = ""

                    common = set(a.v_cache_summary.keys()) & set(b.v_cache_summary.keys())
                    for layer, head in sorted(common):
                        l2, cos = self._compare_vectors(
                            a.v_cache_summary[(layer, head)],
                            b.v_cache_summary[(layer, head)],
                        )
                        w.writerow(
                            [
                                self.run_id,
                                a.selected_tool or "",
                                a.call_index,
                                b.selected_tool or "",
                                b.call_index,
                                layer,
                                head,
                                f"{l2:.6f}",
                                f"{cos:.6f}",
                                na,
                                nb,
                                overlap_s,
                                pct_a,
                                pct_b,
                            ]
                        )

    def _write_sensitivity_csv(self) -> None:
        """Write T2/T3/T5 head-level influence metrics in long format."""
        if not self._sensitivity_rows:
            return
        path = self._path("sensitivity", "csv")
        columns = [
            "run_id",
            "step_index",
            "call_index",
            "selected_tool",
            "decision_token_pos",
            "decision_token_id",
            "layer",
            "head",
            "head_type",
            "kv_group",
            "metric_type",
            "value",
        ]
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(columns)
            for row in self._sensitivity_rows:
                w.writerow([row.get(c, "") for c in columns])

    def _write_sensitivity_meta_json(self) -> None:
        """Write model / analysis metadata alongside the sensitivity CSV."""
        if not self._sensitivity_rows and self._hf_model_meta is None:
            return
        path = self._path("sensitivity", "meta.json")
        meta = dict(self._hf_model_meta or {})
        meta.setdefault("run_id", self.run_id)
        meta.setdefault("num_sensitivity_rows", len(self._sensitivity_rows))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    # ---- helpers ------------------------------------------------------------------
    @staticmethod
    def _contiguous_overlap(a_ids: List[int], b_ids: List[int], min_run: int = 3) -> int:
        from difflib import SequenceMatcher

        sm = SequenceMatcher(a=a_ids, b=b_ids, autojunk=False)
        total = 0
        for block in sm.get_matching_blocks():
            if block.size >= min_run:
                total += block.size
        return total

    @staticmethod
    def _compare_vectors(va: List[float], vb: List[float]) -> Tuple[float, float]:
        if np is not None:
            a = np.asarray(va, dtype=np.float64)
            b = np.asarray(vb, dtype=np.float64)
            l2 = float(np.linalg.norm(a - b))
            na = float(np.linalg.norm(a))
            nb = float(np.linalg.norm(b))
            cos = float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0
            return l2, cos
        l2 = math.sqrt(sum((x - y) ** 2 for x, y in zip(va, vb)))
        dot = sum(x * y for x, y in zip(va, vb))
        na = math.sqrt(sum(x * x for x in va))
        nb = math.sqrt(sum(y * y for y in vb))
        cos = dot / (na * nb) if na > 0 and nb > 0 else 0.0
        return l2, cos
