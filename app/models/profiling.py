"""
Profiling recorder for the API-main / local-HF-sub OpenManus execution path.

One ProfilingRecorder instance is created per full user request (per PlanningFlow.execute).
It is registered as the process-wide "active recorder" so that:

  - app/llm.py        -> records main-LLM and sub-LLM latency / token metrics
  - app/agent/toolcall.py -> records per-tool latency
  - app/models/hf_local.py (indirectly, via the message._profiling payload) -> V-cache summaries

Outputs (under {profile_dir}/...):
  record/response/{run_id}.txt
  record/profiling/{run_id}.csv
  record/tool_compare/{run_id}.csv

Notes
-----
* "head" in the V-cache summary refers to the KV (key/value) head index. Qwen3 uses grouped-query
  attention, so num_kv_heads < num_attention_heads. We compare the same (layer, kv_head) across calls.
* We never store full V tensors; only a mean-pooled vector of shape [head_dim] per (layer, kv_head).
"""

import csv
import math
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
except Exception:  # numpy is in requirements, but stay defensive
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
    tool_names: Optional[List[str]] = None
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
        return ev

    # ---- tool latency -------------------------------------------------------------
    def record_tool_latency(
        self, tool_name: str, latency: float, success: bool = True
    ) -> None:
        ev = self.current_sub_event
        # The most recent sub-LLM call is the one that selected this tool.
        if ev is not None and ev.tool_latency_sec is None:
            ev.tool_latency_sec = float(latency)
            ev.tool_success = bool(success)
            if not ev.selected_tool:
                ev.selected_tool = tool_name
            return

        # Same ask_tool produced more than one tool call: store an extra tool-only
        # event (no V-cache summary, so it does not enter the pairwise comparison).
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
        logger.info(f"[profiling] finalized run {self.run_id} -> {self.profile_dir}/")

    # ---- writers ------------------------------------------------------------------
    def _path(self, sub: str, ext: str) -> str:
        d = os.path.join(self.profile_dir, sub)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{self.run_id}.{ext}")

    def _write_response_txt(self) -> None:
        path = self._path("response", "txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"Run ID: {self.run_id}\n")
            f.write("=" * 70 + "\n")
            f.write("ORIGINAL REQUEST\n")
            f.write("-" * 70 + "\n")
            f.write(self.request_text.strip() + "\n\n")

            f.write("STEPWISE TASKS\n")
            f.write("-" * 70 + "\n")
            for idx, text in self.step_prompts:
                f.write(f"[step {idx}] {text}\n")
            f.write("\n")

            f.write("LLM CALLS (in order)\n")
            f.write("-" * 70 + "\n")
            for ev in self.events:
                if ev.llm_type == "sub" and ev.llm_latency_sec == 0.0 and ev.generated_text == "":
                    # tool-only auxiliary event; skip in the readable log
                    continue
                header = (
                    f"[{ev.llm_type}] step={ev.step_index} call={ev.call_index} "
                    f"model={ev.model} tool={ev.selected_tool} "
                    f"in={ev.input_tokens} out={ev.output_tokens} "
                    f"out+think={ev.output_tokens_including_think} "
                    f"llm={ev.llm_latency_sec:.3f}s tool={ev.tool_latency_sec}"
                )
                f.write(header + "\n")
                if ev.llm_type == "main":
                    f.write(
                        f"  finish_reason={ev.finish_reason} "
                        f"tool_calls={ev.tool_names or []}\n"
                    )
                    if ev.finish_reason == "length" and not ev.tool_names:
                        f.write(
                            "  ** WARNING: hit the output-token ceiling before emitting "
                            "a tool call. Increase max_tokens for this model. **\n"
                        )
                if ev.generated_text:
                    f.write("  generated (raw, incl. <think>):\n")
                    for line in ev.generated_text.splitlines():
                        f.write("    " + line + "\n")
                f.write("\n")

    def _write_profiling_csv(self) -> None:
        path = self._path("profiling", "csv")
        # NOTE: raw prompt_text / generated_text are intentionally NOT written here.
        # They are large and already preserved verbatim in response/{run_id}.txt;
        # duplicating them in the CSV bloats it and makes it hard to parse.
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
        # Only sub-LLM events that actually carry a V-cache summary participate.
        cache_events = [
            ev
            for ev in self.events
            if ev.llm_type == "sub" and ev.v_cache_summary
        ]
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "run_id", "tool_a", "call_a", "tool_b", "call_b",
                    "layer", "head", "l2", "cosine",
                    # input-overlap of the two compared calls (contiguous runs >= 3 tokens)
                    "input_tokens_a", "input_tokens_b", "overlap_tokens",
                    "overlap_pct_a", "overlap_pct_b",
                ]
            )
            for i in range(len(cache_events)):
                for j in range(i + 1, len(cache_events)):
                    a = cache_events[i]
                    b = cache_events[j]

                    # --- input-token overlap (computed once per pair) ---------------
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
                        # token ids unavailable (e.g. older run) -> leave blank
                        overlap_s = pct_a = pct_b = ""

                    common = set(a.v_cache_summary.keys()) & set(b.v_cache_summary.keys())
                    for (layer, head) in sorted(common):
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

    @staticmethod
    def _contiguous_overlap(a_ids: List[int], b_ids: List[int], min_run: int = 3) -> int:
        """Number of tokens that lie in a contiguous run of length >= min_run that
        appears identically (same order) in both token-id sequences.

        Uses difflib's non-overlapping matching blocks (autojunk disabled so that
        common tokens are not dropped) and sums the sizes of blocks of length
        >= min_run. The count is symmetric: each matched block contributes `size`
        tokens to both sequences, so it is divided by each side's own length to get
        the two overlap percentages.
        """
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
        # pure-python fallback
        l2 = math.sqrt(sum((x - y) ** 2 for x, y in zip(va, vb)))
        dot = sum(x * y for x, y in zip(va, vb))
        na = math.sqrt(sum(x * x for x in va))
        nb = math.sqrt(sum(y * y for y in vb))
        cos = dot / (na * nb) if na > 0 and nb > 0 else 0.0
        return l2, cos
