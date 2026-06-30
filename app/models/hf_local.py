"""
Local HuggingFace chat backend that mimics the OpenAI async client interface
( client.chat.completions.create(...) ) so it can be dropped into app/llm.py
in place of AsyncOpenAI for `api_type == "hf_local"`.

Design constraints:
  * Compatible with LLM.ask and LLM.ask_tool return expectations.
  * tool_calls are real app.schema.ToolCall objects.
  * Non-streaming only.
  * Heavy deps (torch/transformers) are imported lazily inside __init__.

KV / V-cache and influence profiling — two phases per call:

Phase A (generation)
  - use_cache=True, torch.no_grad().
  - T1: collect V-cache with mean-pool over decision window [p-W, p].

Phase B (analysis re-forward) — only when profile_influence=True
  - Teacher-forced forward over (prompt + generated) with use_cache=False,
    attn_implementation="eager", and gradients enabled.
  - o_proj pre-hooks capture head-concat activations at decision position p.
  - One backward pass (loss = log_softmax(logits_p)[y*]).
  - T2: head output contribution norm  ||c^h||  where c^h = z^h @ W_O_h.T
  - T3: direct logit attribution  dla^h = (c^h ⊙ γ·s) · W_U[y*,:]
         (linearised RMSNorm; direct path only — indirect effects not captured)
  - T5: gradient-based sensitivity  ||∂loss/∂z^h||
         (first-order / local approximation)
  Phase B is wrapped in try/except; generation result survives any failure.
  Additional GPU memory is required for the full forward graph during Phase B.

GQA indexing:
  - T1 is indexed by KV head (num_kv_heads).
  - T2/T3/T5 are indexed by query head (num_attention_heads).
  - kv_group = query_head // (num_attention_heads // num_key_value_heads).
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple

from app.config import LLMSettings
from app.logger import logger
from app.models.tool_parser import find_decision_token_pos, parse_generated_output


# --------------------------------------------------------------------------------------
# OpenAI-like response shells
# --------------------------------------------------------------------------------------
class _Message:
    def __init__(self, content: Optional[str], tool_calls: List[Any]):
        self.role = "assistant"
        self.content = content
        self.tool_calls = tool_calls or []
        self._profiling: Dict[str, Any] = {}


class _Choice:
    def __init__(self, message: "_Message"):
        self.index = 0
        self.message = message
        self.finish_reason = "stop"


class _Usage:
    def __init__(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


class _Response:
    def __init__(self, choices: List[_Choice], usage: Optional[_Usage] = None):
        self.choices = choices
        self.usage = usage


class _Completions:
    def __init__(self, client: "LocalHFClient"):
        self._client = client

    async def create(self, **kwargs):
        return await self._client._create(**kwargs)


class _Chat:
    def __init__(self, client: "LocalHFClient"):
        self.completions = _Completions(client)


# --------------------------------------------------------------------------------------
# The client
# --------------------------------------------------------------------------------------
class LocalHFClient:
    def __init__(self, settings: LLMSettings):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.settings = settings
        self.model_name = settings.model
        self.enable_thinking = bool(getattr(settings, "enable_thinking", False))
        self.profile_cache = bool(getattr(settings, "profile_cache", False))
        self.profile_influence = bool(getattr(settings, "profile_influence", False))
        # Window width W for T1 windowed V-cache pooling around decision position p.
        self._v_cache_window_W: int = 5

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "auto": "auto",
        }
        torch_dtype = dtype_map.get(
            getattr(settings, "torch_dtype", None) or "auto", "auto"
        )

        quant_config = None
        if getattr(settings, "load_in_4bit", False) or getattr(
            settings, "load_in_8bit", False
        ):
            try:
                from transformers import BitsAndBytesConfig

                quant_config = BitsAndBytesConfig(
                    load_in_4bit=bool(getattr(settings, "load_in_4bit", False)),
                    load_in_8bit=bool(getattr(settings, "load_in_8bit", False)),
                    bnb_4bit_compute_dtype=(
                        torch.bfloat16 if torch_dtype == "auto" else torch_dtype
                    ),
                )
            except Exception as e:
                logger.warning(f"[hf_local] bitsandbytes config failed: {e}")

        logger.info(f"[hf_local] loading tokenizer/model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )

        # Attention backend. We default to (and prefer) SDPA: eager materialises a
        # full [seq, seq] attention matrix per layer, which — combined with the
        # grad-enabled, full-graph Phase B forward — is the dominant OOM source.
        # The influence profiling reads ONLY the o_proj input (head-concat), never the
        # attention probabilities, so SDPA is sufficient and far cheaper. If a config
        # still asks for eager while influence profiling is on, override it.
        attn_impl = getattr(settings, "attn_implementation", None) or "sdpa"
        if self.profile_influence and attn_impl == "eager":
            logger.warning(
                "[hf_local] influence profiling does not need attention matrices; "
                "overriding attn_implementation 'eager' -> 'sdpa' to avoid OOM."
            )
            attn_impl = "sdpa"

        # Single 48GB GPU: pin everything to cuda:0 so accelerate never silently
        # offloads layers to CPU (which would make the Phase B backward crawl/break).
        _dev = getattr(settings, "device_map", None)
        device_map = {"": 0} if _dev in (None, "auto") else _dev

        model_kwargs: Dict[str, Any] = {
            "device_map": device_map,
            "trust_remote_code": True,
            "attn_implementation": attn_impl,
        }
        if quant_config is not None:
            model_kwargs["quantization_config"] = quant_config
        else:
            model_kwargs["torch_dtype"] = torch_dtype

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **model_kwargs
        )
        self.model.eval()
        self._attn_impl = attn_impl

        self.chat = _Chat(self)
        logger.info("[hf_local] model ready")

    # ---- public (async) create ----------------------------------------------------
    async def _create(
        self,
        model: str = None,
        messages: List[dict] = None,
        tools: Optional[List[dict]] = None,
        tool_choice: Any = "auto",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        stream: bool = False,
        **kwargs,
    ) -> _Response:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._generate_sync,
            messages or [],
            tools,
            tool_choice,
            float(temperature or 0.0),
            int(max_tokens or 1024),
        )

    # ---- prompt construction -------------------------------------------------------
    def _render_tools(self, tools: List[dict]) -> str:
        lines = [
            "You have access to the following tools. When a tool is needed, respond with "
            'EXACTLY ONE JSON object and nothing else:',
            '{"function_call": {"name": "<tool_name>", "arguments": {<args>}}}',
            "",
            "Available tools:",
        ]
        for tool in tools:
            if tool.get("type") != "function":
                continue
            fn = tool.get("function", {})
            lines.append(f"- {fn.get('name')}: {fn.get('description', '')}")
            params = (fn.get("parameters") or {}).get("properties", {})
            required = set((fn.get("parameters") or {}).get("required", []))
            for pname, pdef in params.items():
                req = " (required)" if pname in required else ""
                lines.append(
                    f"    * {pname} ({pdef.get('type', 'any')}){req}: "
                    f"{pdef.get('description', '')}"
                )
        return "\n".join(lines)

    def _build_prompt(self, messages: List[dict], tools: Optional[List[dict]]) -> str:
        msgs = list(messages)
        if tools:
            tool_msg = {"role": "system", "content": self._render_tools(tools)}
            if msgs and msgs[0].get("role") == "system":
                msgs = [msgs[0], tool_msg] + msgs[1:]
            else:
                msgs = [tool_msg] + msgs

        try:
            return self.tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )

    # ---- main sync entry point (Phase A + Phase B) ---------------------------------
    def _generate_sync(
        self,
        messages: List[dict],
        tools: Optional[List[dict]],
        tool_choice: Any,
        temperature: float,
        max_tokens: int,
    ) -> _Response:
        torch = self._torch

        use_tools = bool(tools) and str(tool_choice) != "none"
        prompt_text = self._build_prompt(messages, tools if use_tools else None)

        enc = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        input_len = int(enc.input_ids.shape[1])
        input_token_ids = enc.input_ids[0].tolist()

        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            use_cache=True,
            return_dict_in_generate=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature)
        else:
            gen_kwargs.update(do_sample=False)

        # ---- Phase A: generation ---------------------------------------------------
        with torch.no_grad():
            out = self.model.generate(**enc, **gen_kwargs)

        full_ids = out.sequences[0]                    # [seq_len]
        gen_ids = full_ids[input_len:]
        output_tokens_including_think = int(gen_ids.shape[0])
        gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        think_text, content, tool_calls = parse_generated_output(gen_text)

        think_len = 0
        if think_text:
            try:
                think_len = len(
                    self.tokenizer(think_text, add_special_tokens=False).input_ids
                )
            except Exception:
                think_len = 0
        output_tokens = max(0, output_tokens_including_think - think_len)

        selected_tool = tool_calls[0].function.name if tool_calls else None

        # Find decision token position p and y* (first token of selected tool name)
        decision_pos: Optional[int] = None
        decision_token_id: Optional[int] = None
        decision_token_str: str = ""
        if selected_tool:
            try:
                full_ids_list = full_ids.tolist()
                decision_pos, decision_token_id = find_decision_token_pos(
                    self.tokenizer,
                    full_ids_list,
                    input_len,
                    gen_text,
                    selected_tool,
                )
                if decision_token_id is not None:
                    decision_token_str = self.tokenizer.decode(
                        [decision_token_id], skip_special_tokens=False
                    )
            except Exception as e:
                logger.warning(f"[hf_local] decision token search failed: {e}")

        # T1: windowed V-cache around decision position
        v_cache_summary = None
        if self.profile_cache:
            try:
                v_cache_summary = self._collect_v_cache_windowed(
                    full_ids.unsqueeze(0), decision_pos, self._v_cache_window_W
                )
            except Exception as e:
                logger.warning(f"[hf_local] V-cache profiling failed: {e}")

        # ---- Phase B: analysis re-forward for T2/T3/T5 ----------------------------
        head_metrics: List[Dict[str, Any]] = []
        hf_model_meta: Optional[Dict[str, Any]] = None
        if self.profile_influence and decision_pos is not None and decision_token_id is not None:
            hf_model_meta = self._build_model_meta()
            try:
                head_metrics = self._run_analysis_forward(
                    full_ids.unsqueeze(0), decision_pos, decision_token_id
                )
            except Exception as e:
                logger.warning(
                    f"[hf_local] Phase B analysis failed (generation result preserved): {e}",
                    exc_info=True,
                )

        message = _Message(content=content or None, tool_calls=tool_calls)
        message._profiling = {
            "input_tokens": input_len,
            "output_tokens": output_tokens,
            "output_tokens_including_think": output_tokens_including_think,
            "generated_text": gen_text,
            "prompt_text": prompt_text,
            "v_cache_summary": v_cache_summary,
            "selected_tool": selected_tool,
            "input_token_ids": input_token_ids,
            # new fields
            "decision_token_pos": decision_pos,
            "decision_token_id": decision_token_id,
            "decision_token_str": decision_token_str,
            "head_metrics": head_metrics,
            "hf_model_meta": hf_model_meta,
        }
        usage = _Usage(
            prompt_tokens=input_len,
            completion_tokens=output_tokens_including_think,
        )
        return _Response([_Choice(message)], usage=usage)

    # ---- T1: windowed V-cache -----------------------------------------------------
    def _layer_values(self, past_key_values):
        """Return list of value tensors (one per layer) from legacy tuples or Cache obj."""
        if past_key_values is None:
            return []
        if hasattr(past_key_values, "to_legacy_cache"):
            legacy = past_key_values.to_legacy_cache()
            return [kv[1] for kv in legacy]
        return [kv[1] for kv in past_key_values]

    def _collect_v_cache_windowed(
        self,
        full_ids,           # [1, seq_len]
        decision_pos: Optional[int],
        window_W: int = 5,
    ) -> Dict[Tuple[int, int], List[float]]:
        """Mean-pool V-cache over window [p-W, p] around decision position p.

        Falls back to full-sequence pooling if decision_pos is None (backward compat).
        T1 is indexed by KV head (not query head).
        """
        torch = self._torch
        seq_len = full_ids.shape[1]

        with torch.no_grad():
            out = self.model(input_ids=full_ids, use_cache=True, return_dict=True)

        # Determine pool window
        if decision_pos is not None:
            win_start = max(0, decision_pos - window_W)
            win_end = min(decision_pos + 1, seq_len)  # p inclusive
        else:
            win_start = 0
            win_end = seq_len

        summary: Dict[Tuple[int, int], List[float]] = {}
        values = self._layer_values(getattr(out, "past_key_values", None))
        for layer_idx, value in enumerate(values):
            # value: [batch, num_kv_heads, seq, head_dim]
            v = value[0]                                         # [kv_heads, seq, head_dim]
            v_win = v[:, win_start:win_end, :]                  # [kv_heads, win, head_dim]
            if v_win.shape[1] == 0:
                continue
            pooled = v_win.mean(dim=1)                          # [kv_heads, head_dim]
            pooled = pooled.to(torch.float32).cpu().numpy()
            for head_idx in range(pooled.shape[0]):
                summary[(layer_idx, head_idx)] = pooled[head_idx].tolist()

        del out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return summary

    # ---- Phase B: T2 / T3 / T5 analysis re-forward --------------------------------
    def _build_model_meta(self) -> Dict[str, Any]:
        """Collect model config info for meta.json."""
        cfg = self.model.config
        num_q = getattr(cfg, "num_attention_heads", None)
        num_kv = getattr(cfg, "num_key_value_heads", None)
        h_dim = getattr(cfg, "head_dim", None)
        if h_dim is None and num_q:
            h_dim = getattr(cfg, "hidden_size", 0) // num_q
        return {
            "model_name": self.model_name,
            "num_attention_heads": num_q,
            "num_key_value_heads": num_kv,
            "head_dim": h_dim,
            "num_hidden_layers": getattr(cfg, "num_hidden_layers", None),
            "hidden_size": getattr(cfg, "hidden_size", None),
            "attn_implementation": self._attn_impl,
            "v_cache_window_W": self._v_cache_window_W,
            "profile_influence": self.profile_influence,
            "t1_head_type": "kv",
            "t2_t3_t5_head_type": "query",
            "dla_note": (
                "Direct path only (linearised RMSNorm). "
                "Indirect effects not captured; later layers dominate."
            ),
            "t5_note": "First-order / local approximation via one backward pass.",
        }

    def _run_analysis_forward(
        self,
        full_ids,           # [1, seq_len]
        p: int,             # prediction position (logits at p predict full_ids[p+1])
        y_star: int,        # decision token id
    ) -> List[Dict[str, Any]]:
        """Phase B: one teacher-forced forward + one backward for T2/T3/T5.

        Additional GPU memory is needed to hold the full autograd graph.
        Any exception is caught by the caller; generation result is unaffected.
        """
        torch = self._torch

        # SDPA is fine here: we capture the o_proj input (head-concat) via hooks and
        # never read attention probabilities, so there is no eager requirement.

        # ---- extract model architecture ------------------------------------------
        try:
            layers = self.model.model.layers
            final_norm = self.model.model.norm
            lm_head = self.model.lm_head
        except AttributeError as e:
            logger.warning(f"[hf_local] unexpected model structure for T2/T3/T5: {e}")
            return []

        cfg = self.model.config
        num_q_heads = cfg.num_attention_heads
        num_kv_heads = getattr(cfg, "num_key_value_heads", num_q_heads)
        head_dim = getattr(cfg, "head_dim", None) or (
            cfg.hidden_size // num_q_heads
        )
        num_layers = cfg.num_hidden_layers
        gqa_ratio = max(1, num_q_heads // num_kv_heads)

        orig_seq_len = full_ids.shape[1]
        if not (0 < p < orig_seq_len):
            logger.warning(
                f"[hf_local] decision_pos={p} out of range for seq_len={orig_seq_len}"
            )
            return []

        if not (0 <= y_star < lm_head.weight.shape[0]):
            logger.warning(f"[hf_local] y_star={y_star} out of vocab range")
            return []

        # Only logits at position p matter, so drop everything after p. This shrinks
        # the Phase B sequence (and hence the autograd graph) to [0 .. p]; p becomes
        # the last index of the truncated sequence.
        analysis_ids = full_ids[:, : p + 1]
        seq_len = analysis_ids.shape[1]  # == p + 1

        # ---- memory-saving setup for the grad-enabled re-forward ------------------
        # 1) Freeze ALL params: we only need gradients w.r.t. ACTIVATIONS (z^h), not
        #    weights. This removes large param-grad buffers (e.g. lm_head ~1.5GB) and
        #    avoids cross-call grad accumulation. enable_input_require_grads() then
        #    re-introduces a grad source at the embeddings so the activation graph
        #    (and our backward hooks) still receive gradients.
        # 2) Gradient checkpointing (non-reentrant): recompute layer activations in
        #    backward instead of retaining all layers' activations -> the single
        #    biggest Phase B memory saver. Requires use_cache=False.
        prev_use_cache = getattr(self.model.config, "use_cache", None)
        ckpt_enabled = False
        input_req_grads = False

        # z_at_p[layer]:  detached head-concat VALUES at p          -> T2 / T3
        # gz_at_p[layer]: grad of loss w.r.t. that head-concat at p -> T5
        # Two hooks (forward values + module backward grad) so capture survives
        # gradient checkpointing, where retain_grad() on a recomputed intermediate
        # would NOT be populated.
        z_at_p: Dict[int, Any] = {}
        gz_at_p: Dict[int, Any] = {}
        resid_pre_norm: Dict[str, Any] = {}

        try:
            self.model.requires_grad_(False)
            self.model.enable_input_require_grads()
            input_req_grads = True
            self.model.config.use_cache = False
            try:
                self.model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                self.model.gradient_checkpointing_enable()
            ckpt_enabled = True

            # ---- install hooks ----------------------------------------------------
            hooks = []
            for i in range(num_layers):
                try:
                    o_proj = layers[i].self_attn.o_proj
                except AttributeError:
                    continue

                def _make_fwd(layer_idx: int):
                    def _hook(module, args):
                        x = args[0]  # [batch, seq, num_q_heads * head_dim]
                        if x.shape[1] > p:
                            z_at_p[layer_idx] = x[0, p, :].detach().to(torch.float32)
                    return _hook

                def _make_bwd(layer_idx: int):
                    def _hook(module, grad_input, grad_output):
                        gi = grad_input[0] if grad_input else None
                        if gi is not None and gi.shape[1] > p:
                            gz_at_p[layer_idx] = gi[0, p, :].detach().to(torch.float32)
                    return _hook

                hooks.append(o_proj.register_forward_pre_hook(_make_fwd(i)))
                hooks.append(o_proj.register_full_backward_hook(_make_bwd(i)))

            def _norm_pre_hook(module, args):
                x = args[0]  # [batch, seq, hidden]
                if x.shape[1] > p:
                    resid_pre_norm["r"] = x[0, p, :].detach().clone().float()

            hooks.append(final_norm.register_forward_pre_hook(_norm_pre_hook))

            # ---- forward (+ isolated backward) -----------------------------------
            try:
                with torch.enable_grad():
                    out = self.model(
                        input_ids=analysis_ids,
                        use_cache=False,
                        return_dict=True,
                    )
                    logits_p = out.logits[0, p, :]  # [vocab]
                    loss = torch.nn.functional.log_softmax(logits_p, dim=0)[y_star]
                    # Backward is isolated: T2/T3 only need the forward values
                    # (z_at_p / residual), so a backward OOM/failure must not discard
                    # them — it only drops T5.
                    try:
                        loss.backward()
                    except Exception as e:
                        logger.warning(
                            f"[hf_local] Phase B backward failed; keeping T2/T3, "
                            f"skipping T5: {e}"
                        )
            finally:
                for h in hooks:
                    h.remove()
                hooks.clear()
        finally:
            # Restore the model to its normal inference configuration.
            if ckpt_enabled:
                try:
                    self.model.gradient_checkpointing_disable()
                except Exception:
                    pass
            if input_req_grads:
                try:
                    self.model.disable_input_require_grads()
                except Exception:
                    pass
            if prev_use_cache is not None:
                self.model.config.use_cache = prev_use_cache

        # ---- compute T2 / T3 / T5 ------------------------------------------------
        # Index the single unembedding row BEFORE up-casting; casting the whole
        # [vocab, hidden] lm_head to float32 would waste ~3GB for one row.
        gamma = final_norm.weight.detach().float()              # [hidden]
        w_u_y = lm_head.weight[y_star, :].detach().float()      # [hidden]

        # RMSNorm linearisation scale at position p
        r = resid_pre_norm.get("r")
        norm_scale: Optional[float] = None
        if r is not None:
            eps = getattr(final_norm, "variance_epsilon", None) or getattr(
                final_norm, "eps", 1e-6
            )
            rms = float((r.pow(2).mean() + eps).sqrt())
            norm_scale = 1.0 / rms if rms > 0 else None

        rows: List[Dict[str, Any]] = []

        for layer_idx in range(num_layers):
            h_float = z_at_p.get(layer_idx)          # [num_q_heads * head_dim] or None
            if h_float is None:
                continue

            try:
                o_proj_w = (
                    layers[layer_idx].self_attn.o_proj.weight.detach().float()
                )  # [hidden_out, num_q_heads * head_dim]
            except AttributeError:
                continue

            grad_h = gz_at_p.get(layer_idx)          # [num_q_heads * head_dim] or None

            for q_h in range(num_q_heads):
                kv_group = q_h // gqa_ratio
                s = q_h * head_dim
                e = s + head_dim

                z_h = h_float[s:e]                              # [head_dim]
                W_O_h = o_proj_w[:, s:e]                       # [hidden_out, head_dim]
                c_h = z_h @ W_O_h.t()                          # [hidden_out]

                # T2 — head output contribution norm
                contrib_norm = float(c_h.norm())
                rows.append(
                    {
                        "layer": layer_idx,
                        "head": q_h,
                        "head_type": "query",
                        "kv_group": kv_group,
                        "metric_type": "contrib_norm",
                        "value": contrib_norm,
                    }
                )

                # T3 — direct logit attribution (linearised RMSNorm; direct path only)
                if norm_scale is not None:
                    c_h_norm = c_h * gamma * norm_scale         # [hidden]
                    dla = float((c_h_norm * w_u_y).sum())
                    rows.append(
                        {
                            "layer": layer_idx,
                            "head": q_h,
                            "head_type": "query",
                            "kv_group": kv_group,
                            "metric_type": "dla",
                            "value": dla,
                        }
                    )

                # T5 — gradient sensitivity (first-order approximation)
                if grad_h is not None:
                    g_h = grad_h[s:e].float()                   # [head_dim]
                    grad_sens = float(g_h.norm())
                    rows.append(
                        {
                            "layer": layer_idx,
                            "head": q_h,
                            "head_type": "query",
                            "kv_group": kv_group,
                            "metric_type": "grad_sens",
                            "value": grad_sens,
                        }
                    )

        # free graph memory
        del out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return rows
