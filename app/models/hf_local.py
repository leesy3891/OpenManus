"""
Local HuggingFace chat backend that mimics the OpenAI async client interface
( client.chat.completions.create(...) ) so it can be dropped into app/llm.py
in place of AsyncOpenAI for `api_type == "hf_local"`.

Design constraints (see understanding_OpenManus_code_structure):
  * Compatible with LLM.ask and LLM.ask_tool return expectations:
    response.choices[0].message has `.content` and `.tool_calls`.
  * tool_calls are real app.schema.ToolCall objects (model_dump-able), never dummies.
  * Non-streaming only.
  * Heavy deps (torch/transformers) are imported lazily inside __init__ so the
    openai/azure/hf paths keep working in environments without them.

KV / V-cache profiling:
  * Generation runs with use_cache=True.
  * When profile_cache is enabled, a single forward pass over (prompt + generated)
    is used to read past_key_values and mean-pool each layer's value tensor to a
    compact [num_kv_heads, head_dim] summary. We never keep full V tensors.
  * "head" = KV head index (Qwen3 uses grouped-query attention).
"""

import asyncio
import time
from typing import Any, Dict, List, Optional

from app.config import LLMSettings
from app.logger import logger
from app.models.tool_parser import parse_generated_output


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
    def __init__(self, message: _Message):
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
        # lazy heavy imports
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.settings = settings
        self.model_name = settings.model
        self.enable_thinking = bool(getattr(settings, "enable_thinking", False))
        self.profile_cache = bool(getattr(settings, "profile_cache", False))

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "auto": "auto",
        }
        torch_dtype = dtype_map.get(getattr(settings, "torch_dtype", None) or "auto", "auto")

        quant_config = None
        if getattr(settings, "load_in_4bit", False) or getattr(settings, "load_in_8bit", False):
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

        model_kwargs: Dict[str, Any] = {
            "device_map": getattr(settings, "device_map", None) or "auto",
            "trust_remote_code": True,
            "attn_implementation": getattr(settings, "attn_implementation", None) or "eager",
        }
        if quant_config is not None:
            model_kwargs["quantization_config"] = quant_config
        else:
            model_kwargs["torch_dtype"] = torch_dtype

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        self.model.eval()

        # OpenAI-compatible surface
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
        # generation is blocking -> run in a thread so we don't stall the event loop
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
                    f"    * {pname} ({pdef.get('type', 'any')}){req}: {pdef.get('description', '')}"
                )
        return "\n".join(lines)

    def _build_prompt(self, messages: List[dict], tools: Optional[List[dict]]) -> str:
        msgs = list(messages)
        if tools:
            tool_msg = {"role": "system", "content": self._render_tools(tools)}
            # keep any existing system message first, then the tool spec
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
            # older templates without enable_thinking kwarg
            return self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )

    # ---- sync generation + profiling ----------------------------------------------
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

        with torch.no_grad():
            out = self.model.generate(**enc, **gen_kwargs)

        full_ids = out.sequences[0]
        gen_ids = full_ids[input_len:]
        output_tokens_including_think = int(gen_ids.shape[0])
        gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        think_text, content, tool_calls = parse_generated_output(gen_text)

        # output tokens excluding <think>
        think_len = 0
        if think_text:
            try:
                think_len = len(self.tokenizer(think_text, add_special_tokens=False).input_ids)
            except Exception:
                think_len = 0
        output_tokens = max(0, output_tokens_including_think - think_len)

        selected_tool = tool_calls[0].function.name if tool_calls else None

        # V-cache summary (optional, expensive)
        v_cache_summary = None
        if self.profile_cache:
            try:
                v_cache_summary = self._collect_v_cache(full_ids.unsqueeze(0))
            except Exception as e:
                logger.warning(f"[hf_local] V-cache profiling failed: {e}")

        message = _Message(content=content or None, tool_calls=tool_calls)
        message._profiling = {
            "input_tokens": input_len,
            "output_tokens": output_tokens,
            "output_tokens_including_think": output_tokens_including_think,
            "generated_text": gen_text,
            "prompt_text": prompt_text,
            "v_cache_summary": v_cache_summary,
            "selected_tool": selected_tool,
        }
        usage = _Usage(prompt_tokens=input_len, completion_tokens=output_tokens_including_think)
        return _Response([_Choice(message)], usage=usage)

    def _layer_values(self, past_key_values):
        """Return a list of value tensors (one per layer), handling both legacy tuples
        and the newer Cache objects."""
        if past_key_values is None:
            return []
        if hasattr(past_key_values, "to_legacy_cache"):
            legacy = past_key_values.to_legacy_cache()
            return [kv[1] for kv in legacy]
        # legacy tuple of (key, value) per layer
        return [kv[1] for kv in past_key_values]

    def _collect_v_cache(self, full_ids) -> Dict[tuple, List[float]]:
        torch = self._torch
        with torch.no_grad():
            out = self.model(input_ids=full_ids, use_cache=True, return_dict=True)

        summary: Dict[tuple, List[float]] = {}
        values = self._layer_values(getattr(out, "past_key_values", None))
        for layer_idx, value in enumerate(values):
            # value: [batch, num_kv_heads, seq, head_dim]
            v = value[0]                  # [num_kv_heads, seq, head_dim]
            pooled = v.mean(dim=1)        # mean-pool over sequence -> [num_kv_heads, head_dim]
            pooled = pooled.to(torch.float32).cpu().numpy()
            for head_idx in range(pooled.shape[0]):
                summary[(layer_idx, head_idx)] = pooled[head_idx].tolist()

        # free memory
        del out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return summary
