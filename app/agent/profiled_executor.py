"""
Local-HF profiled executor agent.

This is the SUB agent in the new execution path. It is a normal ToolCallAgent whose
LLM is the local HuggingFace backend (config_name="profiled_qwen"). It chooses tools;
the selected tool name + call index are what differentiate calls for KV profiling.
"""

from typing import Any

from pydantic import Field

from app.agent.toolcall import ToolCallAgent
from app.llm import LLM
from app.prompt.manus import NEXT_STEP_PROMPT as _MANUS_NEXT_STEP_PROMPT
from app.prompt.manus import SYSTEM_PROMPT
from app.tool import ImageQuery, Terminate, ToolCollection
from app.tool.browser_use_tool import BrowserUseTool
from app.tool.file_creator_viewer import FileCreatorViewer
from app.tool.file_saver import FileSaver
from app.tool.python_execute import PythonExecute
from app.tool.web_search import WebSearch


# Reinforce termination behaviour for the profiled executor WITHOUT touching the
# shared Manus prompt. In OpenManus the loop ends ONLY when the `terminate` tool is
# called; declaring completion in natural language does not stop execution. Qwen3
# tends to announce "task complete" in prose (selecting 0 tools), which — combined
# with the structural early-exit fix — would otherwise burn plan steps needlessly.
_TERMINATE_REMINDER = """

IMPORTANT — how to finish:
- The ONLY way to end the task is to call the `terminate` tool. Saying the task is
  done in natural language does NOT stop execution.
- As soon as you have produced the final answer (or if you genuinely cannot make
  further progress), you MUST immediately call the `terminate` tool:
  use status="success" when the task is solved, status="failure" otherwise.
- Do not keep repeating a tool call that is not making progress. If you are stuck,
  call `terminate` with status="failure" instead of looping.

If the task references an image file path, FIRST call `image_query` with that path \
and a specific question, then reason over the returned text. Do not attempt to read \
images any other way.
"""


class ProfiledExecutorAgent(ToolCallAgent):
    """A ToolCallAgent backed by a local HuggingFace model, used for KV profiling.

    It owns the same general-purpose tools as Manus so that tool selection behaviour
    is comparable to the original single-agent path.
    """

    name: str = "profiled_executor"
    description: str = (
        "A local-HF profiled executor that selects and runs OpenManus tools."
    )

    system_prompt: str = SYSTEM_PROMPT
    # Profiled-executor-specific next-step prompt: Manus prompt + terminate reminder.
    next_step_prompt: str = _MANUS_NEXT_STEP_PROMPT + _TERMINATE_REMINDER

    # Local HuggingFace sub-LLM (Qwen/Qwen3-32B via [llm.profiled_qwen]).
    llm: LLM = Field(default_factory=lambda: LLM(config_name="profiled_qwen"))

    max_observe: int = 2000
    max_steps: int = 20

    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            PythonExecute(),
            WebSearch(),
            FileSaver(),
            BrowserUseTool(),
            FileCreatorViewer(),
            ImageQuery(),
            Terminate(),
        )
    )

    async def _handle_special_tool(self, name: str, result: Any, **kwargs):
        if not self._is_special_tool(name):
            return
        else:
            await self.available_tools.get_tool(BrowserUseTool().name).cleanup()
            await super()._handle_special_tool(name, result, **kwargs)


# Backward-compatible alias mentioned in the spec.
ProfiledToolCallAgent = ProfiledExecutorAgent
ProfiledManusExecutor = ProfiledExecutorAgent
