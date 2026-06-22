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
from app.prompt.manus import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.tool import Terminate, ToolCollection
from app.tool.browser_use_tool import BrowserUseTool
from app.tool.file_creator_viewer import FileCreatorViewer
from app.tool.file_saver import FileSaver
from app.tool.python_execute import PythonExecute
from app.tool.web_search import WebSearch


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
    next_step_prompt: str = NEXT_STEP_PROMPT

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
ProfiledManusExecutor = ProfiledExecutorAgent
