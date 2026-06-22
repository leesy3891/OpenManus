"""
Profiled runner: API main orchestration (PlanningFlow) + local-HF sub-LLM execution.

Run:
    python main_profiled.py

This does NOT touch the existing main.py / Manus().run() path.
Profiling artifacts are written to {profile_dir}/{response,profiling,tool_compare}/{run_id}.*
"""

import asyncio

from app.agent.profiled_executor import ProfiledExecutorAgent
from app.flow.planning import PlanningFlow
from app.llm import LLM
from app.logger import logger


async def main():
    # Main orchestration LLM = API-based [llm] (config_name="default").
    main_llm = LLM(config_name="default")

    # Sub executor LLM = local HuggingFace [llm.profiled_qwen] (built by the agent).
    executor = ProfiledExecutorAgent()

    flow = PlanningFlow(
        agents={"profiled_executor": executor},
        primary_agent_key="profiled_executor",
        llm=main_llm,
        executors=["profiled_executor"],
    )

    try:
        prompt = input("Enter your prompt: ")
        if not prompt.strip():
            logger.warning("Empty prompt provided.")
            return

        logger.warning("Processing your request (profiled path)...")
        result = await flow.execute(prompt)
        logger.info("Request processing completed.")
        print("\n========== RESULT ==========")
        print(result)
    except KeyboardInterrupt:
        logger.warning("Operation interrupted.")


if __name__ == "__main__":
    asyncio.run(main())
