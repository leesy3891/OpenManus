"""
Profiled runner: API main orchestration (PlanningFlow) + local-HF sub-LLM execution,
with optional GAIA benchmark driving.

Two run modes
-------------
1) Single-prompt (original behaviour, unchanged):
       python main_profiled.py
       python main_profiled.py --prompt "your task"

2) GAIA benchmark:
       python main_profiled.py --gaia --data-dir /path/to/GAIA --on valid --level all
       python main_profiled.py --gaia --data-dir /path/to/GAIA --start 0 --end 10
       python main_profiled.py --gaia --data-dir /path/to/GAIA --level 2 --start 0 --end 5

Benchmark selection semantics (the two requested features)
----------------------------------------------------------
* --level {1,2,3,all}:  first EXTRACT only the test cases of that level
                        (level="all" keeps every level). This is feature #2.
* --start / --end:      then SLICE the (level-filtered) list as tasks[start:end],
                        i.e. run exactly (end - start) test cases. This is feature #1.
  The two compose: with a specific --level you get "this level only, run 1..N of them";
  with --level all you get a plain global [start:end] slice over the whole split.

Output / profiling contract (kept intact)
------------------------------------------
Each task is run through its OWN flow.execute(question), so the existing profiling
behaviour is preserved 1:1 -- every task produces one timestamp run_id and one set of
    {profile_dir}/{response,profiling,tool_compare}/{run_id}.*
files (written by app/flow/planning.py's recorder lifecycle). This runner only adds
benchmark driving + scoring on top; it does NOT touch main.py / Manus().run().

GAIA's own run()/run_role_playing()/run_workforce_with_retry() are CAMEL-specific
(they drive a camel ChatAgent / Workforce), so they are intentionally NOT used here.
We reuse only the dataset-loading + scoring helpers from utils.gaia.GAIABenchmark:
    _load_tasks, _prepare_task, question_scorer, _save_results_to_file, _generate_summary
and feed each task's Question into the OpenManus PlanningFlow.
"""

import argparse
import asyncio
from typing import Any, Dict, List, Optional, Union

from app.agent.profiled_executor import ProfiledExecutorAgent
from app.flow.planning import PlanningFlow
from app.llm import LLM
from app.logger import logger


# --------------------------------------------------------------------------------------
# Flow construction
# --------------------------------------------------------------------------------------
def build_flow(main_llm: LLM, plan_id: Optional[str] = None) -> PlanningFlow:
    """Build a fresh PlanningFlow + local-HF executor.

    A NEW flow (and a NEW executor) is built per task so that plan state and agent
    memory never leak across benchmark tasks. The heavy local-HF model is NOT reloaded:
    LLM(config_name=...) is a singleton keyed by config name, so the Qwen weights are
    loaded once and shared by every executor instance.
    """
    executor = ProfiledExecutorAgent()

    data: Dict[str, Any] = dict(
        agents={"profiled_executor": executor},
        primary_agent_key="profiled_executor",
        llm=main_llm,
        executors=["profiled_executor"],
    )
    if plan_id is not None:
        # PlanningFlow.__init__ accepts plan_id and maps it onto active_plan_id.
        # Explicit per-task plan_id avoids the time.time() collision when two tasks
        # start within the same second.
        data["plan_id"] = plan_id

    return PlanningFlow(**data)


# --------------------------------------------------------------------------------------
# Single-prompt mode (original behaviour)
# --------------------------------------------------------------------------------------
async def run_single(main_llm: LLM, prompt: Optional[str]) -> None:
    if not prompt:
        try:
            prompt = input("Enter your prompt: ")
        except EOFError:
            prompt = ""
    if not prompt or not prompt.strip():
        logger.warning("Empty prompt provided.")
        return

    flow = build_flow(main_llm)

    logger.warning("Processing your request (profiled path)...")
    result = await flow.execute(prompt)
    logger.info("Request processing completed.")
    print("\n========== RESULT ==========")
    print(result)


# --------------------------------------------------------------------------------------
# GAIA benchmark mode
# --------------------------------------------------------------------------------------
def _parse_level(raw: str) -> Union[int, str]:
    """argparse string -> _load_tasks() level argument (int or 'all')."""
    raw = str(raw).strip().lower()
    if raw == "all":
        return "all"
    try:
        return int(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--level must be one of 1, 2, 3, all (got {raw!r})"
        )


def _load_gaia_tasks(
    benchmark,
    on: str,
    level: Union[int, str],
    start: int,
    end: Optional[int],
) -> List[Dict[str, Any]]:
    """Load -> level-filter -> [start:end] slice.

    Feature #2 (level extraction) is handled by GAIABenchmark._load_tasks(level=...).
    Feature #1 (count via slicing) is the tasks[start:end] applied afterwards.
    """
    # level filter happens inside _load_tasks; randomize/subset/idx left at defaults so
    # the ordering is stable and our [start:end] slice is reproducible.
    tasks = benchmark._load_tasks(
        on=on, level=level, randomize=False, subset=None, idx=None
    )

    total = len(tasks)
    start = max(0, start)
    end = total if end is None else min(end, total)
    if start >= end:
        logger.warning(
            f"Empty task slice: start={start}, end={end}, level-filtered total={total}."
        )
        return []

    sliced = tasks[start:end]
    logger.info(
        f"GAIA: on={on} level={level} -> {total} tasks after level filter; "
        f"running [{start}:{end}] = {len(sliced)} tasks."
    )
    return sliced


async def run_gaia(main_llm: LLM, args: argparse.Namespace) -> None:
    # Imported lazily: utils.gaia pulls in CAMEL/OWL deps that single-prompt mode
    # does not need, so we only require them when --gaia is actually requested.
    try:
        from utils.gaia import GAIABenchmark
    except Exception as e:  # pragma: no cover - depends on user env
        logger.error(
            f"Failed to import utils.gaia.GAIABenchmark ({e}). "
            f"Make sure utils/gaia.py and its dependencies are importable."
        )
        return

    benchmark = GAIABenchmark(
        data_dir=args.data_dir,
        save_to=args.save_to,
        processes=1,
    )

    level = _parse_level(args.level)
    tasks = _load_gaia_tasks(benchmark, args.on, level, args.start, args.end)
    if not tasks:
        print("\n========== GAIA SUMMARY ==========")
        print("No tasks to run for the given level/slice.")
        return

    # Fresh results list for this run (we drive scoring ourselves, not benchmark.run()).
    benchmark._results = []

    for i, task in enumerate(tasks):
        task_id = task.get("task_id")
        # _prepare_task enriches task["Question"] with any attached file paths (and
        # tells us to skip if a referenced file is missing), exactly as benchmark.run does.
        ok, info = benchmark._prepare_task(task)
        if not ok:
            logger.warning(f"[{i}] Skipping task {task_id}: {info}")
            benchmark._results.append(
                {
                    "task_id": task_id,
                    "question": task.get("Question"),
                    "level": task.get("Level"),
                    "model_answer": None,
                    "ground_truth": task.get("Final answer"),
                    "score": False,
                    "raw_answer": None,
                }
            )
            continue

        question = task["Question"]
        logger.warning(
            f"[{i + 1}/{len(tasks)}] Running GAIA task {task_id} "
            f"(level {task.get('Level')})..."
        )

        # One flow.execute per task == one profiling run_id (record/.../{run_id}.*).
        flow = build_flow(main_llm, plan_id=f"plan_gaia_{i}_{task_id}")

        raw_answer: Optional[str] = None
        answer: Optional[str] = None
        try:
            raw_answer = await flow.execute(question)

            # By default we score the raw flow output. With --formal we additionally
            # normalise it into GAIA's strict answer format via the benchmark's own
            # GPT-4o reformatter (requires the camel OpenAI backend to be configured).
            if args.formal and raw_answer:
                try:
                    answer = benchmark.get_formal_answer(question, raw_answer)
                except Exception as e:
                    logger.error(f"get_formal_answer failed, using raw answer: {e}")
                    answer = raw_answer
            else:
                answer = raw_answer
        except Exception as e:
            logger.error(f"Error running task {task_id}: {e}")

        ground_truth = task.get("Final answer")
        try:
            score = (
                benchmark.question_scorer(answer, ground_truth)
                if answer is not None
                else False
            )
        except Exception as e:
            logger.error(f"Scoring failed for task {task_id}: {e}")
            score = False

        logger.info(f"Task {task_id} -> answer={answer!r} gt={ground_truth!r} score={score}")

        benchmark._results.append(
            {
                "task_id": task_id,
                "question": question,
                "level": task.get("Level"),
                "model_answer": answer,
                "ground_truth": ground_truth,
                "score": score,
                "raw_answer": raw_answer,
            }
        )

        # Persist incrementally so a crash mid-run still leaves partial results.
        if args.save_to:
            try:
                benchmark._save_results_to_file(benchmark._results, benchmark.save_to)
            except Exception as e:
                logger.warning(f"Failed to save results to {benchmark.save_to}: {e}")

    # Final summary using the benchmark's own aggregator (total / correct / accuracy).
    summary = benchmark._generate_summary()
    print("\n========== GAIA SUMMARY ==========")
    print(f"on        : {args.on}")
    print(f"level     : {level}")
    print(f"slice     : [{args.start}:{args.end}]")
    print(f"total     : {summary['total']}")
    print(f"correct   : {summary['correct']}")
    print(f"accuracy  : {summary['accuracy']:.4f}")
    if args.save_to:
        print(f"results   : {benchmark.save_to}")
    print("profiling : per-task artifacts under the profiler's record dir "
          "(one run_id per task).")


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Profiled OpenManus runner (PlanningFlow + local-HF executor) "
        "with optional GAIA benchmark driving."
    )

    # Mode / single-prompt
    p.add_argument(
        "--gaia",
        action="store_true",
        help="Run the GAIA benchmark instead of a single prompt.",
    )
    p.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single-prompt mode prompt (if omitted and not --gaia, reads from stdin).",
    )

    # GAIA options
    p.add_argument(
        "--data-dir",
        type=str,
        default="data/gaia",
        help="GAIA dataset directory (passed to GAIABenchmark).",
    )
    p.add_argument(
        "--save-to",
        type=str,
        default="record/gaia_results.json",
        help="Where to write the GAIA results JSON.",
    )
    p.add_argument(
        "--on",
        type=str,
        default="valid",
        choices=["valid", "test"],
        help="Which GAIA split to run.",
    )
    p.add_argument(
        "--level",
        type=str,
        default="all",
        help="GAIA level to extract: 1, 2, 3, or all (feature #2).",
    )
    p.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index of the [start:end] slice (feature #1).",
    )
    p.add_argument(
        "--end",
        type=int,
        default=None,
        help="End index (exclusive) of the [start:end] slice; default = run to the end.",
    )
    p.add_argument(
        "--formal",
        action="store_true",
        help="Normalise each answer into GAIA's strict format via "
        "GAIABenchmark.get_formal_answer (needs the camel OpenAI backend).",
    )

    return p.parse_args()


async def main() -> None:
    args = parse_args()

    # Main orchestration LLM = API-based [llm] (config_name="default").
    main_llm = LLM(config_name="default")

    try:
        if args.gaia:
            logger.warning("Processing GAIA benchmark (profiled path)...")
            await run_gaia(main_llm, args)
        else:
            await run_single(main_llm, args.prompt)
        logger.info("Request processing completed.")
    except KeyboardInterrupt:
        logger.warning("Operation interrupted.")


if __name__ == "__main__":
    asyncio.run(main())
