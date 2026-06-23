"""
Profiled runner: API main orchestration (PlanningFlow) + local-HF sub-LLM execution,
with optional GAIA benchmark driving (camel-free).

Two run modes
-------------
1) Single-prompt (original behaviour, unchanged):
       python main_profiled.py
       python main_profiled.py --prompt "your task"

2) GAIA benchmark (no camel required):
       python main_profiled.py --gaia --on valid --level all
       python main_profiled.py --gaia --on valid --start 0 --end 10
       python main_profiled.py --gaia --on valid --level 1 --start 0 --end 5

Benchmark selection semantics (the two requested features)
----------------------------------------------------------
* --level {1,2,3,all}:  EXTRACT only the test cases of that level. GAIA ships per-level
                        HF configs, so this is just config selection
                        (level n -> "2023_level{n}", all -> "2023_all"). Feature #2.
* --start / --end:      then SLICE the (level-filtered) list as tasks[start:end], i.e.
                        run exactly (end - start) test cases. Feature #1.
  The two compose: with a specific --level you get "this level only, run 1..N of them";
  with --level all you get a plain global [start:end] slice over the whole split.

Dataset comes straight from the Hugging Face hub via `datasets.load_dataset`
("gaia-benchmark/GAIA"). It is gated: run `huggingface-cli login` once and accept the
terms on the dataset page. No CAMEL/OWL dependency is used anywhere on this path.

Output / profiling contract (kept intact)
------------------------------------------
Each task is run through its OWN flow.execute(question), so the existing profiling
behaviour is preserved 1:1 -- every task produces one timestamp run_id and one set of
    {profile_dir}/{response,profiling,tool_compare}/{run_id}.*
files (written by app/flow/planning.py's recorder lifecycle). This runner only adds
benchmark driving + scoring on top; it does NOT touch main.py / Manus().run().
"""

import argparse
import asyncio
from typing import Any, Dict, List, Optional, Union

from app.agent.profiled_executor import ProfiledExecutorAgent
from app.flow.planning import PlanningFlow
from app.llm import LLM
from app.logger import logger
from app.schema import Message


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
        # PlanningFlow.__init__ accepts plan_id -> active_plan_id. Explicit per-task
        # plan_id avoids the time.time() collision when two tasks start in the same second.
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
# GAIA benchmark mode (camel-free)
# --------------------------------------------------------------------------------------
def _parse_level(raw: str) -> Union[int, str]:
    raw = str(raw).strip().lower()
    if raw == "all":
        return "all"
    try:
        return int(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--level must be one of 1, 2, 3, all (got {raw!r})"
        )


def _slice_tasks(
    tasks: List[Dict[str, Any]], start: int, end: Optional[int]
) -> List[Dict[str, Any]]:
    """tasks[start:end] with clamping. (Feature #1; level already applied by config.)"""
    total = len(tasks)
    start = max(0, start)
    end = total if end is None else min(end, total)
    if start >= end:
        logger.warning(f"Empty task slice: start={start}, end={end}, total={total}.")
        return []
    sliced = tasks[start:end]
    logger.info(
        f"GAIA: {total} tasks after level filter; running [{start}:{end}] "
        f"= {len(sliced)} tasks."
    )
    return sliced


async def _formalize_answer(main_llm: LLM, question: str, raw_answer: str) -> str:
    """Normalise a raw answer into GAIA's strict format using the API main LLM.

    Replaces gaia.py's camel-based get_formal_answer: we reuse the existing API LLM
    (config_name="default") instead of pulling in camel's ModelFactory.
    """
    from gaia_bench import FORMAL_ANSWER_PROMPT

    prompt = FORMAL_ANSWER_PROMPT.format(question=question, text=raw_answer)
    resp = await main_llm.ask(
        messages=[Message.user_message(prompt)],
        stream=False,
        temperature=0.0,
    )
    return (resp or raw_answer).strip()


async def run_gaia(main_llm: LLM, args: argparse.Namespace) -> None:
    try:
        from gaia_bench import GaiaTasks
    except ImportError as e:
        logger.error(f"Failed to import gaia_bench ({e}).")
        return

    level = _parse_level(args.level)

    if args.on == "test":
        logger.warning(
            "GAIA 'test' split has hidden ground-truth answers; scores will not be "
            "meaningful (use it only to produce leaderboard submissions)."
        )

    gaia = GaiaTasks(level=level, split=args.on, save_to=args.save_to)

    try:
        all_tasks = gaia.load()
    except Exception as e:
        logger.error(
            f"Failed to load GAIA dataset (config={gaia.config}, split={gaia.split}): {e}. "
            f"Ensure `pip install datasets` and `huggingface-cli login` (gated dataset)."
        )
        return

    tasks = _slice_tasks(all_tasks, args.start, args.end)
    if not tasks:
        print("\n========== GAIA SUMMARY ==========")
        print("No tasks to run for the given level/slice.")
        return

    results: List[Dict[str, Any]] = []

    for i, task in enumerate(tasks):
        task_id = task.get("task_id")

        # Enrich Question with any attached-file hint (skip if a referenced file is gone).
        ok, info = gaia.prepare_task(task)
        if not ok:
            logger.warning(f"[{i}] Skipping task {task_id}: {info}")
            results.append(
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

            if args.formal and raw_answer:
                try:
                    answer = await _formalize_answer(main_llm, question, raw_answer)
                except Exception as e:
                    logger.error(f"Formalisation failed, using raw answer: {e}")
                    answer = raw_answer
            else:
                answer = raw_answer
        except Exception as e:
            logger.error(f"Error running task {task_id}: {e}")

        ground_truth = task.get("Final answer")
        try:
            score = (
                gaia.question_scorer(answer, ground_truth)
                if answer is not None and ground_truth is not None
                else False
            )
        except Exception as e:
            logger.error(f"Scoring failed for task {task_id}: {e}")
            score = False

        logger.info(
            f"Task {task_id} -> answer={answer!r} gt={ground_truth!r} score={score}"
        )

        results.append(
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
                gaia.save_results(results, gaia.save_to)
            except Exception as e:
                logger.warning(f"Failed to save results to {gaia.save_to}: {e}")

    summary = gaia.summary(results)
    print("\n========== GAIA SUMMARY ==========")
    print(f"on        : {gaia.split}")
    print(f"level     : {level}  (config={gaia.config})")
    print(f"slice     : [{args.start}:{args.end}]")
    print(f"total     : {summary['total']}")
    print(f"correct   : {summary['correct']}")
    print(f"accuracy  : {summary['accuracy']:.4f}")
    if args.save_to:
        print(f"results   : {gaia.save_to}")
    print(
        "profiling : per-task artifacts under the profiler's record dir "
        "(one run_id per task)."
    )


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Profiled OpenManus runner (PlanningFlow + local-HF executor) "
        "with optional GAIA benchmark driving."
    )

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
        "--on",
        type=str,
        default="valid",
        choices=["valid", "validation", "test"],
        help="Which GAIA split to run (valid==validation).",
    )
    p.add_argument(
        "--level",
        type=str,
        default="all",
        help="GAIA level to extract: 1, 2, 3, or all (feature #2). Selects the HF config.",
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
        "--save-to",
        type=str,
        default="record/gaia_results.json",
        help="Where to write the GAIA results JSON.",
    )
    p.add_argument(
        "--formal",
        action="store_true",
        help="Normalise each answer into GAIA's strict format using the API main LLM.",
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
