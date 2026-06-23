"""
Standalone, camel-free GAIA benchmark helper.

Why this exists
---------------
The repo's utils/gaia.py imports CAMEL/OWL at module top (camel.benchmarks,
camel.societies.workforce, .enhanced_role_playing, ...), so merely importing
GAIABenchmark pulls in the whole CAMEL stack ("No module named 'camel'").
We don't run a camel agent here -- the OpenManus PlanningFlow drives the tasks --
so this module reimplements ONLY the dataset-loading + scoring pieces with no
external agent deps.

Dataset loading
---------------
Loaded straight from the Hugging Face hub (gated dataset; run `huggingface-cli login`
once and accept the terms on the GAIA repo page):

    from datasets import load_dataset
    ds = load_dataset("gaia-benchmark/GAIA", "2023_all",    split="validation")
    ds = load_dataset("gaia-benchmark/GAIA", "2023_level1", split="validation")
    ds = load_dataset("gaia-benchmark/GAIA", "2023_level2", split="validation")

The dataset ships per-level configs, so LEVEL FILTERING IS JUST CONFIG SELECTION:
    level "all" -> "2023_all", level 1/2/3 -> "2023_level{n}".
Columns: task_id, Question, Level, Final answer, file_name, file_path,
and struct-valued "Annotator Metadata".

The scorer (question_scorer / normalize_* / split_string) is ported verbatim from
utils/gaia.py, which itself mirrors the official GAIA leaderboard scorer. Pure stdlib.
"""

import json
import os
import re
import string
from typing import Any, Dict, List, Optional, Union


# level -> HF config name
_CONFIG_BY_LEVEL = {
    "all": "2023_all",
    1: "2023_level1",
    2: "2023_level2",
    3: "2023_level3",
}

# accept a few spellings for the split
_SPLIT_ALIASES = {
    "valid": "validation",
    "validation": "validation",
    "dev": "validation",
    "test": "test",
}


class GaiaTasks:
    """Camel-free GAIA loader + scorer.

    Args:
        level: 1, 2, 3, or "all". Selects the HF per-level config.
        split: "valid"/"validation" or "test".
        save_to: optional path for the results JSON (used by save_results()).
    """

    def __init__(
        self,
        level: Union[int, str] = "all",
        split: str = "validation",
        save_to: Optional[str] = None,
    ):
        self.level = self._normalize_level(level)
        self.split = self._normalize_split(split)
        self.config = _CONFIG_BY_LEVEL[self.level]
        self.save_to = save_to

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _normalize_level(level: Union[int, str]) -> Union[int, str]:
        if isinstance(level, str):
            s = level.strip().lower()
            if s == "all":
                return "all"
            level = int(s)
        if level not in (1, 2, 3):
            raise ValueError(f"level must be 1, 2, 3 or 'all' (got {level!r})")
        return level

    @staticmethod
    def _normalize_split(split: str) -> str:
        s = str(split).strip().lower()
        if s not in _SPLIT_ALIASES:
            raise ValueError(f"split must be valid/validation/test (got {split!r})")
        return _SPLIT_ALIASES[s]

    # ----------------------------------------------------------------- loading
    def load(self) -> List[Dict[str, Any]]:
        """Load the (level-filtered) split as a list of plain task dicts."""
        try:
            from datasets import load_dataset
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "The `datasets` package is required to load GAIA. "
                "Install it with `pip install datasets`."
            ) from e

        ds = load_dataset("gaia-benchmark/GAIA", self.config, split=self.split)
        # Each row already maps the columns; make plain dicts so callers can mutate.
        tasks = [dict(row) for row in ds]

        # Drop the dummy/sentinel row if the dataset still ships it.
        tasks = [t for t in tasks if t.get("task_id") != "0-0-0-0-0"]

        # Normalise Level to int when possible (HF may serialise it as a string).
        for t in tasks:
            lvl = t.get("Level")
            try:
                t["Level"] = int(lvl)
            except (TypeError, ValueError):
                pass
        return tasks

    # ------------------------------------------------------------- task prep
    def prepare_task(self, task: Dict[str, Any]) -> "tuple[bool, Optional[str]]":
        """Append any attached-file hint to the Question (ported from gaia.py).

        Uses `file_path` (the local cached path the HF dataset provides) rather than
        the bare `file_name`. Returns (ok, info); ok=False means skip this task.
        """
        file_path = task.get("file_path") or ""
        if not file_path:
            return True, None

        if not os.path.exists(file_path):
            return False, f"Skipping task because file not found: {file_path}"

        suffix = os.path.splitext(file_path)[1].lower()
        if suffix in (".pdf", ".docx", ".doc", ".txt"):
            task["Question"] += f" Here are the necessary document files: {file_path}"
        elif suffix in (".jpg", ".jpeg", ".png"):
            task["Question"] += f" Here are the necessary image files: {file_path}"
        elif suffix in (".xlsx", ".xls", ".csv"):
            task["Question"] += (
                f" Here are the necessary table files: {file_path}, for processing excel "
                f"file, you can write python code and leverage excel toolkit to process "
                f"the file step-by-step and get the information."
            )
        elif suffix in (".py",):
            task["Question"] += f" Here are the necessary python files: {file_path}"
        else:
            task["Question"] += f" Here are the necessary files: {file_path}"

        return True, None

    # -------------------------------------------------------------- results io
    @staticmethod
    def save_results(results: List[Dict[str, Any]], file_path: str) -> None:
        base_dir = os.path.dirname(file_path)
        if base_dir:
            os.makedirs(base_dir, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

    @staticmethod
    def summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        correct = sum(1 for r in results if r.get("score"))
        total = len(results)
        return {
            "total": total,
            "correct": correct,
            "results": results,
            "accuracy": correct / total if total > 0 else 0.0,
        }

    # ----------------------------------------------------------------- scoring
    # The following are ported verbatim from utils/gaia.py (official GAIA scorer).
    def question_scorer(self, model_answer: str, ground_truth: str) -> bool:
        def is_float(element: Any) -> bool:
            try:
                float(element)
                return True
            except (ValueError, TypeError):
                return False

        if is_float(ground_truth):
            normalized_answer = self.normalize_number_str(model_answer)
            return normalized_answer == float(ground_truth)

        elif any(char in ground_truth for char in [",", ";"]):
            gt_elems = self.split_string(ground_truth)
            ma_elems = self.split_string(model_answer)
            if len(gt_elems) != len(ma_elems):
                return False
            comparisons = []
            for ma_elem, gt_elem in zip(ma_elems, gt_elems):
                if is_float(gt_elem):
                    normalized_ma_elem = self.normalize_number_str(ma_elem)
                    comparisons.append(normalized_ma_elem == float(gt_elem))
                else:
                    ma_elem = self.normalize_str(ma_elem, remove_punct=False)
                    gt_elem = self.normalize_str(gt_elem, remove_punct=False)
                    comparisons.append(ma_elem == gt_elem)
            return all(comparisons)
        else:
            ma_elem = self.normalize_str(model_answer)
            gt_elem = self.normalize_str(ground_truth)
            return ma_elem == gt_elem

    @staticmethod
    def normalize_number_str(number_str: str) -> float:
        for char in ["$", "%", ","]:
            number_str = str(number_str).replace(char, "")
        try:
            return float(number_str)
        except ValueError:
            return float("inf")

    @staticmethod
    def split_string(s: str, char_list: Optional[List[str]] = None) -> List[str]:
        if char_list is None:
            char_list = [",", ";"]
        pattern = f"[{''.join(char_list)}]"
        return re.split(pattern, s)

    @staticmethod
    def normalize_str(input_str, remove_punct: bool = True) -> str:
        no_spaces = re.sub(r"\s", "", str(input_str))
        if remove_punct:
            translator = str.maketrans("", "", string.punctuation)
            return no_spaces.lower().translate(translator)
        return no_spaces.lower()


# GAIA's strict-format reformatting prompt (ported from gaia.py get_formal_answer),
# but with the model call left to the caller so we don't depend on camel.
FORMAL_ANSWER_PROMPT = """\
I am solving a question:
<question>
{question}
</question>

Now, I have solved the question, the primary answer is as follows:
<answer>
{text}
</answer>

Now, I need you to determine the final answer. Do not try to solve the question, just \
pay attention to ONLY the format in which the answer is presented. DO NOT CHANGE THE \
MEANING OF THE PRIMARY ANSWER.
You should first analyze the answer format required by the question and then output the \
final answer that meets the format requirements.
Here are the requirements for the final answer:
<requirements>
The final answer must be output exactly in the format specified by the question. Your \
final answer should be a number OR as few words as possible OR a comma separated list of \
numbers and/or strings.
If you are asked for a number, don't use comma to write your number neither use units \
such as $ or percent sign unless specified otherwise. Numbers do not need to be written \
as words, but as digits.
If you are asked for a string, don't use articles, neither abbreviations (e.g. for \
cities), and write the digits in plain text unless specified otherwise.
If you are asked for a comma separated list, apply the above rules depending of whether \
the element to be put in the list is a number or a string.
</requirements>

Please output with the final answer according to the requirements without any other \
text. If the primary answer is already a final answer with the correct format, just \
output the primary answer.
"""
