from dataclasses import dataclass
from typing import List, Optional
import json


@dataclass
class ProbeExample:
    id: str
    benchmark: str  # "mmlu_pro" or "arc_easy"
    question: str
    choices: List[str]
    target_label: str
    prompt_text: str = ""
    input_ids: Optional[List[int]] = None
    student_correct: bool = False
    teacher_correct: bool = False
    student_answer_T1: str = ""
    teacher_answer: str = ""


@dataclass
class ProbeSet:
    probes: List[ProbeExample]
    checkpoint_source: str
    seed: int = 42

    def to_dict(self) -> dict:
        return {
            "probes": [
                {
                    "id": p.id,
                    "benchmark": p.benchmark,
                    "question": p.question,
                    "choices": p.choices,
                    "target_label": p.target_label,
                    "prompt_text": p.prompt_text,
                    "input_ids": p.input_ids,
                    "student_correct": p.student_correct,
                    "teacher_correct": p.teacher_correct,
                    "student_answer_T1": p.student_answer_T1,
                    "teacher_answer": p.teacher_answer,
                }
                for p in self.probes
            ],
            "checkpoint_source": self.checkpoint_source,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProbeSet":
        probes = [
            ProbeExample(
                id=p["id"],
                benchmark=p["benchmark"],
                question=p["question"],
                choices=p["choices"],
                target_label=p["target_label"],
                prompt_text=p.get("prompt_text", ""),
                input_ids=p.get("input_ids"),
                student_correct=p.get("student_correct", False),
                teacher_correct=p.get("teacher_correct", False),
                student_answer_T1=p.get("student_answer_T1", ""),
                teacher_answer=p.get("teacher_answer", ""),
            )
            for p in data["probes"]
        ]
        return cls(
            probes=probes,
            checkpoint_source=data["checkpoint_source"],
            seed=data.get("seed", 42),
        )


def _load_stress_test_results(path: str) -> List[dict]:
    """Load stress test results and flatten detailed_results from T-grouped structure."""
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Results file not found: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in results: {e}")

    # Stress test results are grouped by T (num_steps), each with detailed_results
    all_results = []
    if isinstance(data, dict) and "results" in data:
        for t_group in data["results"]:
            if isinstance(t_group, dict) and "detailed_results" in t_group:
                all_results.extend(t_group["detailed_results"])
    elif isinstance(data, list):
        all_results = data
    else:
        raise ValueError(f"Unexpected results structure in {path}")

    return all_results


def _is_mmlu_wrong_student_t1(item: dict) -> bool:
    """Check if MMLU question has student wrong at T=1."""
    # MMLU uses letters A-G for answers; we want questions where answer is E-J (harder)
    # and student was wrong at T=1
    correct_answer = item.get("correct_answer", "")
    is_hard = correct_answer in "EFGHIJ"
    num_steps = item.get("num_steps", 1)
    is_wrong = not item.get("is_correct", True)
    return is_hard and num_steps == 1 and is_wrong


def _is_arc_teacher_right_student_wrong_t1(item: dict) -> bool:
    """Check if ARC question has teacher correct but student wrong at T=1."""
    # For ARC we need to check if teacher got it right - but stress test only has student results
    # We check if student was wrong at T=1
    num_steps = item.get("num_steps", 1)
    is_wrong = not item.get("is_correct", True)
    return num_steps == 1 and is_wrong


def select_probes(
    mmlu_pro_path: str,
    arc_easy_path: str,
    min_mmlu: int = 20,
    min_arc: int = 20,
) -> List[ProbeExample]:
    probes = []

    # Load MMLU results
    mmlu_data = _load_stress_test_results(mmlu_pro_path)
    mmlu_candidates = [item for item in mmlu_data if _is_mmlu_wrong_student_t1(item)]

    for item in mmlu_candidates[:min_mmlu]:
        probes.append(ProbeExample(
            id=item.get("id", f"mmlu_{len(probes)}"),
            benchmark="mmlu_pro",
            question=item.get("question", ""),
            choices=item.get("options", []),  # stress test uses "options"
            target_label=item.get("correct_answer", ""),
            student_correct=item.get("is_correct", False),
            teacher_correct=False,  # Not available in stress test
            student_answer_T1=item.get("predicted_answer", ""),
            teacher_answer="",
            prompt_text=item.get("prompt_text", ""),
        ))

    # Load ARC results
    arc_data = _load_stress_test_results(arc_easy_path)
    arc_candidates = [item for item in arc_data if _is_arc_teacher_right_student_wrong_t1(item)]

    for item in arc_candidates[:min_arc]:
        probes.append(ProbeExample(
            id=item.get("id", f"arc_{len(probes)}"),
            benchmark="arc_easy",
            question=item.get("question", ""),
            choices=item.get("options", []),  # stress test uses "options"
            target_label=item.get("correct_answer", ""),
            student_correct=item.get("is_correct", False),
            teacher_correct=False,  # Not available in stress test
            student_answer_T1=item.get("predicted_answer", ""),
            teacher_answer="",
            prompt_text=item.get("prompt_text", ""),
        ))

    return probes
