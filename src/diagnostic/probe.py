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


def select_probes(
    mmlu_pro_path: str,
    arc_easy_path: str,
    min_mmlu: int = 20,
    min_arc: int = 20,
) -> List[ProbeExample]:
    probes = []

    try:
        with open(mmlu_pro_path) as f:
            mmlu_data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"MMLU-Pro results file not found: {mmlu_pro_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in MMLU-Pro results: {e}")
    mmlu_candidates = [
        item for item in mmlu_data
        if item.get("answer", "") in "EFGHIJ"
        and item.get("student_correct_T1", True) is False
    ]
    for item in mmlu_candidates[:min_mmlu]:
        probes.append(ProbeExample(
            id=item.get("id", ""),
            benchmark="mmlu_pro",
            question=item.get("question", ""),
            choices=item.get("choices", []),
            target_label=item.get("answer", ""),
            student_correct=item.get("student_correct_T1", False),
            teacher_correct=item.get("teacher_correct", False),
            student_answer_T1=item.get("student_answer_T1", ""),
            teacher_answer=item.get("teacher_answer", ""),
        ))

    try:
        with open(arc_easy_path) as f:
            arc_data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"ARC-Easy results file not found: {arc_easy_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in ARC-Easy results: {e}")
    arc_candidates = [
        item for item in arc_data
        if item.get("teacher_correct") is True
        and item.get("student_correct_T1") is False
    ]
    for item in arc_candidates[:min_arc]:
        probes.append(ProbeExample(
            id=item.get("id", ""),
            benchmark="arc_easy",
            question=item.get("question", ""),
            choices=item.get("choices", []),
            target_label=item.get("answer", ""),
            student_correct=item.get("student_correct_T1", False),
            teacher_correct=item.get("teacher_correct", True),
            student_answer_T1=item.get("student_answer_T1", ""),
            teacher_answer=item.get("teacher_answer", ""),
        ))

    return probes
