"""SFT evaluation metrics — accuracy, oracle, prediction-change rate."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SFTSampleResult:
    question_id: int
    question: str
    correct_answer: str
    predicted_answer: str
    is_correct: bool
    num_steps: int
    latency_ms: float
    category: str = "unknown"

    def to_dict(self) -> Dict:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "correct_answer": self.correct_answer,
            "predicted_answer": self.predicted_answer,
            "is_correct": self.is_correct,
            "num_steps": self.num_steps,
            "latency_ms": self.latency_ms,
            "category": self.category,
        }


@dataclass
class MultiTReport:
    experiment_id: str
    t_values: List[int]
    per_t_accuracy: Dict[int, float]
    per_t_correct: Dict[int, int]
    num_total: int
    oracle_accuracy: float
    oracle_correct: int
    prediction_change_rate: float
    avg_latency_by_t: Dict[int, float]
    detailed_results: List[SFTSampleResult] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            f"SFT MULTI-T EVALUATION — {self.experiment_id}",
            "=" * 60,
            f"Total questions: {self.num_total}",
            "",
            "Per-T Accuracy:",
        ]
        for t in self.t_values:
            acc = self.per_t_accuracy.get(t, 0.0)
            corr = self.per_t_correct.get(t, 0)
            lat = self.avg_latency_by_t.get(t, 0.0)
            lines.append(f"  T={t:2d}: {acc:.1%} ({corr}/{self.num_total})  {lat:.0f}ms")

        lines.extend([
            "",
            f"Oracle (best-of-T):  {self.oracle_accuracy:.1%} ({self.oracle_correct}/{self.num_total})",
            f"Prediction-change rate: {self.prediction_change_rate:.1%}",
            "=" * 60,
        ])
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return {
            "experiment_id": self.experiment_id,
            "t_values": self.t_values,
            "per_t_accuracy": self.per_t_accuracy,
            "per_t_correct": self.per_t_correct,
            "num_total": self.num_total,
            "oracle_accuracy": self.oracle_accuracy,
            "oracle_correct": self.oracle_correct,
            "prediction_change_rate": self.prediction_change_rate,
            "avg_latency_by_t": self.avg_latency_by_t,
        }


def compute_fixed_t_accuracy(results: List[SFTSampleResult], t: int) -> float:
    """Accuracy at a specific thinking level T."""
    t_results = [r for r in results if r.num_steps == t]
    if not t_results:
        return 0.0
    return sum(1 for r in t_results if r.is_correct) / len(t_results)


def compute_oracle_accuracy(results: List[SFTSampleResult], t_values: List[int]) -> float:
    """Oracle accuracy: correct if ANY T value gets the right answer.

    Groups results by question_id, and marks a question correct if
    at least one T value produces the correct answer.
    """
    grouped = {}
    for r in results:
        if r.question_id not in grouped:
            grouped[r.question_id] = False
        if r.is_correct:
            grouped[r.question_id] = True

    if not grouped:
        return 0.0
    return sum(1 for v in grouped.values() if v) / len(grouped)


def compute_prediction_change_rate(results: List[SFTSampleResult], t_values: List[int]) -> float:
    """Fraction of questions where the prediction changed between any two T values.

    A question shows prediction change if the predicted answer differs
    across at least one pair of T values.
    """
    grouped = {}
    for r in results:
        grouped.setdefault(r.question_id, {})[r.num_steps] = r.predicted_answer

    changed = 0
    for qid, t_answers in grouped.items():
        answers = list(t_answers.values())
        if len(set(answers)) > 1:
            changed += 1

    if not grouped:
        return 0.0
    return changed / len(grouped)


def compute_multi_t_report(
    results: List[SFTSampleResult],
    t_values: List[int],
    experiment_id: str = "",
) -> MultiTReport:
    """Compute full multi-T evaluation report from per-sample results."""
    num_total = len(set(r.question_id for r in results))
    if num_total == 0:
        return MultiTReport(
            experiment_id=experiment_id,
            t_values=t_values,
            per_t_accuracy={},
            per_t_correct={},
            num_total=0,
            oracle_accuracy=0.0,
            oracle_correct=0,
            prediction_change_rate=0.0,
            avg_latency_by_t={},
            detailed_results=results,
        )

    per_t_accuracy = {}
    per_t_correct = {}
    avg_latency_by_t = {}

    for t in t_values:
        t_results = [r for r in results if r.num_steps == t]
        if t_results:
            per_t_accuracy[t] = sum(1 for r in t_results if r.is_correct) / len(t_results)
            per_t_correct[t] = sum(1 for r in t_results if r.is_correct)
            avg_latency_by_t[t] = sum(r.latency_ms for r in t_results) / len(t_results)
        else:
            per_t_accuracy[t] = 0.0
            per_t_correct[t] = 0
            avg_latency_by_t[t] = 0.0

    oracle_correct_count = 0
    grouped = {}
    for r in results:
        grouped.setdefault(r.question_id, False)
        if r.is_correct:
            grouped[r.question_id] = True
    oracle_correct_count = sum(1 for v in grouped.values() if v)
    oracle_accuracy = oracle_correct_count / num_total if num_total > 0 else 0.0

    prediction_change_rate = compute_prediction_change_rate(results, t_values)

    return MultiTReport(
        experiment_id=experiment_id,
        t_values=t_values,
        per_t_accuracy=per_t_accuracy,
        per_t_correct=per_t_correct,
        num_total=num_total,
        oracle_accuracy=oracle_accuracy,
        oracle_correct=oracle_correct_count,
        prediction_change_rate=prediction_change_rate,
        avg_latency_by_t=avg_latency_by_t,
        detailed_results=results,
    )
