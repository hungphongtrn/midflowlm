from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class FlowTrace:
    probe_id: str
    benchmark: str
    T: int
    endpoint_hidden_norm: float
    per_step_velocity_norms: List[float] = field(default_factory=list)
    trajectory_endpoint_norm: float = 0.0
    trajectory_divergence_from_T1: float = 0.0
    teacher_anchor_distances: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "probe_id": self.probe_id,
            "benchmark": self.benchmark,
            "T": self.T,
            "endpoint_hidden_norm": self.endpoint_hidden_norm,
            "per_step_velocity_norms": self.per_step_velocity_norms,
            "trajectory_endpoint_norm": self.trajectory_endpoint_norm,
            "trajectory_divergence_from_T1": self.trajectory_divergence_from_T1,
            "teacher_anchor_distances": self.teacher_anchor_distances,
        }


@dataclass
class DecoderTrace:
    probe_id: str
    benchmark: str
    T: int
    logits_answer_tokens: Dict[str, float] = field(default_factory=dict)
    predicted_answer: str = ""
    predicted_token_id: int = -1
    ground_truth_label: str = ""
    parsed_answer_match: bool = False
    teacher_logits_answer_tokens: Dict[str, float] = field(default_factory=dict)
    kl_divergence: float = 0.0
    js_divergence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "probe_id": self.probe_id,
            "benchmark": self.benchmark,
            "T": self.T,
            "logits_answer_tokens": self.logits_answer_tokens,
            "predicted_answer": self.predicted_answer,
            "predicted_token_id": self.predicted_token_id,
            "ground_truth_label": self.ground_truth_label,
            "parsed_answer_match": self.parsed_answer_match,
            "teacher_logits_answer_tokens": self.teacher_logits_answer_tokens,
            "kl_divergence": self.kl_divergence,
            "js_divergence": self.js_divergence,
        }
