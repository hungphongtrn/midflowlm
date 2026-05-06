from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class FlowAnalysisResult:
    per_T_stats: Dict[int, dict] = field(default_factory=dict)
    pairwise_tests: List[dict] = field(default_factory=list)
    velocity_analysis: Dict[int, dict] = field(default_factory=dict)
    divergence_from_T1: Dict[int, dict] = field(default_factory=dict)


@dataclass
class DecoderAnalysisResult:
    per_T_stats: Dict[int, dict] = field(default_factory=dict)
    logit_shift_tests: List[dict] = field(default_factory=list)
    answer_coverage: Dict[int, dict] = field(default_factory=dict)
    prediction_stability: Dict[str, int] = field(default_factory=dict)
