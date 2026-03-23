"""Evaluation module for baseline comparisons and metrics."""

from src.eval.baselines import (
    IdentityBaseline,
    T1SharedBlockBaseline,
    SimpleRecurrentBaseline,
    MetricsReport,
    compute_endpoint_error,
    compute_trajectory_error,
    compute_kl_divergence,
    compute_perplexity,
    compute_latency_metrics,
    compute_stability_metrics,
)
from src.eval.text_checkpoint_sweep import (
    run_text_sweep,
    compute_repetition_metrics,
    aggregate_repetition_metrics,
)
from src.eval.mmlu_pro_behavior import (
    extract_first_valid_answer,
    run_mmlu_pro_behavior_observation,
    summarize_behavior_records,
)

__all__ = [
    "IdentityBaseline",
    "T1SharedBlockBaseline",
    "SimpleRecurrentBaseline",
    "MetricsReport",
    "compute_endpoint_error",
    "compute_trajectory_error",
    "compute_kl_divergence",
    "compute_perplexity",
    "compute_latency_metrics",
    "compute_stability_metrics",
    "run_text_sweep",
    "compute_repetition_metrics",
    "aggregate_repetition_metrics",
    "extract_first_valid_answer",
    "run_mmlu_pro_behavior_observation",
    "summarize_behavior_records",
]
