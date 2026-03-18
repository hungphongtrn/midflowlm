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
from src.eval.text_checkpoint_sweep import run_text_sweep

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
]
