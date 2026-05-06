import json
import pytest
import numpy as np
from pathlib import Path


class TestFlowAnalysisResult:
    def test_dataclass_fields(self):
        from src.diagnostic.analysis import FlowAnalysisResult
        result = FlowAnalysisResult(
            per_T_stats={
                1: {"mean": 0.5, "std": 0.1, "median": 0.48, "n": 10},
                8: {"mean": 0.7, "std": 0.15, "median": 0.68, "n": 10},
            },
            pairwise_tests=[
                {"T_a": 1, "T_b": 8, "t_stat": 3.5, "p_value": 0.001, "cohens_d": 0.95, "mean_delta": 0.2, "significant": True, "interpretation": "large"},
            ],
            velocity_analysis={
                1: {"mean": 0.1, "std": 0.05, "median": 0.09, "nonzero_count": 10},
                8: {"mean": 0.08, "std": 0.04, "median": 0.07, "nonzero_count": 10},
            },
            divergence_from_T1={
                2: {"mean": 0.05, "median": 0.04, "proportion_diverged": 0.8},
                8: {"mean": 0.12, "median": 0.11, "proportion_diverged": 1.0},
                64: {"mean": 0.18, "median": 0.17, "proportion_diverged": 1.0},
            },
        )
        assert result.per_T_stats[1]["mean"] == 0.5
        assert result.pairwise_tests[0]["significant"] is True
        assert result.divergence_from_T1[8]["proportion_diverged"] == 1.0


class TestDecoderAnalysisResult:
    def test_dataclass_fields(self):
        from src.diagnostic.analysis import DecoderAnalysisResult
        result = DecoderAnalysisResult(
            per_T_stats={
                1: {"accuracy": 0.0, "mean_kl": 0.5, "mean_js": 0.3, "median_kl": 0.48, "median_js": 0.28, "n": 10},
                8: {"accuracy": 0.0, "mean_kl": 1.2, "mean_js": 0.7, "median_kl": 1.15, "median_js": 0.68, "n": 10},
            },
            logit_shift_tests=[
                {"T_a": 1, "T_b": 8, "mean_kl_delta": 0.8, "proportion_shifted": 0.9},
            ],
            answer_coverage={
                1: {"mean_prob": 0.1, "median_prob": 0.08, "max_prob": 0.15, "probes_gt_50pct": 0},
                8: {"mean_prob": 0.12, "median_prob": 0.09, "max_prob": 0.18, "probes_gt_50pct": 0},
            },
            prediction_stability={
                "always_wrong": 8, "flipped": 2, "became_correct": 0,
            },
        )
        assert result.per_T_stats[1]["accuracy"] == 0.0
        assert result.prediction_stability["always_wrong"] == 8
        assert result.logit_shift_tests[0]["mean_kl_delta"] == 0.8
