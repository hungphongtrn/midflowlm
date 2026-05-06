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


def _make_fake_traces(tmp_path, T_values, endpoint_norms=None):
    traces_dir = Path(tmp_path) / "traces"
    probes = [f"probe_{i:03d}" for i in range(5)]
    default_norms = {t: t * 0.01 for t in T_values}
    endpoint_norms = endpoint_norms or default_norms
    for T in T_values:
        T_dir = traces_dir / f"T{T}"
        T_dir.mkdir(parents=True, exist_ok=True)
        flow = {}
        decoder = {}
        for pid in probes:
            flow[pid] = {
                "probe_id": pid, "benchmark": "mmlu_pro", "T": T,
                "endpoint_hidden_norm": endpoint_norms[T] + hash(pid) % 100 * 0.001,
                "per_step_velocity_norms": [0.1] * T,
                "trajectory_endpoint_norm": endpoint_norms[T] + 0.01,
                "trajectory_divergence_from_T1": 0.0,
                "teacher_anchor_distances": {"h8": 0.3},
            }
            decoder[pid] = {
                "probe_id": pid, "benchmark": "mmlu_pro", "T": T,
                "logits_answer_tokens": {l: -1.0 + hash(pid) % 10 * 0.1 for l in "ABCDEFGHIJ"},
                "predicted_answer": "C", "predicted_token_id": 17627,
                "ground_truth_label": "E", "parsed_answer_match": False,
                "teacher_logits_answer_tokens": {l: -2.0 for l in "ABCDEFGHIJ"},
                "kl_divergence": 0.5, "js_divergence": 0.3,
            }
        with open(T_dir / "flow_traces.json", "w") as f:
            json.dump(flow, f)
        with open(T_dir / "decoder_traces.json", "w") as f:
            json.dump(decoder, f)
    return str(traces_dir), probes, endpoint_norms


class TestAnalyzeFlow:
    def test_analyze_flow_computes_stats(self, tmp_path):
        traces_dir, probes, _ = _make_fake_traces(tmp_path, [1, 8])
        from src.diagnostic.analysis import analyze_flow
        result = analyze_flow(traces_dir, [1, 8])
        assert 1 in result.per_T_stats
        assert 8 in result.per_T_stats
        assert result.per_T_stats[1]["n"] == len(probes)
        assert "mean" in result.per_T_stats[1]
        assert "std" in result.per_T_stats[1]

    def test_analyze_flow_no_change_detected(self, tmp_path):
        traces_dir, probes, _ = _make_fake_traces(tmp_path, [1, 8], endpoint_norms={1: 0.1, 8: 0.1})
        from src.diagnostic.analysis import analyze_flow
        result = analyze_flow(traces_dir, [1, 8])
        pair_test = [t for t in result.pairwise_tests if t["T_a"] == 1 and t["T_b"] == 8][0]
        assert pair_test["significant"] is False

    def test_analyze_flow_significant_change(self, tmp_path):
        traces_dir, probes, _ = _make_fake_traces(tmp_path, [1, 8], endpoint_norms={1: 0.1, 8: 0.5})
        from src.diagnostic.analysis import analyze_flow
        result = analyze_flow(traces_dir, [1, 8])
        pair_test = [t for t in result.pairwise_tests if t["T_a"] == 1 and t["T_b"] == 8][0]
        assert pair_test["significant"] is True
        assert abs(pair_test["cohens_d"]) > 0.5
