import json
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


class TestAnalyzeDecoder:
    def test_analyze_decoder_accuracy(self, tmp_path):
        traces_dir = Path(tmp_path) / "traces"
        probes = [f"probe_{i:03d}" for i in range(4)]
        for T in [1, 8]:
            T_dir = traces_dir / f"T{T}"
            T_dir.mkdir(parents=True, exist_ok=True)
            flow = {pid: {"probe_id": pid, "benchmark": "mmlu_pro", "T": T, "endpoint_hidden_norm": 0.1, "per_step_velocity_norms": [0.1] * T, "trajectory_endpoint_norm": 0.1, "trajectory_divergence_from_T1": 0.0, "teacher_anchor_distances": {}} for pid in probes}
            decoder = {}
            for i, pid in enumerate(probes):
                decoder[pid] = {
                    "probe_id": pid, "benchmark": "mmlu_pro", "T": T,
                    "logits_answer_tokens": {l: -1.0 for l in "ABCDEFGHIJ"},
                    "predicted_answer": "C", "predicted_token_id": 17627,
                    "ground_truth_label": "C" if i < 2 else "E",
                    "parsed_answer_match": i < 2,
                    "teacher_logits_answer_tokens": {l: -2.0 for l in "ABCDEFGHIJ"},
                    "kl_divergence": 0.5, "js_divergence": 0.3,
                }
            with open(T_dir / "flow_traces.json", "w") as f:
                json.dump(flow, f)
            with open(T_dir / "decoder_traces.json", "w") as f:
                json.dump(decoder, f)
        from src.diagnostic.analysis import analyze_decoder
        result = analyze_decoder(str(traces_dir), [1, 8])
        assert result.per_T_stats[1]["accuracy"] == 0.5
        assert result.per_T_stats[8]["accuracy"] == 0.5

    def test_analyze_decoder_coverage(self, tmp_path):
        traces_dir = Path(tmp_path) / "traces"
        probes = [f"probe_{i:03d}" for i in range(3)]
        for T in [1, 8]:
            T_dir = traces_dir / f"T{T}"
            T_dir.mkdir(parents=True, exist_ok=True)
            flow = {pid: {"probe_id": pid, "benchmark": "mmlu_pro", "T": T, "endpoint_hidden_norm": 0.1, "per_step_velocity_norms": [0.1] * T, "trajectory_endpoint_norm": 0.1, "trajectory_divergence_from_T1": 0.0, "teacher_anchor_distances": {}} for pid in probes}
            decoder = {}
            for i, pid in enumerate(probes):
                probs = {"A": 0.1, "B": 0.55, "C": 0.2, "D": 0.1, "E": 0.05, "F": 0.0, "G": 0.0, "H": 0.0, "I": 0.0, "J": 0.0}
                decoder[pid] = {
                    "probe_id": pid, "benchmark": "mmlu_pro", "T": T,
                    "logits_answer_tokens": {l: np.log(p) for l, p in probs.items()},
                    "predicted_answer": "B", "predicted_token_id": 17626,
                    "ground_truth_label": "B",
                    "parsed_answer_match": True,
                    "teacher_logits_answer_tokens": {},
                    "kl_divergence": 0.0, "js_divergence": 0.0,
                }
            with open(T_dir / "flow_traces.json", "w") as f:
                json.dump(flow, f)
            with open(T_dir / "decoder_traces.json", "w") as f:
                json.dump(decoder, f)
        from src.diagnostic.analysis import analyze_decoder
        result = analyze_decoder(str(traces_dir), [1, 8])
        assert result.answer_coverage[1]["probes_gt_50pct"] == 3

    def test_analyze_decoder_stability(self, tmp_path):
        traces_dir = Path(tmp_path) / "traces"
        probes = [f"probe_{i:03d}" for i in range(4)]
        for T_idx, T in enumerate([1, 8, 64]):
            T_dir = traces_dir / f"T{T}"
            T_dir.mkdir(parents=True, exist_ok=True)
            flow = {pid: {"probe_id": pid, "benchmark": "mmlu_pro", "T": T, "endpoint_hidden_norm": 0.1, "per_step_velocity_norms": [0.1] * T, "trajectory_endpoint_norm": 0.1, "trajectory_divergence_from_T1": 0.0, "teacher_anchor_distances": {}} for pid in probes}
            decoder = {}
            for i, pid in enumerate(probes):
                answers = [[["C", False], ["C", False], ["C", False]],
                           [["C", False], ["D", False], ["E", False]],
                           [["C", False], ["C", False], ["C", True]],
                           [["D", True], ["D", True], ["D", True]]]
                pred, match = answers[i][T_idx]
                decoder[pid] = {
                    "probe_id": pid, "benchmark": "mmlu_pro", "T": T,
                    "logits_answer_tokens": {l: -1.0 for l in "ABCDEFGHIJ"},
                    "predicted_answer": pred, "predicted_token_id": 17627,
                    "ground_truth_label": "D",
                    "parsed_answer_match": match,
                    "teacher_logits_answer_tokens": {},
                    "kl_divergence": 0.0, "js_divergence": 0.0,
                }
            with open(T_dir / "flow_traces.json", "w") as f:
                json.dump(flow, f)
            with open(T_dir / "decoder_traces.json", "w") as f:
                json.dump(decoder, f)
        from src.diagnostic.analysis import analyze_decoder
        result = analyze_decoder(str(traces_dir), [1, 8, 64])
        assert result.prediction_stability["always_wrong"] == 2
        assert result.prediction_stability["flipped"] == 1
        assert result.prediction_stability["became_correct"] == 1

    def test_analyze_decoder_logit_shift(self, tmp_path):
        traces_dir = Path(tmp_path) / "traces"
        probes = [f"probe_{i:03d}" for i in range(3)]
        for T in [1, 8]:
            T_dir = traces_dir / f"T{T}"
            T_dir.mkdir(parents=True, exist_ok=True)
            flow = {pid: {"probe_id": pid, "benchmark": "mmlu_pro", "T": T, "endpoint_hidden_norm": 0.1, "per_step_velocity_norms": [0.1] * T, "trajectory_endpoint_norm": 0.1, "trajectory_divergence_from_T1": 0.0, "teacher_anchor_distances": {}} for pid in probes}
            decoder = {}
            for i, pid in enumerate(probes):
                labels = "ABCDEFGHIJ"
                t_factor = 1.0 if T == 1 else 5.0
                decoder[pid] = {
                    "probe_id": pid, "benchmark": "mmlu_pro", "T": T,
                    "logits_answer_tokens": {
                        l: t_factor * (ord(l) - 64) for l in labels
                    },
                    "predicted_answer": "C", "predicted_token_id": 17627,
                    "ground_truth_label": "E",
                    "parsed_answer_match": False,
                    "teacher_logits_answer_tokens": {},
                    "kl_divergence": 0.0, "js_divergence": 0.0,
                }
            with open(T_dir / "flow_traces.json", "w") as f:
                json.dump(flow, f)
            with open(T_dir / "decoder_traces.json", "w") as f:
                json.dump(decoder, f)
        from src.diagnostic.analysis import analyze_decoder
        result = analyze_decoder(str(traces_dir), [1, 8])
        shift = [t for t in result.logit_shift_tests if t["T_a"] == 1 and t["T_b"] == 8][0]
        assert shift["mean_kl_delta"] > 0


class TestRunAnalysis:
    def test_run_analysis_integration(self, tmp_path):
        traces_dir, probes, _ = _make_fake_traces(tmp_path, [1, 8, 64])
        from src.diagnostic.analysis import run_analysis, FlowAnalysisResult, DecoderAnalysisResult
        flow_result, decoder_result = run_analysis(traces_dir, [1, 8, 64])
        assert isinstance(flow_result, FlowAnalysisResult)
        assert isinstance(decoder_result, DecoderAnalysisResult)
        assert 1 in flow_result.per_T_stats
        assert 1 in decoder_result.per_T_stats
