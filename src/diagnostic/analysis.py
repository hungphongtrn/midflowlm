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


answer_labels = "ABCDEFGHIJ"


def _load_flow_traces(traces_dir, T_values):
    import json
    from pathlib import Path

    traces_by_T = {}
    for T in T_values:
        path = Path(traces_dir) / f"T{T}" / "flow_traces.json"
        with open(path) as f:
            data = json.load(f)
        traces_by_T[T] = list(data.values())
    return traces_by_T


def _compute_per_T_stats(traces, field):
    import numpy as np

    values = np.array([t[field] for t in traces])
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "median": float(np.median(values)),
        "n": len(values),
    }


def _cohens_d(values1, values2):
    import numpy as np

    diff = np.mean(values1 - values2)
    var1 = np.var(values1, ddof=1)
    var2 = np.var(values2, ddof=1)
    n1, n2 = len(values1), len(values2)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-12:
        return 0.0
    return float(diff / pooled_std)


def _cohens_d_interpretation(d):
    if abs(d) < 0.5:
        return "small"
    elif abs(d) < 0.8:
        return "medium"
    else:
        return "large"


def analyze_flow(traces_dir, T_values):
    import numpy as np
    from scipy.stats import ttest_rel

    traces_by_T = _load_flow_traces(traces_dir, T_values)

    per_T_stats = {}
    for T in T_values:
        per_T_stats[T] = _compute_per_T_stats(traces_by_T[T], "endpoint_hidden_norm")

    pairwise_tests = []
    n_pairs = len(T_values) * (len(T_values) - 1) // 2
    bonferroni_alpha = 0.05 / max(n_pairs, 1)

    for i, T_a in enumerate(T_values):
        for T_b in T_values[i + 1:]:
            traces_a = {t["probe_id"]: t for t in traces_by_T[T_a]}
            traces_b = {t["probe_id"]: t for t in traces_by_T[T_b]}
            common_ids = sorted(set(traces_a) & set(traces_b))
            if len(common_ids) < 2:
                continue

            values_a = np.array([traces_a[pid]["endpoint_hidden_norm"] for pid in common_ids])
            values_b = np.array([traces_b[pid]["endpoint_hidden_norm"] for pid in common_ids])

            t_stat, p_value = ttest_rel(values_a, values_b)
            d = _cohens_d(values_a, values_b)
            mean_delta = float(np.mean(values_b - values_a))
            significant = bool(p_value < bonferroni_alpha)

            pairwise_tests.append({
                "T_a": T_a,
                "T_b": T_b,
                "t_stat": float(t_stat),
                "p_value": float(p_value),
                "cohens_d": float(d),
                "mean_delta": mean_delta,
                "significant": significant,
                "interpretation": _cohens_d_interpretation(d),
            })

    velocity_analysis = {}
    for T in T_values:
        all_velocity_means = []
        nonzero_count = 0
        for t in traces_by_T[T]:
            vels = t.get("per_step_velocity_norms", [])
            if vels:
                mean_vel = np.mean(vels)
                all_velocity_means.append(mean_vel)
                if mean_vel > 1e-6:
                    nonzero_count += 1
        if all_velocity_means:
            vals = np.array(all_velocity_means)
            velocity_analysis[T] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)),
                "median": float(np.median(vals)),
                "nonzero_count": nonzero_count,
            }
        else:
            velocity_analysis[T] = {"mean": 0.0, "std": 0.0, "median": 0.0, "nonzero_count": 0}

    divergence_from_T1 = {}
    for T in T_values[1:]:
        div_values = []
        diverged_count = 0
        for t in traces_by_T[T]:
            divergence = t.get("trajectory_divergence_from_T1", 0.0)
            div_values.append(divergence)
            if divergence > 1e-6:
                diverged_count += 1
        if div_values:
            vals = np.array(div_values)
            divergence_from_T1[T] = {
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "proportion_diverged": diverged_count / len(div_values),
            }
        else:
            divergence_from_T1[T] = {"mean": 0.0, "median": 0.0, "proportion_diverged": 0.0}

    return FlowAnalysisResult(
        per_T_stats=per_T_stats,
        pairwise_tests=pairwise_tests,
        velocity_analysis=velocity_analysis,
        divergence_from_T1=divergence_from_T1,
    )


def _load_decoder_traces(traces_dir, T_values):
    import json
    from pathlib import Path

    traces_by_T = {}
    for T in T_values:
        path = Path(traces_dir) / f"T{T}" / "decoder_traces.json"
        with open(path) as f:
            data = json.load(f)
        traces_by_T[T] = list(data.values())
    return traces_by_T


def _probe_kl(a_logits, b_logits):
    import numpy as np

    labels = sorted(set(a_logits.keys()) & set(b_logits.keys()))
    if not labels:
        return 0.0
    a_vals = np.array([a_logits[l] for l in labels])
    b_vals = np.array([b_logits[l] for l in labels])
    a_probs = np.exp(a_vals - np.max(a_vals))
    a_probs = a_probs / a_probs.sum()
    b_probs = np.exp(b_vals - np.max(b_vals))
    b_probs = b_probs / b_probs.sum()
    kl = np.sum(a_probs * np.log((a_probs + 1e-12) / (b_probs + 1e-12)))
    return float(max(kl, 0.0))


def analyze_decoder(traces_dir, T_values):
    import numpy as np

    traces_by_T = _load_decoder_traces(traces_dir, T_values)

    per_T_stats = {}
    for T in T_values:
        traces = traces_by_T[T]
        n = len(traces)
        accuracy = sum(1 for t in traces if t.get("parsed_answer_match", False)) / n if n > 0 else 0.0
        kl_vals = np.array([t.get("kl_divergence", 0.0) for t in traces])
        js_vals = np.array([t.get("js_divergence", 0.0) for t in traces])
        per_T_stats[T] = {
            "accuracy": accuracy,
            "mean_kl": float(np.mean(kl_vals)),
            "mean_js": float(np.mean(js_vals)),
            "median_kl": float(np.median(kl_vals)),
            "median_js": float(np.median(js_vals)),
            "n": n,
        }

    logit_shift_tests = []
    for i, T_a in enumerate(T_values):
        for T_b in T_values[i + 1:]:
            traces_a = {t["probe_id"]: t for t in traces_by_T[T_a]}
            traces_b = {t["probe_id"]: t for t in traces_by_T[T_b]}
            common_ids = sorted(set(traces_a) & set(traces_b))
            if len(common_ids) < 1:
                continue
            kl_deltas = []
            shifted_count = 0
            for pid in common_ids:
                a_logits = traces_a[pid].get("logits_answer_tokens", {})
                b_logits = traces_b[pid].get("logits_answer_tokens", {})
                kl = _probe_kl(a_logits, b_logits)
                kl_deltas.append(kl)
                if kl > 1e-3:
                    shifted_count += 1
            logit_shift_tests.append({
                "T_a": T_a,
                "T_b": T_b,
                "mean_kl_delta": float(np.mean(kl_deltas)),
                "proportion_shifted": shifted_count / len(kl_deltas) if kl_deltas else 0.0,
            })

    answer_coverage = {}
    for T in T_values:
        prob_on_correct_vals = []
        gt_50_count = 0
        for t in traces_by_T[T]:
            logits = t.get("logits_answer_tokens", {})
            gt_label = t.get("ground_truth_label", "")
            if logits and gt_label in answer_labels:
                vals = np.array(list(logits.values()))
                probs = np.exp(vals - np.max(vals))
                probs = probs / probs.sum()
                label_idx = answer_labels.index(gt_label)
                prob = probs[label_idx]
                prob_on_correct_vals.append(prob)
                if prob > 0.5:
                    gt_50_count += 1
        if prob_on_correct_vals:
            vals = np.array(prob_on_correct_vals)
            answer_coverage[T] = {
                "mean_prob": float(np.mean(vals)),
                "median_prob": float(np.median(vals)),
                "max_prob": float(np.max(vals)),
                "probes_gt_50pct": gt_50_count,
            }
        else:
            answer_coverage[T] = {"mean_prob": 0.0, "median_prob": 0.0, "max_prob": 0.0, "probes_gt_50pct": 0}

    all_ids = sorted(set().union(*[
        {t["probe_id"] for t in traces_by_T[T]}
        for T in T_values
    ]))
    always_wrong = 0
    flipped = 0
    became_correct = 0
    for pid in all_ids:
        predictions = []
        matches = []
        for T in T_values:
            pid_traces = {t["probe_id"]: t for t in traces_by_T[T]}
            if pid in pid_traces:
                predictions.append(pid_traces[pid].get("predicted_answer", ""))
                matches.append(pid_traces[pid].get("parsed_answer_match", False))
        if not predictions:
            continue
        unique_preds = set(predictions)
        if not any(matches):
            always_wrong += 1
        if len(unique_preds) > 1:
            flipped += 1
        if not matches[0] and any(matches):
            became_correct += 1

    prediction_stability = {
        "always_wrong": always_wrong,
        "flipped": flipped,
        "became_correct": became_correct,
    }

    return DecoderAnalysisResult(
        per_T_stats=per_T_stats,
        logit_shift_tests=logit_shift_tests,
        answer_coverage=answer_coverage,
        prediction_stability=prediction_stability,
    )
