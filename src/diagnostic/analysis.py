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
