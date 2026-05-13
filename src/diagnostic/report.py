"""Diagnostic report generation for P3-D3 T-scaling investigation.

Reads AnalysisResult dataclasses and writes a self-contained markdown report.
"""



def _verdict_q1(flow):
    pair = None
    for p in flow.pairwise_tests:
        if p["T_a"] == 1 and p["T_b"] == 64:
            pair = p
            break
    if pair is None:
        return {"status": "FAIL", "verdict": "FAIL", "detail": "No T1->T64 comparison available"}
    mean_delta = abs(pair["mean_delta"])
    p_value = pair["p_value"]
    if mean_delta > 0.01 and p_value < 0.05:
        return {
            "status": "PASS",
            "verdict": "PASS",
            "detail": f"mean_delta={mean_delta:.4f}, p={p_value:.4f}, d={pair['cohens_d']:.2f} ({pair['interpretation']})",
        }
    return {
        "status": "FAIL",
        "verdict": "FAIL",
        "detail": f"mean_delta={mean_delta:.4f}, p={p_value:.4f} — hidden states do not change with T",
    }


def _verdict_q2(decoder):
    any_kl = False
    for s in decoder.logit_shift_tests:
        if s["mean_kl_delta"] > 0.01:
            any_kl = True
            break
    all_zero_accuracy = all(v["accuracy"] == 0.0 for v in decoder.per_T_stats.values())
    if any_kl and all_zero_accuracy:
        max_kl = max((s["mean_kl_delta"] for s in decoder.logit_shift_tests), default=0.0)
        return {"status": "PASS", "verdict": "PASS", "detail": f"Logits change (max pairwise KL={max_kl:.4f}) but predictions remain wrong — logit shifts don't reach correct answer"}
    if any_kl:
        max_kl = max((s["mean_kl_delta"] for s in decoder.logit_shift_tests), default=0.0)
        return {"status": "WARN", "verdict": "WARN", "detail": f"Logits change (max pairwise KL={max_kl:.4f}) and some predictions change too"}
    return {"status": "WARN", "verdict": "WARN", "detail": "Logits do not change measurably between T values"}


def _verdict_q3(decoder):
    max_mean_prob = max((v["mean_prob"] for v in decoder.answer_coverage.values()), default=0.0)
    if max_mean_prob < 0.3:
        return {"status": "FAIL", "verdict": "FAIL", "detail": f"Max mean prob on correct answer = {max_mean_prob:.3f} < 0.3 — answer-token supervision likely needed"}
    elif max_mean_prob < 0.5:
        return {"status": "WARN", "verdict": "WARN", "detail": f"Max mean prob on correct answer = {max_mean_prob:.3f} < 0.5 — answer coverage borderline"}
    else:
        return {"status": "PASS", "verdict": "PASS", "detail": f"Max mean prob on correct answer = {max_mean_prob:.3f} >= 0.5"}


def _determine_root_cause(v1, v2, v3):
    q1_pass = v1["status"] == "PASS"
    q2_pass = v2["status"] == "PASS"
    q3_pass = v3["status"] == "PASS"

    lines = [
        "```",
        "Q1: Hidden states change with T?",
    ]
    if not q1_pass:
        lines.extend([
            "├── NO → ROOT CAUSE: Flow integration dead",
            "│        Vector field returns near-zero, ODE solver broken, or timestep conditioning failing",
        ])
    else:
        lines.extend([
            "└── YES → Q2: Logits change without prediction change?",
        ])
        if not q2_pass:
            lines.extend([
            "    ├── NO → ROOT CAUSE: Flow integration too weak",
            "    │        Logits stuck despite hidden-state changes — timestep conditioning or target scale too small",
            ])
        else:
            lines.extend([
            "    └── YES → Q3: Probability reaches answer labels?",
            ])
            if not q3_pass:
                lines.extend([
                "        ├── NO → ROOT CAUSE: Answer-space collapse",
                "        │        Model can't put mass on correct labels — answer-token supervision or readout calibration needed",
                ])
            else:
                lines.extend([
                "        └── YES → ROOT CAUSE: Unknown threshold",
                "                 Something else limits accuracy — investigate data or training objective",
                ])
    lines.append("```")
    return "\n".join(lines)


def _generate_recommendations(v1, v2, v3):
    recommendations = []
    if v1["status"] == "FAIL":
        recommendations.extend([
            "1. **Fix flow integration** → Verify MidblockVectorField returns non-zero velocities; check ODE solver configuration (atol/rtol too loose? dt too small?) → Expected: endpoint_hidden_norm varies measurably with T → Effort: Medium",
            "2. **Add timestep-conditioning signal** → Ensure flow midblock receives and uses the timestep parameter; check `midblock.forward(t=...)` path → Expected: velocity vectors become timestep-dependent → Effort: Medium",
        ])
    elif v2["status"] == "PASS" and v3["status"] == "FAIL":
        recommendations.extend([
            "1. **Add answer-token supervision** → Train with cross-entropy on logits over A-J tokens; can use teacher_logits as soft targets → Expected: prob_on_correct > 0.5 for reachable probes → Effort: Low",
            "2. **Calibrate readout layer** → Scale or bias-correct the final logit projection over answer tokens; verify that hidden states carry sufficient signal → Expected: sharper answer distributions → Effort: Low",
            "3. **Increase target scale** → If hidden-state deltas are small, scale up trajectory targets (h8-h11 anchors) to produce stronger flow → Expected: larger hidden-state deltas with T → Effort: Medium",
        ])
    elif v2["status"] == "WARN":
        recommendations.extend([
            "1. **Audit flow integration strength** → Check whether hidden-state deltas from T1→T64 are statistically significant; if borderline, increase flow block capacity → Expected: clearer separation between T values → Effort: Medium",
            "2. **Review prediction parity** → Verify that logit shifts, when they occur, are in the direction of correct answer labels → Expected: flipped probes become correct more often → Effort: Low",
        ])
    else:
        recommendations.extend([
            "1. **Investigate data/training objective** → Accuracy may be limited by non-flow factors (data noise, insufficient teacher signal, wrong objective weight) → Expected: identify bottleneck → Effort: High",
        ])
    return "\n".join(recommendations)


def generate_report(flow_result, decoder_result, probes_path="", traces_dir=""):
    v1 = _verdict_q1(flow_result)
    v2 = _verdict_q2(decoder_result)
    v3 = _verdict_q3(decoder_result)
    root_cause = _determine_root_cause(v1, v2, v3)
    recommendations = _generate_recommendations(v1, v2, v3)

    lines = []
    lines.append("# P3-D3 Diagnostic Report: T-Scaling Investigation\n")

    lines.append("## 1. Executive Summary\n")
    lines.append(f"**Q1: Does increasing T change hidden states before decoding?** — **{v1['verdict']}** ({v1['detail']})\n")
    lines.append(f"**Q2: Does increasing T change logits without changing parsed predictions?** — **{v2['verdict']}** ({v2['detail']})\n")
    lines.append(f"**Q3: Does increasing T fail to put probability mass on reachable answer labels?** — **{v3['verdict']}** ({v3['detail']})\n")

    lines.append("## 2. Flow Integration Analysis\n")
    lines.append("### Endpoint Hidden Norm per T\n")
    lines.append("| T | Probes | Mean Norm | Std Norm | Median Norm |")
    lines.append("|---|--------|-----------|----------|-------------|")
    for T in sorted(flow_result.per_T_stats.keys()):
        s = flow_result.per_T_stats[T]
        lines.append(f"| {T} | {s['n']} | {s['mean']:.4f} | {s['std']:.4f} | {s['median']:.4f} |")
    lines.append("")

    lines.append("### Pairwise T-Test Results (Bonferroni-corrected)\n")
    lines.append("| T_a | T_b | t-stat | p-value | Significant | Cohen's d | Interpretation | Mean Delta |")
    lines.append("|-----|-----|--------|---------|-------------|-----------|---------------|------------|")
    for p in flow_result.pairwise_tests:
        sig = "YES" if p["significant"] else "no"
        lines.append(f"| {p['T_a']} | {p['T_b']} | {p['t_stat']:.3f} | {p['p_value']:.4f} | {sig} | {p['cohens_d']:.3f} | {p['interpretation']} | {p['mean_delta']:.4f} |")
    lines.append("")

    lines.append("### Velocity Norm per T\n")
    lines.append("| T | Mean Velocity | Std Velocity | Nonzero Probes |")
    lines.append("|---|---------------|--------------|----------------|")
    for T in sorted(flow_result.velocity_analysis.keys()):
        v = flow_result.velocity_analysis[T]
        lines.append(f"| {T} | {v['mean']:.4f} | {v['std']:.4f} | {v['nonzero_count']} |")
    lines.append("")

    lines.append("### Divergence from T1\n")
    lines.append("| T | Mean Divergence | Proportion Diverged |")
    lines.append("|---|-----------------|---------------------|")
    for T in sorted(flow_result.divergence_from_T1.keys()):
        d = flow_result.divergence_from_T1[T]
        lines.append(f"| {T} | {d['mean']:.4f} | {d['proportion_diverged']:.2%} |")
    lines.append("")

    lines.append("## 3. Decoder/Readout Analysis\n")
    lines.append("### Per-T Accuracy\n")
    lines.append("| T | Probes | Correct | Accuracy |")
    lines.append("|---|--------|---------|----------|")
    for T in sorted(decoder_result.per_T_stats.keys()):
        s = decoder_result.per_T_stats[T]
        correct = int(s["accuracy"] * s["n"])
        lines.append(f"| {T} | {s['n']} | {correct} | {s['accuracy']:.2%} |")
    lines.append("")

    lines.append("### KL/JS Divergence from Teacher per T\n")
    lines.append("| T | Mean KL | Mean JS | Median KL | Median JS |")
    lines.append("|---|---------|---------|-----------|-----------|")
    for T in sorted(decoder_result.per_T_stats.keys()):
        s = decoder_result.per_T_stats[T]
        lines.append(f"| {T} | {s['mean_kl']:.4f} | {s['mean_js']:.4f} | {s['median_kl']:.4f} | {s['median_js']:.4f} |")
    lines.append("")

    lines.append("### Answer Coverage per T\n")
    lines.append("| T | Mean Pr(correct) | Median Pr(correct) | Probes Pr > 50% |")
    lines.append("|---|------------------|---------------------|-----------------|")
    for T in sorted(decoder_result.answer_coverage.keys()):
        c = decoder_result.answer_coverage[T]
        lines.append(f"| {T} | {c['mean_prob']:.4f} | {c['median_prob']:.4f} | {c['probes_gt_50pct']} |")
    lines.append("")

    lines.append("### Prediction Stability Across T\n")
    ps = decoder_result.prediction_stability
    lines.append(f"- Always wrong (all T): {ps['always_wrong']}")
    lines.append(f"- Flipped prediction (changed between T values): {ps['flipped']}")
    lines.append(f"- Became correct at higher T: {ps['became_correct']}\n")

    lines.append("## 4. Root Cause Decision Tree\n")
    lines.append(root_cause)
    lines.append("")

    lines.append("## 5. Recommendations\n")
    lines.append(recommendations)
    lines.append("")

    lines.append(f"---\n*Report generated from traces at `{traces_dir}`. Probes loaded from `{probes_path}`.*\n")

    return "\n".join(lines)
