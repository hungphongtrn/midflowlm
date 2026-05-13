from src.diagnostic.analysis import FlowAnalysisResult, DecoderAnalysisResult


def _make_empty_results():
    flow = FlowAnalysisResult(
        per_T_stats={
            T: {"mean": 0.1, "std": 0.0, "median": 0.1, "n": 10}
            for T in [1, 2, 8, 64]
        },
        pairwise_tests=[
            {"T_a": 1, "T_b": 64, "t_stat": 0.0, "p_value": 1.0, "cohens_d": 0.0, "mean_delta": 0.0, "significant": False, "interpretation": "small"},
        ],
        velocity_analysis={
            T: {"mean": 0.0, "std": 0.0, "median": 0.0, "nonzero_count": 0}
            for T in [1, 2, 8, 64]
        },
        divergence_from_T1={
            T: {"mean": 0.0, "median": 0.0, "proportion_diverged": 0.0}
            for T in [2, 8, 64]
        },
    )
    decoder = DecoderAnalysisResult(
        per_T_stats={
            T: {"accuracy": 0.0, "mean_kl": 0.0, "mean_js": 0.0, "median_kl": 0.0, "median_js": 0.0, "n": 10}
            for T in [1, 2, 8, 64]
        },
        logit_shift_tests=[
            {"T_a": 1, "T_b": 64, "mean_kl_delta": 0.0, "proportion_shifted": 0.0},
        ],
        answer_coverage={
            T: {"mean_prob": 0.05, "median_prob": 0.04, "max_prob": 0.1, "probes_gt_50pct": 0}
            for T in [1, 2, 8, 64]
        },
        prediction_stability={"always_wrong": 10, "flipped": 0, "became_correct": 0},
    )
    return flow, decoder


class TestGenerateReport:
    def test_generate_report_returns_string(self):
        flow, decoder = _make_empty_results()
        from src.diagnostic.report import generate_report
        report = generate_report(flow, decoder, probes_path="", traces_dir="")
        assert isinstance(report, str)
        assert len(report) > 100

    def test_report_contains_all_sections(self):
        flow, decoder = _make_empty_results()
        from src.diagnostic.report import generate_report
        report = generate_report(flow, decoder, probes_path="", traces_dir="")
        assert "Executive Summary" in report
        assert "Flow Integration Analysis" in report
        assert "Decoder/Readout Analysis" in report
        assert "Root Cause Decision Tree" in report
        assert "Recommendations" in report

    def test_report_verdicts_on_no_change(self):
        flow, decoder = _make_empty_results()
        from src.diagnostic.report import generate_report
        report = generate_report(flow, decoder, probes_path="", traces_dir="")
        assert "Q1:" in report
        assert "Q2:" in report
        assert "Q3:" in report
        assert "FAIL" in report

    def test_report_tables_valid_markdown(self):
        flow, decoder = _make_empty_results()
        from src.diagnostic.report import generate_report
        report = generate_report(flow, decoder, probes_path="", traces_dir="")
        assert "|" in report
        lines = report.split("\n")
        table_headers = [l for l in lines if l.strip().startswith("|---")]
        assert len(table_headers) >= 4
