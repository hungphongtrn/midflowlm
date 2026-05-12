import pytest
from src.eval.sft_metrics import (
    SFTSampleResult,
    compute_fixed_t_accuracy,
    compute_oracle_accuracy,
    compute_prediction_change_rate,
    compute_multi_t_report,
)


def _make_result(qid, t, correct, answer="A"):
    return SFTSampleResult(
        question_id=qid,
        question=f"Q{qid}",
        correct_answer="A",
        predicted_answer=answer if correct else "B",
        is_correct=correct,
        num_steps=t,
        latency_ms=100.0,
    )


class TestFixedTAccuracy:
    def test_all_correct(self):
        results = [_make_result(0, 4, True) for _ in range(10)]
        assert compute_fixed_t_accuracy(results, 4) == 1.0

    def test_half_correct(self):
        results = [_make_result(i, 4, i % 2 == 0) for i in range(10)]
        assert compute_fixed_t_accuracy(results, 4) == 0.5

    def test_empty(self):
        assert compute_fixed_t_accuracy([], 4) == 0.0

    def test_no_matching_t(self):
        results = [_make_result(0, 1, True)]
        assert compute_fixed_t_accuracy(results, 4) == 0.0

    def test_mixed_t_values(self):
        results = [
            _make_result(0, 1, True),
            _make_result(0, 4, True),
            _make_result(0, 8, False),
        ]
        assert compute_fixed_t_accuracy(results, 4) == 1.0


class TestOracleAccuracy:
    def test_all_correct_at_least_one_t(self):
        results = [
            _make_result(0, 1, True),
            _make_result(0, 4, False),
            _make_result(1, 1, False),
            _make_result(1, 4, True),
        ]
        assert compute_oracle_accuracy(results, [1, 4]) == 1.0

    def test_none_correct(self):
        results = [
            _make_result(0, 1, False),
            _make_result(0, 4, False),
        ]
        assert compute_oracle_accuracy(results, [1, 4]) == 0.0

    def test_empty(self):
        assert compute_oracle_accuracy([], [1, 4]) == 0.0


class TestPredictionChangeRate:
    def test_no_changes(self):
        results = [
            _make_result(0, 1, True, answer="A"),
            _make_result(0, 4, True, answer="A"),
            _make_result(1, 1, True, answer="A"),
            _make_result(1, 4, True, answer="A"),
        ]
        assert compute_prediction_change_rate(results, [1, 4]) == 0.0

    def test_half_changed(self):
        results = [
            _make_result(0, 1, True, answer="A"),
            _make_result(0, 4, False, answer="B"),
            _make_result(1, 1, True, answer="C"),
            _make_result(1, 4, True, answer="C"),
        ]
        assert compute_prediction_change_rate(results, [1, 4]) == 0.5

    def test_all_changed(self):
        results = [
            _make_result(0, 1, True, answer="A"),
            _make_result(0, 4, True, answer="B"),
            _make_result(1, 1, True, answer="C"),
            _make_result(1, 4, True, answer="D"),
        ]
        assert compute_prediction_change_rate(results, [1, 4]) == 1.0

    def test_empty(self):
        assert compute_prediction_change_rate([], [1, 4]) == 0.0


class TestMultiTReport:
    def test_basic_report(self):
        results = [
            _make_result(0, 1, False, answer="B"),
            _make_result(0, 4, True, answer="A"),
            _make_result(1, 1, True, answer="A"),
            _make_result(1, 4, False, answer="B"),
        ]
        report = compute_multi_t_report(results, [1, 4], experiment_id="test")

        assert report.experiment_id == "test"
        assert report.num_total == 2
        assert report.per_t_accuracy[1] == 0.5
        assert report.per_t_accuracy[4] == 0.5
        assert report.oracle_accuracy == 1.0
        assert report.prediction_change_rate == 1.0

        summary = report.summary()
        assert "Oracle" in summary
        assert "50.0%" in summary
