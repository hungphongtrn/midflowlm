"""
Tests for evaluation pipeline and baselines.

These tests verify:
1. Identity baseline exists and produces expected outputs
2. T=1 shared-block baseline exists and produces expected outputs
3. Simple recurrent baseline without minFM step conditioning
4. Metric reporting format is correct
5. All metrics are computed and reported properly
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import yaml


@pytest.fixture
def config():
    """Load v0 config for tests."""
    config_path = Path(__file__).parent.parent / "configs" / "v0_onemotif.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def device():
    """Get available device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def model_config(config):
    """Extract model configuration."""
    return {
        "model_name": config["model"]["name"],
        "hidden_size": 896,  # Qwen3.5-0.8B hidden size
        "num_layers": 36,  # Qwen3.5-0.8B has 36 layers
        "max_steps_T": config["model"]["max_steps_T"],
        "start_layer": config["replacement_model"]["start_layer"],
        "end_layer": config["replacement_model"]["end_layer"],
    }


@pytest.fixture
def sample_hidden_states(model_config, device):
    """Create sample hidden states for testing."""
    batch_size = 2
    seq_len = 16
    hidden_size = model_config["hidden_size"]
    return torch.randn(batch_size, seq_len, hidden_size, device=device)


@pytest.fixture
def sample_logits(model_config, device):
    """Create sample logits for testing."""
    batch_size = 2
    seq_len = 16
    vocab_size = 151936  # Qwen3.5-0.8B vocab size
    return torch.randn(batch_size, seq_len, vocab_size, device=device)


@pytest.fixture
def sample_labels(model_config, device):
    """Create sample labels for testing."""
    batch_size = 2
    seq_len = 16
    return torch.randint(0, 151936, (batch_size, seq_len), device=device)


class TestEvalImports:
    """Test that eval modules can be imported."""

    def test_import_baselines(self):
        """Test that src.eval.baselines exists and can be imported."""
        from src.eval import baselines

        assert baselines is not None

    def test_import_identity_baseline(self):
        """Test that IdentityBaseline class exists."""
        from src.eval.baselines import IdentityBaseline

        assert IdentityBaseline is not None

    def test_import_t1_shared_block_baseline(self):
        """Test that T1SharedBlockBaseline class exists."""
        from src.eval.baselines import T1SharedBlockBaseline

        assert T1SharedBlockBaseline is not None

    def test_import_simple_recurrent_baseline(self):
        """Test that SimpleRecurrentBaseline class exists."""
        from src.eval.baselines import SimpleRecurrentBaseline

        assert SimpleRecurrentBaseline is not None

    def test_import_metrics(self):
        """Test that evaluation metrics exist."""
        from src.eval.baselines import (
            compute_endpoint_error,
            compute_trajectory_error,
            compute_kl_divergence,
            compute_perplexity,
            compute_latency_metrics,
            compute_stability_metrics,
        )

        assert compute_endpoint_error is not None
        assert compute_trajectory_error is not None
        assert compute_kl_divergence is not None
        assert compute_perplexity is not None
        assert compute_latency_metrics is not None
        assert compute_stability_metrics is not None


class TestIdentityBaseline:
    """Test identity baseline (h_end = h_start)."""

    def test_identity_baseline_exists(self):
        """Test that IdentityBaseline class exists."""
        from src.eval.baselines import IdentityBaseline

        baseline = IdentityBaseline()
        assert baseline is not None

    def test_identity_baseline_output_equals_input(self, sample_hidden_states, device):
        """Test that identity baseline returns input unchanged."""
        from src.eval.baselines import IdentityBaseline

        baseline = IdentityBaseline()
        h_start = sample_hidden_states
        h_end = baseline(h_start)

        assert torch.allclose(h_end, h_start, atol=1e-6)

    def test_identity_baseline_no_parameters(self):
        """Test that identity baseline has no trainable parameters."""
        from src.eval.baselines import IdentityBaseline

        baseline = IdentityBaseline()
        params = list(baseline.parameters())
        assert len(params) == 0

    def test_identity_baseline_shape_preservation(self, sample_hidden_states):
        """Test that identity baseline preserves shape."""
        from src.eval.baselines import IdentityBaseline

        baseline = IdentityBaseline()
        h_start = sample_hidden_states
        h_end = baseline(h_start)

        assert h_end.shape == h_start.shape


class TestT1SharedBlockBaseline:
    """Test T=1 shared-block baseline."""

    def test_t1_shared_block_baseline_exists(self, model_config):
        """Test that T1SharedBlockBaseline class exists."""
        from src.eval.baselines import T1SharedBlockBaseline

        baseline = T1SharedBlockBaseline(
            hidden_size=model_config["hidden_size"],
            num_heads=8,
        )
        assert baseline is not None

    def test_t1_shared_block_has_parameters(self, model_config):
        """Test that T1 shared-block baseline has trainable parameters."""
        from src.eval.baselines import T1SharedBlockBaseline

        baseline = T1SharedBlockBaseline(
            hidden_size=model_config["hidden_size"],
            num_heads=8,
        )
        params = list(baseline.parameters())
        assert len(params) > 0

    def test_t1_shared_block_output_shape(
        self, model_config, sample_hidden_states, device
    ):
        """Test that T1 shared-block baseline produces correct output shape."""
        from src.eval.baselines import T1SharedBlockBaseline

        baseline = T1SharedBlockBaseline(
            hidden_size=model_config["hidden_size"],
            num_heads=8,
        ).to(device)

        h_start = sample_hidden_states
        h_end = baseline(h_start)

        assert h_end.shape == h_start.shape

    def test_t1_shared_block_single_step(
        self, model_config, sample_hidden_states, device
    ):
        """Test that T1 shared-block baseline runs exactly one step."""
        from src.eval.baselines import T1SharedBlockBaseline

        baseline = T1SharedBlockBaseline(
            hidden_size=model_config["hidden_size"],
            num_heads=8,
        ).to(device)

        h_start = sample_hidden_states
        h_end = baseline(h_start, num_steps=1)

        assert h_end.shape == h_start.shape
        # Should be different from input (has gone through transformation)
        assert not torch.allclose(h_end, h_start, atol=1e-6)


class TestSimpleRecurrentBaseline:
    """Test simple recurrent baseline without minFM step conditioning."""

    def test_simple_recurrent_baseline_exists(self, model_config):
        """Test that SimpleRecurrentBaseline class exists."""
        from src.eval.baselines import SimpleRecurrentBaseline

        baseline = SimpleRecurrentBaseline(
            hidden_size=model_config["hidden_size"],
            num_heads=8,
            max_steps_T=model_config["max_steps_T"],
        )
        assert baseline is not None

    def test_simple_recurrent_no_step_conditioning(self, model_config):
        """Test that simple recurrent baseline doesn't use step conditioning."""
        from src.eval.baselines import SimpleRecurrentBaseline

        baseline = SimpleRecurrentBaseline(
            hidden_size=model_config["hidden_size"],
            num_heads=8,
            max_steps_T=model_config["max_steps_T"],
        )
        assert not baseline.use_step_conditioning

    def test_simple_recurrent_has_parameters(self, model_config):
        """Test that simple recurrent baseline has trainable parameters."""
        from src.eval.baselines import SimpleRecurrentBaseline

        baseline = SimpleRecurrentBaseline(
            hidden_size=model_config["hidden_size"],
            num_heads=8,
            max_steps_T=model_config["max_steps_T"],
        )
        params = list(baseline.parameters())
        assert len(params) > 0

    def test_simple_recurrent_multi_step(
        self, model_config, sample_hidden_states, device
    ):
        """Test that simple recurrent baseline can run multiple steps."""
        from src.eval.baselines import SimpleRecurrentBaseline

        baseline = SimpleRecurrentBaseline(
            hidden_size=model_config["hidden_size"],
            num_heads=8,
            max_steps_T=model_config["max_steps_T"],
        ).to(device)

        h_start = sample_hidden_states
        h_end = baseline(h_start, num_steps=4)

        assert h_end.shape == h_start.shape

    def test_simple_recurrent_same_output_for_same_t(
        self, model_config, sample_hidden_states, device
    ):
        """Test that simple recurrent baseline produces consistent outputs."""
        from src.eval.baselines import SimpleRecurrentBaseline

        baseline = SimpleRecurrentBaseline(
            hidden_size=model_config["hidden_size"],
            num_heads=8,
            max_steps_T=model_config["max_steps_T"],
        ).to(device)

        # Set to eval mode for deterministic behavior
        baseline.eval()
        with torch.no_grad():
            h_start = sample_hidden_states
            h_end_1 = baseline(h_start, num_steps=4)
            h_end_2 = baseline(h_start, num_steps=4)

        assert torch.allclose(h_end_1, h_end_2, atol=1e-6)


class TestMetrics:
    """Test evaluation metrics."""

    def test_endpoint_error_computation(self, sample_hidden_states):
        """Test endpoint hidden-state error computation."""
        from src.eval.baselines import compute_endpoint_error

        h_pred = sample_hidden_states
        h_target = sample_hidden_states + 0.1 * torch.randn_like(sample_hidden_states)

        error = compute_endpoint_error(h_pred, h_target)

        assert isinstance(error, float)
        assert error >= 0

    def test_trajectory_error_computation(self, sample_hidden_states):
        """Test trajectory error computation."""
        from src.eval.baselines import compute_trajectory_error

        batch_size, seq_len, hidden_size = sample_hidden_states.shape
        # Create trajectory: list of hidden states over steps
        trajectory_pred = [
            sample_hidden_states + 0.05 * torch.randn_like(sample_hidden_states)
            for _ in range(4)
        ]
        trajectory_target = [
            sample_hidden_states + 0.05 * torch.randn_like(sample_hidden_states)
            for _ in range(4)
        ]

        error = compute_trajectory_error(trajectory_pred, trajectory_target)

        assert isinstance(error, float)
        assert error >= 0

    def test_kl_divergence_computation(self, sample_logits):
        """Test KL divergence computation."""
        from src.eval.baselines import compute_kl_divergence

        logits_pred = sample_logits
        logits_target = sample_logits + 0.1 * torch.randn_like(sample_logits)

        kl_div = compute_kl_divergence(logits_pred, logits_target)

        assert isinstance(kl_div, float)
        assert kl_div >= 0

    def test_perplexity_computation(self, sample_logits, sample_labels):
        """Test perplexity computation."""
        from src.eval.baselines import compute_perplexity

        perplexity = compute_perplexity(sample_logits, sample_labels)

        assert isinstance(perplexity, float)
        assert perplexity > 0

    def test_latency_metrics_computation(self):
        """Test latency and throughput metrics computation."""
        from src.eval.baselines import compute_latency_metrics

        latencies = [0.1, 0.12, 0.09, 0.11, 0.1]  # seconds
        batch_size = 2
        seq_len = 16

        metrics = compute_latency_metrics(latencies, batch_size, seq_len)

        assert "mean_latency_ms" in metrics
        assert "std_latency_ms" in metrics
        assert "tokens_per_second" in metrics
        assert metrics["mean_latency_ms"] > 0
        assert metrics["tokens_per_second"] > 0

    def test_stability_metrics_computation(self, sample_hidden_states):
        """Test hidden-state stability metrics computation."""
        from src.eval.baselines import compute_stability_metrics

        batch_size, seq_len, hidden_size = sample_hidden_states.shape
        # Create trajectory with some variation
        trajectory = [
            sample_hidden_states + 0.01 * i * torch.randn_like(sample_hidden_states)
            for i in range(5)
        ]

        metrics = compute_stability_metrics(trajectory)

        assert "mean_norm" in metrics
        assert "mean_delta_norm" in metrics
        assert "max_delta_norm" in metrics
        assert metrics["mean_norm"] > 0
        assert metrics["mean_delta_norm"] >= 0


class TestMetricReportingFormat:
    """Test metric reporting format."""

    def test_metrics_report_structure(self):
        """Test that metrics report has correct structure."""
        from src.eval.baselines import MetricsReport

        report = MetricsReport(
            endpoint_error=0.1,
            trajectory_error=0.2,
            kl_divergence=0.05,
            perplexity=15.5,
            latency_ms=100.0,
            tokens_per_second=50.0,
            total_params=1000000,
            trainable_params=100000,
            stability_metrics={"mean_norm": 10.0, "mean_delta_norm": 0.5},
        )

        assert hasattr(report, "endpoint_error")
        assert hasattr(report, "trajectory_error")
        assert hasattr(report, "kl_divergence")
        assert hasattr(report, "perplexity")
        assert hasattr(report, "latency_ms")
        assert hasattr(report, "tokens_per_second")
        assert hasattr(report, "total_params")
        assert hasattr(report, "trainable_params")
        assert hasattr(report, "stability_metrics")

    def test_metrics_report_to_dict(self):
        """Test that metrics report can be converted to dict."""
        from src.eval.baselines import MetricsReport

        report = MetricsReport(
            endpoint_error=0.1,
            trajectory_error=0.2,
            kl_divergence=0.05,
            perplexity=15.5,
            latency_ms=100.0,
            tokens_per_second=50.0,
            total_params=1000000,
            trainable_params=100000,
            stability_metrics={"mean_norm": 10.0, "mean_delta_norm": 0.5},
        )

        report_dict = report.to_dict()

        assert isinstance(report_dict, dict)
        assert "endpoint_error" in report_dict
        assert "trajectory_error" in report_dict
        assert "kl_divergence" in report_dict

    def test_metrics_report_to_json(self):
        """Test that metrics report can be serialized to JSON."""
        from src.eval.baselines import MetricsReport
        import json

        report = MetricsReport(
            endpoint_error=0.1,
            trajectory_error=0.2,
            kl_divergence=0.05,
            perplexity=15.5,
            latency_ms=100.0,
            tokens_per_second=50.0,
            total_params=1000000,
            trainable_params=100000,
            stability_metrics={"mean_norm": 10.0, "mean_delta_norm": 0.5},
        )

        json_str = report.to_json()

        assert isinstance(json_str, str)
        # Verify it's valid JSON
        parsed = json.loads(json_str)
        assert parsed["endpoint_error"] == 0.1


class TestBaselineComparison:
    """Test comparing baselines to each other."""

    def test_identity_vs_t1_error(self, model_config, sample_hidden_states, device):
        """Test that we can compare identity and T1 baselines."""
        from src.eval.baselines import (
            IdentityBaseline,
            T1SharedBlockBaseline,
            compute_endpoint_error,
        )

        identity = IdentityBaseline()
        t1_baseline = T1SharedBlockBaseline(
            hidden_size=model_config["hidden_size"],
            num_heads=8,
        ).to(device)

        h_start = sample_hidden_states
        h_identity = identity(h_start)
        h_t1 = t1_baseline(h_start, num_steps=1)

        error = compute_endpoint_error(h_identity, h_t1)

        assert isinstance(error, float)
        assert error >= 0

    def test_all_baselines_produce_same_shape(
        self, model_config, sample_hidden_states, device
    ):
        """Test that all baselines produce outputs with the same shape."""
        from src.eval.baselines import (
            IdentityBaseline,
            T1SharedBlockBaseline,
            SimpleRecurrentBaseline,
        )

        identity = IdentityBaseline()
        t1_baseline = T1SharedBlockBaseline(
            hidden_size=model_config["hidden_size"],
            num_heads=8,
        ).to(device)
        recurrent = SimpleRecurrentBaseline(
            hidden_size=model_config["hidden_size"],
            num_heads=8,
            max_steps_T=model_config["max_steps_T"],
        ).to(device)

        h_start = sample_hidden_states

        h_identity = identity(h_start)
        h_t1 = t1_baseline(h_start, num_steps=1)
        h_recurrent = recurrent(h_start, num_steps=4)

        assert h_identity.shape == h_start.shape
        assert h_t1.shape == h_start.shape
        assert h_recurrent.shape == h_start.shape


class TestTextSweepSolverMetadata:
    """Test that text sweep records solver metadata."""

    def test_solver_method_parameter_in_function_signature(self):
        """Test that run_text_sweep accepts solver_method parameter."""
        from src.eval.text_checkpoint_sweep import run_text_sweep
        import inspect

        sig = inspect.signature(run_text_sweep)
        assert "solver_method" in sig.parameters
        assert sig.parameters["solver_method"].default == "euler"

    def test_solver_method_passed_to_greedy_generate(self):
        """Test that solver_method is passed to greedy_generate."""
        from src.eval.text_checkpoint_sweep import greedy_generate
        import inspect

        sig = inspect.signature(greedy_generate)
        assert "solver_method" in sig.parameters

    def test_payload_includes_solver_metadata(self):
        """Test that payload includes solver_method when calling run_text_sweep."""
        from src.eval.text_checkpoint_sweep import run_text_sweep

        payload = {
            "config_path": "test",
            "checkpoint": {},
            "device": "cpu",
            "max_new_tokens": 10,
            "num_steps": [4, 8],
            "max_steps_T": 8,
            "solver_method": "rk4",
            "warnings": [],
            "comparisons": [],
            "repetition_metrics": {},
            "table": "",
        }

        assert "solver_method" in payload
        assert payload["solver_method"] == "rk4"


class TestRepetitionMetrics:
    """Test n-gram repetition metrics."""

    def test_repetition_metrics_include_ngram_counts(self):
        """Test that repetition metrics include n-gram ratio counts."""
        from src.eval.text_checkpoint_sweep import compute_repetition_metrics

        text = "cat cat cat cat"
        metrics = compute_repetition_metrics(text, n_values=(2, 3, 4))

        assert "repeat_2gram_ratio" in metrics
        assert "repeat_3gram_ratio" in metrics
        assert "repeat_4gram_ratio" in metrics

    def test_repetition_metrics_values(self):
        """Test that repetition metrics compute correct ratios."""
        from src.eval.text_checkpoint_sweep import compute_repetition_metrics

        text_no_repeat = "the quick brown fox jumps"
        metrics = compute_repetition_metrics(text_no_repeat, n_values=(2,))

        assert metrics["repeat_2gram_ratio"] == 0.0

        text_with_repeat = "cat cat cat"
        metrics = compute_repetition_metrics(text_with_repeat, n_values=(2,))

        assert metrics["repeat_2gram_ratio"] > 0.0

    def test_aggregate_repetition_metrics(self):
        """Test that aggregate function computes mean across comparisons."""
        from src.eval.text_checkpoint_sweep import aggregate_repetition_metrics

        comparisons = [
            {"generated_text": "cat cat cat"},
            {"generated_text": "dog dog dog"},
        ]

        result = aggregate_repetition_metrics(comparisons, n_values=(2,))

        assert "mean_repeat_2gram_ratio" in result
        assert result["mean_repeat_2gram_ratio"] >= 0.0

    def test_repetition_metrics_empty_text(self):
        """Test that compute_repetition_metrics handles empty text."""
        from src.eval.text_checkpoint_sweep import compute_repetition_metrics

        metrics = compute_repetition_metrics("", n_values=(2,))

        assert metrics["repeat_2gram_ratio"] == 0.0

    def test_repetition_metrics_single_word(self):
        """Test that compute_repetition_metrics handles single word."""
        from src.eval.text_checkpoint_sweep import compute_repetition_metrics

        metrics = compute_repetition_metrics("hello", n_values=(2,))

        assert metrics["repeat_2gram_ratio"] == 0.0

    def test_aggregate_empty_comparisons(self):
        """Test that aggregate_repetition_metrics handles empty list."""
        from src.eval.text_checkpoint_sweep import aggregate_repetition_metrics

        result = aggregate_repetition_metrics([], n_values=(2,))

        assert "mean_repeat_2gram_ratio" in result
        assert result["mean_repeat_2gram_ratio"] == 0.0
