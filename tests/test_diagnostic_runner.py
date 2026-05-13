"""Tests for diagnostic runner - TraceRecord serialization and deterministic behavior."""
import json
import random
import torch
import numpy as np

from src.diagnostic.runner import TraceRecord, set_deterministic


class TestTraceRecord:
    """Test TraceRecord dataclass serialization."""
    
    def test_trace_record_to_dict(self):
        """TraceRecord should serialize to dictionary correctly."""
        record = TraceRecord(
            probe_id="mmlu_001",
            benchmark="mmlu_pro",
            T=1,
            seed=42,
            endpoint_hidden_norm=12.34,
            logits_answer_tokens={"A": 1.0, "B": 2.0},
            predicted_answer="B",
            predicted_token_id=66,
            full_logits_shape="[1, 151936]",
        )
        
        data = record.to_dict()
        
        assert data["probe_id"] == "mmlu_001"
        assert data["benchmark"] == "mmlu_pro"
        assert data["T"] == 1
        assert data["seed"] == 42
        assert data["endpoint_hidden_norm"] == 12.34
        assert data["logits_answer_tokens"] == {"A": 1.0, "B": 2.0}
        assert data["predicted_answer"] == "B"
        assert data["predicted_token_id"] == 66
        assert data["full_logits_shape"] == "[1, 151936]"
    
    def test_trace_record_roundtrip(self):
        """TraceRecord should survive JSON roundtrip."""
        record = TraceRecord(
            probe_id="arc_001",
            benchmark="arc_easy",
            T=8,
            seed=123,
            endpoint_hidden_norm=5.67,
            logits_answer_tokens={"A": 0.5, "B": -0.3, "C": 1.2},
            predicted_answer="C",
            predicted_token_id=67,
            full_logits_shape="[1, 32000]",
        )
        
        data = record.to_dict()
        json_str = json.dumps(data)
        reloaded_data = json.loads(json_str)
        
        assert reloaded_data["probe_id"] == "arc_001"
        assert reloaded_data["benchmark"] == "arc_easy"
        assert reloaded_data["T"] == 8
        assert reloaded_data["endpoint_hidden_norm"] == 5.67


class TestSetDeterministic:
    """Test set_deterministic function for reproducibility."""
    
    def test_same_seed_same_random_numbers(self):
        """Same seed should produce identical random numbers."""
        set_deterministic(42)
        random_values_1 = [random.random() for _ in range(10)]
        
        set_deterministic(42)
        random_values_2 = [random.random() for _ in range(10)]
        
        assert random_values_1 == random_values_2
    
    def test_different_seed_different_random_numbers(self):
        """Different seeds should produce different random numbers."""
        set_deterministic(42)
        random_values_1 = [random.random() for _ in range(10)]
        
        set_deterministic(123)
        random_values_2 = [random.random() for _ in range(10)]
        
        assert random_values_1 != random_values_2
    
    def test_numpy_reproducibility(self):
        """NumPy random should be deterministic."""
        set_deterministic(42)
        numpy_values_1 = np.random.randn(5).tolist()
        
        set_deterministic(42)
        numpy_values_2 = np.random.randn(5).tolist()
        
        assert numpy_values_1 == numpy_values_2
    
    def test_torch_reproducibility(self):
        """PyTorch random should be deterministic."""
        set_deterministic(42)
        torch_values_1 = torch.randn(5).tolist()
        
        set_deterministic(42)
        torch_values_2 = torch.randn(5).tolist()
        
        # Compare element by element with tolerance
        for v1, v2 in zip(torch_values_1, torch_values_2):
            assert abs(v1 - v2) < 1e-6


class TestDeterministicTraceRunnerSkeleton:
    """Test DeterministicTraceRunner class structure (skeleton for Phase 1)."""
    
    def test_runner_class_exists(self):
        """DeterministicTraceRunner class should exist."""
        from src.diagnostic.runner import DeterministicTraceRunner
        assert DeterministicTraceRunner is not None
    
    def test_runner_initialization(self):
        """Runner should initialize with model, tokenizer, device, seed."""
        from src.diagnostic.runner import DeterministicTraceRunner
        
        # Mock model and tokenizer
        mock_model = object()
        mock_tokenizer = object()
        
        runner = DeterministicTraceRunner(
            model=mock_model,
            tokenizer=mock_tokenizer,
            device=torch.device("cpu"),
            seed=42,
        )
        
        assert runner.model is mock_model
        assert runner.tokenizer is mock_tokenizer
        assert runner.seed == 42


class MockModelOutput:
    """Mock output for testing deterministic behavior."""
    def __init__(self, hidden_norm, logits):
        self.endpoint_hidden = torch.randn(1, 10, 128) * hidden_norm
        self.logits = logits


class MockTokenizer:
    """Mock tokenizer for testing."""
    def encode(self, text, add_special_tokens=False):
        # Return deterministic token IDs for A-J
        return [ord(text)] if len(text) == 1 else [ord(c) for c in text]
    
    def decode(self, token_ids):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return "".join(chr(t) for t in token_ids if 65 <= t <= 74)


class TestLoadModelFromCheckpoint:
    """Test model loading from checkpoint."""
    
    def test_load_model_from_checkpoint_exists(self):
        """load_model_from_checkpoint function should exist."""
        from src.diagnostic.runner import load_model_from_checkpoint
        assert load_model_from_checkpoint is not None
    
    def test_load_model_from_checkpoint_signature(self):
        """Function should accept checkpoint_path, config_path, device."""
        import inspect
        from src.diagnostic.runner import load_model_from_checkpoint
        
        sig = inspect.signature(load_model_from_checkpoint)
        params = list(sig.parameters.keys())
        
        assert "checkpoint_path" in params
        assert "config_path" in params
        assert "device" in params


class TestDeterministicReproducibility:
    """Test that deterministic behavior produces reproducible results."""
    
    def test_deterministic_is_reproducible(self):
        """Same seed + same T should yield identical TraceRecord."""
        from src.diagnostic.runner import DeterministicTraceRunner
        from src.diagnostic.probe import ProbeExample
        
        # Create mock model that produces deterministic output based on global state
        class DeterministicMockModel:
            def __init__(self):
                self.eval_mode = False
            
            def eval(self):
                self.eval_mode = True
                return self
            
            def to(self, device):
                return self
            
            def forward(self, input_ids, attention_mask, num_steps, return_dict=True):
                # Use torch seed to generate deterministic hidden state
                torch.manual_seed(42 + num_steps)
                hidden_norm = 1.0 + num_steps * 0.1
                logits = torch.randn(1, input_ids.shape[1], 100) * 0.5
                logits[:, -1, 65:75] = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
                return MockModelOutput(hidden_norm, logits)
        
        # Create probe example
        probe = ProbeExample(
            id="test_probe_001",
            benchmark="test",
            question="Test question?",
            choices=["A", "B", "C"],
            target_label="B",
            input_ids=[1, 2, 3, 4, 5],
        )
        
        # Create runner with deterministic mock model
        model = DeterministicMockModel()
        tokenizer = MockTokenizer()
        runner = DeterministicTraceRunner(
            model=model,
            tokenizer=tokenizer,
            device=torch.device("cpu"),
            seed=42,
        )
        
        # Run twice with same T
        record1 = runner.run_single(probe, T=4)
        record2 = runner.run_single(probe, T=4)
        
        # Verify records are identical
        assert record1.probe_id == record2.probe_id
        assert record1.T == record2.T
        assert record1.seed == record2.seed
        assert abs(record1.endpoint_hidden_norm - record2.endpoint_hidden_norm) < 1e-6
        assert record1.predicted_token_id == record2.predicted_token_id
        assert record1.predicted_answer == record2.predicted_answer
    
    def test_T_changes_hidden_states(self):
        """T=1 vs T=8 should produce different endpoint_hidden norms."""
        from src.diagnostic.runner import DeterministicTraceRunner
        from src.diagnostic.probe import ProbeExample
        
        # Create mock model with T-dependent output
        class TDependentMockModel:
            def __init__(self):
                self.eval_mode = False
            
            def eval(self):
                self.eval_mode = True
                return self
            
            def to(self, device):
                return self
            
            def forward(self, input_ids, attention_mask, num_steps, return_dict=True):
                # T-dependent hidden norm
                hidden_norm = float(num_steps) * 2.5  # Different norm for each T
                logits = torch.zeros(1, input_ids.shape[1], 100)
                # Set answer logits with T-dependent pattern
                for i, label in enumerate("ABCDEFGHIJ"):
                    logits[:, -1, ord(label)] = float(num_steps) * (i + 1) * 0.1
                return MockModelOutput(hidden_norm, logits)
        
        # Create probe example
        probe = ProbeExample(
            id="test_probe_002",
            benchmark="test",
            question="Test question?",
            choices=["A", "B", "C"],
            target_label="A",
            input_ids=[1, 2, 3, 4, 5],
        )
        
        # Create runner
        model = TDependentMockModel()
        tokenizer = MockTokenizer()
        runner = DeterministicTraceRunner(
            model=model,
            tokenizer=tokenizer,
            device=torch.device("cpu"),
            seed=42,
        )
        
        # Run with T=1 and T=8
        record_t1 = runner.run_single(probe, T=1)
        record_t8 = runner.run_single(probe, T=8)
        
        # Verify different T produces different results
        assert abs(record_t1.endpoint_hidden_norm - record_t8.endpoint_hidden_norm) > 1e-6
        # T=8 should have higher norm (2.5 * 8 vs 2.5 * 1)
        assert record_t8.endpoint_hidden_norm > record_t1.endpoint_hidden_norm
