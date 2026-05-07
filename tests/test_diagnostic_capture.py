import json
import pytest
import torch
from src.diagnostic.traces import FlowTrace, DecoderTrace
from src.diagnostic.probe import ProbeExample

class TestFlowTrace:
    def test_flow_trace_fields(self):
        ft = FlowTrace(
            probe_id="mmlu_001",
            benchmark="mmlu_pro",
            T=8,
            endpoint_hidden_norm=1.23,
            per_step_velocity_norms=[0.1, 0.2, 0.15, 0.18, 0.12, 0.14, 0.11, 0.13],
            trajectory_endpoint_norm=1.45,
            trajectory_divergence_from_T1=0.05,
            teacher_anchor_distances={"h8": 0.3, "h9": 0.4, "h10": 0.35, "h11": 0.45},
        )
        assert ft.T == 8
        assert len(ft.per_step_velocity_norms) == 8

    def test_flow_trace_serialization(self):
        ft = FlowTrace(
            probe_id="arc_001",
            benchmark="arc_easy",
            T=2,
            endpoint_hidden_norm=2.34,
            per_step_velocity_norms=[0.5, 0.6],
            trajectory_endpoint_norm=2.45,
            trajectory_divergence_from_T1=0.0,
            teacher_anchor_distances={"h8": 0.1},
        )
        d = ft.to_dict()
        assert d["probe_id"] == "arc_001"
        assert d["per_step_velocity_norms"] == [0.5, 0.6]
        assert d == FlowTrace(**d).to_dict()


class TestDecoderTrace:
    def test_decoder_trace_fields(self):
        dt = DecoderTrace(
            probe_id="mmlu_001",
            benchmark="mmlu_pro",
            T=64,
            logits_answer_tokens={"A": -2.1, "B": -1.2, "C": -0.5, "D": -3.0, "E": -4.0},
            predicted_answer="C",
            predicted_token_id=17627,
            ground_truth_label="E",
            parsed_answer_match=False,
            teacher_logits_answer_tokens={"A": -3.0, "B": -2.0, "C": -1.5, "D": -0.8, "E": -0.2},
            kl_divergence=0.45,
            js_divergence=0.12,
        )
        assert dt.T == 64
        assert dt.predicted_answer == "C"
        assert not dt.parsed_answer_match

    def test_decoder_trace_serialization(self):
        dt = DecoderTrace(
            probe_id="arc_001",
            benchmark="arc_easy",
            T=1,
            logits_answer_tokens={"A": 0.1},
            predicted_answer="A",
            predicted_token_id=17625,
            ground_truth_label="A",
            parsed_answer_match=True,
        )
        d = dt.to_dict()
        assert d["probe_id"] == "arc_001"
        assert d["parsed_answer_match"] is True


class MockModelFlow:
    def __init__(self):
        self.device = torch.device("cpu")
        self._calls = []
    def eval(self):
        pass
    def forward(self, input_ids, attention_mask=None, num_steps=None, return_dict=False):
        batch, seq, hidden = 1, 10, 768
        h_mid = torch.randn(batch, seq, hidden) * num_steps * 0.1
        trajectory = torch.randn(batch, seq, num_steps, hidden) if return_dict else None
        logits = torch.randn(batch, seq, 151936)
        from types import SimpleNamespace
        return SimpleNamespace(
            logits=logits,
            endpoint_hidden=h_mid,
            trajectory_hidden=trajectory,
        )
    def extract_teacher_targets(self, input_ids, attention_mask=None, need_trajectory_anchors=False):
        batch, seq, hidden = 1, 10, 768
        return {
            "h_start": torch.randn(batch, seq, hidden),
            "h_target": torch.randn(batch, seq, hidden),
            "teacher_logits": torch.randn(batch, seq, 151936),
            "trajectory_anchors": {
                "h8": torch.randn(batch, seq, hidden),
                "h9": torch.randn(batch, seq, hidden),
                "h10": torch.randn(batch, seq, hidden),
                "h11": torch.randn(batch, seq, hidden),
            } if need_trajectory_anchors else None,
        }


class TestCaptureFlow:
    def test_capture_flow_traces_returns_list(self):
        from src.diagnostic.capture import capture_flow_traces
        model = MockModelFlow()
        example = ProbeExample(id="p1", benchmark="mmlu_pro", question="?", choices=["A"], target_label="A", input_ids=[1,2,3])
        traces = capture_flow_traces(model, example, T_values=[1, 8], device=torch.device("cpu"), seed=42)
        assert len(traces) == 2
        assert traces[0].T == 1
        assert traces[1].T == 8

    def test_capture_flow_traces_has_velocity_norms(self):
        from src.diagnostic.capture import capture_flow_traces
        model = MockModelFlow()
        example = ProbeExample(id="p1", benchmark="mmlu_pro", question="?", choices=["A"], target_label="A", input_ids=[1,2,3])
        traces = capture_flow_traces(model, example, T_values=[2], device=torch.device("cpu"), seed=42)
        assert len(traces[0].per_step_velocity_norms) == 2

    def test_capture_flow_traces_divergence_from_T1(self):
        from src.diagnostic.capture import capture_flow_traces
        model = MockModelFlow()
        example = ProbeExample(id="p1", benchmark="mmlu_pro", question="?", choices=["A"], target_label="A", input_ids=[1,2,3])
        traces = capture_flow_traces(model, example, T_values=[1, 4, 8], device=torch.device("cpu"), seed=42)
        t4_trace = [t for t in traces if t.T == 4][0]
        assert t4_trace.trajectory_divergence_from_T1 > 0

    def test_capture_teacher_traces_returns_anchors(self):
        from src.diagnostic.capture import capture_teacher_traces
        model = MockModelFlow()
        example = ProbeExample(id="p1", benchmark="mmlu_pro", question="?", choices=["A"], target_label="E", input_ids=[1,2,3])
        teacher = capture_teacher_traces(model, example, torch.device("cpu"))
        assert teacher is not None
        assert "h8" in teacher["teacher_anchor_distances"]
        assert "teacher_logits_answer_tokens" in teacher


class MockTokenizer:
    """Mock tokenizer for testing decoder capture."""
    def __init__(self, vocab_size=151936):
        self.vocab_size = vocab_size
        # Map answer labels A-J to fixed token IDs
        self._answer_ids = {label: 17625 + i for i, label in enumerate("ABCDEFGHIJ")}
    
    def encode(self, text, add_special_tokens=True):
        # Return single token id for answer labels
        if text in self._answer_ids:
            return [self._answer_ids[text]]
        # Default to token 0 for unknown
        return [0]
    
    def decode(self, token_ids, skip_special_tokens=True):
        # Reverse lookup for answer labels
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        for label, tid in self._answer_ids.items():
            if tid in token_ids:
                return label
        return "OTHER"


class TestCaptureDecoder:
    def test_capture_decoder_traces_has_kl_js(self):
        from src.diagnostic.capture import capture_decoder_traces
        model = MockModelFlow()
        tokenizer = MockTokenizer()
        example = ProbeExample(id="p1", benchmark="mmlu_pro", question="?", choices=["A"], target_label="E", input_ids=[1,2,3])
        traces = capture_decoder_traces(
            model, tokenizer, example,
            T_values=[1, 8], device=torch.device("cpu"), seed=42,
        )
        assert len(traces) == 2
        dt = [t for t in traces if t.T == 1][0]
        assert dt.T == 1
        assert dt.ground_truth_label == "E"
        assert "A" in dt.logits_answer_tokens
        assert "B" in dt.logits_answer_tokens
        assert "C" in dt.logits_answer_tokens
        assert "D" in dt.logits_answer_tokens
        assert "E" in dt.logits_answer_tokens

    def test_capture_decoder_traces_with_teacher(self):
        from src.diagnostic.capture import capture_decoder_traces, capture_teacher_traces
        model = MockModelFlow()
        tokenizer = MockTokenizer()
        example = ProbeExample(id="p1", benchmark="mmlu_pro", question="?", choices=["A"], target_label="E", input_ids=[1,2,3])
        teacher_data = capture_teacher_traces(model, example, torch.device("cpu"), tokenizer)
        traces = capture_decoder_traces(
            model, tokenizer, example,
            T_values=[1], device=torch.device("cpu"), seed=42,
            teacher_data=teacher_data,
        )
        assert traces[0].kl_divergence >= 0
        assert traces[0].js_divergence >= 0
