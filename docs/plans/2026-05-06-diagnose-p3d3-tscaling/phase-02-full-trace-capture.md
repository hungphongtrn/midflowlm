# Phase 2: Full Trace Capture Across T

## Phase Goal

Runner captures two trace families (flow integration + decoder/readout) across T values `{1, 2, 8, 64}` for the full probe set. Teacher targets extracted once via `extract_teacher_targets()`. All artifacts saved as JSON in `results/diagnostic_p3d3/traces/`. Aggregate stats written as CSV.

## Files to Touch

- Create: `src/diagnostic/traces.py` — FlowTrace, DecoderTrace dataclasses
- Create: `src/diagnostic/capture.py` — capture_flow_traces(), capture_decoder_traces(), capture_teacher_traces()
- Modify: `src/diagnostic/runner.py` — add capture integration methods
- Modify: `scripts/diagnose_p3d3.py` — run full capture pipeline
- Create: `tests/test_diagnostic_capture.py` — unit + integration tests

## Architecture Notes from Phase 1

- Token IDs use `tokenizer.encode()` (not ASCII), resolved in Phase 1 review
- `model.forward(return_dict=True)` returns `StudentOutput(endpoint_hidden=h_mid, trajectory_hidden=trajectory_stacked)` where trajectory_stacked is `[batch, seq, num_steps, hidden]`
- `model._forward_ode(return_trajectory=True)` gives per-step trajectory
- `model.extract_teacher_targets(need_trajectory_anchors=True)` gives h8,h9,h10,h11 + teacher_logits
- `src/model/ode.py` has `MidblockVectorField` wrapping `midblock.get_velocity()`
- `Velocity norms` = ||h_{t+dt} - h_t|| / dt along the ODE trajectory

## Tasks

### Task 4: Trace Dataclasses

**Files:**
- Create: `src/diagnostic/traces.py`
- Test: `tests/test_diagnostic_capture.py`

- [ ] **Step 1: Write the failing test**

```python
import json
import pytest
from src.diagnostic.traces import FlowTrace, DecoderTrace

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
        # Roundtrip
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_diagnostic_capture.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.diagnostic.traces'`

- [ ] **Step 3: Write minimal implementation**

Create `src/diagnostic/traces.py`:

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class FlowTrace:
    probe_id: str
    benchmark: str
    T: int
    endpoint_hidden_norm: float
    per_step_velocity_norms: List[float] = field(default_factory=list)
    trajectory_endpoint_norm: float = 0.0
    trajectory_divergence_from_T1: float = 0.0
    teacher_anchor_distances: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "probe_id": self.probe_id,
            "benchmark": self.benchmark,
            "T": self.T,
            "endpoint_hidden_norm": self.endpoint_hidden_norm,
            "per_step_velocity_norms": self.per_step_velocity_norms,
            "trajectory_endpoint_norm": self.trajectory_endpoint_norm,
            "trajectory_divergence_from_T1": self.trajectory_divergence_from_T1,
            "teacher_anchor_distances": self.teacher_anchor_distances,
        }


@dataclass
class DecoderTrace:
    probe_id: str
    benchmark: str
    T: int
    logits_answer_tokens: Dict[str, float] = field(default_factory=dict)
    predicted_answer: str = ""
    predicted_token_id: int = -1
    ground_truth_label: str = ""
    parsed_answer_match: bool = False
    teacher_logits_answer_tokens: Dict[str, float] = field(default_factory=dict)
    kl_divergence: float = 0.0
    js_divergence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "probe_id": self.probe_id,
            "benchmark": self.benchmark,
            "T": self.T,
            "logits_answer_tokens": self.logits_answer_tokens,
            "predicted_answer": self.predicted_answer,
            "predicted_token_id": self.predicted_token_id,
            "ground_truth_label": self.ground_truth_label,
            "parsed_answer_match": self.parsed_answer_match,
            "teacher_logits_answer_tokens": self.teacher_logits_answer_tokens,
            "kl_divergence": self.kl_divergence,
            "js_divergence": self.js_divergence,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_diagnostic_capture.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/diagnostic/traces.py tests/test_diagnostic_capture.py
git commit -m "feat: add FlowTrace and DecoderTrace dataclasses for Phase 2"
```

---

### Task 5: Flow Integration Capture Functions

**Files:**
- Create: `src/diagnostic/capture.py`
- Add tests: `tests/test_diagnostic_capture.py`

- [ ] **Step 1: Write the failing test**

```python
import torch
from src.diagnostic.capture import capture_flow_traces, capture_teacher_traces

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
        model = MockModelFlow()
        example = ProbeExample(id="p1", benchmark="mmlu_pro", question="?", choices=["A"], target_label="A", input_ids=[1,2,3])
        traces = capture_flow_traces(model, example, T_values=[1, 8], device=torch.device("cpu"), seed=42)
        assert len(traces) == 2
        assert traces[0].T == 1
        assert traces[1].T == 8

    def test_capture_flow_traces_has_velocity_norms(self):
        model = MockModelFlow()
        example = ProbeExample(id="p1", benchmark="mmlu_pro", question="?", choices=["A"], target_label="A", input_ids=[1,2,3])
        traces = capture_flow_traces(model, example, T_values=[2], device=torch.device("cpu"), seed=42)
        assert len(traces[0].per_step_velocity_norms) == 2

    def test_capture_flow_traces_divergence_from_T1(self):
        model = MockModelFlow()
        example = ProbeExample(id="p1", benchmark="mmlu_pro", question="?", choices=["A"], target_label="A", input_ids=[1,2,3])
        traces = capture_flow_traces(model, example, T_values=[1, 4, 8], device=torch.device("cpu"), seed=42)
        t4_trace = [t for t in traces if t.T == 4][0]
        assert t4_trace.trajectory_divergence_from_T1 > 0

    def test_capture_teacher_traces_returns_anchors(self):
        model = MockModelFlow()
        example = ProbeExample(id="p1", benchmark="mmlu_pro", question="?", choices=["A"], target_label="E", input_ids=[1,2,3])
        teacher = capture_teacher_traces(model, example, torch.device("cpu"))
        assert teacher is not None
        assert "h8" in teacher["teacher_anchor_distances"]
        assert "teacher_logits_answer_tokens" in teacher
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_diagnostic_capture.py::TestCaptureFlow -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

Create `src/diagnostic/capture.py`:

```python
import torch
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path

from src.diagnostic.probe import ProbeExample
from src.diagnostic.traces import FlowTrace, DecoderTrace
from src.diagnostic.runner import set_deterministic


def capture_flow_traces(
    model,
    example: ProbeExample,
    T_values: List[int],
    device: torch.device,
    seed: int = 42,
) -> List[FlowTrace]:
    set_deterministic(seed)

    input_ids = torch.tensor([example.input_ids], device=device)
    attention_mask = torch.ones_like(input_ids)

    traces = []
    ref_endpoint = None
    ref_trajectory = None

    for T in T_values:
        set_deterministic(seed)

        with torch.no_grad():
            output = model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_steps=T,
                return_dict=True,
            )

        endpoint_hidden = output.endpoint_hidden
        endpoint_hidden_norm = endpoint_hidden.norm(p=2, dim=-1).mean().item()

        per_step_velocity_norms = []
        trajectory_endpoint_norm = endpoint_hidden_norm

        if output.trajectory_hidden is not None:
            traj = output.trajectory_hidden  # [batch, seq, num_steps, hidden]
            num_steps_captured = traj.shape[2]
            for step in range(1, num_steps_captured):
                delta = traj[:, :, step, :] - traj[:, :, step - 1, :]
                dt = 1.0 / T
                vel_norm = (delta.norm(p=2, dim=-1).mean().item()) / dt
                per_step_velocity_norms.append(vel_norm)

            trajectory_endpoint = traj[:, :, -1, :]  # last step in trajectory
            trajectory_endpoint_norm = trajectory_endpoint.norm(p=2, dim=-1).mean().item()

        divergence_from_T1 = 0.0
        if T == 1:
            ref_endpoint = endpoint_hidden
            ref_trajectory = output.trajectory_hidden
        elif ref_endpoint is not None:
            divergence_from_T1 = (endpoint_hidden - ref_endpoint).norm(p=2, dim=-1).mean().item()

        traces.append(FlowTrace(
            probe_id=example.id,
            benchmark=example.benchmark,
            T=T,
            endpoint_hidden_norm=endpoint_hidden_norm,
            per_step_velocity_norms=per_step_velocity_norms,
            trajectory_endpoint_norm=trajectory_endpoint_norm,
            trajectory_divergence_from_T1=divergence_from_T1,
            teacher_anchor_distances={},
        ))

    return traces


def capture_teacher_traces(
    model,
    example: ProbeExample,
    device: torch.device,
    tokenizer=None,
) -> Optional[Dict]:
    input_ids = torch.tensor([example.input_ids], device=device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        teacher = model.extract_teacher_targets(
            input_ids=input_ids,
            attention_mask=attention_mask,
            need_teacher_logits=True,
            need_trajectory_anchors=True,
        )

    answer_labels = "ABCDEFGHIJ"
    teacher_anchor_distances = {}
    if teacher.get("trajectory_anchors"):
        for key, anchor in teacher["trajectory_anchors"].items():
            teacher_anchor_distances[key] = anchor.norm(p=2, dim=-1).mean().item()

    teacher_logits_answer = {}
    if teacher.get("teacher_logits") is not None and tokenizer is not None:
        logits = teacher["teacher_logits"][:, -1, :]
        ANSWER_TOKEN_IDS = {
            label: tokenizer.encode(label, add_special_tokens=False)[0]
            for label in answer_labels
        }
        for label, tid in ANSWER_TOKEN_IDS.items():
            teacher_logits_answer[label] = logits[0, tid].item()

    return {
        "teacher_anchor_distances": teacher_anchor_distances,
        "teacher_logits_answer_tokens": teacher_logits_answer,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_diagnostic_capture.py::TestCaptureFlow -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/diagnostic/capture.py tests/test_diagnostic_capture.py
git commit -m "feat: add flow integration capture functions"
```

---

### Task 6: Decoder/Readout Capture + KL/JS Divergence

**Files:**
- Modify: `src/diagnostic/capture.py` — add capture_decoder_traces()
- Add tests: `tests/test_diagnostic_capture.py`

- [ ] **Step 1: Write the failing test**

```python
class TestCaptureDecoder:
    def test_capture_decoder_traces_has_kl_js(self):
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

    def test_capture_decoder_traces_with_teacher(self):
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_diagnostic_capture.py::TestCaptureDecoder -v`
Expected: FAIL — `capture_decoder_traces` not defined

- [ ] **Step 3: Write implementation**

In `src/diagnostic/capture.py`, add:

```python
import math


def _kl_divergence(p_logits: Dict[str, float], q_logits: Dict[str, float]) -> float:
    labels = sorted(set(p_logits.keys()) & set(q_logits.keys()))
    if not labels:
        return 0.0
    p_vals = np.array([p_logits[l] for l in labels])
    q_vals = np.array([q_logits[l] for l in labels])
    p_probs = np.exp(p_vals - np.max(p_vals))
    p_probs = p_probs / p_probs.sum()
    q_probs = np.exp(q_vals - np.max(q_vals))
    q_probs = q_probs / q_probs.sum()
    kl = np.sum(p_probs * np.log((p_probs + 1e-12) / (q_probs + 1e-12)))
    return float(max(kl, 0.0))


def _js_divergence(p_logits: Dict[str, float], q_logits: Dict[str, float]) -> float:
    labels = sorted(set(p_logits.keys()) & set(q_logits.keys()))
    if not labels:
        return 0.0
    p_vals = np.array([p_logits[l] for l in labels])
    q_vals = np.array([q_logits[l] for l in labels])
    p_probs = np.exp(p_vals - np.max(p_vals))
    p_probs = p_probs / p_probs.sum()
    q_probs = np.exp(q_vals - np.max(q_vals))
    q_probs = q_probs / q_probs.sum()
    m_probs = 0.5 * (p_probs + q_probs)
    kl_pm = np.sum(p_probs * np.log((p_probs + 1e-12) / (m_probs + 1e-12)))
    kl_qm = np.sum(q_probs * np.log((q_probs + 1e-12) / (m_probs + 1e-12)))
    js = 0.5 * kl_pm + 0.5 * kl_qm
    return float(max(js, 0.0))


def capture_decoder_traces(
    model,
    tokenizer,
    example: ProbeExample,
    T_values: List[int],
    device: torch.device,
    seed: int = 42,
    teacher_data: Optional[Dict] = None,
) -> List[DecoderTrace]:
    set_deterministic(seed)

    input_ids = torch.tensor([example.input_ids], device=device)
    attention_mask = torch.ones_like(input_ids)

    answer_labels = "ABCDEFGHIJ"
    ANSWER_TOKEN_IDS = {
        label: tokenizer.encode(label, add_special_tokens=False)[0]
        for label in answer_labels
    }

    traces = []
    for T in T_values:
        set_deterministic(seed)

        with torch.no_grad():
            output = model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_steps=T,
                return_dict=True,
            )

        logits = output.logits[:, -1, :]

        logits_answer_tokens = {}
        for label, token_id in ANSWER_TOKEN_IDS.items():
            logits_answer_tokens[label] = logits[0, token_id].item()

        predicted_token_id = logits[0].argmax().item()
        predicted_answer = tokenizer.decode([predicted_token_id])
        if predicted_answer not in answer_labels:
            predicted_answer = "OTHER"

        parsed_answer_match = predicted_answer == example.target_label

        teacher_logits_answer = {}
        kl = 0.0
        js = 0.0
        if teacher_data is not None and teacher_data.get("teacher_logits_answer_tokens"):
            teacher_logits_answer = teacher_data["teacher_logits_answer_tokens"]
            kl = _kl_divergence(logits_answer_tokens, teacher_logits_answer)
            js = _js_divergence(logits_answer_tokens, teacher_logits_answer)

        traces.append(DecoderTrace(
            probe_id=example.id,
            benchmark=example.benchmark,
            T=T,
            logits_answer_tokens=logits_answer_tokens,
            predicted_answer=predicted_answer,
            predicted_token_id=predicted_token_id,
            ground_truth_label=example.target_label,
            parsed_answer_match=parsed_answer_match,
            teacher_logits_answer_tokens=teacher_logits_answer,
            kl_divergence=kl,
            js_divergence=js,
        ))

    return traces
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_diagnostic_capture.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/diagnostic/capture.py tests/test_diagnostic_capture.py
git commit -m "feat: add decoder/readout capture with KL/JS divergence"
```

---

### Task 7: CLI Integration — Full Capture Pipeline

**Files:**
- Modify: `scripts/diagnose_p3d3.py`
- Modify: `src/diagnostic/runner.py` — add `run_full_capture()` method

- [ ] **Step 1: Add `run_full_capture()` to DeterministicTraceRunner**

In `src/diagnostic/runner.py`:

```python
def run_full_capture(
    self, probe_set: ProbeSet, T_values: List[int]
) -> Dict[str, List]:
    from src.diagnostic.capture import (
        capture_flow_traces,
        capture_decoder_traces,
        capture_teacher_traces,
    )

    flow_results = {}
    decoder_results = {}

    for probe in probe_set.probes:
        teacher_data = capture_teacher_traces(
            self.model, probe, self.device, self.tokenizer
        )

        flow_traces = capture_flow_traces(
            self.model, probe, T_values, self.device, self.seed
        )

        for ft in flow_traces:
            if teacher_data and teacher_data.get("teacher_anchor_distances"):
                for d in flow_traces:
                    d.teacher_anchor_distances = teacher_data["teacher_anchor_distances"]

        decoder_traces = capture_decoder_traces(
            self.model, self.tokenizer, probe,
            T_values, self.device, self.seed,
            teacher_data=teacher_data,
        )

        flow_results[probe.id] = ft_list = [ft.to_dict() for ft in flow_traces]
        decoder_results[probe.id] = dt_list = [dt.to_dict() for dt in decoder_traces]

    return {
        "flow_traces": flow_results,
        "decoder_traces": decoder_results,
    }
```

- [ ] **Step 2: Update CLI to run full capture**

In `scripts/diagnose_p3d3.py`, after model loading, add:

```python
runner = DeterministicTraceRunner(model, tokenizer, device, seed=args.seed)
results = runner.run_full_capture(probe_set, args.T)

# Save per T
out_root = Path(args.output_dir) / "traces"
out_root.mkdir(parents=True, exist_ok=True)
for T_val in args.T:
    T_dir = out_root / f"T{T_val}"
    T_dir.mkdir(exist_ok=True)

    flow_T = {}
    decoder_T = {}
    for pid, traces in results["flow_traces"].items():
        for t in traces:
            if t["T"] == T_val:
                flow_T[pid] = t
    for pid, traces in results["decoder_traces"].items():
        for t in traces:
            if t["T"] == T_val:
                decoder_T[pid] = t

    with open(T_dir / "flow_traces.json", "w") as f:
        json.dump(flow_T, f, indent=2)
    with open(T_dir / "decoder_traces.json", "w") as f:
        json.dump(decoder_T, f, indent=2)

# Generate summary CSV
import csv
summary_path = Path(args.output_dir) / "summary.csv"
with open(summary_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["T", "probe_id", "benchmark", "endpoint_hidden_norm",
                      "predicted_answer", "parsed_answer_match",
                      "kl_divergence", "js_divergence",
                      "mean_velocity_norm"])
    for T_val in args.T:
        for pid in sorted(results["flow_traces"].keys()):
            ft = next((t for t in results["flow_traces"][pid] if t["T"] == T_val), None)
            dt = next((t for t in results["decoder_traces"][pid] if t["T"] == T_val), None)
            if ft and dt:
                mean_vel = np.mean(ft["per_step_velocity_norms"]) if ft["per_step_velocity_norms"] else 0
                writer.writerow([
                    T_val, pid, dt["benchmark"],
                    ft["endpoint_hidden_norm"],
                    dt["predicted_answer"],
                    dt["parsed_answer_match"],
                    dt["kl_divergence"], dt["js_divergence"],
                    mean_vel,
                ])

print(f"Traces written to {out_root}")
print(f"Summary written to {summary_path}")
```

- [ ] **Step 3: Run the smoke capture** (requires checkpoint)

Run: `python scripts/diagnose_p3d3.py --checkpoint ./models/p3_d3_mix_c/checkpoint.pth --config configs/v0_1_matrix/midflow_qwen_8to11_p3_d3_flow_mixc_endtrajkl_trainT_r2468.yaml --mmlu-path results/stress_test/mmlu_pro_results.json --arc-path results/stress_test/arc_easy_results.json --T 1 8 --device cpu`

Expected: Creates `results/diagnostic_p3d3/traces/T1/` and `T8/` with flow_traces.json and decoder_traces.json

- [ ] **Step 4: Commit**

```bash
git add src/diagnostic/runner.py scripts/diagnose_p3d3.py
git commit -m "feat: integrate full capture pipeline into CLI"
```

---

### Task 8: End-to-End Smoke Test & Verification

- [ ] **Step 1: Verify the full pipeline** against actual checkpoint

Run with actual model on 2 probes (T=1 and T=8):
```bash
python scripts/diagnose_p3d3.py \
    --checkpoint ./models/p3_d3_mix_c/checkpoint.pth \
    --config configs/v0_1_matrix/midflow_qwen_8to11_p3_d3_flow_mixc_endtrajkl_trainT_r2468.yaml \
    --mmlu-path results/stress_test/mmlu_pro_results.json \
    --arc-path results/stress_test/arc_easy_results.json \
    --T 1 8 \
    --output-dir results/diagnostic_p3d3
```

- [ ] **Step 2: Verify artifacts exist and are valid JSON**

- [ ] **Step 3: Verify endpoint_hidden_norm changes between T=1 and T=8**

```python
import json
with open("results/diagnostic_p3d3/traces/T1/flow_traces.json") as f: t1 = json.load(f)
with open("results/diagnostic_p3d3/traces/T8/flow_traces.json") as f: t8 = json.load(f)
deltas = [abs(t8[pid]["endpoint_hidden_norm"] - t1[pid]["endpoint_hidden_norm"]) for pid in t1]
assert any(d > 0 for d in deltas), "endpoint_hidden_norm is identical across T — flow is dead"
```

- [ ] **Step 5: Commit (if clean-up needed)**

---

## Phase Completion Criteria
- [ ] Flow traces capture working: endpoint_hidden_norm, per_step_velocity_norms, trajectory_endpoint_norm, divergence_from_T1
- [ ] Decoder traces capture working: logits_answer_tokens, predicted_answer, parsed_answer_match, KL/JS divergence with teacher
- [ ] Teacher traces captured: teacher_anchor_distances, teacher_logits_answer_tokens
- [ ] CLI writes per-T trace JSONs in `results/diagnostic_p3d3/traces/`
- [ ] Summary CSV generated
- [ ] Endpoint_hidden_norm measurably changes between T=1 and T=8
- [ ] All tests pass

## Handoff Notes
- `results/diagnostic_p3d3/summary.csv` is the primary artifact for Phase 3 report generation
- The `trajectory_divergence_from_T1` field answers "Does increasing T change hidden states before decoding?"
- The `kl_divergence` field answers "Does increasing T change answer distributions?"
- The `parsed_answer_match` field answers "Does increasing T change parsed predictions?"