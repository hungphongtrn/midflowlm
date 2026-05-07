# Phase 1: Probe Selection & Trace Runner Skeleton

## Phase Goal

A curated fixed probe set is selected from existing stress-test outputs. The checkpoint loads deterministically. A single-example forward pass at T=1 produces trace output (endpoint_hidden + logits) saved as JSON. The skeleton is verified with a manual T=1 vs T=8 comparison showing hidden states differ.

## Files to Touch

- `results/stress_test/mmlu_pro_results.json` - Source for MMLU-Pro probe selection (read-only)
- `results/stress_test/arc_easy_results.json` - Source for ARC-Easy probe selection (read-only)
- `results/diagnostic_p3d3/probes.json` - Output: selected probe examples
- `src/diagnostic/probe.py` - Create: ProbeSet dataclass and loader
- `src/diagnostic/runner.py` - Create: DeterministicTraceRunner class
- `scripts/diagnose_p3d3.py` - Create: CLI entry point
- `tests/test_diagnostic_probe.py` - Create: Unit tests for probe selection
- `tests/test_diagnostic_runner.py` - Create: Unit tests for deterministic runner

## Tasks

### Task 1: Probe Set Selection

**Files:**
- Create: `src/diagnostic/__init__.py`
- Create: `src/diagnostic/probe.py`
- Test: `tests/test_diagnostic_probe.py`

- [ ] **Step 1: Write the failing test**

```python
import json
import pytest
from src.diagnostic.probe import ProbeExample, ProbeSet, select_probes

@pytest.fixture
def sample_stress_test_data(tmp_path):
    mmlu_results = tmp_path / "mmlu_pro_results.json"
    mmlu_results.write_text(json.dumps([
        {
            "id": "mmlu_001",
            "question": "What is X?",
            "choices": ["A. ...", "B. ...", "C. ..."],
            "answer": "E",
            "student_answer_T1": "C",
            "student_answer_T8": "C",
            "teacher_answer": "E",
            "student_correct_T1": False,
            "teacher_correct": True,
        },
        {
            "id": "mmlu_002",
            "question": "What is Y?",
            "choices": ["A. ...", "B. ..."],
            "answer": "F",
            "student_answer_T1": "F",
            "teacher_answer": "F",
            "student_correct_T1": True,
            "teacher_correct": True,
        },
    ]))
    arc_results = tmp_path / "arc_easy_results.json"
    arc_results.write_text(json.dumps([
        {
            "id": "arc_001",
            "question": "What is Z?",
            "choices": ["A", "B", "C", "D"],
            "answer": "A",
            "student_answer_T1": "B",
            "teacher_answer": "A",
            "student_correct_T1": False,
            "teacher_correct": True,
        },
    ]))
    return str(mmlu_results), str(arc_results)


class TestProbeSelection:
    def test_select_mmlu_probes_E_J_labels(self, sample_stress_test_data):
        mmlu_path, arc_path = sample_stress_test_data
        probes = select_probes(mmlu_pro_path=mmlu_path, arc_easy_path=arc_path)
        
        mmlu_probes = [p for p in probes if p.benchmark == "mmlu_pro"]
        assert len(mmlu_probes) == 1
        assert mmlu_probes[0].target_label in "EFGHIJ"
        assert not mmlu_probes[0].teacher_correct or not mmlu_probes[0].student_correct

    def test_select_arc_probes_teacher_correct_student_wrong(self, sample_stress_test_data):
        mmlu_path, arc_path = sample_stress_test_data
        probes = select_probes(mmlu_pro_path=mmlu_path, arc_easy_path=arc_path)
        
        arc_probes = [p for p in probes if p.benchmark == "arc_easy"]
        assert len(arc_probes) == 1
        assert arc_probes[0].teacher_correct
        assert not arc_probes[0].student_correct

    def test_probe_set_minimum_size(self, sample_stress_test_data):
        mmlu_path, arc_path = sample_stress_test_data
        probes = select_probes(
            mmlu_pro_path=mmlu_path, arc_easy_path=arc_path,
            min_mmlu=1, min_arc=1
        )
        assert len(probes) >= 2

    def test_probe_serialization(self, sample_stress_test_data):
        mmlu_path, arc_path = sample_stress_test_data
        probes = select_probes(mmlu_pro_path=mmlu_path, arc_easy_path=arc_path)
        probe_set = ProbeSet(probes=probes, checkpoint_source="test", seed=42)
        
        data = probe_set.to_dict()
        assert "probes" in data
        assert "checkpoint_source" in data
        assert data["seed"] == 42
        
        reloaded = ProbeSet.from_dict(data)
        assert len(reloaded.probes) == len(probes)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_diagnostic_probe.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.diagnostic.probe'`

- [ ] **Step 3: Write minimal implementation**

Create `src/diagnostic/__init__.py` (empty).

Create `src/diagnostic/probe.py`:

```python
from dataclasses import dataclass, field
from typing import List, Optional
import json

@dataclass
class ProbeExample:
    id: str
    benchmark: str  # "mmlu_pro" or "arc_easy"
    question: str
    choices: List[str]
    target_label: str
    prompt_text: str = ""
    input_ids: Optional[List[int]] = None
    student_correct: bool = False
    teacher_correct: bool = False
    student_answer_T1: str = ""
    teacher_answer: str = ""

@dataclass
class ProbeSet:
    probes: List[ProbeExample]
    checkpoint_source: str
    seed: int = 42

    def to_dict(self) -> dict:
        return {
            "probes": [
                {
                    "id": p.id,
                    "benchmark": p.benchmark,
                    "question": p.question,
                    "choices": p.choices,
                    "target_label": p.target_label,
                    "prompt_text": p.prompt_text,
                    "student_correct": p.student_correct,
                    "teacher_correct": p.teacher_correct,
                    "student_answer_T1": p.student_answer_T1,
                    "teacher_answer": p.teacher_answer,
                }
                for p in self.probes
            ],
            "checkpoint_source": self.checkpoint_source,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProbeSet":
        probes = [
            ProbeExample(
                id=p["id"],
                benchmark=p["benchmark"],
                question=p["question"],
                choices=p["choices"],
                target_label=p["target_label"],
                prompt_text=p.get("prompt_text", ""),
                student_correct=p.get("student_correct", False),
                teacher_correct=p.get("teacher_correct", False),
                student_answer_T1=p.get("student_answer_T1", ""),
                teacher_answer=p.get("teacher_answer", ""),
            )
            for p in data["probes"]
        ]
        return cls(
            probes=probes,
            checkpoint_source=data["checkpoint_source"],
            seed=data.get("seed", 42),
        )


def select_probes(
    mmlu_pro_path: str,
    arc_easy_path: str,
    min_mmlu: int = 20,
    min_arc: int = 20,
) -> List[ProbeExample]:
    probes = []

    with open(mmlu_pro_path) as f:
        mmlu_data = json.load(f)
    mmlu_candidates = [
        item for item in mmlu_data
        if item.get("answer", "") in "EFGHIJ"
        and item.get("student_correct_T1", True) is False
    ]
    for item in mmlu_candidates[:min_mmlu]:
        probes.append(ProbeExample(
            id=item.get("id", ""),
            benchmark="mmlu_pro",
            question=item.get("question", ""),
            choices=item.get("choices", []),
            target_label=item.get("answer", ""),
            student_correct=item.get("student_correct_T1", False),
            teacher_correct=item.get("teacher_correct", False),
            student_answer_T1=item.get("student_answer_T1", ""),
            teacher_answer=item.get("teacher_answer", ""),
        ))

    with open(arc_easy_path) as f:
        arc_data = json.load(f)
    arc_candidates = [
        item for item in arc_data
        if item.get("teacher_correct") is True
        and item.get("student_correct_T1") is False
    ]
    for item in arc_candidates[:min_arc]:
        probes.append(ProbeExample(
            id=item.get("id", ""),
            benchmark="arc_easy",
            question=item.get("question", ""),
            choices=item.get("choices", []),
            target_label=item.get("answer", ""),
            student_correct=item.get("student_correct_T1", False),
            teacher_correct=item.get("teacher_correct", True),
            student_answer_T1=item.get("student_answer_T1", ""),
            teacher_answer=item.get("teacher_answer", ""),
        ))

    return probes
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_diagnostic_probe.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/diagnostic/__init__.py src/diagnostic/probe.py tests/test_diagnostic_probe.py
git commit -m "feat: add ProbeSet dataclass and stress-test probe selection"
```

---

### Task 2: Determine Checkpoint Path and Seed

**Files:**
- Create: `src/diagnostic/runner.py`

- [ ] **Step 1: Locate the P3-D3 checkpoint**

Search the stress-test orchestration script or results metadata for the exact checkpoint path. Check `scripts/run_p3d3_stress_test.sh` and any config files used during stress test evaluation.

- [ ] **Step 2: Write the minimal DeterministicTraceRunner skeleton**

In `src/diagnostic/runner.py`:

```python
import json
import torch
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path

from src.diagnostic.probe import ProbeSet


@dataclass
class TraceRecord:
    probe_id: str
    benchmark: str
    T: int
    seed: int
    endpoint_hidden_norm: float
    logits_answer_tokens: Dict[str, float]
    predicted_answer: str
    predicted_token_id: int
    full_logits_shape: str  # serialized shape for sanity check

    def to_dict(self) -> dict:
        return {
            "probe_id": self.probe_id,
            "benchmark": self.benchmark,
            "T": self.T,
            "seed": self.seed,
            "endpoint_hidden_norm": self.endpoint_hidden_norm,
            "logits_answer_tokens": self.logits_answer_tokens,
            "predicted_answer": self.predicted_answer,
            "predicted_token_id": self.predicted_token_id,
            "full_logits_shape": self.full_logits_shape,
        }


def set_deterministic(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DeterministicTraceRunner:
    def __init__(
        self,
        model,
        tokenizer,
        device: torch.device,
        seed: int = 42,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.seed = seed
        self.model.eval()

    def run_single(
        self, example, T: int
    ) -> TraceRecord:
        set_deterministic(self.seed)

        input_ids = torch.tensor([example.input_ids], device=self.device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            output = self.model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_steps=T,
                return_dict=True,
            )

        endpoint_hidden = output.endpoint_hidden
        logits = output.logits[:, -1, :]  # last position

        endpoint_hidden_norm = endpoint_hidden.norm(p=2, dim=-1).mean().item()

        ANSWER_TOKEN_IDS = [ord(c) for c in "ABCDEFGHIJ"]
        logits_answer_tokens = {}
        for i, token_id in enumerate(ANSWER_TOKEN_IDS):
            label = chr(ord("A") + i)
            logits_answer_tokens[label] = logits[0, token_id].item()

        predicted_token_id = logits[0].argmax().item()
        predicted_answer = chr(predicted_token_id) if 65 <= predicted_token_id <= 74 else "OTHER"

        return TraceRecord(
            probe_id=example.id,
            benchmark=example.benchmark,
            T=T,
            seed=self.seed,
            endpoint_hidden_norm=endpoint_hidden_norm,
            logits_answer_tokens=logits_answer_tokens,
            predicted_answer=predicted_answer,
            predicted_token_id=predicted_token_id,
            full_logits_shape=str(list(logits.shape)),
        )

    def run_probe_set(self, probe_set: ProbeSet, T_values: List[int]) -> Dict[int, List[TraceRecord]]:
        results = {}
        for T in T_values:
            records = []
            for probe in probe_set.probes:
                record = self.run_single(probe, T)
                records.append(record)
            results[T] = records
        return results
```

- [ ] **Step 3: Write the CLI entry point skeleton**

Create `scripts/diagnose_p3d3.py`:

```python
#!/usr/bin/env python
"""P3-D3 diagnostic probe — Phase 1 skeleton."""
import argparse
import json
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.diagnostic.probe import ProbeSet, select_probes
from src.diagnostic.runner import DeterministicTraceRunner

def main():
    parser = argparse.ArgumentParser(description="P3-D3 diagnostic probe")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--mmlu-path", type=str, required=True)
    parser.add_argument("--arc-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results/diagnostic_p3d3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--T", type=int, nargs="+", default=[1, 2, 8, 64])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    probes = select_probes(
        mmlu_pro_path=args.mmlu_path,
        arc_easy_path=args.arc_path,
    )
    probe_set = ProbeSet(
        probes=probes,
        checkpoint_source=args.checkpoint,
        seed=args.seed,
    )

    # Write probes
    with open(output_dir / "probes.json", "w") as f:
        json.dump(probe_set.to_dict(), f, indent=2)
    print(f"Selected {len(probes)} probes -> {output_dir / 'probes.json'}")

    # Placeholder: model loading + trace running in Phase 2
    print("Phase 1 skeleton complete. Model loading coming in Phase 2.")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"T values: {args.T}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verify the skeleton runs**

Run:
```bash
python scripts/diagnose_p3d3.py \
    --checkpoint "PLACEHOLDER" \
    --mmlu-path results/stress_test/mmlu_pro_results.json \
    --arc-path results/stress_test/arc_easy_results.json \
    --output-dir results/diagnostic_p3d3
```
Expected: Creates `results/diagnostic_p3d3/probes.json` with selected probes.

---

### Task 3: Deterministic Model Loading + T=1 vs T=8 Smoke Test

**Files:**
- Modify: `src/diagnostic/runner.py` (add model loading)
- Test: `tests/test_diagnostic_runner.py`

- [ ] **Step 1: Identify the checkpoint path**

Read `scripts/run_p3d3_stress_test.sh` to find which checkpoint was used for the stress test. Resolve the path.

- [ ] **Step 2: Write a deterministic loading function**

In `src/diagnostic/runner.py`, add:

```python
def load_model_from_checkpoint(
    checkpoint_path: str,
    config_path: str,
    device: torch.device,
) -> tuple:
    import yaml
    from src.model.student_qwen import FrozenQwenStudent

    with open(config_path) as f:
        config = yaml.safe_load(f)

    model = FrozenQwenStudent(config["model"], config["replacement_model"])
    model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_midblock(checkpoint_path)

    model.eval()
    return model
```

- [ ] **Step 3: Write the minimal smoke test**

Create `tests/test_diagnostic_runner.py`:

```python
def test_deterministic_is_reproducible():
    """Same seed + same T should yield identical hidden states."""

def test_T_changes_hidden_states():
    """T=1 vs T=8 should produce different endpoint_hidden norms."""
```

- [ ] **Step 4: Run the smoke test manually**

Once the checkpoint and config are resolved, verify:
1. Two runs with same seed and T produce identical results.
2. T=1 vs T=8 produce measurably different `endpoint_hidden_norm` values.

- [ ] **Step 5: Commit**

```bash
git add src/diagnostic/runner.py tests/test_diagnostic_runner.py scripts/diagnose_p3d3.py
git commit -m "feat: add deterministic trace runner skeleton and CLI entry point"
```

---

## Phase Completion Criteria
- [ ] `results/diagnostic_p3d3/probes.json` exists with >= 40 examples (>= 20 MMLU-Pro E-J label, >= 20 ARC-Easy teacher-correct/student-wrong)
- [ ] `src/diagnostic/` package with `probe.py` and `runner.py`
- [ ] `scripts/diagnose_p3d3.py` CLI runs the probe selection step
- [ ] Deterministic smoke test confirms same seed yields identical hidden states
- [ ] T=1 vs T=8 produces different `endpoint_hidden_norm` values
- [ ] Checkpoint path is resolved and documented

## Handoff Notes
- The checkpoint path must be pinned in `probes.json` under `checkpoint_source`.
- The exact config file (e.g., `configs/v0_1_matrix/midflow_qwen_8to11_p3_d3_flow_mixc_endtrajkl_trainT_r2468.yaml`) and checkpoint path are the critical parameters passed to Phase 2.
- Phase 2 will expand the runner to capture ODE trajectories, per-step velocity norms, and full decoder logit distributions across all answer tokens.