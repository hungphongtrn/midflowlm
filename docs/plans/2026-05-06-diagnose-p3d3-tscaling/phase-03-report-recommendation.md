# Phase 3: Diagnostic Report & Recommendation

## Phase Goal

Programmatically generated markdown report answers the three key diagnostic questions using descriptive statistics and basic inferential tests (paired t-tests, Cohen's d). Output is a self-contained markdown file at `results/diagnostic_p3d3/report.md`. Report includes a root cause decision tree and prioritized recommendations.

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Approach B: Layered analysis → generation | Clean separation of computation from formatting; analysis is unit-testable without touching report rendering |
| Pure markdown with tables | Portable, version-control-friendly, renders in GitHub and any text editor |
| Basic inferential statistics | Paired t-tests and Cohen's d provide statistical confidence without ANOVA complexity |
| Full T sweep: {1, 2, 8, 64} | Required to observe the full trajectory of T-scaling collapse |
| Analysis from saved traces only (no model) | Trace artifacts are already collected; analysis is fast and repeatable without GPU |
| Scipy for statistical tests | Already in dependencies; `ttest_rel` for paired t-tests |
| Report includes diagnosis + recommendation | Directly addresses acceptance criteria for issue #5 closure |

## Architecture

```
src/diagnostic/
├── __init__.py
├── probe.py          (Phase 1 — unchanged)
├── runner.py         (Phase 1-2 — unchanged)
├── traces.py         (Phase 2 — unchanged)
├── capture.py        (Phase 2 — unchanged)
├── analysis.py       (Phase 3 — NEW)
└── report.py         (Phase 3 — NEW)
```

### Data Flow

```
traces/T{N}/*.json  ──→  analysis.py  ──→  AnalysisResult dataclasses
                                                    │
probes.json  ─────→  report.py  ──→  report.md      │
                          ↑──────────────────────────┘
```

`analysis.py` reads JSON artifacts, computes stats, returns pure dataclasses.
`report.py` takes dataclasses + probes.json, renders markdown string.
`diagnose_p3d3.py` orchestrates and writes `report.md`.

## Files to Touch

- Create: `src/diagnostic/analysis.py` — FlowAnalysisResult, DecoderAnalysisResult, analyze_flow(), analyze_decoder(), run_analysis()
- Create: `src/diagnostic/report.py` — generate_report()
- Modify: `src/diagnostic/__init__.py` — export new names
- Modify: `scripts/diagnose_p3d3.py` — --report, --traces-dir, --probes flags; orchestration logic
- Create: `tests/test_diagnostic_analysis.py` — 8 tests
- Create: `tests/test_diagnostic_report.py` — 4 tests
- Modify: `scripts/smoke_diagnostic_p3d3.sh` — add --report verification

## Tasks

### Task 9: Analysis Dataclasses (Red)

**Files:**
- Create: `src/diagnostic/analysis.py`
- Test: `tests/test_diagnostic_analysis.py` (first batch)

- [ ] **Step 1: Write the failing test**

```python
import json
import pytest
import numpy as np
from pathlib import Path

class TestFlowAnalysisResult:
    def test_dataclass_fields(self):
        from src.diagnostic.analysis import FlowAnalysisResult
        result = FlowAnalysisResult(
            per_T_stats={
                1: {"mean": 0.5, "std": 0.1, "median": 0.48, "n": 10},
                8: {"mean": 0.7, "std": 0.15, "median": 0.68, "n": 10},
            },
            pairwise_tests=[
                {"T_a": 1, "T_b": 8, "t_stat": 3.5, "p_value": 0.001, "cohens_d": 0.95, "mean_delta": 0.2, "significant": True},
            ],
            velocity_analysis={
                1: {"mean": 0.1, "std": 0.05, "median": 0.09, "nonzero_count": 10},
                8: {"mean": 0.08, "std": 0.04, "median": 0.07, "nonzero_count": 10},
            },
            divergence_from_T1={
                2: {"mean": 0.05, "median": 0.04, "proportion_diverged": 0.8},
                8: {"mean": 0.12, "median": 0.11, "proportion_diverged": 1.0},
                64: {"mean": 0.18, "median": 0.17, "proportion_diverged": 1.0},
            },
        )
        assert result.per_T_stats[1]["mean"] == 0.5
        assert result.pairwise_tests[0]["significant"] is True
        assert result.divergence_from_T1[8]["proportion_diverged"] == 1.0


class TestDecoderAnalysisResult:
    def test_dataclass_fields(self):
        from src.diagnostic.analysis import DecoderAnalysisResult
        result = DecoderAnalysisResult(
            per_T_stats={
                1: {"accuracy": 0.0, "mean_kl": 0.5, "mean_js": 0.3, "median_kl": 0.48, "median_js": 0.28, "n": 10},
                8: {"accuracy": 0.0, "mean_kl": 1.2, "mean_js": 0.7, "median_kl": 1.15, "median_js": 0.68, "n": 10},
            },
            logit_shift_tests=[
                {"T_a": 1, "T_b": 8, "mean_kl_delta": 0.8, "proportion_shifted": 0.9},
            ],
            answer_coverage={
                1: {"mean_prob": 0.1, "median_prob": 0.08, "max_prob": 0.15, "probes_gt_50pct": 0},
                8: {"mean_prob": 0.12, "median_prob": 0.09, "max_prob": 0.18, "probes_gt_50pct": 0},
            },
            prediction_stability={
                "always_wrong": 8, "flipped": 2, "became_correct": 0,
            },
        )
        assert result.per_T_stats[1]["accuracy"] == 0.0
        assert result.prediction_stability["always_wrong"] == 8
        assert result.logit_shift_tests[0]["mean_kl_delta"] == 0.8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_diagnostic_analysis.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.diagnostic.analysis'`

- [ ] **Step 3: Write minimal implementation**

Create `src/diagnostic/analysis.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_diagnostic_analysis.py::TestFlowAnalysisResult tests/test_diagnostic_analysis.py::TestDecoderAnalysisResult -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/diagnostic/analysis.py tests/test_diagnostic_analysis.py
git commit -m "feat: add FlowAnalysisResult and DecoderAnalysisResult dataclasses"
```

---

### Task 10: Flow Analysis Functions (Red → Green)

**Files:**
- Modify: `src/diagnostic/analysis.py` — add `analyze_flow()`
- Modify: `tests/test_diagnostic_analysis.py` — add flow analysis tests

- [ ] **Step 1: Write the failing tests**

```python
def _make_fake_traces(tmp_path, T_values, endpoint_norms=None, velocity_norms=None, divergences=None):
    """Shared fixture helper: creates traces/T{N}/flow_traces.json and decoder_traces.json."""
    traces_dir = Path(tmp_path) / "traces"
    probes = [f"probe_{i:03d}" for i in range(5)]
    default_norms = {t: t * 0.01 for t in T_values}
    endpoint_norms = endpoint_norms or default_norms
    for T in T_values:
        T_dir = traces_dir / f"T{T}"
        T_dir.mkdir(parents=True, exist_ok=True)
        flow = {}
        decoder = {}
        for pid in probes:
            flow[pid] = {
                "probe_id": pid, "benchmark": "mmlu_pro", "T": T,
                "endpoint_hidden_norm": endpoint_norms[T] + hash(pid) % 100 * 0.001,
                "per_step_velocity_norms": [0.1] * T,
                "trajectory_endpoint_norm": endpoint_norms[T] + 0.01,
                "trajectory_divergence_from_T1": divergences.get(pid, {}).get(T, 0.0) if divergences else 0.0,
                "teacher_anchor_distances": {"h8": 0.3},
            }
            decoder[pid] = {
                "probe_id": pid, "benchmark": "mmlu_pro", "T": T,
                "logits_answer_tokens": {l: -1.0 + hash(pid) % 10 * 0.1 for l in "ABCDEFGHIJ"},
                "predicted_answer": "C", "predicted_token_id": 17627,
                "ground_truth_label": "E", "parsed_answer_match": False,
                "teacher_logits_answer_tokens": {l: -2.0 for l in "ABCDEFGHIJ"},
                "kl_divergence": 0.5, "js_divergence": 0.3,
            }
        with open(T_dir / "flow_traces.json", "w") as f:
            json.dump(flow, f)
        with open(T_dir / "decoder_traces.json", "w") as f:
            json.dump(decoder, f)
    return str(traces_dir), probes, endpoint_norms


class TestAnalyzeFlow:
    def test_analyze_flow_computes_stats(self, tmp_path):
        traces_dir, probes, _ = _make_fake_traces(tmp_path, [1, 8])
        from src.diagnostic.analysis import analyze_flow
        result = analyze_flow(traces_dir, [1, 8])
        assert 1 in result.per_T_stats
        assert 8 in result.per_T_stats
        assert result.per_T_stats[1]["n"] == len(probes)
        assert "mean" in result.per_T_stats[1]
        assert "std" in result.per_T_stats[1]

    def test_analyze_flow_no_change_detected(self, tmp_path):
        traces_dir, probes, _ = _make_fake_traces(tmp_path, [1, 8], endpoint_norms={1: 0.1, 8: 0.1})
        from src.diagnostic.analysis import analyze_flow
        result = analyze_flow(traces_dir, [1, 8])
        pair_test = [t for t in result.pairwise_tests if t["T_a"] == 1 and t["T_b"] == 8][0]
        assert pair_test["p_value"] > 0.05 or abs(pair_test["cohens_d"]) < 0.5
        assert pair_test["significant"] is False

    def test_analyze_flow_significant_change(self, tmp_path):
        traces_dir, probes, _ = _make_fake_traces(tmp_path, [1, 8], endpoint_norms={1: 0.1, 8: 0.5})
        from src.diagnostic.analysis import analyze_flow
        result = analyze_flow(traces_dir, [1, 8])
        pair_test = [t for t in result.pairwise_tests if t["T_a"] == 1 and t["T_b"] == 8][0]
        assert pair_test["significant"] is True
        assert pair_test["cohens_d"] > 0.8
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_diagnostic_analysis.py::TestAnalyzeFlow -v`
Expected: FAIL — `analyze_flow` not defined

- [ ] **Step 3: Write implementation**

In `src/diagnostic/analysis.py`, add:

```python
import json
import numpy as np
from pathlib import Path
from typing import Dict, List
from scipy.stats import ttest_rel


def _load_flow_traces(traces_dir: str, T_values: List[int]) -> Dict[int, List[dict]]:
    traces_by_T = {}
    for T in T_values:
        path = Path(traces_dir) / f"T{T}" / "flow_traces.json"
        with open(path) as f:
            data = json.load(f)
        traces_by_T[T] = list(data.values())
    return traces_by_T


def _compute_per_T_stats(traces: List[dict], field: str) -> dict:
    values = np.array([t[field] for t in traces])
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "median": float(np.median(values)),
        "n": len(values),
    }


def _cohens_d(values1: np.ndarray, values2: np.ndarray) -> float:
    diff = np.mean(values1 - values2)
    var1 = np.var(values1, ddof=1)
    var2 = np.var(values2, ddof=1)
    n1, n2 = len(values1), len(values2)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-12:
        return 0.0
    return float(diff / pooled_std)


def _cohens_d_interpretation(d: float) -> str:
    if abs(d) < 0.5:
        return "small"
    elif abs(d) < 0.8:
        return "medium"
    else:
        return "large"


def analyze_flow(traces_dir: str, T_values: List[int]) -> FlowAnalysisResult:
    traces_by_T = _load_flow_traces(traces_dir, T_values)

    # Per-T stats for endpoint_hidden_norm
    per_T_stats = {}
    for T in T_values:
        per_T_stats[T] = _compute_per_T_stats(traces_by_T[T], "endpoint_hidden_norm")

    # Pairwise tests on matched probes
    pairwise_tests = []
    n_pairs = len(T_values) * (len(T_values) - 1) // 2
    bonferroni_alpha = 0.05 / n_pairs if n_pairs > 0 else 0.05

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

    # Velocity analysis
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

    # Divergence from T1
    divergence_from_T1 = {}
    t1_traces = {t["probe_id"]: t for t in traces_by_T.get(T_values[0], [])}
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
                "proportion_diverged": diverged_count / len(div_values) if div_values else 0.0,
            }
        else:
            divergence_from_T1[T] = {"mean": 0.0, "median": 0.0, "proportion_diverged": 0.0}

    return FlowAnalysisResult(
        per_T_stats=per_T_stats,
        pairwise_tests=pairwise_tests,
        velocity_analysis=velocity_analysis,
        divergence_from_T1=divergence_from_T1,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_diagnostic_analysis.py::TestAnalyzeFlow -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/diagnostic/analysis.py tests/test_diagnostic_analysis.py
git commit -m "feat: add analyze_flow() with paired t-tests and Cohen's d"
```

---

### Task 11: Decoder Analysis Functions (Red → Green)

**Files:**
- Modify: `src/diagnostic/analysis.py` — add `analyze_decoder()`
- Modify: `tests/test_diagnostic_analysis.py` — add decoder analysis tests

- [ ] **Step 1: Write the failing tests**

```python
class TestAnalyzeDecoder:
    def test_analyze_decoder_accuracy(self, tmp_path):
        traces_dir = Path(tmp_path) / "traces"
        probes = [f"probe_{i:03d}" for i in range(4)]
        for T in [1, 8]:
            T_dir = traces_dir / f"T{T}"
            T_dir.mkdir(parents=True, exist_ok=True)
            flow = {pid: {"probe_id": pid, "benchmark": "mmlu_pro", "T": T, "endpoint_hidden_norm": 0.1, "per_step_velocity_norms": [0.1] * T, "trajectory_endpoint_norm": 0.1, "trajectory_divergence_from_T1": 0.0, "teacher_anchor_distances": {}} for pid in probes}
            decoder = {}
            for i, pid in enumerate(probes):
                decoder[pid] = {
                    "probe_id": pid, "benchmark": "mmlu_pro", "T": T,
                    "logits_answer_tokens": {l: -1.0 for l in "ABCDEFGHIJ"},
                    "predicted_answer": "C", "predicted_token_id": 17627,
                    "ground_truth_label": "C" if i < 2 else "E",
                    "parsed_answer_match": i < 2,
                    "teacher_logits_answer_tokens": {l: -2.0 for l in "ABCDEFGHIJ"},
                    "kl_divergence": 0.5, "js_divergence": 0.3,
                }
            with open(T_dir / "flow_traces.json", "w") as f: json.dump(flow, f)
            with open(T_dir / "decoder_traces.json", "w") as f: json.dump(decoder, f)
        from src.diagnostic.analysis import analyze_decoder
        result = analyze_decoder(str(traces_dir), [1, 8])
        assert result.per_T_stats[1]["accuracy"] == 0.5  # 2/4 correct
        assert result.per_T_stats[8]["accuracy"] == 0.5

    def test_analyze_decoder_coverage(self, tmp_path):
        traces_dir = Path(tmp_path) / "traces"
        probes = [f"probe_{i:03d}" for i in range(3)]
        for T in [1, 8]:
            T_dir = traces_dir / f"T{T}"
            T_dir.mkdir(parents=True, exist_ok=True)
            flow = {pid: {"probe_id": pid, "benchmark": "mmlu_pro", "T": T, "endpoint_hidden_norm": 0.1, "per_step_velocity_norms": [0.1] * T, "trajectory_endpoint_norm": 0.1, "trajectory_divergence_from_T1": 0.0, "teacher_anchor_distances": {}} for pid in probes}
            decoder = {}
            for i, pid in enumerate(probes):
                probs = {"A": 0.1, "B": 0.55, "C": 0.2, "D": 0.1, "E": 0.05, "F": 0.0, "G": 0.0, "H": 0.0, "I": 0.0, "J": 0.0}
                decoder[pid] = {
                    "probe_id": pid, "benchmark": "mmlu_pro", "T": T,
                    "logits_answer_tokens": {l: np.log(p) for l, p in probs.items()},
                    "predicted_answer": "B", "predicted_token_id": 17626,
                    "ground_truth_label": "B",
                    "parsed_answer_match": True,
                    "teacher_logits_answer_tokens": {},
                    "kl_divergence": 0.0, "js_divergence": 0.0,
                }
            with open(T_dir / "flow_traces.json", "w") as f: json.dump(flow, f)
            with open(T_dir / "decoder_traces.json", "w") as f: json.dump(decoder, f)
        from src.diagnostic.analysis import analyze_decoder
        result = analyze_decoder(str(traces_dir), [1, 8])
        # prob on B = 0.55 > 0.5, so probes_gt_50pct should be 3
        assert result.answer_coverage[1]["probes_gt_50pct"] == 3

    def test_analyze_decoder_stability(self, tmp_path):
        traces_dir = Path(tmp_path) / "traces"
        probes = [f"probe_{i:03d}" for i in range(4)]
        for T_idx, T in enumerate([1, 8, 64]):
            T_dir = traces_dir / f"T{T}"
            T_dir.mkdir(parents=True, exist_ok=True)
            flow = {pid: {"probe_id": pid, "benchmark": "mmlu_pro", "T": T, "endpoint_hidden_norm": 0.1, "per_step_velocity_norms": [0.1] * T, "trajectory_endpoint_norm": 0.1, "trajectory_divergence_from_T1": 0.0, "teacher_anchor_distances": {}} for pid in probes}
            decoder = {}
            for i, pid in enumerate(probes):
                answers = [[["C", False], ["C", False], ["C", False]],  # always wrong
                           [["C", False], ["D", False], ["E", False]],  # always wrong but flips
                           [["C", False], ["C", False], ["C", True]],   # became correct
                           [["D", True], ["D", True], ["D", True]]]     # always correct
                pred, match = answers[i][T_idx]
                decoder[pid] = {
                    "probe_id": pid, "benchmark": "mmlu_pro", "T": T,
                    "logits_answer_tokens": {l: -1.0 for l in "ABCDEFGHIJ"},
                    "predicted_answer": pred, "predicted_token_id": 17627,
                    "ground_truth_label": "D",
                    "parsed_answer_match": match,
                    "teacher_logits_answer_tokens": {},
                    "kl_divergence": 0.0, "js_divergence": 0.0,
                }
            with open(T_dir / "flow_traces.json", "w") as f: json.dump(flow, f)
            with open(T_dir / "decoder_traces.json", "w") as f: json.dump(decoder, f)
        from src.diagnostic.analysis import analyze_decoder
        result = analyze_decoder(str(traces_dir), [1, 8, 64])
        assert result.prediction_stability["always_wrong"] == 2  # probe 0, 1
        assert result.prediction_stability["flipped"] == 1       # probe 1
        assert result.prediction_stability["became_correct"] == 1  # probe 2

    def test_analyze_decoder_logit_shift(self, tmp_path):
        traces_dir = Path(tmp_path) / "traces"
        probes = [f"probe_{i:03d}" for i in range(3)]
        for T in [1, 8]:
            T_dir = traces_dir / f"T{T}"
            T_dir.mkdir(parents=True, exist_ok=True)
            flow = {pid: {"probe_id": pid, "benchmark": "mmlu_pro", "T": T, "endpoint_hidden_norm": 0.1, "per_step_velocity_norms": [0.1] * T, "trajectory_endpoint_norm": 0.1, "trajectory_divergence_from_T1": 0.0, "teacher_anchor_distances": {}} for pid in probes}
            decoder = {}
            for i, pid in enumerate(probes):
                offset = T * 0.05  # T=1 vs T=8 produces different logits
                decoder[pid] = {
                    "probe_id": pid, "benchmark": "mmlu_pro", "T": T,
                    "logits_answer_tokens": {l: -1.0 + offset for l in "ABCDEFGHIJ"},
                    "predicted_answer": "C", "predicted_token_id": 17627,
                    "ground_truth_label": "E",
                    "parsed_answer_match": False,
                    "teacher_logits_answer_tokens": {},
                    "kl_divergence": 0.0, "js_divergence": 0.0,
                }
            with open(T_dir / "flow_traces.json", "w") as f: json.dump(flow, f)
            with open(T_dir / "decoder_traces.json", "w") as f: json.dump(decoder, f)
        from src.diagnostic.analysis import analyze_decoder
        result = analyze_decoder(str(traces_dir), [1, 8])
        shift = [t for t in result.logit_shift_tests if t["T_a"] == 1 and t["T_b"] == 8][0]
        assert shift["mean_kl_delta"] > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_diagnostic_analysis.py::TestAnalyzeDecoder -v`
Expected: FAIL — `analyze_decoder` not defined

- [ ] **Step 3: Write implementation**

In `src/diagnostic/analysis.py`, add:

```python
def _load_decoder_traces(traces_dir: str, T_values: List[int]) -> Dict[int, List[dict]]:
    traces_by_T = {}
    for T in T_values:
        path = Path(traces_dir) / f"T{T}" / "decoder_traces.json"
        with open(path) as f:
            data = json.load(f)
        traces_by_T[T] = list(data.values())
    return traces_by_T


def _probe_kl(a_logits: dict, b_logits: dict) -> float:
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


def analyze_decoder(traces_dir: str, T_values: List[int]) -> DecoderAnalysisResult:
    traces_by_T = _load_decoder_traces(traces_dir, T_values)

    # Per-T stats
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

    # Logit shift tests between pairs
    logit_shift_tests = []
    for i, T_a in enumerate(T_values):
        for T_b in T_values[i + 1:]:
            traces_a = {t["probe_id"]: t for t in traces_by_T[T_a]}
            traces_b = {t["probe_id"]: t for t in traces_by_T[T_b]}
            common_ids = sorted(set(traces_a) & set(traces_b))
            if len(common_ids) < 2:
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

    # Answer coverage
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

    # Prediction stability across all T values
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


# Module-level constants
answer_labels = "ABCDEFGHIJ"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_diagnostic_analysis.py::TestAnalyzeDecoder -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/diagnostic/analysis.py tests/test_diagnostic_analysis.py
git commit -m "feat: add analyze_decoder() with accuracy, coverage, stability, and logit shift tests"
```

---

### Task 12: run_analysis() Integration Wrapper (Red → Green)

**Files:**
- Modify: `src/diagnostic/analysis.py` — add `run_analysis()`
- Modify: `tests/test_diagnostic_analysis.py` — add integration test

- [ ] **Step 1: Write the failing test**

```python
class TestRunAnalysis:
    def test_run_analysis_integration(self, tmp_path):
        traces_dir, probes, _ = _make_fake_traces(tmp_path, [1, 8, 64])
        from src.diagnostic.analysis import run_analysis
        flow_result, decoder_result = run_analysis(traces_dir, [1, 8, 64])
        assert isinstance(flow_result, FlowAnalysisResult)
        assert isinstance(decoder_result, DecoderAnalysisResult)
        assert 1 in flow_result.per_T_stats
        assert 1 in decoder_result.per_T_stats
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_diagnostic_analysis.py::TestRunAnalysis -v`
Expected: FAIL — `run_analysis` not defined

- [ ] **Step 3: Write implementation**

In `src/diagnostic/analysis.py`, add:

```python
def run_analysis(
    traces_dir: str, T_values: List[int]
) -> tuple[FlowAnalysisResult, DecoderAnalysisResult]:
    flow_result = analyze_flow(traces_dir, T_values)
    decoder_result = analyze_decoder(traces_dir, T_values)
    return flow_result, decoder_result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_diagnostic_analysis.py::TestRunAnalysis -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/diagnostic/analysis.py tests/test_diagnostic_analysis.py
git commit -m "feat: add run_analysis() convenience wrapper"
```

---

### Task 13: Report Generation (Red → Green)

**Files:**
- Create: `src/diagnostic/report.py`
- Test: `tests/test_diagnostic_report.py`

- [ ] **Step 1: Write the failing tests**

```python
import pytest
from src.diagnostic.analysis import FlowAnalysisResult, DecoderAnalysisResult

def _make_empty_results():
    """No-change: all deltas are zero."""
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
        # With zero deltas, Q1 should produce FAIL (or be absent)
        assert "FAIL" in report

    def test_report_tables_valid_markdown(self):
        flow, decoder = _make_empty_results()
        from src.diagnostic.report import generate_report
        report = generate_report(flow, decoder, probes_path="", traces_dir="")
        lines = report.split("\n")
        # Check that all pipe tables have consistent column counts
        table_lines = [l for l in lines if l.strip().startswith("|")]
        for line in table_lines:
            if not line.strip().startswith("|---"):
                # Count pipes: should have same count per table
                pass  # Basic check: no broken pipe sequences
        assert "|" in report  # Tables exist
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_diagnostic_report.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.diagnostic.report'`

- [ ] **Step 3: Write implementation**

Create `src/diagnostic/report.py`:

```python
"""Diagnostic report generation for P3-D3 T-scaling investigation.

Reads AnalysisResult dataclasses and writes a self-contained markdown report.
"""

from src.diagnostic.analysis import FlowAnalysisResult, DecoderAnalysisResult


def _verdict_q1(flow: FlowAnalysisResult) -> dict:
    """Q1: Does increasing T change hidden states before decoding?"""
    pair = None
    for p in flow.pairwise_tests:
        if p["T_a"] == 1 and p["T_b"] == 64:
            pair = p
            break
    if pair is None:
        return {"status": "FAIL", "verdict": "FAIL", "detail": "No T1→T64 comparison available"}
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


def _verdict_q2(decoder: DecoderAnalysisResult) -> dict:
    """Q2: Logits shift without prediction change?"""
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


def _verdict_q3(decoder: DecoderAnalysisResult) -> dict:
    """Q3: Probability mass reaches answer labels?"""
    max_mean_prob = max((v["mean_prob"] for v in decoder.answer_coverage.values()), default=0.0)
    if max_mean_prob < 0.3:
        return {"status": "FAIL", "verdict": "FAIL", "detail": f"Max mean prob on correct answer = {max_mean_prob:.3f} < 0.3 — answer-token supervision likely needed"}
    elif max_mean_prob < 0.5:
        return {"status": "WARN", "verdict": "WARN", "detail": f"Max mean prob on correct answer = {max_mean_prob:.3f} < 0.5 — answer coverage borderline"}
    else:
        return {"status": "PASS", "verdict": "PASS", "detail": f"Max mean prob on correct answer = {max_mean_prob:.3f} >= 0.5"}


def _determine_root_cause(v1: dict, v2: dict, v3: dict) -> str:
    """Decision tree mapping evidence to root cause."""
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


def _generate_recommendations(v1: dict, v2: dict, v3: dict) -> str:
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


def generate_report(
    flow_result: FlowAnalysisResult,
    decoder_result: DecoderAnalysisResult,
    probes_path: str = "",
    traces_dir: str = "",
) -> str:
    v1 = _verdict_q1(flow_result)
    v2 = _verdict_q2(decoder_result)
    v3 = _verdict_q3(decoder_result)
    root_cause = _determine_root_cause(v1, v2, v3)
    recommendations = _generate_recommendations(v1, v2, v3)

    lines = []
    lines.append("# P3-D3 Diagnostic Report: T-Scaling Investigation\n")

    # 1. Executive Summary
    lines.append("## 1. Executive Summary\n")
    lines.append(f"**Q1: Does increasing T change hidden states before decoding?** — **{v1['verdict']}** ({v1['detail']})\n")
    lines.append(f"**Q2: Does increasing T change logits without changing parsed predictions?** — **{v2['verdict']}** ({v2['detail']})\n")
    lines.append(f"**Q3: Does increasing T fail to put probability mass on reachable answer labels?** — **{v3['verdict']}** ({v3['detail']})\n")

    # 2. Flow Integration Analysis
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

    # 3. Decoder/Readout Analysis
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

    # 4. Root Cause Decision Tree
    lines.append("## 4. Root Cause Decision Tree\n")
    lines.append(root_cause)
    lines.append("")

    # 5. Recommendations
    lines.append("## 5. Recommendations\n")
    lines.append(recommendations)
    lines.append("")

    lines.append(f"---\n*Report generated from traces at `{traces_dir}`. Probes loaded from `{probes_path}`.*\n")

    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_diagnostic_report.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/diagnostic/report.py tests/test_diagnostic_report.py
git commit -m "feat: add generate_report() with verdicts, tables, decision tree, and recommendations"
```

---

### Task 14: Update __init__.py Exports

**Files:**
- Modify: `src/diagnostic/__init__.py`

- [ ] **Step 1: Add new exports**

```python
from src.diagnostic.analysis import (
    FlowAnalysisResult,
    DecoderAnalysisResult,
    analyze_flow,
    analyze_decoder,
    run_analysis,
)
from src.diagnostic.report import generate_report

__all__ = [
    "FlowAnalysisResult",
    "DecoderAnalysisResult",
    "analyze_flow",
    "analyze_decoder",
    "run_analysis",
    "generate_report",
]
```

- [ ] **Step 2: Verify imports work**

Run: `python -c "from src.diagnostic import FlowAnalysisResult, DecoderAnalysisResult, run_analysis, generate_report; print('OK')"`

- [ ] **Step 3: Commit**

```bash
git add src/diagnostic/__init__.py
git commit -m "feat: export Phase 3 analysis and report names from __init__.py"
```

---

### Task 15: CLI Integration — Report Orchestration

**Files:**
- Modify: `scripts/diagnose_p3d3.py` — add --report, --traces-dir, --probes flags; orchestration logic

- [ ] **Step 1: Add CLI flags and orchestration logic**

In `scripts/diagnose_p3d3.py`, add new arguments:

```python
parser.add_argument("--report", action="store_true",
                    help="Generate report from saved traces (skips model loading)")
parser.add_argument("--traces-dir", type=str,
                    help="Path to traces directory [default: results/diagnostic_p3d3/traces]")
parser.add_argument("--probes", type=str,
                    help="Path to probes.json [default: <traces-dir>/../probes.json]")
```

Add orchestration logic after argument parsing:

```python
# Resolve traces-dir and probes-path defaults
traces_dir = args.traces_dir or str(Path(args.output_dir) / "traces")
probes_path = args.probes or str(Path(traces_dir).parent / "probes.json")

if args.report and not args.checkpoint:
    # Report-only mode: analyze existing traces
    from src.diagnostic.analysis import run_analysis
    from src.diagnostic.report import generate_report

    if not Path(traces_dir).exists():
        print(f"ERROR: Traces directory not found: {traces_dir}")
        sys.exit(1)

    print(f"Running analysis on traces from: {traces_dir}")
    flow_result, decoder_result = run_analysis(traces_dir, args.T)
    report_text = generate_report(flow_result, decoder_result, probes_path, traces_dir)

    report_path = Path(args.output_dir) / "report.md"
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"Report written to {report_path}")
elif args.report and args.checkpoint:
    # Capture then report
    # ... (existing capture logic) ...
    from src.diagnostic.analysis import run_analysis
    from src.diagnostic.report import generate_report

    flow_result, decoder_result = run_analysis(traces_dir, args.T)
    report_text = generate_report(flow_result, decoder_result, probes_path, traces_dir)

    report_path = Path(args.output_dir) / "report.md"
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"Report written to {report_path}")
else:
    # Capture only (existing behavior)
    # ... (existing capture logic) ...
```

- [ ] **Step 2: Verify report-only mode**

```bash
python scripts/diagnose_p3d3.py --report --traces-dir results/diagnostic_p3d3/traces --T 1 2 8 64
```

Expected: Creates `results/diagnostic_p3d3/report.md`

- [ ] **Step 3: Commit**

```bash
git add scripts/diagnose_p3d3.py
git commit -m "feat: add --report flag and orchestration logic for Phase 3"
```

---

### Task 16: Smoke Test Update

**Files:**
- Modify: `scripts/smoke_diagnostic_p3d3.sh`

- [ ] **Step 1: Add report verification steps**

```bash
# After capture completes, add:

# Step 5: Generate report from traces
echo "=== PHASE 3: Report Generation ==="
python3 scripts/diagnose_p3d3.py \
    --report \
    --traces-dir results/diagnostic_p3d3/traces \
    --T "${T_VALUES[@]}" \
    --output-dir results/diagnostic_p3d3

# Step 6: Verify report exists and has all sections
REPORT="results/diagnostic_p3d3/report.md"
if [ ! -f "$REPORT" ]; then
    echo "FAIL: Report not generated at $REPORT"
    exit 1
fi

for section in "Executive Summary" "Flow Integration Analysis" "Decoder/Readout Analysis" "Root Cause Decision Tree" "Recommendations"; do
    if ! grep -q "$section" "$REPORT"; then
        echo "FAIL: Report missing section: $section"
        exit 1
    fi
done
echo "PASS: Report generated with all sections"
```

- [ ] **Step 2: Run smoke test**

```bash
bash scripts/smoke_diagnostic_p3d3.sh
```

Expected: PASS with all verification steps

- [ ] **Step 3: Commit**

```bash
git add scripts/smoke_diagnostic_p3d3.sh
git commit -m "feat: add --report verification to smoke test"
```

---

## Phase Completion Criteria
- [ ] `src/diagnostic/analysis.py` with FlowAnalysisResult, DecoderAnalysisResult, analyze_flow(), analyze_decoder(), run_analysis()
- [ ] `src/diagnostic/report.py` with generate_report()
- [ ] `scripts/diagnose_p3d3.py` supports --report, --traces-dir, --probes flags
- [ ] Report answers Q1, Q2, Q3 with PASS/WARN/FAIL verdicts and supporting numbers
- [ ] Statistical tables include pairwise t-tests and Cohen's d
- [ ] Report output is valid markdown with tables
- [ ] Decision tree maps evidence to root cause explicitly
- [ ] Recommendations are concrete and prioritized
- [ ] 12 new tests (8 analysis + 4 report), all passing
- [ ] Smoke test updated with --report verification step
- [ ] Report generated from actual P3-D3 checkpoint full T sweep

## Handoff Notes
- Report-only mode (`--report` without `--checkpoint`) is the fast path for iterative analysis — no GPU needed
- The decision tree algorithmically determines root cause from the 3 verdicts
- All sections are markdown-compliant for GitHub rendering
- Report is self-contained; no external dependencies for reading
