# Phase 1: v0.1 Support Closure

**Goal:** Make the training stack ready for the v0.1 matrix with all three architecture families selectable by config, loss-conditional target extraction, wandb integration, and truncation-aware data loading.

**Deliverables:**
- [ ] A1 one-shot projector trainable under the same interface as A3
- [ ] A2 shared recurrent residual trainable under the same interface as A3
- [ ] Loss-conditional teacher target extraction that actually skips computation
- [ ] Trainer wired to use conditional flags based on loss config
- [ ] wandb integration in the trainer with proper run naming
- [ ] Truncation rate logging per dataset component
- [ ] Parity test: verify existing A3 training still works after changes

**Files in scope:**
- Create: `src/model/student_families.py`
- Create: `src/model/student_interface.py`
- Create: `tests/test_loss_conditional_targets.py`
- Create: `tests/test_student_family_selection.py`
- Create: `tests/test_wandb_integration.py`
- Create: `tests/test_truncation_logging.py`
- Modify: `src/model/student_qwen.py`
- Modify: `src/training/trainer.py`
- Modify: `src/data/mixed_corpus.py`

---

## Task 1: Make Teacher Target Extraction Loss-Conditional

### Step 1: Write failing test

Create `tests/test_loss_conditional_targets.py`:

```python
import pytest
import torch
from unittest.mock import patch, MagicMock
from src.model.student_qwen import FrozenQwenStudent


def test_extract_teacher_targets_conditional():
    """Test that extract_teacher_targets can skip unneeded targets."""
    model = FrozenQwenStudent(
        model_name="Qwen/Qwen3.5-0.8B",
        start_layer=8,
        end_layer=11,
        max_steps_T=4,
        device="cpu",
    )
    
    input_ids = torch.randint(0, 1000, (2, 64))
    attention_mask = torch.ones(2, 64)
    
    # When only endpoint needed, teacher_logits should not be returned
    targets = model.extract_teacher_targets(
        input_ids, attention_mask,
        need_teacher_logits=False,
        need_velocity=False,
    )
    assert "teacher_logits" not in targets
    assert "velocity_target" not in targets
    assert "h_start" in targets
    assert "h_target" in targets


def test_teacher_logits_not_computed_when_not_needed():
    """Verify that teacher model forward is NOT called with output_logits when not needed."""
    model = FrozenQwenStudent(
        model_name="Qwen/Qwen3.5-0.8B",
        start_layer=8,
        end_layer=11,
        max_steps_T=4,
        device="cpu",
    )
    
    input_ids = torch.randint(0, 1000, (2, 64))
    attention_mask = torch.ones(2, 64)
    
    # Mock the underlying model forward
    with patch.object(model.model, '__call__') as mock_forward:
        mock_output = MagicMock()
        mock_output.hidden_states = [torch.randn(2, 64, 896) for _ in range(13)]
        mock_output.logits = torch.randn(2, 64, 1000)
        mock_forward.return_value = mock_output
        
        # Call with need_teacher_logits=False
        targets = model.extract_teacher_targets(
            input_ids, attention_mask,
            need_teacher_logits=False,
            need_velocity=True,
        )
        
        # Verify forward was called with output_logits=False
        call_kwargs = mock_forward.call_args[1]
        assert call_kwargs.get('output_logits', True) is False or 'logits' not in targets
```

### Step 2: Run test to verify it fails

Run: `pytest tests/test_loss_conditional_targets.py::test_extract_teacher_targets_conditional -v`

Expected: FAIL with "TypeError: extract_teacher_targets() got an unexpected keyword argument 'need_teacher_logits'"

### Step 3: Implement loss-conditional extraction

Modify `src/model/student_qwen.py`:

Add optional flags to `extract_teacher_targets()`:
- `need_teacher_logits: bool = True`
- `need_velocity: bool = True`
- `need_trajectory_anchors: bool = False` (for v0.1 trajectory loss at layers 8,9,10,11)

Implementation:
- If `need_teacher_logits=False`, call model forward WITHOUT `output_logits=True`
- If `need_velocity=False`, skip computing velocity_target
- If `need_trajectory_anchors=True`, extract h8,h9,h10,h11 from hidden_states[8:12]

### Step 4: Run test to verify it passes

Run: `pytest tests/test_loss_conditional_targets.py -v`

Expected: PASS

### Step 5: Commit

```bash
git add tests/test_loss_conditional_targets.py src/model/student_qwen.py
git commit -m "feat: make teacher target extraction loss-conditional"
```

---

## Task 2: Implement A1 One-Shot Projector Trainable Family

### Step 1: Write failing test

Create `tests/test_a1_projector.py`:

```python
import pytest
import torch
from src.model.student_families import OneShotProjector


def test_one_shot_projector_forward():
    """Test A1 one-shot projector produces valid hidden states."""
    projector = OneShotProjector(
        hidden_size=896,
        mlp_ratio=4.0,
    )
    
    h_start = torch.randn(2, 64, 896)
    h_end_hat = projector(h_start)
    
    assert h_end_hat.shape == h_start.shape
    # A1 is residual MLP: h_start + g_theta(h_start)
    # Output should be different from input
    assert not torch.allclose(h_end_hat, h_start)


def test_one_shot_projector_is_residual_mlp():
    """Verify A1 implements residual projection per v0.1 spec."""
    from torch import nn
    
    projector = OneShotProjector(hidden_size=896)
    
    # Should have simple MLP layers, not transformer blocks
    has_transformer = any(
        isinstance(m, (nn.MultiheadAttention, nn.TransformerEncoderLayer))
        for m in projector.modules()
    )
    assert not has_transformer, "A1 should be simple MLP, not transformer"
```

### Step 2: Run test to verify it fails

Run: `pytest tests/test_a1_projector.py -v`

Expected: FAIL with "ModuleNotFoundError: No module named 'src.model.student_families'"

### Step 3: Create student families module

Create `src/model/student_families.py`:

Implement `OneShotProjector` class:
- Per v0.1 spec: `h_end_hat = h_start + g_theta(h_start)` where g_theta is a simple MLP
- Two-layer MLP with residual connection around it
- Input: hidden states [batch, seq, hidden]
- Output: transformed hidden states [batch, seq, hidden]
- No attention, no recurrence, no step conditioning

### Step 4: Run test to verify it passes

Run: `pytest tests/test_a1_projector.py -v`

Expected: PASS

### Step 5: Commit

```bash
git add src/model/student_families.py tests/test_a1_projector.py
git commit -m "feat: add A1 one-shot projector trainable family"
```

---

## Task 3: Implement A2 Shared Recurrent Residual Trainable Family

### Step 1: Write failing test

Create `tests/test_a2_recurrent.py`:

```python
import pytest
import torch
from src.model.student_families import SharedRecurrentResidual


def test_shared_recurrent_variable_t():
    """Test A2 shared recurrent block works with different T values."""
    model = SharedRecurrentResidual(
        hidden_size=896,
        max_steps_T=8,
        mlp_ratio=4.0,
    )
    
    h_start = torch.randn(2, 64, 896)
    
    # Test T=1
    h_t1 = model(h_start, num_steps=1)
    assert h_t1.shape == h_start.shape
    
    # Test T=4
    h_t4 = model(h_start, num_steps=4)
    assert h_t4.shape == h_start.shape
    
    # With more steps, output should be different
    assert not torch.allclose(h_t1, h_t4)


def test_shared_recurrent_uses_same_block():
    """Verify A2 reuses same parameters across steps."""
    from torch import nn
    
    model = SharedRecurrentResidual(hidden_size=896)
    
    # Should have single transformer block, not T separate blocks
    transformer_blocks = [
        m for m in model.modules()
        if isinstance(m, (nn.TransformerEncoderLayer, nn.MultiheadAttention))
    ]
    # One or few blocks, not T separate blocks
    assert len(transformer_blocks) <= 2
```

### Step 2: Run test to verify it fails

Run: `pytest tests/test_a2_recurrent.py -v`

Expected: FAIL with "ImportError: cannot import name 'SharedRecurrentResidual'"

### Step 3: Add SharedRecurrentResidual to student_families

Extend `src/model/student_families.py`:

Implement `SharedRecurrentResidual`:
- Single shared refiner block (attention + MLP)
- Applied T times iteratively
- No step conditioning (tests whether simple iteration is enough)
- Variable num_steps support
- Input: (h_start, num_steps)
- Output: refined hidden states

### Step 4: Run test to verify it passes

Run: `pytest tests/test_a2_recurrent.py -v`

Expected: PASS

### Step 5: Commit

```bash
git add src/model/student_families.py tests/test_a2_recurrent.py
git commit -m "feat: add A2 shared recurrent residual trainable family"
```

---

## Task 4: Create Common Student Family Interface

### Step 1: Write failing test

Create `tests/test_student_interface.py`:

```python
import pytest
import torch
from src.model.student_interface import StudentFamilyInterface
from src.model.student_families import OneShotProjector, SharedRecurrentResidual


def test_a1_interface_compliance():
    """Test A1 conforms to student family interface."""
    family = OneShotProjector(hidden_size=896)
    interface = StudentFamilyInterface(family, family_type="one_shot_projector")
    
    h_start = torch.randn(2, 64, 896)
    
    # Should support unified forward
    h_end = interface.forward_refinement(h_start, num_steps=1)
    assert h_end.shape == h_start.shape


def test_a2_interface_compliance():
    """Test A2 conforms to student family interface."""
    family = SharedRecurrentResidual(hidden_size=896, max_steps_T=8)
    interface = StudentFamilyInterface(family, family_type="shared_recurrent")
    
    h_start = torch.randn(2, 64, 896)
    
    # Should support unified forward with variable T
    h_end = interface.forward_refinement(h_start, num_steps=4)
    assert h_end.shape == h_start.shape


def test_interface_returns_dict_for_trajectory():
    """Test interface returns dict with endpoint_hidden and optionally trajectory_hidden."""
    family = SharedRecurrentResidual(hidden_size=896, max_steps_T=8)
    interface = StudentFamilyInterface(family, family_type="shared_recurrent")
    
    h_start = torch.randn(2, 64, 896)
    
    # Without trajectory
    result = interface.forward_refinement(h_start, num_steps=4, return_trajectory=False)
    assert isinstance(result, dict)
    assert "endpoint_hidden" in result
    assert "trajectory_hidden" not in result
    
    # With trajectory
    result = interface.forward_refinement(h_start, num_steps=4, return_trajectory=True)
    assert "endpoint_hidden" in result
    assert "trajectory_hidden" in result
```

### Step 2: Run test to verify it fails

Run: `pytest tests/test_student_interface.py -v`

Expected: FAIL with "ModuleNotFoundError: No module named 'src.model.student_interface'"

### Step 3: Create student interface module

Create `src/model/student_interface.py`:

Implement `StudentFamilyInterface`:
- Wraps A1, A2, A3 families with unified API
- `forward_refinement(h_start, num_steps, attention_mask=None, return_trajectory=False)` → dict
- `forward_refinement_with_velocity(...)` for flow families
- Family-specific dispatch internally
- Maintains same return format as current FlowMidblock path

This decouples `FrozenQwenStudent.forward()` from family-specific logic.

### Step 4: Run test to verify it passes

Run: `pytest tests/test_student_interface.py -v`

Expected: PASS

### Step 5: Commit

```bash
git add src/model/student_interface.py tests/test_student_interface.py
git commit -m "feat: add unified student family interface"
```

---

## Task 5: Integrate Student Families into FrozenQwenStudent

### Step 1: Write failing test

Create `tests/test_student_family_selection.py`:

```python
import pytest
import torch
from src.model.student_qwen import FrozenQwenStudent


def test_create_a1_projector_student():
    """Test creating student with A1 family via config."""
    student = FrozenQwenStudent(
        model_name="Qwen/Qwen3.5-0.8B",
        start_layer=8,
        end_layer=11,
        max_steps_T=8,
        device="cpu",
        family="one_shot_projector",
    )
    
    assert student.family == "one_shot_projector"
    assert student.family_interface is not None
    
    # Should be able to forward pass
    input_ids = torch.randint(0, 1000, (2, 32))
    outputs = student(input_ids, num_steps=1, return_dict=True)
    assert isinstance(outputs, dict)
    assert "logits" in outputs
    assert outputs["logits"].shape[0] == 2


def test_create_a2_recurrent_student():
    """Test creating student with A2 family."""
    student = FrozenQwenStudent(
        model_name="Qwen/Qwen3.5-0.8B",
        start_layer=8,
        end_layer=11,
        max_steps_T=8,
        device="cpu",
        family="shared_recurrent_residual",
    )
    
    assert student.family == "shared_recurrent_residual"
    
    input_ids = torch.randint(0, 1000, (2, 32))
    outputs = student(input_ids, num_steps=4, return_dict=True)
    assert "logits" in outputs
```

### Step 2: Run test to verify it fails

Run: `pytest tests/test_student_family_selection.py::test_create_a1_projector_student -v`

Expected: FAIL with "TypeError: FrozenQwenStudent.__init__() got an unexpected keyword argument 'family'"

### Step 3: Add family selection to FrozenQwenStudent

Modify `src/model/student_qwen.py`:

Add `family` parameter to `__init__()`:
- `family: str = "flow_midblock"` (default for backward compatibility)
- Support "one_shot_projector", "shared_recurrent_residual", "flow_midblock"
- Delegate to `StudentFamilyInterface` for family-specific logic
- Refactor `forward()` to use interface instead of hardcoded ODE path

Key changes:
```python
def forward(self, ...):
    h_start = self._extract_h_start(...)
    
    # Use family interface instead of hardcoded midblock/ODE
    refinement_result = self.family_interface.forward_refinement(
        h_start, num_steps, attention_mask, return_dict
    )
    h_mid = refinement_result["endpoint_hidden"]
    
    # Continue through upper layers
    logits = self._continue_from_hidden_state(h_mid, attention_mask)
    return logits
```

### Step 4: Run test to verify it passes

Run: `pytest tests/test_student_family_selection.py -v`

Expected: PASS

### Step 5: Commit

```bash
git add src/model/student_qwen.py tests/test_student_family_selection.py
git commit -m "feat: add family selection to FrozenQwenStudent with unified interface"
```

---

## Task 6: Wire Loss-Conditional Extraction into Trainer

### Step 1: Write failing test

Create `tests/test_trainer_conditional_targets.py`:

```python
import pytest
from unittest.mock import MagicMock, patch
from src.training.trainer import Trainer


def test_trainer_passes_conditional_flags_based_on_loss_config():
    """Test that trainer passes correct flags to extract_teacher_targets based on loss weights."""
    
    config = {
        "model": {"name": "Qwen/Qwen3.5-0.8B", "max_steps_T": 4},
        "replacement_model": {"start_layer": 8, "end_layer": 11},
        "loss": {
            "endpoint_weight": 1.0,
            "trajectory_weight": 0.0,  # No trajectory
            "kl_weight": 0.0,  # No KL
            "ce_weight": 0.0,
            "velocity_weight": 0.0,
        },
        "train_loop": {},
        "optimizer": {},
        "scheduler": {},
    }
    
    mock_model = MagicMock()
    mock_model.extract_teacher_targets.return_value = {
        "h_start": MagicMock(),
        "h_target": MagicMock(),
    }
    
    trainer = Trainer(
        model=mock_model,
        loss_fn=MagicMock(),
        config=config,
        device="cpu",
    )
    
    batch = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
    
    # Patch train_step to just test extraction
    with patch.object(trainer, '_get_loss_flags') as mock_get_flags:
        mock_get_flags.return_value = {
            "need_teacher_logits": False,
            "need_velocity": False,
            "need_trajectory_anchors": False,
        }
        
        trainer.train_step(batch, T=4)
        
        # Verify extraction was called with correct flags
        mock_model.extract_teacher_targets.assert_called_once()
        call_kwargs = mock_model.extract_teacher_targets.call_args[1]
        assert call_kwargs.get('need_teacher_logits') is False
        assert call_kwargs.get('need_velocity') is False
```

### Step 2: Run test to verify it fails

Run: `pytest tests/test_trainer_conditional_targets.py -v`

Expected: FAIL with various errors since trainer doesn't yet pass these flags

### Step 3: Update trainer to use conditional extraction

Modify `src/training/trainer.py`:

Add `_get_loss_flags()` method:
- Reads loss weights from config
- Returns dict with `need_teacher_logits`, `need_velocity`, `need_trajectory_anchors`
- `need_teacher_logits = kl_weight > 0`
- `need_velocity = velocity_weight > 0`
- `need_trajectory_anchors = trajectory_weight > 0`

Update `train_step()`:
- Call `_get_loss_flags()` at start
- Pass flags to `extract_teacher_targets()`

Update `val_step()`:
- Same conditional logic

### Step 4: Run test to verify it passes

Run: `pytest tests/test_trainer_conditional_targets.py -v`

Expected: PASS

### Step 5: Commit

```bash
git add src/training/trainer.py tests/test_trainer_conditional_targets.py
git commit -m "feat: wire loss-conditional extraction into trainer"
```

---

## Task 7: Add wandb Integration to Trainer

### Step 1: Write failing test

Create `tests/test_wandb_integration.py`:

```python
import pytest
from unittest.mock import MagicMock, patch
from src.training.trainer import Trainer


@patch('src.training.trainer.wandb')
def test_wandb_init_with_config(mock_wandb):
    """Test that wandb is initialized with config."""
    mock_run = MagicMock()
    mock_wandb.init.return_value = mock_run
    
    config = {
        "experiment_name": "test_exp",
        "seed": 1337,
        "logging": {
            "wandb": {
                "enabled": True,
                "project": "midflowlm",
                "entity": "myteam",
                "tags": ["v0.1", "test"],
            }
        },
        "model": {"name": "Qwen/Qwen3.5-0.8B", "max_steps_T": 4},
        "replacement_model": {"start_layer": 8, "end_layer": 11},
        "loss": {},
        "train_loop": {},
        "optimizer": {},
        "scheduler": {},
    }
    
    mock_model = MagicMock()
    mock_model.extract_teacher_targets.return_value = {
        "h_start": MagicMock(),
        "h_target": MagicMock(),
    }
    
    trainer = Trainer(
        model=mock_model,
        loss_fn=MagicMock(),
        config=config,
        device="cpu",
    )
    
    assert trainer.use_wandb is True
    mock_wandb.init.assert_called_once()
    
    # Verify init args
    call_kwargs = mock_wandb.init.call_args[1]
    assert call_kwargs['project'] == "midflowlm"
    assert call_kwargs['entity'] == "myteam"
    assert call_kwargs['tags'] == ["v0.1", "test"]


@patch('src.training.trainer.wandb')
def test_wandb_log_metrics(mock_wandb):
    """Test that metrics are logged to wandb."""
    mock_wandb.init.return_value = MagicMock()
    
    config = {
        "experiment_name": "test",
        "logging": {"wandb": {"enabled": True, "project": "test"}},
        "model": {"name": "Qwen/Qwen3.5-0.8B", "max_steps_T": 4},
        "replacement_model": {"start_layer": 8, "end_layer": 11},
        "loss": {},
        "train_loop": {},
        "optimizer": {},
        "scheduler": {},
    }
    
    trainer = Trainer(
        model=MagicMock(),
        loss_fn=MagicMock(),
        config=config,
        device="cpu",
    )
    
    # Simulate logging
    trainer._log_to_wandb({"train/loss": 0.5, "train/kl": 0.1}, step=100)
    
    mock_wandb.log.assert_called_once()
    call_args = mock_wandb.log.call_args[0][0]
    assert call_args["train/loss"] == 0.5
    assert call_args["train/kl"] == 0.1
    assert call_args["step"] == 100
```

### Step 2: Run test to verify it fails

Run: `pytest tests/test_wandb_integration.py::test_wandb_init_with_config -v`

Expected: FAIL with "AttributeError: 'Trainer' object has no attribute 'use_wandb'"

### Step 3: Add wandb support to Trainer

Modify `src/training/trainer.py`:

Add wandb configuration:
- Read from `config.get("logging", {}).get("wandb", {})`
- Support `enabled`, `project`, `entity`, `tags`, `name`
- Initialize wandb in `__init__` if enabled
- Add `_log_to_wandb(metrics, step)` method
- Call `_log_to_wandb()` alongside `_log_to_tensorboard()` in train/val steps
- Handle wandb not installed gracefully (warning, not crash)

### Step 4: Run test to verify it passes

Run: `pytest tests/test_wandb_integration.py -v`

Expected: PASS

### Step 5: Commit

```bash
git add src/training/trainer.py tests/test_wandb_integration.py
git commit -m "feat: add wandb integration to Trainer"
```

---

## Task 8: Add Truncation Logging

### Step 1: Write failing test

Create `tests/test_truncation_logging.py`:

```python
import pytest
from unittest.mock import MagicMock, patch
from src.data.mixed_corpus import build_mixture_split_with_stats, get_truncation_stats


def test_build_mixture_split_with_stats_returns_dataset_and_stats():
    """Test that truncation statistics are collected during dataset building."""
    
    config = MagicMock()
    config.data.seq_len = 1024
    config.data.mixture_components = [
        {
            "name": "test_dataset",
            "dataset_name": "test/dataset",
            "train_split": "train",
            "val_split": "validation",
            "train_samples": 100,
            "val_samples": 10,
        }
    ]
    config.data.shuffle_seed = 42
    
    tokenizer = MagicMock()
    tokenizer.__call__ = MagicMock(return_value={
        "input_ids": [[1, 2, 3]],
        "attention_mask": [[1, 1, 1]],
    })
    
    with patch('src.data.mixed_corpus.load_dataset') as mock_load:
        mock_dataset = MagicMock()
        mock_dataset.shuffle.return_value = mock_dataset
        mock_dataset.select.return_value = mock_dataset
        mock_dataset.map.return_value = mock_dataset
        mock_dataset.remove_columns.return_value = mock_dataset
        mock_dataset.__len__ = MagicMock(return_value=100)
        mock_load.return_value = mock_dataset
        
        dataset, stats = build_mixture_split_with_stats(
            config=config,
            split="train",
            tokenizer=tokenizer,
        )
    
    assert dataset is not None
    assert isinstance(stats, dict)
    assert "by_component" in stats
    assert "test_dataset" in stats["by_component"]
    component_stats = stats["by_component"]["test_dataset"]
    assert "truncation_rate" in component_stats
    assert "mean_original_length" in component_stats


def test_get_truncation_stats_returns_per_component_stats():
    """Test get_truncation_stats produces summary metrics."""
    stats = {
        "by_component": {
            "fineweb": {"truncation_rate": 0.05, "mean_original_length": 1500},
            "ultrachat": {"truncation_rate": 0.12, "mean_original_length": 1800},
        }
    }
    
    summary = get_truncation_stats(stats)
    
    assert "overall_truncation_rate" in summary
    assert "per_component" in summary
    assert summary["by_component"]["fineweb"]["truncation_rate"] == 0.05
```

### Step 2: Run test to verify it fails

Run: `pytest tests/test_truncation_logging.py::test_build_mixture_split_with_stats_returns_dataset_and_stats -v`

Expected: FAIL with "ImportError: cannot import name 'build_mixture_split_with_stats'"

### Step 3: Add truncation logging to mixed corpus

Modify `src/data/mixed_corpus.py`:

Create `build_mixture_split_with_stats()`:
- Call existing `build_mixture_split()` for dataset construction
- Track per-component statistics:
  - Total sequences
  - Truncated sequences
  - Mean token count before truncation
- Return `(dataset, stats_dict)` tuple

Create `get_truncation_stats()`:
- Take stats dict
- Return formatted summary for logging
- Compute overall truncation rate across components

Preserve backward compatibility: keep existing `build_mixture_split()` unchanged.

### Step 4: Run test to verify it passes

Run: `pytest tests/test_truncation_logging.py -v`

Expected: PASS

### Step 5: Commit

```bash
git add src/data/mixed_corpus.py tests/test_truncation_logging.py
git commit -m "feat: add truncation rate logging per dataset component"
```

---

## Phase Completion Gate

After all tasks pass:
- [ ] All tests in `tests/` pass
- [ ] No regressions in existing tests (run `pytest tests/`)
- [ ] Smoke test: verify existing A3 training still works: `python scripts/train.py --config configs/v0_smoke_run.yaml --fast-dev-run`
- [ ] A1 and A2 forward passes work with test configs
- [ ] Commit message captures phase deliverables

**Next:** Write Phase 2 plan (hardware calibration)
