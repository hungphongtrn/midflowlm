# Phase 1: Autograd Fix

**Goal:** Restore CE/KL gradient flow through frozen Qwen upper stack without updating frozen weights.

**Deliverables:**
- [ ] `torch.no_grad()` removed from `_continue_from_hidden_state`
- [ ] Smoke test proving CE/KL gradients reach midblock parameters
- [ ] Verification that frozen Qwen parameters receive no gradients after backward
- [ ] One batch forward/backward/optimizer step runs successfully

**Files in scope:**
- Modify: `src/model/student_qwen.py:432` — Remove `with torch.no_grad():` from `_continue_from_hidden_state`
- Modify: `tests/test_student_qwen.py` — Add `TestGradientFlow` class

---

## Task 1: Write Failing Test for Gradient Flow

- [ ] **Step 1: Write test that verifies CE/KL gradients reach midblock**

```python
# tests/test_student_qwen.py

class TestGradientFlow:
    """Test that CE/KL gradients flow through frozen Qwen to midblock."""

    def test_ce_loss_gradients_reach_midblock(self, model_config, device):
        """Test that CE loss gradients flow to midblock parameters."""
        from src.model.student_qwen import FrozenQwenStudent
        import torch.nn.functional as F

        student = FrozenQwenStudent(
            model_name=model_config["model_name"],
            start_layer=model_config["start_layer"],
            end_layer=model_config["end_layer"],
            max_steps_T=model_config["max_steps_T"],
            device=device,
        )

        # Create input
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        attention_mask = torch.ones(batch_size, seq_len, device=device)

        # Forward pass
        logits = student(input_ids, attention_mask, num_steps=4)
        
        # Create CE loss target
        labels = input_ids[:, 1:]
        pred_logits = logits[:, :-1, :]
        
        # Compute CE loss
        ce_loss = F.cross_entropy(
            pred_logits.reshape(-1, pred_logits.size(-1)),
            labels.reshape(-1)
        )

        # Backward
        ce_loss.backward()

        # Verify midblock has gradients
        midblock_has_grad = False
        for name, param in student.named_parameters():
            if "midblock" in name and param.grad is not None:
                if param.grad.abs().sum() > 0:
                    midblock_has_grad = True
                    break

        assert midblock_has_grad, "Midblock should have non-zero gradients after CE loss backward"

    def test_frozen_qwen_has_no_gradients(self, model_config, device):
        """Test that frozen Qwen parameters have no gradients after backward."""
        from src.model.student_qwen import FrozenQwenStudent
        import torch.nn.functional as F

        student = FrozenQwenStudent(
            model_name=model_config["model_name"],
            start_layer=model_config["start_layer"],
            end_layer=model_config["end_layer"],
            max_steps_T=model_config["max_steps_T"],
            device=device,
        )

        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        attention_mask = torch.ones(batch_size, seq_len, device=device)

        logits = student(input_ids, attention_mask, num_steps=4)
        labels = input_ids[:, 1:]
        pred_logits = logits[:, :-1, :]
        
        ce_loss = F.cross_entropy(
            pred_logits.reshape(-1, pred_logits.size(-1)),
            labels.reshape(-1)
        )
        ce_loss.backward()

        # Verify frozen Qwen parameters have no gradients
        for name, param in student.named_parameters():
            if "model." in name and "midblock" not in name:
                assert param.grad is None or param.grad.abs().sum() == 0, (
                    f"Frozen Qwen parameter should have no gradient: {name}"
                )

    def test_kl_loss_gradients_reach_midblock(self, model_config, device):
        """Test that KL divergence gradients flow to midblock parameters."""
        from src.model.student_qwen import FrozenQwenStudent
        import torch.nn.functional as F

        student = FrozenQwenStudent(
            model_name=model_config["model_name"],
            start_layer=model_config["start_layer"],
            end_layer=model_config["end_layer"],
            max_steps_T=model_config["max_steps_T"],
            device=device,
        )

        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        attention_mask = torch.ones(batch_size, seq_len, device=device)

        # Get teacher logits (no_grad)
        teacher_targets = student.extract_teacher_targets(input_ids, attention_mask)
        teacher_logits = teacher_targets["teacher_logits"]

        # Forward pass (with_grad, since we removed no_grad)
        student_logits = student(input_ids, attention_mask, num_steps=4)

        # KL divergence loss
        kl_loss = F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            F.softmax(teacher_logits, dim=-1),
            reduction="batchmean"
        )

        kl_loss.backward()

        # Verify midblock has gradients
        midblock_has_grad = False
        for name, param in student.named_parameters():
            if "midblock" in name and param.grad is not None:
                if param.grad.abs().sum() > 0:
                    midblock_has_grad = True
                    break

        assert midblock_has_grad, "Midblock should have non-zero gradients after KL loss backward"

    def test_optimizer_step_updates_only_midblock(self, model_config, device):
        """Test that optimizer step only changes midblock parameters."""
        from src.model.student_qwen import FrozenQwenStudent
        import torch.nn.functional as F

        student = FrozenQwenStudent(
            model_name=model_config["model_name"],
            start_layer=model_config["start_layer"],
            end_layer=model_config["end_layer"],
            max_steps_T=model_config["max_steps_T"],
            device=device,
        )

        # Store initial parameter values
        midblock_params_before = {}
        qwen_params_before = {}
        for name, param in student.named_parameters():
            if "midblock" in name:
                midblock_params_before[name] = param.data.clone()
            elif "model." in name:
                qwen_params_before[name] = param.data.clone()

        # Forward + backward + optimizer step
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        attention_mask = torch.ones(batch_size, seq_len, device=device)

        logits = student(input_ids, attention_mask, num_steps=4)
        labels = input_ids[:, 1:]
        pred_logits = logits[:, :-1, :]
        
        ce_loss = F.cross_entropy(
            pred_logits.reshape(-1, pred_logits.size(-1)),
            labels.reshape(-1)
        )

        optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)
        optimizer.zero_grad()
        ce_loss.backward()
        optimizer.step()

        # Verify Qwen parameters unchanged
        for name, param in student.named_parameters():
            if "model." in name and "midblock" not in name:
                if name in qwen_params_before:
                    assert torch.allclose(param.data, qwen_params_before[name], atol=1e-7), (
                        f"Frozen Qwen parameter changed after optimizer step: {name}"
                    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_student_qwen.py::TestGradientFlow -v`
Expected: Tests FAIL because `torch.no_grad()` prevents gradient flow

---

## Task 2: Remove torch.no_grad() from _continue_from_hidden_state

- [ ] **Step 1: Remove the context manager from `_continue_from_hidden_state`**

In `src/model/student_qwen.py`, find `_continue_from_hidden_state` (around line 412-492) and remove the `with torch.no_grad():` wrapper while keeping the inner code.

Before:
```python
def _continue_from_hidden_state(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    base_model = self._get_base_model()

    with torch.no_grad():  # <-- REMOVE THIS LINE
        # Get the layers module
        ...
        return logits
```

After:
```python
def _continue_from_hidden_state(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    base_model = self._get_base_model()

    # Get the layers module
    ...
    return logits
```

Remove one level of indentation from all code inside the removed `with torch.no_grad():` block.

- [ ] **Step 2: Run tests to verify gradient tests pass**

Run: `pytest tests/test_student_qwen.py::TestGradientFlow -v`
Expected: Tests PASS

---

## Task 3: Run Full Test Suite

- [ ] **Step 1: Run all student_qwen tests**

Run: `pytest tests/test_student_qwen.py -v`
Expected: All tests PASS

- [ ] **Step 2: Verify parameter counts still correct**

Run: `pytest tests/test_student_qwen.py::TestTrainableParameters -v`
Expected: Tests PASS — frozen params still frozen, midblock still trainable

---

## Task 4: Memory Impact Check

- [ ] **Step 1: Run one training batch and record GPU memory**

Create a quick script `scripts/memory_check.py`:

```python
#!/usr/bin/env python3
"""Quick memory check for gradient flow fix."""

import torch
from src.model.student_qwen import FrozenQwenStudent
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

if device == "cpu":
    print("CPU mode - memory measurement not applicable")
    exit(0)

student = FrozenQwenStudent(
    model_name="Qwen/Qwen3.5-0.8B",
    start_layer=8,
    end_layer=11,
    max_steps_T=4,
    device=device,
    dtype=torch.bfloat16,
)

torch.cuda.reset_peak_memory_stats()

# Forward + backward
input_ids = torch.randint(0, 1000, (2, 64), device=device)
attention_mask = torch.ones(2, 64, device=device)

logits = student(input_ids, attention_mask, num_steps=4)
labels = input_ids[:, 1:]
pred_logits = logits[:, :-1, :]
ce_loss = F.cross_entropy(
    pred_logits.reshape(-1, pred_logits.size(-1)),
    labels.reshape(-1)
)
ce_loss.backward()

peak_memory = torch.cuda.max_memory_allocated() / 1e9
print(f"Peak GPU memory: {peak_memory:.2f} GB")

# Compare with baseline (document what baseline should be)
# Note: Memory will increase because we now track gradients through frozen layers
```

Run: `python scripts/memory_check.py`
Expected: Record peak memory, compare to baseline if available

---

## Phase Completion Gate

After all tasks pass:
- [ ] All tests in `tests/test_student_qwen.py` pass
- [ ] No regressions in existing tests
- [ ] Memory impact documented in memory_notes.md
- [ ] Commit message: `fix: remove torch.no_grad from upper stack to restore CE/KL gradient flow`

---

## Notes

**Why this fix works:**
- `requires_grad=False` on parameters prevents optimizer updates
- Removing `torch.no_grad()` allows autograd to build the computation graph
- Gradients flow from loss → logits → upper frozen layers → midblock
- Frozen layers don't accumulate gradients (they have `requires_grad=False`), so `.grad` stays `None` or zero

**Potential memory impact:**
- Enabling gradient tracking through frozen layers increases activation memory
- Upper layers (end_layer+1 to num_layers) now store activations for backprop
- Expected increase: proportional to hidden_size × seq_len × num_upper_layers