# Continuous ODE Flow Midblock Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor MidflowLM's discrete iterative midblock into a continuous-time velocity field trained with flow-matching targets so inference can safely scale `num_steps` beyond the training schedule.

**Architecture:** Replace integer step conditioning with continuous `t in [0, 1]` embeddings, make the midblock predict `v_theta(h_t, t)` instead of absolute deltas, and integrate that vector field with `torchdiffeq.odeint`. Update teacher-cache generation, loss computation, trainer sampling, and text-sweep evaluation so the whole pipeline is consistently velocity-based and ODE-driven.

**Tech Stack:** PyTorch, Hugging Face Transformers, torchdiffeq, PyYAML, safetensors, vendored `external/minFM` references

---

## Scope and sequencing notes

- This is one coherent plan, not multiple independent subsystems: architecture, cache schema, training loss, and inference all have to change together to avoid a half-discrete / half-continuous pipeline.
- Reuse design patterns from `external/minFM/utils_fm/sampler.py` and `external/minFM/models/latent_fm.py` for velocity-model wrapping and time-driven sampling behavior; do not copy image-specific packing, noiser, or CFG code into MidflowLM.
- Keep the frozen Qwen boundary extraction path intact. The refactor should only change the replacement span behavior and downstream training / eval plumbing.
- Preserve checkpoint save/load and frozen/trainable parameter reporting throughout the refactor.

## File map

**Modify**
- `requirements.txt` - add `torchdiffeq`
- `configs/v0_onemotif.yaml` - replace discrete-step config with continuous-time / ODE config
- `src/model/adapter.py` - replace `StepConditioningAdapter` with continuous-time embedding utilities
- `src/model/midblock.py` - replace `IterativeMidblock` behavior with `FlowMidblock` velocity prediction API
- `src/model/student_qwen.py` - integrate ODE solve path into the student forward pass
- `src/model/__init__.py` - export new flow/ODE types
- `src/data/teacher_cache.py` - store velocity targets and new metadata fields
- `src/training/data.py` - load velocity targets instead of endpoint-only trajectory tensors where applicable
- `src/training/losses.py` - compute continuous-time velocity MSE
- `src/training/trainer.py` - sample `t ~ U(0, 1)` and pass continuous-time training inputs
- `scripts/build_teacher_cache.py` - compute and write `h11 - h7` velocity targets
- `scripts/train_v0.py` - wire new config and smoke commands
- `src/eval/text_checkpoint_sweep.py` - use ODE solver controls and compute repetition metrics
- `scripts/run_checkpoint_text_sweep.py` - expose solver options and step counts
- `src/eval/__init__.py` - export updated sweep helpers if signatures change
- `tests/test_midblock.py` - rewrite around continuous-time embedding and velocity prediction
- `tests/test_student_qwen.py` - rewrite around ODE-backed inference behavior
- `tests/test_teacher_cache.py` - rewrite for velocity-target cache contract
- `tests/test_losses.py` - rewrite for velocity MSE contract
- `tests/test_train_smoke.py` - update mock batches and trainer expectations
- `tests/test_eval_pipeline.py` - add ODE sweep / repetition metric coverage
- `tests/test_architecture_loss_contract.py` - keep the no-logits architecture contract, but assert velocity-target supervision instead of endpoint hidden matching

**Create**
- `src/model/ode.py` - `MidblockVectorField`, ODE time-grid helpers, and solver option normalization
- `tests/test_ode_integration.py` - focused tests for solver wiring and `dt` normalization
- `tests/test_continuous_time_embedding.py` - focused tests for `ContinuousTimeEmbedding`

**Do not modify unless the implementation proves it is necessary**
- `src/model/qwen_parity.py` - boundary extraction should remain the source of truth
- `src/eval/baselines.py` - only touch if metric helper reuse is cleaner than duplicating sweep logic

---

## Fixed design decisions

1. Integer `step_id` is removed from model-facing APIs; only continuous `t` values are allowed in the refactored path.
2. The training target is a velocity field based on teacher boundaries: `v_target = h_end - h_start`, where `h_start` is teacher layer 7 output and `h_end` is teacher layer 11 output for the default span.
3. The sampled training state is constructed explicitly as straight-line interpolation on the teacher path: `h_t = h_start + t * v_target`. Do not train only at `h_start`.
4. Inference time is always normalized to `[0.0, 1.0]`, regardless of `num_steps`.
5. Euler integration must use `step_size = 1.0 / num_steps`; higher-order solvers may ignore `step_size` but must still integrate over the same `[0, 1]` interval.
6. `torchdiffeq.odeint` is the standard solver entrypoint; do not keep the old manual refinement loop in the active inference path.
7. Architecture training remains hidden-state-only by default; logits stay optional and disabled in the default config.
8. The refactor may keep short-term compatibility aliases such as `IterativeMidblock = FlowMidblock` only if needed to reduce churn, but all new tests and configs should use the continuous names.

---

### Task 1: Add the continuous-time config surface and dependency

**Files:**
- Modify: `requirements.txt`
- Modify: `configs/v0_onemotif.yaml`
- Test: `tests/test_architecture_loss_contract.py`

- [ ] **Step 1: Write the failing config contract test**

```python
def test_architecture_config_uses_continuous_time_defaults():
    import yaml
    cfg = yaml.safe_load(open("configs/v0_onemotif.yaml"))
    assert cfg["model"]["time_domain"] == [0.0, 1.0]
    assert cfg["model"]["ode_solver"]["method"] == "euler"
    assert "step_embedding" not in cfg["model"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_architecture_loss_contract.py -k continuous_time_defaults -v`
Expected: FAIL because the config still exposes discrete-step fields.

- [ ] **Step 3: Add `torchdiffeq` and rewrite the config fields**

```yaml
model:
  max_steps_T: 8
  time_domain: [0.0, 1.0]
  time_embedding: "sinusoidal"
  ode_solver:
    method: "euler"
    atol: 1.0e-5
    rtol: 1.0e-5
replacement_model:
  predict_velocity: true
  use_step_conditioning: false
  use_time_conditioning: true
train_loop:
  sample_continuous_time: true
```

- [ ] **Step 4: Run the targeted contract test**

Run: `python -m pytest tests/test_architecture_loss_contract.py -k continuous_time_defaults -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt configs/v0_onemotif.yaml tests/test_architecture_loss_contract.py
git commit -m "chore: add continuous-time ODE config surface"
```

### Task 2: Replace discrete step conditioning with continuous time embeddings

**Files:**
- Modify: `src/model/adapter.py`
- Create: `tests/test_continuous_time_embedding.py`
- Modify: `tests/test_midblock.py`

- [ ] **Step 1: Write the failing embedding tests**

```python
def test_continuous_time_embedding_accepts_fractional_t():
    emb = ContinuousTimeEmbedding(hidden_size=64)
    t = torch.tensor([0.0, 0.125, 0.5, 1.0])
    out = emb(t)
    assert out.shape == (4, 64)

def test_continuous_time_embedding_rejects_integer_step_api():
    emb = ContinuousTimeEmbedding(hidden_size=64)
    with pytest.raises(TypeError):
        emb(step_id=3)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_continuous_time_embedding.py tests/test_midblock.py -k "continuous_time or step_conditioning" -v`
Expected: FAIL because `ContinuousTimeEmbedding` does not exist and the current tests still assume `step_id`.

- [ ] **Step 3: Implement the embedding module and migrate the midblock-facing API**

```python
class ContinuousTimeEmbedding(nn.Module):
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t shape: [batch] or scalar tensor, values in [0, 1]
        ...
```

- [ ] **Step 4: Rewrite midblock tests around continuous `t`**

```python
velocity = midblock.get_velocity(
    h_t=sample_hidden_states,
    h_start=sample_hidden_states,
    attention_mask=sample_attention_mask,
    t=torch.full((batch_size,), 0.25, device=device),
)
assert velocity.shape == sample_hidden_states.shape
```

- [ ] **Step 5: Run the tests to pass**

Run: `python -m pytest tests/test_continuous_time_embedding.py tests/test_midblock.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/model/adapter.py tests/test_continuous_time_embedding.py tests/test_midblock.py
git commit -m "feat: replace discrete step embeddings with continuous time embeddings"
```

### Task 3: Refactor the midblock into a velocity predictor

**Files:**
- Modify: `src/model/midblock.py`
- Modify: `src/model/__init__.py`
- Modify: `tests/test_midblock.py`

- [ ] **Step 1: Write the failing velocity API tests**

```python
def test_flow_midblock_get_velocity_matches_hidden_shape(sample_hidden_states):
    block = FlowMidblock(hidden_size=sample_hidden_states.shape[-1])
    t = torch.tensor([0.5, 0.5], device=sample_hidden_states.device)
    velocity = block.get_velocity(sample_hidden_states, sample_hidden_states, None, t)
    assert velocity.shape == sample_hidden_states.shape

def test_flow_midblock_forward_euler_step_uses_dt(sample_hidden_states):
    block = FlowMidblock(hidden_size=sample_hidden_states.shape[-1])
    t = torch.tensor([0.0, 0.0], device=sample_hidden_states.device)
    h_next = block.forward(sample_hidden_states, sample_hidden_states, None, t, dt=0.25)
    assert h_next.shape == sample_hidden_states.shape
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_midblock.py -k "FlowMidblock or get_velocity or dt" -v`
Expected: FAIL because the module still exposes `IterativeMidblock` and discrete-step updates.

- [ ] **Step 3: Implement `FlowMidblock` and keep only optional compatibility aliases**

```python
class FlowMidblock(nn.Module):
    def get_velocity(self, h_t, h_start, attention_mask, t):
        ...

    def forward(self, h_t, h_start, attention_mask, t, dt):
        return h_t + self.get_velocity(h_t, h_start, attention_mask, t) * dt
```

- [ ] **Step 4: Run the full midblock suite**

Run: `python -m pytest tests/test_midblock.py tests/test_continuous_time_embedding.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/model/midblock.py src/model/__init__.py tests/test_midblock.py
git commit -m "feat: refactor midblock into continuous-time velocity predictor"
```

### Task 4: Add the ODE vector-field wrapper and integrate it into the student model

**Files:**
- Create: `src/model/ode.py`
- Modify: `src/model/student_qwen.py`
- Modify: `src/model/__init__.py`
- Create: `tests/test_ode_integration.py`
- Modify: `tests/test_student_qwen.py`

- [ ] **Step 1: Write the failing ODE integration tests**

```python
def test_midblock_vector_field_matches_torchdiffeq_signature():
    field = MidblockVectorField(midblock=mock_midblock, h_start=h_start, attention_mask=mask)
    out = field(torch.tensor(0.5), h_start)
    assert out.shape == h_start.shape

def test_euler_solver_normalizes_dt_between_one_and_many_steps():
    opts_1 = build_solver_options(method="euler", num_steps=1)
    opts_100 = build_solver_options(method="euler", num_steps=100)
    assert opts_1["step_size"] == 1.0
    assert opts_100["step_size"] == 0.01
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_ode_integration.py tests/test_student_qwen.py -k "ode or solver or num_steps_can_exceed_max" -v`
Expected: FAIL because there is no ODE wrapper and the student still uses a manual loop.

- [ ] **Step 3: Implement the ODE helper module and wire `FrozenQwenStudent.forward` through `odeint`**

```python
class MidblockVectorField(nn.Module):
    def __init__(self, midblock, h_start, attention_mask):
        ...

    def forward(self, t, h_t):
        batch = h_t.shape[0]
        t_batch = torch.full((batch,), float(t), device=h_t.device, dtype=h_t.dtype)
        return self.midblock.get_velocity(h_t, self.h_start, self.attention_mask, t_batch)
```

- [ ] **Step 4: Add student-model tests for solver controls**

```python
output = student(input_ids, attention_mask, num_steps=100, solver_method="euler")
assert output.shape[:2] == input_ids.shape
```

- [ ] **Step 5: Run the targeted solver and student tests**

Run: `python -m pytest tests/test_ode_integration.py tests/test_student_qwen.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/model/ode.py src/model/student_qwen.py src/model/__init__.py tests/test_ode_integration.py tests/test_student_qwen.py
git commit -m "feat: integrate torchdiffeq-backed ODE inference into student wrapper"
```

### Task 5: Refactor teacher-cache generation to store velocity targets

**Files:**
- Modify: `src/data/teacher_cache.py`
- Modify: `scripts/build_teacher_cache.py`
- Modify: `src/training/data.py`
- Modify: `tests/test_teacher_cache.py`
- Modify: `tests/test_train_smoke.py`

- [ ] **Step 1: Write the failing cache contract tests**

```python
def test_cache_writer_stores_velocity_target(temp_cache_dir):
    sample = {
        "input_ids": torch.randint(0, 10, (16,)),
        "attention_mask": torch.ones(16),
        "h_start": torch.randn(16, 32),
        "velocity_target": torch.randn(16, 32),
    }
    ...
    assert "velocity_target" in loaded

def test_build_teacher_cache_uses_h_end_minus_h_start(mock_inspector):
    assert torch.allclose(cache["velocity_target"], outputs["h_target"] - outputs["h_start"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_teacher_cache.py tests/test_train_smoke.py -k "velocity_target or cache" -v`
Expected: FAIL because the cache still stores `trajectory_targets` and `h_target` only.

- [ ] **Step 3: Update the cache schema and loader contract**

```python
cache_data = {
    "input_ids": ...,
    "attention_mask": ...,
    "h_start": outputs["h_start"],
    "velocity_target": outputs["h_target"] - outputs["h_start"],
}
```

- [ ] **Step 4: Make the training-state reconstruction explicit in cache metadata or loader docs**

```python
h_t = h_start + t[:, None, None] * velocity_target
```

This reconstruction rule must be documented in both `src/data/teacher_cache.py` metadata handling and `src/training/data.py` loader-facing docs so the trainer and tests do not invent different interpolation rules.

- [ ] **Step 5: Keep metadata explicit about the new target type**

```json
{
  "target_type": "velocity",
  "time_domain": [0.0, 1.0],
  "training_state_rule": "h_t = h_start + t * velocity_target"
}
```

- [ ] **Step 6: Run cache and dataloader tests**

Run: `python -m pytest tests/test_teacher_cache.py tests/test_train_smoke.py -v`
Expected: PASS.

- [ ] **Step 7: Run a real cache smoke build**

Run: `python scripts/build_teacher_cache.py --config configs/v0_onemotif.yaml --limit 8 --split train --overwrite --verify`
Expected: cache metadata includes `target_type = velocity`, and shard verification prints `velocity_target shape`.

- [ ] **Step 8: Commit**

```bash
git add src/data/teacher_cache.py src/training/data.py scripts/build_teacher_cache.py tests/test_teacher_cache.py tests/test_train_smoke.py
git commit -m "feat: cache teacher velocity targets for continuous-time training"
```

### Task 6: Switch training from endpoint matching to continuous-time velocity MSE

**Files:**
- Modify: `src/training/losses.py`
- Modify: `src/training/trainer.py`
- Modify: `scripts/train_v0.py`
- Modify: `tests/test_losses.py`
- Modify: `tests/test_architecture_loss_contract.py`
- Modify: `tests/test_train_smoke.py`

- [ ] **Step 1: Write the failing loss and trainer tests**

```python
def test_distillation_loss_matches_velocity_target(student_outputs, sample_batch):
    total_loss, metrics = loss_fn(student_outputs, sample_batch, T=8)
    assert "velocity_loss" in metrics

def test_trainer_samples_continuous_t_between_zero_and_one(trainer, batch):
    t = trainer.sample_continuous_time(batch_size=2, device=torch.device("cpu"))
    assert torch.all(t >= 0.0)
    assert torch.all(t <= 1.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_losses.py tests/test_train_smoke.py tests/test_architecture_loss_contract.py -k "velocity or continuous_t" -v`
Expected: FAIL because the loss still expects endpoint and trajectory hidden targets.

- [ ] **Step 3: Implement the minimal continuous-time training contract**

```python
t = torch.rand(batch_size, device=device)
h_t = batch["h_start"] + t[:, None, None] * batch["velocity_target"]
v_pred = model.midblock.get_velocity(h_t=h_t, h_start=batch["h_start"], attention_mask=batch.get("attention_mask"), t=t)
loss = F.mse_loss(v_pred, batch["velocity_target"])
```

- [ ] **Step 4: Keep architecture-training defaults free of logits**

```python
if "teacher_logits" not in teacher_batch:
    assert self.config.kl_weight == 0.0
```

- [ ] **Step 5: Run the updated training tests**

Run: `python -m pytest tests/test_losses.py tests/test_train_smoke.py tests/test_architecture_loss_contract.py -v`
Expected: PASS.

- [ ] **Step 6: Run a fast training smoke test**

Run: `python scripts/train_v0.py --config configs/v0_onemotif.yaml --fast-dev-run --device cpu`
Expected: one training step and one validation step complete with a logged `velocity_loss`, and debug logging confirms sampled non-boundary `t` values are used.

- [ ] **Step 7: Commit**

```bash
git add src/training/losses.py src/training/trainer.py scripts/train_v0.py tests/test_losses.py tests/test_architecture_loss_contract.py tests/test_train_smoke.py
git commit -m "feat: train flow midblock with continuous-time velocity loss"
```

### Task 7: Update text-sweep inference and repetition metrics

**Files:**
- Modify: `src/eval/text_checkpoint_sweep.py`
- Modify: `scripts/run_checkpoint_text_sweep.py`
- Modify: `src/eval/__init__.py`
- Modify: `tests/test_eval_pipeline.py`

- [ ] **Step 1: Write the failing evaluation tests**

```python
def test_run_text_sweep_records_solver_metadata(tmp_path):
    payload = run_text_sweep(..., num_steps=[4, 8], solver_method="rk4")
    assert payload["solver_method"] == "rk4"

def test_repetition_metrics_include_ngram_counts():
    metrics = compute_repetition_metrics("cat cat cat cat", n_values=(2, 3, 4))
    assert "repeat_2gram_ratio" in metrics
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_eval_pipeline.py -k "solver or repetition or text_sweep" -v`
Expected: FAIL because the sweep output does not include solver metadata or repetition metrics.

- [ ] **Step 3: Expose solver controls and add repetition helpers**

```python
payload = {
    "solver_method": solver_method,
    "num_steps": num_steps,
    "comparisons": comparisons,
    "repetition_metrics": aggregate_repetition_metrics(comparisons),
}
```

- [ ] **Step 4: Run the eval tests**

Run: `python -m pytest tests/test_eval_pipeline.py -v`
Expected: PASS.

- [ ] **Step 5: Run the milestone sweep**

Run: `python scripts/run_checkpoint_text_sweep.py --config configs/v0_onemotif.yaml --checkpoint outputs/v0_qwen_iterative_midblock/checkpoints/best.ckpt --num-steps 4 8 16 32 64 --solver-method euler --output outputs/text_sweep_euler.json`
Expected: JSON output includes solver metadata and repetition metrics for each prompt.

Run: `python scripts/run_checkpoint_text_sweep.py --config configs/v0_onemotif.yaml --checkpoint outputs/v0_qwen_iterative_midblock/checkpoints/best.ckpt --num-steps 4 8 16 32 64 --solver-method rk4 --output outputs/text_sweep_rk4.json`
Expected: second JSON output with the same schema for solver comparison.

- [ ] **Step 6: Commit**

```bash
git add src/eval/text_checkpoint_sweep.py scripts/run_checkpoint_text_sweep.py src/eval/__init__.py tests/test_eval_pipeline.py
git commit -m "feat: add ODE solver sweep controls and repetition metrics"
```

### Task 8: Run end-to-end verification for the refactor

**Files:**
- Modify: `docs/superpowers/plans/2026-03-18-continuous-ode-flow-midblock.md`

- [ ] **Step 1: Run the focused model and training suites**

Run: `python -m pytest tests/test_continuous_time_embedding.py tests/test_midblock.py tests/test_ode_integration.py tests/test_student_qwen.py tests/test_teacher_cache.py tests/test_losses.py tests/test_train_smoke.py tests/test_eval_pipeline.py -v`
Expected: PASS.

- [ ] **Step 2: Run a cache-build smoke test**

Run: `python scripts/build_teacher_cache.py --config configs/v0_onemotif.yaml --limit 8 --split train --overwrite --verify`
Expected: PASS with velocity-target metadata and readable shards.

- [ ] **Step 3: Run a training smoke test**

Run: `python scripts/train_v0.py --config configs/v0_onemotif.yaml --fast-dev-run`
Expected: PASS with non-NaN `velocity_loss`, checkpoint write, and logged evidence that `h_t = h_start + t * velocity_target` is used for sampled `t` values beyond only `0.0`.

- [ ] **Step 4: Verify checkpoint reload and parameter counts still work**

Run: `python -m pytest tests/test_student_qwen.py -k "ParameterCounts or state_dict" -v`
Expected: PASS, confirming the refactor did not break save/load compatibility or frozen/trainable count reporting.

- [ ] **Step 5: Run the solver normalization check**

Run: `python -m pytest tests/test_ode_integration.py -k euler_solver_normalizes_dt_between_one_and_many_steps -v`
Expected: PASS.

- [ ] **Step 6: Record the actual command matrix in the plan notes if any command differs from this draft**

```text
Update the plan file with the exact smoke-test commands that proved the implementation.
```

- [ ] **Step 7: Commit**

```bash
git add docs/superpowers/plans/2026-03-18-continuous-ode-flow-midblock.md
git commit -m "docs: finalize continuous ODE flow midblock verification plan"
```

---

## Milestone mapping

1. **Milestone 1 - Continuous embeddings and forward pass**: complete Tasks 1-3.
2. **Milestone 2 - ODE solver integration**: complete Task 4 and the solver-focused parts of Task 8.
3. **Milestone 3 - Target refactor and training**: complete Tasks 5-6.
4. **Milestone 4 - Evaluation updates**: complete Task 7 and the sweep commands in Task 8.

## Upstream reuse checklist

- Reuse the velocity-wrapper ideas from `external/minFM/models/latent_fm.py:373` when shaping `MidblockVectorField`, but keep the wrapper LM-hidden-state-specific.
- Reuse the time-grid / update semantics from `external/minFM/utils_fm/sampler.py:68`, but hand the actual integration loop to `torchdiffeq.odeint`.
- Do not port minFM image-packing helpers, noiser abstractions, VAE code, or classifier-free guidance into MidflowLM.

## Go / no-go rule

Do not trust the refactor until all of the following are true:
- `torchdiffeq` is the active inference path.
- No model test passes an integer `step_id` into the refactored midblock.
- Teacher cache metadata says `target_type = velocity`.
- Training logs report `velocity_loss` instead of endpoint / trajectory loss as the primary architecture objective.
- Training uses sampled intermediate states `h_t = h_start + t * velocity_target`, not only the boundary state `h_start`.
- Sweep outputs for `num_steps` in `[4, 8, 16, 32, 64]` are generated for both `euler` and `rk4`.
- Repetition metrics stop worsening monotonically as `num_steps` grows beyond the old training maximum.
