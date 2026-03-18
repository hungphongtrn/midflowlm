# Continuous ODE Flow Midblock Spec

## Objective

Refactor MidflowLM from discrete iterative residual updates into a continuous ODE / flow-matching formulation so inference can safely scale `num_steps` beyond the trained maximum without OOD overshoot or feature collapse.

## Required architecture changes

1. Remove discrete step embeddings from the student midblock.
2. Introduce a `ContinuousTimeEmbedding` that maps scalar `t in [0, 1]` to the hidden dimension using a frequency-based embedding.
3. Refactor `IterativeMidblock` into a velocity-predicting `FlowMidblock` with signature `get_velocity(self, h_t, h_start, attention_mask, t)`.
4. Add an ODE wrapper:

```python
class MidblockVectorField(nn.Module):
    # forward(self, t, h_t)
    # holds h_start and attention_mask as state
```

## Required inference changes

1. Add `torchdiffeq` to project dependencies.
2. Replace the manual refinement loop with `torchdiffeq.odeint` over `t in [0.0, 1.0]`.
3. Expose `solver_method` and `num_steps` in inference scripts.
4. For Euler, set `step_size = 1.0 / num_steps`.

## Required training and cache changes

1. Update `scripts/build_teacher_cache.py` to store `velocity_target = h11 - h7` instead of endpoint-only matching targets.
2. During training, sample `t ~ U(0, 1)`.
3. Train with MSE against the cached velocity target:

```python
Loss = ||v_theta(h_t, t) - (h11 - h7)||^2
```

## Required evaluation changes

1. Add automated n-gram repetition metrics to sweep outputs.
2. Run sweeps over `num_steps = [4, 8, 16, 32, 64]`.
3. Compare both `euler` and `rk4` solver behavior.

## Upstream reuse constraint

Use `external/minFM` as the design reference instead of manually inventing flow-matching patterns where the vendored code already provides a suitable template.
