# Decision Log

## 2026-05-08: ADR 0001 — Switch to HuggingFace Trainer with CE-only training

**Context:** Issue #9 introduces SFT on reasoning data. The existing custom `Trainer` (`src/training/trainer.py`) supports velocity/endpoint/trajectory/KL losses but is overbuilt for plain CE-only SFT. We need compatibility with Liger Kernel fused ops.

**Decision:** From issue #9 onward, training uses HF `Trainer` with `AutoLigerKernelForCausalLM`. The existing custom trainer is renamed `DistillationTrainer` in `src/training/distillation_trainer.py`.

**Rationale:**
- HF Trainer handles batching, gradient accumulation, checkpointing, logging natively
- Liger Kernel provides fused CE that's faster than manual `F.cross_entropy`
- Simpler code path for CE-only training
- Existing trainer preserved for distillation experiments

**Consequences:**
- New model wrapper must output HF Trainer compatible forward (returns `{"loss": ..., "logits": ...}`)
- Loss computation built into the model forward (Liger Kernel computes CE internally)
- No more `extract_teacher_targets()` — targets are the labels tensor

## 2026-05-08: ADR 0002 — Monkey-patch FlowMidblock into Qwen layers

**Context:** We need the FlowMidblock to intercept hidden states between Qwen layers 7 and 12, replacing layers 8-11. Two approaches: (a) override the Qwen model's forward method, or (b) physically replace layers in the ModuleList.

**Decision:** Override the Qwen model's `forward` method. Access `model.model` (the base Qwen3Model), replace its `forward` with one that routes layer 7 output → FlowMidblock → layer 12 input.

**Rationale:**
- Physically replacing ModuleList entries risks breaking Qwen's save/load, state dict keys, and config
- Forward override is a single clean patch point
- FlowMidblock's ODE integration (`torchdiffeq.odeint`) is fully differentiable with HF Trainer's `loss.backward()`

**Consequences:**
- The wrapper class stores both the Qwen model and the FlowMidblock
- Forward override must handle attention mask and position IDs correctly
- Checkpoint save/load needs custom handling for FlowMidblock weights

## 2026-05-08: ADR 0003 — Fixed T=32, no variable-step training

**Context:** P3-D3 trained with variable T [2,4,6,8] and continuous time sampling. For SFT, we want a fixed high T to maximize refinement quality.

**Decision:** T is hardcoded as `self.thinking_level = 32` inside the model. No per-batch sampling.

**Rationale:**
- Simplifies training — single forward path, predictable compute
- Higher T = more ODE steps = potentially better reasoning
- Eval at multiple T values post-training to find the accuracy-compute tradeoff

**Consequences:**
- Model init parameter: `thinking_level: int = 32`
- Forward always uses `num_steps = self.thinking_level`
- Future adaptive-T work can add sampling back

## 2026-05-08: ADR 0004 — New dependencies for Qwen3.5 compatibility

**Context:** Qwen3.5-0.8B uses GatedDeltaNet layers (75%) and full-attention layers (25%). These require `causal-conv1d`, `flash-linear-attention`, and `flash-attn`.

**Decision:** Add `causal-conv1d`, `flash-linear-attention`, `flash-attn`, `liger-kernel` as dependencies.

**Consequences:**
- Installation via prebuilt wheels for CUDA 13 / PyTorch 2.10
- `causal-conv1d` and `flash-attn` from prebuilt GitHub release wheels
- `flash-linear-attention` from git source
- `liger-kernel` from PyPI
