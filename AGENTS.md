# AGENTS.md

- **Always run "beans prime" at the task! **
- All implemenetations go into `src` at root

Repository working rules for Hermes and other coding agents.

## 1. Reuse before building
- Before implementing any model block, trainer, data pipeline, optimizer wrapper, or loss utility, first search for an established implementation in the current stack.
- Default preference order:
  1. Existing code already in this repository
  2. Official upstream library implementation
  3. Widely used library with active maintenance
  4. New custom implementation only if the above do not fit

## 2. Do not reimplement standard transformer internals without a written justification
- **Prefer existing Hugging Face Qwen modules over custom RMSNorm / SwiGLU / GQA implementations**
- Avoid hand-rolling RMSNorm, RoPE, GQA, SwiGLU, decoder layers, caching, masking, or trainer infrastructure if Hugging Face Transformers already provide the needed behavior
- **Avoid custom trainer frameworks when raw PyTorch suffices**
- If custom code is still required, document:
  - why reuse failed
  - the exact upstream reference implementation reviewed
  - parity tests proving compatibility

## 3. For this repository specifically (midflowlm)
- Treat this project as iterative latent matching / distillation unless a task explicitly requires formal continuous-time flow matching
- **Require parity tests before student training** - verify Qwen boundary extraction and bypass wrapper reproduction of teacher outputs
- **Require cache generation before training** - offline teacher-state caching is mandatory before any student training
- Prefer offline teacher-state caching before online dual-model training when memory is tight
- minFM is the upstream training reference for design patterns, but its image/video latent preprocessing and denoiser stack are not reused unchanged
- Raw PyTorch training loops are the default training path

## 4. Required pre-coding checklist
- Identify the exact upstream component(s) to reuse
- Pin versions if importing non-public internals
- **Define parity tests before modifying architecture**
- **Ensure teacher cache is generated before training**
- Define smoke-test commands before large training runs

## 5. Required post-change checklist
- Verify frozen/trainable parameter counts
- Verify checkpoint save/load
- Verify one batch forward pass and one optimizer step
- Record any library/API assumptions in docs
