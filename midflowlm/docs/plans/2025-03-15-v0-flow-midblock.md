# v0: Qwen Iterative Mid-Block Implementation Plan

> For Hermes: use subagent-driven-development to execute this plan task-by-task. Reuse existing library components before writing custom model internals. Do not reimplement standard transformer pieces when Hugging Face Qwen modules already provide them.

**Goal:** Replace a configurable middle span of Qwen3.5-0.8B, defaulting to layers 8-11, with a trainable iterative refinement block that starts from `h_start`, runs for `T` residual steps, and distills both the teacher endpoint hidden state and the teacher intermediate trajectory.

**Architecture:** Keep all Qwen modules outside the replacement span frozen. Run an offline teacher inference pass to cache `h_start`, full span trajectory states, `h_target`, and optional logits. Train a project-local causal hidden-state refiner inspired by minFM training patterns, but adapted to language-model hidden states rather than image latents.

**Tech Stack:** PyTorch, Hugging Face Transformers, Datasets, TorchMetrics, safetensors, PyYAML, einops

---

## Fixed design decisions

1. Replacement span is configurable through config keys `replacement_model.start_layer` and `replacement_model.end_layer`, with default values `8` and `11`.
2. All Qwen layers outside the replacement span remain frozen during v0 training.
3. The replacement block starts from `h_start` and refines with residual updates: `h_{k+1} = h_k + delta_k`.
4. Every iterative step uses causal attention semantics and original sequence positions unless an explicit ablation says otherwise.
5. Offline teacher caching is required before student training.
6. Endpoint supervision is required.
7. Intermediate trajectory supervision is also required. Endpoint-only training is out of scope for v0.
8. The default endpoint loss is hidden-state MSE.
9. For `T < depth(span)`, training must use an explicit compression target-alignment policy.
10. For `T = depth(span)`, training must use exact layerwise trajectory matching.
11. For `T > depth(span)`, training must use an explicit expansion / interpolation policy and label it approximate.
12. Raw PyTorch training loops are the default training path. Lightning is out of scope.
13. minFM is the upstream training reference, but its image/video latent preprocessing and denoiser stack are not reused unchanged.

---

## Working assumptions about the current repository

Current observed files before implementation:
- `requirements.txt`
- `configs/v0_onemotif.yaml`
- `src/model/flow_block.py`
- `docs/flux-adaptation-notes.md`
- `external/minFM/`

The repository does not yet contain the planned `tests/`, `scripts/`, `src/data/`, `src/training/`, or wrapper-model files, so this plan creates them explicitly.

---

## Task 1: Align repository guardrails and dependencies

**Objective:** Make the repository constraints explicit and remove stale FLUX/Lightning assumptions from the v0 config surface.

**Files:**
- Create: `AGENTS.md`
- Modify: `requirements.txt`
- Modify: `configs/v0_onemotif.yaml`

**Step 1: Write the failing dependency/config test checklist**
- Create a temporary checklist in the plan implementation notes or task notes covering:
  - raw PyTorch only
  - minFM as upstream reference
  - configurable replacement span
  - no `lightning` dependency in the final v0 requirements unless another existing file still truly needs it

**Step 2: Add repository guardrails**
- Create `AGENTS.md` at repo root.
- Add instructions to:
  - prefer existing Hugging Face Qwen modules over custom RMSNorm / SwiGLU / GQA
  - avoid custom trainer frameworks when raw PyTorch suffices
  - require parity tests before student training
  - require cache generation before training

**Step 3: Update requirements**
- Remove `lightning>=2.4.0` from `requirements.txt` unless a newly discovered hard dependency requires it.
- Keep only dependencies actually needed for the planned implementation.

**Step 4: Rewrite config structure**
- Update `configs/v0_onemotif.yaml` to use these top-level sections:
  - `experiment_name`
  - `seed`
  - `model`
  - `replacement_model`
  - `teacher_cache`
  - `data`
  - `loss`
  - `optimizer`
  - `scheduler`
  - `train_loop`
  - `logging`
- Replace stale fields such as `replacement_model.family: "flux-derived"` with `replacement_model.family: "minfm-hidden-refiner"`.
- Add:
  - `replacement_model.start_layer: 8`
  - `replacement_model.end_layer: 11`
  - explicit trajectory-alignment config for `T < depth(span)`, `T = depth(span)`, and `T > depth(span)`

**Step 5: Verify config and requirements**
Run:
- `cd /home/hungphongtrn/Workspace/midflowlm/midflowlm && python3 -c "import yaml; cfg=yaml.safe_load(open('configs/v0_onemotif.yaml')); print(cfg['replacement_model']['family']); print(cfg['replacement_model']['start_layer'], cfg['replacement_model']['end_layer']); print('train_loop' in cfg)"`
Expected:
- line 1: `minfm-hidden-refiner`
- line 2: `8 11`
- line 3: `True`

Run:
- `cd /home/hungphongtrn/Workspace/midflowlm/midflowlm && python3 -c "from pathlib import Path; txt=Path('requirements.txt').read_text(); print('lightning' in txt.lower())"`
Expected:
- `False` unless the plan is explicitly amended with justification

**Step 6: Commit**
Run:
- `cd /home/hungphongtrn/Workspace/midflowlm/midflowlm && git add AGENTS.md requirements.txt configs/v0_onemotif.yaml && git commit -m "docs: align v0 config and guardrails with minFM plan"`

---

## Task 2: Record the vendored minFM revision used for development

**Objective:** Pin the upstream training reference used for design and adaptation.

**Files:**
- Modify: `configs/v0_onemotif.yaml`
- Modify: `docs/plans/2025-03-15-v0-flow-midblock.md`

**Step 1: Record current minFM commit**
Run:
- `cd /home/hungphongtrn/Workspace/midflowlm/midflowlm && git -C external/minFM rev-parse HEAD`
Expected:
- one concrete commit hash

**Recorded:** `2b15c5ffa9f6a083650994e385ee11ca23ee7ab7`

**Step 2: Store the commit in config**
- Add a field such as `replacement_model.source_repo`, `replacement_model.source_commit`, or equivalent.

**Step 3: Verify the pin exists**
Run:
- `cd /home/hungphongtrn/Workspace/midflowlm/midflowlm && python3 -c "import yaml; cfg=yaml.safe_load(open('configs/v0_onemotif.yaml')); print(cfg['replacement_model']['source_repo']); print(cfg['replacement_model']['source_commit'])"`
Expected:
- source repo path or URL on line 1
- non-empty commit hash on line 2

**Step 4: Commit**
Run:
- `cd /home/hungphongtrn/Workspace/midflowlm/midflowlm && git add configs/v0_onemotif.yaml docs/plans/2025-03-15-v0-flow-midblock.md && git commit -m "docs: pin minFM reference commit for v0"`

---

## Task 3: Replace the hand-rolled flow block with a parity-first Qwen inspection module

**Objective:** Stop building on `src/model/flow_block.py` as the primary abstraction before proving correct Qwen boundary extraction.

**Files:**
- Modify: `src/model/flow_block.py`
- Create: `src/model/qwen_parity.py`
- Create: `tests/test_qwen_parity.py`

**Step 1: Write failing parity tests**
- Create `tests/test_qwen_parity.py`.
- Add tests for:
  1. extracting `h_start`, span states, and final logits for configurable `start_layer` / `end_layer`
  2. default span 8-11 indexing
  3. bypass wrapper reproducing teacher logits within tolerance
  4. frozen/trainable parameter counts matching expectations

**Step 2: Run the parity tests to confirm failure**
Run:
- `cd /home/hungphongtrn/Workspace/midflowlm/midflowlm && python3 -m pytest tests/test_qwen_parity.py -v`
Expected:
- FAIL because `src/model/qwen_parity.py` does not exist yet

**Step 3: Implement Qwen parity utilities**
- Create `src/model/qwen_parity.py`.
- Reuse loaded Qwen modules from `transformers`; do not reimplement decoder internals.
- Expose helpers to collect:
  - embeddings output
  - hidden state immediately before the replacement span
  - hidden states for each layer inside the replacement span
  - final logits
- Add a bypass/no-op wrapper path for exact teacher comparison.

**Step 4: De-emphasize or remove stale custom internals from `src/model/flow_block.py`**
- Either:
  - reduce `src/model/flow_block.py` to a placeholder module that is no longer treated as the architectural source of truth, or
  - refactor it so it no longer contains custom RMSNorm / SwiGLU / GQA implementations that violate repository rules.
- The final source of truth for model boundaries must be the real Qwen model, not custom approximations.

**Step 5: Run parity tests to pass**
Run:
- `cd /home/hungphongtrn/Workspace/midflowlm/midflowlm && python3 -m pytest tests/test_qwen_parity.py -v`
Expected:
- PASS for all parity/indexing tests

**Step 6: Commit**
Run:
- `cd /home/hungphongtrn/Workspace/midflowlm/midflowlm && git add src/model/flow_block.py src/model/qwen_parity.py tests/test_qwen_parity.py && git commit -m "test: add Qwen parity harness for configurable replacement span"`

---

## Task 4: Build TinyStories tokenization and dataset plumbing

**Objective:** Create deterministic data loading for both teacher-cache generation and student training.

**Files:**
- Create: `src/data/tinystories.py`
- Create: `tests/test_tinystories_data.py`

**Step 1: Write failing data tests**
- Add tests for:
  1. deterministic shuffled splits
  2. tokenization to fixed `seq_len`
  3. expected keys: `input_ids`, `attention_mask`
  4. sample count limits for train/val/test

**Step 2: Run the tests to confirm failure**
Run:
- `cd /home/hungphongtrn/Workspace/midflowlm/midflowlm && python3 -m pytest tests/test_tinystories_data.py -v`
Expected:
- FAIL because the data module does not exist yet

**Step 3: Implement dataset loader**
- Create `src/data/tinystories.py`.
- Use `datasets` and the Qwen tokenizer.
- Honor config fields for dataset revision, `seq_len`, sample limits, and shuffle seed.

**Step 4: Run tests to pass**
Run:
- `cd /home/hungphongtrn/Workspace/midflowlm/midflowlm && python3 -m pytest tests/test_tinystories_data.py -v`
Expected:
- PASS

**Step 5: Commit**
Run:
- `cd /home/hungphongtrn/Workspace/midflowlm/midflowlm && git add src/data/tinystories.py tests/test_tinystories_data.py && git commit -m "feat: add deterministic TinyStories data loader"`

---

## Task 5: Implement offline teacher-cache generation with full trajectory targets

**Objective:** Cache `h_start`, full span trajectory states, `h_target`, and optional logits before any student training.

**Files:**
- Create: `src/data/teacher_cache.py`
- Create: `scripts/build_teacher_cache.py`
- Create: `tests/test_teacher_cache.py`

**Step 1: Write failing teacher-cache tests**
- Add tests for:
  1. metadata writing
  2. resumable / idempotent shard writing
  3. presence of `h_start`, `trajectory_targets`, `h_target`
  4. span metadata storage: `start_layer`, `end_layer`, `span_depth`
  5. optional logits storage

**Step 2: Run tests to confirm failure**
Run:
- `cd /home/hungphongtrn/Workspace/midflowlm/midflowlm && python3 -m pytest tests/test_teacher_cache.py -v`
Expected:
- FAIL because cache code does not exist yet

**Step 3: Implement cache writer**
- Create `src/data/teacher_cache.py` and `scripts/build_teacher_cache.py`.
- Build teacher inference over tokenized TinyStories samples.
- For each sample, cache:
  - `input_ids`
  - `attention_mask`
  - `h_start`
  - `trajectory_targets`
  - `h_target`
  - optional `teacher_logits`
  - metadata for model revision and replacement span
- Ensure trajectory states are stored directly, not reconstructed later.

**Step 4: Run unit tests to pass**
Run:
- `cd /home/hungphongtrn/Workspace/midflowlm/midflowlm && python3 -m pytest tests/test_teacher_cache.py -v`
Expected:
- PASS

**Step 5: Run a real cache smoke test**
Run:
- `cd /home/hungphongtrn/Workspace/midflowlm/midflowlm && python3 scripts/build_teacher_cache.py --config configs/v0_onemotif.yaml --limit 8`
Expected:
- cache directory created
- metadata file created
- sample shards created for 8 examples or fewer if a shard format packs multiple samples

**Step 6: Verify cache contents programmatically**
Run:
- `cd /home/hungphongtrn/Workspace/midflowlm/midflowlm && python3 -c "from pathlib import Path; root=Path('cache'); print(root.exists()); print(any(root.rglob('*metadata*'))); print(any(root.rglob('*.pt')) or any(root.rglob('*.safetensors')) or any(root.rglob('*.jsonl')))"`
Expected:
- `True`
- `True`
- `True`

**Step 7: Commit**
Run:
- `cd /home/hungphongtrn/Workspace/midflowlm/midflowlm && git add src/data/teacher_cache.py scripts/build_teacher_cache.py tests/test_teacher_cache.py && git commit -m "feat: add teacher cache builder with full trajectory targets"`

---

## Task 6: Implement the trajectory-alignment policy module

**Objective:** Make mandatory trajectory supervision explicit for every `T` regime.

**Files:**
- Create: `src/training/alignment.py`
- Create: `tests/test_alignment.py`

**Step 1: Write failing alignment tests**
- Add tests for:
  1. exact alignment when `T = depth(span)`
  2. compression policy when `T < depth(span)`
  3. expansion / interpolation policy when `T > depth(span)`
  4. fail-fast behavior when required trajectory targets are missing

**Step 2: Run tests to confirm failure**
Run:
- `cd /home/hungphongtrn/Workspace/midflowlm/midflowlm && python3 -m pytest tests/test_alignment.py -v`
Expected:
- FAIL because the alignment module does not exist yet

**Step 3: Implement alignment policy**
- Create `src/training/alignment.py`.
- Define one explicit policy for each regime:
  - exact teacher layer matching for `T = depth(span)`
  - documented compression mapping for `T < depth(span)`
  - documented interpolation / expansion mapping for `T > depth(span)`
- Return aligned targets in a form directly consumable by the loss module.

**Step 4: Run tests to pass**
Run:
- `cd /home/hungphongtrn/Workspace/midflowlm/midflowlm && python3 -m pytest tests/test_alignment.py -v`
Expected:
- PASS

**Step 5: Commit**
Run:
- `cd /home/hungphongtrn/Workspace/midflowlm/midflowlm && git add src/training/alignment.py tests/test_alignment.py && git commit -m "feat: add trajectory alignment policy for variable T"`

---

## Task 7: Implement the iterative hidden-state refiner

**Objective:** Build a project-local causal refinement block that operates on Qwen hidden states and step conditioning.

**Files:**
- Create: `src/model/midblock.py`
- Create: `src/model/adapter.py`
- Create: `tests/test_midblock.py`

**Step 1: Write failing model tests**
- Add tests for:
  1. output shape preservation `[batch, seq, hidden]`
  2. causal-mask support
  3. configurable `start_layer` / `end_layer`
  4. `T=1` and `T=max_steps_T`
  5. save/load round-trip
  6. per-step stability for longer unrolls
  7. residual update behavior

**Step 2: Run tests to confirm failure**
Run:
- `cd /home/hungphongtrn/Workspace/midflowlm/midflowlm && python3 -m pytest tests/test_midblock.py -v`
Expected:
- FAIL because the midblock and adapter modules do not exist yet

**Step 3: Implement the midblock**
- Create `src/model/midblock.py` and `src/model/adapter.py`.
- Reuse PyTorch / HF components where possible.
- Input interface must accept:
  - current hidden states
  - `h_start`
  - `attention_mask`
  - `position_ids`
  - current step id
  - optional normalized step features such as `t/T`
- Output interface must preserve Qwen hidden shape.
- The update rule must be residual by default.

**Step 4: Run tests to pass**
Run:
- `cd /home/hungphongtrn/Workspace/midflowlm/midflowlm && python3 -m pytest tests/test_midblock.py -v`
Expected:
- PASS

**Step 5: Commit**
Run:
- `cd /home/hungphongtrn/Workspace/midflowlm/midflowlm && git add src/model/midblock.py src/model/adapter.py tests/test_midblock.py && git commit -m "feat: add iterative hidden-state midblock with step conditioning"`

---

## Task 8: Build the frozen student wrapper around Qwen

**Objective:** Assemble the real trainable student model with frozen outer Qwen layers and configurable replacement span.

**Files:**
- Create: `src/model/student_qwen.py`
- Create: `tests/test_student_qwen.py`

**Step 1: Write failing student-wrapper tests**
- Add tests for:
  1. only replacement block parameters are trainable
  2. bypass mode reproduces teacher outputs
  3. wrapper honors configurable `start_layer` / `end_layer`
  4. validation can run with different `T` values without rebuilding the model

**Step 2: Run tests to confirm failure**
Run:
- `cd /home/hungphongtrn/Workspace/midflowlm/midflowlm && python3 -m pytest tests/test_student_qwen.py -v`
Expected:
- FAIL because wrapper code does not exist yet

**Step 3: Implement the wrapper**
- Create `src/model/student_qwen.py`.
- Wire:
  - embeddings and lower frozen layers
  - iterative replacement block
  - upper frozen layers
  - final norm and LM head
- Keep HF-style output compatibility where practical.

**Step 4: Run tests to pass**
Run:
- `cd /home/hungphongtrn/Workspace/midflowlm/midflowlm && python3 -m pytest tests/test_student_qwen.py -v`
Expected:
- PASS

**Step 5: Commit**
Run:
- `cd /home/hungphongtrn/Workspace/midflowlm/midflowlm && git add src/model/student_qwen.py tests/test_student_qwen.py && git commit -m "feat: wrap frozen Qwen skeleton around iterative midblock"`

---

## Task 9: Implement loss functions for endpoint and mandatory trajectory distillation

**Objective:** Make the supervision objective explicit, testable, and fail-fast.

**Files:**
- Create: `src/training/losses.py`
- Create: `tests/test_losses.py`

**Step 1: Write failing loss tests**
- Add tests for:
  1. endpoint MSE computation
  2. mandatory trajectory loss computation
  3. weighting of endpoint / trajectory / KL / CE terms
  4. fail-fast behavior when trajectory targets are missing
  5. grouped metric outputs

**Step 2: Run tests to confirm failure**
Run:
- `cd /home/hungphongtrn/Workspace/midflowlm/midflowlm && python3 -m pytest tests/test_losses.py -v`
Expected:
- FAIL because the losses module does not exist yet

**Step 3: Implement losses**
- Create `src/training/losses.py`.
- Required losses:
  - endpoint hidden-state MSE
  - trajectory hidden-state loss using the alignment policy
- Optional losses:
  - KL on logits if present
  - CE on labels if enabled
- The loss module must reject missing trajectory targets for configured training runs.

**Step 4: Run tests to pass**
Run:
- `cd /home/hungphongtrn/Workspace/midflowlm/midflowlm && python3 -m pytest tests/test_losses.py -v`
Expected:
- PASS

**Step 5: Commit**
Run:
- `cd /home/hungphongtrn/Workspace/midflowlm/midflowlm && git add src/training/losses.py tests/test_losses.py && git commit -m "feat: add endpoint and trajectory distillation losses"`

---

## Task 10: Implement raw PyTorch data loading and trainer loop

**Objective:** Build the training loop used for fast iteration and smoke tests.

**Files:**
- Create: `src/training/data.py`
- Create: `src/training/trainer.py`
- Create: `scripts/train_v0.py`
- Create: `tests/test_train_smoke.py`

**Step 1: Write failing trainer smoke tests**
- Add tests for:
  1. deterministic dataloaders from cache
  2. one train step
  3. one val step
  4. checkpoint save/load
  5. variable `T` sampling from config

**Step 2: Run smoke tests to confirm failure**
Run:
- `cd /home/hungphongtrn/Workspace/midflowlm/midflowlm && python3 -m pytest tests/test_train_smoke.py -v`
Expected:
- FAIL because trainer code does not exist yet

**Step 3: Implement trainer**
- Create `src/training/data.py`, `src/training/trainer.py`, and `scripts/train_v0.py`.
- Support:
  - fixed-`T=depth(span)` training
  - variable-`T` training from config
  - optimizer and scheduler setup
  - AMP / bf16 as configured
  - checkpoint save/load
  - logging of train and val metrics

**Step 4: Run unit smoke tests to pass**
Run:
- `cd /home/hungphongtrn/Workspace/midflowlm/midflowlm && python3 -m pytest tests/test_train_smoke.py -v`
Expected:
- PASS

**Step 5: Run fast-dev training smoke test**
Run:
- `cd /home/hungphongtrn/Workspace/midflowlm/midflowlm && python3 scripts/train_v0.py --config configs/v0_onemotif.yaml --fast-dev-run`
Expected:
- one train step completes
- one val step completes
- process exits successfully

**Step 6: Run limited real loop smoke test**
Run:
- `cd /home/hungphongtrn/Workspace/midflowlm/midflowlm && python3 scripts/train_v0.py --config configs/v0_onemotif.yaml --limit-train-batches 2 --limit-val-batches 2`
Expected:
- 2 train batches complete
- up to 2 val batches complete
- checkpoint or log output directory appears

**Step 7: Commit**
Run:
- `cd /home/hungphongtrn/Workspace/midflowlm/midflowlm && git add src/training/data.py src/training/trainer.py scripts/train_v0.py tests/test_train_smoke.py && git commit -m "feat: add raw PyTorch training loop for iterative midblock"`

---

## Task 11: Add baselines and evaluation pipeline

**Objective:** Make the experiment falsifiable before scaling up.

**Files:**
- Create: `src/eval/baselines.py`
- Create: `scripts/eval_v0.py`
- Create: `tests/test_eval_pipeline.py`

**Step 1: Write failing evaluation tests**
- Add tests for:
  1. identity baseline
  2. `T=1` shared-block baseline
  3. simple recurrent baseline without minFM-inspired step conditioning
  4. metric reporting format

**Step 2: Run tests to confirm failure**
Run:
- `cd /home/hungphongtrn/Workspace/midflowlm/midflowlm && python3 -m pytest tests/test_eval_pipeline.py -v`
Expected:
- FAIL because eval code does not exist yet

**Step 3: Implement evaluation path**
- Create `src/eval/baselines.py` and `scripts/eval_v0.py`.
- Required outputs:
  - endpoint hidden-state error
  - trajectory error
  - KL divergence if logits exist
  - perplexity if labels/logits are present
  - latency / tokens-per-second by `T`
  - total and trainable parameter counts
  - hidden-state norm and delta stability metrics

**Step 4: Run tests to pass**
Run:
- `cd /home/hungphongtrn/Workspace/midflowlm/midflowlm && python3 -m pytest tests/test_eval_pipeline.py -v`
Expected:
- PASS

**Step 5: Run evaluation smoke test**
Run:
- `cd /home/hungphongtrn/Workspace/midflowlm/midflowlm && python3 scripts/eval_v0.py --config configs/v0_onemotif.yaml --limit 8`
Expected:
- evaluation summary prints successfully
- baseline metrics are reported

**Step 6: Commit**
Run:
- `cd /home/hungphongtrn/Workspace/midflowlm/midflowlm && git add src/eval/baselines.py scripts/eval_v0.py tests/test_eval_pipeline.py && git commit -m "feat: add evaluation pipeline and baselines"`

---

## Task 12: Add operational helpers and final documentation pass

**Objective:** Make the implementation runnable by someone with zero prior context.

**Files:**
- Create: `scripts/print_model_stats.py`
- Modify: `configs/v0_onemotif.yaml`
- Modify: `docs/plans/2025-03-15-v0-flow-midblock.md`

**Step 1: Add model-stats helper**
- Create `scripts/print_model_stats.py`.
- Report:
  - total parameters
  - trainable parameters
  - frozen parameters
  - replacement span
  - allowed `T` values

**Step 2: Document operational assumptions in config or plan**
- Record:
  - minimum tested GPU memory
  - expected cache size
  - checkpoint naming scheme
  - resume command
  - expected smoke-test runtime

**Step 3: Verify model-stats helper**
Run:
- `cd /home/hungphongtrn/Workspace/midflowlm/midflowlm && python3 scripts/print_model_stats.py --config configs/v0_onemotif.yaml`
Expected:
- parameter summary prints successfully
- replacement span and `T` schedule are visible in output

**Step 4: Final repo-wide test pass**
Run:
- `cd /home/hungphongtrn/Workspace/midflowlm/midflowlm && python3 -m pytest tests -v`
Expected:
- all tests pass

**Step 5: Final smoke sequence**
Run in order:
- `cd /home/hungphongtrn/Workspace/midflowlm/midflowlm && python3 scripts/build_teacher_cache.py --config configs/v0_onemotif.yaml --limit 8`
- `cd /home/hungphongtrn/Workspace/midflowlm/midflowlm && python3 scripts/train_v0.py --config configs/v0_onemotif.yaml --fast-dev-run`
- `cd /home/hungphongtrn/Workspace/midflowlm/midflowlm && python3 scripts/train_v0.py --config configs/v0_onemotif.yaml --limit-train-batches 2 --limit-val-batches 2`
- `cd /home/hungphongtrn/Workspace/midflowlm/midflowlm && python3 scripts/eval_v0.py --config configs/v0_onemotif.yaml --limit 8`
Expected:
- all four commands exit successfully

**Step 6: Commit**
Run:
- `cd /home/hungphongtrn/Workspace/midflowlm/midflowlm && git add scripts/print_model_stats.py configs/v0_onemotif.yaml docs/plans/2025-03-15-v0-flow-midblock.md && git commit -m "docs: finalize execution-ready v0 plan and helpers"`

---

## Go / no-go rule for v0

Do not scale the project to larger datasets or more complex objectives unless all of the following are true:
1. teacher-cache generation is deterministic
2. parity tests pass
3. raw PyTorch train/val smoke tests pass
4. trajectory supervision is active in actual runs
5. the iterative block beats the identity replacement baseline
6. the iterative block is not materially worse than a simple distilled baseline at comparable compute

---

## Final expected file layout after implementation

```text
midflowlm/
  AGENTS.md
  configs/
    v0_onemotif.yaml
  docs/
    plans/
      2025-03-15-v0-flow-midblock.md
  external/
    minFM/
  scripts/
    build_teacher_cache.py
    eval_v0.py
    print_model_stats.py
    train_v0.py
  src/
    data/
      teacher_cache.py
      tinystories.py
    eval/
      baselines.py
    model/
      adapter.py
      flow_block.py
      midblock.py
      qwen_parity.py
      student_qwen.py
    training/
      alignment.py
      data.py
      losses.py
      trainer.py
  tests/
    test_alignment.py
    test_eval_pipeline.py
    test_losses.py
    test_midblock.py
    test_qwen_parity.py
    test_student_qwen.py
    test_teacher_cache.py
    test_tinystories_data.py
    test_train_smoke.py
```

---

## Recommended execution order

Execute tasks in this exact order:
1. Task 1
2. Task 2
3. Task 3
4. Task 4
5. Task 5
6. Task 6
7. Task 7
8. Task 8
9. Task 9
10. Task 10
11. Task 11
12. Task 12

Do not start student training implementation before Tasks 3, 5, and 6 are complete.
Do not claim success before Task 11 baselines exist.
