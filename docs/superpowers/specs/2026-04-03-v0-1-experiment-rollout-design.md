# v0.1 Experiment Rollout Design

## Goal

Prepare the repository to execute the `docs/v0_1_exp_plan.md` and
`docs/v0_1_exp_matrix.md` experiment program on a Vast machine with `3x3090`,
using a single config-driven training and evaluation stack that supports the
full matrix, logs useful metrics to Weights & Biases, and keeps experiments
running continuously.

## Scope

This design covers:

- support for all v0.1 experiment axes through YAML config
- integration of `wandb` into the active training and evaluation path
- fixed-context `seq_len=1024` execution for v0.1
- one worst-case memory profile across the comparison matrix
- config generation for each experiment
- sequential queueing across 3 GPUs on one Vast machine

This design does not cover:

- changing the scientific content of the v0.1 plan
- optimizing for a different hardware class
- introducing a second independent training framework

## Current Repository State

The current repository already contains most of the v0.1 skeleton:

- `scripts/train.py` is the active YAML-driven training entrypoint
- `src/training/trainer.py` performs online teacher-target extraction and mixed
  precision training
- `src/data/mixed_corpus.py` supports mixed dataset composition, including plain
  text, chat, and MCQ-style datasets
- `scripts/eval_mmlu_pro.py` provides downstream MMLU-Pro probing
- `scripts/run_checkpoint_text_sweep.py` and
  `src/eval/text_checkpoint_sweep.py` provide post-training step sweeps

The main gaps relative to the v0.1 matrix are:

- the active train path does not yet expose all architecture variants through
  config
- `A1` and `A2` currently exist only as evaluation baselines, not as trainable
  model families in the active training path
- `wandb` is not integrated into the active path
- trajectory-anchor supervision for online training is not fully wired
- the loss contract is still velocity-first rather than v0.1 endpoint and
  behavior-first
- the trainer currently computes some targets and outputs even when a selected
  loss does not need them, which inflates memory usage
- there is no built-in multi-config queue runner for continuous execution on the
  Vast host
- there is no existing calibration harness or persisted hardware-profile
  contract for locking one `3090` resource profile across the matrix

## Design Principles

1. Reuse the current active training stack instead of building a separate v0.1
   framework.
2. Keep all experiment choices config-driven so the matrix can be audited and
   reproduced from YAML alone.
3. Use one fixed hardware profile across the comparison matrix to keep results
   comparable.
4. Size that profile against the worst-case intended loss configuration rather
   than against a cheaper special case.
5. Log enough train, validation, evaluation, and systems metrics to support the
   v0.1 plots without needing ad hoc reruns.
6. Keep the execution system robust to interruptions by checkpointing and by
   treating each config as an independent queued job.

## Recommended Architecture

### 1. Single Train/Eval Stack

The existing `scripts/train.py` and evaluation scripts remain the primary entry
points. The repository should gain a small model-selection layer so the train
script can instantiate the experiment family requested by config.

The supported v0.1 architecture families should be:

- `one_shot_projector` for `A1`, defined as the residual projector from the
  v0.1 plan: `h_end_hat = h_start + g_theta(h_start)`
- `shared_recurrent_residual` for `A2`, defined as one shared residual refiner
  block applied for `T` steps
- `flow_midblock` for `A3`, defined as the current step-conditioned velocity or
  ODE-style refinement path

The training, evaluation, checkpointing, and logging interfaces should stay the
same across all three so the experiment matrix is only a config problem, not a
script-selection problem.

Because only the flow-style path is trainable today, Phase 0 must promote the
existing `A1` and `A2` concepts from `src/eval/baselines.py` into trainable
student-family implementations under the same external interface used by the
active trainer.

That common student-family interface should preserve the current external
contract used by the trainer:

- accept `input_ids`, `attention_mask`, and `num_steps`
- expose endpoint hidden states when endpoint supervision is enabled
- expose trajectory hidden states only when trajectory supervision is enabled
- produce logits through the same frozen upper-stack continuation path
- reuse the same checkpointing and evaluation entrypoints regardless of family

### 2. Loss-Conditional Execution

The trainer should compute only the targets and outputs required by the active
loss weights.

Required behavior:

- if KL is disabled, do not materialize teacher logits
- if trajectory loss is disabled, do not extract anchor targets and do not keep
  student trajectories
- if endpoint loss is disabled, do not compute endpoint-only auxiliary targets
  beyond what is already needed for the architecture
- if CE is disabled, do not compute CE-only logging artifacts

This is necessary because the current path always requests dictionary outputs
from the student and always extracts teacher logits, which makes memory sizing
for a 3090 artificially pessimistic.

### 3. Fixed Context Policy

For v0.1, use `seq_len=1024` for all matrix experiments.

Truncation is accepted rather than used as a gating criterion. However,
truncation should be measured and logged so that results on different dataset
mixes remain interpretable.

The logged statistics should include at least:

- truncation rate per dataset component on train and validation splits
- mean pre-truncation token length per component when feasible
- effective tokens processed per optimization step

### 4. Fixed Worst-Case Hardware Profile

The matrix should use one GPU resource profile across all comparison runs.

The profile is chosen by calibrating on the heaviest intended training regime:

- `seq_len=1024`
- mixed precision enabled
- gradient checkpointing enabled if needed
- largest intended evaluation/training path overhead
- heaviest v0.1 loss combination, effectively `End + Traj + KL + CE`

The calibrated profile should lock:

- microbatch size
- gradient accumulation
- precision mode
- gradient checkpointing policy
- validation cadence if validation materially affects stability or runtime

Lighter experiments can run under the same profile even if they would have fit a
larger microbatch.

The calibration result should be written to a small durable artifact, such as a
profile YAML or JSON record, so matrix configs and queue workers consume one
agreed hardware contract instead of re-deriving resource choices ad hoc.

### 5. Config Matrix Representation

Each experiment in the matrix should have its own explicit config file with a
stable run name. Shared defaults can be factored into common config fragments if
the repo already supports that cleanly; otherwise, explicit files are preferred
over introducing a new config framework.

Config fields must cover:

- experiment metadata and run naming
- model family selection
- replacement span
- train-time `T` values and weights
- eval-time `T` sweep values
- loss weights under a stable schema that explicitly supports
  `velocity_weight`, `endpoint_weight`, `trajectory_weight`, `kl_weight`, and
  `ce_weight`
- data mix composition
- logging destinations
- queue metadata for execution ordering and recovery

For v0.1, the data mixes are:

- Mix A: FineWeb-Edu only
- Mix B: FineWeb-Edu plus UltraChat SFT
- Mix C: FineWeb-Edu, UltraChat SFT, ARC-Challenge, ARC-Easy,
  CommonsenseQA, and OpenBookQA

### 6. Queue-Based Vast Execution

The execution model should be a simple queue with three workers, one bound to
each `3090`.

Each worker should:

1. claim the next pending experiment config
2. launch training on its assigned GPU
3. resume from the latest checkpoint if the run already started
4. run post-training evaluation steps automatically
5. write completion or failure status to a durable run ledger
6. move on to the next config

This keeps the machine fully utilized while preserving clean per-experiment run
boundaries.

## Data Flow

### Training Flow

1. A queue worker selects an experiment config.
2. `scripts/train.py` loads the config and instantiates the selected model
   family.
3. The dataloader builds the requested data mix at `seq_len=1024`.
4. The trainer samples train-time `T` from the config.
5. The trainer requests only the teacher targets required by the active loss.
6. The student produces only the outputs required by the active loss.
7. Loss terms are computed and logged.
8. Checkpoints, structured metrics, and `wandb` artifacts are written.

### Evaluation Flow

1. After training, the queue worker launches the configured evaluation suite.
2. Eval sweeps run over the configured `T` values.
3. Metrics are logged to local artifacts and `wandb`.
4. The queue marks the experiment complete only after eval succeeds or records a
   clearly classified partial-failure state.

## Metrics and Logging

### Weights & Biases

`wandb` should be the primary experiment-tracking surface for v0.1.

The run schema should capture:

- experiment ID and matrix phase
- model family, loss mode, data mix, and span
- training and evaluation `T` settings
- fixed hardware profile values
- checkpoint path and resume metadata
- git commit hash if available

### Metrics

The minimum logged metrics should cover the v0.1 story.

Hidden fidelity:

- endpoint MSE
- endpoint cosine similarity
- trajectory anchor MSE
- hidden norm drift

Output fidelity:

- token KL versus teacher
- top-1 agreement versus teacher
- top-k agreement versus teacher

Language and task behavior:

- validation CE or perplexity by mixture component
- MCQ accuracy on ARC, CommonsenseQA, and OpenBookQA where applicable
- MMLU-Pro accuracy

Systems:

- peak GPU memory
- latency by eval `T`
- tokens per second by eval `T`
- wall-clock time per training step

## Error Handling and Recovery

The queue and training stack should handle common operational failures without
manual cleanup.

Required behaviors:

- if training is interrupted, resume from the latest checkpoint for that run
- if a run fails repeatedly due to a deterministic config or code issue, mark it
  failed and continue to the next queued run
- if a run fails due to infrastructure loss, leave it resumable
- if post-training eval fails, preserve the training artifact and mark eval as
  incomplete rather than losing the run record

## Validation Strategy

Validation should happen in this order:

1. smoke-test one config locally or on a single GPU path
2. confirm the fixed worst-case hardware profile on a single `3090`
3. verify queue execution with at least two short runs
4. then launch the full matrix in the planned phase order

The matrix order remains:

1. Phase 1 architecture sanity on Mix B
2. Phase 2 loss ablation on the best architecture
3. Phase 3 data mix ablation on the best architecture and loss
4. Phase 4 final probing and step-count study

## Risks

1. Supporting `A1/A2/A3` under one interface may require a small model-factory
   refactor rather than pure config additions.
2. Anchor extraction for trajectory supervision may add non-trivial compute and
   memory overhead even after making it conditional.
3. `seq_len=1024` with accepted truncation may still reduce throughput enough to
   force prioritization within the matrix.
4. `wandb` logging can become noisy if run naming and metric namespaces are not
   standardized before the first batch of runs.
5. The existing velocity-centered path may need careful adaptation so v0.1 loss
   modes do not regress old training behavior unintentionally.

## Recommended Implementation Phases

### Phase 0: v0.1 support closure

- add config-selectable architecture families
- implement trainable `A1` and `A2` student-family paths under the same trainer
  interface as `A3`
- make teacher target and student output production loss-conditional
- add missing v0.1 metrics and `wandb`
- define the fixed hardware-profile artifact consumed by matrix runs

### Phase 1: hardware calibration

- calibrate a fixed `1024`-context profile on the heaviest intended loss regime
- lock microbatch, accumulation, precision, and checkpointing for the matrix
- persist the resulting profile for reuse by the queue runner and experiment
  configs

### Phase 2: matrix config generation

- create one config per matrix experiment with stable names and metadata

### Phase 3: continuous execution

- add the Vast queue runner, durable run ledger, and post-run evaluation
  automation

### Phase 4: launch and monitor

- smoke run first, then execute the full matrix in queue order across `3x3090`

## Success Criteria

The design is successful when:

- every planned v0.1 experiment can be represented and launched from YAML
- train and eval metrics are visible in `wandb` and on disk
- the matrix runs continuously on a `3x3090` Vast machine without manual
  babysitting between experiments
- runs are resumable after interruption
- the resulting artifacts are sufficient to produce the quality-vs-steps,
  architecture, data-mix, and latency-vs-quality plots described in the v0.1
  plan
