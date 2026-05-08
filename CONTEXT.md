# MidFlowLM Context

MidFlowLM explores latent refinement inside a frozen Qwen model, with experiments judging whether iterative hidden-state computation can improve answer quality beyond teacher matching.

## Language

**Latent Compute Budget**:
The number of hidden-state refinement steps available to the replacement midblock for a single forward pass.
_Avoid_: Thinking tokens, reasoning length

**T**:
The concrete value of the latent compute budget used for one model run.
_Avoid_: Solver resolution when discussing answer-quality experiments

**Adaptive T**:
A policy or rule that chooses a per-example latent compute budget instead of using one fixed T for all examples.
_Avoid_: Thinking budget when the budget is not visible token length

**Difficulty-Aware T**:
An adaptive T policy that spends more latent compute on examples estimated to be harder and less latent compute on easier examples.
_Avoid_: Fixed high T

**Model-Derived Hardness**:
An example is hard when the model's low-budget run is wrong or uncertain and a higher T run improves correctness or confidence.
_Avoid_: Domain label as hardness proxy

**T Request Policy**:
A policy learned by the model that emits a requested latent compute budget for the current assistant turn.
_Avoid_: Model self-awareness

**In-Turn T Request**:
A T request emitted at the start of an assistant turn and applied by the runtime to the remaining generation in that same turn.
_Avoid_: Next-turn budget setting

**Budget-Request Segment**:
The initial low-cost generation segment that emits the in-turn T request before answer generation begins.
_Avoid_: Full reasoning pass

**Verifier-Gated T Escalation**:
A multi-attempt training loop where the model first answers with low T, a verifier checks correctness, and the model is rewarded for requesting higher T only when the cheap attempt fails.
_Avoid_: Single-pass budget request

**Verifier-Friendly Task**:
A task whose answer can be checked automatically during rollout, such as math, multiple-choice QA, or code tests.
_Avoid_: Open-ended preference task

**Oracle T Envelope**:
An offline analysis that scores each example at multiple T values and reports the best achievable result if T were chosen perfectly per example.
_Avoid_: Learned T policy

**Adaptive-T Feasibility Probe**:
An experiment that tests whether the oracle T envelope beats fixed-T baselines enough to justify training an adaptive T policy.
_Avoid_: Generic CE baseline

**Reasoning SFT Output**:
A supervised response target that includes both the visible `<think>...</think>` reasoning trace and the final answer.
_Avoid_: Final-answer-only target

**Distillation Training**:
Training mode that uses on-the-fly teacher hidden-state extraction (velocity, KL, trajectory losses) to match student midblock outputs to teacher Qwen layers. Implemented in `src/training/distillation_trainer.py`.
_Avoid_: Online training, cached training

**CE-Only Training**:
Training mode that uses only next-token prediction loss via Liger Kernel fused cross-entropy, with the FlowMidblock running as an internal computation layer. Implemented via HuggingFace `Trainer`.
_Avoid_: Distillation training, flow matching loss

**Monkey-Patch**:
The strategy of replacing `model.language_model.layers[start:end+1]` with a custom midblock module that intercepts hidden states between lower and upper frozen layers. Used instead of `FrozenQwenStudent` wrapper for HF Trainer compatibility.
_Avoid_: Model wrapper, subclassing

**HF Dataset**:
A HuggingFace `Dataset` object with `input_ids`, `attention_mask`, and `labels` columns, compatible with HF `Trainer`. Replaces the `DataLoader`-returning `dataset_factory.py` for CE-only training.
_Avoid_: Dataloader, token batch

**Pack**:
Concatenating multiple short training samples into one sequence up to max context length, with proper cross-boundary loss masking, using TRL's `pack_dataset`.
_Avoid_: Batching, chunking

**Meta-Filter**:
Filtering dataset samples by `meta.input_tokens + meta.output_tokens <= N` before packing, to exclude samples that would exceed the model's context window.
_Avoid_: Truncation, length clipping

**Thinking Level**:
The hardcoded `num_steps` (T) value stored as `self.thinking_level` inside the model, used by the FlowMidblock during every forward pass for CE-only training.
_Avoid_: T sampling, adaptive T

## Relationships

- A **Latent Compute Budget** is instantiated by a specific **T**.
- **Adaptive T** is only useful when the **Oracle T Envelope** beats fixed-T baselines.
- **Difficulty-Aware T** is the desired human-like behavior for MidFlowLM: harder examples should receive larger **T** only when extra latent compute improves answer quality.
- **Model-Derived Hardness** is the hardness signal for **Difficulty-Aware T** in issue #8.
- A **T Request Policy** can implement **Difficulty-Aware T** when the runtime honors its requested **T**.
- An **In-Turn T Request** requires generation to be split into a budget-request segment and an answer segment in the same assistant turn.
- An **In-Turn T Request** uses XML-style syntax: `<think-level>N</think-level>`.
- The **Budget-Request Segment** for issue #8 uses **T** = 1.
- **Verifier-Gated T Escalation** is the preferred training setup for **Difficulty-Aware T** when using **Verifier-Friendly Tasks**.
- An **Adaptive-T Feasibility Probe** precedes GRPO or learned-halting work.
- A **Reasoning SFT Output** can train visible reasoning format, but issue #8 evaluates final-answer correctness and **Oracle T Envelope** rather than visible trace quality alone.
- From issue #9 onward: **CE-Only Training** uses HuggingFace `Trainer` with **HF Dataset** and **Liger Kernel**; the **Monkey-Patch** strategy replaces `FrozenQwenStudent`; **Thinking Level** replaces per-batch T sampling.
- **Distillation Training** remains available in `distillation_trainer.py` for legacy experiments but is not used in CE-only path.
- **Meta-Filter** is applied before **Pack** to exclude samples exceeding context length, then **Pack** fills the remaining capacity with multiple short samples.

## Example Dialogue

> **Dev:** "Should issue #8 train the model to choose T?"
> **Domain expert:** "Not yet. First run an **Adaptive-T Feasibility Probe**: after SFT, sweep several **T** values and measure the **Oracle T Envelope**. If the envelope is flat, there is no adaptive compute signal to learn."

> **Dev:** "Is training at fixed T=32 enough?"
> **Domain expert:** "No. Fixed high T only tests forced longer thinking. The target behavior is **Difficulty-Aware T**, where hard examples use more latent compute and easy examples stop earlier."

> **Dev:** "Do we use dataset domain as the hard/easy label?"
> **Domain expert:** "No. Use **Model-Derived Hardness**: the example is hard for this model if low T struggles and higher T helps."

> **Dev:** "Can the model choose a think level inside the same response?"
> **Domain expert:** "Yes, as an **In-Turn T Request**: generate a budget tag first, parse it, then use the requested **T** for the remaining answer generation."

> **Dev:** "How does the model learn to think longer when it is wrong?"
> **Domain expert:** "Use **Verifier-Gated T Escalation** on **Verifier-Friendly Tasks**: try cheap first, verify, then reward higher **T** requests only when escalation fixes the answer."

> **Dev:** "Should SFT strip the reasoning trace and train only final answers?"
> **Domain expert:** "No. For issue #8, use the full **Reasoning SFT Output**, then judge the run by final-answer correctness and T-dependent behavior."

## Flagged Ambiguities

- "Thinking budget" can mean visible chain-of-thought token length in Qwen-style systems or latent refinement steps in MidFlowLM; resolved: use **Latent Compute Budget** for MidFlowLM's hidden-state steps.
- "Model self-awareness" was used to describe emitting `<think-level>`; resolved: call this a **T Request Policy** unless the model receives explicit latent-state introspection signals.
- "Multi-turn GRPO" was used for retry-after-failure training; resolved: call this **Verifier-Gated T Escalation** because the verifier gates whether higher T is useful.
- "Trainer" ambiguously refers to both the legacy custom `DistillationTrainer` and HuggingFace `Trainer`; resolved: the custom one is **Distillation Training** in `distillation_trainer.py`; HF `Trainer` refers to **CE-Only Training**.
- "T" was previously sampled per batch in the trainer; resolved: for CE-only training, **Thinking Level** is hardcoded inside the model.
