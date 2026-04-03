# **v0.1 Experiment Plan** for **MidflowLM 8→11**.

## 1. Goal of v0.1

Establish whether a trainable iterative midblock can replace **Qwen layers 8–11** while:

* preserving teacher hidden-state transport,
* preserving downstream LM behavior,
* and giving a usable **quality vs steps** tradeoff at inference.

This version is not yet about full optimization. It is about **proving the hypothesis clearly**.

---

## 2. Main hypotheses

### H1. Span replacement works

A learned midblock can approximate the teacher mapping from layer-7 boundary state to layer-11 boundary state with limited degradation in loss and benchmark behavior.

### H2. Iteration matters

A shared iterative block outperforms a one-shot projector at similar parameter scale.

### H3. Step count is a real knob

Increasing inference refinement steps improves fidelity and task performance up to some point.

### H4. Mixed-regime training helps

Training on plain text + chat + MCQ preserves the replaced span more robustly across different behavior modes than plain text alone.

### H5. Hidden-state matching alone is not enough

Adding teacher-logit distillation materially improves downstream behavior beyond endpoint hidden-state loss alone.

---

## 3. Fixed setup

### Model

* Base teacher: frozen Qwen
* Frozen student parts:

  * embeddings
  * layers 0–7
  * layers 12–23
  * LM head
* Trainable:

  * replacement module for layers 8–11 only

### Boundaries

* `h_start`: hidden state after layer 7
* `h_end`: teacher hidden state after layer 11

Optional intermediate anchors:

* `h_8`, `h_9`, `h_10`, `h_11`

---

## 4. Dataset mixes

Use three mixes.

### Mix A: plain text only

* FineWeb-Edu only

Purpose:

* isolate generic LM transport

### Mix B: plain text + chat

* FineWeb-Edu
* UltraChat SFT

Purpose:

* test whether instruction behavior needs explicit preservation

### Mix C: full mix

* FineWeb-Edu
* UltraChat SFT
* ARC-Challenge
* ARC-Easy
* CommonsenseQA
* OpenBookQA

Purpose:

* stress cross-regime preservation

This gives a very clean ablation story.

---

## 5. Architecture ablations

You need at least these three.

### A0. Teacher

Original frozen Qwen.

### A1. One-shot projector

A residual MLP:
[
\hat h_{end} = h_{start} + g_\theta(h_{start})
]

Purpose:

* test whether iteration is needed at all

### A2. Shared recurrent residual block

A shared block applied for `T` steps:
[
h_{k+1} = h_k + f_\theta(h_k, t_k)
]

No flow-style velocity objective required.

Purpose:

* test whether simple iterative reuse already explains gains

### A3. FlowMidblock

Continuous/step-conditioned velocity-style refinement:
[
\frac{dh}{dt} = v_\theta(h,t)
]

Euler rollout with `T` steps.

Purpose:

* your main method

---

## 6. Loss ablations

Keep this disciplined.

### L1. Endpoint only

[
L_{end} = | \mathrm{Norm}(\hat h_{11}) - \mathrm{Norm}(h_{11}) |^2
]

This is the simplest transport objective.

### L2. Endpoint + KL

[
L = \lambda_{end}L_{end} + \lambda_{KL}L_{KL}
]

where
[
L_{KL} = KL(p_{teacher} | p_{student})
]

This is likely your strongest practical baseline.

### L3. Endpoint + trajectory + KL

Add supervision for intermediate anchors:
[
L_{traj} = \sum_{i \in {8,9,10,11}} w_i |\mathrm{Norm}(\hat h_i) - \mathrm{Norm}(h_i)|^2
]

Then:
[
L = \lambda_{end}L_{end} + \lambda_{traj}L_{traj} + \lambda_{KL}L_{KL}
]

### L4. Endpoint + trajectory + KL + CE

[
L = \lambda_{end}L_{end} + \lambda_{traj}L_{traj} + \lambda_{KL}L_{KL} + \lambda_{CE}L_{CE}
]

Use this only after the simpler ones, because CE can blur whether you are preserving teacher transport or just adapting behavior.

---

## 7. My recommended default loss for v0.1

Start with:

[
L = 1.0 L_{end} + 1.0 L_{traj} + 0.5 L_{KL}
]

Optional later:
[

* 0.1 L_{CE}
  ]

Why:

* endpoint keeps the destination aligned,
* trajectory prevents degenerate shortcuting,
* KL preserves behavior at the output.

I would **not** start with CE-heavy training.

---

## 8. Training-time step strategy

This matters a lot.

### Bad option

Train only at fixed `T=4`, then hope `T=2` and `T=8` work.

### Better option

Train with random step count:
[
T \sim {2,4,6,8}
]

This makes variable-step inference much more believable.

For each batch:

* sample `T`
* run the same midblock for `T` steps
* compute endpoint / trajectory / KL losses

This directly supports H3.

---

## 9. Inference-time evaluation steps

Evaluate each iterative model at:

* `T = 1`
* `T = 2`
* `T = 4`
* `T = 8`

Optional:

* `T = 12`

The key result is not just best score. It is the **curve**.

You want plots like:

* x-axis: inference steps
* y-axis: val loss / KL / MMLU-Pro

---

## 10. Evaluation metrics

Use four groups.

### Group A: hidden transport fidelity

* endpoint MSE
* endpoint cosine similarity
* trajectory anchor MSE
* hidden norm drift

### Group B: output fidelity

* token-level KL vs teacher
* top-1 agreement with teacher
* top-k agreement with teacher

### Group C: language/task behavior

* val CE / perplexity on each mix component
* MCQ accuracy on ARC / CSQA / OBQA
* MMLU-Pro accuracy

### Group D: systems metrics

* latency
* tokens/sec
* memory
* wall-clock per token for each `T`

---

## 11. What to use for model selection

Use a **primary validation score**, not just one metric.

I recommend this order:

1. held-out **teacher KL**
2. held-out **endpoint cosine / endpoint loss**
3. held-out **val CE**
4. only then benchmark probes like MMLU-Pro

Reason:
MMLU-Pro is a probe of downstream preservation, not the most direct signal of whether span transport is learned.

---

## 12. Exact ablation matrix

Here is the smallest serious matrix.

### Phase 1: architecture sanity

On **Mix B** only:

* A1 one-shot projector + L2
* A2 recurrent residual + L2
* A3 flow midblock + L2

Evaluate at:

* `T = 1, 2, 4, 8` for iterative models
* normal single pass for projector

Question answered:

* does iteration help?
* does flow-style conditioning help beyond simple recurrence?

### Phase 2: loss ablation

On best architecture from phase 1:

* L1 endpoint only
* L2 endpoint + KL
* L3 endpoint + trajectory + KL
* L4 endpoint + trajectory + KL + CE

Question answered:

* what supervision is actually necessary?

### Phase 3: data mix ablation

On best architecture/loss:

* Mix A
* Mix B
* Mix C

Question answered:

* does cross-regime training improve robustness?

### Phase 4: step-count study

On final selected setup:

* infer at `T = 1, 2, 4, 8, 12`

Question answered:

* is there a usable compute-quality tradeoff?

---

## 13. What a good result looks like

You do **not** need to beat the teacher.

You need something like:

* one-shot projector degrades clearly,
* shared recurrence improves over one-shot,
* FlowMidblock improves further or is more robust across `T`,
* higher `T` improves metrics monotonically or near-monotonically up to a point,
* Mix C preserves MMLU-Pro and chat behavior better than Mix A,
* KL materially helps more than endpoint-only.

That is already a strong result.

---

## 14. What failure patterns mean

### Case 1: endpoint good, MMLU-Pro bad

Interpretation:

* hidden Euclidean matching is not enough
* need stronger KL / trajectory supervision

### Case 2: `T=4` works, `T=8` gets worse

Interpretation:

* rollout is unstable
* step-conditioning or normalization is weak

### Case 3: recurrent residual matches FlowMidblock

Interpretation:

* the real gain is shared iterative computation, not specifically flow-style transport

This is still useful. It just changes the paper claim.

### Case 4: Mix A strong on val loss, weak on MMLU-Pro/chat

Interpretation:

* generic LM transport is preserved, but cross-regime behavior is not
* mixed training is justified

---

## 15. Strongest plots to produce

Make these first.

### Plot 1: quality vs inference steps

For best iterative model:

* x: `T`
* y: val CE / KL / MMLU-Pro

### Plot 2: architecture comparison

Bars or lines for:

* projector
* recurrent residual
* flow midblock

Metrics:

* endpoint cosine
* KL
* MMLU-Pro

### Plot 3: data mix comparison

* Mix A / B / C
* evaluated on plain text, chat, MCQ, MMLU-Pro

### Plot 4: latency vs quality

This one is important for the iterative story.

---

## 16. Recommended execution order

Do this in order:

### Step 1

Run **Mix B** only, because it is balanced and cheaper than full Mix C.

### Step 2

Compare:

* one-shot projector
* recurrent residual
* flow midblock

Use:

* endpoint + KL

### Step 3

If FlowMidblock wins or is competitive, add trajectory loss.

### Step 4

Only after that, run Mix C and MMLU-Pro.

This keeps the loop tight.

---

## 17. My suggested v0.1 default config

If I had to choose one starting config:

* Span: 8→11
* Data: Mix B
* Architecture: FlowMidblock
* Train-time `T`: random in {2,4,6,8}
* Eval-time `T`: 1,2,4,8
* Loss:
  [
  L = 1.0L_{end} + 1.0L_{traj} + 0.5L_{KL}
  ]
* Model selection:

  * first by val KL
  * second by endpoint cosine
  * third by val CE

Then probe:

* ARC family
* MMLU-Pro

---

## 18. The clean story if it works

If results go well, your claim becomes:

> A shared iterative latent transport module can replace a mid-layer transformer span with controllable inference-time refinement, preserving hidden-state transport and downstream behavior better than one-shot compression baselines.