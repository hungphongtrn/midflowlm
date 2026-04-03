# v0.1 Experiment Matrix.

## 1. Core matrix

### Phase 1 — architecture sanity

Use **Mix B** only.

| Exp ID | Model                           |     Loss |          Train T |  Eval T | Purpose                             |
| ------ | ------------------------------- | -------: | ---------------: | ------: | ----------------------------------- |
| P1-A1  | One-shot projector              | End + KL |                1 |       1 | Direct compression baseline         |
| P1-A2  | Shared recurrent residual block | End + KL | {2,4,6,8} random | 1,2,4,8 | Tests whether iteration alone helps |
| P1-A3  | Flow midblock                   | End + KL | {2,4,6,8} random | 1,2,4,8 | Main method                         |

Decision rule for Phase 1:

* pick top 2 by **val KL**
* break ties with **endpoint cosine**
* then check **val CE**

---

### Phase 2 — loss ablation

Use best architecture from Phase 1, still on **Mix B**.

| Exp ID |                 Loss |   Train T |  Eval T | Purpose                                 |
| ------ | -------------------: | --------: | ------: | --------------------------------------- |
| P2-L1  |             End only | {2,4,6,8} | 1,2,4,8 | Tests whether hidden endpoint is enough |
| P2-L2  |             End + KL | {2,4,6,8} | 1,2,4,8 | Teacher-behavior preservation           |
| P2-L3  |      End + Traj + KL | {2,4,6,8} | 1,2,4,8 | Tests whether trajectory anchors help   |
| P2-L4  | End + Traj + KL + CE | {2,4,6,8} | 1,2,4,8 | Tests whether task CE adds value        |

Decision rule for Phase 2:

* pick best by **val KL**
* verify it does not hurt **val CE**
* inspect whether larger eval `T` gives a smoother curve

---

### Phase 3 — data mix ablation

Use best model and best loss.

| Exp ID | Data Mix                    |   Train T |  Eval T | Purpose                   |
| ------ | --------------------------- | --------: | ------: | ------------------------- |
| P3-D1  | Mix A = FineWeb only        | {2,4,6,8} | 1,2,4,8 | Generic LM transport only |
| P3-D2  | Mix B = FineWeb + UltraChat | {2,4,6,8} | 1,2,4,8 | Adds instruction behavior |
| P3-D3  | Mix C = Full mix            | {2,4,6,8} | 1,2,4,8 | Cross-regime preservation |

Decision rule for Phase 3:

* compare per-regime robustness, not just one global number

---

### Phase 4 — final probing

Use final selected setup.

| Exp ID |                    Probe |     Eval T | Purpose                           |
| ------ | -----------------------: | ---------: | --------------------------------- |
| P4-E1  |              FineWeb val | 1,2,4,8,12 | PPL / CE stability                |
| P4-E2  |            UltraChat val | 1,2,4,8,12 | Chat behavior preservation        |
| P4-E3  |    ARC / CSQA / OBQA val | 1,2,4,8,12 | Structured reasoning preservation |
| P4-E4  |                 MMLU-Pro | 1,2,4,8,12 | Hard external probe               |
| P4-E5  | Latency / tok/s / memory | 1,2,4,8,12 | Quality vs compute curve          |

---

## 2. Exact data mixes

### Mix A

* FineWeb-Edu only

### Mix B

* FineWeb-Edu
* UltraChat SFT

### Mix C

* FineWeb-Edu
* UltraChat SFT
* ARC-Challenge
* ARC-Easy
* CommonsenseQA
* OpenBookQA

---

## 3. Default loss definitions

Let:

* `h7` = hidden after layer 7
* `h8,h9,h10,h11` = teacher anchors
* `ĥ11` = predicted end state
* `p_t, p_s` = teacher/student logits

### Endpoint loss

[
L_{end} = |\text{Norm}(\hat h_{11}) - \text{Norm}(h_{11})|_2^2
]

### Trajectory loss

[
L_{traj} = \sum_{i \in {8,9,10,11}} w_i |\text{Norm}(\hat h_i) - \text{Norm}(h_i)|_2^2
]

Good default:

* `w8=0.2, w9=0.3, w10=0.5, w11=1.0`

### KL loss

[
L_{KL} = KL(p_t | p_s)
]

### CE loss

Standard next-token cross-entropy.

### Recommended default weights

Start with:
[
L = 1.0L_{end} + 1.0L_{traj} + 0.5L_{KL}
]

Then test:
[
L + 0.1L_{CE}
]

---

## 4. Metrics table

| Group           | Metric                   | Use                                   |
| --------------- | ------------------------ | ------------------------------------- |
| Hidden fidelity | endpoint MSE             | direct transport quality              |
| Hidden fidelity | endpoint cosine          | geometry preservation                 |
| Hidden fidelity | trajectory anchor MSE    | whether refinement path is meaningful |
| Output fidelity | token KL                 | best teacher-faithfulness metric      |
| Output fidelity | top-1/top-k agreement    | easy behavior sanity                  |
| Language        | val CE / PPL             | generic LM preservation               |
| Task            | ARC / CSQA / OBQA acc    | reasoning-style stress test           |
| Task            | MMLU-Pro acc             | hard external probe                   |
| Systems         | latency / tok/s / memory | quality-compute tradeoff              |

---

## 5. Recommended success criteria

You do not need to beat teacher. You need this pattern:

### Minimum success

* iterative models beat one-shot projector on **val KL**
* best model shows useful improvement from `T=1` to `T=4`
* no catastrophic collapse on MMLU-Pro

### Strong success

* Flow midblock beats shared recurrent residual block
* Mix C preserves MMLU-Pro and MCQ better than Mix A/B
* quality improves with `T` at least up to 4 or 8
* latency-quality curve is clearly usable

### Important negative but still publishable

* shared recurrent residual block matches or beats Flow midblock
  This would mean:
* the key insight is **iterative shared computation**, not specifically flow-style transport

That is still a good result.

---

## 6. Recommended execution order

Do this exact order:

1. **P1-A1 / A2 / A3** on Mix B
2. pick best architecture
3. **P2-L1..L4**
4. pick best loss
5. **P3-D1..D3**
6. final `T` sweep + MMLU-Pro

This keeps cost under control.

---

## 7. My recommended starting point

Start with this single config first:

| Item             | Choice                |
| ---------------- | --------------------- |
| Span             | 8→11                  |
| Data             | Mix B                 |
| Model            | Flow midblock         |
| Train T          | random from {2,4,6,8} |
| Eval T           | 1,2,4,8               |
| Loss             | End + Traj + KL       |
| Selection metric | val KL                |
| Main probe       | MMLU-Pro              |

---

## 8. Suggested run naming scheme

Use something like:

`midflow_qwen_8to11_[model]_[mix]_[loss]_trainT-r2468`

Examples:

* `midflow_qwen_8to11_proj_mixB_endkl`
* `midflow_qwen_8to11_rrb_mixB_endtrajkl_trainT-r2468`
* `midflow_qwen_8to11_flow_mixC_endtrajklce_trainT-r2468`

This will save you a lot of pain later.

---

## 9. The one figure to prioritize

If you only make one key figure early, make this:

**x-axis:** eval steps `T`
**y-axis:**

* val KL
* val CE
* MMLU-Pro accuracy

for:

* one-shot projector
* recurrent residual block
* flow midblock

That single figure tells most of the story.