# KL Follow-up Inference Probe Notes
**Generated:** 2026-03-22  
**Artifacts:** `results/kl_followup_default_512.jsonl`, `results/kl_followup_closed_think_512.jsonl`  
**Probe config:** 8 samples × {1,4,8,32} steps, max_new_tokens=512, temp=1.0, top_p=0.95  
**Models:** `trained_midblock` (24 rows) + `teacher_original` (8 rows) per file  

---

## 1. Answer Extraction Hit Rate

| Variant | trained_midblock | teacher_original | Overall |
|---------|:-----------------:|:-----------------:|:-------:|
| default | 21/24 (87.5%) | 8/8 (100%) | 29/32 (90.6%) |
| closed_think | 18/24 (75.0%) | 8/8 (100%) | 26/32 (81.2%) |

**closed_think costs ~12.5 pp on trained_midblock.** teacher_original is a degenerate baseline (2-token completions) and is not informative.

---

## 2. Completion Length — trained_midblock only

| Variant | Median | Max |
|---------|-------:|----:|
| default | 258 | 512 |
| closed_think | 191 | 512 |

Shorter median under closed_think is consistent with the model spending tokens re-stating the question rather than producing a direct answer.

---

## 3. Looping `<think>` Behavior

**No looping detected in either variant.** Zero generations contain any `<think>` / `</think>` token sequence in the output, and zero generations contain a stray `<|im_start|>` (prompt-loopback). Truncation at 512 tokens occurs in both variants but is a hard stop, not a loop.

---

## 4. Notable First-Token Shifts

| First token | default | closed_think |
|-------------|--------:|-----------:|
| "The" | 21 | 21 |
| "A" | 2 | **7** |
| "B" | 3 | 0 |
| "C" | 2 | 0 |
| "D" | 4 | 0 |
| "F" | 0 | 1 |
| "E" | 0 | 1 |

`closed_think` shifts the first token toward bare option letters ("A", "F", "E") rather than prose ("The"), suggesting the closed-think prefill biases the model toward option-recitation mode rather than free-text reasoning. This correlates with the higher extraction failure rate: in sample 0 (default: len=512, extracts **B** ✓) vs closed_think: len=218, extracts **B** but correct answer is **F** — the model is listing option labels without committing to the right one.

---

## 5. Qualitative Failure Examples

**closed_think wrong-option flip (sample 0):**  
```
default:  "The correct answer is **B**."  (correct, len=512 truncated)
closed:   "The correct answer is **B**."  (extracts B; correct=F, stopped at 218 tokens)
```
The model states an answer but is factually wrong. The closed-think preamble seems to have biased it toward a wrong option early.

**closed_think non-answer preamble (sample 4):**  
```
closed: "A. Female catheters are used more frequently than male catheters.\n\nB. Male catheters are bigger..."  
```
First token shifts to "A" — the model immediately enters option enumeration mode without stating a final answer, exhausting the 512-token budget before producing a extractable answer letter.

---

## 6. Caveats

- **Sample size is small** (8 samples × 3 step-counts = 24 trained_midblock rows per variant). Differences of 1–2 extractions (87.5% vs 75%) are suggestive but not statistically significant.
- **teacher_original is a degenerate baseline** producing only 2 tokens (letter + EOS); its 100% hit rate is meaningless for comparison.
- **`num_steps` is conflated with model identity**: teacher_original uses only `steps=1` while trained_midblock uses {4,8,32}. Direct step-count comparisons across models are not valid.
- **Answer extraction** (`found_valid_answer`) is regex-based; it may miss cases where the model gives correct text answer but with slightly different formatting.
- **No visible `<think>` in outputs** does not prove absence of internal thought iterations — it only shows the model does not emit `<think>` tokens in the final output.
