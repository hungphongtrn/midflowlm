# MMLU-Pro Downstream Task Evaluation Results

**Dataset:** TIGER-Lab/MMLU-Pro (validation split)  
**Samples:** 70 questions  
**Model:** Qwen/Qwen3.5-0.8B  
**Checkpoint:** outputs/11-12-19-03-2026-v0_qwen_iterative_midblock/checkpoints/best.ckpt  
**Evaluation Date:** 2026-03-19

## Results Summary

| Model | Steps (T) | Accuracy | Correct / Total | Avg Latency (ms) |
|-------|-----------|----------|-----------------|------------------|
| trained_midblock | 1 | **0.00%** | 0 / 70 | 95.39 |
| trained_midblock | 4 | **0.00%** | 0 / 70 | 94.46 |
| trained_midblock | 8 | **0.00%** | 0 / 70 | 103.14 |
| trained_midblock | 32 | **0.00%** | 0 / 70 | 154.13 |
| teacher_original | 1 | **17.14%** | 12 / 70 | 30.93 |

## Critical Finding: Output Token Analysis

### Student Model Outputs
The student model outputs token **248068** which maps to the Chinese character **"思考"** (thinking), and token **248069** which is **"】"** (end thinking).

```
Token 248068 → "思考" (thinking)
Token 248069 → "】" (end thinking bracket)
```

These are Qwen's **chain-of-thought (CoT) tokens**. The model is attempting to reason/think before answering rather than directly outputting the answer letter.

### Teacher Model Outputs
The teacher correctly outputs answer tokens:
```
Token 32 → "A"
Token 33 → "B"
Token 34 → "C"
...
```

## Root Cause Analysis

The student model checkpoint was likely trained with:
1. **CoT-style prompting** - expecting the model to think before answering
2. **Different prompt format** - the training objective may have been different from direct QA

The midblock appears to be *correctly producing reasoning tokens* according to how it was trained, but this conflicts with our direct-answer evaluation format.

## Recommendations

1. **Modify prompt format** to include thinking tokens if the model expects them:
   ```text
   <|im_start|>system
   Think step by step, then answer with only the letter.
   <|im_end|>
   ```

2. **Continue generation** past the thinking tokens to get the actual answer:
   - Instead of `logits[:, -1, :]` for next token after prompt
   - Continue generating tokens until we see an answer letter (A-J)

3. **Use teacher-style prompt** if the midblock was trained to match teacher behavior at specific steps

4. **Analyze checkpoint training config** to understand what behavior was expected

## Example Inputs/Outputs

### Prompt Template (with special tokens)
```
<|im_start|>system
You are a helpful assistant that answers multiple choice questions. Respond with only the letter of the correct answer.<|im_end|>
<|im_start|>user
Answer the following multiple choice question. Respond with only the letter of the correct answer (A, B, C, etc.).

Question: A microwave oven is connected to an outlet, 120 V, and draws a current of 2 amps. At what rate is energy being used by the microwave oven?

Options:
A. 240 W
B. 120 W
...
J. 300 W

Answer:<|im_end|>
<|im_start|>assistant
```

### Student Model Response
```
Token: 248068 → "思考"
```

### Teacher Model Response
```
Token: 32 → "A" (correct answer)
```

## Configuration

```yaml
config: configs/v0_onemotif.yaml
checkpoint: outputs/11-12-19-03-2026-v0_qwen_iterative_midblock/checkpoints/best.ckpt
num_samples: 70
seed: 42
device: cuda
num_steps: [1, 4, 8, 32]
```

## Raw Results (JSON)

See `mmlu_pro_eval.json` for:
- Full prompt text with special tokens
- Token IDs for prompts
- Raw output token ID
- Raw output text (with special tokens)
- Detailed per-question results