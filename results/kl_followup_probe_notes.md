# KL Follow-up Probe Notes

Date: 2026-03-22

Artifacts:
- `results/kl_followup_default_512.jsonl`
- `results/kl_followup_closed_think_512.jsonl`

## Trained Midblock Comparison

| Prompt behavior | Rows | Answer hit rate | Median completion length | Max completion length | `<think>` or chat-loop output |
| --- | ---: | ---: | ---: | ---: | --- |
| `default` | 24 | 21/24 (87.5%) | 258.5 | 512 | none observed |
| `closed_think` | 24 | 18/24 (75.0%) | 191.0 | 512 | none observed |

Per-step trained-midblock hit rates were flat within each variant:
- `default`: `7/8` at `num_steps=4`, `8`, and `32`
- `closed_think`: `6/8` at `num_steps=4`, `8`, and `32`

## First-token shifts

- The trained model still starts with prose (`"The"`) on `21/24` rows for both variants.
- `default` uses bare option letters as the first token on the remaining `3/24` rows, always `"D"`.
- `closed_think` also uses bare option letters on `3/24` rows, but it shifts those starts to `"A"` instead of `"D"`.
- The closed-think outputs are shorter on median and more likely to miss extraction despite not changing the dominant prose opening token.

## Looping behavior

- No generated row emitted literal `<think>` text.
- No generated row emitted `<|im_start|>` chat-template loopbacks.
- The earlier long-form reasoning failure persists as long explanatory prose, not as a visible think-tag loop.

## Representative misses

- `default` misses are mostly long explanatory answers that never surface a plain option letter early enough for the current extractor, e.g. text that begins with `The embryological origin of the hyoid bone is the **second pharyngeal arch**...`.
- `closed_think` misses show the same pattern with slightly more direct answer formatting, e.g. `The correct answer is **B**. **Explanation:** ...`; these still miss because the extractor does not currently normalize markdown emphasis.

## Takeaway

`closed_think` does not fix the KL-checkpoint's long-form answer behavior. It lowers trained-midblock answer extraction by `12.5` percentage points (`87.5%` to `75.0%`), shortens completions somewhat, and does not remove the tendency to answer with explanatory prose. Because neither variant showed literal `<think>` looping, the better interpretation is that the failure mode is answer-format drift rather than visible chain-of-thought tag recursion. Keep `default` as the preferred prompt behavior for follow-up evaluation.

## Caveats

- The comparison uses only `8` questions per step and prompt behavior.
- `teacher_original` is not very informative here because it answers in two tokens at `num_steps=1` only.
- `found_valid_answer` depends on regex extraction and currently misses markdown-emphasized answers like `**B**`.
