---
# midflowlm-u7k7
title: Plan next MidflowLM experiments after MMLU-Pro CoT failure
status: completed
type: task
priority: normal
created_at: 2026-03-19T09:42:30Z
updated_at: 2026-03-20T05:04:45Z
---

Goal: decide next experiments for the current Qwen3.5-0.8B hidden-state distillation setup after discovering that the student emits <think> tokens on MMLU-Pro instead of direct answer letters.

Checklist:
- [x] Explore project context: docs, eval script, results, recent commits
- [x] Ask one clarifying question about experiment goals and constraints
- [x] Propose 2-3 experimental approaches with trade-offs and recommendation
- [x] Present a concrete experiment design and get approval
- [ ] Write design doc for approved experiment direction
- [ ] Run spec review loop on design doc
- [ ] Ask user to review the written spec
- [x] Transition to implementation planning with writing-plans

## 2026-03-20 follow-up observation

- Free-form MMLU-Pro inference on `results/mmlu_pro_free_inference_best.jsonl` is coherent but low-accuracy: forgiving letter extraction gives 3/24 correct (12.5%).
- 19/24 outputs are verbose despite the prompt requesting a single letter, so this is not just a parser failure.
- Several responses are internally inconsistent (letter/text mismatch or correct letter with wrong explanation), suggesting decoupled fluency vs answer selection.
- The prompt prefix includes an empty `<think></think>` assistant prefill, which is a likely control-format confound to test next.

## 2026-03-20 decoding and training-path follow-up

- Deterministic rerun appended to `results/mmlu_pro_free_inference_best.jsonl` shows no real improvement: subagent analysis found 3/20 correct (15.0%) vs 3/24 (12.5%) originally, with 0/20 strict letter-only responses and much higher latency/length.
- Training-path audit shows KL currently consumes cached `teacher_logits`, CE is not wired through the cache-backed training path, and no on-the-fly teacher recomputation path is implemented yet.
