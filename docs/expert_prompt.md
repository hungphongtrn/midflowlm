# Expert Discussion Prompt

I am working on an experimental project called MidflowLM. The core idea is to replace a span of frozen Qwen layers with a trainable iterative hidden-state refiner, while keeping the rest of the base model frozen.

Current setup:

- Base model: `Qwen/Qwen3.5-0.8B`
- Replacement span: layers `8..11`
- Student wrapper: `src/model/student_qwen.py`
- Step conditioning: `src/model/adapter.py`
- Config: `configs/v0_onemotif.yaml`
- Dataset: `roneneldan/TinyStories`
- Training mode so far: architecture training from cached teacher hidden states
- Cache policy: hidden-state-only by default, no full teacher logits in the offline cache
- Trained refinement range: `max_steps_T = 8`, with train values `[1, 2, 4, 6, 8]`
- Losses currently active: endpoint + trajectory matching only (`kl_weight=0`, `ce_weight=0`)

Recent status:

- I trained a checkpoint saved at `outputs/v0_qwen_iterative_midblock/checkpoints/final.ckpt`
- I added a qualitative sweep script that compares:
  - original frozen/non-replaced model output
  - trained model output at different `num_steps`
- The sweep output is in `outputs/text_sweep.json`

Recent qualitative findings:

- For a story-style prompt, `num_steps=8` is the best among `1/4/8`, but it is still generic and somewhat repetitive
- For a robot-themed prompt, both the original model and trained model collapse into repeated sentence loops
- When I push `num_steps` beyond the trained range to `32` and `64`, outputs degrade badly into junk/repeated/multilingual token loops
- This makes sense because the step-conditioning adapter clamps the discrete step embedding at `max_steps_T-1`, so larger `num_steps` are effectively extrapolative
- Generations often run all the way to the token budget instead of stopping naturally, suggesting decoding/repetition issues are a major factor

What I think is happening:

1. Hidden-state architecture training is sufficient to get some coherent generations
2. It is not sufficient to reliably improve behavior quality
3. The current bottleneck may be a mix of:
   - repetition from greedy decoding
   - lack of behavioral training objectives
   - mismatch between hidden-state alignment quality and text-generation quality
4. Increasing `num_steps` beyond the trained range is not a valid way to improve quality in the current design

Questions I want expert feedback on:

1. Given this setup, what is the most promising next step: better decoding, behavior distillation (KL/CE), preference optimization, or architecture changes?
2. Is hidden-state-only training likely to plateau qualitatively unless I add an explicit behavior objective?
3. Is the current step-conditioning design fundamentally limiting generalization to larger `num_steps`? If so, what conditioning design would you recommend?
4. How would you structure evaluation so that I can separate:
   - hidden-state imitation quality
   - language modeling quality
   - repetition/degeneration failure modes
5. If the goal is better generation quality rather than only hidden-state matching, what training objective would you try first and why?

Constraints and preferences:

- I want to keep most of the base Qwen model frozen if possible
- I prefer reuse of HF Qwen internals rather than custom transformer reimplementation
- I want experiments that are practical on limited compute
- I would like recommendations that distinguish fast validation steps from larger architecture changes

Please respond with:

1. your diagnosis of the current failure mode,
2. the top 3 next experiments you would run in order,
3. what success criteria I should use for each,
4. whether you think behavior training is now necessary.
