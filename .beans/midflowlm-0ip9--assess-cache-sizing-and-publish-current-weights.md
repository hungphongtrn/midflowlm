---
# midflowlm-0ip9
title: Assess cache sizing and publish current weights
status: in-progress
type: task
priority: normal
created_at: 2026-03-23T03:45:39Z
updated_at: 2026-03-23T04:26:08Z
---

Handle three requests:
- [x] Check ability to publish the current outputs checkpoint to Hugging Face repo hungphongtrn/midflowlm-23-Mar-26\n  - [x] Create the repo\n  - [x] Upload all checkpoint files under outputs\n  - [x] Verify uploaded files
- [x] Compute a context length covering at least 80% of samples in the current mixed dataset
- [x] Estimate cache disk usage for hidden states and logits at that context length
