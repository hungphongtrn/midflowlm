---
# midflowlm-wue9
title: 'Task 6: Wire Loss-Conditional Extraction into Trainer'
status: completed
type: task
created_at: 2026-04-03T16:35:44Z
updated_at: 2026-04-03T16:35:44Z
---

Wires the conditional target extraction into the trainer so that it actually gets used during training.

- Add _get_loss_flags() method to Trainer
- Update train_step() and val_step() to pass flags to extract_teacher_targets()
- Enables memory savings when certain losses are disabled
