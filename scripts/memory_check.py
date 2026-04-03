#!/usr/bin/env python3
"""Quick memory check for gradient flow fix."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.model.student_qwen import FrozenQwenStudent
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

if device == "cpu":
    print("CPU mode - memory measurement not applicable")
    exit(0)

student = FrozenQwenStudent(
    model_name="Qwen/Qwen3.5-0.8B",
    start_layer=8,
    end_layer=11,
    max_steps_T=4,
    device=device,
)

torch.cuda.reset_peak_memory_stats()

# Forward + backward
input_ids = torch.randint(0, 1000, (2, 64), device=device)
attention_mask = torch.ones(2, 64, device=device)

logits = student(input_ids, attention_mask, num_steps=4)
labels = input_ids[:, 1:]
pred_logits = logits[:, :-1, :]
ce_loss = F.cross_entropy(
    pred_logits.reshape(-1, pred_logits.size(-1)), labels.reshape(-1)
)
ce_loss.backward()

peak_memory = torch.cuda.max_memory_allocated() / 1e9
print(f"Peak GPU memory: {peak_memory:.2f} GB")

# Compare with baseline (document what baseline should be)
# Note: Memory will increase because we now track gradients through frozen layers
