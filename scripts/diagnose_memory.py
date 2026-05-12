#!/usr/bin/env python3
"""Deterministic VRAM diagnosis harness for issue #12."""

import json
import os
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.sft_flow_midblock import SFTFlowMidblockModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_mb(num_bytes: int) -> float:
    return round(num_bytes / (1024 * 1024), 2)


def run_one(model, seq_len: int, vocab_size: int, device: torch.device):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.reset_accumulated_memory_stats(device)

    input_ids = torch.randint(0, vocab_size, (1, seq_len), device=device, dtype=torch.long)
    attention_mask = torch.ones((1, seq_len), device=device, dtype=torch.long)

    out = {"seq_len": seq_len, "ok": True}
    try:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        stats = torch.cuda.memory_stats(device)
        out["peak_allocated_mb"] = to_mb(torch.cuda.max_memory_allocated(device))
        out["peak_reserved_mb"] = to_mb(torch.cuda.max_memory_reserved(device))
        out["active_peak_mb"] = to_mb(int(stats.get("active_bytes.all.peak", 0)))
        out["allocated_peak_mb"] = to_mb(int(stats.get("allocated_bytes.all.peak", 0)))
        out["num_alloc_retries"] = int(stats.get("num_alloc_retries", 0))
        out["num_ooms"] = int(stats.get("num_ooms", 0))
    except torch.cuda.OutOfMemoryError as exc:
        out["ok"] = False
        out["oom_error"] = str(exc)
        out["memory_summary"] = torch.cuda.memory_summary(device=device, abbreviated=True)
    finally:
        model.zero_grad(set_to_none=True)

    return out


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for diagnose_memory.py")

    seed = 1337
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    device = torch.device("cuda")
    model = SFTFlowMidblockModel(
        model_name="Qwen/Qwen3.5-0.8B",
        start_layer=8,
        end_layer=11,
        thinking_level=32,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.train()

    seq_lengths = [512, 768, 1024, 1280, 1536, 1792, 2048]
    results = []
    first_oom = None
    vocab_size = int(model.embed_tokens.num_embeddings)
    for seq_len in seq_lengths:
        result = run_one(model, seq_len, vocab_size, device)
        results.append(result)
        if (not result["ok"]) and first_oom is None:
            first_oom = seq_len
            break

    report = {
        "issue": 12,
        "seed": seed,
        "device": torch.cuda.get_device_name(device),
        "thinking_level": 32,
        "results": results,
        "first_oom_seq_len": first_oom,
    }

    out_arg = sys.argv[1] if len(sys.argv) > 1 else "reports/issue-12-memory-report.json"
    out_path = Path(out_arg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote report: {out_path}")
    if first_oom is None:
        print("No OOM detected in tested seq lengths")
    else:
        print(f"OOM detected at seq_len={first_oom}")


if __name__ == "__main__":
    main()
