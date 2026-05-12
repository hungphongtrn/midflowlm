#!/usr/bin/env python3
"""Prepare GLM-5.1-Reasoning-1M-Cleaned for SFT training.

Usage:
    python scripts/prepare_reasoning_sft_data.py \
        --max_length 8192 \
        --num_proc 4 \
        --max_train_samples 100000 \
        --output_dir ./data/reasoning_sft \
        [--smoke_test]
"""

import argparse
import os
import time

from datasets import load_dataset
from transformers import AutoTokenizer

from src.data.reasoning_sft import create_reasoning_sft_datasets

DATASET_NAME = "Jackrong/GLM-5.1-Reasoning-1M-Cleaned"
MODEL_NAME = "Qwen/Qwen3.5-0.8B"
IGNORE_INDEX = -100
SMOKE_TEST_SAMPLE_LIMIT = 1000
SMOKE_TEST_DEFAULT_TRAIN = 500


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare reasoning SFT dataset")
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--num_proc", type=int, default=4)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=200)
    parser.add_argument("--val_split", type=float, default=0.02)
    parser.add_argument("--output_dir", type=str, default="./data/reasoning_sft")
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Use only 1000 samples for quick testing",
    )
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    t0 = time.time()

    print("=" * 80)
    print("PREPARING REASONING SFT DATASET")
    print("=" * 80)
    print(f"  Dataset: {DATASET_NAME}")
    print(f"  Model:   {MODEL_NAME}")
    print(f"  Max length: {args.max_length}")
    print(f"  Workers: {args.num_proc}")
    print(f"  Smoke test: {args.smoke_test}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Tokenizer vocab_size: {tokenizer.vocab_size}")
    print(f"  pad_token: {tokenizer.pad_token!r}  ({tokenizer.pad_token_id})")
    print(f"  eos_token: {tokenizer.eos_token!r}  ({tokenizer.eos_token_id})")

    print("\nLoading dataset from HuggingFace Hub...")
    ds = load_dataset(DATASET_NAME, split="train")
    print(f"  Loaded {len(ds):,} raw samples")

    if args.smoke_test:
        ds = ds.select(range(min(SMOKE_TEST_SAMPLE_LIMIT, len(ds))))
        args.max_train_samples = min(args.max_train_samples or SMOKE_TEST_DEFAULT_TRAIN, len(ds))
        print(f"  [SMOKE TEST] Subset to {len(ds)} samples")

    train_ds, eval_ds = create_reasoning_sft_datasets(
        ds,
        tokenizer,
        max_length=args.max_length,
        num_proc=args.num_proc,
        val_split=args.val_split,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        seed=args.seed,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train")
    eval_path = os.path.join(args.output_dir, "eval")

    print(f"\nSaving datasets to {args.output_dir}/...")
    train_ds.save_to_disk(train_path)
    eval_ds.save_to_disk(eval_path)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Train: {train_path} ({len(train_ds):,} sequences)")
    print(f"  Eval:  {eval_path} ({len(eval_ds):,} sequences)")
    print("\nSample token budgets (train):")
    for i in range(min(3, len(train_ds))):
        ids = train_ds[i]["input_ids"]
        labels = train_ds[i]["labels"]
        active = sum(1 for label in labels if label != IGNORE_INDEX)
        total_tokens = len(ids)
        print(f"  [{i}] total_tokens={total_tokens}, active_labels={active}")


if __name__ == "__main__":
    main()
