#!/usr/bin/env python3
"""Inspect the schema and sample records of GLM-5.1-Reasoning-1M-Cleaned."""

import sys
from datasets import load_dataset

DATASET_NAME = "Jackrong/GLM-5.1-Reasoning-1M-Cleaned"


def main():
    ds = load_dataset(DATASET_NAME, split="train", streaming=True)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Split: train (streaming)")

    count = 0
    for example in ds:
        count += 1
        if count <= 3:
            print(f"\n{'='*60}")
            print(f"Example #{count}:")
            print(f"  Keys: {list(example.keys())}")
            for key in example:
                val = example[key]
                if isinstance(val, dict):
                    print(f"  {key}: (dict) keys={list(val.keys())}")
                    for k, v in val.items():
                        v_str = str(v)
                        if len(v_str) > 200:
                            v_str = v_str[:200] + "..."
                        print(f"    {k}: {v_str}")
                elif isinstance(val, list):
                    print(f"  {key}: (list, len={len(val)})")
                    if len(val) > 0 and isinstance(val[0], dict):
                        print(f"    First item keys: {list(val[0].keys())}")
                        item_str = str(val[0])
                        if len(item_str) > 300:
                            item_str = item_str[:300] + "..."
                        print(f"    First item: {item_str}")
                else:
                    v_str = str(val)
                    if len(v_str) > 200:
                        v_str = v_str[:200] + "..."
                    print(f"  {key}: {v_str}")
        if count >= 5:
            break

    # Summary
    sample_count = 0
    total_input_tokens = 0
    total_output_tokens = 0
    has_meta = False
    num_over_budget = 0

    for example in ds:
        sample_count += 1
        if "meta" in example and isinstance(example["meta"], dict):
            has_meta = True
            it = example["meta"].get("input_tokens", 0) or 0
            ot = example["meta"].get("output_tokens", 0) or 0
            total_input_tokens += it
            total_output_tokens += ot
            if it + ot > 8192:
                num_over_budget += 1
        if sample_count >= 1000:
            break

    print(f"\n{'='*60}")
    print(f"Summary (first {sample_count} samples):")
    print(f"  Has 'meta' field: {has_meta}")
    if has_meta:
        print(f"  Mean input_tokens:  {total_input_tokens / sample_count:.0f}")
        print(f"  Mean output_tokens: {total_output_tokens / sample_count:.0f}")
        print(f"  Over 8192 budget:   {num_over_budget}/{sample_count} ({100*num_over_budget/sample_count:.1f}%)")


if __name__ == "__main__":
    main()
