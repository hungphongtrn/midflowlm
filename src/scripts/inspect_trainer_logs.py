"""Utility to inspect trainer logs and detect duplicate global_step entries.

This helps verify that train metrics/TensorBoard are aligned to optimizer steps
and that validation/checkpointing happen once per optimizer step boundary.

Usage:
    python -m src.scripts.inspect_trainer_logs <log_file>
"""

import re
import sys
from collections import defaultdict
from pathlib import Path


def parse_log_file(log_path: Path):
    step_entries = defaultdict(list)
    current_step = None
    entry_lines = []

    pattern_step = re.compile(r"Step (\d+):")
    pattern_val = re.compile(r"Validation at step (\d+):")
    pattern_best = re.compile(r"New best checkpoint saved.*at step (\d+)")

    for line in log_path.read_text().splitlines():
        m = pattern_step.search(line)
        if m:
            if current_step is not None and entry_lines:
                step_entries[current_step].extend(entry_lines)
            current_step = int(m.group(1))
            entry_lines = [line.strip()]
        elif pattern_val.search(line) or pattern_best.search(line):
            if current_step is not None:
                step_entries[current_step].append(line.strip())
        elif current_step is not None:
            entry_lines.append(line.strip())

    if current_step is not None and entry_lines:
        step_entries[current_step].extend(entry_lines)

    return step_entries


def detect_duplicate_steps(step_entries):
    duplicates = {}
    for step, entries in step_entries.items():
        val_count = sum(
            1
            for e in entries
            if "Validation at step" in e or "New best checkpoint" in e
        )
        if val_count > 1:
            duplicates[step] = val_count
    return duplicates


def analyze_log_file(log_path: Path):
    step_entries = parse_log_file(log_path)

    print(f"\n=== Log Analysis for {log_path} ===")
    print(f"Total unique global_steps: {len(step_entries)}")

    steps_sorted = sorted(step_entries.keys())
    if steps_sorted:
        print(f"Step range: {steps_sorted[0]} to {steps_sorted[-1]}")

        gaps = []
        for i in range(len(steps_sorted) - 1):
            diff = steps_sorted[i + 1] - steps_sorted[i]
            if diff > 1:
                gaps.append((steps_sorted[i], steps_sorted[i + 1], diff - 1))

        if gaps:
            print(f"Skipped steps: {gaps}")
        else:
            print("No gaps in step sequence")

    duplicates = detect_duplicate_steps(step_entries)
    if duplicates:
        print(f"\nWARNING: Duplicate validation/checkpoint entries found:")
        for step, count in duplicates.items():
            print(f"  Step {step}: {count} validation/checkpoint events")
    else:
        print("\nNo duplicate validation/checkpoint entries detected")

    return step_entries


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.scripts.inspect_trainer_logs <log_file>")
        sys.exit(1)

    log_path = Path(sys.argv[1])
    if not log_path.exists():
        print(f"Error: Log file not found: {log_path}")
        sys.exit(1)

    analyze_log_file(log_path)
