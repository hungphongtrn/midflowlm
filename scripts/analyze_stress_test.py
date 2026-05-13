#!/usr/bin/env python3
"""Analyze P3-D3 stress test results across benchmarks."""

import argparse
import csv
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Any


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


def load_json_results(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        data = json.load(f)
    return data.get("results", data) if isinstance(data, dict) else data


def analyze_benchmark(benchmark_name: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    student_results = [r for r in results if r.get("model_name") == "trained_midblock"]
    teacher_results = [r for r in results if r.get("model_name") == "teacher_original"]

    student_sorted = sorted(student_results, key=lambda r: r.get("num_steps", 0))
    teacher_acc = None
    if teacher_results:
        tr = teacher_results[0]
        if "mc1_accuracy" in tr:
            teacher_acc = tr["mc1_accuracy"]
        elif "accuracy" in tr:
            teacher_acc = tr["accuracy"]

    t_values = []
    accuracies = []
    latencies = []
    for r in student_sorted:
        t_values.append(r.get("num_steps", 0))
        if "mc1_accuracy" in r:
            accuracies.append(r.get("mc1_accuracy", 0))
        elif "accuracy" in r:
            accuracies.append(r.get("accuracy", 0))
        else:
            accuracies.append(0)
        latencies.append(r.get("avg_latency_ms", 0))

    answer_distribution = {}
    for r in student_sorted:
        T = r.get("num_steps", 0)
        if "detailed_results" in r:
            preds = [d.get("predicted_answer", "INVALID") for d in r["detailed_results"]]
            answer_distribution[T] = Counter(preds)

    return {
        "benchmark": benchmark_name,
        "teacher_accuracy": teacher_acc,
        "t_scaling": {
            "T": t_values,
            "accuracy": accuracies,
            "latency_ms": latencies,
        },
        "answer_distribution": {str(k): dict(v.most_common(20)) for k, v in answer_distribution.items()},
        "best_t": t_values[accuracies.index(max(accuracies))] if accuracies else None,
        "best_accuracy": max(accuracies) if accuracies else 0,
        "t1_accuracy": accuracies[0] if accuracies and len(accuracies) > 0 else 0,
    }


def print_t_scaling_table(benchmark_analyses: Dict[str, Dict[str, Any]]):
    print("\n" + "=" * 100)
    print("T-SCALING TABLE: Accuracy per Benchmark per T value")
    print("=" * 100)

    benchmarks = sorted(benchmark_analyses.keys())
    all_T = set()
    for ba in benchmark_analyses.values():
        all_T.update(ba["t_scaling"]["T"])
    all_T = sorted(all_T)

    header = f"{'T':>4} | " + " | ".join(f"{b:>12}" for b in benchmarks)
    print(header)
    print("-" * len(header))

    for T in all_T:
        row = f"{T:>4} | "
        for b in benchmarks:
            ts = benchmark_analyses[b]["t_scaling"]
            if T in ts["T"]:
                idx = ts["T"].index(T)
                row += f"{ts['accuracy'][idx]:>11.2%} | "
            else:
                row += f"{'N/A':>11} | "
        print(row)

    print("=" * 100)


def print_answer_distribution(analyses: Dict[str, Dict[str, Any]]):
    print("\n" + "=" * 100)
    print("ANSWER DISTRIBUTION ANALYSIS")
    print("=" * 100)
    for b_name, analysis in sorted(analyses.items()):
        print(f"\n--- {b_name} ---")
        for T, dist in sorted(analysis.get("answer_distribution", {}).items(), key=lambda x: int(x[0])):
            total = sum(dist.values())
            top5 = list(dist.items())[:5]
            dist_str = " | ".join(f"{letter}:{count}({count/total:.1%})" for letter, count in top5)
            print(f"  T={T:>3}: {dist_str}")


def save_summary_csv(analyses: Dict[str, Dict[str, Any]], output_path: str):
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "benchmark", "T", "accuracy", "latency_ms",
            "teacher_accuracy", "gap_vs_teacher", "is_best",
        ])
        for b_name, analysis in sorted(analyses.items()):
            ts = analysis["t_scaling"]
            teacher = analysis.get("teacher_accuracy")
            best_t = analysis.get("best_t")
            for i in range(len(ts["T"])):
                gap = (ts["accuracy"][i] - teacher) if teacher is not None else 0
                writer.writerow([
                    b_name,
                    ts["T"][i],
                    f"{ts['accuracy'][i]:.6f}",
                    f"{ts['latency_ms'][i]:.2f}",
                    f"{teacher:.6f}" if teacher is not None else "",
                    f"{gap:.6f}",
                    "YES" if ts["T"][i] == best_t else "",
                ])

    logging.getLogger(__name__).info(f"Summary CSV saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze P3-D3 stress test results")
    parser.add_argument("--results-dir", type=str, default="results/stress_test")
    parser.add_argument("--csv-output", type=str, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logger = setup_logging(args.log_level)

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        logger.error(f"Results directory {results_dir} does not exist")
        return

    analyses = {}
    for json_file in sorted(results_dir.glob("*_results.json")):
        benchmark_name = json_file.stem.replace("_results", "")
        results = load_json_results(str(json_file))
        analyses[benchmark_name] = analyze_benchmark(benchmark_name, results)

    print_t_scaling_table(analyses)
    print_answer_distribution(analyses)

    print("\n" + "=" * 100)
    print("PER-BENCHMARK SUMMARY")
    print("=" * 100)
    print(f"{'Benchmark':>20} | {'Teacher':>8} | {'Best T':>6} | {'Best Acc':>8} | {'T=1 Acc':>8} | {'Peak Gain':>10}")
    print("-" * 85)
    for b_name, analysis in sorted(analyses.items()):
        teacher = analysis.get("teacher_accuracy")
        t1 = analysis.get("t1_accuracy", 0)
        gain = analysis["best_accuracy"] - t1
        print(
            f"{b_name:>20} | "
            f"{teacher:>7.2%} | "
            f"{analysis['best_t']:>6} | "
            f"{analysis['best_accuracy']:>7.2%} | "
            f"{t1:>7.2%} | "
            f"{gain:>+9.2%}"
        )

    if args.csv_output:
        save_summary_csv(analyses, args.csv_output)

    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()
