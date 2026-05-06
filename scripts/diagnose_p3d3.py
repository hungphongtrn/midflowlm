#!/usr/bin/env python
"""P3-D3 diagnostic probe and report generation.

This script performs diagnostic analysis on the P3-D3 checkpoint to investigate
flat T-scaling behavior. It supports two modes:

1. Capture mode (default): Select probes, run model, save traces to JSON/CSV
2. Report mode (--report): Generate markdown diagnostic report from saved traces

Usage:
    # Full pipeline: capture + report
    python scripts/diagnose_p3d3.py --checkpoint ... --config ... --mmlu-path ... --arc-path ... --report

    # Report only from saved traces (fast, no model)
    python scripts/diagnose_p3d3.py --report --traces-dir results/diagnostic_p3d3/traces

    # Capture only (existing behavior)
    python scripts/diagnose_p3d3.py --checkpoint ... --config ... --mmlu-path ... --arc-path ...
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _do_capture(args):
    import torch

    from src.diagnostic.probe import ProbeSet, select_probes
    from src.diagnostic.runner import DeterministicTraceRunner, load_model_from_checkpoint

    checkpoint_path = Path(args.checkpoint)
    config_path = Path(args.config)
    mmlu_path = Path(args.mmlu_path)
    arc_path = Path(args.arc_path)

    for path, name in [
        (checkpoint_path, "checkpoint"),
        (config_path, "config"),
        (mmlu_path, "mmlu-path"),
        (arc_path, "arc-path"),
    ]:
        if not path.exists():
            print(f"Error: {name} not found: {path}", file=sys.stderr)
            sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    probes = select_probes(
        mmlu_pro_path=str(mmlu_path),
        arc_easy_path=str(arc_path),
    )
    probe_set = ProbeSet(
        probes=probes,
        checkpoint_source=str(checkpoint_path),
        seed=args.seed,
    )

    probes_file = output_dir / "probes.json"
    with open(probes_file, "w") as f:
        json.dump(probe_set.to_dict(), f, indent=2)
    print(f"Selected {len(probes)} probes -> {probes_file}")

    mmlu_count = sum(1 for p in probes if p.benchmark == "mmlu_pro")
    arc_count = sum(1 for p in probes if p.benchmark == "arc_easy")
    print(f"  - MMLU-Pro: {mmlu_count}")
    print(f"  - ARC-Easy: {arc_count}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    try:
        model, tokenizer = load_model_from_checkpoint(
            checkpoint_path=str(checkpoint_path),
            config_path=str(config_path),
            device=device,
        )
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    for probe in probes:
        if probe.input_ids is None or len(probe.input_ids) == 0:
            if probe.prompt_text:
                probe.input_ids = tokenizer.encode(probe.prompt_text, return_tensors=None)
            else:
                prompt = f"Question: {probe.question}\nChoices:\n"
                for i, choice in enumerate(probe.choices):
                    prompt += f"{chr(65+i)}. {choice}\n"
                prompt += "Answer:"
                probe.input_ids = tokenizer.encode(prompt, return_tensors=None)

    runner = DeterministicTraceRunner(model, tokenizer, device, seed=args.seed)
    results = runner.run_full_capture(probe_set, args.T)

    import csv
    import numpy as np

    out_root = Path(args.output_dir) / "traces"
    out_root.mkdir(parents=True, exist_ok=True)

    for T_val in args.T:
        T_dir = out_root / f"T{T_val}"
        T_dir.mkdir(exist_ok=True)

        flow_T = {}
        decoder_T = {}
        for pid, traces in results["flow_traces"].items():
            for t in traces:
                if t["T"] == T_val:
                    flow_T[pid] = t
        for pid, traces in results["decoder_traces"].items():
            for t in traces:
                if t["T"] == T_val:
                    decoder_T[pid] = t
        with open(T_dir / "flow_traces.json", "w") as f:
            json.dump(flow_T, f, indent=2)
        with open(T_dir / "decoder_traces.json", "w") as f:
            json.dump(decoder_T, f, indent=2)
        print(f"  T={T_val}: {len(flow_T)} flow traces, {len(decoder_T)} decoder traces")

    summary_path = Path(args.output_dir) / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["T", "probe_id", "benchmark", "endpoint_hidden_norm",
                          "trajectory_divergence_from_T1",
                          "predicted_answer", "parsed_answer_match",
                          "kl_divergence", "js_divergence",
                          "mean_velocity_norm"])
        for T_val in args.T:
            for pid in sorted(results["flow_traces"].keys()):
                ft = next((t for t in results["flow_traces"][pid] if t["T"] == T_val), None)
                dt = next((t for t in results["decoder_traces"][pid] if t["T"] == T_val), None)
                if ft and dt:
                    mean_vel = float(np.mean(ft["per_step_velocity_norms"])) if ft.get("per_step_velocity_norms") else 0.0
                    writer.writerow([
                        T_val, pid, dt["benchmark"],
                        ft["endpoint_hidden_norm"],
                        ft.get("trajectory_divergence_from_T1", 0),
                        dt["predicted_answer"],
                        dt["parsed_answer_match"],
                        dt.get("kl_divergence", 0),
                        dt.get("js_divergence", 0),
                        mean_vel,
                    ])
    print(f"Summary written to {summary_path}")


def _do_report(args):
    from src.diagnostic.analysis import run_analysis
    from src.diagnostic.report import generate_report

    traces_dir = args.traces_dir or str(Path(args.output_dir) / "traces")
    probes_path = args.probes or str(Path(traces_dir).parent / "probes.json")

    if not Path(traces_dir).exists():
        print(f"ERROR: Traces directory not found: {traces_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Running analysis on traces from: {traces_dir}")
    flow_result, decoder_result = run_analysis(traces_dir, args.T)
    report_text = generate_report(flow_result, decoder_result, probes_path, traces_dir)

    report_path = Path(args.output_dir) / "report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"Report written to {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="P3-D3 diagnostic probe and report generation"
    )

    parser.add_argument(
        "--checkpoint", type=str,
        help="Path to the P3-D3 checkpoint file"
    )
    parser.add_argument(
        "--config", type=str,
        help="Path to the model config YAML file"
    )
    parser.add_argument(
        "--mmlu-path", type=str,
        help="Path to MMLU-Pro stress test results JSON"
    )
    parser.add_argument(
        "--arc-path", type=str,
        help="Path to ARC-Easy stress test results JSON"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/diagnostic_p3d3",
        help="Directory for output files (default: results/diagnostic_p3d3)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for deterministic behavior (default: 42)"
    )
    parser.add_argument(
        "--T", type=int, nargs="+", default=[1, 2, 8, 64],
        help="List of T values to test (default: 1 2 8 64)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"],
        help="Device to run on (default: cuda)"
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Generate report from saved traces (skips model loading)"
    )
    parser.add_argument(
        "--traces-dir", type=str,
        help="Path to traces directory [default: results/diagnostic_p3d3/traces]"
    )
    parser.add_argument(
        "--probes", type=str,
        help="Path to probes.json [default: <traces-dir>/../probes.json]"
    )

    args = parser.parse_args()

    if args.report and not args.checkpoint:
        _do_report(args)
    elif args.report and args.checkpoint:
        _do_capture(args)
        _do_report(args)
    elif args.checkpoint:
        _do_capture(args)
    else:
        parser.error("Either --checkpoint (for capture mode) or --report (for report-only mode) is required")


if __name__ == "__main__":
    main()
