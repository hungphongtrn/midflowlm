#!/usr/bin/env python
"""P3-D3 diagnostic probe — Phase 1 skeleton.

This script performs diagnostic analysis on the P3-D3 checkpoint to investigate
flat T-scaling behavior. It selects probe examples from stress test results and
prepares for trace-based analysis.

Usage:
    python scripts/diagnose_p3d3.py \
        --checkpoint ./models/p3_d3_mix_c/checkpoint.pth \
        --config configs/v0_1_matrix/midflow_qwen_8to11_p3_d3_flow_mixc_endtrajkl_trainT_r2468.yaml \
        --mmlu-path results/stress_test/mmlu_pro_results.json \
        --arc-path results/stress_test/arc_easy_results.json \
        --output-dir results/diagnostic_p3d3
"""
import argparse
import json
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.diagnostic.probe import ProbeSet, select_probes
from src.diagnostic.runner import DeterministicTraceRunner, load_model_from_checkpoint


def main():
    parser = argparse.ArgumentParser(
        description="P3-D3 diagnostic probe for investigating flat T-scaling"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the P3-D3 checkpoint file (e.g., ./models/p3_d3_mix_c/checkpoint.pth)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the model config YAML file"
    )
    parser.add_argument(
        "--mmlu-path",
        type=str,
        required=True,
        help="Path to MMLU-Pro stress test results JSON"
    )
    parser.add_argument(
        "--arc-path",
        type=str,
        required=True,
        help="Path to ARC-Easy stress test results JSON"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/diagnostic_p3d3",
        help="Directory for output files (default: results/diagnostic_p3d3)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic behavior (default: 42)"
    )
    parser.add_argument(
        "--T",
        type=int,
        nargs="+",
        default=[1, 2, 8, 64],
        help="List of T values to test (default: 1 2 8 64)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run on (default: cuda)"
    )

    args = parser.parse_args()

    # Validate input paths exist
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

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Select probes from stress test results
    print(f"\nSelecting probes from stress test results...")
    print(f"  MMLU-Pro: {mmlu_path}")
    print(f"  ARC-Easy: {arc_path}")
    
    probes = select_probes(
        mmlu_pro_path=str(mmlu_path),
        arc_easy_path=str(arc_path),
    )
    
    # Create probe set with metadata
    probe_set = ProbeSet(
        probes=probes,
        checkpoint_source=str(checkpoint_path),
        seed=args.seed,
    )

    # Write probes to JSON
    probes_file = output_dir / "probes.json"
    with open(probes_file, "w") as f:
        json.dump(probe_set.to_dict(), f, indent=2)
    
    print(f"\nSelected {len(probes)} probes -> {probes_file}")
    
    # Print breakdown by benchmark
    mmlu_count = sum(1 for p in probes if p.benchmark == "mmlu_pro")
    arc_count = sum(1 for p in probes if p.benchmark == "arc_easy")
    print(f"  - MMLU-Pro: {mmlu_count}")
    print(f"  - ARC-Easy: {arc_count}")

    # Print summary
    print(f"\n{'='*60}")
    print("Phase 1 Skeleton Complete - Starting Trace Collection")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Config: {config_path}")
    print(f"Seed: {args.seed}")
    print(f"T values: {args.T}")

    # Load model and run traces
    print(f"\n{'='*60}")
    print("Loading model from checkpoint...")
    print(f"{'='*60}")
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    try:
        model, tokenizer = load_model_from_checkpoint(
            checkpoint_path=str(checkpoint_path),
            config_path=str(config_path),
            device=device,
        )
        print(f"Model loaded successfully!")
        print(f"Model family: {model.family}")
        print(f"Model layers: {model.start_layer}-{model.end_layer}")
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Create trace runner and run full capture
    print(f"\n{'='*60}")
    print("Running diagnostic traces...")
    print(f"{'='*60}")

    # Ensure all probes have input_ids
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

    # Final summary
    print(f"\n{'='*60}")
    print("Diagnostic Trace Collection Complete")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Probe set: {probes_file}")
    print(f"Traces: {out_root}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
