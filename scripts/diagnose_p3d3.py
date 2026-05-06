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
    
    # Create trace runner
    print(f"\n{'='*60}")
    print("Running diagnostic traces...")
    print(f"{'='*60}")
    
    runner = DeterministicTraceRunner(
        model=model,
        tokenizer=tokenizer,
        device=device,
        seed=args.seed,
    )
    
    # Run probes at each T value
    for T in args.T:
        print(f"\nRunning {len(probes)} probes at T={T}...")
        
        # Prepare probe set for this T value
        records = []
        for probe in probes:
            # Check if probe has input_ids, if not, tokenize the prompt
            if probe.input_ids is None or len(probe.input_ids) == 0:
                if probe.prompt_text:
                    probe.input_ids = tokenizer.encode(probe.prompt_text, return_tensors=None)
                else:
                    # Create a simple prompt from question and choices
                    prompt = f"Question: {probe.question}\nChoices:\n"
                    for i, choice in enumerate(probe.choices):
                        prompt += f"{chr(65+i)}. {choice}\n"
                    prompt += "Answer:"
                    probe.input_ids = tokenizer.encode(prompt, return_tensors=None)
            
            # Run the probe
            try:
                record = runner.run_single(probe, T=T)
                records.append(record)
                print(f"  {probe.id}: norm={record.endpoint_hidden_norm:.4f}, pred={record.predicted_answer}")
            except Exception as e:
                print(f"  {probe.id}: ERROR - {e}", file=sys.stderr)
        
        # Save traces for this T
        traces_file = output_dir / f"traces_T{T}.json"
        traces_data = {
            "T": T,
            "seed": args.seed,
            "checkpoint": str(checkpoint_path),
            "num_probes": len(records),
            "traces": [record.to_dict() for record in records],
        }
        
        with open(traces_file, "w") as f:
            json.dump(traces_data, f, indent=2)
        
        print(f"Saved {len(records)} traces to {traces_file}")
    
    # Final summary
    print(f"\n{'='*60}")
    print("Diagnostic Trace Collection Complete")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Probe set: {probes_file}")
    for T in args.T:
        traces_file = output_dir / f"traces_T{T}.json"
        print(f"  T={T}: {traces_file}")


if __name__ == "__main__":
    main()
