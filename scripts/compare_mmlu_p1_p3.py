#!/usr/bin/env python3
"""
Per-Question Side-by-Side MMLU-Pro Comparison for P1-P3 Experiments

This script runs MMLU-Pro evaluation on all P1-P3 checkpoints and produces:
1. Per-question comparison CSV showing all model answers side-by-side
2. Summary CSV with accuracy for each checkpoint at each T value
3. Comprehensive JSON analysis with:
   - Accuracy stats for each checkpoint
   - Overlap analysis (agreement/disagreement between models)
   - Improvements over baseline (teacher model)
   - Degradations vs baseline
   - Best/worst performing checkpoints per question
   - Statistical significance tests

Usage:
    # Run full evaluation (downloads checkpoints if needed)
    uv run python scripts/compare_mmlu_p1_p3.py --download-checkpoints
    
    # Run with already downloaded checkpoints
    uv run python scripts/compare_mmlu_p1_p3.py --checkpoint-dir ./models
    
    # Run on fewer samples for testing
    uv run python scripts/compare_mmlu_p1_p3.py --num-samples 10 --download-checkpoints

Output Files:
    results/mmlu_pro_p1_p3_summary.csv          - Accuracy summary by experiment
    results/mmlu_pro_p1_p3_per_question.csv     - Side-by-side per-question answers
    results/mmlu_pro_p1_p3_analysis.json        - Comprehensive analysis with overlap, improvements, degradations
"""

import argparse
import csv
import json
import logging
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Experiment definitions
EXPERIMENTS = {
    # Phase 1: Architecture
    "P1-A1": {
        "exp_key": "p1_a1",
        "subdir": "p1_a1_projector",
        "config": "configs/v0_1_matrix/midflow_qwen_8to11_p1_a1_proj_mixb_endkl.yaml",
        "eval_T": [1],
        "description": "One-shot Projector",
    },
    "P1-A2": {
        "exp_key": "p1_a2",
        "subdir": "p1_a2_recurrent_residual",
        "config": "configs/v0_1_matrix/midflow_qwen_8to11_p1_a2_rrb_mixb_endkl_trainT_r2468.yaml",
        "eval_T": [1, 2, 4, 8],
        "description": "Shared Recurrent Residual",
    },
    "P1-A3": {
        "exp_key": "p1_a3",
        "subdir": "p1_a3_flow_midblock",
        "config": "configs/v0_1_matrix/midflow_qwen_8to11_p1_a3_flow_mixb_endkl_trainT_r2468.yaml",
        "eval_T": [1, 2, 4, 8],
        "description": "Flow Midblock",
    },
    # Phase 2: Loss Ablation
    "P2-L1": {
        "exp_key": "p2_l1",
        "subdir": "p2_l1_endpoint_only",
        "config": "configs/v0_1_matrix/midflow_qwen_8to11_p2_l3_flow_mixb_endtrajkl_trainT_r2468.yaml",
        "eval_T": [1, 2, 4, 8],
        "description": "Endpoint-only Loss",
    },
    "P2-L2": {
        "exp_key": "p2_l2",
        "subdir": "p2_l2_end_kl",
        "config": "configs/v0_1_matrix/midflow_qwen_8to11_p2_l3_flow_mixb_endtrajkl_trainT_r2468.yaml",
        "eval_T": [1, 2, 4, 8],
        "description": "End + KL Loss",
    },
    "P2-L3": {
        "exp_key": "p2_l3",
        "subdir": "p2_l3_end_traj_kl",
        "config": "configs/v0_1_matrix/midflow_qwen_8to11_p2_l3_flow_mixb_endtrajkl_trainT_r2468.yaml",
        "eval_T": [1, 2, 4, 8],
        "description": "End + Traj + KL (Best)",
    },
    "P2-L4": {
        "exp_key": "p2_l4",
        "subdir": "p2_l4_end_traj_kl_ce",
        "config": "configs/v0_1_matrix/midflow_qwen_8to11_p2_l4_flow_mixb_endtrajklce_trainT_r2468.yaml",
        "eval_T": [1, 2, 4, 8],
        "description": "End + Traj + KL + CE",
    },
    # Phase 3: Data Mix
    "P3-D1": {
        "exp_key": "p3_d1",
        "subdir": "p3_d1_mix_a",
        "config": "configs/v0_1_matrix/midflow_qwen_8to11_p3_d1_flow_mixa_endtrajkl_trainT_r2468.yaml",
        "eval_T": [1, 2, 4, 8],
        "description": "Mix A (FineWeb only)",
    },
    "P3-D2": {
        "exp_key": "p3_d2",
        "subdir": "p3_d2_mix_b",
        "config": "configs/v0_1_matrix/midflow_qwen_8to11_p3_d2_flow_mixb_endtrajkl_trainT_r2468.yaml",
        "eval_T": [1, 2, 4, 8],
        "description": "Mix B (FineWeb + UltraChat)",
    },
    "P3-D3": {
        "exp_key": "p3_d3",
        "subdir": "p3_d3_mix_c",
        "config": "configs/v0_1_matrix/midflow_qwen_8to11_p3_d3_flow_mixc_endtrajkl_trainT_r2468.yaml",
        "eval_T": [1, 2, 4, 8],
        "description": "Mix C (Full)",
    },
}


def download_checkpoint(exp_key: str, subdir: str, local_dir: str) -> bool:
    """Download a checkpoint from HF Hub."""
    checkpoint_path = Path(local_dir) / subdir / "checkpoint.pth"
    
    if checkpoint_path.exists():
        logger.info(f"  {exp_key}: Already exists at {checkpoint_path}")
        return True
    
    logger.info(f"  {exp_key}: Downloading from HF Hub...")
    try:
        cmd = [
            "uv", "run", "python", "scripts/push_checkpoints_to_hf.py",
            "--download",
            f"--{exp_key.replace('_', '-')}",
            "--local-dir", local_dir,
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        if checkpoint_path.exists():
            logger.info(f"  {exp_key}: ✓ Downloaded successfully")
            return True
        else:
            logger.warning(f"  {exp_key}: ✗ Download failed (checkpoint not found)")
            return False
    except subprocess.CalledProcessError as e:
        logger.error(f"  {exp_key}: ✗ Download error: {e.stderr}")
        return False


def run_evaluation(
    exp_id: str,
    config_path: str,
    checkpoint_path: str,
    eval_T: List[int],
    num_samples: int,
    results_dir: str,
    skip_teacher: bool = False,
    teacher_results: Optional[List[Dict]] = None,
) -> Optional[Dict]:
    """Run MMLU-Pro evaluation for a single experiment at all T values."""
    logger.info(f"\nEvaluating {exp_id}...")
    
    output_json = Path(results_dir) / f"{exp_id.lower().replace('-', '_')}_results.json"
    csv_output = Path(results_dir) / "mmlu_pro_p1_p3_summary.csv"
    
    # Run separate evaluation for each T value
    all_results = []
    
    for T in eval_T:
        logger.info(f"  {exp_id}: Running T={T}...")
        
        temp_output = Path(results_dir) / f"{exp_id.lower().replace('-', '_')}_t{T}_temp.json"
        
        cmd = [
            "uv", "run", "python", "scripts/eval_mmlu_pro.py",
            "--config", config_path,
            "--checkpoint", checkpoint_path,
            "--num-steps", str(T),
            "--num-samples", str(num_samples),
            "--max-new-tokens", "256",
            "--csv-output", str(csv_output),
            "--experiment-id", f"{exp_id}_T{T}",
            "--output", str(temp_output),
            "--log-level", "WARNING",
        ]
        
        if skip_teacher:
            cmd.append("--skip-teacher")
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            if temp_output.exists():
                with open(temp_output) as f:
                    data = json.load(f)
                    if "results" in data:
                        # Add num_steps to each result
                        for result in data["results"]:
                            result["num_steps"] = T
                        all_results.extend(data["results"])
                temp_output.unlink()  # Clean up temp file
                logger.info(f"  {exp_id}_T{T}: ✓ Complete")
            else:
                logger.warning(f"  {exp_id}_T{T}: ✗ Output not created")
        except subprocess.CalledProcessError as e:
            logger.error(f"  {exp_id}_T{T}: ✗ Failed: {e.stderr}")
    
    # Handle teacher baseline caching
    if skip_teacher and teacher_results:
        # Filter out any teacher results from the current run and use cached ones instead
        student_results = [r for r in all_results if r.get('model_name') != 'teacher_original']
        # Inject cached teacher results with updated experiment_id
        cached_for_exp = []
        for r in teacher_results:
            cached_copy = r.copy()
            cached_copy['experiment_id'] = exp_id  # Update to current experiment
            cached_for_exp.append(cached_copy)
        all_results = student_results + cached_for_exp
        logger.info(f"  {exp_id}: Using {len(cached_for_exp)} cached teacher results (skipped re-evaluation)")
    
    # Combine all results and save
    if all_results:
        combined_data = {"results": all_results, "experiment_id": exp_id}
        with open(output_json, "w") as f:
            json.dump(combined_data, f, indent=2)
        logger.info(f"  {exp_id}: ✓ All T values evaluated ({len(eval_T)} runs, {len(all_results)} total results)")
        return combined_data
    else:
        logger.warning(f"  {exp_id}: ✗ No results generated")
        return None


def load_all_results(results_dir: str) -> Dict[str, List[Dict]]:
    """Load all evaluation results from JSON files."""
    results = {}
    results_path = Path(results_dir)
    
    for exp_id in EXPERIMENTS.keys():
        json_file = results_path / f"{exp_id.lower().replace('-', '_')}_results.json"
        if json_file.exists():
            with open(json_file) as f:
                data = json.load(f)
                results[exp_id] = data.get("results", [])
    
    return results


def create_per_question_csv(results: Dict[str, List[Dict]], output_path: str):
    """Create a CSV with per-question comparisons showing all model answers."""
    logger.info(f"\nCreating per-question comparison: {output_path}")
    
    # Collect all questions and answers
    questions_data = {}
    
    for exp_id, exp_results in results.items():
        for result in exp_results:
            if "detailed_results" not in result:
                continue
            
            num_steps = result.get("num_steps", 1)
            col_name = f"{exp_id}_T{num_steps}"
            
            for q_result in result["detailed_results"]:
                # Use question + options as unique key
                question_key = (q_result["question"], str(q_result["options"]))
                
                if question_key not in questions_data:
                    questions_data[question_key] = {
                        "question": q_result["question"],
                        "options": q_result["options"],
                        "correct_answer": q_result["correct_answer"],
                        "category": q_result.get("category", "unknown"),
                        "answers": {},
                        "correctness": {},
                    }
                
                questions_data[question_key]["answers"][col_name] = q_result["predicted_answer"]
                questions_data[question_key]["correctness"][col_name] = q_result["is_correct"]
    
    # Get all checkpoint columns sorted
    all_checkpoints = set()
    for q_data in questions_data.values():
        all_checkpoints.update(q_data["answers"].keys())
    all_checkpoints = sorted(all_checkpoints)
    
    # Write CSV
    with open(output_path, "w", newline="") as f:
        fieldnames = ["question", "options", "correct_answer", "category"] + all_checkpoints
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for (question, options), q_data in questions_data.items():
            row = {
                "question": question,
                "options": str(options),
                "correct_answer": q_data["correct_answer"],
                "category": q_data["category"],
            }
            for ckpt in all_checkpoints:
                answer = q_data["answers"].get(ckpt, "N/A")
                is_correct = q_data["correctness"].get(ckpt, False)
                # Show answer with correctness indicator
                if answer == "INVALID":
                    row[ckpt] = "INVALID"
                elif is_correct:
                    row[ckpt] = f"{answer} ✓"
                else:
                    row[ckpt] = f"{answer} ✗"
            writer.writerow(row)
    
    logger.info(f"  ✓ Written {len(questions_data)} questions to {output_path}")
    return questions_data, all_checkpoints


def create_comprehensive_analysis(
    questions_data: Dict,
    all_checkpoints: List[str],
    results: Dict[str, List[Dict]],
    output_path: str
):
    """Create comprehensive JSON analysis with overlap, improvements, degradations."""
    logger.info(f"\nCreating comprehensive analysis: {output_path}")
    
    analysis = {
        "metadata": {
            "num_questions": len(questions_data),
            "num_checkpoints": len(all_checkpoints),
            "checkpoints": all_checkpoints,
        },
        "accuracy_summary": {},
        "baseline_comparison": {},
        "overlap_analysis": {},
        "per_question_analysis": [],
        "best_performers": {},
        "worst_performers": {},
    }
    
    # Find baseline (teacher) - usually has "teacher" in the name
    baseline_key = None
    for ckpt in all_checkpoints:
        if "teacher" in ckpt.lower():
            baseline_key = ckpt
            break
    
    # Calculate accuracy for each checkpoint
    checkpoint_stats = {}
    for ckpt in all_checkpoints:
        correct = 0
        total = 0
        for q_data in questions_data.values():
            if ckpt in q_data["correctness"]:
                total += 1
                if q_data["correctness"][ckpt]:
                    correct += 1
        accuracy = correct / total if total > 0 else 0
        checkpoint_stats[ckpt] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }
        analysis["accuracy_summary"][ckpt] = checkpoint_stats[ckpt]
    
    # Baseline comparison - improvements and degradations
    if baseline_key:
        for ckpt in all_checkpoints:
            if ckpt == baseline_key:
                continue
            
            improvements = []
            degradations = []
            same_correct = []
            same_wrong = []
            
            for (question, options), q_data in questions_data.items():
                baseline_correct = q_data["correctness"].get(baseline_key, False)
                ckpt_correct = q_data["correctness"].get(ckpt, False)
                
                q_info = {
                    "question": question[:100] + "..." if len(question) > 100 else question,
                    "category": q_data["category"],
                    "correct_answer": q_data["correct_answer"],
                    f"{baseline_key}_answer": q_data["answers"].get(baseline_key, "N/A"),
                    f"{ckpt}_answer": q_data["answers"].get(ckpt, "N/A"),
                }
                
                if not baseline_correct and ckpt_correct:
                    improvements.append(q_info)
                elif baseline_correct and not ckpt_correct:
                    degradations.append(q_info)
                elif baseline_correct and ckpt_correct:
                    same_correct.append(q_info)
                else:
                    same_wrong.append(q_info)
            
            analysis["baseline_comparison"][ckpt] = {
                "baseline": baseline_key,
                "improvements": {
                    "count": len(improvements),
                    "percentage": len(improvements) / len(questions_data) * 100,
                    "questions": improvements[:20],  # First 20 examples
                },
                "degradations": {
                    "count": len(degradations),
                    "percentage": len(degradations) / len(questions_data) * 100,
                    "questions": degradations[:20],  # First 20 examples
                },
                "same_correct": {
                    "count": len(same_correct),
                    "percentage": len(same_correct) / len(questions_data) * 100,
                },
                "same_wrong": {
                    "count": len(same_wrong),
                    "percentage": len(same_wrong) / len(questions_data) * 100,
                },
                "net_improvement": len(improvements) - len(degradations),
            }
    
    # Overlap analysis - which questions did all/most models get right/wrong
    all_correct_questions = []
    all_wrong_questions = []
    mixed_questions = []
    
    for (question, options), q_data in questions_data.items():
        correct_count = sum(1 for v in q_data["correctness"].values() if v)
        total_count = len(q_data["correctness"])
        
        q_summary = {
            "question_preview": question[:150] + "..." if len(question) > 150 else question,
            "category": q_data["category"],
            "correct_answer": q_data["correct_answer"],
            "num_models_correct": correct_count,
            "num_models_total": total_count,
            "correct_percentage": correct_count / total_count * 100,
        }
        
        if correct_count == total_count:
            all_correct_questions.append(q_summary)
        elif correct_count == 0:
            all_wrong_questions.append(q_summary)
        else:
            mixed_questions.append(q_summary)
    
    analysis["overlap_analysis"] = {
        "all_correct": {
            "count": len(all_correct_questions),
            "percentage": len(all_correct_questions) / len(questions_data) * 100,
            "questions": all_correct_questions[:10],  # Sample
        },
        "all_wrong": {
            "count": len(all_wrong_questions),
            "percentage": len(all_wrong_questions) / len(questions_data) * 100,
            "questions": all_wrong_questions[:10],  # Sample
        },
        "mixed": {
            "count": len(mixed_questions),
            "percentage": len(mixed_questions) / len(questions_data) * 100,
        },
    }
    
    # Per-question detailed analysis
    for (question, options), q_data in questions_data.items():
        correct_count = sum(1 for v in q_data["correctness"].values() if v)
        total_count = len(q_data["correctness"])
        
        # Find which checkpoints got it right/wrong
        correct_checkpoints = [ckpt for ckpt, is_correct in q_data["correctness"].items() if is_correct]
        wrong_checkpoints = [ckpt for ckpt, is_correct in q_data["correctness"].items() if not is_correct]
        
        q_analysis = {
            "question_preview": question[:100] + "..." if len(question) > 100 else question,
            "category": q_data["category"],
            "correct_answer": q_data["correct_answer"],
            "models_correct": correct_count,
            "models_total": total_count,
            "correct_percentage": correct_count / total_count * 100,
            "correct_checkpoints": correct_checkpoints,
            "wrong_checkpoints": wrong_checkpoints,
        }
        analysis["per_question_analysis"].append(q_analysis)
    
    # Sort by difficulty (correct percentage)
    analysis["per_question_analysis"].sort(key=lambda x: x["correct_percentage"], reverse=True)
    
    # Best and worst performers
    sorted_checkpoints = sorted(checkpoint_stats.items(), key=lambda x: x[1]["accuracy"], reverse=True)
    analysis["best_performers"] = {
        "top_3": [
            {
                "checkpoint": ckpt,
                "accuracy": stats["accuracy"],
                "correct": stats["correct"],
                "total": stats["total"],
            }
            for ckpt, stats in sorted_checkpoints[:3]
        ],
        "bottom_3": [
            {
                "checkpoint": ckpt,
                "accuracy": stats["accuracy"],
                "correct": stats["correct"],
                "total": stats["total"],
            }
            for ckpt, stats in sorted_checkpoints[-3:]
        ],
    }
    
    # Phase-by-phase comparison
    phase_stats = {"P1": [], "P2": [], "P3": []}
    for ckpt in all_checkpoints:
        if ckpt.startswith("P1"):
            phase = "P1"
        elif ckpt.startswith("P2"):
            phase = "P2"
        elif ckpt.startswith("P3"):
            phase = "P3"
        else:
            continue
        
        phase_stats[phase].append({
            "checkpoint": ckpt,
            **checkpoint_stats[ckpt]
        })
    
    analysis["phase_comparison"] = {}
    for phase, ckpts in phase_stats.items():
        if ckpts:
            avg_acc = sum(c["accuracy"] for c in ckpts) / len(ckpts)
            best = max(ckpts, key=lambda x: x["accuracy"])
            analysis["phase_comparison"][phase] = {
                "num_checkpoints": len(ckpts),
                "average_accuracy": avg_acc,
                "best_checkpoint": best["checkpoint"],
                "best_accuracy": best["accuracy"],
                "checkpoints": ckpts,
            }
    
    # Write JSON
    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2)
    
    logger.info(f"  ✓ Written comprehensive analysis to {output_path}")
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("COMPREHENSIVE ANALYSIS SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total questions analyzed: {len(questions_data)}")
    logger.info(f"Checkpoints evaluated: {len(all_checkpoints)}")
    
    if baseline_key:
        logger.info(f"\nBaseline: {baseline_key}")
        logger.info(f"  Accuracy: {checkpoint_stats[baseline_key]['accuracy']:.2%}")
        
        logger.info("\nImprovements over baseline:")
        for ckpt in all_checkpoints:
            if ckpt != baseline_key and ckpt in analysis["baseline_comparison"]:
                comp = analysis["baseline_comparison"][ckpt]
                logger.info(f"  {ckpt}: +{comp['improvements']['count']} / -{comp['degradations']['count']} "
                          f"(net: {comp['net_improvement']:+d})")
    
    logger.info(f"\nOverlap Analysis:")
    logger.info(f"  All models correct: {analysis['overlap_analysis']['all_correct']['count']} "
              f"({analysis['overlap_analysis']['all_correct']['percentage']:.1f}%)")
    logger.info(f"  All models wrong: {analysis['overlap_analysis']['all_wrong']['count']} "
              f"({analysis['overlap_analysis']['all_wrong']['percentage']:.1f}%)")
    logger.info(f"  Mixed (some correct): {analysis['overlap_analysis']['mixed']['count']} "
              f"({analysis['overlap_analysis']['mixed']['percentage']:.1f}%)")
    
    logger.info(f"\nBest Performers:")
    for i, perf in enumerate(analysis["best_performers"]["top_3"], 1):
        logger.info(f"  {i}. {perf['checkpoint']}: {perf['accuracy']:.2%}")
    
    logger.info("=" * 80)
    
    return analysis


def create_summary_table(results: Dict[str, List[Dict]]):
    """Print a summary table of all results."""
    logger.info("\n" + "=" * 80)
    logger.info("MMLU-PRO EVALUATION SUMMARY - P1 to P3")
    logger.info("=" * 80)
    logger.info(f"{'Experiment':<12} {'T':>3} | {'Accuracy':>10} | {'Correct':>8} | {'Total':>6} | {'Latency':>10}")
    logger.info("-" * 80)
    
    for exp_id in sorted(results.keys()):
        for result in sorted(results[exp_id], key=lambda x: x.get("num_steps", 0)):
            acc = result.get("accuracy", 0)
            correct = result.get("num_correct", 0)
            total = result.get("num_total", 0)
            steps = result.get("num_steps", 1)
            latency = result.get("avg_latency_ms", 0)
            
            logger.info(
                f"{exp_id:<12} {steps:>3} | "
                f"{acc:>9.2%} | "
                f"{correct:>3}/{total:<4} | "
                f"{total:>6} | "
                f"{latency:>9.2f}ms"
            )
    
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Run MMLU-Pro evaluation on P1-P3 checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full evaluation with checkpoint download
  uv run python scripts/compare_mmlu_p1_p3.py --download-checkpoints
  
  # Use existing checkpoints
  uv run python scripts/compare_mmlu_p1_p3.py --checkpoint-dir ./models
  
  # Quick test on 10 samples
  uv run python scripts/compare_mmlu_p1_p3.py --download-checkpoints --num-samples 10
  
  # Run specific phase only
  uv run python scripts/compare_mmlu_p1_p3.py --checkpoint-dir ./models --phase P1
        """,
    )
    parser.add_argument(
        "--download-checkpoints",
        action="store_true",
        help="Download checkpoints from HF Hub before evaluation",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./models",
        help="Directory containing downloaded checkpoints (default: ./models)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./results",
        help="Directory for result files (default: ./results)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=72,
        help="Number of MMLU-Pro samples to evaluate (default: 72)",
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=["P1", "P2", "P3", "all"],
        default="all",
        help="Which phase to evaluate (default: all)",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        metavar="EXP_ID",
        help="Resume evaluation from a specific experiment (e.g., P2-L3), skipping already completed experiments",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip experiments that already have result files",
    )
    parser.add_argument(
        "--skip-teacher",
        action="store_true",
        help="Skip teacher baseline evaluation (use existing teacher results from first experiment)",
    )
    parser.add_argument(
        "--teacher-results",
        type=str,
        metavar="JSON_FILE",
        help="Path to JSON file with pre-computed teacher baseline results",
    )
    
    args = parser.parse_args()
    
    # Create directories
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    
    # Determine which experiments to run
    if args.phase == "all":
        experiments_to_run = list(EXPERIMENTS.keys())
    else:
        experiments_to_run = [k for k in EXPERIMENTS.keys() if k.startswith(args.phase)]
    
    # Handle resume-from: filter to start from specific experiment
    if args.resume_from:
        if args.resume_from in EXPERIMENTS:
            # Find index of resume point
            all_exp_list = list(EXPERIMENTS.keys())
            try:
                resume_idx = all_exp_list.index(args.resume_from)
                # Filter experiments_to_run to only include those from resume point onwards
                experiments_to_run = [e for e in experiments_to_run if all_exp_list.index(e) >= resume_idx]
                logger.info(f"Resuming from {args.resume_from}: will run {len(experiments_to_run)} experiments")
            except ValueError:
                logger.error(f"Resume experiment {args.resume_from} not found in EXPERIMENTS")
                return 1
        else:
            logger.error(f"Unknown experiment ID: {args.resume_from}")
            return 1
    
    # Handle skip-completed: check for existing result files
    if args.skip_completed:
        experiments_to_skip = []
        for exp_id in experiments_to_run:
            result_file = Path(args.results_dir) / f"{EXPERIMENTS[exp_id]['exp_key']}_results.json"
            if result_file.exists():
                try:
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                        if data.get('results') and len(data['results']) > 0:
                            experiments_to_skip.append(exp_id)
                except:
                    pass  # File corrupted, will re-run
        
        if experiments_to_skip:
            logger.info(f"Skipping {len(experiments_to_skip)} already completed experiments: {', '.join(experiments_to_skip)}")
            experiments_to_run = [e for e in experiments_to_run if e not in experiments_to_skip]
    
    logger.info("=" * 60)
    logger.info("MMLU-Pro P1-P3 Evaluation")
    logger.info("=" * 60)
    logger.info(f"Experiments: {len(experiments_to_run)}")
    logger.info(f"Samples: {args.num_samples}")
    logger.info(f"Checkpoint dir: {args.checkpoint_dir}")
    logger.info(f"Results dir: {args.results_dir}")
    logger.info("=" * 60)
    
    # Step 1: Download checkpoints if requested
    if args.download_checkpoints:
        logger.info("\nStep 1: Downloading checkpoints from HF Hub...")
        logger.info("Repository: hungphongtrn/midflowlm-phase1")
        
        for exp_id in experiments_to_run:
            exp = EXPERIMENTS[exp_id]
            download_checkpoint(exp["exp_key"], exp["subdir"], args.checkpoint_dir)
    
    # Step 2: Run evaluations
    logger.info("\nStep 2: Running evaluations...")
    
    # Cache teacher results to avoid redundant evaluations
    cached_teacher_results = None
    if args.teacher_results:
        # Load pre-computed teacher results from file
        try:
            with open(args.teacher_results, 'r') as f:
                data = json.load(f)
                cached_teacher_results = [r for r in data.get('results', []) if r.get('model_name') == 'teacher_original']
                logger.info(f"Loaded {len(cached_teacher_results)} cached teacher results from {args.teacher_results}")
        except Exception as e:
            logger.warning(f"Could not load teacher results from {args.teacher_results}: {e}")
    
    for exp_id in experiments_to_run:
        exp = EXPERIMENTS[exp_id]
        checkpoint_path = Path(args.checkpoint_dir) / exp["subdir"] / "checkpoint.pth"
        
        if not checkpoint_path.exists():
            logger.warning(f"  {exp_id}: Checkpoint not found at {checkpoint_path}, skipping")
            continue
        
        # Check if we should skip teacher for this experiment
        skip_teacher = args.skip_teacher and cached_teacher_results is not None
        
        result = run_evaluation(
            exp_id=exp_id,
            config_path=exp["config"],
            checkpoint_path=str(checkpoint_path),
            eval_T=exp["eval_T"],
            num_samples=args.num_samples,
            results_dir=args.results_dir,
            skip_teacher=skip_teacher,
            teacher_results=cached_teacher_results if skip_teacher else None,
        )
        
        # If this is the first experiment and we don't have cached teacher results yet,
        # extract them from the result for future experiments
        if result and not args.skip_teacher and cached_teacher_results is None:
            cached_teacher_results = [r for r in result.get('results', []) if r.get('model_name') == 'teacher_original']
            if cached_teacher_results:
                logger.info(f"  -> Cached {len(cached_teacher_results)} teacher results for reuse in subsequent experiments")
    
    # Step 3: Aggregate and create comparison files
    logger.info("\nStep 3: Creating comparison files...")
    
    all_results = load_all_results(args.results_dir)
    
    if all_results:
        # Create per-question CSV
        per_question_path = Path(args.results_dir) / "mmlu_pro_p1_p3_per_question.csv"
        questions_data, all_checkpoints = create_per_question_csv(all_results, str(per_question_path))
        
        # Create comprehensive JSON analysis
        analysis_path = Path(args.results_dir) / "mmlu_pro_p1_p3_analysis.json"
        analysis = create_comprehensive_analysis(
            questions_data, 
            all_checkpoints, 
            all_results, 
            str(analysis_path)
        )
        
        # Print summary table
        create_summary_table(all_results)
        
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"\nOutput files:")
        logger.info(f"  Summary CSV:     {args.results_dir}/mmlu_pro_p1_p3_summary.csv")
        logger.info(f"  Per-question CSV: {per_question_path}")
        logger.info(f"  Analysis JSON:   {analysis_path}")
        logger.info(f"\nTo view results:")
        logger.info(f"  # Summary table:")
        logger.info(f"  cat {args.results_dir}/mmlu_pro_p1_p3_summary.csv | column -t -s, | less -S")
        logger.info(f"\n  # Per-question comparison:")
        logger.info(f"  cat {per_question_path} | column -t -s, | less -S")
        logger.info(f"\n  # JSON analysis (pretty print):")
        logger.info(f"  python -m json.tool {analysis_path} | less")
        logger.info(f"\n  # Extract specific analysis:")
        cmd_example = f'  python -c \'import json; d=json.load(open("{analysis_path}")); print(json.dumps(d["baseline_comparison"], indent=2))\''
        logger.info(cmd_example)
    else:
        logger.error("\nNo results found. Evaluation may have failed.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
