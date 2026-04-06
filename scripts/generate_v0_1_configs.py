#!/usr/bin/env python3
"""
Generate all v0.1 experiment matrix YAML configs.

Usage:
    python scripts/generate_v0_1_configs.py

This creates all P1/P2/P3/P4 experiment configs in configs/v0_1_matrix/
"""

import os
import yaml
import multiprocessing
from pathlib import Path

# Constants
BASE_DIR = Path(__file__).parent.parent
CONFIGS_DIR = BASE_DIR / "configs" / "v0_1_matrix"

# Auto-detect optimal workers based on CPU cores
# Formula: Leave 2 cores for system, distribute rest among parallel experiments
CPU_COUNT = multiprocessing.cpu_count()
NUM_GPUS = 3  # Default for 3090 setup
OPTIMAL_WORKERS = max(2, (CPU_COUNT - 2) // NUM_GPUS)  # Leave 2 cores for system

print(
    f"Detected {CPU_COUNT} CPU cores, configuring {OPTIMAL_WORKERS} dataloader workers per experiment"
)

# Hardware profile (auto-optimized for available CPUs)
HARDWARE_PROFILE = {
    "seq_len": 1024,
    "batch_size": 3,  # microbatch = 3
    "num_workers": OPTIMAL_WORKERS,  # Auto-calculated: (cores - 2) / 3 GPUs
    "pin_memory": True,  # Enable for GPU training
    "persistent_workers": True,  # Keep workers alive between epochs
    "prefetch_factor": 4,  # Prefetch 4 batches per worker
}

# Training settings (fixed for all experiments)
TRAINING_DEFAULTS = {
    "max_epochs": 3,
    "accumulate_grad_batches": 5,  # grad accum = 5 (3*5=15 effective batch)
    "sample_continuous_time": True,
    "log_every_n_steps": 10,
    "val_check_interval": 250,
    "gradient_checkpointing": True,
    "precision": "bf16-mixed",
}

# Optimizer settings (fixed for all experiments)
OPTIMIZER_DEFAULTS = {
    "name": "adamw",
    "learning_rate": 1.0e-4,
    "weight_decay": 0.01,
    "betas": [0.9, 0.95],
    "eps": 1.0e-8,
    "grad_clip_norm": 1.0,
}

# Scheduler settings
SCHEDULER_DEFAULTS = {
    "name": "cosine_with_warmup",
    "warmup_steps": 100,
}

# Model settings
MODEL_DEFAULTS = {
    "name": "Qwen/Qwen3.5-0.8B",
    "revision": None,
    "max_steps_T": 8,
    "step_embedding": "discrete",
    "reuse_qwen_modules": True,
}

# Teacher state mode
TEACHER_STATE = {"mode": "online_no_cache"}

# Teacher cache (disabled by default)
TEACHER_CACHE = {
    "enabled": False,
    "cache_dir": "./cache/experiment",
    "store_logits": True,
    "store_hidden_states": True,
    "overwrite": False,
}

# Data mix components
MIX_A_COMPONENTS = [
    {
        "name": "fineweb_edu",
        "dataset_name": "HuggingFaceFW/fineweb-edu",
        "dataset_config": "sample-10BT",
        "train_split": "train",
        "val_split": "train",
        "format_type": "plain_text",
        "text_field": "text",
        "train_samples": 12000,
        "val_samples": 600,
    }
]

MIX_B_COMPONENTS = MIX_A_COMPONENTS + [
    {
        "name": "ultrachat_sft",
        "dataset_name": "HuggingFaceH4/ultrachat_200k",
        "dataset_config": "default",
        "train_split": "train_sft",
        "val_split": "test_sft",
        "format_type": "chat_messages",
        "messages_field": "messages",
        "train_samples": 5000,
        "val_samples": 250,
    }
]

MIX_C_COMPONENTS = MIX_B_COMPONENTS + [
    {
        "name": "arc_challenge",
        "dataset_name": "allenai/ai2_arc",
        "dataset_config": "ARC-Challenge",
        "train_split": "train",
        "val_split": "validation",
        "format_type": "mcq_choices",
        "use_chat_template": True,
        "question_field": "question",
        "choices_field": "choices",
        "answer_field": "answerKey",
        "train_samples": 900,
        "val_samples": 120,
    },
    {
        "name": "arc_easy",
        "dataset_name": "allenai/ai2_arc",
        "dataset_config": "ARC-Easy",
        "train_split": "train",
        "val_split": "validation",
        "format_type": "mcq_choices",
        "use_chat_template": True,
        "question_field": "question",
        "choices_field": "choices",
        "answer_field": "answerKey",
        "train_samples": 1200,
        "val_samples": 150,
    },
    {
        "name": "commonsense_qa",
        "dataset_name": "tau/commonsense_qa",
        "dataset_config": "default",
        "train_split": "train",
        "val_split": "validation",
        "format_type": "mcq_choices",
        "use_chat_template": True,
        "question_field": "question",
        "choices_field": "choices",
        "answer_field": "answerKey",
        "train_samples": 1500,
        "val_samples": 150,
    },
    {
        "name": "openbookqa",
        "dataset_name": "allenai/openbookqa",
        "dataset_config": "main",
        "train_split": "train",
        "val_split": "validation",
        "format_type": "mcq_choices",
        "use_chat_template": True,
        "question_field": "question_stem",
        "choices_field": "choices",
        "answer_field": "answerKey",
        "train_samples": 900,
        "val_samples": 100,
    },
]

# Architecture family definitions
ARCH_ONE_SHOT = {
    "family": "one_shot_projector",
    "start_layer": 8,
    "end_layer": 11,
    "depth": 4,
    "mlp_ratio": 4.0,
    "qkv_bias": True,
    "use_qwen_causal_mask": True,
    "use_step_conditioning": False,
    "conditioning_mode": "timestep_plus_layer_boundary",
    "init_strategy": "fresh",
}

ARCH_RECURRENT = {
    "family": "shared_recurrent_residual",
    "start_layer": 8,
    "end_layer": 11,
    "depth": 4,
    "mlp_ratio": 4.0,
    "qkv_bias": True,
    "use_qwen_causal_mask": True,
    "use_step_conditioning": True,
    "conditioning_mode": "timestep_plus_layer_boundary",
    "init_strategy": "fresh",
}

ARCH_FLOW = {
    "family": "flow_midblock",
    "start_layer": 8,
    "end_layer": 11,
    "depth": 4,
    "mlp_ratio": 4.0,
    "qkv_bias": True,
    "use_qwen_causal_mask": True,
    "use_step_conditioning": True,
    "conditioning_mode": "timestep_plus_layer_boundary",
    "init_strategy": "fresh",
}


def build_config(
    experiment_name: str,
    replacement_model: dict,
    mixture_components: list,
    loss_config: dict,
    train_T_values: list,
    train_T_weights: list,
    eval_T_values: list,
    tags: list,
    seed: int = 1337,
    shuffle_seed: int = 1337,
) -> dict:
    """Build a complete experiment config."""

    # Build paths from experiment name
    safe_name = experiment_name.replace("-", "_")
    output_base = f"./outputs/{safe_name}"

    config = {
        "experiment_name": experiment_name,
        "seed": seed,
        "teacher_state": TEACHER_STATE,
        "teacher_cache": {
            **TEACHER_CACHE,
            "cache_dir": f"./cache/{safe_name}",
        },
        "model": {
            **MODEL_DEFAULTS,
            "train_T_values": train_T_values,
            "train_T_weights": train_T_weights,
        },
        "replacement_model": replacement_model,
        "data": {
            "loader": "mixture",
            "seq_len": HARDWARE_PROFILE["seq_len"],
            "batch_size": HARDWARE_PROFILE["batch_size"],
            "num_workers": HARDWARE_PROFILE["num_workers"],
            "pin_memory": HARDWARE_PROFILE["pin_memory"],
            "persistent_workers": HARDWARE_PROFILE["persistent_workers"],
            "prefetch_factor": HARDWARE_PROFILE["prefetch_factor"],
            "shuffle_seed": shuffle_seed,
            "mixture_components": mixture_components,
        },
        "loss": {
            **loss_config,
            "mask_padding_tokens": True,
        },
        "optimizer": OPTIMIZER_DEFAULTS,
        "scheduler": SCHEDULER_DEFAULTS,
        "train_loop": {
            **TRAINING_DEFAULTS,
            "checkpoint_dir": f"{output_base}/checkpoints",
        },
        "logging": {
            "log_dir": f"{output_base}/logs",
            "save_top_k": 2,
            "monitor": "val/total_loss",
            "mode": "min",
        },
        "tensorboard": {
            "enabled": True,
            "log_dir": f"{output_base}/tensorboard",
            "log_every_n_steps": 10,
        },
        "wandb": {
            "enabled": True,
            "project": "midflowlm-v0-1",
            "entity": None,
            "tags": tags,
        },
        # Extra metadata for queue system
        "eval": {
            "eval_T_values": eval_T_values,
        },
    }

    return config


def generate_p1_configs():
    """Generate Phase 1: Architecture Sanity configs."""
    configs = []

    # Loss: End + KL
    loss_endkl = {
        "velocity_weight": 0.0,
        "endpoint_weight": 1.0,
        "trajectory_weight": 0.0,
        "kl_weight": 0.5,
        "teacher_logits_source": "online",
        "ce_weight": 0.0,
    }

    # P1-A1: One-shot projector, Mix B, End + KL, T=1
    configs.append(
        build_config(
            experiment_name="midflow_qwen_8to11_p1_a1_proj_mixb_endkl",
            replacement_model=ARCH_ONE_SHOT,
            mixture_components=MIX_B_COMPONENTS,
            loss_config=loss_endkl,
            train_T_values=[1],
            train_T_weights=[1.0],
            eval_T_values=[1],
            tags=["v0.1", "p1", "a1", "architecture", "baseline", "mix-b"],
        )
    )

    # P1-A2: Shared recurrent residual, Mix B, End + KL, T={2,4,6,8}
    configs.append(
        build_config(
            experiment_name="midflow_qwen_8to11_p1_a2_rrb_mixb_endkl_trainT-r2468",
            replacement_model=ARCH_RECURRENT,
            mixture_components=MIX_B_COMPONENTS,
            loss_config=loss_endkl,
            train_T_values=[2, 4, 6, 8],
            train_T_weights=[0.25, 0.25, 0.25, 0.25],
            eval_T_values=[1, 2, 4, 8],
            tags=["v0.1", "p1", "a2", "architecture", "recurrent", "mix-b"],
        )
    )

    # P1-A3: Flow midblock, Mix B, End + KL, T={2,4,6,8}
    configs.append(
        build_config(
            experiment_name="midflow_qwen_8to11_p1_a3_flow_mixb_endkl_trainT-r2468",
            replacement_model=ARCH_FLOW,
            mixture_components=MIX_B_COMPONENTS,
            loss_config=loss_endkl,
            train_T_values=[2, 4, 6, 8],
            train_T_weights=[0.25, 0.25, 0.25, 0.25],
            eval_T_values=[1, 2, 4, 8],
            tags=["v0.1", "p1", "a3", "architecture", "flow", "mix-b"],
        )
    )

    return configs


def generate_p2_configs():
    """Generate Phase 2: Loss Ablation configs."""
    configs = []

    # Using Flow midblock as default for loss ablations
    # (can be changed based on P1 results)

    # L1: End only
    loss_end = {
        "velocity_weight": 0.0,
        "endpoint_weight": 1.0,
        "trajectory_weight": 0.0,
        "kl_weight": 0.0,
        "teacher_logits_source": "online",
        "ce_weight": 0.0,
    }

    # L2: End + KL
    loss_endkl = {
        "velocity_weight": 0.0,
        "endpoint_weight": 1.0,
        "trajectory_weight": 0.0,
        "kl_weight": 0.5,
        "teacher_logits_source": "online",
        "ce_weight": 0.0,
    }

    # L3: End + Traj + KL
    loss_endtrajkl = {
        "velocity_weight": 0.0,
        "endpoint_weight": 1.0,
        "trajectory_weight": 1.0,
        "kl_weight": 0.5,
        "teacher_logits_source": "online",
        "ce_weight": 0.0,
    }

    # L4: End + Traj + KL + CE
    loss_endtrajklce = {
        "velocity_weight": 0.0,
        "endpoint_weight": 1.0,
        "trajectory_weight": 1.0,
        "kl_weight": 0.5,
        "teacher_logits_source": "online",
        "ce_weight": 0.1,
    }

    # P2-L1: Flow, Mix B, End only
    configs.append(
        build_config(
            experiment_name="midflow_qwen_8to11_p2_l1_flow_mixb_end_trainT-r2468",
            replacement_model=ARCH_FLOW,
            mixture_components=MIX_B_COMPONENTS,
            loss_config=loss_end,
            train_T_values=[2, 4, 6, 8],
            train_T_weights=[0.25, 0.25, 0.25, 0.25],
            eval_T_values=[1, 2, 4, 8],
            tags=["v0.1", "p2", "l1", "loss", "endpoint-only", "mix-b"],
        )
    )

    # P2-L2: Flow, Mix B, End + KL
    configs.append(
        build_config(
            experiment_name="midflow_qwen_8to11_p2_l2_flow_mixb_endkl_trainT-r2468",
            replacement_model=ARCH_FLOW,
            mixture_components=MIX_B_COMPONENTS,
            loss_config=loss_endkl,
            train_T_values=[2, 4, 6, 8],
            train_T_weights=[0.25, 0.25, 0.25, 0.25],
            eval_T_values=[1, 2, 4, 8],
            tags=["v0.1", "p2", "l2", "loss", "end+kl", "mix-b"],
        )
    )

    # P2-L3: Flow, Mix B, End + Traj + KL
    configs.append(
        build_config(
            experiment_name="midflow_qwen_8to11_p2_l3_flow_mixb_endtrajkl_trainT-r2468",
            replacement_model=ARCH_FLOW,
            mixture_components=MIX_B_COMPONENTS,
            loss_config=loss_endtrajkl,
            train_T_values=[2, 4, 6, 8],
            train_T_weights=[0.25, 0.25, 0.25, 0.25],
            eval_T_values=[1, 2, 4, 8],
            tags=["v0.1", "p2", "l3", "loss", "end+traj+kl", "mix-b"],
        )
    )

    # P2-L4: Flow, Mix B, End + Traj + KL + CE
    configs.append(
        build_config(
            experiment_name="midflow_qwen_8to11_p2_l4_flow_mixb_endtrajklce_trainT-r2468",
            replacement_model=ARCH_FLOW,
            mixture_components=MIX_B_COMPONENTS,
            loss_config=loss_endtrajklce,
            train_T_values=[2, 4, 6, 8],
            train_T_weights=[0.25, 0.25, 0.25, 0.25],
            eval_T_values=[1, 2, 4, 8],
            tags=["v0.1", "p2", "l4", "loss", "end+traj+kl+ce", "mix-b"],
        )
    )

    return configs


def generate_p3_configs():
    """Generate Phase 3: Data Mix Ablation configs."""
    configs = []

    # Using End + Traj + KL as default for data ablations
    # (can be changed based on P2 results)
    loss_endtrajkl = {
        "velocity_weight": 0.0,
        "endpoint_weight": 1.0,
        "trajectory_weight": 1.0,
        "kl_weight": 0.5,
        "teacher_logits_source": "online",
        "ce_weight": 0.0,
    }

    # P3-D1: Flow, Mix A, End + Traj + KL
    configs.append(
        build_config(
            experiment_name="midflow_qwen_8to11_p3_d1_flow_mixa_endtrajkl_trainT-r2468",
            replacement_model=ARCH_FLOW,
            mixture_components=MIX_A_COMPONENTS,
            loss_config=loss_endtrajkl,
            train_T_values=[2, 4, 6, 8],
            train_T_weights=[0.25, 0.25, 0.25, 0.25],
            eval_T_values=[1, 2, 4, 8],
            tags=["v0.1", "p3", "d1", "data", "mix-a"],
        )
    )

    # P3-D2: Flow, Mix B, End + Traj + KL
    configs.append(
        build_config(
            experiment_name="midflow_qwen_8to11_p3_d2_flow_mixb_endtrajkl_trainT-r2468",
            replacement_model=ARCH_FLOW,
            mixture_components=MIX_B_COMPONENTS,
            loss_config=loss_endtrajkl,
            train_T_values=[2, 4, 6, 8],
            train_T_weights=[0.25, 0.25, 0.25, 0.25],
            eval_T_values=[1, 2, 4, 8],
            tags=["v0.1", "p3", "d2", "data", "mix-b"],
        )
    )

    # P3-D3: Flow, Mix C, End + Traj + KL
    configs.append(
        build_config(
            experiment_name="midflow_qwen_8to11_p3_d3_flow_mixc_endtrajkl_trainT-r2468",
            replacement_model=ARCH_FLOW,
            mixture_components=MIX_C_COMPONENTS,
            loss_config=loss_endtrajkl,
            train_T_values=[2, 4, 6, 8],
            train_T_weights=[0.25, 0.25, 0.25, 0.25],
            eval_T_values=[1, 2, 4, 8],
            tags=["v0.1", "p3", "d3", "data", "mix-c"],
        )
    )

    return configs


def generate_p4_configs():
    """Generate Phase 4: T Sweep configs (eval only)."""
    configs = []

    # Using best from P3 (Flow, Mix C, End + Traj + KL)
    # but with different eval T values
    loss_endtrajkl = {
        "velocity_weight": 0.0,
        "endpoint_weight": 1.0,
        "trajectory_weight": 1.0,
        "kl_weight": 0.5,
        "teacher_logits_source": "online",
        "ce_weight": 0.0,
    }

    # P4 configs for each eval T value
    for eval_T in [1, 2, 4, 8, 12]:
        configs.append(
            build_config(
                experiment_name=f"midflow_qwen_8to11_p4_flow_mixc_endtrajkl_evalT{eval_T}",
                replacement_model=ARCH_FLOW,
                mixture_components=MIX_C_COMPONENTS,
                loss_config=loss_endtrajkl,
                train_T_values=[2, 4, 6, 8],
                train_T_weights=[0.25, 0.25, 0.25, 0.25],
                eval_T_values=[eval_T],  # Single eval T for this config
                tags=["v0.1", "p4", "tsweep", f"evalT{eval_T}", "mix-c"],
            )
        )

    return configs


def save_config(config: dict, filename: str):
    """Save a config to YAML file."""
    filepath = CONFIGS_DIR / filename

    # Custom YAML representer to handle None values properly
    def represent_none(dumper, _):
        return dumper.represent_scalar("tag:yaml.org,2002:null", "null")

    yaml.add_representer(type(None), represent_none)

    with open(filepath, "w") as f:
        # Add header comment
        f.write(f"# {config['experiment_name']}\n")
        f.write(f"# Phase: {config['wandb']['tags'][1].upper()}\n")
        f.write(f"# Architecture: {config['replacement_model']['family']}\n")
        f.write(f"# Loss: endpoint={config['loss']['endpoint_weight']}, ")
        f.write(f"trajectory={config['loss']['trajectory_weight']}, ")
        f.write(f"kl={config['loss']['kl_weight']}, ")
        f.write(f"ce={config['loss']['ce_weight']}\n")
        f.write(f"# Train T: {config['model']['train_T_values']}\n")
        f.write(f"# Eval T: {config['eval']['eval_T_values']}\n")
        f.write(f"# Data: {len(config['data']['mixture_components'])} components\n")
        f.write("#\n")
        f.write(
            "# This config was auto-generated by scripts/generate_v0_1_configs.py\n"
        )
        f.write("# Do not edit manually; regenerate instead.\n\n")

        # Write the actual config
        yaml.dump(
            config, f, default_flow_style=False, sort_keys=False, allow_unicode=True
        )

    print(f"  ✓ {filename}")


def main():
    """Generate all v0.1 experiment configs."""
    print("Generating v0.1 Experiment Matrix Configs")
    print("=" * 50)

    # Ensure directory exists
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    # Generate Phase 1 configs
    print("\nPhase 1: Architecture Sanity (Mix B)")
    for config in generate_p1_configs():
        safe_name = config["experiment_name"].replace("-", "_")
        save_config(config, f"{safe_name}.yaml")

    # Generate Phase 2 configs
    print("\nPhase 2: Loss Ablations (Flow, Mix B)")
    for config in generate_p2_configs():
        safe_name = config["experiment_name"].replace("-", "_")
        save_config(config, f"{safe_name}.yaml")

    # Generate Phase 3 configs
    print("\nPhase 3: Data Mix Ablations (Flow, End+Traj+KL)")
    for config in generate_p3_configs():
        safe_name = config["experiment_name"].replace("-", "_")
        save_config(config, f"{safe_name}.yaml")

    # Generate Phase 4 configs
    print("\nPhase 4: T Sweep (Flow, Mix C, End+Traj+KL)")
    for config in generate_p4_configs():
        safe_name = config["experiment_name"].replace("-", "_")
        save_config(config, f"{safe_name}.yaml")

    # Generate README
    print("\nGenerating README...")
    readme_path = CONFIGS_DIR / "README.md"
    with open(readme_path, "w") as f:
        f.write("# v0.1 Experiment Matrix Configs\n\n")
        f.write(
            "This directory contains all YAML configs for the v0.1 experiment matrix.\n\n"
        )
        f.write("## Structure\n\n")
        f.write("- `p1_*`: Phase 1 - Architecture Sanity (3 configs)\n")
        f.write("  - A1: One-shot projector\n")
        f.write("  - A2: Shared recurrent residual\n")
        f.write("  - A3: Flow midblock\n")
        f.write("- `p2_*`: Phase 2 - Loss Ablations (4 configs)\n")
        f.write("  - L1: Endpoint only\n")
        f.write("  - L2: Endpoint + KL\n")
        f.write("  - L3: Endpoint + Trajectory + KL\n")
        f.write("  - L4: Endpoint + Trajectory + KL + CE\n")
        f.write("- `p3_*`: Phase 3 - Data Mix Ablations (3 configs)\n")
        f.write("  - D1: Mix A (FineWeb only)\n")
        f.write("  - D2: Mix B (FineWeb + UltraChat)\n")
        f.write("  - D3: Mix C (Full mix)\n")
        f.write("- `p4_*`: Phase 4 - T Sweep (5 configs)\n")
        f.write("  - Eval at T=1, 2, 4, 8, 12\n\n")
        f.write("## Regeneration\n\n")
        f.write("To regenerate all configs:\n\n")
        f.write("```bash\n")
        f.write("python scripts/generate_v0_1_configs.py\n")
        f.write("```\n\n")
        f.write("## Hardware Profile\n\n")
        f.write("All configs use the same hardware profile:\n")
        f.write("- seq_len: 1024\n")
        f.write("- batch_size: 3 (microbatch)\n")
        f.write("- accumulate_grad_batches: 5\n")
        f.write("- effective_batch_size: 15\n")
        f.write("- precision: bf16-mixed\n")
        f.write("- gradient_checkpointing: true\n\n")
        f.write("See `profiles/v0_1_3090_profile.json` for details.\n")

    print(f"  ✓ {readme_path.name}")

    print("\n" + "=" * 50)
    print("Done! Generated 15 configs in configs/v0_1_matrix/")


if __name__ == "__main__":
    main()
