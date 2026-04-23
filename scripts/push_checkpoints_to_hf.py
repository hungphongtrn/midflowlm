#!/usr/bin/env python3
"""Push MidFlowLM Phase 1 checkpoints to HuggingFace Hub.

This script uploads trained model checkpoints (P1-A1, P1-A2, P1-A3) to
HuggingFace Hub for easy download and inference on local machines.

Authentication (checked in order):
    1. --token argument
    2. HF_TOKEN environment variable  
    3. HUGGING_FACE_HUB_TOKEN environment variable
    4. Locally stored token from `huggingface-cli login`

Usage:
    # If already logged in via huggingface-cli, just run:
    uv run python scripts/push_checkpoints_to_hf.py --all

    # Or set token via environment variable:
    export HF_TOKEN=your_hf_token_here
    uv run python scripts/push_checkpoints_to_hf.py --all

    # Or use --token directly:
    uv run python scripts/push_checkpoints_to_hf.py --all --token your_token_here

    # Push specific checkpoints:
    uv run python scripts/push_checkpoints_to_hf.py --p1-a1 --p1-a2

    # Push with custom repo name:
    uv run python scripts/push_checkpoints_to_hf.py --all --repo-id your-username/midflowlm-phase1

    # Download locally after pushing:
    from huggingface_hub import hf_hub_download
    checkpoint_path = hf_hub_download(
        repo_id="hungphongtrn/midflowlm-phase1",
        filename="p1_a3_flow_midblock/checkpoint.pth",
        local_dir="./downloads"
    )

Repository Structure on HF Hub:
    hungphongtrn/midflowlm-phase1/
    ├── p1_a1_projector/
    │   ├── checkpoint.pth          # Model weights
    │   ├── config.yaml             # Training config
    │   ├── experiment_info.json    # Metadata (loss, T values, etc.)
    │   └── README.md               # Model card
    ├── p1_a2_recurrent_residual/
    │   ├── checkpoint.pth
    │   ├── config.yaml
    │   ├── experiment_info.json
    │   └── README.md
    └── p1_a3_flow_midblock/
        ├── checkpoint.pth
        ├── config.yaml
        ├── experiment_info.json
        └── README.md

"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from huggingface_hub import HfApi, create_repo, upload_file, hf_hub_download
    from huggingface_hub.utils import RepositoryNotFoundError
    from huggingface_hub.constants import HF_TOKEN_PATH
    from huggingface_hub._login import get_token
except ImportError:
    print("Error: huggingface_hub not installed. Install with: uv add huggingface-hub")
    sys.exit(1)

import torch
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Default repository ID
DEFAULT_REPO_ID = "hungphongtrn/midflowlm-phase1"

# Experiment configurations
EXPERIMENTS = {
    "p1_a1": {
        "name": "P1-A1: One-shot Projector",
        "subdir": "p1_a1_projector",
        "config": "configs/v0_1_matrix/midflow_qwen_8to11_p1_a1_proj_mixb_endkl.yaml",
        "checkpoint": "./outputs/midflow_qwen_8to11_p1_a1_proj_mixb_endkl/checkpoints/best.ckpt",
        "architecture": "projector",
        "train_T": [1],
        "eval_T": [1],
        "wandb_run": None,  # Fill in if available
    },
    "p1_a2": {
        "name": "P1-A2: Shared Recurrent Residual",
        "subdir": "p1_a2_recurrent_residual",
        "config": "configs/v0_1_matrix/midflow_qwen_8to11_p1_a2_rrb_mixb_endkl_trainT_r2468.yaml",
        "checkpoint": "./outputs/midflow_qwen_8to11_p1_a2_rrb_mixb_endkl_trainT_r2468/checkpoints/best.ckpt",
        "architecture": "shared_recurrent_residual",
        "train_T": [2, 4, 6, 8],
        "eval_T": [1, 2, 4, 8],
        "wandb_run": "ze54okvs",
    },
    "p1_a3": {
        "name": "P1-A3: Flow Midblock",
        "subdir": "p1_a3_flow_midblock",
        "config": "configs/v0_1_matrix/midflow_qwen_8to11_p1_a3_flow_mixb_endkl_trainT_r2468.yaml",
        "checkpoint": "./outputs/midflow_qwen_8to11_p1_a3_flow_mixb_endkl_trainT_r2468/checkpoints/best.ckpt",
        "architecture": "flow_midblock",
        "train_T": [2, 4, 6, 8],
        "eval_T": [1, 2, 4, 8],
        "wandb_run": "5q0mthbl",
    },
}


def load_checkpoint_info(checkpoint_path: str) -> Dict:
    """Load checkpoint and extract metadata.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Dictionary with checkpoint metadata
    """
    if not Path(checkpoint_path).exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return {}

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        info = {
            "has_model_state": "model_state_dict" in checkpoint,
            "has_optimizer_state": "optimizer_state_dict" in checkpoint,
            "has_scheduler_state": "scheduler_state_dict" in checkpoint,
            "keys": list(checkpoint.keys()),
        }

        # Extract training info if available
        if "global_step" in checkpoint:
            info["global_step"] = checkpoint["global_step"]
        if "epoch" in checkpoint:
            info["epoch"] = checkpoint["epoch"]
        if "val_loss" in checkpoint:
            info["val_loss"] = float(checkpoint["val_loss"])

        # Count parameters if model state available
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            total_params = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
            trainable_params = sum(
                p.numel() for k, p in state_dict.items()
                if isinstance(p, torch.Tensor) and not k.startswith("frozen_")
            )
            info["total_params"] = total_params
            info["trainable_params"] = trainable_params

        return info

    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        return {}


def create_model_card(experiment_key: str, config: Dict, checkpoint_info: Dict) -> str:
    """Create a README.md model card for the checkpoint.

    Args:
        experiment_key: Key like "p1_a1"
        config: Experiment configuration dictionary
        checkpoint_info: Checkpoint metadata

    Returns:
        Markdown content for README.md
    """
    exp = EXPERIMENTS[experiment_key]

    readme = f"""# {exp['name']}

## Model Description

This is a trained MidFlowLM student model checkpoint from Phase 1 of the v0.1 experiment matrix.

**Architecture:** {exp['architecture']}
**Base Model:** Qwen/Qwen3.5-0.8B
**Replacement Layers:** 8-11 (4 layers replaced)

## Training Configuration

- **Training T values:** {exp['train_T']}
- **Evaluation T values:** {exp['eval_T']}
- **Loss weights:** Endpoint=1.0, KL=0.5, Trajectory=0.0, CE=0.0
- **Data Mix:** Mix B (FineWeb-Edu + UltraChat)

## Checkpoint Info

"""

    if checkpoint_info:
        val_loss = checkpoint_info.get('val_loss')
        val_loss_str = f"{val_loss:.4f}" if val_loss is not None else "N/A"

        total_params = checkpoint_info.get('total_params')
        total_params_str = f"{total_params:,}" if total_params is not None else "N/A"

        trainable_params = checkpoint_info.get('trainable_params')
        trainable_params_str = f"{trainable_params:,}" if trainable_params is not None else "N/A"

        readme += f"""- **Global step:** {checkpoint_info.get('global_step', 'N/A')}
- **Epoch:** {checkpoint_info.get('epoch', 'N/A')}
- **Validation loss:** {val_loss_str}
- **Total parameters:** {total_params_str}
- **Trainable parameters:** {trainable_params_str}

"""

    if exp['wandb_run']:
        readme += f"""## Weights & Biases

- **Run ID:** {exp['wandb_run']}
- **Project:** midflowlm-v0-1

"""

    readme += f"""## Usage

### Load for Inference

```python
import torch
from src.model.student_qwen import FrozenQwenStudent

# Load student model
model = FrozenQwenStudent(
    model_name="Qwen/Qwen3.5-0.8B",
    start_layer=8,
    end_layer=11,
    max_steps_T=8,
    device="cuda",
    dtype=torch.bfloat16,
    bypass_mode=False,
)

# Load checkpoint
checkpoint = torch.load("checkpoint.pth", map_location="cuda")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Generate with T=4
with torch.no_grad():
    logits = model(input_ids, attention_mask, num_steps=4)
```

### Evaluation

See the [MidFlowLM repository](https://github.com/hungphongtrn/midflowlm) for evaluation scripts.

## Citation

```bibtex
@software{{midflowlm,
  author = {{Tran, Hung Phong}},
  title = {{MidFlowLM: Flow-Based Language Model Distillation}},
  year = {{2025}},
  url = {{https://github.com/hungphongtrn/midflowlm}}
}}
```

## License

This checkpoint is shared for research purposes. The base model (Qwen3.5-0.8B) follows its original license.
"""

    return readme


def push_checkpoint_to_hub(
    experiment_key: str,
    repo_id: str,
    token: str,
    create_if_missing: bool = True,
) -> bool:
    """Push a single checkpoint to HuggingFace Hub.

    Args:
        experiment_key: Key like "p1_a1"
        repo_id: HuggingFace Hub repository ID
        token: HuggingFace Hub token
        create_if_missing: Whether to create repo if it doesn't exist

    Returns:
        True if successful, False otherwise
    """
    exp = EXPERIMENTS[experiment_key]
    api = HfApi(token=token)

    # Check if checkpoint exists
    checkpoint_path = Path(exp["checkpoint"])
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.error(f"Please ensure training has completed and checkpoint exists at:")
        logger.error(f"  {checkpoint_path.absolute()}")
        return False

    # Load checkpoint info
    logger.info(f"Loading checkpoint info for {exp['name']}...")
    checkpoint_info = load_checkpoint_info(str(checkpoint_path))

    # Load config
    config_path = Path(exp["config"])
    if config_path.exists():
        with open(config_path) as f:
            config_data = yaml.safe_load(f)
    else:
        logger.warning(f"Config not found: {config_path}")
        config_data = {}

    # Create repo if needed
    if create_if_missing:
        try:
            create_repo(repo_id, token=token, exist_ok=True, repo_type="model")
            logger.info(f"Repository ready: {repo_id}")
        except Exception as e:
            logger.error(f"Failed to create repository: {e}")
            return False

    # Upload checkpoint
    subdir = exp["subdir"]
    try:
        logger.info(f"Uploading checkpoint for {exp['name']}...")

        # Upload checkpoint file
        upload_file(
            path_or_fileobj=str(checkpoint_path),
            path_in_repo=f"{subdir}/checkpoint.pth",
            repo_id=repo_id,
            token=token,
            repo_type="model",
        )
        logger.info(f"  ✓ checkpoint.pth uploaded")

        # Upload config
        if config_path.exists():
            upload_file(
                path_or_fileobj=str(config_path),
                path_in_repo=f"{subdir}/config.yaml",
                repo_id=repo_id,
                token=token,
                repo_type="model",
            )
            logger.info(f"  ✓ config.yaml uploaded")

        # Create and upload experiment info
        experiment_info = {
            "experiment_key": experiment_key,
            "name": exp["name"],
            "architecture": exp["architecture"],
            "train_T": exp["train_T"],
            "eval_T": exp["eval_T"],
            "wandb_run": exp["wandb_run"],
            "checkpoint_info": checkpoint_info,
            "base_model": "Qwen/Qwen3.5-0.8B",
            "start_layer": 8,
            "end_layer": 11,
        }

        info_path = f"/tmp/{experiment_key}_info.json"
        with open(info_path, "w") as f:
            json.dump(experiment_info, f, indent=2)

        upload_file(
            path_or_fileobj=info_path,
            path_in_repo=f"{subdir}/experiment_info.json",
            repo_id=repo_id,
            token=token,
            repo_type="model",
        )
        logger.info(f"  ✓ experiment_info.json uploaded")

        # Create and upload README
        readme = create_model_card(experiment_key, config_data, checkpoint_info)
        readme_path = f"/tmp/{experiment_key}_README.md"
        with open(readme_path, "w") as f:
            f.write(readme)

        upload_file(
            path_or_fileobj=readme_path,
            path_in_repo=f"{subdir}/README.md",
            repo_id=repo_id,
            token=token,
            repo_type="model",
        )
        logger.info(f"  ✓ README.md uploaded")

        logger.info(f"✅ Successfully pushed {exp['name']} to {repo_id}/{subdir}")
        return True

    except Exception as e:
        logger.error(f"Failed to upload {exp['name']}: {e}")
        return False


def download_checkpoint_locally(
    experiment_key: str,
    repo_id: str,
    local_dir: str,
    token: Optional[str] = None,
) -> bool:
    """Download a checkpoint from HuggingFace Hub to local machine.

    Args:
        experiment_key: Key like "p1_a1"
        repo_id: HuggingFace Hub repository ID
        local_dir: Local directory to save files
        token: Optional HuggingFace Hub token (for private repos)

    Returns:
        True if successful, False otherwise
    """
    exp = EXPERIMENTS[experiment_key]
    subdir = exp["subdir"]

    try:
        logger.info(f"Downloading {exp['name']} from {repo_id}...")

        # Download checkpoint
        checkpoint_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{subdir}/checkpoint.pth",
            local_dir=local_dir,
            token=token,
            repo_type="model",
        )
        logger.info(f"  ✓ checkpoint.pth -> {checkpoint_path}")

        # Download config
        config_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{subdir}/config.yaml",
            local_dir=local_dir,
            token=token,
            repo_type="model",
        )
        logger.info(f"  ✓ config.yaml -> {config_path}")

        # Download experiment info
        info_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{subdir}/experiment_info.json",
            local_dir=local_dir,
            token=token,
            repo_type="model",
        )
        logger.info(f"  ✓ experiment_info.json -> {info_path}")

        # Create best.ckpt symlink for compatibility with eval scripts
        import os
        subdir_path = os.path.join(local_dir, subdir)
        best_ckpt_path = os.path.join(subdir_path, "best.ckpt")
        checkpoint_real_path = os.path.join(subdir_path, "checkpoint.pth")

        if os.path.exists(checkpoint_real_path) and not os.path.exists(best_ckpt_path):
            try:
                os.symlink("checkpoint.pth", best_ckpt_path)
                logger.info(f"  ✓ Created best.ckpt -> checkpoint.pth symlink")
            except OSError:
                # Fallback: copy the file if symlinks aren't supported (Windows)
                import shutil
                shutil.copy2(checkpoint_real_path, best_ckpt_path)
                logger.info(f"  ✓ Created best.ckpt (copied from checkpoint.pth)")

        logger.info(f"✅ Successfully downloaded {exp['name']} to {local_dir}/{subdir}")
        return True

    except Exception as e:
        logger.error(f"Failed to download {exp['name']}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Push MidFlowLM Phase 1 checkpoints to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # If already logged in via huggingface-cli:
  uv run python scripts/push_checkpoints_to_hf.py --all

  # Or with explicit token:
  uv run python scripts/push_checkpoints_to_hf.py --all --token YOUR_TOKEN

  # Or with environment variable:
  HF_TOKEN=YOUR_TOKEN uv run python scripts/push_checkpoints_to_hf.py --all

  # Push specific checkpoints:
  uv run python scripts/push_checkpoints_to_hf.py --p1-a1 --p1-a2

  # Push to custom repo:
  uv run python scripts/push_checkpoints_to_hf.py --all --repo-id myusername/myrepo

  # Download checkpoints locally:
  uv run python scripts/push_checkpoints_to_hf.py --download --p1-a3 --local-dir ./models

Authentication Methods (checked in order):
  1. --token argument
  2. HF_TOKEN environment variable
  3. HUGGING_FACE_HUB_TOKEN environment variable
  4. Token from `huggingface-cli login` (stored in ~/.huggingface/token)
        """,
    )

    # Push options
    parser.add_argument("--all", action="store_true", help="Push all P1 checkpoints")
    parser.add_argument("--p1-a1", action="store_true", help="Push P1-A1 (One-shot Projector)")
    parser.add_argument("--p1-a2", action="store_true", help="Push P1-A2 (Shared Recurrent Residual)")
    parser.add_argument("--p1-a3", action="store_true", help="Push P1-A3 (Flow Midblock)")

    # Repo options
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_REPO_ID,
        help=f"HuggingFace Hub repository ID (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace Hub token (falls back to HF_TOKEN env var or huggingface-cli login)",
    )

    # Download options
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download mode: fetch checkpoints from HF Hub instead of pushing",
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default="./downloads",
        help="Local directory for downloads (default: ./downloads)",
    )

    args = parser.parse_args()

    # Get token from various sources (in order of priority):
    # 1. --token argument
    # 2. HF_TOKEN environment variable
    # 3. HUGGING_FACE_HUB_TOKEN environment variable
    # 4. Locally stored token from `huggingface-cli login`
    token = (
        args.token 
        or os.environ.get("HF_TOKEN") 
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or get_token()  # Reads from ~/.huggingface/token
    )
    
    if not token:
        if args.download:
            logger.warning("No token found. Public repos may still work, but private repos require authentication.")
        else:
            logger.error("HuggingFace token required to push checkpoints.")
            logger.error("Options:")
            logger.error("  1. Login with CLI: huggingface-cli login")
            logger.error("  2. Set HF_TOKEN environment variable")
            logger.error("  3. Use --token argument")
            logger.error("Get your token from: https://huggingface.co/settings/tokens")
            sys.exit(1)
    else:
        # Verify token source for user clarity
        if args.token:
            logger.info("Using token from --token argument")
        elif os.environ.get("HF_TOKEN"):
            logger.info("Using token from HF_TOKEN environment variable")
        elif os.environ.get("HUGGING_FACE_HUB_TOKEN"):
            logger.info("Using token from HUGGING_FACE_HUB_TOKEN environment variable")
        elif get_token():
            logger.info(f"Using token from locally stored credentials ({HF_TOKEN_PATH})")
            logger.info("(You logged in via: huggingface-cli login)")

    # Determine which experiments to process
    experiments_to_process = []
    if args.all:
        experiments_to_process = ["p1_a1", "p1_a2", "p1_a3"]
    else:
        if args.p1_a1:
            experiments_to_process.append("p1_a1")
        if args.p1_a2:
            experiments_to_process.append("p1_a2")
        if args.p1_a3:
            experiments_to_process.append("p1_a3")

    if not experiments_to_process:
        logger.error("No experiments selected. Use --all or --p1-a1/--p1-a2/--p1-a3")
        sys.exit(1)

    # Process each experiment
    success_count = 0
    for exp_key in experiments_to_process:
        if args.download:
            # Download mode
            if download_checkpoint_locally(exp_key, args.repo_id, args.local_dir, token):
                success_count += 1
        else:
            # Upload mode
            if push_checkpoint_to_hub(exp_key, args.repo_id, token):
                success_count += 1

    # Summary
    total = len(experiments_to_process)
    if args.download:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Download Summary: {success_count}/{total} successful")
        logger.info(f"Files saved to: {Path(args.local_dir).absolute()}")
        logger.info(f"{'=' * 60}")
    else:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Push Summary: {success_count}/{total} successful")
        logger.info(f"Repository: https://huggingface.co/{args.repo_id}")
        logger.info(f"{'=' * 60}")

        if success_count == total:
            logger.info("\n✅ All checkpoints pushed successfully!")
            logger.info("\nTo download on your local machine:")
            logger.info(f"  export HF_TOKEN=your_token_here")
            for exp_key in experiments_to_process:
                subdir = EXPERIMENTS[exp_key]["subdir"]
                logger.info(f"  python scripts/push_checkpoints_to_hf.py --download --{exp_key.replace('_', '-')} --local-dir ./models")
        else:
            logger.warning("\n⚠️ Some checkpoints failed to push. Check errors above.")
            sys.exit(1)


if __name__ == "__main__":
    main()
