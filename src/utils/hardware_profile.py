"""Hardware profile loader for v0.1 experiment matrix.

Provides utilities to load and apply hardware profiles from JSON artifacts.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional


def load_hardware_profile(profile_path: str) -> Dict[str, Any]:
    """Load a hardware profile from a JSON file.

    Args:
        profile_path: Path to the profile JSON file

    Returns:
        Dictionary containing hardware profile settings

    Raises:
        FileNotFoundError: If profile file doesn't exist
        ValueError: If profile is malformed
    """
    path = Path(profile_path)

    if not path.exists():
        # Try relative to project root
        project_root = Path(__file__).parent.parent.parent
        alt_path = project_root / profile_path
        if alt_path.exists():
            path = alt_path
        else:
            raise FileNotFoundError(f"Hardware profile not found: {profile_path}")

    with open(path, "r") as f:
        profile = json.load(f)

    # Validate required fields
    required_fields = [
        "hardware",
        "seq_len",
        "microbatch_size",
        "gradient_accumulation",
        "effective_batch_size",
        "precision",
        "gradient_checkpointing",
    ]

    missing = [f for f in required_fields if f not in profile]
    if missing:
        raise ValueError(f"Hardware profile missing required fields: {missing}")

    return profile


def apply_hardware_profile_to_config(
    config: Dict[str, Any],
    profile: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply a hardware profile to a training config.

    Updates config with batch size, gradient accumulation, precision,
    and gradient checkpointing settings from the profile.

    Args:
        config: Original training configuration dict
        profile: Hardware profile dict

    Returns:
        Updated configuration dict
    """
    # Deep copy to avoid modifying original
    import copy

    updated = copy.deepcopy(config)

    # Apply data settings
    if "data" not in updated:
        updated["data"] = {}
    updated["data"]["batch_size"] = profile["microbatch_size"]
    updated["data"]["seq_len"] = profile["seq_len"]

    # Apply training loop settings
    if "train_loop" not in updated:
        updated["train_loop"] = {}
    updated["train_loop"]["accumulate_grad_batches"] = profile["gradient_accumulation"]
    updated["train_loop"]["precision"] = profile["precision"]
    updated["train_loop"]["gradient_checkpointing"] = profile["gradient_checkpointing"]

    # Add metadata
    if "_hardware_profile" not in updated:
        updated["_hardware_profile"] = {}
    updated["_hardware_profile"]["source"] = profile.get("calibrated_on", "unknown")
    updated["_hardware_profile"]["peak_vram_gb"] = profile.get("peak_vram_gb", 0)
    updated["_hardware_profile"]["tokens_per_sec"] = profile.get("tokens_per_sec", 0)

    return updated


def get_default_profile_path() -> str:
    """Get the default hardware profile path for v0.1 experiments."""
    project_root = Path(__file__).parent.parent.parent
    return str(project_root / "profiles" / "v0_1_3090_profile.json")


def load_v0_1_profile() -> Dict[str, Any]:
    """Load the default v0.1 3090 profile.

    Returns:
        Hardware profile dict

    Raises:
        FileNotFoundError: If profile hasn't been calibrated yet
    """
    return load_hardware_profile(get_default_profile_path())


# Example profile structure for reference
EXAMPLE_PROFILE = {
    "hardware": "NVIDIA RTX 3090 24GB",
    "seq_len": 1024,
    "microbatch_size": 2,
    "gradient_accumulation": 8,
    "effective_batch_size": 16,
    "precision": "bf16-mixed",
    "gradient_checkpointing": True,
    "peak_vram_gb": 21.5,
    "tokens_per_sec": 420.0,
    "calibrated_on": "FlowMidblock_EndTrajKLCe_MixC",
    "calibration_timestamp": "2024-01-01T00:00:00Z",
}
