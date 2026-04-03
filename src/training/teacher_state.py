"""Teacher-state mode contract for midflowlm KL training.

This module centralizes the control-plane contract for teacher-state modes:
- offline_cache: Uses pre-built teacher cache (existing offline path)
- online_no_cache: Uses live teacher extraction, no cache writing
- online_write_through_cache: Uses live teacher extraction with cache writing

Runtime behavior (Tasks 5/6) will route through these helpers:
- create_dataloaders() delegates based on mode
- Trainer computes or consumes teacher states per mode
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

from src.training.data import validate_cache_compatibility as _validate_cache_compat


class TeacherStateMode(Enum):
    OFFLINE_CACHE = "offline_cache"
    ONLINE_NO_CACHE = "online_no_cache"
    ONLINE_WRITE_THROUGH_CACHE = "online_write_through_cache"

    def requires_cache(self) -> bool:
        return self == TeacherStateMode.OFFLINE_CACHE

    def requires_live_teacher(self) -> bool:
        return self in (
            TeacherStateMode.ONLINE_NO_CACHE,
            TeacherStateMode.ONLINE_WRITE_THROUGH_CACHE,
        )

    def allow_cache_write(self) -> bool:
        return self == TeacherStateMode.ONLINE_WRITE_THROUGH_CACHE


VALID_MODES = {"offline_cache", "online_no_cache", "online_write_through_cache"}


def resolve_teacher_state_mode(config: Dict[str, Any]) -> str:
    """Resolve the teacher_state mode from config.

    Args:
        config: Configuration dictionary

    Returns:
        One of: "offline_cache", "online_no_cache", "online_write_through_cache"

    Raises:
        ValueError: If mode is not a recognized value
    """
    teacher_state = config.get("teacher_state", {})
    mode = teacher_state.get("mode", "offline_cache")

    if mode not in VALID_MODES:
        raise ValueError(
            f"Invalid teacher_state.mode: {mode!r}. Valid modes: {sorted(VALID_MODES)}"
        )

    return mode


def get_teacher_state_mode(config: Dict[str, Any]) -> TeacherStateMode:
    """Get the TeacherStateMode enum from config.

    Args:
        config: Configuration dictionary

    Returns:
        TeacherStateMode enum value
    """
    mode_str = resolve_teacher_state_mode(config)
    return TeacherStateMode(mode_str)


def validate_teacher_state_config(config: Dict[str, Any]) -> None:
    """Validate teacher_state config for the resolved mode.

    Args:
        config: Configuration dictionary

    Raises:
        ValueError: If config is invalid for the mode
    """
    mode = get_teacher_state_mode(config)
    teacher_state = config.get("teacher_state", {})
    teacher_cache = config.get("teacher_cache", {})
    model_config = config.get("model", {})

    if mode == TeacherStateMode.OFFLINE_CACHE:
        cache_dir = teacher_cache.get("cache_dir", "./cache")
        cache_path = Path(cache_dir)
        if not cache_path.exists():
            raise ValueError(
                f"offline_cache mode requires cache_dir to exist: {cache_dir}"
            )
        _validate_cache_compat(config, cache_dir)

    elif mode == TeacherStateMode.ONLINE_NO_CACHE:
        model_name = model_config.get("name")
        if not model_name:
            raise ValueError(
                "online_no_cache mode requires model.name to be set for live teacher extraction"
            )

    elif mode == TeacherStateMode.ONLINE_WRITE_THROUGH_CACHE:
        if not teacher_cache.get("enabled", False):
            raise ValueError(
                "online_write_through_cache mode requires teacher_cache.enabled=True"
            )
        model_name = model_config.get("name")
        if not model_name:
            raise ValueError(
                "online_write_through_cache mode requires model.name to be set"
            )
