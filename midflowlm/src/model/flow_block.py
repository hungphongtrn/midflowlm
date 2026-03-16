"""
DEPRECATED / PLACEHOLDER MODULE

This module is no longer the primary abstraction for model architecture.
It has been replaced by src.model.qwen_parity which provides proper Qwen
boundary extraction using Hugging Face transformers modules.

DO NOT ADD NEW CODE HERE. Use src.model.qwen_parity instead.

This placeholder exists only for backwards compatibility and will be removed
in a future version.
"""

import warnings

warnings.warn(
    "src.model.flow_block is deprecated. Use src.model.qwen_parity instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from qwen_parity for backwards compatibility
from src.model.qwen_parity import (
    QwenInspector,
    BypassWrapper,
    get_frozen_parameter_count,
    get_trainable_parameter_count,
    get_total_parameter_count,
)

__all__ = [
    "QwenInspector",
    "BypassWrapper",
    "get_frozen_parameter_count",
    "get_trainable_parameter_count",
    "get_total_parameter_count",
]
