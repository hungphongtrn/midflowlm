"""
SFT-compatible model wrapper: frozen Qwen3.5-0.8B with trainable FlowMidblock.

Monkey-patches layers 8..11 with a FlowMidblock for CE-only SFT training
via HuggingFace Trainer + AutoLigerKernelForCausalLM.
"""

import logging
import os
import torch
import torch.nn as nn
from typing import Optional, Dict
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast

from src.model.midblock import FlowMidblock

logger = logging.getLogger(__name__)


class SFTFlowMidblockModel(nn.Module):
    """Frozen Qwen with trainable FlowMidblock replacing layers start:end+1.

    Architecture:
        1. Qwen Embeddings (frozen)
        2. Qwen Layers 0..start_layer-1 (frozen)
        3. FlowMidblock replacing layers start_layer..end_layer (trainable, warm-start P3-D3)
        4. Qwen Layers end_layer+1..N (frozen)
        5. Final Norm + LM Head (frozen)

    Args:
        model_name: HuggingFace model name (default: Qwen/Qwen3.5-0.8B)
        start_layer: First layer of replacement span (inclusive)
        end_layer: Last layer of replacement span (inclusive)
        thinking_level: Number of FlowMidblock refinement steps (fixed T)
        checkpoint_path: Path to P3-D3 checkpoint for warm-start
        torch_dtype: Data type for model weights
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-0.8B",
        start_layer: int = 8,
        end_layer: int = 11,
        thinking_level: int = 32,
        checkpoint_path: Optional[str] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.model_name = model_name
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.thinking_level = thinking_level
        self.span_depth = end_layer - start_layer + 1
        self.torch_dtype = torch_dtype

        self.config = AutoConfig.from_pretrained(model_name)
        self._resolve_layer_count()
        self._resolve_hidden_size()
        self._resolve_num_heads()

        self._load_qwen()
        self._freeze_qwen()
        self.midblock = self._create_midblock()

        if checkpoint_path is not None:
            self._warm_start_midblock(checkpoint_path)

        self._patch_forward()

    def _resolve_layer_count(self):
        if hasattr(self.config, "num_hidden_layers"):
            self.num_layers = self.config.num_hidden_layers
        elif hasattr(self.config, "num_layers"):
            self.num_layers = self.config.num_layers
        elif hasattr(self.config, "text_config") and hasattr(self.config.text_config, "num_hidden_layers"):
            self.num_layers = self.config.text_config.num_hidden_layers
        elif hasattr(self.config, "text_config") and hasattr(self.config.text_config, "num_layers"):
            self.num_layers = self.config.text_config.num_layers
        else:
            raise AttributeError("Config has no num_hidden_layers or num_layers attribute")

        if self.start_layer < 0 or self.end_layer >= self.num_layers:
            raise ValueError(
                f"Invalid layer range: start_layer={self.start_layer}, end_layer={self.end_layer}, "
                f"model has {self.num_layers} layers (0-{self.num_layers - 1})"
            )

    def _resolve_hidden_size(self):
        if hasattr(self.config, "hidden_size"):
            self.hidden_size = self.config.hidden_size
        elif hasattr(self.config, "text_config") and hasattr(self.config.text_config, "hidden_size"):
            self.hidden_size = self.config.text_config.hidden_size
        else:
            raise AttributeError("Config has no hidden_size attribute")

    def _resolve_num_heads(self):
        if hasattr(self.config, "num_attention_heads"):
            self.num_heads = self.config.num_attention_heads
        elif hasattr(self.config, "num_heads"):
            self.num_heads = self.config.num_heads
        elif hasattr(self.config, "text_config") and hasattr(self.config.text_config, "num_attention_heads"):
            self.num_heads = self.config.text_config.num_attention_heads
        else:
            self.num_heads = 8

    def _load_qwen(self):
        try:
            from liger_kernel.transformers import AutoLigerKernelForCausalLM
            self.qwen = AutoLigerKernelForCausalLM.from_pretrained(
                self.model_name,
                config=self.config,
                torch_dtype=self.torch_dtype,
            )
            logger.info("Loaded Qwen via AutoLigerKernelForCausalLM")
        except ImportError as e:
            logger.warning(f"AutoLigerKernelForCausalLM not available ({e}), falling back to AutoModelForCausalLM")
            self.qwen = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                config=self.config,
                torch_dtype=self.torch_dtype,
            )

        self.base_model = self.qwen.model
        self.layers = self.base_model.layers
        self.embed_tokens = self.base_model.embed_tokens
        self.norm = self.base_model.norm
        self.lm_head = self.qwen.lm_head

    def _freeze_qwen(self):
        for param in self.qwen.parameters():
            param.requires_grad = False
        self.qwen.eval()

    def _create_midblock(self):
        return FlowMidblock(
            hidden_size=self.hidden_size,
            max_steps_T=self.thinking_level,
            start_layer=self.start_layer,
            end_layer=self.end_layer,
            num_heads=self.num_heads,
            mlp_ratio=4.0,
            dropout=0.0,
            qkv_bias=True,
            use_causal_mask=True,
            use_step_conditioning=True,
        )

    def _warm_start_midblock(self, checkpoint_path: str):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        if "model_state_dict" in checkpoint:
            # Old format: {"model_state_dict": {"midblock.time_proj.weight": ..., ...}}
            model_state = checkpoint["model_state_dict"]
            midblock_state = {}
            for key, value in model_state.items():
                if key.startswith("midblock."):
                    new_key = key[len("midblock."):]
                    midblock_state[new_key] = value
        elif "midblock_state_dict" in checkpoint:
            # Intermediate format: {"midblock_state_dict": {...}}
            midblock_state = checkpoint["midblock_state_dict"]
        elif isinstance(checkpoint, dict) and any("proj" in k or "norm" in k or "refiner" in k or "adapt" in k for k in list(checkpoint.keys())[:3]):
            # New format: flat FlowMidblock state_dict directly
            # (keys look like "time_proj.weight", "velocity_proj.1.weight", etc.)
            midblock_state = checkpoint
        else:
            raise ValueError(
                f"Unrecognized checkpoint format at {checkpoint_path}. "
                f"Expected 'model_state_dict', 'midblock_state_dict', or flat FlowMidblock keys. "
                f"Got top-level keys: {list(checkpoint.keys())[:5]}"
            )

        missing, unexpected = self.midblock.load_state_dict(midblock_state, strict=False)
        logger.info(f"Warm-started midblock from {checkpoint_path}")
        if missing:
            logger.warning(f"Missing keys: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys: {unexpected}")

    def _patch_forward(self):
        midblock = self.midblock
        start_layer = self.start_layer
        end_layer = self.end_layer
        embed_tokens = self.embed_tokens
        layers = self.layers
        norm = self.norm
        num_layers = self.num_layers
        rotary_emb = getattr(self.base_model, "rotary_emb", None)
        parent_model = self

        def patched_forward(
            self_model,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            inputs_embeds=None,
            **kwargs,
        ):
            if inputs_embeds is None:
                hidden_states = embed_tokens(input_ids)
            else:
                hidden_states = inputs_embeds

            # Qwen3.5 layers expect position_embeddings, not position_ids
            position_embeddings = None
            if rotary_emb is not None:
                batch_size, seq_len = hidden_states.shape[:2]
                if position_ids is None:
                    position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)
                position_embeddings = rotary_emb(hidden_states, position_ids)

            qwen_dtype = hidden_states.dtype

            # Qwen3.5 layers use causal masking by default internally.
            # Passing a 2D attention_mask directly causes shape mismatches with SDPA.
            # The FrozenQwenStudent reference also skips attention_mask for layer calls.
            for i in range(start_layer):
                layer_output = layers[i](hidden_states, position_embeddings=position_embeddings)
                hidden_states = layer_output[0] if isinstance(layer_output, tuple) else layer_output

            num_steps = kwargs.pop("num_steps", parent_model.thinking_level)
            hidden_states = midblock.iterative_refinement(
                h_start=hidden_states,
                num_steps=num_steps,
            )
            # FlowMidblock may produce float32; cast back to Qwen dtype
            hidden_states = hidden_states.to(qwen_dtype)

            for i in range(end_layer + 1, num_layers):
                layer_output = layers[i](hidden_states, position_embeddings=position_embeddings)
                hidden_states = layer_output[0] if isinstance(layer_output, tuple) else layer_output

            hidden_states = norm(hidden_states)
            return BaseModelOutputWithPast(last_hidden_state=hidden_states)

        self.base_model.forward = patched_forward.__get__(self.base_model, type(self.base_model))

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs,
    ):
        outputs = self.qwen(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )
        return outputs

    def state_dict(self, *args, **kwargs):
        """Return a deduplicated state_dict for safetensors compatibility.

        Qwen ties some weights (for example input embeddings and lm_head), and this
        wrapper also exposes aliases to the same underlying modules. Safetensors
        rejects duplicate references to the same storage. We keep only the first key
        per unique tensor storage in the returned state_dict.
        """
        raw_state = super().state_dict(*args, **kwargs)
        deduped_state = {}
        seen_storages = set()

        for key, tensor in raw_state.items():
            storage_key = (tensor.untyped_storage().data_ptr(), tensor.storage_offset(), tuple(tensor.size()))
            if storage_key in seen_storages:
                continue
            seen_storages.add(storage_key)
            deduped_state[key] = tensor

        return deduped_state

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: Optional[Dict] = None):
        if hasattr(self.qwen, "gradient_checkpointing_enable"):
            self.qwen.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        if hasattr(self.qwen, "gradient_checkpointing_disable"):
            self.qwen.gradient_checkpointing_disable()

    @property
    def trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def frozen_params(self):
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)

    @property
    def device(self):
        return next(self.parameters()).device
