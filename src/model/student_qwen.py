"""
Frozen student wrapper around Qwen with trainable iterative midblock.

This module implements a student model that:
- Uses frozen Qwen layers outside the replacement span
- Uses a trainable IterativeMidblock within the replacement span
- Maintains HF-style output compatibility
- Supports bypass mode for teacher comparison
- Supports variable T values without model rebuild
"""

import logging
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union
from transformers import AutoModelForCausalLM, AutoConfig
from dataclasses import dataclass

from src.model.midblock import IterativeMidblock
from src.model.ode import MidblockVectorField, build_solver_options

logger = logging.getLogger(__name__)


class StudentOutput(dict):
    """Output class that acts like both a dict and an object.

    Supports:
    - dict-style access: output["logits"]
    - attribute access: output.logits
    - isinstance checks: isinstance(output, dict)
    """

    def __init__(
        self,
        logits: torch.Tensor,
        endpoint_hidden: Optional[torch.Tensor] = None,
        trajectory_hidden: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self["logits"] = logits
        self._logits = logits
        if endpoint_hidden is not None:
            self["endpoint_hidden"] = endpoint_hidden
            self._endpoint_hidden = endpoint_hidden
        if trajectory_hidden is not None:
            self["trajectory_hidden"] = trajectory_hidden
            self._trajectory_hidden = trajectory_hidden

    @property
    def logits(self) -> torch.Tensor:
        return self._logits

    @property
    def endpoint_hidden(self) -> Optional[torch.Tensor]:
        return getattr(self, "_endpoint_hidden", None)

    @property
    def trajectory_hidden(self) -> Optional[torch.Tensor]:
        return getattr(self, "_trajectory_hidden", None)

    def __getattr__(self, name: str) -> Any:
        if name == "logits":
            return self._logits
        elif name == "endpoint_hidden":
            return getattr(self, "_endpoint_hidden", None)
        elif name == "trajectory_hidden":
            return getattr(self, "_trajectory_hidden", None)
        raise AttributeError(
            f'"{type(self).__name__}" object has no attribute "{name}"'
        )

    def __repr__(self) -> str:
        return f"StudentOutput(logits={self._logits.shape})"


def get_frozen_parameter_count(model: nn.Module) -> int:
    """Get number of frozen parameters."""
    return sum(p.numel() for p in model.parameters() if not p.requires_grad)


def get_trainable_parameter_count(model: nn.Module) -> int:
    """Get number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_total_parameter_count(model: nn.Module) -> int:
    """Get total number of parameters."""
    return sum(p.numel() for p in model.parameters())


class FrozenQwenStudent(nn.Module):
    """
    Student model with frozen Qwen layers and trainable iterative midblock.

    Architecture:
    1. Embeddings (frozen)
    2. Lower Qwen layers [0:start_layer] (frozen)
    3. IterativeMidblock [start_layer:end_layer+1] (trainable)
    4. Upper Qwen layers [end_layer+1:] (frozen)
    5. Final norm and LM head (frozen)

    Args:
        model_name: HuggingFace model name (default: Qwen/Qwen3.5-0.8B)
        start_layer: First layer of replacement span (inclusive)
        end_layer: Last layer of replacement span (inclusive)
        max_steps_T: Maximum number of refinement steps
        device: Device to load model on
        dtype: Data type for model weights
        bypass_mode: If True, use full model without midblock (for comparison)
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-0.8B",
        start_layer: int = 8,
        end_layer: int = 11,
        max_steps_T: int = 8,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        bypass_mode: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.max_steps_T = max_steps_T
        self.device = device
        self.dtype = dtype
        self.bypass_mode = bypass_mode
        self.span_depth = end_layer - start_layer + 1

        # Load model config
        self.config = AutoConfig.from_pretrained(model_name)

        # Get number of layers
        if hasattr(self.config, "num_hidden_layers"):
            self.num_layers = self.config.num_hidden_layers
        elif hasattr(self.config, "num_layers"):
            self.num_layers = self.config.num_layers
        elif hasattr(self.config, "text_config") and hasattr(
            self.config.text_config, "num_hidden_layers"
        ):
            self.num_layers = self.config.text_config.num_hidden_layers
        elif hasattr(self.config, "text_config") and hasattr(
            self.config.text_config, "num_layers"
        ):
            self.num_layers = self.config.text_config.num_layers
        else:
            raise AttributeError(
                "Config has no num_hidden_layers or num_layers attribute"
            )

        # Validate layer indices
        if start_layer < 0 or end_layer >= self.num_layers:
            raise ValueError(
                f"Invalid layer range: start_layer={start_layer}, end_layer={end_layer}, "
                f"but model only has {self.num_layers} layers (0-{self.num_layers - 1})"
            )

        # Load the full Qwen model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=self.config,
            torch_dtype=dtype,
        ).to(device)

        if not bypass_mode:
            # Freeze all Qwen parameters
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

            # Get hidden size from config
            if hasattr(self.config, "hidden_size"):
                hidden_size = self.config.hidden_size
            elif hasattr(self.config, "text_config") and hasattr(
                self.config.text_config, "hidden_size"
            ):
                hidden_size = self.config.text_config.hidden_size
            else:
                raise AttributeError("Config has no hidden_size attribute")

            # Get number of heads for attention
            if hasattr(self.config, "num_attention_heads"):
                num_heads = self.config.num_attention_heads
            elif hasattr(self.config, "num_heads"):
                num_heads = self.config.num_heads
            elif hasattr(self.config, "text_config") and hasattr(
                self.config.text_config, "num_attention_heads"
            ):
                num_heads = self.config.text_config.num_attention_heads
            else:
                num_heads = 8  # Default for Qwen3.5-0.8B

            # Create the trainable midblock
            self.midblock = IterativeMidblock(
                hidden_size=hidden_size,
                max_steps_T=max_steps_T,
                start_layer=start_layer,
                end_layer=end_layer,
                num_heads=num_heads,
                mlp_ratio=4.0,
                dropout=0.0,
                qkv_bias=True,
                use_causal_mask=True,
                use_step_conditioning=True,
                use_residual=True,
                step_encoding_mode="combined",
            ).to(device)
        else:
            # In bypass mode, no midblock
            self.midblock = None
            # Still freeze parameters
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

    def _get_base_model(self) -> nn.Module:
        """Get the base model (handles different HF model structures)."""
        if hasattr(self.model, "model"):
            return self.model.model
        elif hasattr(self.model, "transformer"):
            return self.model.transformer
        else:
            return self.model

    def _extract_h_start(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract hidden state at start_layer (output of layer start_layer-1).

        Uses the model's forward with output_hidden_states to get the
        hidden state that would be fed into layer start_layer.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Hidden state tensor [batch_size, seq_len, hidden_size]
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            # hidden_states[0] is embeddings output
            # hidden_states[i] is the output of layer i-1
            # So h_start (input to layer start_layer) is hidden_states[start_layer]
            h_start = outputs.hidden_states[self.start_layer]
        return h_start

    def _run_upper_layers_from_hidden(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run upper frozen layers from a hidden state and return logits.

        This runs layers [end_layer+1:] and the LM head.

        Args:
            hidden_states: Hidden states from midblock [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Logits tensor [batch_size, seq_len, vocab_size]
        """
        with torch.no_grad():
            base_model = self._get_base_model()

            # Get the layers module
            if hasattr(base_model, "layers"):
                layers = base_model.layers
            else:
                raise AttributeError("Cannot find layers module in base model")

            # Run through layers end_layer+1 to num_layers-1
            # Note: We need to use the model's internal forward for each layer
            # This is tricky because we need to handle position embeddings properly
            # Instead, we'll use a simpler approach: run the full model and swap the span

            # Get the full hidden states to end_layer
            outputs = self.model(
                input_ids=None,  # We provide inputs_embeds instead
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            logits = outputs.logits

        return logits

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        solver_method: str = "euler",
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the student model.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            num_steps: Number of iterative refinement steps (default: max_steps_T)
            solver_method: ODE solver method ("euler", "rk4", "dopri5", etc.)
            return_dict: If True, return dict/HF-style output with hidden states

        Returns:
            Logits tensor or dict with logits, endpoint_hidden, trajectory_hidden
        """
        if self.bypass_mode:
            # In bypass mode, just run the full model
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits

            if return_dict:
                return StudentOutput(logits=logits)
            return logits
        else:
            # Use num_steps or default to max_steps_T
            if num_steps is None:
                num_steps = self.max_steps_T

            # Extract h_start using model's forward (handles all the complexity)
            h_start = self._extract_h_start(input_ids, attention_mask)

            # Create vector field wrapper for ODE integration
            vector_field = MidblockVectorField(
                midblock=self.midblock,
                h_start=h_start,
                attention_mask=attention_mask,
            )

            # Build solver options based on method and num_steps
            solver_options = build_solver_options(solver_method, num_steps)

            # Import torchdiffeq for ODE integration
            try:
                from torchdiffeq import odeint
            except ImportError:
                raise ImportError(
                    "torchdiffeq is required for ODE integration. "
                    "Install with: pip install torchdiffeq"
                )

            # Integration time points: start at 0, end at 1
            # Using [0, 1] normalizes inference time regardless of num_steps
            t = torch.tensor([0.0, 1.0], device=h_start.device, dtype=h_start.dtype)

            # Run ODE integration
            # odeint returns [time_points, batch, seq, hidden]
            solution = odeint(
                vector_field,
                h_start,
                t,
                method=solver_method,
                options=solver_options,
            )

            # Extract final state at t=1
            h_mid = solution[-1]

            # For trajectory tracking, interpolate at each step if return_dict
            if return_dict:
                # Create time points for each step to capture trajectory
                t_trajectory = torch.linspace(
                    0, 1, num_steps + 1, device=h_start.device
                )
                trajectory_solution = odeint(
                    vector_field,
                    h_start,
                    t_trajectory,
                    method=solver_method,
                    options=solver_options,
                )
                # Skip the initial state at t=0, take remaining steps
                # Shape: [num_steps+1, batch, seq, hidden] -> [num_steps, batch, seq, hidden]
                trajectory = trajectory_solution[1:]
                # Transpose to [batch, seq, num_steps, hidden]
                trajectory_stacked = trajectory.transpose(0, 1).transpose(1, 2)
            else:
                trajectory_stacked = None

            # Run upper layers from h_mid
            logits = self._continue_from_hidden_state(h_mid, attention_mask)

            if return_dict:
                return StudentOutput(
                    logits=logits,
                    endpoint_hidden=h_mid,
                    trajectory_hidden=trajectory_stacked,
                )
            else:
                return logits

    def _continue_from_hidden_state(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Continue forward pass from a hidden state through upper layers.

        This manually runs layers from end_layer+1 through the final layer
        and applies the LM head.

        Args:
            hidden_states: Hidden states to start from (output of end_layer)
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Logits tensor [batch_size, seq_len, vocab_size]
        """
        base_model = self._get_base_model()

        with torch.no_grad():
            # Get the layers module
            if hasattr(base_model, "layers"):
                layers = base_model.layers
            else:
                raise AttributeError("Cannot find layers module in base model")

            # Compute position embeddings if needed (Qwen3.5)
            position_embeddings = None
            if hasattr(base_model, "rotary_emb"):
                batch_size, seq_len, _ = hidden_states.shape
                position_ids = (
                    torch.arange(seq_len, device=hidden_states.device)
                    .unsqueeze(0)
                    .expand(batch_size, -1)
                )
                position_embeddings = base_model.rotary_emb(hidden_states, position_ids)

            # Manually run through layers end_layer+1 to num_layers-1
            h = hidden_states
            for layer_idx in range(self.end_layer + 1, self.num_layers):
                layer = layers[layer_idx]

                # Build layer kwargs based on signature
                import inspect

                sig = inspect.signature(layer.forward)
                layer_kwargs = {"hidden_states": h}

                # Note: We don't pass attention_mask because:
                # 1. The model uses causal masking by default
                # 2. The attention_mask format [batch, seq] doesn't work with
                #    the layer's expected format and causes shape mismatches
                # 3. For training, causal masking is typically what we want

                if (
                    "position_embeddings" in sig.parameters
                    and position_embeddings is not None
                ):
                    layer_kwargs["position_embeddings"] = position_embeddings

                # Use layer.forward directly to avoid shape issues with __call__
                layer_output = layer.forward(**layer_kwargs)

                # Layer returns a tuple: (hidden_states, ...) or just hidden_states
                h = layer_output[0] if isinstance(layer_output, tuple) else layer_output

            # Apply final norm if it exists
            if hasattr(base_model, "norm"):
                h = base_model.norm(h)

            # Apply LM head
            if hasattr(self.model, "lm_head"):
                logits = self.model.lm_head(h)
            elif hasattr(base_model, "lm_head"):
                logits = base_model.lm_head(h)
            else:
                # Last resort: try to get lm_head from the model
                logits = self.model.lm_head(h)

        return logits

    def get_trainable_parameter_count(self) -> int:
        """Get number of trainable parameters."""
        return get_trainable_parameter_count(self)

    def get_total_parameter_count(self) -> int:
        """Get total number of parameters."""
        return get_total_parameter_count(self)

    def get_frozen_parameter_count(self) -> int:
        """Get number of frozen parameters."""
        return get_frozen_parameter_count(self)

    def get_parameter_summary(self) -> Dict[str, int]:
        """Get summary of parameter counts."""
        return {
            "trainable": self.get_trainable_parameter_count(),
            "frozen": self.get_frozen_parameter_count(),
            "total": self.get_total_parameter_count(),
        }

    def freeze_all(self):
        """Freeze all parameters (for evaluation mode)."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_midblock(self):
        """Unfreeze only the midblock parameters."""
        if self.midblock is not None:
            for param in self.midblock.parameters():
                param.requires_grad = True

    def save_midblock(self, path: str):
        """Save only the midblock state dict."""
        if self.midblock is not None:
            torch.save(self.midblock.state_dict(), path)
        else:
            raise ValueError("No midblock to save (bypass mode)")

    def load_midblock(self, path: str):
        """Load midblock state dict."""
        if self.midblock is not None:
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
            self.midblock.load_state_dict(state_dict)
        else:
            raise ValueError("No midblock to load into (bypass mode)")

    def extract_teacher_targets(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract teacher targets from frozen Qwen using one forward pass.

        This method reuses the frozen Qwen weights already inside the student model
        to produce teacher targets for online-no-cache training. Uses a single
        forward pass with output_hidden_states=True.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Dictionary containing:
                - h_start: Hidden state at start_layer (input to layer start_layer)
                - h_target: Hidden state after end_layer (output of layer end_layer)
                - velocity_target: h_target - h_start (flow matching velocity)
                - teacher_logits: Final logits from the full teacher model
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

            hidden_states = outputs.hidden_states

            h_start = hidden_states[self.start_layer]
            h_target = hidden_states[self.end_layer + 1]
            velocity_target = h_target - h_start
            teacher_logits = outputs.logits

        return {
            "h_start": h_start,
            "h_target": h_target,
            "velocity_target": velocity_target,
            "teacher_logits": teacher_logits,
        }

    def gradient_checkpointing_enable(self, gradient_checkpointing_func=None):
        """Enable gradient checkpointing to reduce memory usage.

        This enables gradient checkpointing on the base Qwen model layers
        and the midblock to trade compute for memory during training.

        Args:
            gradient_checkpointing_func: Optional custom checkpointing function.
                If None, uses torch.utils.checkpoint.checkpoint.
        """
        if gradient_checkpointing_func is None:
            from torch.utils.checkpoint import checkpoint

            gradient_checkpointing_func = checkpoint

        # Enable gradient checkpointing on the base model
        base_model = self._get_base_model()

        # Enable HF-style gradient checkpointing if available
        if hasattr(base_model, "gradient_checkpointing_enable"):
            base_model.gradient_checkpointing_enable()
            logger.info("Enabled HF gradient checkpointing on base model")
        elif hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
            logger.info("Enabled HF gradient checkpointing on full model")

        # Store checkpointing function for manual use in forward
        self._gradient_checkpointing_func = gradient_checkpointing_func
        self._gradient_checkpointing_enabled = True

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        base_model = self._get_base_model()

        if hasattr(base_model, "gradient_checkpointing_disable"):
            base_model.gradient_checkpointing_disable()
        elif hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()

        self._gradient_checkpointing_enabled = False
        if hasattr(self, "_gradient_checkpointing_func"):
            delattr(self, "_gradient_checkpointing_func")
