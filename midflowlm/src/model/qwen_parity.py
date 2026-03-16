"""
Qwen parity inspection module.

This module provides utilities for extracting hidden states and logits from
Qwen models at configurable layer boundaries. It reuses Hugging Face transformers
Qwen modules and does NOT reimplement decoder internals.

Key features:
- Extract embeddings output
- Extract h_start (hidden state before replacement span)
- Extract hidden states for each layer inside replacement span
- Extract final logits
- Bypass/no-op wrapper for exact teacher comparison
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


class QwenInspector:
    """
    Inspector for Qwen models that extracts hidden states at layer boundaries.

    Reuses Hugging Face transformers Qwen modules - does NOT reimplement
    decoder internals like RMSNorm, SwiGLU, or GQA.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-0.8B",
        start_layer: int = 8,
        end_layer: int = 11,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize QwenInspector.

        Args:
            model_name: HuggingFace model name (default: Qwen/Qwen3.5-0.8B)
            start_layer: First layer of replacement span (inclusive)
            end_layer: Last layer of replacement span (inclusive)
            device: Device to load model on
            dtype: Data type for model weights
        """
        self.model_name = model_name
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.device = device
        self.dtype = dtype
        self.span_depth = end_layer - start_layer + 1

        # Load model and tokenizer
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=self.config,
            torch_dtype=dtype,
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Freeze all parameters - this is for teacher/inspection only
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()

        # Get number of layers in the model
        # Handle different config attribute names across model versions
        # Some multimodal configs have text_config with num_hidden_layers
        if hasattr(self.config, 'num_hidden_layers'):
            self.num_layers = self.config.num_hidden_layers
        elif hasattr(self.config, 'num_layers'):
            self.num_layers = self.config.num_layers
        elif hasattr(self.config, 'text_config') and hasattr(self.config.text_config, 'num_hidden_layers'):
            self.num_layers = self.config.text_config.num_hidden_layers
        elif hasattr(self.config, 'text_config') and hasattr(self.config.text_config, 'num_layers'):
            self.num_layers = self.config.text_config.num_layers
        else:
            raise AttributeError("Config has no num_hidden_layers or num_layers attribute")

        # Validate layer indices
        if start_layer < 0 or end_layer >= self.num_layers:
            raise ValueError(
                f"Invalid layer range: start_layer={start_layer}, end_layer={end_layer}, "
                f"but model only has {self.num_layers} layers (0-{self.num_layers-1})"
            )

    def extract_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract embeddings output (before any transformer layers).

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Embeddings tensor [batch_size, seq_len, hidden_size]
        """
        with torch.no_grad():
            # Get embeddings from the model's embed_tokens layer
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
                # For models like Qwen2/3 with nested structure
                embeddings = self.model.model.embed_tokens(input_ids)
            elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'wte'):
                # For GPT-style models
                embeddings = self.model.transformer.wte(input_ids)
            else:
                # Fallback: try to find embeddings
                embeddings = self.model.get_input_embeddings()(input_ids)

        return embeddings

    def extract_h_start(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract hidden state immediately before the replacement span.

        This runs the model through layers 0 to start_layer-1 and returns
        the hidden state that would be fed into layer start_layer.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Hidden state tensor [batch_size, seq_len, hidden_size]
        """
        with torch.no_grad():
            # Use the model's forward with output_hidden_states to get all layer outputs
            # Then pick the one at start_layer (which is the input to layer start_layer)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

            # hidden_states[0] is embeddings output
            # hidden_states[i] is the output of layer i-1
            # So h_start (input to layer start_layer) is hidden_states[start_layer]
            hidden_states_list = outputs.hidden_states
            h_start = hidden_states_list[self.start_layer]

        return h_start

    def extract_span_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Extract hidden states for each layer inside the replacement span.

        Returns the output of each layer from start_layer to end_layer (inclusive).

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            List of hidden state tensors, one per layer in the span
        """
        with torch.no_grad():
            # Use the model's forward with output_hidden_states to get all layer outputs
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

            # hidden_states[0] is embeddings output
            # hidden_states[i] is the output of layer i-1
            # So span_states are hidden_states[start_layer+1] to hidden_states[end_layer+1]
            hidden_states_list = outputs.hidden_states
            span_states = []
            for layer_idx in range(self.start_layer, self.end_layer + 1):
                # hidden_states[layer_idx+1] is the output of layer_idx
                span_states.append(hidden_states_list[layer_idx + 1])

        return span_states

    def extract_h_target(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract h_target - the hidden state after the replacement span.

        This is the output of end_layer, which would be fed into end_layer+1.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Hidden state tensor [batch_size, seq_len, hidden_size]
        """
        span_states = self.extract_span_states(input_ids, attention_mask)
        return span_states[-1]  # Last state in the span

    def extract_final_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract final logits from the teacher model.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Logits tensor [batch_size, seq_len, vocab_size]
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits

        return logits

    def extract_all(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Extract all teacher outputs at once.

        Returns:
            Dictionary with keys:
                - embeddings: Initial embeddings
                - h_start: Hidden state before replacement span
                - span_states: List of hidden states within span
                - h_target: Hidden state after replacement span
                - logits: Final logits
        """
        embeddings = self.extract_embeddings(input_ids, attention_mask)
        h_start = self.extract_h_start(input_ids, attention_mask)
        span_states = self.extract_span_states(input_ids, attention_mask)
        h_target = span_states[-1] if span_states else h_start
        logits = self.extract_final_logits(input_ids, attention_mask)

        return {
            "embeddings": embeddings,
            "h_start": h_start,
            "span_states": span_states,
            "h_target": h_target,
            "logits": logits,
        }

    def _get_base_model(self):
        """Get the base model (handles different HF model structures)."""
        if hasattr(self.model, 'model'):
            return self.model.model
        elif hasattr(self.model, 'transformer'):
            return self.model.transformer
        else:
            return self.model


class BypassWrapper(nn.Module):
    """
    Bypass/no-op wrapper for exact teacher comparison.

    This wrapper runs the full Qwen model without any modifications,
    allowing for exact comparison with teacher outputs.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-0.8B",
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize BypassWrapper.

        Args:
            model_name: HuggingFace model name
            start_layer: Ignored for bypass (for API compatibility)
            end_layer: Ignored for bypass (for API compatibility)
            device: Device to load model on
            dtype: Data type for model weights
        """
        super().__init__()
        self.model_name = model_name
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.device = device
        self.dtype = dtype

        # Load the full model
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=self.config,
            torch_dtype=dtype,
        ).to(device)

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass returning final logits.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Logits tensor [batch_size, seq_len, vocab_size]
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        return outputs.logits

    def forward_with_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass returning logits and hidden states.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Dictionary with logits, h_start, span_states, h_target
        """
        # Create a temporary inspector to extract hidden states
        inspector = QwenInspector(
            model_name=self.model_name,
            start_layer=self.start_layer or 8,
            end_layer=self.end_layer or 11,
            device=self.device,
            dtype=self.dtype,
        )

        outputs = inspector.extract_all(input_ids, attention_mask)
        return outputs


def get_frozen_parameter_count(model: nn.Module) -> int:
    """Get number of frozen parameters."""
    return sum(p.numel() for p in model.parameters() if not p.requires_grad)


def get_trainable_parameter_count(model: nn.Module) -> int:
    """Get number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_total_parameter_count(model: nn.Module) -> int:
    """Get total number of parameters."""
    return sum(p.numel() for p in model.parameters())
