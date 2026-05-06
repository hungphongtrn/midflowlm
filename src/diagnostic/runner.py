"""Deterministic trace runner for diagnostic probing."""
import json
import random
import torch
import numpy as np
import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

from src.diagnostic.probe import ProbeSet, ProbeExample
from transformers import AutoTokenizer


@dataclass
class TraceRecord:
    """A single trace record from running a probe through the model."""
    probe_id: str
    benchmark: str
    T: int
    seed: int
    endpoint_hidden_norm: float
    logits_answer_tokens: Dict[str, float]
    predicted_answer: str
    predicted_token_id: int
    full_logits_shape: str  # serialized shape for sanity check

    def to_dict(self) -> dict:
        """Serialize the trace record to a dictionary."""
        return {
            "probe_id": self.probe_id,
            "benchmark": self.benchmark,
            "T": self.T,
            "seed": self.seed,
            "endpoint_hidden_norm": self.endpoint_hidden_norm,
            "logits_answer_tokens": self.logits_answer_tokens,
            "predicted_answer": self.predicted_answer,
            "predicted_token_id": self.predicted_token_id,
            "full_logits_shape": self.full_logits_shape,
        }


def set_deterministic(seed: int):
    """Set all random seeds for reproducibility.
    
    Args:
        seed: The random seed to use for all random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DeterministicTraceRunner:
    """Runner for executing probe examples deterministically through a model.
    
    This runner ensures reproducible results by setting random seeds before
    each forward pass and capturing detailed trace information.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device: torch.device,
        seed: int = 42,
    ):
        """Initialize the trace runner.
        
        Args:
            model: The model to run traces through.
            tokenizer: The tokenizer for decoding outputs.
            device: The torch device to run on.
            seed: The random seed for deterministic behavior.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.seed = seed
        # Model should be in eval mode for deterministic inference
        if hasattr(model, 'eval'):
            self.model.eval()

    def run_single(
        self, example: ProbeExample, T: int
    ) -> TraceRecord:
        """Run a single example through the model at a specific T.
        
        Args:
            example: A ProbeExample with input_ids set.
            T: The number of flow steps to use.
            
        Returns:
            A TraceRecord containing the results.
        """
        set_deterministic(self.seed)

        input_ids = torch.tensor([example.input_ids], device=self.device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            output = self.model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_steps=T,
                return_dict=True,
            )

        endpoint_hidden = output.endpoint_hidden
        logits = output.logits[:, -1, :]  # last position

        endpoint_hidden_norm = endpoint_hidden.norm(p=2, dim=-1).mean().item()

        # Use tokenizer to get actual token IDs for answer labels A-J
        # Qwen tokenizer token IDs are NOT ASCII values (e.g., "A" is ~17625, not 65)
        ANSWER_TOKEN_IDS = {
            label: self.tokenizer.encode(label, add_special_tokens=False)[0]
            for label in "ABCDEFGHIJ"
        }
        logits_answer_tokens = {}
        for label, token_id in ANSWER_TOKEN_IDS.items():
            logits_answer_tokens[label] = logits[0, token_id].item()

        predicted_token_id = logits[0].argmax().item()
        predicted_token_decoded = self.tokenizer.decode([predicted_token_id])
        predicted_answer = predicted_token_decoded if predicted_token_decoded in "ABCDEFGHIJ" else "OTHER"

        return TraceRecord(
            probe_id=example.id,
            benchmark=example.benchmark,
            T=T,
            seed=self.seed,
            endpoint_hidden_norm=endpoint_hidden_norm,
            logits_answer_tokens=logits_answer_tokens,
            predicted_answer=predicted_answer,
            predicted_token_id=predicted_token_id,
            full_logits_shape=str(list(logits.shape)),
        )

    def run_probe_set(self, probe_set: ProbeSet, T_values: List[int]) -> Dict[int, List[TraceRecord]]:
        """Run all probes in a ProbeSet at multiple T values.
        
        Args:
            probe_set: The set of probes to run.
            T_values: List of T values to test each probe at.
            
        Returns:
            Dictionary mapping T values to lists of TraceRecords.
        """
        results = {}
        for T in T_values:
            records = []
            for probe in probe_set.probes:
                record = self.run_single(probe, T)
                records.append(record)
            results[T] = records
        return results

    def run_full_capture(self, probe_set: ProbeSet, T_values: List[int]) -> Dict[str, Any]:
        """Run full capture pipeline: flow traces, decoder traces, and teacher data.
        
        Args:
            probe_set: The set of probes to run.
            T_values: List of T values to test each probe at.
            
        Returns:
            Dictionary with "flow_traces" and "decoder_traces" keys,
            each mapping probe_id to list of trace dicts.
        """
        from src.diagnostic.capture import (
            capture_flow_traces,
            capture_decoder_traces,
            capture_teacher_traces,
        )
        flow_results = {}
        decoder_results = {}
        for probe in probe_set.probes:
            teacher_data = capture_teacher_traces(
                self.model, probe, self.device, self.tokenizer
            )
            flow_traces = capture_flow_traces(
                self.model, probe, T_values, self.device, self.seed
            )
            for ft in flow_traces:
                if teacher_data and teacher_data.get("teacher_anchor_distances"):
                    ft.teacher_anchor_distances = teacher_data["teacher_anchor_distances"]
            decoder_traces = capture_decoder_traces(
                self.model, self.tokenizer, probe,
                T_values, self.device, self.seed,
                teacher_data=teacher_data,
            )
            flow_results[probe.id] = [ft.to_dict() for ft in flow_traces]
            decoder_results[probe.id] = [dt.to_dict() for dt in decoder_traces]
        return {"flow_traces": flow_results, "decoder_traces": decoder_results}


def load_model_from_checkpoint(
    checkpoint_path: str,
    config_path: str,
    device: torch.device,
) -> Tuple[torch.nn.Module, Any]:
    """Load a FrozenQwenStudent model from checkpoint and config.
    
    This function handles both:
    1. Trainer checkpoint format with model_state_dict
    2. Raw midblock state dict (loaded via load_midblock)
    
    Args:
        checkpoint_path: Path to the checkpoint file (.pth)
        config_path: Path to the YAML config file
        device: torch device to load the model on
        
    Returns:
        Tuple of (model, tokenizer) where:
        - model: FrozenQwenStudent instance in eval mode on device
        - tokenizer: AutoTokenizer instance for the model
    """
    from src.model.student_qwen import FrozenQwenStudent
    
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    model_config = config.get("model", {})
    replacement_config = config.get("replacement_model", {})
    
    # Create model
    model = FrozenQwenStudent(
        model_name=model_config.get("name", "Qwen/Qwen3.5-0.8B"),
        start_layer=replacement_config.get("start_layer", 8),
        end_layer=replacement_config.get("end_layer", 11),
        max_steps_T=replacement_config.get("max_steps_T", 8),
        device=str(device),
        dtype=torch.float32,
        bypass_mode=False,
        family=model_config.get("family", "flow_midblock"),
    )
    
    # Load checkpoint (handles both formats)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        # Trainer checkpoint format
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Raw midblock state dict
        model.load_midblock(checkpoint_path)
    
    # Set to eval mode and move to device
    model.eval()
    model.to(device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.get("name", "Qwen/Qwen3.5-0.8B"),
        trust_remote_code=True
    )
    
    return model, tokenizer
