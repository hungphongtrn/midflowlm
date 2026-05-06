"""Deterministic trace runner for diagnostic probing."""
import json
import random
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path

from src.diagnostic.probe import ProbeSet, ProbeExample


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
