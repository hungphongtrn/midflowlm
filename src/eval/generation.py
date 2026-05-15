"""Shared generation utilities for eval modules."""

from __future__ import annotations

from typing import Any, Optional

import torch


def greedy_generate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    max_new_tokens: int = 64,
    num_steps: Optional[int] = None,
    solver_method: str = "euler",
    stop_on_eos: bool = True,
    eos_token_id: Optional[int] = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    **model_kwargs: Any,
) -> dict[str, Any]:
    """Autoregressive generation with optional temperature/top-p sampling."""
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    generated_token_ids: list[int] = []
    stopped_on_eos = False

    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            call_kwargs = dict(model_kwargs)
            if num_steps is not None:
                call_kwargs["num_steps"] = num_steps
            if solver_method is not None:
                call_kwargs["solver_method"] = solver_method
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, **call_kwargs)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            next_token_logits = logits[:, -1, :]

            if temperature > 0:
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        -1, sorted_indices, sorted_indices_to_remove
                    )
                    probs = probs.masked_fill(indices_to_remove, 0.0)
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            next_token_id = int(next_token.item())
            generated_token_ids.append(next_token_id)

            input_ids = torch.cat([input_ids, next_token], dim=1)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones_like(next_token, device=attention_mask.device),
                ],
                dim=1,
            )

            if stop_on_eos and eos_token_id is not None and next_token_id == eos_token_id:
                stopped_on_eos = True
                break

    return {
        "input_ids": input_ids,
        "generated_token_ids": generated_token_ids,
        "stopped_on_eos": stopped_on_eos,
    }
