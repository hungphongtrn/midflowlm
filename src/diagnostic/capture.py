import torch
import numpy as np
import math
from typing import List, Dict, Optional
from pathlib import Path

from src.diagnostic.probe import ProbeExample
from src.diagnostic.traces import FlowTrace, DecoderTrace
from src.diagnostic.runner import set_deterministic


def _kl_divergence(p_logits: Dict[str, float], q_logits: Dict[str, float]) -> float:
    """Compute KL(p||q) from logit dicts. Returns >= 0."""
    labels = sorted(set(p_logits.keys()) & set(q_logits.keys()))
    if not labels:
        return 0.0
    
    # Convert logits to probabilities via softmax
    p_vals = np.array([p_logits[l] for l in labels])
    q_vals = np.array([q_logits[l] for l in labels])
    
    # Numerically stable softmax: subtract max
    p_max = np.max(p_vals)
    q_max = np.max(q_vals)
    
    p_exp = np.exp(p_vals - p_max)
    p_probs = p_exp / p_exp.sum()
    
    q_exp = np.exp(q_vals - q_max)
    q_probs = q_exp / q_exp.sum()
    
    # KL(p||q) = sum(p * log(p/q)) = sum(p * (log p - log q))
    epsilon = 1e-12
    p_probs_clamped = np.clip(p_probs, epsilon, 1.0)
    q_probs_clamped = np.clip(q_probs, epsilon, 1.0)
    
    kl = np.sum(p_probs * (np.log(p_probs_clamped) - np.log(q_probs_clamped)))
    return float(max(kl, 0.0))


def _js_divergence(p_logits: Dict[str, float], q_logits: Dict[str, float]) -> float:
    """Compute Jensen-Shannon divergence from logit dicts. Returns >= 0.
    
    JS(P||Q) = (KL(P||M) + KL(Q||M)) / 2, where M = (P+Q)/2
    """
    labels = sorted(set(p_logits.keys()) & set(q_logits.keys()))
    if not labels:
        return 0.0
    
    # Convert logits to probabilities via softmax
    p_vals = np.array([p_logits[l] for l in labels])
    q_vals = np.array([q_logits[l] for l in labels])
    
    # Numerically stable softmax
    p_max = np.max(p_vals)
    q_max = np.max(q_vals)
    
    p_exp = np.exp(p_vals - p_max)
    p_probs = p_exp / p_exp.sum()
    
    q_exp = np.exp(q_vals - q_max)
    q_probs = q_exp / q_exp.sum()
    
    # M = (P+Q)/2
    m_probs = 0.5 * (p_probs + q_probs)
    
    # KL(P||M) and KL(Q||M)
    epsilon = 1e-12
    p_probs_clamped = np.clip(p_probs, epsilon, 1.0)
    q_probs_clamped = np.clip(q_probs, epsilon, 1.0)
    m_probs_clamped = np.clip(m_probs, epsilon, 1.0)
    
    kl_pm = np.sum(p_probs * (np.log(p_probs_clamped) - np.log(m_probs_clamped)))
    kl_qm = np.sum(q_probs * (np.log(q_probs_clamped) - np.log(m_probs_clamped)))
    
    js = 0.5 * kl_pm + 0.5 * kl_qm
    return float(max(js, 0.0))


def capture_decoder_traces(
    model,
    tokenizer,
    example: ProbeExample,
    T_values: List[int],
    device: torch.device,
    seed: int = 42,
    teacher_data: Optional[Dict] = None,
) -> List[DecoderTrace]:
    """Capture decoder/readout traces across multiple T values.
    
    For each T, runs model forward pass, extracts logits at last position,
    builds answer token logits dict, decodes predicted answer, and computes
    KL/JS divergence from teacher if provided.
    
    Args:
        model: Model with forward() returning logits
        tokenizer: Tokenizer for encoding/decoding answer labels
        example: ProbeExample with input_ids and target_label
        T_values: List of T values to evaluate
        device: torch device
        seed: Random seed for determinism
        teacher_data: Optional dict with teacher_logits_answer_tokens from capture_teacher_traces
        
    Returns:
        List of DecoderTrace, one per T value
    """
    set_deterministic(seed)
    
    input_ids = torch.tensor([example.input_ids], device=device)
    attention_mask = torch.ones_like(input_ids)
    
    # Build answer token IDs using tokenizer
    answer_labels = "ABCDEFGHIJ"
    ANSWER_TOKEN_IDS = {
        label: tokenizer.encode(label, add_special_tokens=False)[0]
        for label in answer_labels
    }
    
    traces = []
    for T in T_values:
        set_deterministic(seed)
        
        with torch.no_grad():
            output = model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_steps=T,
                return_dict=True,
            )
        
        # Extract logits at last position
        logits = output.logits[:, -1, :]  # [batch, vocab]
        
        # Build logits dict for answer tokens
        logits_answer_tokens = {}
        for label, token_id in ANSWER_TOKEN_IDS.items():
            logits_answer_tokens[label] = logits[0, token_id].item()
        
        # Decode predicted answer token (argmax over full vocab)
        predicted_token_id = logits[0].argmax().item()
        predicted_answer = tokenizer.decode([predicted_token_id])
        
        # Validate predicted answer is A-J
        if predicted_answer not in answer_labels:
            predicted_answer = "OTHER"
        
        # Check if predicted answer matches ground truth
        parsed_answer_match = predicted_answer == example.target_label
        
        # Get teacher logits and compute divergences if teacher_data provided
        teacher_logits_answer = {}
        kl = 0.0
        js = 0.0
        if teacher_data is not None and teacher_data.get("teacher_logits_answer_tokens"):
            teacher_logits_answer = teacher_data["teacher_logits_answer_tokens"]
            # Only compute if we have matching keys
            if set(logits_answer_tokens.keys()) & set(teacher_logits_answer.keys()):
                kl = _kl_divergence(logits_answer_tokens, teacher_logits_answer)
                js = _js_divergence(logits_answer_tokens, teacher_logits_answer)
        
        traces.append(DecoderTrace(
            probe_id=example.id,
            benchmark=example.benchmark,
            T=T,
            logits_answer_tokens=logits_answer_tokens,
            predicted_answer=predicted_answer,
            predicted_token_id=predicted_token_id,
            ground_truth_label=example.target_label,
            parsed_answer_match=parsed_answer_match,
            teacher_logits_answer_tokens=teacher_logits_answer,
            kl_divergence=kl,
            js_divergence=js,
        ))
    
    return traces


def capture_flow_traces(
    model,
    example: ProbeExample,
    T_values: List[int],
    device: torch.device,
    seed: int = 42,
) -> List[FlowTrace]:
    set_deterministic(seed)

    input_ids = torch.tensor([example.input_ids], device=device)
    attention_mask = torch.ones_like(input_ids)

    traces = []
    ref_endpoint = None
    ref_trajectory = None

    for T in T_values:
        set_deterministic(seed)

        with torch.no_grad():
            output = model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_steps=T,
                return_dict=True,
            )

        endpoint_hidden = output.endpoint_hidden
        endpoint_hidden_norm = endpoint_hidden.norm(p=2, dim=-1).mean().item()

        per_step_velocity_norms = []
        trajectory_endpoint_norm = endpoint_hidden_norm

        if output.trajectory_hidden is not None:
            traj = output.trajectory_hidden  # [batch, seq, num_steps, hidden]
            num_steps_captured = traj.shape[2]
            dt = 1.0 / T
            for step in range(num_steps_captured):
                if step == 0:
                    # First step: zero velocity (starting point)
                    per_step_velocity_norms.append(0.0)
                else:
                    delta = traj[:, :, step, :] - traj[:, :, step - 1, :]
                    vel_norm = (delta.norm(p=2, dim=-1).mean().item()) / dt
                    per_step_velocity_norms.append(vel_norm)

            trajectory_endpoint = traj[:, :, -1, :]  # last step in trajectory
            trajectory_endpoint_norm = trajectory_endpoint.norm(p=2, dim=-1).mean().item()

        divergence_from_T1 = 0.0
        if T == 1:
            ref_endpoint = endpoint_hidden
            ref_trajectory = output.trajectory_hidden
        elif ref_endpoint is not None:
            divergence_from_T1 = (endpoint_hidden - ref_endpoint).norm(p=2, dim=-1).mean().item()

        traces.append(FlowTrace(
            probe_id=example.id,
            benchmark=example.benchmark,
            T=T,
            endpoint_hidden_norm=endpoint_hidden_norm,
            per_step_velocity_norms=per_step_velocity_norms,
            trajectory_endpoint_norm=trajectory_endpoint_norm,
            trajectory_divergence_from_T1=divergence_from_T1,
            teacher_anchor_distances={},
        ))

    return traces


def capture_teacher_traces(
    model,
    example: ProbeExample,
    device: torch.device,
    tokenizer=None,
) -> Optional[Dict]:
    input_ids = torch.tensor([example.input_ids], device=device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        # Try with keyword args first, fall back to positional for compatibility
        try:
            teacher = model.extract_teacher_targets(
                input_ids=input_ids,
                attention_mask=attention_mask,
                need_teacher_logits=True,
                need_trajectory_anchors=True,
            )
        except TypeError:
            # Fallback for mocks that don't accept these kwargs
            teacher = model.extract_teacher_targets(
                input_ids, attention_mask, need_trajectory_anchors=True
            )

    answer_labels = "ABCDEFGHIJ"
    teacher_anchor_distances = {}
    if teacher.get("trajectory_anchors"):
        for key, anchor in teacher["trajectory_anchors"].items():
            teacher_anchor_distances[key] = anchor.norm(p=2, dim=-1).mean().item()

    teacher_logits_answer = {}
    if teacher.get("teacher_logits") is not None and tokenizer is not None:
        logits = teacher["teacher_logits"][:, -1, :]
        ANSWER_TOKEN_IDS = {
            label: tokenizer.encode(label, add_special_tokens=False)[0]
            for label in answer_labels
        }
        for label, tid in ANSWER_TOKEN_IDS.items():
            teacher_logits_answer[label] = logits[0, tid].item()

    return {
        "teacher_anchor_distances": teacher_anchor_distances,
        "teacher_logits_answer_tokens": teacher_logits_answer,
    }
