# Phase 1: Model Wrapper & SFT Setup

## Phase Goal
`SFTFlowMidblockModel` loads frozen Qwen3.5-0.8B with FlowMidblock patched into layers 8-11, warm-started from P3-D3, produces HF Trainer compatible forward pass with CE loss. Verified parameter counts: ~20M trainable, ~752M frozen.

## Files to Touch

| File | Action | Responsibility |
|------|--------|----------------|
| `src/model/sft_flow_midblock.py` | Create | Wrapper model class |
| `tests/test_sft_flow_midblock.py` | Create | Tests for param counts, warm-start, forward |
| `src/model/midblock.py` | Read | Existing FlowMidblock (reused as-is) |
| `src/model/student_qwen.py` | Read | Reference for freezing patterns, h_start extraction |
| `models/p3_d3_mix_c/checkpoint.pth` | Read | P3-D3 checkpoint for warm-start |

## Tasks

### Task 1: Create `SFTFlowMidblockModel` class

**Files:**
- Create: `src/model/sft_flow_midblock.py`

**Design:**

```python
class SFTFlowMidblockModel(nn.Module):
    """Qwen3.5-0.8B with FlowMidblock patched into layers 8-11 for SFT.
    
    Architecture:
        1. Qwen Embeddings (frozen)
        2. Qwen Layers 0..7 (frozen)
        3. FlowMidblock replacing layers 8..11 (trainable, warm-start P3-D3)
        4. Qwen Layers 12..23 (frozen)
        5. Final Norm + LM Head (frozen)
    
    Compatible with HuggingFace Trainer: forward returns CE loss via Liger Kernel.
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
        # 1. Load full Qwen via AutoLigerKernelForCausalLM
        # 2. Freeze ALL Qwen parameters
        # 3. Extract hidden_size, num_heads from config
        # 4. Create FlowMidblock with max_steps_T=thinking_level
        # 5. Warm-start FlowMidblock from P3-D3 checkpoint if provided
        # 6. Monkey-patch model.forward to intercept layers 8-11
        # 7. Store start_layer, end_layer for boundary extraction
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # Patched forward will be called
        # Returns {"loss": ..., "logits": ...} (HF Trainer compatible)
```

**Monkey-patch forward implementation:**

The patched forward replaces `Qwen3Model.forward` (the base transformer forward, not the CausalLM wrapper). It must:

1. Run embeddings: `hidden_states = self.embed_tokens(input_ids)`
2. Run lower layers 0..7: iterate `self.layers[0:8]`
3. Run FlowMidblock on layer 7 output with `num_steps=self.thinking_level`
4. Run upper layers 12..N: iterate `self.layers[12:]`
5. Run final norm
6. Compute logits via LM head
7. Compute CE loss via Liger Kernel if labels provided

Key considerations:
- Position embeddings: Qwen3.5 uses RoPE internally in each attention layer, so position encoding is handled per-layer — no special handling needed when splitting layers
- Attention mask: Pass through unchanged to both lower and upper layers
- FlowMidblock expects `(h_t, attention_mask)` — we pass layer 7 output as both `h_t` and use internal `h_start` conditioning

- [ ] **Step 1: Create the file with the class skeleton**

Write `src/model/sft_flow_midblock.py`:

```python
"""
SFT-compatible model wrapper: frozen Qwen3.5-0.8B with trainable FlowMidblock.

Monkey-patches layers 8..11 with a FlowMidblock for CE-only SFT training
via HuggingFace Trainer + AutoLigerKernelForCausalLM.
"""

import logging
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from transformers import AutoConfig
from src.model.midblock import FlowMidblock

logger = logging.getLogger(__name__)


class SFTFlowMidblockModel(nn.Module):
    """Frozen Qwen with trainable FlowMidblock replacing layers start:end+1."""

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
        
        # 1. Load config
        self.config = AutoConfig.from_pretrained(model_name)
        self._resolve_layer_count()
        self._resolve_hidden_size()
        self._resolve_num_heads()
        
        # 2. Load Qwen model
        self._load_qwen()
        
        # 3. Freeze all Qwen parameters
        self._freeze_qwen()
        
        # 4. Create FlowMidblock
        self.midblock = self._create_midblock()
        
        # 5. Warm-start from checkpoint
        if checkpoint_path is not None:
            self._warm_start_midblock(checkpoint_path)
        
        # 6. Monkey-patch forward
        self._patch_forward()
    
    def _resolve_layer_count(self):
        ...
    
    def _resolve_hidden_size(self):
        ...
    
    def _resolve_num_heads(self):
        ...
    
    def _load_qwen(self):
        ...
    
    def _freeze_qwen(self):
        ...
    
    def _create_midblock(self):
        ...
    
    def _warm_start_midblock(self, checkpoint_path: str):
        ...
    
    def _patch_forward(self):
        """Override Qwen3Model.forward to route through FlowMidblock."""
        ...
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """HF Trainer compatible forward. Delegates to patched Qwen."""
        ...
```

- [ ] **Step 2: Implement `_resolve_layer_count`, `_resolve_hidden_size`, `_resolve_num_heads`**

These are straightforward from `student_qwen.py:140-197`.

- [ ] **Step 3: Implement `_load_qwen`**

```python
def _load_qwen(self):
    from liger_kernel.transformers import AutoLigerKernelForCausalLM
    
    self.qwen = AutoLigerKernelForCausalLM.from_pretrained(
        self.model_name,
        config=self.config,
        torch_dtype=self.torch_dtype,
    )
    
    # Access the base transformer model for patching
    self.base_model = self.qwen.model  # Qwen3Model
    self.layers = self.base_model.layers  # ModuleList of decoder layers
    self.embed_tokens = self.base_model.embed_tokens
    self.norm = self.base_model.norm
    self.lm_head = self.qwen.lm_head
```

- [ ] **Step 4: Implement `_freeze_qwen`**

```python
def _freeze_qwen(self):
    for param in self.qwen.parameters():
        param.requires_grad = False
    self.qwen.eval()
```

- [ ] **Step 5: Implement `_create_midblock`**

```python
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
```

- [ ] **Step 6: Implement `_warm_start_midblock`**

Load P3-D3 checkpoint, extract `midblock.*` keys (strip prefix), load into midblock.

```python
def _warm_start_midblock(self, checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_state = checkpoint["model_state_dict"]
    
    # Extract midblock keys: "midblock.*" prefix in P3-D3 checkpoint
    midblock_state = {}
    for key, value in model_state.items():
        if key.startswith("midblock."):
            new_key = key[len("midblock."):]  # Strip prefix
            midblock_state[new_key] = value
    
    # Load into FlowMidblock
    missing, unexpected = self.midblock.load_state_dict(midblock_state, strict=False)
    logger.info(f"Warm-started midblock from {checkpoint_path}")
    if missing:
        logger.warning(f"Missing keys: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected}")
```

- [ ] **Step 7: Implement `_patch_forward` — the core monkey-patch**

This is the critical piece. We store the original forward and replace it with one that routes through FlowMidblock.

```python
def _patch_forward(self):
    """Replace Qwen3Model.forward with patched version that routes through FlowMidblock."""
    midblock = self.midblock
    start_layer = self.start_layer
    end_layer = self.end_layer
    thinking_level = self.thinking_level
    embed_tokens = self.embed_tokens
    layers = self.layers
    norm = self.norm
    lm_head = self.lm_head
    
    def patched_forward(
        self_model,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        # Embeddings
        if inputs_embeds is None:
            hidden_states = embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds
        
        # Lower frozen layers: 0 to start_layer-1
        for i in range(start_layer):
            layer_output = layers[i](
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            hidden_states = layer_output[0] if isinstance(layer_output, tuple) else layer_output
        
        # Trainable FlowMidblock: replaces layers start_layer to end_layer
        # FlowMidblock.forward expects (h_start, num_steps, attention_mask)
        hidden_states = midblock(
            hidden_states,
            num_steps=thinking_level,
            attention_mask=attention_mask,
        )
        
        # Upper frozen layers: end_layer+1 to end
        for i in range(end_layer + 1, len(layers)):
            layer_output = layers[i](
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            hidden_states = layer_output[0] if isinstance(layer_output, tuple) else layer_output
        
        # Final norm
        hidden_states = norm(hidden_states)
        
        return hidden_states
    
    # Replace the base model's forward method
    self.base_model.forward = patched_forward.__get__(self.base_model, type(self.base_model))
```

Note: This patches `Qwen3Model.forward` (the internal transformer). The CausalLM wrapper (`self.qwen`) calls `self.model.forward(...)` then `self.lm_head(...)`, so patching at the Qwen3Model level is sufficient.

- [ ] **Step 8: Implement the top-level `forward`**

```python
def forward(
    self,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """HF Trainer compatible forward with CE loss."""
    # The patched base_model.forward handles the midblock routing
    # self.qwen.forward() will call self.base_model.forward() internally
    outputs = self.qwen(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    # Liger Kernel's AutoLigerKernelForCausalLM already computes fused CE loss
    # when labels are provided, returning {"loss": ..., "logits": ...}
    return outputs
```

- [ ] **Step 9: Add helper properties for parameter inspection**

```python
@property
def trainable_params(self) -> int:
    return sum(p.numel() for p in self.parameters() if p.requires_grad)

@property
def frozen_params(self) -> int:
    return sum(p.numel() for p in self.parameters() if not p.requires_grad)

@property
def device(self):
    return next(self.parameters()).device
```

### Task 2: Write failing test — parameter count verification

**File:** `tests/test_sft_flow_midblock.py`

- [ ] **Step 1: Create test file with skeleton**

```python
"""Tests for SFTFlowMidblockModel wrapper."""

import pytest
import torch
from src.model.sft_flow_midblock import SFTFlowMidblockModel


class TestSFTFlowMidblockParameterCounts:
    """Verify frozen/trainable parameter split."""
    
    def test_only_midblock_is_trainable(self):
        ...
    
    def test_midblock_param_count_matches_checkpoint(self):
        ...
    
    def test_total_params_under_one_billion(self):
        ...


class TestSFTFlowMidblockWarmStart:
    """Verify P3-D3 checkpoint loading."""
    
    def test_midblock_weights_match_checkpoint(self):
        ...


class TestSFTFlowMidblockForward:
    """Verify forward pass and HF Trainer compatibility."""
    
    def test_forward_produces_logits(self):
        ...
    
    def test_forward_with_labels_produces_loss(self):
        ...
    
    def test_gradients_only_flow_to_midblock(self):
        ...
```

- [ ] **Step 2: Write `test_only_midblock_is_trainable`**

```python
def test_only_midblock_is_trainable(self):
    model = SFTFlowMidblockModel(
        checkpoint_path="models/p3_d3_mix_c/checkpoint.pth",
    )
    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    frozen_names = [name for name, p in model.named_parameters() if not p.requires_grad]
    
    # All trainable params must contain "midblock"
    for name in trainable_names:
        assert "midblock" in name, f"Trainable param '{name}' not in midblock"
    
    # No frozen param should contain "midblock"
    for name in frozen_names:
        assert "midblock" not in name, f"Frozen param '{name}' is in midblock"
    
    assert model.trainable_params > 0
    assert model.frozen_params > 0
```

- [ ] **Step 3: Write `test_midblock_param_count_matches_checkpoint`**

```python
def test_midblock_param_count_matches_checkpoint(self):
    model = SFTFlowMidblockModel(
        checkpoint_path="models/p3_d3_mix_c/checkpoint.pth",
    )
    # From inspection: ~19.9M midblock params in P3-D3 checkpoint
    assert 19_000_000 <= model.trainable_params <= 21_000_000, \
        f"Expected ~20M trainable params, got {model.trainable_params:,}"
```

- [ ] **Step 4: Write `test_midblock_weights_match_checkpoint`**

```python
def test_midblock_weights_match_checkpoint(self):
    import torch
    checkpoint = torch.load("models/p3_d3_mix_c/checkpoint.pth", map_location="cpu", weights_only=True)
    ckpt_state = checkpoint["model_state_dict"]
    
    model = SFTFlowMidblockModel(
        checkpoint_path="models/p3_d3_mix_c/checkpoint.pth",
    )
    
    # Pick a specific key to compare
    key = "refiner.attn.q_proj.weight"
    checkpoint_weight = ckpt_state[f"midblock.{key}"]
    model_weight = model.midblock.refiner.attn.q_proj.weight.data
    
    assert torch.allclose(checkpoint_weight, model_weight, atol=1e-6), \
        f"Weight mismatch for {key}"
```

- [ ] **Step 5: Write forward pass tests**

```python
def test_forward_produces_logits(self):
    model = SFTFlowMidblockModel(
        checkpoint_path="models/p3_d3_mix_c/checkpoint.pth",
    )
    model.eval()
    
    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    assert "logits" in outputs
    assert outputs["logits"].shape == (batch_size, seq_len, model.config.vocab_size)

def test_forward_with_labels_produces_loss(self):
    model = SFTFlowMidblockModel(
        checkpoint_path="models/p3_d3_mix_c/checkpoint.pth",
    )
    model.train()
    
    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    labels = input_ids.clone()
    
    outputs = model(input_ids=input_ids, labels=labels)
    
    assert "loss" in outputs
    assert outputs["loss"].requires_grad is True
    assert outputs["loss"].item() > 0

def test_gradients_only_flow_to_midblock(self):
    model = SFTFlowMidblockModel(
        checkpoint_path="models/p3_d3_mix_c/checkpoint.pth",
    )
    model.train()
    
    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    labels = input_ids.clone()
    
    outputs = model(input_ids=input_ids, labels=labels)
    outputs["loss"].backward()
    
    # Midblock params should have gradients
    for name, p in model.named_parameters():
        if "midblock" in name:
            assert p.grad is not None, f"Midblock param '{name}' has no gradient"
        else:
            assert p.grad is None, f"Frozen param '{name}' has gradient"
```

- [ ] **Step 6: Run tests to verify they fail (model class not yet existing)**

```bash
pytest tests/test_sft_flow_midblock.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'src.model.sft_flow_midblock'`

### Task 3: Run tests to verify they pass

- [ ] **Step 1: Run all tests**

```bash
pytest tests/test_sft_flow_midblock.py -v
```
Expected: All tests PASS

### Task 4: Commit

```bash
git add src/model/sft_flow_midblock.py tests/test_sft_flow_midblock.py
git commit -m "feat: add SFTFlowMidblockModel — frozen Qwen + trainable FlowMidblock for SFT"
```

## Phase Completion Criteria
- [ ] `SFTFlowMidblockModel` loads Qwen3.5-0.8B and FlowMidblock
- [ ] Warm-start from P3-D3 checkpoint verified
- [ ] Only FlowMidblock parameters are trainable (~20M)
- [ ] Forward pass produces logits with correct shape
- [ ] Forward pass with labels produces CE loss (Liger Kernel)
- [ ] Gradients flow only to midblock, frozen params have None grad
- [ ] All tests in `tests/test_sft_flow_midblock.py` pass

## Handoff Notes
- The monkey-patched forward replaces `Qwen3Model.forward` at the base transformer level. The CausalLM wrapper (`self.qwen`) calls `self.model.forward()` then applies `lm_head`, so this is the right interception point.
- Checkpoint save/load for the full model (midblock + config) will be handled in Phase 3.
- If `AutoLigerKernelForCausalLM` is not available in the environment, Phase 1 should first add the dependency via `uv add liger-kernel --no-sync`.
- Qwen3.5's layer output format: `hidden_states, *optional` — the patched forward handles this with `isinstance(layer_output, tuple)` check.
