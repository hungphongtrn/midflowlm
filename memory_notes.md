# Memory Notes — Autograd Fix (2026-03-26)

## Change
Removed `torch.no_grad()` from `_continue_from_hidden_state` to restore CE/KL gradient flow through frozen Qwen upper layers.

## Impact
- Gradients now flow from loss → logits → upper frozen layers → midblock
- Upper layers (end_layer+1 to num_layers) now store activations for backprop
- Frozen parameters still have `requires_grad=False`, so no weight updates

## Measurement
- **Test config:** Qwen3.5-0.8B, start_layer=8, end_layer=11, num_steps=4, batch_size=2, seq_len=64
- **Peak GPU memory:** 4.13 GB (default float32 dtype)
- **No baseline available** — this was measured after the fix

## Notes
- Memory increase is proportional to hidden_size × seq_len × num_upper_layers
- Upper layers (24 layers after end_layer=11) now participate in gradient computation
- If memory becomes a bottleneck, consider gradient checkpointing for upper layers
