# Phase 1: Spike Investigation - Checkpoint Download & VRAM Profiling

## Phase Goal
Download checkpoint-3000 from HF Hub and determine the maximum viable training configuration for RTX 3060 (12GB VRAM). Output: known VRAM budget and a recommended config strategy.

## Files to Touch
- `scripts/train_sft.py` - May need minor modifications for HF Hub checkpoint download (TBD in Phase 2)
- `configs/issue-9/sft_flow_midblock.yaml` - Reference full config
- `configs/issue-9/sft_flow_midblock_3060.yaml` - Reference 3060 smoke config
- `src/model/sft_flow_midblock.py` - Understanding checkpoint loading path
- `src/training/sft_utils.py` - `estimate_training_budget` utility

## Tasks

### Task 1: Download checkpoint-3000 from HF Hub

**Files:**
- No code changes needed; investigation task

- [ ] **Step 1: Download the full checkpoint directory**

```bash
# Download checkpoint-3000 directory from HF Hub
# huggingface_hub provides snapshot_download for this
uv run python -c "
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id='hungphongtrn/midflowlm-phase2',
    allow_patterns='checkpoint-3000/*',
    local_dir='./outputs/issue-9/checkpoint-3000-continue',
)
print(f'Downloaded to: {path}')
"
```

- [ ] **Step 2: Verify checkpoint contents**

```bash
ls -lh outputs/issue-9/checkpoint-3000-continue/checkpoint-3000/
# Expected: model.safetensors (~1.6GB), midblock.pth (~88MB), optimizer.pt (~45MB),
#           scheduler.pt, rng_state.pth, trainer_state.json, training_args.bin,
#           tokenizer.json, tokenizer_config.json, chat_template.jinja
```

- [ ] **Step 3: Inspect trainer_state.json for step 3000 loss and dataset progress**

```bash
uv run python -c "
import json
with open('outputs/issue-9/checkpoint-3000-continue/checkpoint-3000/trainer_state.json') as f:
    state = json.load(f)

print('Global step:', state.get('global_step'))
print('Best step:', state.get('best_model_checkpoint'))
print('Last log history entry:')
if state.get('log_history'):
    last = state['log_history'][-1]
    for k, v in last.items():
        print(f'  {k}: {v}')
"
```

### Task 2: Profile SFTFlowMidblockModel VRAM at target batch sizes

**Files:**
- No code changes; investigation task

- [ ] **Step 1: Load model and measure baseline VRAM**

```bash
# Load the model fresh (no warm-start needed for profiling) and check VRAM
uv run python -c "
import torch
from src.model.sft_flow_midblock import SFTFlowMidblockModel
from transformers import set_seed

set_seed(1337)
device = torch.device('cuda')

for bs, seq_len in [(1, 8192), (2, 8192), (4, 8192), (4, 4096), (4, 2048)]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = SFTFlowMidblockModel(
        model_name='Qwen/Qwen3.5-0.8B',
        start_layer=8, end_layer=11,
        thinking_level=32,
        checkpoint_path=None,
        torch_dtype=torch.bfloat16,
    ).to(device)

    # Simulate a forward pass
    input_ids = torch.randint(0, 151936, (bs, seq_len), device=device)
    labels = input_ids.clone()
    labels[:, :seq_len//2] = -100  # Half masked

    try:
        with torch.no_grad():
            model.eval()
            out = model(input_ids=input_ids, labels=labels)
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f'bs={bs:<4} seq_len={seq_len:<6} peak={peak:.2f}GB  loss={out.loss.item():.4f}')
    except RuntimeError as e:
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f'bs={bs:<4} seq_len={seq_len:<6} peak={peak:.2f}GB  ERROR: {str(e)[:100]}')
    finally:
        del model
        torch.cuda.empty_cache()

print('Done')
"
```

- [ ] **Step 2: Profile with gradient checkpointing + backward pass (training mode)**

```bash
# Same but with grad + gradient_checkpointing
uv run python -c "
import torch
from src.model.sft_flow_midblock import SFTFlowMidblockModel

device = torch.device('cuda')

for bs, seq_len in [(1, 4096), (1, 8192), (2, 4096), (2, 8192), (4, 4096), (4, 8192)]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = SFTFlowMidblockModel(
        model_name='Qwen/Qwen3.5-0.8B',
        start_layer=8, end_layer=11,
        thinking_level=32,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.gradient_checkpointing_enable()
    model.train()

    input_ids = torch.randint(0, 151936, (bs, seq_len), device=device)
    labels = input_ids.clone()
    labels[:, :seq_len//2] = -100

    try:
        out = model(input_ids=input_ids, labels=labels)
        out.loss.backward()
        peak = torch.cuda.max_memory_allocated() / 1e9
        peak_reserved = torch.cuda.max_memory_reserved() / 1e9
        print(f'bs={bs:<4} seq_len={seq_len:<6} peak_alloc={peak:.2f}GB  peak_reserved={peak_reserved:.2f}GB  loss={out.loss.item():.4f}')
    except RuntimeError as e:
        print(f'bs={bs:<4} seq_len={seq_len:<6} ERROR: OOM ({str(e)[:80]})')
    finally:
        del model
        torch.cuda.empty_cache()
"
```

- [ ] **Step 3: Determine max viable eval batch size (no grad)**

```bash
# Eval mode is more memory-efficient (no grad)
# Profile bs=4 through bs=8 at various seq lengths
uv run python -c "
import torch
from src.model.sft_flow_midblock import SFTFlowMidblockModel

device = torch.device('cuda')

for bs, seq_len in [(4, 8192), (6, 8192), (8, 8192), (8, 4096)]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = SFTFlowMidblockModel(
        model_name='Qwen/Qwen3.5-0.8B',
        start_layer=8, end_layer=11,
        thinking_level=32,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    input_ids = torch.randint(0, 151936, (bs, seq_len), device=device)
    labels = input_ids.clone()
    labels[:, :seq_len//2] = -100

    try:
        with torch.no_grad():
            out = model(input_ids=input_ids, labels=labels)
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f'eval bs={bs:<4} seq_len={seq_len:<6} peak={peak:.2f}GB  loss={out.loss.item():.4f}')
    except RuntimeError as e:
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f'eval bs={bs:<4} seq_len={seq_len:<6} ERROR: {str(e)[:80]}')
    finally:
        del model
        torch.cuda.empty_cache()
"
```

### Task 3: Estimate remaining training budget

- [ ] **Step 1: Calculate steps remaining after step 3000**

From the original full config (`sft_flow_midblock.yaml`):
- Dataset: ~1M samples, seq_len=8192, T=32
- Train batch: 4, grad_accum: 2 → effective batch = 8
- Steps per epoch ≈ 1M / 8 ≈ 125K steps
- Currently at step 3000 → ~2.4% through epoch 1
- Remaining: ~122K steps at full batch size

If bs is reduced to 1 with grad_accum=8 (same effective batch):
- Remaining: still ~122K steps

If bs stays at 4 but seq_len is reduced:
- Dataset packing changes, need to re-process

### Task 4: Document findings

- [ ] **Step 1: Write findings to `phase-01-spike-investigation.md` in the plan directory**

Key outputs:
- Max train batch × seq_len that fits in 12GB
- Max eval batch × seq_len that fits
- Recommended fallback strategy
- Estimated total training time on 3060 at the chosen config

## Phase Completion Criteria
- [ ] checkpoint-3000 downloaded and contents verified
- [ ] Peak VRAM measured for train bs=4/seq_len=8192 and fallback configs
- [ ] Max viable batch × seq_len determined for both train and eval
- [ ] Remaining training budget estimated
- [ ] Recommended config strategy documented in decisions.md

## Handoff Notes
Phase 2 should take the max viable config from Phase 1, create `configs/issue-9/sft_flow_midblock_3060_resume.yaml`, and run a 5-step smoke test to verify resume works. The key deliverable is a config that doesn't OOM and shows loss continuity from step 3000.
