# MidflowLM

**Iterative Latent Matching via Flow-Based Refinement**

MidflowLM is an experimental project that replaces a span of transformer layers with a trainable iterative midblock that learns to match teacher hidden states through flow-based refinement.

## Architecture Overview

```mermaid
flowchart TB
    subgraph "Standard Qwen Architecture"
        direction TB
        Emb[Embeddings<br/>frozen]
        L0_7["Layers 0-7<br/>frozen"]
        L8_11["Layers 8-11<br/>replaced by FlowMidblock"]
        L12_23["Layers 12-23<br/>frozen"]
        LMHead[LM Head<br/>frozen]
    end
    
    subgraph "Flow-Based Refinement (T Steps)"
        direction TB
        h_start["h_start<br/>from Layer 7"]
        FM[FlowMidblock<br/>velocity predictor v_θ]
        ODE["ODE Integration<br/>Euler steps"]
        h_end["h_end<br/>to Layer 12"]
    end
    
    Input[Input Text] --> Emb
    Emb --> L0_7
    L0_7 --> h_start
    h_start --> FM
    FM --> ODE
    ODE --> h_end
    h_end --> L12_23
    L12_23 --> LMHead
    LMHead --> Output[Output Logits]
    
    style L8_11 fill:#ff6b6b,stroke:#333,stroke-width:2px
    style FM fill:#4ecdc4,stroke:#333,stroke-width:2px
    style ODE fill:#45b7d1,stroke:#333,stroke-width:2px
```

## Two-Phase Training Paradigm

```mermaid
flowchart LR
    subgraph "Phase 1: Flow Matching Distillation"
        direction TB
        P1_Data["Mixed Corpus<br/>Diverse text datasets"]
        P1_Teacher["Teacher Qwen<br/>Layers 0-23"]
        P1_Cache["Teacher Cache<br/>h_start, h_target, velocity"]
        P1_Train["Train FlowMidblock<br/>Match layer 7 → 11 transport"]
        P1_Obj["Objective:<br/>Velocity Loss (MSE)"]
        
        P1_Data --> P1_Teacher
        P1_Teacher --> P1_Cache
        P1_Cache --> P1_Train
        P1_Train --> P1_Obj
    end
    
    subgraph "Phase 2: End-to-End LLM Training"
        direction TB
        P2_Frozen["Frozen:<br/>Layers 0-7, 12-23, LM Head"]
        P2_Trainable["Trainable:<br/>FlowMidblock (replaces 8-11)"]
        P2_Data["Standard LM Data"]
        P2_Loss["Loss:<br/>CE / KL / Perplexity"]
        P2_Out["Train as Normal LLM"]
        
        P2_Frozen --> P2_Out
        P2_Trainable --> P2_Out
        P2_Data --> P2_Out
        P2_Out --> P2_Loss
    end
    
    Phase1["✓ Phase 1 Complete<br/>Flow module trained"] --> Phase2["Phase 2: Full LLM<br/>Fine-tuning"]
    
    style P1_Train fill:#4ecdc4,stroke:#333,stroke-width:2px
    style P2_Trainable fill:#4ecdc4,stroke:#333,stroke-width:2px
    style Phase1 fill:#95e1d3,stroke:#333,stroke-width:2px
```

## Detailed Architecture

```
Input Text
    ↓
[Embeddings] (frozen)
    ↓
[Lower Qwen Layers 0-7] (frozen)
    ↓ ← h_start (hidden state at layer 7 boundary)
[FlowMidblock replacing layers 8-11] (trainable)
    │   ↑
    │   └── Continuous-time velocity predictor v_θ(h_t, t)
    │   └── ODE solver: dh/dt = v_θ(h_t, t)
    │   └── T refinement steps (configurable at inference)
    ↓ → h_end (hidden state at layer 11 boundary)
[Upper Qwen Layers 12-23] (frozen)
    ↓
[LM Head] (frozen)
    ↓
Output Logits
```

### Key Components

1. **Frozen Qwen Base**: The teacher and student share the same Qwen architecture, with only layers 8-11 being replaced.

2. **FlowMidblock**: A trainable flow-based module that:
   - Takes hidden states at layer 8 boundary (actually after layer 7)
   - Iteratively refines them through T steps (configurable at inference)
   - Outputs hidden states at layer 11 boundary
   - Uses ODE solvers (Euler) for the iterative process

3. **Teacher Cache**: Pre-computed teacher hidden states and trajectory targets, enabling efficient distillation without loading the teacher during training.

### Loss Functions

- **Velocity Loss**: Matches the derivative/velocity of hidden state changes
- **Endpoint Loss**: Matches final hidden states at span exit
- **Trajectory Loss**: Matches intermediate teacher hidden states
- **KL Divergence**: Matches output token distributions (optional)

### Why Iterative?

- **Compute-Efficiency Tradeoff**: More steps = better quality, fewer steps = faster inference
- **Variable T at Inference**: Same model can run at different speed/quality points
- **Flow Matching**: Continuous-time formulation enables flexible step counts

## What is the cache build?

The cache build is the preprocessing step that runs the frozen teacher model once over the dataset and saves the teacher outputs to disk.

In this project, `scripts/build_teacher_cache.py`:
- loads the teacher model (`Qwen/Qwen3.5-0.8B` in the current v0 config)
- loads/tokenizes the dataset
- runs the teacher forward pass for each sample
- extracts the hidden states around the replacement span
- saves those outputs into `cache/...` as shard files plus metadata

More specifically, each cached sample can contain:
- `input_ids`
- `attention_mask`
- `h_start`: hidden state before the replacement span
- `trajectory_targets`: teacher hidden states inside the span
- `h_target`: hidden state after the span
- `teacher_logits`: final teacher logits

## Why do we build the cache first?

The student is trained to match teacher behavior. Instead of running the full teacher model during every training step, we precompute and save the teacher targets offline.

Benefits:
- training is simpler
- training can be faster
- GPU memory pressure during training is lower
- experiments become more reproducible

Tradeoff:
- cache generation can take a long time
- cache files can become very large

## Where does the cache go?

The cache directory is controlled by the config file.

Examples:
- `configs/v0_onemotif.yaml` -> `./cache/tinystories_qwen_boundary_states`
- `configs/v0_smoke_run.yaml` -> `./cache/tinystories_qwen_boundary_states_smoke`

## How big is the cache expected to be?

Short answer: yes, hidden-state tensors scale like `num_layers * seq_len * hidden_dim`, but in the current setup the largest tensor is actually the saved logits, which scale like `seq_len * vocab_size`.

For the current v0 setup:
- text hidden size = `1024`
- replacement span = layers `8..11` -> span depth `4`
- sequence length = `128`
- vocab size = `248320`
- cache build currently uses float32 for teacher outputs, so each float is `4 bytes`

Per sample, the main cached tensors are approximately:
- `h_start`: `[seq_len, hidden_dim]` -> `128 * 1024 * 4` bytes -> about `0.5 MiB`
- `trajectory_targets`: `span_depth * [seq_len, hidden_dim]` -> `4 * 128 * 1024 * 4` bytes -> about `2.0 MiB`
- `h_target`: `[seq_len, hidden_dim]` -> about `0.5 MiB`
- `teacher_logits`: `[seq_len, vocab_size]` -> `128 * 248320 * 4` bytes -> about `121 MiB`

So the rough total per sample is dominated by logits:
- hidden states only: about `3 MiB / sample`
- logits only: about `121 MiB / sample`
- total: about `124 MiB / sample`

That means for `20,000` samples the raw cache can become extremely large:
- hidden states only: about `60 GiB`
- logits only: about `2.3 TiB`
- total rough upper bound: about `2.4 TiB`

This is why the cache build can become unexpectedly huge. If cache size must be reduced, the first thing to reconsider is whether `teacher_logits` need to be stored for every token of every sample.

## Step-by-step workflow

1. Build teacher cache
2. Train the student using the cached teacher outputs
3. Evaluate the trained student

## Commands

Smoke run:

```bash
cd /home/hungphongtrn/Workspace/midflowlm
source .venv/bin/activate
pip install -r requirements.txt
python scripts/build_teacher_cache.py --config configs/v0_smoke_run.yaml --limit 8 --overwrite
python scripts/train_v0.py --config configs/v0_smoke_run.yaml --fast-dev-run
```

Full v0 run:

```bash
cd /home/hungphongtrn/Workspace/midflowlm
source .venv/bin/activate
pip install -r requirements.txt
python scripts/build_teacher_cache.py --config configs/v0_onemotif.yaml --overwrite
python scripts/train_v0.py --config configs/v0_onemotif.yaml
python scripts/eval_v0.py --config configs/v0_onemotif.yaml
```

## Important notes

- Cache build is required before training.
- Full cache generation may be very large and slow.
- If training fails while loading cached shards, inspect the cache loader and shard format compatibility first.
