# Phase 4: Queue Execution

**Goal:** Implement and test the queue runner that keeps 3×3090 GPUs busy executing the config matrix continuously.

**Deliverables:**
- [ ] Queue runner script that coordinates across 3 GPUs
- [ ] Durable run ledger for tracking status
- [ ] Automatic post-training evaluation
- [ ] Checkpoint resume on interruption
- [ ] Failure isolation (one failed run doesn't block others)

**Depends on:** Phase 3 (config matrix generation)

---

## Queue Design

Simple Python queue with one worker process per GPU:

```
GPU 0 Worker → picks next pending config → trains → runs eval → writes done
GPU 1 Worker → picks next pending config → trains → runs eval → writes done
GPU 2 Worker → picks next pending config → trains → runs eval → writes done
```

Each worker:
1. Scans `configs/v0_1/` for configs without completion markers
2. Claims one via atomic rename or file lock
3. Sets `CUDA_VISIBLE_DEVICES` to its assigned GPU
4. Runs `scripts/train.py --config <config>`
5. On success, runs eval: text sweep, MMLU-Pro
6. Writes `run_ledger.json` entry with status
7. Moves to next config

## Run Ledger

JSON lines file tracking all attempts:

```json
{"run_id": "P1-A1", "config": "midflow_qwen_8to11_proj_mixB_endkl_trainT-r2468.yaml", "status": "completed", "checkpoint": "...", "eval_results": "...", "gpu": 0, "timestamp": "2026-04-10T12:00:00Z"}
{"run_id": "P1-A2", "config": "...", "status": "failed", "error": "OOM", "gpu": 1, "timestamp": "..."}
```

## Phase Completion Gate

- [ ] Queue runner launches and claims configs
- [ ] GPU 0, 1, 2 each run independent configs
- [ ] Interrupted run resumes from checkpoint
- [ ] Failed run is marked failed and queue continues
- [ ] Ledger is readable and accurate

**Next:** Write Phase 5 plan (launch and monitoring)
