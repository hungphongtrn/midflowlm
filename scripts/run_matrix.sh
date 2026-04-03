#!/bin/bash
# run_matrix.sh - Simple runner for v0.1 experiment matrix
# Usage: 
#   bash scripts/run_matrix.sh              # Run all experiments sequentially
#   bash scripts/run_matrix.sh --parallel   # Run on all available GPUs in parallel
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_matrix.sh  # Run on GPU 0 only

set -e

CONFIG_DIR="configs/v0_1_matrix"
LOG_DIR="logs"
STATUS_FILE=".experiment_status"
PARALLEL=0
MAX_JOBS=3  # Default for 3x3090 setup

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel)
            PARALLEL=1
            shift
            ;;
        --max-jobs)
            MAX_JOBS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --help)
            echo "Usage: bash scripts/run_matrix.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --parallel      Run experiments across all GPUs in parallel"
            echo "  --max-jobs N    Limit parallel jobs to N (default: 3)"
            echo "  --dry-run       Show what would be run without executing"
            echo "  --help          Show this help"
            echo ""
            echo "Examples:"
            echo "  bash scripts/run_matrix.sh                    # Sequential"
            echo "  bash scripts/run_matrix.sh --parallel         # Parallel on all GPUs"
            echo "  CUDA_VISIBLE_DEVICES=0 bash scripts/run_matrix.sh  # GPU 0 only"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Setup
mkdir -p "$LOG_DIR"
touch "$STATUS_FILE"

# Get list of configs (excluding README)
CONFIGS=($(ls $CONFIG_DIR/*.yaml | grep -v README | sort))
TOTAL=${#CONFIGS[@]}

echo "=================================="
echo "MidflowLM v0.1 Experiment Runner"
echo "=================================="
echo "Found $TOTAL experiments"
echo "Mode: $([ $PARALLEL -eq 1 ] && echo 'Parallel (max $MAX_JOBS jobs)' || echo 'Sequential')"
echo "Logs: $LOG_DIR/"
echo "Status: $STATUS_FILE"
echo ""

# Check if already running
if [ -f ".runner_lock" ]; then
    PID=$(cat .runner_lock)
    if ps -p $PID > /dev/null 2>&1; then
        echo "WARNING: Runner already active (PID: $PID)"
        echo "Use: kill $PID  # to stop"
        echo "Or: rm .runner_lock  # if stale"
        echo ""
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        rm .runner_lock
    fi
fi

# Create lock file
echo $$ > .runner_lock

# Cleanup on exit
trap "rm -f .runner_lock; echo ''; echo 'Runner stopped.'; exit" INT TERM EXIT

# Function to check if experiment is already completed
is_completed() {
    local config="$1"
    local name=$(basename "$config" .yaml)
    grep -q "^COMPLETED:$name$" "$STATUS_FILE" 2>/dev/null
}

# Function to check if experiment is currently running
is_running() {
    local config="$1"
    local name=$(basename "$config" .yaml)
    grep -q "^RUNNING:$name$" "$STATUS_FILE" 2>/dev/null
}

# Function to mark experiment status
mark_status() {
    local status="$1"
    local config="$2"
    local name=$(basename "$config" .yaml)
    
    # Remove old status
    sed -i "/^.*:$name$/d" "$STATUS_FILE" 2>/dev/null || true
    
    # Add new status with timestamp
    echo "$status:$name:$(date +%Y-%m-%d_%H:%M:%S)" >> "$STATUS_FILE"
}

# Function to run a single experiment
run_experiment() {
    local config="$1"
    local gpu_id="${2:-0}"
    local name=$(basename "$config" .yaml)
    local logfile="$LOG_DIR/${name}.log"
    
    if is_completed "$config"; then
        echo "  [SKIP] $name (already completed)"
        return 0
    fi
    
    if is_running "$config"; then
        echo "  [SKIP] $name (already running)"
        return 0
    fi
    
    mark_status "RUNNING" "$config"
    
    echo "  [START] $name (GPU $gpu_id)"
    echo "  Log: $logfile"
    
    if [ -n "$DRY_RUN" ]; then
        echo "  [DRY-RUN] Would run: CUDA_VISIBLE_DEVICES=$gpu_id python3 scripts/train.py --config $config"
        return 0
    fi
    
    # Run the experiment
    export CUDA_VISIBLE_DEVICES=$gpu_id
    if python3 scripts/train.py --config "$config" > "$logfile" 2>&1; then
        mark_status "COMPLETED" "$config"
        echo "  [DONE] $name"
        return 0
    else
        mark_status "FAILED" "$config"
        echo "  [FAIL] $name (see $logfile)"
        return 1
    fi
}

# Count pending experiments
PENDING=0
for config in "${CONFIGS[@]}"; do
    if ! is_completed "$config"; then
        ((PENDING++))
    fi
done

echo "Pending: $PENDING / $TOTAL experiments"
echo ""

if [ $PENDING -eq 0 ]; then
    echo "All experiments completed!"
    exit 0
fi

# Show summary
echo "Experiments:"
for config in "${CONFIGS[@]}"; do
    name=$(basename "$config" .yaml)
    if is_completed "$config"; then
        echo "  ✓ $name"
    elif is_running "$config"; then
        echo "  ⟳ $name (running)"
    else
        echo "  ○ $name"
    fi
done
echo ""

# Run experiments
COMPLETED=0
FAILED=0

if [ $PARALLEL -eq 1 ]; then
    # Parallel mode - distribute across GPUs
    echo "Running in parallel mode (max $MAX_JOBS concurrent jobs)..."
    echo ""
    
    # Get number of available GPUs
    NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "1")
    echo "Detected $NUM_GPUS GPU(s)"
    echo ""
    
    # Background job tracking
    declare -a PIDS
    declare -a GPU_ASSIGNMENTS
    
    for config in "${CONFIGS[@]}"; do
        # Skip if already done or running
        if is_completed "$config" || is_running "$config"; then
            continue
        fi
        
        # Find an available GPU slot
        while true; do
            # Check for completed jobs
            for i in "${!PIDS[@]}"; do
                if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
                    # Job finished
                    unset PIDS[$i]
                    unset GPU_ASSIGNMENTS[$i]
                    PIDS=("${PIDS[@]}")
                    GPU_ASSIGNMENTS=("${GPU_ASSIGNMENTS[@]}")
                fi
            done
            
            # Check if we have room for another job
            if [ ${#PIDS[@]} -lt $MAX_JOBS ]; then
                break
            fi
            
            # Wait a bit before checking again
            sleep 5
        done
        
        # Find next available GPU
        GPU_ID=$((COMPLETED % NUM_GPUS))
        
        # Run in background
        run_experiment "$config" "$GPU_ID" &
        PIDS+=($!)
        GPU_ASSIGNMENTS+=($GPU_ID)
        ((COMPLETED++))
        
        echo "Launched job $COMPLETED (PID: $!) on GPU $GPU_ID"
    done
    
    # Wait for all remaining jobs
    echo ""
    echo "Waiting for all jobs to complete..."
    wait
    echo "All jobs finished!"
    
else
    # Sequential mode
    echo "Running in sequential mode..."
    echo ""
    
    for config in "${CONFIGS[@]}"; do
        if is_completed "$config"; then
            ((COMPLETED++))
            continue
        fi
        
        run_experiment "$config" 0
        if [ $? -eq 0 ]; then
            ((COMPLETED++))
        else
            ((FAILED++))
            if [ $FAILED -gt 3 ]; then
                echo ""
                echo "Too many failures ($FAILED), stopping."
                break
            fi
        fi
        echo ""
    done
fi

# Final summary
echo ""
echo "=================================="
echo "Run Complete"
echo "=================================="
echo ""

# Count final status
COMPLETED_COUNT=$(grep -c "^COMPLETED:" "$STATUS_FILE" 2>/dev/null || echo "0")
FAILED_COUNT=$(grep -c "^FAILED:" "$STATUS_FILE" 2>/dev/null || echo "0")
RUNNING_COUNT=$(grep -c "^RUNNING:" "$STATUS_FILE" 2>/dev/null || echo "0")

echo "Summary:"
echo "  Completed: $COMPLETED_COUNT"
echo "  Failed: $FAILED_COUNT"
echo "  Running: $RUNNING_COUNT"
echo ""
echo "View results: https://wandb.ai"
echo ""

# Clean up lock
rm -f .runner_lock
