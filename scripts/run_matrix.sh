#!/bin/bash
# run_matrix.sh - Enhanced runner for v0.1 experiment matrix on 3x3090s
# Usage: 
#   bash scripts/run_matrix.sh              # Run all experiments sequentially
#   bash scripts/run_matrix.sh --parallel   # Run on all 3 GPUs in parallel (default for 3090s)
#   bash scripts/run_matrix.sh --resume     # Resume interrupted experiments
#   CUDA_VISIBLE_DEVICES=0,1,2 bash scripts/run_matrix.sh --parallel  # Specific GPUs

# Configuration
CONFIG_DIR="configs/v0_1_matrix"
LOG_DIR="logs/matrix_$(date +%Y%m%d_%H%M%S)"
STATUS_DIR=".experiment_status"
STATUS_FILE="$STATUS_DIR/status_$(date +%Y%m%d).log"
PARALLEL=1  # Default to parallel for 3090s
MAX_JOBS=3
RESUME=0
DRY_RUN=0
START_TIME=$(date +%s)
RUNNER_PID=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel)
            PARALLEL=1
            shift
            ;;
        --sequential)
            PARALLEL=0
            shift
            ;;
        --max-jobs)
            MAX_JOBS="$2"
            shift 2
            ;;
        --resume)
            RESUME=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --config-dir)
            CONFIG_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: bash scripts/run_matrix.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --parallel        Run experiments across all GPUs in parallel (default)"
            echo "  --sequential      Run experiments one at a time"
            echo "  --max-jobs N      Limit parallel jobs to N (default: 3 for 3x3090s)"
            echo "  --resume          Resume from previous run (skip completed)"
            echo "  --dry-run         Show what would be run without executing"
            echo "  --config-dir DIR  Use different config directory"
            echo "  --help            Show this help"
            echo ""
            echo "Examples:"
            echo "  bash scripts/run_matrix.sh                    # Parallel on all GPUs"
            echo "  bash scripts/run_matrix.sh --sequential       # One at a time"
            echo "  bash scripts/run_matrix.sh --resume           # Continue after crash"
            echo "  bash scripts/run_matrix.sh --dry-run          # Preview only"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Setup directories
mkdir -p "$LOG_DIR" 2>/dev/null || true
mkdir -p "$STATUS_DIR" 2>/dev/null || true
touch "$STATUS_FILE" 2>/dev/null || true

# Get list of configs
CONFIGS=($(ls "$CONFIG_DIR"/*.yaml 2>/dev/null | grep -v README | sort))
TOTAL=${#CONFIGS[@]}

if [ $TOTAL -eq 0 ]; then
    echo -e "${RED}ERROR: No configs found in $CONFIG_DIR${NC}"
    exit 1
fi

# Header
echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║     MidflowLM v0.1 Experiment Matrix Runner (3090s)        ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Configuration:${NC}"
echo "  Total experiments: $TOTAL"
echo "  Config directory: $CONFIG_DIR"
echo "  Log directory: $LOG_DIR"
echo "  Status file: $STATUS_FILE"
echo -n "  Mode: "
if [ $PARALLEL -eq 1 ]; then
    echo -e "${GREEN}Parallel${NC} (max $MAX_JOBS jobs)"
else
    echo -e "${YELLOW}Sequential${NC}"
fi
echo ""

# Detect GPUs
NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
if [ "$NUM_GPUS" = "0" ]; then
    echo -e "${RED}WARNING: No GPUs detected!${NC}"
    echo "  Training will be extremely slow on CPU."
    echo "  Set CUDA_VISIBLE_DEVICES if GPUs are available."
    echo ""
    NUM_GPUS=1  # Fallback
fi
echo -e "${GREEN}GPUs detected: $NUM_GPUS${NC}"
echo ""

# Check if already running
if [ -f ".runner_lock" ]; then
    PID=$(cat .runner_lock 2>/dev/null || echo "")
    if [ -n "$PID" ] && ps -p "$PID" > /dev/null 2>&1; then
        echo -e "${YELLOW}WARNING: Runner already active (PID: $PID)${NC}"
        echo "  Another instance is currently running."
        echo "  Use: kill $PID  # to stop existing runner"
        echo "  Or:  rm .runner_lock  # if process crashed"
        echo ""
        if [ $RESUME -eq 0 ]; then
            read -p "Continue anyway? (y/N) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        fi
    else
        rm -f .runner_lock 2>/dev/null || true
    fi
fi

# Create lock file
echo $$ > .runner_lock
RUNNER_PID=$$

# Cleanup function - only shows summary, doesn't exit
cleanup() {
    if [ -f ".runner_lock" ]; then
        local lock_pid=$(cat .runner_lock 2>/dev/null || echo "")
        if [ "$lock_pid" = "$$" ]; then
            rm -f .runner_lock
        fi
    fi
}

trap cleanup INT TERM

# Status tracking functions
is_completed() {
    local config="$1"
    local name=$(basename "$config" .yaml)
    grep -q "^COMPLETED:$name:" "$STATUS_FILE" 2>/dev/null
}

is_running() {
    local config="$1"
    local name=$(basename "$config" .yaml)
    grep -q "^RUNNING:$name:" "$STATUS_FILE" 2>/dev/null
}

is_failed() {
    local config="$1"
    local name=$(basename "$config" .yaml)
    grep -q "^FAILED:$name:" "$STATUS_FILE" 2>/dev/null
}

mark_status() {
    local status="$1"
    local config="$2"
    local name=$(basename "$config" .yaml)
    local duration="${3:-0}"
    
    # Remove old status
    if [ -f "$STATUS_FILE" ]; then
        sed -i "/^.*:$name:/d" "$STATUS_FILE" 2>/dev/null || true
    fi
    
    # Add new status with timestamp and duration
    echo "$status:$name:$(date +%Y-%m-%d_%H:%M:%S):${duration}s" >> "$STATUS_FILE"
}

# Count experiments by status
count_status() {
    local pattern="$1"
    if [ -f "$STATUS_FILE" ]; then
        grep "^$pattern:" "$STATUS_FILE" 2>/dev/null | wc -l
    else
        echo "0"
    fi
}

# Format time duration
format_duration() {
    local seconds=$1
    local hours=$((seconds / 3600))
    local minutes=$(((seconds % 3600) / 60))
    local secs=$((seconds % 60))
    printf "%02d:%02d:%02d" $hours $minutes $secs
}

# Show summary table
show_summary_table() {
    local completed=$1
    local failed=$2
    local running=$3
    local pending=$4
    local elapsed=$5
    
    echo ""
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║              Experiment Progress Summary                   ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    echo -e "${BLUE}Status:${NC}"
    printf "  ${GREEN}✓ Completed: %d/%d${NC}\n" $completed $TOTAL
    printf "  ${YELLOW}⟳ Running:   %d${NC}\n" $running
    printf "  ${RED}✗ Failed:    %d${NC}\n" $failed
    printf "  ${NC}○ Pending:   %d${NC}\n" $pending
    echo ""
    
    echo -e "${BLUE}Time:${NC}"
    echo "  Elapsed: $(format_duration $elapsed)"
    
    if [ $completed -gt 0 ]; then
        local avg_time=$((elapsed / completed))
        local remaining=$((TOTAL - completed))
        local eta=$((avg_time * remaining))
        echo "  Avg per exp: $(format_duration $avg_time)"
        echo "  Est. ETA: $(format_duration $eta)"
    fi
    echo ""
    
    # Show GPU utilization if available
    if command -v nvidia-smi &> /dev/null && [ $NUM_GPUS -gt 0 ]; then
        echo -e "${BLUE}GPU Status:${NC}"
        nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -3 | while IFS=, read -r idx name temp util mem_used mem_total; do
            # Trim whitespace
            idx=$(echo "$idx" | xargs)
            name=$(echo "$name" | xargs)
            temp=$(echo "$temp" | xargs)
            util=$(echo "$util" | xargs)
            mem_used=$(echo "$mem_used" | xargs)
            mem_total=$(echo "$mem_total" | xargs)
            printf "  GPU%s: %s | %3s%% util | %2s°C | %5s/%5s MB\n" "$idx" "$name" "$util" "$temp" "$mem_used" "$mem_total"
        done
        echo ""
    fi
    
    # Recent activity (last 5)
    if [ -f "$STATUS_FILE" ]; then
        local total_lines=$(wc -l < "$STATUS_FILE" 2>/dev/null || echo "0")
        if [ $total_lines -gt 0 ]; then
            echo -e "${BLUE}Recent Activity (last 5):${NC}"
            tail -5 "$STATUS_FILE" | while read -r line; do
                local status=$(echo "$line" | cut -d: -f1)
                local name=$(echo "$line" | cut -d: -f2)
                local duration=$(echo "$line" | cut -d: -f5)
                
                case $status in
                    COMPLETED)
                        echo -e "  ${GREEN}✓${NC} $name ${duration}"
                        ;;
                    RUNNING)
                        echo -e "  ${YELLOW}⟳${NC} $name (running...)"
                        ;;
                    FAILED)
                        echo -e "  ${RED}✗${NC} $name ${duration}"
                        ;;
                esac
            done
            echo ""
        fi
    fi
    
    echo -e "${BLUE}View live results:${NC} https://wandb.ai"
    echo ""
}

# Run a single experiment
run_experiment() {
    local config="$1"
    local gpu_id="${2:-0}"
    local name=$(basename "$config" .yaml)
    local logfile="$LOG_DIR/${name}.log"
    local start_ts=$(date +%s)
    
    # Check if already done
    if is_completed "$config"; then
        echo -e "  ${GREEN}[SKIP]${NC} $name (already completed)"
        return 0
    fi
    
    if is_running "$config"; then
        echo -e "  ${YELLOW}[SKIP]${NC} $name (already running in another process)"
        return 0
    fi
    
    mark_status "RUNNING" "$config"
    
    echo -e "  ${CYAN}[START]${NC} $name (GPU $gpu_id)"
    echo "          Log: $logfile"
    
    if [ $DRY_RUN -eq 1 ]; then
        echo -e "          ${YELLOW}[DRY-RUN]${NC} Would run: CUDA_VISIBLE_DEVICES=$gpu_id python3 scripts/train.py --config $config"
        sleep 0.5
        return 0
    fi
    
    # Run the experiment with proper error handling
    export CUDA_VISIBLE_DEVICES=$gpu_id
    export WANDB_SILENT=true  # Reduce wandb noise in logs
    
    local cmd="python3 scripts/train.py --config $config"
    
    if $cmd > "$logfile" 2>&1; then
        local end_ts=$(date +%s)
        local duration=$((end_ts - start_ts))
        mark_status "COMPLETED" "$config" $duration
        echo -e "  ${GREEN}[DONE]${NC} $name ($(format_duration $duration))"
        return 0
    else
        local end_ts=$(date +%s)
        local duration=$((end_ts - start_ts))
        mark_status "FAILED" "$config" $duration
        echo -e "  ${RED}[FAIL]${NC} $name ($(format_duration $duration))"
        echo -e "          ${YELLOW}Check:${NC} tail -50 $logfile"
        return 1
    fi
}

# Show final summary
show_final_summary() {
    local end_time=$(date +%s)
    local total_time=$((end_time - START_TIME))
    
    local completed=$(count_status "COMPLETED")
    local failed=$(count_status "FAILED")
    local running=$(count_status "RUNNING")
    
    echo ""
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                    Experiment Summary                      ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BLUE}Final Status:${NC}"
    echo -e "  ${GREEN}✓ Completed: $completed${NC}"
    echo -e "  ${RED}✗ Failed: $failed${NC}"
    echo -e "  ${YELLOW}⟳ Still running: $running${NC}"
    echo ""
    echo -e "${BLUE}Total time: $(format_duration $total_time)${NC}"
    echo ""
    
    if [ $completed -gt 0 ]; then
        echo -e "${BLUE}Completed experiments:${NC}"
        grep "^COMPLETED:" "$STATUS_FILE" 2>/dev/null | while read -r line; do
            local name=$(echo "$line" | cut -d: -f2)
            local duration=$(echo "$line" | cut -d: -f5)
            echo -e "  ${GREEN}✓${NC} $name ${duration}"
        done
        echo ""
    fi
    
    if [ $failed -gt 0 ]; then
        echo -e "${BLUE}Failed experiments:${NC}"
        grep "^FAILED:" "$STATUS_FILE" 2>/dev/null | while read -r line; do
            local name=$(echo "$line" | cut -d: -f2)
            local duration=$(echo "$line" | cut -d: -f5)
            echo -e "  ${RED}✗${NC} $name ${duration}"
        done
        echo ""
        echo -e "${YELLOW}To retry failed experiments:${NC}"
        echo "  bash scripts/run_matrix.sh --resume"
        echo ""
    fi
    
    echo -e "${BLUE}View all results:${NC}"
    echo "  wandb: https://wandb.ai"
    echo "  logs:  ls -la $LOG_DIR/"
    echo ""
}

# Main execution
main() {
    # Count initial status
    local INITIAL_COMPLETED=$(count_status "COMPLETED")
    local INITIAL_FAILED=$(count_status "FAILED")
    
    if [ $RESUME -eq 1 ]; then
        echo -e "${YELLOW}Resuming from previous run...${NC}"
        echo "  Skipping $INITIAL_COMPLETED already completed"
        if [ $INITIAL_FAILED -gt 0 ]; then
            echo "  Will retry $INITIAL_FAILED failed experiments"
        fi
        echo ""
    fi
    
    # Count pending
    local PENDING=0
    for config in "${CONFIGS[@]}"; do
        if ! is_completed "$config"; then
            PENDING=$((PENDING + 1))
        fi
    done
    
    echo -e "${BLUE}Queue:${NC} $PENDING experiments pending"
    echo ""
    
    if [ $PENDING -eq 0 ]; then
        echo -e "${GREEN}All experiments already completed!${NC}"
        show_final_summary
        exit 0
    fi
    
    # Run experiments
    local COMPLETED_COUNT=$INITIAL_COMPLETED
    local FAILED_COUNT=$INITIAL_FAILED
    local CURRENT_RUNNING=0
    
    if [ $PARALLEL -eq 1 ]; then
        # Parallel mode with progress updates
        echo -e "${CYAN}Running in parallel mode on $NUM_GPUS GPU(s)...${NC}"
        echo ""
        
        declare -a PIDS
        declare -a GPU_ASSIGNMENTS
        declare -a CONFIG_NAMES
        
        for config in "${CONFIGS[@]}"; do
            # Skip if already done
            if is_completed "$config"; then
                continue
            fi
            
            # If not resuming, skip failed ones too
            if [ $RESUME -eq 0 ] && is_failed "$config"; then
                echo -e "  ${YELLOW}[SKIP]${NC} $(basename "$config" .yaml) (failed previously, use --resume to retry)"
                continue
            fi
            
            # Wait for a GPU slot to open
            while [ ${#PIDS[@]} -ge $MAX_JOBS ]; do
                # Check for completed jobs
                local i=0
                while [ $i -lt ${#PIDS[@]} ]; do
                    if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
                        # Job finished - check exit status
                        wait ${PIDS[$i]}
                        local exit_code=$?
                        if [ $exit_code -eq 0 ]; then
                            COMPLETED_COUNT=$((COMPLETED_COUNT + 1))
                        else
                            FAILED_COUNT=$((FAILED_COUNT + 1))
                        fi
                        
                        # Remove from arrays
                        unset PIDS[$i]
                        unset GPU_ASSIGNMENTS[$i]
                        unset CONFIG_NAMES[$i]
                        
                        # Rebuild arrays (compact)
                        PIDS=("${PIDS[@]}")
                        GPU_ASSIGNMENTS=("${GPU_ASSIGNMENTS[@]}")
                        CONFIG_NAMES=("${CONFIG_NAMES[@]}")
                    else
                        i=$((i + 1))
                    fi
                done
                
                # Show summary periodically
                local CURRENT_TIME=$(date +%s)
                local ELAPSED=$((CURRENT_TIME - START_TIME))
                show_summary_table $COMPLETED_COUNT $FAILED_COUNT ${#PIDS[@]} $((PENDING - COMPLETED_COUNT - FAILED_COUNT)) $ELAPSED
                
                sleep 5
            done
            
            # Find available GPU (round-robin)
            local GPU_ID=$((COMPLETED_COUNT % NUM_GPUS))
            
            # Run in background
            run_experiment "$config" "$GPU_ID" &
            PIDS+=($!)
            GPU_ASSIGNMENTS+=($GPU_ID)
            CONFIG_NAMES+=($(basename "$config" .yaml))
            
            echo "  [BG] Launched job on GPU $GPU_ID (PID: $!)"
            
            # Show summary immediately
            local CURRENT_TIME=$(date +%s)
            local ELAPSED=$((CURRENT_TIME - START_TIME))
            show_summary_table $COMPLETED_COUNT $FAILED_COUNT ${#PIDS[@]} $((PENDING - COMPLETED_COUNT - FAILED_COUNT - ${#PIDS[@]})) $ELAPSED
        done
        
        # Wait for remaining jobs
        echo ""
        echo -e "${CYAN}Waiting for ${#PIDS[@]} remaining jobs to complete...${NC}"
        for pid in "${PIDS[@]}"; do
            wait $pid
            local exit_code=$?
            if [ $exit_code -eq 0 ]; then
                COMPLETED_COUNT=$((COMPLETED_COUNT + 1))
            else
                FAILED_COUNT=$((FAILED_COUNT + 1))
            fi
        done
        
    else
        # Sequential mode
        echo -e "${CYAN}Running in sequential mode...${NC}"
        echo ""
        
        for config in "${CONFIGS[@]}"; do
            if is_completed "$config"; then
                COMPLETED_COUNT=$((COMPLETED_COUNT + 1))
                continue
            fi
            
            if [ $RESUME -eq 0 ] && is_failed "$config"; then
                echo -e "  ${YELLOW}[SKIP]${NC} $(basename "$config" .yaml) (failed previously, use --resume to retry)"
                FAILED_COUNT=$((FAILED_COUNT + 1))
                continue
            fi
            
            run_experiment "$config" 0
            local exit_code=$?
            if [ $exit_code -eq 0 ]; then
                COMPLETED_COUNT=$((COMPLETED_COUNT + 1))
            else
                FAILED_COUNT=$((FAILED_COUNT + 1))
                if [ $FAILED_COUNT -gt 3 ]; then
                    echo ""
                    echo -e "${RED}Too many failures ($FAILED_COUNT), stopping.${NC}"
                    break
                fi
            fi
            
            # Show progress
            local CURRENT_TIME=$(date +%s)
            local ELAPSED=$((CURRENT_TIME - START_TIME))
            show_summary_table $COMPLETED_COUNT $FAILED_COUNT 0 $((PENDING - COMPLETED_COUNT - FAILED_COUNT)) $ELAPSED
        done
    fi
    
    # Clear trap before showing final summary
    trap - INT TERM
    cleanup
    
    # Show final summary
    show_final_summary
}

# Run main
main "$@"
