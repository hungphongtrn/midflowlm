#!/bin/bash
# Smoke test for P3-D3 diagnostic pipeline
# Verifies: report generation from saved traces
#
# Usage:
#   # Report-only (requires existing traces):
#   bash scripts/smoke_diagnostic_p3d3.sh
#   bash scripts/smoke_diagnostic_p3d3.sh --traces-dir path/to/traces
#
#   # Full pipeline (requires checkpoint):
#   bash scripts/smoke_diagnostic_p3d3.sh --checkpoint path.pth --config cfg.yaml --mmlu-path mmlu.json --arc-path arc.json
set -euo pipefail

echo "=== P3-D3 Diagnostic Smoke Test ==="

OUTPUT_DIR="results/diagnostic_p3d3_smoke"
TRACES_DIR=""
CHECKPOINT=""
CONFIG=""
MMLU_PATH=""
ARC_PATH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --traces-dir) TRACES_DIR="$2"; shift 2 ;;
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --mmlu-path) MMLU_PATH="$2"; shift 2 ;;
        --arc-path) ARC_PATH="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$TRACES_DIR" ]; then
    if [ -d "results/diagnostic_p3d3/traces" ]; then
        TRACES_DIR="results/diagnostic_p3d3/traces"
        echo "Using existing traces: $TRACES_DIR"
    elif [ -n "$CHECKPOINT" ]; then
        TRACES_DIR="$OUTPUT_DIR/traces"
        echo "Will capture traces to: $TRACES_DIR"
    else
        echo "FAIL: No traces directory found. Provide --traces-dir or --checkpoint."
        exit 1
    fi
fi

T_VALUES=(1 2 8 64)
if [ -d "$TRACES_DIR" ]; then
    AVAILABLE_T=($(ls -d "$TRACES_DIR"/T*/ 2>/dev/null | sed 's|.*/T||' | sed 's|/||' | sort -n))
    if [ ${#AVAILABLE_T[@]} -gt 0 ]; then
        T_VALUES=("${AVAILABLE_T[@]}")
    fi
fi
echo "T values: ${T_VALUES[*]}"

if [ -n "$CHECKPOINT" ] && [ -n "$CONFIG" ] && [ -n "$MMLU_PATH" ] && [ -n "$ARC_PATH" ]; then
    echo ""
    echo "=== PHASE 1-2: Full Capture Pipeline ==="
    python3 scripts/diagnose_p3d3.py \
        --checkpoint "$CHECKPOINT" \
        --config "$CONFIG" \
        --mmlu-path "$MMLU_PATH" \
        --arc-path "$ARC_PATH" \
        --T "${T_VALUES[@]}" \
        --output-dir "$OUTPUT_DIR" \
        --seed 42
fi

echo ""
echo "=== PHASE 3: Report Generation ==="
python3 scripts/diagnose_p3d3.py \
    --report \
    --traces-dir "$TRACES_DIR" \
    --T "${T_VALUES[@]}" \
    --output-dir "$OUTPUT_DIR"

REPORT="$OUTPUT_DIR/report.md"
if [ ! -f "$REPORT" ]; then
    echo "FAIL: Report not generated at $REPORT"
    exit 1
fi

echo ""
echo "=== Verifying Report Contents ==="
for section in "Executive Summary" "Flow Integration Analysis" "Decoder/Readout Analysis" "Root Cause Decision Tree" "Recommendations"; do
    if ! grep -q "$section" "$REPORT"; then
        echo "FAIL: Report missing section: $section"
        exit 1
    fi
    echo "  ✓ $section"
done

echo ""
echo "PASS: Smoke test complete — report generated with all 5 sections"
