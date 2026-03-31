#!/bin/bash
#
# TurboQuant Quality Test Suite — Qwen3.5-9B Q8_0
#
# Matches 3090 Ti test setup:
#   - Model: Qwen3.5-9B Q8_0
#   - Types: q8_0 (baseline), turbo3, turbo2, turbo1.5
#   - Contexts: 4K through 131K
#   - Phase 1: Passkey retrieval
#   - Phase 2: NIAH (needle-in-a-haystack)
#
# Usage:
#   bash quality-tests/run_quality_suite.sh [phase1|phase2|all]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$REPO_DIR/build/bin"
MODEL="${MODEL:-/home/erol/ai/turboquant/models/Qwen3.5-9B-Q8_0.gguf}"
PORT=8090
MAX_CTX=140000

# Match 3090 Ti: skip turbo4 (broken on small models)
TYPES="q8_0 turbo3 turbo2 turbo1.5"

echo "=============================================="
echo "  TurboQuant Quality Test Suite (5090)"
echo "=============================================="
echo "Model:    $(basename $MODEL)"
echo "Types:    $TYPES"
echo "Port:     $PORT"
echo ""

[ -f "$BUILD_DIR/llama-server" ] || { echo "ERROR: llama-server not found"; exit 1; }
[ -f "$MODEL" ] || { echo "ERROR: Model not found at $MODEL"; exit 1; }
python3 -c "import requests" 2>/dev/null || pip install requests

start_server() {
    local TYPE=$1
    echo ""
    echo "--- Starting server: -ctk $TYPE -ctv $TYPE ---"
    pkill -f "llama-server.*$PORT" 2>/dev/null || true
    sleep 3

    "$BUILD_DIR/llama-server" \
        -m "$MODEL" \
        -ctk "$TYPE" -ctv "$TYPE" \
        -fa on -ngl 99 \
        -c $MAX_CTX \
        --port $PORT \
        --no-mmap \
        --log-disable \
        &
    SERVER_PID=$!

    for i in $(seq 1 120); do
        if curl -s "http://localhost:$PORT/health" 2>/dev/null | grep -q "ok"; then
            echo "Server ready (PID $SERVER_PID) after ${i}s"
            return 0
        fi
        [ $i -eq 120 ] && { echo "ERROR: Server timeout"; kill $SERVER_PID 2>/dev/null; exit 1; }
        sleep 1
    done
}

stop_server() {
    echo "Stopping server..."
    pkill -f "llama-server.*$PORT" 2>/dev/null || true
    wait 2>/dev/null || true
    sleep 3
    echo "Cooling GPU 20s..." && sleep 20
}

PHASE="${1:-all}"

run_phase1() {
    echo ""
    echo "=============================================="
    echo "  PHASE 1: Passkey Retrieval"
    echo "  4 types x 7 contexts x 5 depths x 3 reps"
    echo "  = 420 tests"
    echo "=============================================="

    for TYPE in $TYPES; do
        start_server "$TYPE"
        python3 "$SCRIPT_DIR/passkey_retrieval.py" \
            --port $PORT \
            --label "$TYPE" \
            --contexts "2048,4096,8192,16384,32768,65536,131072" \
            --depths "10,25,50,75,90" \
            --reps 3 \
            --output "$SCRIPT_DIR/passkey_results_${TYPE}.json"
        stop_server
    done
}

run_phase2() {
    echo ""
    echo "=============================================="
    echo "  PHASE 2: Needle-in-a-Haystack (NIAH)"
    echo "  4 types x 6 contexts x 5 depths x 1 rep"
    echo "  = 120 tests"
    echo "=============================================="

    for TYPE in $TYPES; do
        start_server "$TYPE"
        python3 "$SCRIPT_DIR/niah_test.py" \
            --port $PORT \
            --label "$TYPE" \
            --contexts "4096,8192,16384,32768,65536,131072" \
            --depths "10,25,50,75,90" \
            --reps 1 \
            --output "$SCRIPT_DIR/niah_results_${TYPE}.json"
        stop_server
    done
}

case "$PHASE" in
    phase1) run_phase1 ;;
    phase2) run_phase2 ;;
    all)    run_phase1; run_phase2 ;;
    *)      echo "Usage: $0 [phase1|phase2|all]"; exit 1 ;;
esac

echo ""
echo "=============================================="
echo "  Done! Results:"
echo "=============================================="
ls -la "$SCRIPT_DIR"/*.json 2>/dev/null

echo ""
echo "=== Quick comparison ==="
for f in "$SCRIPT_DIR"/passkey_results_*.json; do
    [ -f "$f" ] || continue
    python3 -c "
import json
with open('$f') as fh:
    d = json.load(fh)
print(f'{d[\"label\"]:>10}: passkey {d[\"overall_accuracy\"]:.1f}%')
"
done
for f in "$SCRIPT_DIR"/niah_results_*.json; do
    [ -f "$f" ] || continue
    python3 -c "
import json
with open('$f') as fh:
    d = json.load(fh)
print(f'{d[\"label\"]:>10}: NIAH    {d[\"overall_accuracy\"]:.1f}%')
"
done
