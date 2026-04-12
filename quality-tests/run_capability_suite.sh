#!/bin/bash
# TurboQuant Capability Comparison Suite
# Compares f16, q8_0, and turbo3 KV cache on tool calling, instruction following, reasoning.
#
# Usage: bash quality-tests/run_capability_suite.sh [model_path]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SERVER="$REPO_DIR/build/bin/llama-server"
MODEL="${1:-${MODEL:?Set MODEL env var or pass model path as argument}}"
PORT=8090
MAX_CTX=40000

TYPES="f16 q8_0 turbo3"
CONTEXTS="2048 32768"

echo "=============================================="
echo " TurboQuant Capability Comparison Suite"
echo "=============================================="
echo "Model:    $MODEL"
echo "Types:    $TYPES"
echo "Contexts: $CONTEXTS"
echo "Server:   $SERVER"
echo ""

# --- Server management ---

start_server() {
    local TYPE="$1"
    echo ">>> Starting server with KV type: $TYPE"

    # Kill any existing server on this port
    pkill -f "llama-server.*--port $PORT" 2>/dev/null || true
    sleep 2

    "$SERVER" \
        -m "$MODEL" \
        -ctk "$TYPE" -ctv "$TYPE" \
        -fa on \
        -ngl 99 \
        -c "$MAX_CTX" \
        --port "$PORT" \
        --jinja \
        --reasoning-budget -1 \
        --no-mmap \
        --log-disable &
    SERVER_PID=$!

    # Wait for health
    echo -n "    Waiting for server..."
    for i in $(seq 1 120); do
        if curl -s "http://localhost:$PORT/health" 2>/dev/null | grep -q "ok"; then
            echo " ready (${i}s)"
            return 0
        fi
        sleep 1
    done
    echo " TIMEOUT"
    kill $SERVER_PID 2>/dev/null || true
    return 1
}

stop_server() {
    echo ">>> Stopping server"
    pkill -f "llama-server.*--port $PORT" 2>/dev/null || true
    sleep 2
}

# --- Main loop ---

for TYPE in $TYPES; do
    echo ""
    echo "=============================================="
    echo " KV Type: $TYPE"
    echo "=============================================="

    start_server "$TYPE"

    for CTX in $CONTEXTS; do
        echo ""
        echo "--- Context depth: $CTX ---"
        python3 "$SCRIPT_DIR/capability_test.py" \
            --port "$PORT" \
            --label "$TYPE" \
            --context-depth "$CTX" \
            --reps 3 \
            --output "$SCRIPT_DIR/capability_${TYPE}_ctx${CTX}.json"
    done

    stop_server
done

# --- Comparison Summary ---

echo ""
echo "=============================================="
echo " COMPARISON SUMMARY"
echo "=============================================="

python3 -c "
import json, sys, os

script_dir = '$SCRIPT_DIR'
types = '$TYPES'.split()
contexts = [int(x) for x in '$CONTEXTS'.split()]

# Load all results
data = {}
for t in types:
    for c in contexts:
        path = os.path.join(script_dir, f'capability_{t}_ctx{c}.json')
        if os.path.exists(path):
            with open(path) as f:
                data[(t, c)] = json.load(f)

if not data:
    print('No results found!')
    sys.exit(1)

def get_acc(d, cat):
    return d.get('categories', {}).get(cat, {}).get('accuracy_avg', d.get('categories', {}).get(cat, {}).get('accuracy', 0))

# Print results table
for c in contexts:
    has_agentic = any(get_acc(data.get((t,c),{}), 'agentic_coding') > 0 for t in types)
    print(f'\nContext: {c} tokens (avg across reps)')
    hdr = f'  {\"\":20s} {\"Tool Call\":>10s} {\"Instruct\":>10s} {\"Reason\":>10s}'
    if has_agentic:
        hdr += f' {\"Agentic\":>10s}'
    hdr += f' {\"Overall\":>10s}'
    print(hdr)
    print('  ' + '-' * (72 if has_agentic else 62))
    for t in types:
        key = (t, c)
        if key in data:
            d = data[key]
            tc = get_acc(d, 'tool_calling')
            if_ = get_acc(d, 'instruction_following')
            re = get_acc(d, 'reasoning')
            ac = get_acc(d, 'agentic_coding')
            ov = d.get('overall_accuracy_avg', d.get('overall_accuracy', 0))
            row = f'  {t:20s} {tc:>9.1f}% {if_:>9.1f}% {re:>9.1f}%'
            if has_agentic:
                row += f' {ac:>9.1f}%'
            row += f' {ov:>9.1f}%'
            print(row)

# Print deltas from f16
print(f'\nDELTA from f16 baseline (avg):')
hdr = f'  {\"\":20s} {\"Tool Call\":>10s} {\"Instruct\":>10s} {\"Reason\":>10s} {\"Overall\":>10s}'
print(hdr)
print('  ' + '-' * 62)
for t in types:
    if t == 'f16':
        continue
    for c in contexts:
        base = data.get(('f16', c))
        curr = data.get((t, c))
        if base and curr:
            dtc = get_acc(curr, 'tool_calling') - get_acc(base, 'tool_calling')
            dif = get_acc(curr, 'instruction_following') - get_acc(base, 'instruction_following')
            dre = get_acc(curr, 'reasoning') - get_acc(base, 'reasoning')
            dov = (curr.get('overall_accuracy_avg', curr.get('overall_accuracy', 0)) -
                   base.get('overall_accuracy_avg', base.get('overall_accuracy', 0)))
            print(f'  {t + \" @ \" + str(c//1024) + \"k\":20s} {dtc:>+9.1f}% {dif:>+9.1f}% {dre:>+9.1f}% {dov:>+9.1f}%')
"

echo ""
echo "Done. JSON results in $SCRIPT_DIR/capability_*.json"
