#!/bin/bash
# Live monitor for the llama-tcq stack. Usage: watch.sh [interval_seconds]
# Shows: slot state, context fill, prefill/decode speeds, GPU load. Ctrl-C to exit.
INT=${1:-2}
KEY=$(cat /home/erol/.config/llama-tcq/api.key)
LOG=/home/erol/.config/llama-tcq/server.log

while true; do
  clear
  echo "=== llama-tcq live monitor ($(date +%H:%M:%S)) — Ctrl-C to exit ==="
  # Slot state from the server itself
  curl -s --max-time 2 -H "Authorization: Bearer $KEY" http://127.0.0.1:8131/slots 2>/dev/null | python3 -c "
import sys, json
try:
    for s in json.load(sys.stdin):
        state = 'GENERATING' if s.get('is_processing') else 'idle'
        print(f\"slot {s.get('id')}: {state} | window: {s.get('n_ctx'):,} tokens | temp={s.get('params',{}).get('temperature',0):.2f} top_k={s.get('params',{}).get('top_k')}\")
except Exception:
    print('slot state: server unreachable')
"
  # Context fill + progress from log (last prompt-processing line)
  LASTPROG=$(grep "prompt processing progress" "$LOG" 2>/dev/null | tail -1)
  if [ -n "$LASTPROG" ]; then
    NTOK=$(echo "$LASTPROG" | grep -oP 'n_tokens =\s*\K[0-9]+' | head -1)
    PROG=$(echo "$LASTPROG" | grep -oP 'progress =\s*\K[0-9.]+')
    PCT=$(python3 -c "print(f'{100*$NTOK/409600:.1f}')" 2>/dev/null)
    echo "context fill: ${NTOK:-?} tokens (${PCT:-?}% of 409,600) | last prefill batch progress: ${PROG:-?}"
  fi
  # Speeds (last completed request)
  echo "--- last completed request ---"
  grep -E "prompt eval time|^ +eval time" "$LOG" 2>/dev/null | tail -2 | sed 's/^ *//'
  # GPU
  echo "--- GPU ---"
  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free,power.draw,temperature.gpu --format=csv,noheader 2>/dev/null
  sleep "$INT"
done
