#!/bin/bash
# tqmix gate server launcher — production-matched config on port 8233.
# Usage: gate_server.sh <model.gguf>   (writes pidfile beside this script)
set -u
HERE=/home/erol/ai/turboquant/turboquant-g1/quality-tests/quant-lab
MODEL="$1"
export TURBO_NORM_ALPHA_V=1.00
export TURBO4_NORM_ALPHA_V=1.00
export GGML_TURBO_MMA_FUSED=1
export GGML_CUDA_PDL=0
setsid nohup /home/erol/ai/turboquant/turboquant-kv-cache/build-g1/bin/llama-server \
  -m "$MODEL" \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  -ctkd turbo4 -ctvd turbo4 -ctk turbo4 -ctv turbo4 \
  -fa on -ngl 99 -c 327680 --no-context-shift \
  --rope-scaling yarn --rope-scale 1.25 --yarn-orig-ctx 262144 \
  --jinja --reasoning-format none --parallel 1 -b 512 -ub 512 \
  --temp 1.0 --top-p 0.95 --top-k 20 \
  --host 127.0.0.1 --port 8233 < /dev/null > "$HERE/gate-server.log" 2>&1 &
GPID=$!
echo $GPID > "$HERE/gate.pid"
for i in $(seq 1 120); do
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 http://127.0.0.1:8233/health 2>/dev/null)
    [ "$code" = "200" ] && break
    kill -0 "$GPID" 2>/dev/null || { echo "GATE SERVER DIED"; tail -4 "$HERE/gate-server.log"; exit 1; }
    sleep 3
done
[ "$code" = "200" ] || { echo "GATE SERVER TIMEOUT"; exit 1; }
echo "GATE SERVER HEALTHY pid=$GPID"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
