#!/bin/bash
# LONG-CONTEXT profile — Qwen3.6-27B at 409,600 ctx via YaRN 1.5625.
# VALIDATED 2026-07-28: 5/5 at 381,781 actual tokens (needles @10/50/90% depth,
# 3-hop chain, no-fabrication control). Raw (no YaRN) fabricated on the control
# at 300K — YaRN restored grounding. No short-context PPL tax at this factor
# (5.5394 vs 5.5474 baseline @ctx=512). Results: quality-tests/niah_results/battery_*.json
# Companion to start.sh (daily driver, 262K native, q8_0 K + turbo4 V, no YaRN).
#
# Differences vs start.sh:
#   - KV: turbo4/turbo4 (17 KiB/token) — q8_0 K does not fit past ~300K on 32GB.
#     Measured cost: +0.49% PPL vs q8_0 K (ctx=512, wiki.test.raw 32ch).
#   - YaRN: factor sized to target per Qwen guidance (never 4.0).
#     Measured: NO short-context tax at these factors (PPL 5.5394 @1.5625 vs 5.5474 baseline).
#   - CTX below: set from validation verdict (409600 if 380K multi-hop holds, else 307200).
cd /home/erol/ai/turboquant/turboquant-kv-cache
TCQ_KEY=$(cat /home/erol/.config/llama-tcq/api.key)

CTX=409600            # ← finalize from battery verdict
SCALE=1.5625          # = CTX / 262144; use 1.171875 for CTX=307200

export TURBO_NORM_ALPHA_V=1.10
export TURBO4_NORM_ALPHA_V=1.12

mkdir -p /home/erol/.config/llama-tcq/slots-long
nohup ./build-g1/bin/llama-server \
  -m /home/erol/ai/turboquant/models/Qwen3.6-27B-Q6_K.gguf \
  -ctk turbo4 -ctv turbo4 \
  -fa on -ngl 99 -c $CTX --no-context-shift \
  --rope-scaling yarn --rope-scale $SCALE --yarn-orig-ctx 262144 \
  --jinja --reasoning-format none \
  --chat-template-kwargs '{"preserve_thinking": true}' \
  --parallel 1 --swa-full -b 2048 -ub 512 \
  --temp 0.6 --top-p 0.95 --top-k 20 \
  --slot-save-path /home/erol/.config/llama-tcq/slots-long/ \
  --host 127.0.0.1 --port 8131 \
  --api-key "$TCQ_KEY" \
  > /home/erol/.config/llama-tcq/server.log 2>&1 &
echo $! > /home/erol/.config/llama-tcq/server.pid

for i in $(seq 1 90); do
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 http://127.0.0.1:8131/health 2>/dev/null)
  [ "$code" = "200" ] && break
  sleep 2
done
grep -m1 "new slot, n_ctx" /home/erol/.config/llama-tcq/server.log

nohup python3 /home/erol/.config/llama-tcq/proxy.py \
  --upstream http://127.0.0.1:8131 \
  --host 0.0.0.0 --port 8130 \
  > /home/erol/.config/llama-tcq/proxy.log 2>&1 &
echo $! > /home/erol/.config/llama-tcq/proxy.pid

echo "LONG profile: llama-server PID $(cat /home/erol/.config/llama-tcq/server.pid) at ctx=$CTX yarn=$SCALE"
echo "proxy PID:    $(cat /home/erol/.config/llama-tcq/proxy.pid) (0.0.0.0:8130)"
echo "Switch back:  stop.sh && start.sh"
