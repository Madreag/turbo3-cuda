#!/bin/bash
# LONG-CONTEXT profile — Qwen3.8-27B at 409,600 ctx via YaRN 1.5625.
# VALIDATED 2026-08-14 battery (session f5334395):
#   - Effort ladder: reasoning_effort=xhigh @ temp 1.0 (vendor defaults) wins:
#     3/4 clean pixel-gated renders vs medium 2/4, low 0/2, xhigh@0.6 1/2.
#     xhigh cost: ~87K tokens / ~30 min per pagoda-class artifact turn.
#   - NIAH: effective 5/5 at 130K AND 380K (348,798 real ptok) through
#     turbo4+YaRN; CTRL "FABRICATED" flags are scorer false-positives (model
#     refuses correctly while quoting the real Meridian code).
#   - Binary: build-g1/bin/llama-server with NextN port (commit 815b14d61) —
#     REQUIRED for 3.8 (embedded MTP block); serves 3.6 unchanged.
# Rollback: start-long.sh (3.6) + build-g1/bin/llama-server.pre-nextn.
# 3.6 slot saves archived in slots-long/pre-38-backup/ (incompatible states).
#
# Config notes vs start-long.sh (3.6):
#   - temp 1.0 (vendor default; 0.6 arm showed no benefit and one 131K-cap
#     truncation from a think spiral). top-k 20 / top-p 0.95 unchanged.
#   - reasoning_effort: xhigh is the template default — deliberately not set.
#   - alphas 1.10/1.12 carried from 3.6 (validated empirically by ladder+NIAH;
#     KLD spot-check deferred, geometry identical).
#   - KNOWN TAIL RISK: xhigh think occasionally near/at Hermes's 131072
#     max_tokens (1 of 12 ladder runs hit it) → finish=length, truncated
#     artifact, properly terminated. Knob lives Hermes-side if it bites.
cd /home/erol/ai/turboquant/turboquant-kv-cache
TCQ_KEY=$(cat /home/erol/.config/llama-tcq/api.key)

CTX=409600
SCALE=1.5625          # = CTX / 262144 (native), same factor as 3.6 profile

# Alphas re-tuned for Qwen3.8 (2026-08-15 sweep, corrected methodology:
# 2048-tok prompts w/ cross-ubatch quantized-cache readback, f16 noise floor
# = 0.0, n=60): alpha 1.00 is the bracketed KLD minimum (0.0087 vs 0.0200 at
# the old 3.6-inherited 1.10/1.12; confirmed under production YaRN 0.0150 vs
# 0.0243, top-1 96.7% vs 90.0%). Qwen3.8 wants NO V-norm correction.
export TURBO_NORM_ALPHA_V=1.00
export TURBO4_NORM_ALPHA_V=1.00

mkdir -p /home/erol/.config/llama-tcq/slots-long
# MTP speculative decode (2026-08-15 A/B at production sampling): coding
# 89-98 tok/s vs 57 baseline (1.55-1.71x, draft acceptance 59-63%), general
# 1.66x. Cost: art/tool-grammar turns measured 0.6-0.85x (n=2, grammar-draft
# interaction suspected — investigation lever). Enabled because primary use
# is coding/general per user direction. No VRAM cost at full profile.
nohup ./build-g1/bin/llama-server \
  -m /home/erol/ai/turboquant/models/qwen38/Qwen3.8-27B-Q6_K.gguf \
  --mmproj /home/erol/ai/turboquant/models/qwen38/mmproj-F16.gguf \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  -ctk turbo4 -ctv turbo4 \
  -fa on -ngl 99 -c $CTX --no-context-shift \
  --rope-scaling yarn --rope-scale $SCALE --yarn-orig-ctx 262144 \
  --jinja --reasoning-format none \
  --chat-template-kwargs '{"preserve_thinking": true}' \
  --parallel 1 --swa-full -b 2048 -ub 512 \
  --temp 1.0 --top-p 0.95 --top-k 20 \
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

echo "LONG-38 profile: llama-server PID $(cat /home/erol/.config/llama-tcq/server.pid) at ctx=$CTX yarn=$SCALE (Qwen3.8-27B)"
echo "proxy PID:       $(cat /home/erol/.config/llama-tcq/proxy.pid) (0.0.0.0:8130)"
echo "Rollback:        stop.sh && cp build-g1/bin/llama-server.pre-nextn build-g1/bin/llama-server && start-long.sh"
