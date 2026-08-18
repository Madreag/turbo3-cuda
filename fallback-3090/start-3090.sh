#!/bin/bash
# TurboQuant 3090 FALLBACK launcher — Qwen3.8-27B Q4_K_M (imatrix-v2 custom quant)
# on RTX 3090 24GB. Mirrors prod conventions: detach law, health gate, fixed alias.
# Default ctx 98304 (96K, ~1.3GB VRAM slack). Stretch: 131072 on a HEADLESS box only.
# ALIAS LAW: advertised id MUST be qwen3.8-27b-320k (Hermes breaks on any other
# name). Real ctx here is 96K — keep vacation sessions under ~90K tokens.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
CONF="$HERE"
CTX="${CTX:-98304}"

if ss -tlnp 2>/dev/null | grep -qE ':(8130|8131) '; then
    echo "ERROR: port 8130/8131 already bound — run stop-3090.sh first" >&2; exit 1
fi

TCQ_KEY=$(cat "$CONF/api.key")

# Prod-matched env: 3.8 alphas are 1.00; fused MMA on; PDL irrelevant on Ampere
# but pinned off for parity with the 5090 launchers.
export TURBO_NORM_ALPHA_V=1.00
export TURBO4_NORM_ALPHA_V=1.00
export GGML_TURBO_MMA_FUSED=1
export GGML_CUDA_PDL=0
# Binaries carry an absolute RUNPATH from the build box — point the loader at
# the packaged libs explicitly so the package is relocatable.
export LD_LIBRARY_PATH="$HERE/bin:${LD_LIBRARY_PATH:-}"

mkdir -p "$CONF/slots-3090"
setsid nohup "$HERE/bin/llama-server" \
  -m "$HERE/models/qwen38-q4km-imx2.gguf" \
  --mmproj "$HERE/models/mmproj-F16.gguf" --no-mmproj-offload \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  -ctkd turbo4 -ctvd turbo4 -ctk turbo4 -ctv turbo4 \
  -fa on -ngl 99 -c "$CTX" --no-context-shift \
  --jinja --reasoning-format none \
  --chat-template-kwargs '{"preserve_thinking": true}' \
  --parallel 1 -b 512 -ub 512 \
  --temp 1.0 --top-p 0.95 --top-k 20 \
  --slot-save-path "$CONF/slots-3090/" \
  --alias qwen3.8-27b-320k \
  --host 127.0.0.1 --port 8131 \
  --api-key "$TCQ_KEY" \
  < /dev/null > "$CONF/server.log" 2>&1 &
sleep 2
SP=$(ss -tlnp 2>/dev/null | grep ':8131 ' | grep -oP 'pid=\K[0-9]+' | head -1)
[ -z "${SP:-}" ] && SP=$(pgrep -xn llama-server)
echo "${SP:-0}" > "$CONF/server.pid"

# Cold model load on a fresh box can take minutes — allow 8 min like prod.
for i in $(seq 1 160); do
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 http://127.0.0.1:8131/health 2>/dev/null)
    [ "$code" = "200" ] && break
    kill -0 "$(cat "$CONF/server.pid")" 2>/dev/null || { echo "ERROR: server died during load:"; tail -8 "$CONF/server.log"; exit 1; }
    sleep 3
done
[ "$code" = "200" ] || { echo "ERROR: health timeout"; tail -8 "$CONF/server.log"; exit 1; }

setsid nohup python3 "$CONF/proxy.py" \
  --upstream http://127.0.0.1:8131 \
  --host 0.0.0.0 --port 8130 \
  < /dev/null > "$CONF/proxy.log" 2>&1 &
sleep 1
PP=$(ss -tlnp 2>/dev/null | grep ':8130 ' | grep -oP 'pid=\K[0-9]+' | head -1)
echo "${PP:-0}" > "$CONF/proxy.pid"

echo "3090 FALLBACK UP: server PID $(cat "$CONF/server.pid") (127.0.0.1:8131), proxy PID $(cat "$CONF/proxy.pid") (0.0.0.0:8130)"
echo "Hermes endpoint: http://<this-box-LAN-IP>:8130/v1  (model id: qwen3.8-27b-320k)"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
