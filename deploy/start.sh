#!/bin/bash
cd /home/erol/ai/turboquant/turboquant-kv-cache
TCQ_KEY=$(cat /home/erol/.config/llama-tcq/api.key)

# Calibrated V-norm alpha values (Qwen3.6-27B Q6_K, wiki.test.raw 32 chunks, 2026-07-27)
# Previous model (Qwopus3.5-27B-v3): TURBO_NORM_ALPHA_V=1.04, TURBO4_NORM_ALPHA_V=1.10
export TURBO_NORM_ALPHA_V=1.10
export TURBO4_NORM_ALPHA_V=1.12

# Start upstream llama-server on 127.0.0.1:8131 (proxy will forward to this)
# --reasoning-format none keeps <think> inline in content → cache works
# Proxy extracts <think>...</think> to reasoning_content field for clean client UX
# Qwen3.6: model-card sampling (temp 1.0 / top-p 0.95 / top-k 20). top-k 20 is REQUIRED —
# at server-default top-k 40 thinking never terminates. presence-penalty 1.5 was a
# Qwen3.5-only fix, dropped (risks penalizing repeated JSON keys in tool calls).
mkdir -p /home/erol/.config/llama-tcq/slots
nohup ./build-g1/bin/llama-server \
  -m /home/erol/ai/turboquant/models/Qwen3.6-27B-Q6_K.gguf \
  -ctk q8_0 -ctv turbo4 \
  -fa on -ngl 99 -c 262144 --no-context-shift \
  --jinja --reasoning-format none \
  --chat-template-kwargs '{"preserve_thinking": true}' \
  --parallel 1 --swa-full \
  --temp 0.6 --top-p 0.95 --top-k 20 \
  --slot-save-path /home/erol/.config/llama-tcq/slots/ \
  --host 127.0.0.1 --port 8131 \
  --api-key "$TCQ_KEY" \
  > /home/erol/.config/llama-tcq/server.log 2>&1 &
echo $! > /home/erol/.config/llama-tcq/server.pid

# Wait for upstream to be ready before starting proxy
for i in $(seq 1 60); do
  if curl -s --max-time 1 http://127.0.0.1:8131/health >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

# Start reasoning-extraction proxy on 0.0.0.0:8130 (public-facing)
# This is what clients (Droid) point at.
nohup python3 /home/erol/.config/llama-tcq/proxy.py \
  --upstream http://127.0.0.1:8131 \
  --host 0.0.0.0 --port 8130 \
  > /home/erol/.config/llama-tcq/proxy.log 2>&1 &
echo $! > /home/erol/.config/llama-tcq/proxy.pid

echo "llama-server PID: $(cat /home/erol/.config/llama-tcq/server.pid) (127.0.0.1:8131, internal)"
echo "proxy PID:         $(cat /home/erol/.config/llama-tcq/proxy.pid) (0.0.0.0:8130, client-facing)"
echo ""
echo "Clients (Droid) should use: http://<IP>:8130/v1"
echo "Log: tail -f /home/erol/.config/llama-tcq/server.log"
echo "Proxy log: tail -f /home/erol/.config/llama-tcq/proxy.log"
