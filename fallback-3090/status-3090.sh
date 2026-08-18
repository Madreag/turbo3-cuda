#!/bin/bash
HERE="$(cd "$(dirname "$0")" && pwd)"
echo "== health: $(curl -s -o /dev/null -w "%{http_code}" --max-time 2 http://127.0.0.1:8131/health 2>/dev/null) (200 = up)"
echo "== proxy:  $(curl -s -o /dev/null -w "%{http_code}" --max-time 2 http://127.0.0.1:8130/v1/models 2>/dev/null)"
for F in server.pid proxy.pid; do
    P=$(cat "$HERE/$F" 2>/dev/null)
    if [ -n "${P:-}" ] && kill -0 "$P" 2>/dev/null; then echo "== $F $P ALIVE"; else echo "== $F ${P:-none} DEAD"; fi
done
nvidia-smi --query-gpu=memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader 2>/dev/null
tail -3 "$HERE/server.log" 2>/dev/null
