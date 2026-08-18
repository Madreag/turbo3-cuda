#!/bin/bash
# The ONLY sanctioned way to stop the 3090 fallback stack.
HERE="$(cd "$(dirname "$0")" && pwd)"
for F in server.pid proxy.pid; do
    P=$(cat "$HERE/$F" 2>/dev/null)
    [ -n "${P:-}" ] && kill "$P" 2>/dev/null && echo "stopped $F ($P)"
done
sleep 2
for PORT in 8130 8131; do
    P=$(ss -tlnp 2>/dev/null | grep ":$PORT " | grep -oP 'pid=\K[0-9]+' | head -1)
    [ -n "${P:-}" ] && kill -9 "$P" 2>/dev/null && echo "force-killed port $PORT ($P)"
done
echo "stack down"
