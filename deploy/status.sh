#!/bin/bash
# Stack status. Truth = ports + /health probes, not pidfiles — a stale or
# missing pidfile with live processes previously read as "Not running" and
# invited a double-start (2026-08-15 bughunt #11).
CONF=/home/erol/.config/llama-tcq
for name in server proxy; do
    PID_FILE=$CONF/$name.pid
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "$name: pidfile PID $(cat "$PID_FILE") alive"
    else
        echo "$name: no live pidfile"
    fi
done
echo "--- ports"
ss -tlnp 2>/dev/null | grep -E ':(8130|8131)\s' || echo "none bound"
echo "--- health"
for p in 8131 8130; do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://127.0.0.1:$p/health" 2>/dev/null)
    echo ":$p → ${code:-no response}"
done
echo "--- gpu"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null
# swap tripwire (club-3090 P6): serving-process pages in swap silently
# invalidate every perf number — surface it here.
spid=$(ss -tlnp 2>/dev/null | grep ':8131 ' | grep -oP 'pid=\K[0-9]+' | head -1)
if [ -n "$spid" ]; then
    vsw=$(grep VmSwap /proc/$spid/status 2>/dev/null | awk '{print $2, $3}')
    echo "--- swap: server VmSwap = ${vsw:-n/a}"
fi
