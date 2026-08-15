#!/bin/bash
# Verified stop for the llama-tcq stack (2026-08-15 bughunt A5 rewrite).
# The old script fired SIGTERM, deleted pidfiles, then IMMEDIATELY pkilled the
# same processes (a second SIGTERM landing mid-graceful-shutdown, aborting the
# proxy's shutdown slot-save) and exited 0 while ports were still bound —
# `stop.sh && start-…` reproduced the bind-conflict incident deterministically.
# This one: SIGTERM → poll until dead (proxy's slot-save can take ~30s) →
# SIGKILL after grace → safety net only for survivors → verify ports free.
CONF=/home/erol/.config/llama-tcq

stop_one() {
    local name=$1 grace=$2
    local pid_file=$CONF/$name.pid
    [ -f "$pid_file" ] || { echo "$name: no pidfile"; return 0; }
    local pid
    pid=$(cat "$pid_file")
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "$name PID $pid not running"
        rm -f "$pid_file"
        return 0
    fi
    kill "$pid"
    local i
    for i in $(seq 1 "$grace"); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
        echo "$name PID $pid still alive after ${grace}s — SIGKILL"
        kill -9 "$pid" 2>/dev/null
        sleep 1
    fi
    echo "stopped $name (PID $pid)"
    rm -f "$pid_file"
}

# Proxy first: its shutdown hook saves the current slot THROUGH the server,
# so the server must still be up while the proxy exits.
stop_one proxy 35
stop_one server 20

# Safety net — only if something is still bound after the verified kills.
if ss -tln 2>/dev/null | grep -qE ':(8130|8131)\s'; then
    pkill -f 'build-g1/bin/llama-server' 2>/dev/null && echo "killed stray llama-server"
    pkill -f 'proxy.py --upstream' 2>/dev/null && echo "killed stray proxy"
    sleep 2
fi
if ss -tln 2>/dev/null | grep -qE ':(8130|8131)\s'; then
    echo "ERROR: ports 8130/8131 still bound after stop" >&2
    ss -tlnp 2>/dev/null | grep -E ':(8130|8131)\s' >&2
    exit 1
fi
echo "stack down, ports free"
