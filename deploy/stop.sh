#!/bin/bash
for name in proxy server; do
    PID_FILE=/home/erol/.config/llama-tcq/${name}.pid
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID"
            echo "Stopped ${name} PID $PID"
        else
            echo "${name} PID $PID not running"
        fi
        rm -f "$PID_FILE"
    fi
done
# Safety net
pkill -f 'llama-server.*8131' 2>/dev/null && echo "killed extra llama-server"
pkill -f 'proxy.py --upstream' 2>/dev/null && echo "killed extra proxy"
exit 0
