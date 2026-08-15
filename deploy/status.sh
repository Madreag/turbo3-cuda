#!/bin/bash
for name in server proxy; do
    PID_FILE=/home/erol/.config/llama-tcq/${name}.pid
    if [ -f "$PID_FILE" ] && kill -0 "$(cat $PID_FILE)" 2>/dev/null; then
        echo "${name}: Running — PID $(cat $PID_FILE)"
    else
        echo "${name}: Not running"
    fi
done
echo ""
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>/dev/null
