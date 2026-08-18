#!/bin/bash
# Overnight yarn-B' battery + stability SOAK (2026-08-17 night, stock clocks).
# Dual purpose: (1) rope-unification verdict per AUTO-TIER-DESIGN B' spec;
# (2) discriminating soak after OC removal — any GPU death tonight refutes
# the OC conviction. Detached-safe, resumable (ledger skip), per-arm timeout.
set -u
Q=/home/erol/ai/turboquant/turboquant-g1/quality-tests
YB=$Q/yarnB
CONF=/home/erol/.config/llama-tcq
BIN=/home/erol/ai/turboquant/turboquant-kv-cache/build-g1/bin/llama-server
MODEL=/home/erol/ai/turboquant/models/qwen38/Qwen3.8-27B-Q6_K.gguf
LEDGER=$YB/ledger.txt
mkdir -p "$YB"
touch "$LEDGER"
log(){ echo "$(date +%H:%M:%S) $*" >> "$YB/orchestrator.log"; }

stop_all(){ bash $CONF/stop.sh >/dev/null 2>&1; SPID=$(ss -tlnp 2>/dev/null | grep ':8131 ' | grep -oP 'pid=\K[0-9]+' | head -1); [ -n "${SPID:-}" ] && kill $SPID 2>/dev/null; sleep 3; }

launch(){ # $1=scale-name (none|125|156)
    local ROPE=""
    case "$1" in
        125) ROPE="--rope-scaling yarn --rope-scale 1.25 --yarn-orig-ctx 262144";;
        156) ROPE="--rope-scaling yarn --rope-scale 1.5625 --yarn-orig-ctx 262144";;
    esac
    export TURBO_NORM_ALPHA_V=1.00 TURBO4_NORM_ALPHA_V=1.00 GGML_TURBO_MMA_FUSED=1
    setsid nohup $BIN -m $MODEL \
      --spec-type draft-mtp --spec-draft-n-max 3 \
      -ctkd turbo4 -ctvd turbo4 -ctk turbo4 -ctv turbo4 \
      -fa on -ngl 99 -c 262144 --no-context-shift $ROPE \
      --jinja --reasoning-format none --parallel 1 -b 512 -ub 512 \
      --temp 1.0 --top-p 0.95 --top-k 20 \
      --host 127.0.0.1 --port 8131 < /dev/null > "$YB/server-$1.log" 2>&1 &
    echo $! > "$YB/server.pid"
    for i in $(seq 1 160); do
        c=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 http://127.0.0.1:8131/health 2>/dev/null)
        [ "$c" = "200" ] && return 0
        kill -0 $(cat "$YB/server.pid") 2>/dev/null || { log "SERVER DIED during launch scale=$1"; return 1; }
        sleep 3
    done
    log "SERVER TIMEOUT scale=$1"; return 1
}

run_arm(){ # $1=scale $2=seed $3=depths $4=labelsuffix
    local LABEL="yarnB_$1_s$2$4"
    grep -q "^DONE $LABEL$" "$LEDGER" && { log "skip $LABEL (done)"; return 0; }
    log "arm $LABEL start"
    timeout 4800 python3 $Q/trajectory_battery.py --port 8131 --depths "$3" \
        --temp 1.0 --seed "$2" --label "$LABEL" > "$YB/$LABEL.out" 2>&1
    local RC=$?
    if [ $RC -eq 0 ]; then echo "DONE $LABEL" >> "$LEDGER"; log "arm $LABEL done";
    elif [ $RC -eq 124 ]; then echo "TIMEOUT $LABEL" >> "$LEDGER"; log "arm $LABEL TIMEOUT";
    else echo "FAIL $LABEL rc=$RC" >> "$LEDGER"; log "arm $LABEL FAIL rc=$RC";
         kill -0 $(cat "$YB/server.pid") 2>/dev/null || { log "SERVER DEAD at $LABEL - SOAK EVENT"; return 1; }
    fi
    return 0
}

log "=== OVERNIGHT START (stock clocks) ==="
# Step 0: imatrix v2 (GPU) — once
if ! grep -q "^DONE imatrix_v2$" "$LEDGER"; then
    stop_all
    log "imatrix v2 start (1200 chunks)"
    timeout 4200 /home/erol/ai/turboquant/turboquant-sync/build/bin/llama-imatrix \
        -m $MODEL -f $Q/quant-lab/calib_v2.txt -o $Q/quant-lab/imatrix_v2.dat \
        -ngl 99 -c 4096 --chunks 1200 > "$YB/imatrix_v2.out" 2>&1 \
        && echo "DONE imatrix_v2" >> "$LEDGER" && log "imatrix v2 done" \
        || log "imatrix v2 FAILED/TIMEOUT"
fi
# B' matrix: 3 scales x 3 seeds at 64K+128K
for SCALE in none 125 156; do
    stop_all
    launch $SCALE || { log "launch failed scale=$SCALE — trying next scale"; continue; }
    for SEED in 42 43 44; do
        run_arm $SCALE $SEED "64000,128000" "" || break
    done
done
# Finalists: 1.25 vs 1.5625 at 256K depth, seed 42 (none excluded: can't exceed native)
for SCALE in 125 156; do
    stop_all
    launch $SCALE || continue
    run_arm $SCALE 42 "250000" "_final" || true
done
stop_all
log "=== OVERNIGHT COMPLETE ==="
echo "COMPLETE $(date +%H:%M:%S)" >> "$LEDGER"
