#!/bin/bash
# NIGHT QUEUE 2026-08-18→19: post-soak decision-data campaign (user-directed).
# Waits for the 320K soak verdict; on COMPLETE runs, one at a time:
#   1. NATIVE-262 arm: speed probe + battery s42 @64K,128K (serving-choice data)
#   2. YARN-1.25 arm:  speed probe + battery s42 @64K,128K (B' head-to-head)
#   3. seeds 43 on both arms if time allows
#   4. v3-imatrix KLD gates (tqmix-v3-imx2, -imx1) if before 08:00
#   5. chunked-prefill op-tests (no serving risk) if before 08:40
# Then RESTORES 320K SPEED serving + writes MORNING-SUMMARY.
# Deadlines self-skip late steps. All milestones to D: (survive machine death).
# NO PDL, NO new kernels in serving path — stability-safe configs only.
set -u
Q=/home/erol/ai/turboquant/turboquant-g1/quality-tests
NQ=$Q/nightqueue
CONF=/home/erol/.config/llama-tcq
BIN=/home/erol/ai/turboquant/turboquant-kv-cache/build-g1/bin/llama-server
PROD=/home/erol/ai/turboquant/models/qwen38/Qwen3.8-27B-Q6_K.gguf
CUSTOM=/home/erol/ai/turboquant/models/qwen38-custom
D=/mnt/d/spill/qwen36-test
M=$D/nightqueue-milestones.log
mkdir -p "$NQ"
echo $$ > $D/nightqueue.pid
log(){ echo "$(date +%H:%M:%S) $*" >> "$M"; }
hour_lt(){ [ "$(date +%H%M)" -lt "$1" ]; }   # e.g. hour_lt 0800

stop_all(){ bash $CONF/stop.sh >/dev/null 2>&1; local P=$(ss -tlnp 2>/dev/null | grep -E ':(8131|8130) ' | grep -oP 'pid=\K[0-9]+' | head -1); [ -n "${P:-}" ] && kill $P 2>/dev/null; sleep 4; }

launch(){ # $1=scale(none|125) — omega-style keyless test server, port 8131
    local ROPE=""
    [ "$1" = "125" ] && ROPE="--rope-scaling yarn --rope-scale 1.25 --yarn-orig-ctx 262144"
    export TURBO_NORM_ALPHA_V=1.00 TURBO4_NORM_ALPHA_V=1.00 GGML_TURBO_MMA_FUSED=1 GGML_CUDA_PDL=0
    setsid nohup $BIN -m "$2" \
      --spec-type draft-mtp --spec-draft-n-max 3 \
      -ctkd turbo4 -ctvd turbo4 -ctk turbo4 -ctv turbo4 \
      -fa on -ngl 99 -c 262144 --no-context-shift $ROPE \
      --jinja --reasoning-format none --parallel 1 -b 512 -ub 512 \
      --temp 1.0 --top-p 0.95 --top-k 20 \
      --host 127.0.0.1 --port 8131 < /dev/null > "$NQ/server-cur.log" 2>&1 &
    sleep 3
    local SP=$(ss -tlnp 2>/dev/null | grep ':8131 ' | grep -oP 'pid=\K[0-9]+' | head -1)
    [ -z "$SP" ] && SP=$(pgrep -xn llama-server)
    echo "${SP:-0}" > $D/nq-server.pid
    for i in $(seq 1 160); do
        c=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 http://127.0.0.1:8131/health 2>/dev/null)
        [ "$c" = "200" ] && return 0
        kill -0 $(cat $D/nq-server.pid) 2>/dev/null || { log "NQEVENT: server died in launch $1"; return 1; }
        sleep 3
    done
    log "NQEVENT: launch $1 timeout"; return 1
}

gpu_ok(){ nvidia-smi --query-gpu=name --format=csv,noheader > /dev/null 2>&1; }

# ── STEP 0: wait for soak verdict (deadline 04:15) ──────────────────────────
log "NIGHTQUEUE armed — waiting for soak verdict"
while true; do
    grep -q "SOAK COMPLETE" $D/soak-milestones.log && { log "soak verdict: COMPLETE (PASS)"; break; }
    grep -qE "SOAK ENDED EARLY|SOAKEVENT: NVML" $D/soak-milestones.log && { log "ABORT: soak ended early — no science on an unproven box"; exit 1; }
    [ "$(date +%H%M)" -gt "0415" ] && [ "$(date +%H%M)" -lt "1200" ] && { log "ABORT: 04:15 deadline, no verdict"; exit 1; }
    gpu_ok || { log "ABORT: GPU gone during wait"; exit 1; }
    sleep 60
done

run_battery(){ # $1=scale $2=seed
    local LABEL="nq_${1}_s${2}"
    log "battery $LABEL start"
    timeout 4200 python3 $Q/trajectory_battery.py --port 8131 --depths "64000,128000" \
        --temp 1.0 --seed "$2" --label "$LABEL" > "$NQ/$LABEL.out" 2>&1
    log "battery $LABEL end rc=$?"
    gpu_ok || return 1
}
run_speed(){ # $1=tag
    log "speed probe $1"
    IGNORE_EOS=1 PORT=8131 timeout 900 python3 $Q/nmax/nmax_probe.py "nq-$1" > "$NQ/speed-$1.out" 2>&1
    log "speed probe $1 rc=$?"
}

# ── STEP 1+2: profile arms (native then yarn125): speed + battery s42 ──────
for ARM in none 125; do
    hour_lt 0700 || { log "skip arm $ARM (deadline)"; continue; }
    stop_all
    launch $ARM $PROD || continue
    run_speed "$ARM" || true
    run_battery "$ARM" 42 || { log "NQEVENT: GPU lost in battery $ARM s42"; exit 1; }
done
# seeds 43 if early
for ARM in none 125; do
    hour_lt 0630 || { log "skip s43 $ARM (deadline)"; continue; }
    stop_all; launch $ARM $PROD || continue
    run_battery "$ARM" 43 || { log "NQEVENT: GPU lost in battery $ARM s43"; exit 1; }
done

# ── STEP 3: v3 imatrix KLD gates ────────────────────────────────────────────
for V in v3imx2 v3imx1; do
    hour_lt 0800 || { log "skip gate $V (deadline)"; continue; }
    F=$CUSTOM/tqmix-$V.gguf; [ -f "$F" ] || F=$CUSTOM/tqmix-${V/imx/-imx}.gguf
    [ -f "$F" ] || { log "gate $V: model missing, skip"; continue; }
    stop_all
    launch 125 "$F" || { log "gate $V launch failed"; continue; }
    log "gate $V: KLD 157-prompt"
    (cd $Q && timeout 2700 python3 kl_divergence.py --port 8131 --type "$V" \
        --n-prompts 157 --prompt-tokens 2048 --output-dir kldnvfp4 > "$NQ/gate-$V-kld.out" 2>&1)
    RES=$(python3 -c "import json;d=json.load(open('$Q/kldnvfp4/kld_summary_$V.json'));print(f\"mean={d.get('kld_mean',d.get('mean','?'))} top1={d.get('top1_agree',d.get('top1','?'))}\")" 2>/dev/null || echo parse-fail)
    log "gate $V RESULT: $RES"
done

# ── STEP 4: chunked-prefill op-tests (GPU free, no serving) ────────────────
if hour_lt 0840; then
    stop_all
    log "chunked op-tests start"
    timeout 1200 /home/erol/ai/turboquant/turboquant-kv-cache/build-g1/test-backend-ops.chunked \
        -o GATED_DELTA_NET > "$NQ/chunked-optests.out" 2>&1
    log "chunked op-tests rc=$? : $(grep -cE 'OK|FAIL' $NQ/chunked-optests.out 2>/dev/null) result lines, $(grep -c FAIL $NQ/chunked-optests.out 2>/dev/null) FAIL"
fi

# ── FINAL: restore proven serving + summary ────────────────────────────────
stop_all
log "restoring 320K SPEED serving"
bash $CONF/start-long-38.sh > "$NQ/serving-restore.out" 2>&1 && log "SERVING RESTORED (320K SPEED)" || log "NQEVENT: SERVING RESTORE FAILED — check serving-restore.out"

{ echo "=== MORNING SUMMARY $(date +%F) ==="
  echo "--- milestones:"; cat "$M"
  echo "--- battery tails:"
  for f in $NQ/nq_*.out; do [ -f "$f" ] && echo "== $(basename $f)" && tail -3 "$f"; done
  echo "--- speed lines:"
  for f in $NQ/speed-*.out; do [ -f "$f" ] && echo "== $(basename $f)" && grep -E "decode=|prefill" "$f" | tail -4; done
  echo "--- KLD gates:"; grep "RESULT" "$M"
} > "$NQ/MORNING-SUMMARY.txt" 2>&1
cp "$NQ/MORNING-SUMMARY.txt" $D/ 2>/dev/null
log "NIGHTQUEUE COMPLETE"
