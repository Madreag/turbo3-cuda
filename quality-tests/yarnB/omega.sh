#!/bin/bash
# OMEGA overnight 2026-08-17→18 (stock clocks): yarn-B' verdict + soak +
# quant-lab v3 (imatrix A/B science) + DRY anti-spiral arm + conditional
# unified-scale gate. Resumable via ledger.txt (skips DONE lines). Detached.
set -u
Q=/home/erol/ai/turboquant/turboquant-g1/quality-tests
YB=$Q/yarnB
CONF=/home/erol/.config/llama-tcq
BIN=/home/erol/ai/turboquant/turboquant-kv-cache/build-g1/bin/llama-server
PROD=/home/erol/ai/turboquant/models/qwen38/Qwen3.8-27B-Q6_K.gguf
CUSTOM=/home/erol/ai/turboquant/models/qwen38-custom
LEDGER=$YB/ledger.txt
mkdir -p "$YB" "$CUSTOM"
touch "$LEDGER"
log(){ echo "$(date +%H:%M:%S) $*" >> "$YB/orchestrator.log"; }
mark(){ echo "$1" >> "$LEDGER"; }
done_has(){ grep -q "^DONE $1$" "$LEDGER"; }

stop_all(){ bash $CONF/stop.sh >/dev/null 2>&1; local P=$(ss -tlnp 2>/dev/null | grep -E ':(8131|8233) ' | grep -oP 'pid=\K[0-9]+' | head -1); [ -n "${P:-}" ] && kill $P 2>/dev/null; sleep 3; }

launch(){ # $1=scale(none|125|156) $2=model $3=extra flags
    local ROPE=""
    case "$1" in
        125) ROPE="--rope-scaling yarn --rope-scale 1.25 --yarn-orig-ctx 262144";;
        156) ROPE="--rope-scaling yarn --rope-scale 1.5625 --yarn-orig-ctx 262144";;
    esac
    export TURBO_NORM_ALPHA_V=1.00 TURBO4_NORM_ALPHA_V=1.00 GGML_TURBO_MMA_FUSED=1 && export GGML_CUDA_PDL=0
    setsid nohup $BIN -m "$2" \
      --spec-type draft-mtp --spec-draft-n-max 3 \
      -ctkd turbo4 -ctvd turbo4 -ctk turbo4 -ctv turbo4 \
      -fa on -ngl 99 -c 262144 --no-context-shift $ROPE $3 \
      --jinja --reasoning-format none --parallel 1 -b 512 -ub 512 \
      --temp 1.0 --top-p 0.95 --top-k 20 \
      --host 127.0.0.1 --port 8131 < /dev/null > "$YB/server-cur.log" 2>&1 &
    sleep 2
    local SP=$(ss -tlnp 2>/dev/null | grep ':8131 ' | grep -oP 'pid=\K[0-9]+' | head -1)
    [ -z "$SP" ] && SP=$(pgrep -xn llama-server)
    echo "${SP:-0}" > "$YB/server.pid"
    for i in $(seq 1 160); do
        c=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 http://127.0.0.1:8131/health 2>/dev/null)
        [ "$c" = "200" ] && return 0
        kill -0 $(cat "$YB/server.pid") 2>/dev/null || { log "SERVER DIED launch $1 $(basename $2)"; mark "SOAKEVENT launch_$1_$(basename $2)"; return 1; }
        sleep 3
    done
    log "SERVER TIMEOUT $1"; return 1
}

run_arm(){ # $1=scale $2=seed $3=depths $4=suffix
    local LABEL="yarnB_$1_s$2$4"
    done_has "$LABEL" && { log "skip $LABEL"; return 0; }
    log "arm $LABEL start"
    timeout 4800 python3 $Q/trajectory_battery.py --port 8131 --depths "$3" \
        --temp 1.0 --seed "$2" --label "$LABEL" > "$YB/$LABEL.out" 2>&1
    local RC=$?
    if [ $RC -eq 0 ]; then mark "DONE $LABEL"; log "arm $LABEL done";
    elif [ $RC -eq 124 ]; then mark "TIMEOUT $LABEL"; log "arm $LABEL TIMEOUT";
    else mark "FAIL $LABEL rc=$RC"; log "arm $LABEL FAIL rc=$RC";
         kill -0 $(cat "$YB/server.pid") 2>/dev/null || { log "SERVER DEAD at $LABEL — SOAK EVENT"; mark "SOAKEVENT $LABEL"; return 1; }
    fi
    return 0
}

gate_quant(){ # $1=model-file $2=label
    done_has "gate_$2" && { log "skip gate_$2"; return 0; }
    [ -f "$1" ] || { log "gate_$2: model missing, deferring"; return 0; }
    stop_all
    launch 125 "$1" "" || { mark "FAIL gate_$2 launch"; return 0; }
    log "gate_$2: KLD"
    (cd $Q && timeout 2400 python3 kl_divergence.py --port 8131 --type "$2" \
        --n-prompts 157 --prompt-tokens 2048 --output-dir kldnvfp4 > "$YB/gate-$2-kld.out" 2>&1)
    local M=$(python3 -c "import json;d=json.load(open('$Q/kldnvfp4/kld_summary_$2.json'));print(f\"mean={d.get('kld_mean',d.get('mean','?'))} top1={d.get('top1_agree',d.get('top1','?'))}\")" 2>/dev/null || echo "parse-fail")
    log "gate_$2: speed probe"
    IGNORE_EOS=1 PORT=8131 timeout 900 python3 $Q/nmax/nmax_probe.py "spd-$2" > "$YB/gate-$2-speed.out" 2>&1
    mark "DONE gate_$2 $M"
    log "gate_$2 done: $M"
    stop_all
}

log "=== OMEGA START (stock clocks) ==="

# 0: imatrix v2
if ! done_has imatrix_v2; then
    stop_all
    log "imatrix v2 (1200 chunks of calib_v2)"
    timeout 4200 /home/erol/ai/turboquant/turboquant-sync/build/bin/llama-imatrix \
        -m $PROD -f $Q/quant-lab/calib_v2.txt -o $Q/quant-lab/imatrix_v2.dat \
        -ngl 99 -c 4096 --chunks 1200 > "$YB/imatrix_v2.out" 2>&1 \
        && mark "DONE imatrix_v2" && log "imatrix v2 done" || log "imatrix v2 FAIL/TIMEOUT"
fi

# 0.5: async CPU quant build (v3 recipe x two imatrices) — runs during battery
if ! done_has quants_built; then
    ( set -u
      log "quant-build: convert from D: source"
      /home/erol/ai/turboquant/venv-convert/bin/python \
        /home/erol/ai/turboquant/turboquant-sync/convert_hf_to_gguf.py \
        /mnt/d/spill/qwen38-bf16-src --outfile $CUSTOM/bf16-work.gguf --outtype bf16 \
        > "$YB/convert.out" 2>&1 || { log "convert FAILED"; exit 1; }
      QT=/home/erol/ai/turboquant/turboquant-sync/build/bin/llama-quantize
      T="--tensor-type ffn_gate=q5_k --tensor-type ffn_up=q5_k --tensor-type ffn_down=q5_k"
      log "quant-build: v3-imx2"
      $QT --imatrix $Q/quant-lab/imatrix_v2.dat $T $CUSTOM/bf16-work.gguf $CUSTOM/tqmix-v3-imx2.gguf Q6_K 10 > "$YB/quant-v3imx2.out" 2>&1
      log "quant-build: v3-imx1"
      $QT --imatrix $Q/quant-lab/imatrix_v1.dat $T $CUSTOM/bf16-work.gguf $CUSTOM/tqmix-v3-imx1.gguf Q6_K 10 > "$YB/quant-v3imx1.out" 2>&1
      rm -f $CUSTOM/bf16-work.gguf
      mark "DONE quants_built"
      log "quant-build complete, work file deleted"
    ) &
fi

# B' block 1: native rope
stop_all; launch none $PROD "" && for S in 42 43 44; do run_arm none $S "64000,128000" "" || break; done

# gate slot A (needs quants_built; harmless skip if not ready)
gate_quant $CUSTOM/tqmix-v3-imx2.gguf v3imx2

# B' block 2: yarn 1.25
stop_all; launch 125 $PROD "" && for S in 42 43 44; do run_arm 125 $S "64000,128000" "" || break; done

# gate slot B (imatrix A/B second arm)
gate_quant $CUSTOM/tqmix-v3-imx1.gguf v3imx1

# B' block 3: yarn 1.5625
stop_all; launch 156 $PROD "" && for S in 42 43 44; do run_arm 156 $S "64000,128000" "" || break; done

# late gate slots in case quant build finished late
gate_quant $CUSTOM/tqmix-v3-imx2.gguf v3imx2
gate_quant $CUSTOM/tqmix-v3-imx1.gguf v3imx1

# finalists @250K depth
stop_all; launch 125 $PROD "" && run_arm 125 42 "250000" "_final"
stop_all; launch 156 $PROD "" && run_arm 156 42 "250000" "_final"

# DRY anti-spiral arm: prod scale, 250K, dry 0.8 (vs the 125 finalist as control)
stop_all; launch 125 $PROD "--dry-multiplier 0.8 --dry-allowed-length 2" \
    && run_arm 125 42 "250000" "_dry08"

stop_all
log "=== OMEGA COMPLETE — restoring serving ==="
bash $CONF/start-long-38.sh > "$YB/serving-restore.out" 2>&1 \
    && log "serving restored" || log "SERVING RESTORE FAILED"
mark "COMPLETE $(date +%H:%M:%S)"
