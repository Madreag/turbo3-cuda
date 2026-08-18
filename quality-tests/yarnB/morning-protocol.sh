#!/bin/bash
# MORNING PROTOCOL 2026-08-18 (FINAL) — execute after reboot, ONE phase at a
# time, babysat. Root cause identified overnight: GDN row-per-warp kernel
# violated the PDL/__restrict__ race rule (#24030 class). Binary D carries the
# fix; PDL=0 is layered in all launchers regardless.
set -u
Q=/home/erol/ai/turboquant/turboquant-g1/quality-tests
BG=/home/erol/ai/turboquant/turboquant-kv-cache/build-g1
IMX_D=$BG/llama-imatrix.restrictfix
MODEL=/home/erol/ai/turboquant/models/qwen38/Qwen3.8-27B-Q6_K.gguf
PHASE="${1:-1}"
run_killer(){ # $1=label $2=imatrix-binary $3=pdl(0/1)
    echo "--- killer run [$1] PDL=$3 start $(date +%H:%M:%S)"
    GGML_CUDA_PDL=$3 timeout 5400 "$2" -m $MODEL -f $Q/quant-lab/calib_v2.txt \
      -o /tmp/imx-$1.dat -ngl 99 -c 4096 --chunks 1200 2>&1 | tail -2
    nvidia-smi --query-gpu=memory.used --format=csv,noheader >/dev/null 2>&1 \
      || { echo "!!! GPU LOST during [$1]"; return 2; }
    echo "--- [$1] SURVIVED $(date +%H:%M:%S)"
}
case "$PHASE" in
  1) echo "PHASE 1: binary D (restrict-fixed) + PDL=0 — the shipping config, killer x2"
     run_killer d-pdl0-a "$IMX_D" 0 || exit 2
     run_killer d-pdl0-b "$IMX_D" 0 || exit 2
     echo "PHASE 1 PASS -> run phase 3 (promote + battery soak). Optionally phase 1b for science."
     ;;
  1b) echo "PHASE 1b (science): binary D + PDL=1 — does the restrict fix alone hold?"
     run_killer d-pdl1 "$IMX_D" 1 || { echo "D+PDL1 died -> PDL stays off permanently; fix insufficient alone"; exit 2; }
     echo "D+PDL1 survived -> restrict fix alone suffices; PDL re-enable is a future option"
     ;;
  2) echo "PHASE 2 (only if phase 1 died): binary C mainline-GDN kernel"
     IMX_C=/home/erol/ai/turboquant/turboquant-sync/build/bin/llama-imatrix  # rebuild on debug/gdn-mainline first!
     run_killer c-test "$IMX_C" 1 || { echo "C died too -> driver/hw track (see report fault tree)"; exit 2; }
     ;;
  3) echo "PHASE 3: promote binary D to prod + relaunch omega battery as soak"
     echo "(promote already done 04:45 — binary IS prod)" #cp $BG/bin/llama-server $BG/bin/llama-server.pre-restrictfix
     : #cp $BG/bin/llama-server.restrictfix $BG/bin/llama-server
     echo "promoted (rollback: .pre-restrictfix). Relaunching omega (resumable ledger)..."
     setsid nohup bash $Q/yarnB/omega.sh < /dev/null > $Q/yarnB/omega-nohup2.out 2>&1 &
     sleep 3; pgrep -f "[o]mega.sh" >/dev/null && echo "omega running (battery+gates+DRY on fixed stack)"
     ;;
esac
# --- appended (goal v2 night): phases 0 and 4 ---
# Usage additions: "0" = op-test gate FIRST (run before phase 1);
#                  "4" = chunked-prefill validation (after phase 3 stability)
case "$PHASE" in
  0)
    echo "PHASE 0: op-test gate on prod binary (includes nothing new; sanity)"
    timeout 2400 /home/erol/ai/turboquant/turboquant-sync/build/bin/test-backend-ops 2>&1 | tail -3
    ;;
  4)
    echo "PHASE 4: chunked-prefill validation (feature/gdn-chunked-prefill)"
    echo "  a) op-tests incl. chunked suite:"
    timeout 3600 /home/erol/ai/turboquant/turboquant-kv-cache/build-g1/test-backend-ops.chunked -o GATED_DELTA_NET 2>&1 | tail -3
    echo "  b) if PASS: prefill A/B — launch .chunked binary via gate_server.sh, run"
    echo "     nmax probe prefill columns vs same-day prod numbers at 38K/121K depth."
    echo "  c) promote decision per ledger gates (KLD not needed: same math, new schedule)."
    ;;
esac
