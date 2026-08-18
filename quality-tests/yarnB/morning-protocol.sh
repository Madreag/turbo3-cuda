#!/bin/bash
# MORNING PROTOCOL 2026-08-18 — PDL/GDN suspect isolation (run after reboot).
# Phase 1: GGML_CUDA_PDL=0 + the exact killer workload (imatrix 1200 chunks,
#          died at ~30min/[310] with PDL on). Twice. Survive = PDL convicted.
# Phase 2 (only if Phase 1 dies): binary C (.gdnmainline, PDL ON) same test.
# Verdict matrix in OVERNIGHT-REPORT. Run ONE phase at a time, babysat.
set -u
Q=/home/erol/ai/turboquant/turboquant-g1/quality-tests
IMX=/home/erol/ai/turboquant/turboquant-sync/build/bin/llama-imatrix
PROD_BIN=/home/erol/ai/turboquant/turboquant-kv-cache/build-g1/bin/llama-server
MODEL=/home/erol/ai/turboquant/models/qwen38/Qwen3.8-27B-Q6_K.gguf
PHASE="${1:-1}"
case "$PHASE" in
  1)
    echo "PHASE 1: PDL OFF, killer workload x2"
    for i in 1 2; do
      echo "--- run $i start $(date +%H:%M:%S)"
      GGML_CUDA_PDL=0 timeout 5400 $IMX -m $MODEL -f $Q/quant-lab/calib_v2.txt \
        -o /tmp/imx-probe-$i.dat -ngl 99 -c 4096 --chunks 1200 2>&1 | tail -3
      RC=$?
      nvidia-smi --query-gpu=memory.used --format=csv,noheader || { echo "GPU LOST during run $i (PDL OFF!) — PDL exonerated, GDN/other still suspect"; exit 2; }
      [ $RC -ne 0 ] && echo "run $i rc=$RC (nonzero but GPU alive)"
      echo "--- run $i survived $(date +%H:%M:%S)"
    done
    echo "PHASE 1 PASS: PDL-off survives the killer x2 -> PDL CONVICTED."
    echo "NEXT: add 'export GGML_CUDA_PDL=0' to all launchers, relaunch battery."
    ;;
  2)
    echo "PHASE 2: binary C (mainline GDN kernel, PDL ON), killer workload"
    # imatrix binary is from the same build tree as C after the revert build;
    # rebuild imatrix target on debug/gdn-mainline first if not done.
    timeout 5400 $IMX -m $MODEL -f $Q/quant-lab/calib_v2.txt \
      -o /tmp/imx-probeC.dat -ngl 99 -c 4096 --chunks 1200 2>&1 | tail -3
    nvidia-smi --query-gpu=memory.used --format=csv,noheader || { echo "GPU LOST on binary C too — kernel+PDL both exonerated -> driver/hw track"; exit 2; }
    echo "PHASE 2 survived -> row-per-warp GDN kernel convicted (with PDL as trigger-amplifier)"
    ;;
esac
