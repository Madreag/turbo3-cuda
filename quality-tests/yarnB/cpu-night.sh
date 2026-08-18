#!/bin/bash
# CPU salvage night: wait for convert -> build v3 quants (imatrix A/B) -> clean.
set -u
Q=/home/erol/ai/turboquant/turboquant-g1/quality-tests
YB=$Q/yarnB
CUSTOM=/home/erol/ai/turboquant/models/qwen38-custom
log(){ echo "$(date +%H:%M:%S) $*" >> "$YB/orchestrator.log"; }
log "cpu-night: waiting for convert"
while pgrep -f "[c]onvert_hf_to_gguf" >/dev/null; do sleep 60; done
[ -f "$CUSTOM/bf16-work.gguf" ] || { log "cpu-night: NO bf16-work after convert — abort"; exit 1; }
SZ=$(stat -c %s "$CUSTOM/bf16-work.gguf")
[ "$SZ" -lt 50000000000 ] && { log "cpu-night: bf16-work too small ($SZ) — convert failed"; exit 1; }
QT=/home/erol/ai/turboquant/turboquant-sync/build/bin/llama-quantize
T="--tensor-type ffn_gate=q5_k --tensor-type ffn_up=q5_k --tensor-type ffn_down=q5_k"
log "cpu-night: quantize v3-imx2 (session-calibrated)"
$QT --imatrix $Q/quant-lab/imatrix_v2.dat $T $CUSTOM/bf16-work.gguf $CUSTOM/tqmix-v3-imx2.gguf Q6_K 12 > "$YB/quant-v3imx2.out" 2>&1 && log "v3-imx2 built" || log "v3-imx2 FAILED"
log "cpu-night: quantize v3-imx1 (synthetic-calibrated)"
$QT --imatrix $Q/quant-lab/imatrix_v1.dat $T $CUSTOM/bf16-work.gguf $CUSTOM/tqmix-v3-imx1.gguf Q6_K 12 > "$YB/quant-v3imx1.out" 2>&1 && log "v3-imx1 built" || log "v3-imx1 FAILED"
rm -f "$CUSTOM/bf16-work.gguf"
log "cpu-night: work file deleted; quants ready for morning gates"
echo "DONE quants_built" >> "$YB/ledger.txt"
ls -la --block-size=G $CUSTOM/*.gguf >> "$YB/orchestrator.log" 2>/dev/null
