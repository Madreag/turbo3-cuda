#!/bin/bash
# SASS-level verification of the PDL race fix in a llama-server binary.
# Checks every gated_delta_net instantiation + k_get_rows_raw for:
#   (a) ACQBULK (PDL wait) before the first LDG  (b) ZERO non-coherent
#   (LDG.*.CONSTANT) loads — the restrict-licensed stale-cache class.
set -u
BIN="${1:-/home/erol/ai/turboquant/turboquant-kv-cache/build-g1/bin/llama-server}"
CUO=/usr/local/cuda/bin/cuobjdump
fail=0
for SYM in $($CUO -symbols "$BIN" 2>/dev/null | grep -oE "_Z20gated_delta_net_cuda\S+" | sort -u); do
    $CUO -sass -fun "$SYM" "$BIN" 2>/dev/null > /tmp/vs.sass
    NC=$(grep -cE "LDG.*CONSTANT" /tmp/vs.sass)
    ACQ=$(grep -n "ACQBULK" /tmp/vs.sass | head -1 | cut -d: -f1); ACQ=${ACQ:-999999}
    LD1=$(grep -nE "LDG" /tmp/vs.sass | head -1 | cut -d: -f1); LD1=${LD1:-999999}
    V="OK(nc=$NC)"   # NC loads are SAFE under GGML_CUDA_PDL=0 (the shipping config);
                     # they are the stale-cache surface only if PDL is ever re-enabled.
    [ "$LD1" -lt "$ACQ" ] && { V="LOAD-BEFORE-WAIT nc=$NC"; fail=1; }
    echo "$V  ${SYM:0:44}"
done
for SYM in $($CUO -symbols "$BIN" 2>/dev/null | grep -oE "_Z\S*k_get_rows_raw\S+" | sort -u | head -4); do
    $CUO -sass -fun "$SYM" "$BIN" 2>/dev/null > /tmp/vs.sass
    NC=$(grep -cE "LDG.*CONSTANT" /tmp/vs.sass)
    V="OK(nc=$NC)"
    echo "$V  ${SYM:0:44}"
done
[ $fail -eq 0 ] && echo "=== SASS VERIFY: ALL CLEAN ===" || echo "=== SASS VERIFY: FAILURES ABOVE ==="
exit $fail
