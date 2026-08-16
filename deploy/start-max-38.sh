#!/bin/bash
# MAX-CONTEXT profile (opt-in) — Qwen3.8-27B at 409,600 ctx via YaRN 1.5625,
# MTP OFF (spec-off frees ~1.9 GB measured -> the bigger window; ~28-31 tok/s
# at 274-329K depth). Sibling default: start-long-38.sh (320K speed profile).
# Gates 2026-08-16: fill-ladder flat to 329K cached; KLD@1.5625 0.0052 /
# p99 0.028 / top-1 95.5% vs matched f16 ref; battery @64K 2-of-3 (seed-42
# trajectory spiral documented); NIAH 5/5 @380K pre-validated at this scale.
#
# VALIDATED (2026-08-14/15 battery, session f5334395):
#   - reasoning_effort=xhigh @ temp 1.0 (vendor defaults) wins the effort
#     ladder: 3/4 clean pixel-gated renders vs medium 2/4, low 0/2.
#     xhigh is the template default — deliberately not set here.
#   - NIAH effective 5/5 at 130K AND 380K through turbo4+YaRN.
#   - Alphas 1.00/1.00 (2026-08-15 corrected-methodology sweep): Qwen3.8 wants
#     NO V-norm correction — the 3.6-inherited 1.10/1.12 cost 2.3x KLD.
#   - MTP is OFF in this profile (SPEC_TYPE=none) — that is the point:
#     the freed draft/spec compute (~1.9 GB) funds the 409K window.
#   - Binary: build-g1/bin/llama-server = gdn22587 build (2026-08-16).
#     Rollback chain: .pre-gdn22587 -> .mainline -> .pre-bughunt.
#
# Rollback (absolute paths, run AFTER a verified stop):
#   bash /home/erol/.config/llama-tcq/stop.sh \
#     && cp /home/erol/ai/turboquant/turboquant-kv-cache/build-g1/bin/llama-server.pre-gdn22587 \
#           /home/erol/ai/turboquant/turboquant-kv-cache/build-g1/bin/llama-server \
#     && bash /home/erol/.config/llama-tcq/start-max-38.sh
#
# Slot-state files are CONFIG-SPECIFIC: changing KV types, ctx, or --spec-*
# invalidates saved .bin files (server refuses them, then the proxy erases and
# re-prefills). Archive slots-max/*.bin when changing config.
CONF=/home/erol/.config/llama-tcq
REPO=/home/erol/ai/turboquant/turboquant-kv-cache

# ── Double-start guard (bughunt A5: starting over a live stack used to kill
#    logs+pidfiles and false-green off the OLD server's /health) ──────────────
if ss -tln 2>/dev/null | grep -qE ':(8130|8131)\s'; then
    echo "ERROR: stack already running (8130/8131 bound) — run stop.sh first" >&2
    ss -tlnp 2>/dev/null | grep -E ':(8130|8131)\s' >&2
    exit 1
fi

cd "$REPO" || { echo "ERROR: repo missing at $REPO" >&2; exit 1; }

# ── Log rotation (was `>` truncation — a double-start once destroyed the live
#    proxy's log mid-write; keep history, prune to last 5) ────────────────────
ts=$(date +%Y%m%d-%H%M%S)
for base in server proxy; do
    [ -s "$CONF/$base.log" ] && mv "$CONF/$base.log" "$CONF/$base.log.$ts"
    ls -t "$CONF/$base".log.* 2>/dev/null | tail -n +6 | xargs -r rm -f
done

CTX=409600
SCALE=1.5625      # = 409600 / 262144 (native) — NIAH 5/5 @380K validated at this scale

export TURBO_NORM_ALPHA_V=1.00
export TURBO4_NORM_ALPHA_V=1.00
# Fused MMA-turbo decode path (2026-08-15 sparse-P0 discovery: the in-tree
# gate is opt-in and prod had silently run the VEC/dequant path since deploy).
# Paired A/B (restarts between arms, ordering repeated): decode +2.2% @38K,
# +8.7% @121K (70.3/70.3 -> 76.5/76.3); prefill unchanged; KLD == baseline.
# Not token-identical to VEC (f16 reduction order) — unset/=0 for strict
# VEC-identity A/Bs.
export GGML_TURBO_MMA_FUSED=${GGML_TURBO_MMA_FUSED:-1}
# CUDA crash forensics (ops round 3): on a CUDA abort, write a coredump
# instead of losing the evidence (WSL CaptureCrash dumps land Windows-side).
export CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1
# KNOWN UPSTREAM RISK (#26558): MTP + CUDA-graph reuse can hard-crash
# (cuBLAS status 7) under KV saturation. If that signature appears in
# server.log, relaunch with:  export LLAMA_GRAPH_REUSE_DISABLE=1  (perf cost)

mkdir -p "$CONF/slots-max"

# ── VRAM settle (club-3090 P6): a fresh stop returns before CUDA frees the
#    residency under WSL; booting into held VRAM caused transient OOMs.
#    Wait until used-VRAM stops falling (two stable reads) or 30s. ──────────
prev=-1
for i in $(seq 1 15); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -z "$used" ] && break
    if [ "$used" = "$prev" ] && [ "$used" -lt 4000 ]; then break; fi
    prev=$used
    sleep 2
done
if [ -n "$used" ] && [ "$used" -ge 4000 ]; then
    echo "WARNING: VRAM still ${used} MiB after settle wait — another GPU workload" >&2
    echo "         may be live (probe/foreign process); boot may OOM. Proceeding." >&2
fi

# Spec-decode config is env-able for A/B arms (defaults = production):
#   SPEC_TYPE=ngram-mod,draft-mtp SPEC_EXTRA="--spec-ngram-mod-n-max 3" bash start-long-38.sh
SPEC_TYPE="${SPEC_TYPE:-none}"   # MAX-CTX PROFILE: MTP OFF frees ~1.9 GB MEASURED (draft/spec compute graph) -> funds 409K vs the 320K speed profile
SPEC_NMAX="${SPEC_NMAX:-3}"   # 3 ADOPTED 2026-08-16: code decode +17% (96→113), copy-heavy +22% (115→140), prose -6% — coding-primary trade; 5-seed ledger gate {8,8,8,7-wrongval,0-spiral(traj-luck, base-class mode)}; p-min gate + ngram cascade both measured WORSE on our fused stack (see g1 CLUB3090-PORT-BOARD P9/P1)
SPEC_PMIN="${SPEC_PMIN:-}"   # e.g. 0.60 — confidence gate (arg parser has no --flag=value form)
SPEC_EXTRA="${SPEC_EXTRA:-}"
CKPT="${CKPT:-2}"          # ctx-checkpoints (host RAM, ~2.4GB each at full depth)
ALIAS="${ALIAS:-qwen3.8-27b-409k}"  # model id shown in /v1/models — the Hermes selector key
# Key via --api-key-file: the old --api-key "$KEY" form exposed the key in
# /proc/*/cmdline to any local process (bughunt A5).
# --metrics: exposes spec_decode_* acceptance counters (grammar/MTP split
# measurement) — server-side only; the proxy does NOT forward /metrics beyond
# its allowlist policy.
nohup ./build-g1/bin/llama-server \
  -m /home/erol/ai/turboquant/models/qwen38/Qwen3.8-27B-Q6_K.gguf \
  --mmproj /home/erol/ai/turboquant/models/qwen38/mmproj-F16.gguf \
  --no-mmproj-offload \
  --spec-type $SPEC_TYPE --spec-draft-n-max $SPEC_NMAX ${SPEC_PMIN:+--spec-draft-p-min $SPEC_PMIN} $SPEC_EXTRA \
  -ctkd turbo4 -ctvd turbo4 \
  -ctk turbo4 -ctv turbo4 \
  -fa on -ngl 99 -c $CTX --no-context-shift \
  --rope-scaling yarn --rope-scale $SCALE --yarn-orig-ctx 262144 \
  --jinja --reasoning-format none \
  --chat-template-kwargs '{"preserve_thinking": true}' \
  --parallel 1 -b 512 -ub 512 \
  --temp 1.0 --top-p 0.95 --top-k 20 \
  --ctx-checkpoints $CKPT --alias $ALIAS \
  --slot-save-path "$CONF/slots-max/" \
  --host 127.0.0.1 --port 8131 \
  --api-key-file "$CONF/api.key" \
  --metrics \
  > "$CONF/server.log" 2>&1 &
echo $! > "$CONF/server.pid"

# ── Health wait: cold-cache load of 22.9GB + mmproj at this box's ~124MB/s
#    can exceed 3 min; give 8. A timeout is an ERROR — the old script launched
#    the proxy and printed success regardless (bughunt A5). ──────────────────
code=""
for i in $(seq 1 240); do
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 http://127.0.0.1:8131/health 2>/dev/null)
    [ "$code" = "200" ] && break
    kill -0 "$(cat "$CONF/server.pid")" 2>/dev/null || { code="dead"; break; }
    sleep 2
done
if [ "$code" != "200" ]; then
    echo "ERROR: server not healthy after wait (last: ${code:-none}) — proxy NOT started" >&2
    echo "--- server.log tail:" >&2
    tail -5 "$CONF/server.log" >&2
    [ "$code" = "dead" ] || echo "(process still alive — may still be loading; check status.sh)" >&2
    exit 1
fi
grep -m1 "n_ctx_slot" "$CONF/server.log"

nohup python3 "$CONF/proxy.py" \
  --upstream http://127.0.0.1:8131 \
  --host 0.0.0.0 --port 8130 \
  > "$CONF/proxy.log" 2>&1 &
echo $! > "$CONF/proxy.pid"

for i in $(seq 1 10); do
    pcode=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 http://127.0.0.1:8130/health 2>/dev/null)
    [ "$pcode" = "200" ] && break
    sleep 1
done
if [ "$pcode" != "200" ]; then
    echo "ERROR: proxy did not become healthy — see proxy.log" >&2
    tail -5 "$CONF/proxy.log" >&2
    exit 1
fi

echo "MAX-38 profile: llama-server PID $(cat "$CONF/server.pid") at ctx=$CTX yarn=$SCALE (Qwen3.8-27B, MTP OFF — max-ctx profile)"
echo "proxy PID:       $(cat "$CONF/proxy.pid") (0.0.0.0:8130, health OK)"
