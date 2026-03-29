---
description:
alwaysApply: true
---

# CLAUDE.md — TurboQuant CUDA (Madreag/turbo3-cuda)

## Who You Are Working For

Erol Germain (@erolgermain, GitHub: Madreag). Manufacturing Engineer. Direct communicator. Does NOT tolerate:
- Skipping items
- Deferring work to "later" or "next session"
- Stopping to ask "want me to continue?"
- Implementing things halfway
- Moving to the next task before the current one is PROVEN with measurements
- Compromises, fallbacks, or "good enough"

When Erol says "figure it out" — that means investigate, debug, try multiple approaches, and solve the problem yourself. Do not give up. Do not suggest alternatives that avoid the hard work.

## Project Overview

CUDA implementation of [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026) KV cache compression for llama.cpp, targeting NVIDIA RTX 5090 (SM120 Blackwell). Goal: be the **fastest and most complete** TurboQuant CUDA implementation for Blackwell GPUs.

### Repository Layout

- **NEW Repo (active work)**: `/home/erol/ai/turboquant/turboquant-kv-cache/`
  - Forked from TheTom/llama-cpp-turboquant, branch `feature/turboquant-kv-cache`
  - Includes signalnine's CUDA port (94 tok/s decode, parallel SET_ROWS, MMA prefill)
  - GitHub: `Madreag/turbo3-cuda` (default branch: `feature/turboquant-kv-cache`)
  - **This is where all new work goes.**

- **OLD Repo (reference/archive)**: `/home/erol/ai/turboquant/research/llama-cpp-turboquant/`
  - Sessions 1-15 shadow cache architecture
  - Contains turbo1.5, turbo4 full CUDA, attention sinks, LA modes 3-11 (to port)
  - Contains `.trash/research/` with 45 cloned competitor repos and intel reports
  - **Do NOT develop here. Reference only.**

- **Dense model**: `/home/erol/ai/turboquant/models/opus-v2-Q6_K.gguf` (Qwen 3.5 27B, ~21 GB)
- **MoE model**: `/home/erol/ai/turboquant/models/Qwen3.5-35B-A3B-Q4_K_M.gguf` (~21 GB)
- **Hardware**: RTX 5090 32GB GDDR7, SM120, CUDA 12.8, WSL2 Ubuntu 24.04

## ⚠️ ARCHITECTURE — Pre-Rotate-Queries (NOT Shadow Cache)

The new repo uses **signalnine's architecture**: pre-rotate-queries with native turbo dequant in FA kernels. There is NO shadow cache. This architecture achieves 94 tok/s (98.5% of f16) on RTX 5090.

### How It Works

```
Encode (SET_ROWS):
  Token → parallel 128-thread kernel per WHT group
    → Warp __shfl_xor L2 norm reduce
    → Shared memory WHT butterfly (7 stages)
    → Quantize per thread
    → Pack qs (__shfl_sync 4-wide gather)
    → Pack signs (__ballot_sync)
    → Norm correction
    → Write turbo3 blocks to KV cache

Decode (Q->ne[1]==1):
  Q → GGML_OP_TURBO_WHT (forward rotation, graph-level)
  FA VEC kernel reads turbo3 blocks DIRECTLY:
    → vec_dot_KQ: centroid[idx] * norm
    → dequantize_V: ne==4 fast path
    → Sparse V skip: weight < 1e-6 → skip V dequant
  Output → GGML_OP_TURBO_WHT (inverse rotation)

Prefill (Q->ne[1]>1):
  launch_fattn auto-dequants turbo3 → fp16 (built-in need_f16_K/V path)
  MMA/TILE kernel runs at full speed on fp16
```

### Why NOT Shadow Cache

The old repo (sessions 1-15) used a persistent fp16 shadow cache. signalnine proved this is strictly inferior:
- **Shadow reads 8x MORE data** at long context (fp16 = 2 bytes/element vs turbo3 = 0.5 bytes/element)
- **Shadow adds ~8 GB VRAM** overhead
- **Shadow adds per-token sync latency**
- **Shadow prevents direct MMA usage** (forces VEC-only for native turbo types)
- **signalnine's native dequant** with parallel SET_ROWS: 94 tok/s vs our shadow's 54 tok/s

The shadow cache dead ends from sessions 1-11 are documented in `.trash/resume.md` in the OLD repo for historical reference. Do NOT reintroduce shadow.

## Current Performance (signalnine baseline on this repo)

### Dense Model (Qwen 3.5 27B Q6_K, RTX 5090)
| Type | bpv | Short | Notes |
|------|----:|------:|-------|
| f16 | 16 | ~95.5 | Theoretical ceiling |
| q8_0 | 8.5 | ~95.4 | Baseline |
| turbo3 | 3.5 | **~94.0** | 98.5% of f16 (signalnine) |
| turbo2 | 2.5 | TBD | Available, needs benchmarking |

**Prefill**: 0.97x q8_0 at all context lengths (MMA/TILE kernel).
**NIAH**: 11/11 through 256K and 1M context.
**Compression**: 3.47x for turbo3.

### What This Repo Does NOT Have Yet (Port From Old Repo)
- **turbo1.5** (2.0 bpv, 8x compression) — our unique type, nobody else has it
- **turbo4 full CUDA** (4.25 bpv, 16 centroids) — type registered but no CUDA kernels
- **Attention sinks** (TURBO_SINK_SIZE=4) — PPL -1.9% at ctx=2048, zero speed cost
- **Layer-adaptive modes 3-11** — they have 1-2, we need 3-11
- **LUT attention kernel** — novel, zero multiplies in inner loop (not in any repo yet)

## ABSOLUTE RULES

### 1. NEVER skip an item. NEVER defer.
If the plan says to do something, do it. If it's blocked, find another way.

### 2. MEASURE before and after EVERY change.
No commit without benchmark data. PPL at ctx=512 AND ctx=2048 for every code change.

### 3. PPL REJECT THRESHOLDS — non-negotiable.
- ctx=512: turbo3 PPL > 6.89 → **REJECT and revert immediately**
- ctx=2048: turbo3 PPL > 5.80 → **REJECT and revert immediately**

### 4. ONE commit per logical change.
Implement → rebuild → measure → commit → next item.

### 5. Fix regressions IMMEDIATELY.
If any metric gets worse, stop and fix before moving on.

### 6. PROFILE before optimizing.
Use nsys/ncu to find where GPU time is spent. Optimize the bottleneck, not assumptions.

### 7. NEVER ask "want me to continue?"
The answer is always yes.

### 8. Understand the ARCHITECTURE before writing code.
Read signalnine's parallel SET_ROWS kernel in `set-rows.cu` and the FA dispatch in `fattn.cu` before modifying them.

## Build & Test Commands

```bash
# Build (from the NEW repo)
cd /home/erol/ai/turboquant/turboquant-kv-cache
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j$(nproc)

# If cmake not found at that path:
/home/erol/miniconda3/envs/tq/bin/cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
/home/erol/miniconda3/envs/tq/bin/cmake --build build -j$(nproc)

# ALWAYS use -mmp 0 to disable mmap (WSL2 page cache causes GPU stalls).
MODEL=/home/erol/ai/turboquant/models/opus-v2-Q6_K.gguf
WIKI=wikitext-2-raw/wiki.test.raw
# Wiki file may be elsewhere. Search: find /home/erol/ai/turboquant -name "wiki.test.raw" 2>/dev/null

# PPL gate (run for EVERY code change):
./build/bin/llama-perplexity -m $MODEL -f $WIKI -c 512 -ctk turbo3 -ctv turbo3 -fa on --chunks 8 -ngl 99 -mmp 0
./build/bin/llama-perplexity -m $MODEL -f $WIKI -c 2048 -ctk turbo3 -ctv turbo3 -fa on --chunks 8 -ngl 99 -mmp 0

# Decode curve (include 204K — signalnine claims 9.7 t/s there):
for DEPTH in 0 8192 32768 65536 131072 204800; do
  ./build/bin/llama-bench -m $MODEL -fa 1 -ctk turbo3 -ctv turbo3 -d $DEPTH -ngl 99 -t 1 -r 3 -p 0 -n 128 -mmp 0
done

# Prefill:
for PP in 512 2048 4096 8192; do
  ./build/bin/llama-bench -m $MODEL -fa 1 -ctk turbo3 -ctv turbo3 -d 0 -ngl 99 -t 1 -r 3 -p $PP -n 0 -mmp 0
done
```

## Architecture Knowledge

### TurboQuant turbo3 (3.25 bpv, 4.6x compression) — FULLY WORKING
- Block: 14 bytes per 32 values — norm(fp16) + qs[8] (2-bit indices) + signs[4] (1-bit upper)
- WHT rotation group = 128 values = 4 blocks of 32
- Lloyd-Max 8 centroids for N(0, 1/sqrt(128))
- Norm correction: `corrected_norm = raw_norm / ||reconstruction||`
- Parallel SET_ROWS: 128 threads/group, shared memory WHT, warp intrinsics
- Native vec_dot in FA VEC kernel + MMA prefill via launch_fattn pre-dequant
- Sparse V dequant: skip positions with attention weight < 1e-6

### TurboQuant turbo2 (2.5 bpv, 6.4x compression) — WORKING
- 2-bit PolarQuant, 4 centroids, same FWHT rotation as turbo3
- Block: 10 bytes per 32 values
- CUDA VEC kernel + Metal support

### TurboQuant turbo4 (4.25 bpv, 3.8x compression) — CPU ONLY, NEEDS CUDA
- Type registered, CPU ref exists, NO CUDA kernels in this repo
- 4-bit MSE with 16 Lloyd-Max centroids (QJL removed — hurts PPL)
- Nibble-packed: 2 indices per byte
- Our old repo has full CUDA: SET_ROWS, vec_dot, V dequant (port needed)
- TheTom's data: PPL +0.23% vs q8_0, NIAH 31/33 beats q8_0's 30/33

### TurboQuant turbo1.5 (2.0 bpv, 8x compression) — NOT IN THIS REPO
- Ternary quantization: {-C, 0, +C} where C = 0.107632
- Block: 16 bytes per 32 values (norm + 7 trit bytes + 7 pad)
- Our old repo has full implementation (type registration + CUDA kernels)
- Needs native vec_dot (ternary is simpler than 8-centroid — just sign * C * norm)
- **Nobody else has this type.** Our unique differentiator.

### SM120 (RTX 5090) Constraints
- 128 KB shared memory/SM, 99 KB max per block
- 48 max warps/SM (NOT 64 like SM100 datacenter)
- NO WGMMA, NO TMEM — only extended `mma.sync`
- 98 MB L2 cache, 1.79 TB/s GDDR7 bandwidth
- FlashAttention-3/4 does NOT work on SM120
- HAS native FP4 E2M1 Tensor Core support via `mma.sync.m16n8k64`
- CUDA 12.8 only (13.1 segfaults on MMQ kernels)
- **WSL2**: Always use `-mmp 0` to disable mmap.

### Key Files (This Repo)
```
ggml/src/ggml-cuda/set-rows.cu       — Parallel SET_ROWS: turbo3 (128 threads/group, warp intrinsics), turbo2
ggml/src/ggml-cuda/turbo-quant.cuh   — Turbo3 centroid constants, helper functions, sign arrays
ggml/src/ggml-cuda/turbo-wht.cu      — GGML_OP_TURBO_WHT CUDA kernel (forward + inverse)
ggml/src/ggml-cuda/turbo-wht.cuh     — WHT op declarations
ggml/src/ggml-cuda/turbo-innerq.cu   — InnerQ per-channel equalization
ggml/src/ggml-cuda/turbo-innerq.cuh  — InnerQ declarations
ggml/src/ggml-cuda/fattn-common.cuh  — FA vec_dot for turbo3, V dequant (ne==4 fast path), sparse V
ggml/src/ggml-cuda/fattn-vec.cuh     — VEC FA kernel with turbo3 support
ggml/src/ggml-cuda/fattn.cu          — FA dispatch (NO shadow cache, direct native dequant)
ggml/src/ggml-turbo-quant.c          — CPU reference quantize/dequant (turbo2/3/4)
ggml/src/ggml-common.h               — Block structs for turbo2/3/4
src/llama-kv-cache.cpp               — Layer-adaptive modes 1-2, turbo type checks
src/llama-graph.cpp                   — Graph-level WHT rotation via GGML_OP_TURBO_WHT
src/turbo-rotation-data.h            — 4103-line precomputed 128x128 rotation matrices
common/arg.cpp                        — CLI arg parsing for turbo2/turbo3/turbo4
tests/test-turbo-quant.c             — Turbo3 roundtrip correctness tests
scripts/turbo-quality-gate.sh        — CI quality + speed gate
```

### Key Files (Old Repo — Reference for Porting)
```
$OLD/ggml/src/ggml-cuda/turbo-quant.cu    — turbo1.5 SET_ROWS, turbo4 SET_ROWS (serial — rewrite as parallel)
$OLD/ggml/src/ggml-cuda/fattn-common.cuh  — turbo4 vec_dot, turbo4 V dequant
$OLD/ggml/src/ggml-cuda/fattn.cu          — Attention sinks (TURBO_SINK_SIZE), multi-seq temp dequant
$OLD/ggml/src/ggml-cuda/dequantize.cuh    — turbo1.5 trit unpack, turbo4 nibble extract
$OLD/ggml/src/ggml-common.h               — block_turbo1_5 struct
$OLD/src/llama-kv-cache.cpp               — Layer-adaptive modes 3-11
```

## Common Mistakes To Avoid

### ❌ Reintroducing shadow cache
Shadow reads 8x more data than native turbo3 at long context. signalnine proved native is 1.75x faster. Do NOT add shadow back.

### ❌ Porting our old serial SET_ROWS
Our old SET_ROWS used 1 thread per 128-element group (21% of decode budget). signalnine's parallel kernel uses 128 threads with warp intrinsics. All new turbo types (turbo4, turbo1.5) MUST use the parallel pattern.

### ❌ Blocking MMA for turbo types
signalnine enabled MMA/TILE for turbo prefill via launch_fattn's built-in pre-dequant. Do NOT add VEC-only restrictions for turbo types.

### ❌ Batching multiple changes without measuring between them
One change → one measurement → one commit.

### ❌ Skipping PPL validation
Every code change needs PPL at ctx=512 AND ctx=2048. If it exceeds reject thresholds → revert.

### ❌ Optimizing without profiling
Use nsys/ncu FIRST to find where time is spent. Then optimize the actual bottleneck.

### ❌ Using cudaEventSynchronize inside graph compute
CUDA graphs are enabled (USE_GRAPHS=1). Sync inside graph replay = crash.

### ❌ Using cudaMallocAsync for persistent buffers
Causes NaN on CUDA graph replay. Use persistent cudaMalloc instead.

## Commit Message Format

```
<type>: <short description>

<Detailed explanation of what changed and why>

<Benchmark results — REQUIRED>
  PPL impact: turbo3=X.XXX, q8_0=X.XXX, delta=+X.XX% (ctx=512)
  Decode: XX.XX tok/s (ratio vs q8_0)
```

Types: `feat`, `fix`, `perf`, `docs`, `test`, `refactor`

**NEVER add Co-Authored-By lines.** Credit signalnine/spiritbuun/TheTom in the commit message body text if adapted from their code.

## Environment Variables

| Variable | Effect |
|----------|--------|
| `TURBO_LAYER_ADAPTIVE=N` | Per-layer KV type: 0=uniform, 1=first4+last4 K+V→q8_0, 2=last8 |
| `TURBO_INNERQ=N` | InnerQ calibration: collect N tokens of per-channel K stats, then equalize |
| `GGML_TURBO_DECODE_NATIVE=1` | Force native turbo dequant in VEC (default behavior, for debug) |

### Not Yet Implemented (Port From Old Repo)
| Variable | Effect |
|----------|--------|
| `TURBO_SINK_SIZE=N` | Attention sinks: first N positions bypass turbo quantization (PPL -1.9%) |
| `TURBO_LAYER_ADAPTIVE=3-11` | Extended LA modes (we had 1-11, this repo has 1-2) |

## Competitive Landscape

### signalnine (Gabe Ortiz, NVIDIA) — MERGED INTO THIS TREE
- 14 commits, +2200 lines of CUDA
- Parallel SET_ROWS (128 threads, warp intrinsics) — the +31% decode commit
- MMA/TILE prefill for turbo3
- Q/K stride mismatch bugfix
- 94 tok/s decode (98.5% of f16) on RTX 5090
- NIAH 11/11 through 1M context

### spiritbuun — SEPARATE FORK (RTX 3090)
- Feature parity but different architecture (some shadow-like patterns)
- InnerQ equalization (also in this tree via signalnine)
- Inverse-FWHT prefill dequant (not needed — MMA prefill already handles this)

### TheTom — UPSTREAM (Metal + this tree)
- turbo4 resurrection (PPL 679→6.125)
- Sparse V dequant (+22.8% at 32K)
- Cross-model validation (7 models)
- 4-tier test suite
- turbo2 Metal support

### Competitor research repos (45 cloned in old repo's .trash/research/repos/)
- See `.trash/research/megaintelreport.md` for full ecosystem analysis
- See `.trash/research/signalnine-intel.md` for the CUDA kernel deep dive

## What To Build Next (Our Differentiators)

1. **Attention sinks** — Port TURBO_SINK_SIZE from old repo. PPL -1.9% at ctx=2048 for free.
2. **turbo4 CUDA** — Parallel SET_ROWS (16 centroids, nibble packing), native vec_dot, V dequant.
3. **turbo1.5 CUDA** — Parallel SET_ROWS (ternary, trit packing), native vec_dot (simpler than turbo3).
4. **LA modes 3-11** — Port from old repo's llama-kv-cache.cpp.
5. **LUT attention kernel** — Novel: precompute q_rot×centroid table in shared memory. Zero multiplies in inner loop. Nobody has this on CUDA.
6. **FP4 tensor core for turbo1.5** — SM120 moonshot: map ternary to native FP4 E2M1 mma.sync.

## Remember

- **This is signalnine's code + TheTom's Metal.** Respect their architecture.
- **No shadow cache.** Native turbo dequant everywhere. MMA prefill via pre-dequant.
- **Parallel SET_ROWS for everything.** 128 threads, warp intrinsics, shared memory WHT.
- **turbo1.5 is our unique type.** Nobody else has sub-2.5 bpv. Port it with native vec_dot.
- **Attention sinks beat q8_0 quality.** turbo3 PPL 5.628 vs q8_0 5.674 (-0.81%). Port this first.
- **Profile before optimizing.** nsys/ncu to find actual bottlenecks.
- **Every optimization must be MEASURED.** Intuition is not data.
- **DO NOT STOP. DO NOT DEFER. DO NOT SKIP. FINISH THE WORK.**
