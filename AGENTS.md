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

**Current state**: Sessions 17-18 complete. All 4 turbo types (1.5/2/3/4) have full CUDA with parallel SET_ROWS, native FA vec_dot, V dequant. 3 critical cross-GPU bugs found by 3090 Ti overnight testing — fixing in Session 19.

### Repository Layout

- **Active Repo**: `/home/erol/ai/turboquant/turboquant-kv-cache/`
  - Forked from TheTom/llama-cpp-turboquant, branch `feature/turboquant-kv-cache`
  - Includes signalnine's CUDA port + our Sessions 17-18 additions (turbo1.5, turbo4, LUT, trit LUT, LA modes, sinks)
  - GitHub: `Madreag/turbo3-cuda` (default branch: `feature/turboquant-kv-cache`)

- **Archive Repo** (reference only): `/home/erol/ai/turboquant/research/llama-cpp-turboquant/`
  - Sessions 1-15 shadow cache architecture (OBSOLETE — do not use)
  - `.trash/research/` — 45 cloned competitor repos + intel reports
  - `.trash/sessions/` — session prompts 13-19

- **Obsidian Vault** (knowledge base): `/mnt/c/vaults/forge/` (Windows: `C:\vaults\forge`)
  - **SEARCH THIS FIRST** when you need context. 90+ markdown files covering:
    - `03 Benchmarks/Benchmark Hub.md` — definitive performance numbers (Sessions 17-18 final)
    - `03 Benchmarks/3090 Ti Overnight Test.md` — SM86 crash matrix, 340+ stability iterations
    - `02 Architecture/Architecture Overview.md` — pre-rotate-queries design
    - `02 Architecture/Dead Ends.md` — 15+ approaches that FAILED (do NOT repeat)
    - `04 Competitors/` — 14 competitor profiles with code techniques
    - `05 Research/` — K/V norm data, QJL ablation, signalnine comparison correction
    - `07 Issues/` — 3 OPEN critical bugs + resolved bugs with root cause
    - `10 Knowledge/` — CUDA specifics, hardware constraints, quantization theory
    - `01 Sessions/` — all 18 session reports with what worked and what didn't

- **Models**: `/home/erol/ai/turboquant/models/opus-v2-Q6_K.gguf` (27B dense), `Qwen3.5-35B-A3B-Q4_K_M.gguf` (MoE)
- **Hardware**: RTX 5090 32GB (SM120), RTX 3090 Ti 24GB (SM86), RTX 4090M 16GB (SM89)

## ⚠️ ARCHITECTURE — Pre-Rotate-Queries (NOT Shadow Cache)

```
Encode (SET_ROWS):
  Token → parallel 128-thread kernel per WHT group
    → Warp __shfl_xor L2 norm reduce → shared memory WHT butterfly
    → Quantize per thread → Pack qs (__shfl_sync) + signs (__ballot_sync)
    → Norm correction → Write turbo blocks to KV cache

Decode (Q->ne[1]==1):
  Q → GGML_OP_TURBO_WHT (forward rotation, graph-level)
  FA VEC kernel reads turbo blocks DIRECTLY (native dequant)
  Output → GGML_OP_TURBO_WHT (inverse rotation)

Prefill (Q->ne[1]>1):
  launch_fattn auto-dequants turbo→fp16 (built-in need_f16_K/V path)
  MMA/TILE kernel runs on fp16
```

## Current Performance (Session 18 Final, RTX 5090)

| Type | bpv | Short | 32K | PPL ctx=512 | Notes |
|------|----:|------:|----:|:-----------:|-------|
| f16 | 16 | 59.09 | — | — | Ceiling |
| q8_0 | 8.5 | 59.68 | 46.42 | 6.759 | Baseline |
| turbo4 | 4.25 | **59.06** | 45.96 | 6.825 (+0.97%) | **99.9% of f16** |
| turbo2 | 2.5 | **58.88** | 38.49 | 7.080 (+4.75%) | |
| turbo1.5 | 2.0 | **57.36** | 44.57 | 7.312 (+8.18%) | **8x compression, 204K=15.26** |
| turbo3 | 3.25 | 57.99 | 40.90 | 6.852 (+1.38%) | LUT disabled (bank conflicts) |

**MoE (Qwen 3.5 35B-A3B)**: turbo1.5 = 113 tok/s, turbo3 = 93.86.

## 3 OPEN CRITICAL BUGS (from 3090 Ti overnight testing)

| Bug | Impact | Root Cause | Vault Reference |
|-----|--------|------------|-----------------|
| **Sinks crash on SM86** | `TURBO_SINK_SIZE>0` → CUDA error | `__managed__` memory breaks graph capture on SM86 | `07 Issues/Sinks SM86 Crash.md` |
| **Asymmetric K=turbo4/1.5 crash** | SEGFAULT with V={f16,turbo3,turbo2} | Missing FA VEC template instances | `07 Issues/Asymmetric K-turbo4-turbo1.5 Crash.md` |
| **K=turbo3/V=turbo4 = 6 tok/s** | 14x perf regression | CPU fallback from missing template | `07 Issues/turbo3-turbo4 Cross-Type Perf Regression.md` |

Full crash matrix: `03 Benchmarks/3090 Ti Overnight Test.md`

## ABSOLUTE RULES

### 1. NEVER skip an item. NEVER defer.

### 2. MEASURE SPEED AND PPL after EVERY change.
Not just PPL — **decode speed too**. Session 18 caught a -3% speed regression that PPL alone missed. Run:
```bash
# BOTH of these after every change:
./build/bin/llama-bench -m $MODEL -fa 1 -ctk turbo3 -ctv turbo3 -d 0 -ngl 99 -t 1 -r 3 -p 0 -n 128 -mmp 0
./build/bin/llama-perplexity -m $MODEL -f $WIKI -c 512 -ctk turbo3 -ctv turbo3 -fa on --chunks 8 -ngl 99 --no-mmap
```

### 3. PPL REJECT THRESHOLDS
- ctx=512: turbo3 PPL > 6.89 → **REJECT**
- ctx=2048: turbo3 PPL > 5.80 → **REJECT**

### 4. SPEED REJECT THRESHOLDS
- turbo3 short: < 55.0 tok/s → **INVESTIGATE** (baseline is 57.99)
- turbo4 short: < 56.0 tok/s → **INVESTIGATE** (baseline is 59.06)
- turbo1.5 short: < 54.0 tok/s → **INVESTIGATE** (baseline is 57.36)

### 5. READ THE CODE before writing code.
Session 18 failures all came from not reading existing code:
- Sink ne0 mismatch: didn't read `sink_get_or_alloc` before modifying FA dispatch
- turbo4 LUT with q8_1 Q: didn't check `K_is_unquantized` before writing LUT path
- V sink perf regression: added `__managed__` reads to hottest loop without profiling

**Before touching ANY kernel**: Read the function you're modifying AND the functions that call it.

### 6. Check `K_is_unquantized` for type-specific FA paths.
The VEC FA kernel loads Q differently for "unquantized" vs "quantized" K types:
- turbo3, turbo2: `K_is_unquantized = true` → float Q path (LUT compatible)
- turbo4, turbo1.5: `K_is_unquantized = false` → q8_1 Q path (LUT NOT compatible)
Check this BEFORE writing any type-specific FA code.

### 7. Do NOT use `__managed__` memory in kernel-accessible paths.
It breaks CUDA graph capture on SM86 (and possibly other architectures). Use:
- `__constant__` memory with `cudaMemcpyToSymbol` (watch for TU issues)
- Regular device memory with explicit `cudaMalloc` / `cudaMemcpy`
- Kernel arguments passed through FA dispatch

### 8. Template instances MUST exist for EVERY K×V combination.
Missing template = SEGFAULT or CPU fallback. Check `ggml/src/ggml-cuda/template-instances/` before committing any new turbo type dispatch.

### 9. ONE commit per logical change. Include BOTH speed AND PPL data.

### 10. NEVER ask "want me to continue?" The answer is always yes.

## Build & Test Commands

```bash
cd /home/erol/ai/turboquant/turboquant-kv-cache
/home/erol/miniconda3/envs/tq/bin/cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
/home/erol/miniconda3/envs/tq/bin/cmake --build build -j$(nproc)

MODEL=/home/erol/ai/turboquant/models/opus-v2-Q6_K.gguf
WIKI=$(find /home/erol/ai/turboquant -name "wiki.test.raw" 2>/dev/null | head -1)

# REGRESSION SUITE (run after EVERY change):
./build/bin/llama-bench -m $MODEL -fa 1 -ctk turbo3 -ctv turbo3 -d 0 -ngl 99 -t 1 -r 3 -p 0 -n 128 -mmp 0
./build/bin/llama-perplexity -m $MODEL -f $WIKI -c 512 -ctk turbo3 -ctv turbo3 -fa on --chunks 8 -ngl 99 --no-mmap
./build/bin/llama-bench -m $MODEL -fa 1 -ctk turbo4 -ctv turbo4 -d 0 -ngl 99 -t 1 -r 3 -p 0 -n 128 -mmp 0
./build/bin/llama-bench -m $MODEL -fa 1 -ctk turbo1.5 -ctv turbo1.5 -d 0 -ngl 99 -t 1 -r 3 -p 0 -n 128 -mmp 0

# Asymmetric check (must not crash):
./build/bin/llama-bench -m $MODEL -fa 1 -ctk turbo4 -ctv turbo3 -d 0 -ngl 99 -t 1 -r 1 -p 0 -n 4 -mmp 0
./build/bin/llama-bench -m $MODEL -fa 1 -ctk turbo1.5 -ctv turbo3 -d 0 -ngl 99 -t 1 -r 1 -p 0 -n 4 -mmp 0
```

## Key Files (This Repo — After Session 18)

```
ggml/src/ggml-cuda/set-rows.cu       — Parallel SET_ROWS: turbo3/turbo4/turbo2/turbo1.5 (128 threads, warp intrinsics)
ggml/src/ggml-cuda/turbo-quant.cuh   — Centroid constants, helper functions, sign arrays, trit LUT
ggml/src/ggml-cuda/turbo-wht.cu      — GGML_OP_TURBO_WHT CUDA kernel
ggml/src/ggml-cuda/turbo-sink.cu     — Attention sinks (TO BE DELETED in Session 19 — 0% PPL benefit, crashes SM86)
ggml/src/ggml-cuda/turbo-innerq.cu   — InnerQ per-channel equalization
ggml/src/ggml-cuda/fattn-common.cuh  — FA vec_dot for turbo3/4/2/1.5, V dequant, sparse V, LUT (turbo2 only)
ggml/src/ggml-cuda/fattn-vec.cuh     — VEC FA kernel, sink dispatch, LUT dispatch
ggml/src/ggml-cuda/fattn.cu          — FA dispatch, type validation
ggml/src/ggml-cuda/template-instances/ — VEC template instances (CHECK these for every K×V combo)
ggml/src/ggml-turbo-quant.c          — CPU reference quantize/dequant (all 4 types)
ggml/src/ggml-common.h               — Block structs for all 4 types
src/llama-kv-cache.cpp               — Layer-adaptive modes 1-11, turbo type checks
src/llama-graph.cpp                   — Graph-level WHT rotation (5 build_attn overloads, all have WHT)
```

## Lessons Learned (From Session 18 Failures)

### `__managed__` memory kills SM86
Session 17-18 used `__managed__` for sink state. Works on SM120 by luck. Crashes on SM86 because `__managed__` triggers implicit page faults during CUDA graph replay. NEVER use `__managed__` in any kernel-accessible path.

### LUT attention has bank conflicts for turbo3
Session 17 showed +7.2% from turbo3 LUT. Session 18 review found it causes -1.8% from shared memory bank conflicts (8 centroids × 256 dims, threads hitting same bank). Disabled for turbo3. turbo2 (4 centroids) works fine. Investigation needed: try `LUT[D][8+1]` padding to stagger banks.

### V sink managed memory in hot loop = -14% at 32K
Adding `__managed__` reads to the VEC V accumulation loop caused -3% short, -14% 32K. Removed in review pass. Sinks must use explicit device memory or kernel args, NOT managed memory.

### Always verify type properties before writing type-specific code
turbo4 and turbo1.5 use q8_1 Q format (`K_is_unquantized = false`). turbo3 and turbo2 use float Q (`K_is_unquantized = true`). The LUT kernel requires float Q. Writing turbo4 LUT without checking this wasted time and produced broken code.

## Environment Variables

| Variable | Effect | Status |
|----------|--------|--------|
| `TURBO_LAYER_ADAPTIVE=N` | Per-layer KV type (modes 0-11) | Working |
| `TURBO_INNERQ=N` | InnerQ calibration | Working |
| `TURBO_SINK_SIZE=N` | Attention sinks (first N positions at fp16) | BUGGY — crashes on SM86, 0% PPL benefit on SM120 |

## Commit Message Format

```
<type>: <short description>

<Detailed explanation>

  Speed: turbo3=XX.XX, turbo4=XX.XX, turbo1.5=XX.XX tok/s
  PPL: turbo3 ctx=512=X.XXXX, ctx=2048=X.XXXX
```

**NEVER add Co-Authored-By lines.**

## Q Format Architecture (Key Insight from Session 18)

The VEC FA kernel has two Q data paths that dominate long-context performance:

| Q Format | Types | Short Decode | Long Context | Why |
|----------|-------|:---:|:---:|-----|
| **float Q** | turbo3, turbo2 | Best (LUT compatible) | **Worst** | 4x Q bandwidth |
| **q8_1 Q** | turbo4, turbo1.5 | Good | **Best** | Minimal bandwidth |

turbo3 at 32K = 40.90 vs turbo4 = 46.51 (-12%). Root cause: `K_is_unquantized=true` for turbo3 → float Q. **Highest-priority optimization: q8_1-compatible turbo3 vec_dot** to eliminate this gap.

## What To Build Next (Priority Order)

1. **Session 19**: Delete sinks (SM86 crash fix), add missing template instances (asymmetric K/V crash fix)
2. **q8_1 turbo3 vec_dot** — eliminate 4x Q bandwidth penalty at long context
3. **turbo3 LUT bank conflict fix** — `LUT[D][8+1]` padding to recover +7% at short
4. **turbo1.5 branchless ternary** — `float(trit) * C * norm` instead of centroid lookup
5. **FP4 tensor core for turbo1.5** — SM120 moonshot: map ternary to FP4 E2M1 `mma.sync`

## Remember

- **turbo4 = 100.1% of f16** at short context (59.06 vs 59.00)
- **turbo1.5 = 113 tok/s on MoE** (beats signalnine's turbo3 at 94)
- **turbo1.5 = 15.26 tok/s at 204K** (1.8x turbo3's 8.42, long-context champion)
- **3 critical bugs OPEN** — sinks SM86 crash, asymmetric turbo4/1.5 crash, turbo3/turbo4 6 tok/s
- **turbo3 needs q8_1 vec_dot** — float Q path is 4x bandwidth penalty at long context
- **turbo3 LUT disabled** — bank conflicts, needs `LUT[D][9]` padding investigation
- **Search the Obsidian vault** at `/mnt/c/vaults/forge/` for any context you need
- **MEASURE SPEED AND PPL** after every change. Not just one or the other.
- **READ THE CODE** before writing code. Every Session 18 bug came from not reading first.
- **DO NOT STOP. DO NOT DEFER. DO NOT SKIP. FINISH THE WORK.**
