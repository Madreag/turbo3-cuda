---
description:
alwaysApply: true
---

# CLAUDE.md — TurboQuant CUDA (Madreag/turbo3-cuda)

## ⛔ BENCHMARKS: FOREGROUND BASH CALL — IT WAKES YOU UP WHEN DONE ⛔
**Run llama-bench/llama-perplexity as a normal foreground Bash tool call (no `run_in_background`, no `timeout` parameter). The Bash tool BLOCKS and returns output when the command finishes — that automatically wakes you up. You do NOT need to poll or check. Just call Bash, wait, get results. ONE process at a time. NEVER background. Session 20 used `run_in_background` → 8 orphan processes × 22GB = 176GB → system crash → hard reboot. See Rule 11.**

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

**Current state**: Sessions 17-19 complete. All 4 turbo types (1.5/2/3/4) have full CUDA with parallel SET_ROWS, native FA vec_dot, V dequant. All 36 K×V asymmetric combos work. Sinks fixed (graph-compatible). LUT scoring paths deleted in S18 review — need restoration with fixes in Session 20.

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

## Current Performance (Session 20 Final, RTX 5090)

| Type | bpv | Short | 32K | PPL ctx=512 | PPL ctx=2048 | Notes |
|------|----:|------:|----:|:-----------:|:------------:|-------|
| f16 | 16 | ~60 | ~51 | — | — | Ceiling |
| q8_0 | 8.5 | 60.7 | 49.6 | 6.759 | 5.674 | Baseline |
| turbo4 | 4.25 | **59.4** | 42.2 | 6.825 (+0.97%) | 5.694 | LUT restored (16 centroids) |
| turbo3 | 3.25 | **60.2** | **45.9** | 6.852 (+1.38%) | **5.674 (=q8_0)** | q8_1 vec_dot + LUT with [D][9] padding |
| turbo2 | 2.5 | **60.4** | **47.9** | 7.080 (+4.75%) | 5.892 | q8_1 vec_dot + LUT, long-ctx champion |
| turbo1.5 | 2.0 | 59.0 | 38.2 | 7.312 (+8.18%) | 6.103 | **8x compression, 113 MoE, 15.26 @204K** |

**MoE (Qwen 3.5 35B-A3B)**: turbo1.5 = 113 tok/s, turbo4 = 98, turbo3 = 94.

## Session 19 Bugs — ALL FIXED

| Bug | Fix | Commit |
|-----|-----|--------|
| Sinks crash SM86 | `__managed__` → `__device__` + `cudaGetSymbolAddress` + `cudaMemcpyAsync` | `01a3b42` |
| Asymmetric K=turbo4/1.5 crash | 20 new VEC template instances for all K×V combos | `cad0533` |
| K=turbo3/V=turbo4 = 6 tok/s | Same fix — missing template caused CPU fallback | `cad0533` |

All 36 K×V combos verified working. Prefill (MMA/TILE) cross-type verified. SM86 verification pending.

## CRITICAL: Code That Was Deleted and Needs Restoration

Session 18 review deleted working code that should be brought back with fixes:

| Code | Git SHA | Why Deleted | How To Restore |
|------|---------|-------------|----------------|
| **turbo3 LUT scoring loop** | `54d119831` | Bank conflicts -1.8% | `[D][9]` padding + q8_1 Q dequant in LUT construction |
| **turbo4 LUT scoring loop** | `729f98db1` | q8_1 Q incompatible | q8_1 Q dequant in LUT construction (one-time cost) |
| **turbo2 LUT scoring** | Never existed | Dead code from day 1 | Build same pattern as turbo3/4 |
| **V sink in V accumulation** | `399549616` | `__managed__` = -14% at 32K | Use Session 19's `__device__` approach (no perf hit) |

**Use `git show <SHA>` to retrieve deleted code. Don't rewrite from scratch.**

## ABSOLUTE RULES

### 1. NEVER skip an item. NEVER defer.

### 2. MEASURE SPEED (short AND 32K) AND PPL after EVERY change.
Not just PPL — **decode speed at short AND long context**. Session 18 caught a -14% regression at 32K that short-only testing missed. Run:
```bash
# ALL THREE of these after every change:
./build/bin/llama-bench -m $MODEL -fa 1 -ctk turbo3 -ctv turbo3 -d 0 -ngl 99 -t 1 -r 3 -p 0 -n 128 -mmp 0
./build/bin/llama-bench -m $MODEL -fa 1 -ctk turbo3 -ctv turbo3 -d 32768 -ngl 99 -t 1 -r 3 -p 0 -n 32 -mmp 0
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
- turbo3, turbo2: `K_is_unquantized = true` → float Q path (4x bandwidth at long ctx)
- turbo4, turbo1.5: `K_is_unquantized = false` → q8_1 Q path (better bandwidth)
Session 20 goal: move turbo3/turbo2 to q8_1 Q, then build LUT with q8_1 Q dequant in construction.
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

### 11. Benchmark execution rules (HARD RULES — violation = system crash).
Session 20 spawned 8 background benchmark processes (8×22GB = 176GB) against 48GB RAM → swap death → hard reboot. NEVER AGAIN.
- **FOREGROUND BASH CALL** — the Bash tool blocks and returns output when done. That WAKES YOU UP automatically. You do NOT need to poll, sleep, or check.
- **NO `run_in_background`** — EVER. For any llama-bench or llama-perplexity command.
- **NO `timeout` parameter** — EVER. Not on the Bash tool, not with shell timeout.
- **ONE AT A TIME** — never parallel llama-bench or llama-perplexity
- **LET IT FINISH** — do NOT kill benchmarks. They return results when done.
- **ONE BASH CALL PER MESSAGE** — NEVER send multiple Bash calls in the same message. Not for benchmarks, not for builds, not for anything in this repo. One call, get result, then next call. Multiple Bash calls = parallel execution = stacked processes = system crash.
- **Same rules for cmake builds** — foreground, one at a time, no background.
- **BEFORE AND AFTER EVERY TEST**, verify process is dead and memory is free:
  ```bash
  pgrep -f "llama" && echo "PROCESS STILL RUNNING — KILL IT" || echo "Clean"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader
  ```
  If pgrep finds anything: STOP. Kill it. Verify again. Do NOT proceed until clean.

## Build & Test Commands

```bash
cd /home/erol/ai/turboquant/turboquant-kv-cache
/home/erol/miniconda3/envs/tq/bin/cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
/home/erol/miniconda3/envs/tq/bin/cmake --build build -j$(nproc)

MODEL=/home/erol/ai/turboquant/models/opus-v2-Q6_K.gguf
WIKI=$(find /home/erol/ai/turboquant -name "wiki.test.raw" 2>/dev/null | head -1)

# REGRESSION SUITE (run after EVERY change — short + 32K + PPL):
./build/bin/llama-bench -m $MODEL -fa 1 -ctk turbo3 -ctv turbo3 -d 0 -ngl 99 -t 1 -r 3 -p 0 -n 128 -mmp 0
./build/bin/llama-bench -m $MODEL -fa 1 -ctk turbo3 -ctv turbo3 -d 32768 -ngl 99 -t 1 -r 3 -p 0 -n 32 -mmp 0
./build/bin/llama-bench -m $MODEL -fa 1 -ctk turbo4 -ctv turbo4 -d 0 -ngl 99 -t 1 -r 3 -p 0 -n 128 -mmp 0
./build/bin/llama-bench -m $MODEL -fa 1 -ctk turbo1.5 -ctv turbo1.5 -d 0 -ngl 99 -t 1 -r 3 -p 0 -n 128 -mmp 0
./build/bin/llama-perplexity -m $MODEL -f $WIKI -c 512 -ctk turbo3 -ctv turbo3 -fa on --chunks 8 -ngl 99 --no-mmap
```

## Key Files (This Repo — After Session 19)

```
ggml/src/ggml-cuda/set-rows.cu       — Parallel SET_ROWS: all 4 turbo types (128 threads, warp intrinsics, rsqrtf)
ggml/src/ggml-cuda/turbo-quant.cuh   — Centroid constants, helpers, sign arrays, trit LUT (5×256)
ggml/src/ggml-cuda/turbo-wht.cu      — GGML_OP_TURBO_WHT CUDA kernel
ggml/src/ggml-cuda/turbo-sink.cu     — Attention sinks (__device__ + async, graph-compatible after S19)
ggml/src/ggml-cuda/turbo-innerq.cu   — InnerQ per-channel equalization
ggml/src/ggml-cuda/fattn-common.cuh  — FA vec_dot for all 4 types, V dequant, sparse V (LUT scoring MISSING)
ggml/src/ggml-cuda/fattn-vec.cuh     — VEC FA kernel, sink dispatch, LUT table built but NOT read
ggml/src/ggml-cuda/fattn.cu          — FA dispatch, type validation, all 36 K×V combos
ggml/src/ggml-cuda/template-instances/ — 32 VEC template instances (all K×V combos, D=64/128/256)
ggml/src/ggml-cuda/dequantize.cuh    — QR_TURBO4, QR_TURBO1_5 defined, all dequant functions
ggml/src/ggml-cuda/convert.cu        — to_fp16, to_fp32, to_fp16_nc for all 4 turbo types
ggml/src/ggml-turbo-quant.c          — CPU reference quantize/dequant (all 4 types)
ggml/src/ggml-common.h               — Block structs for all 4 types
src/llama-kv-cache.cpp               — Layer-adaptive modes 1-11, turbo type checks
src/llama-graph.cpp                   — Graph-level WHT rotation (5 build_attn overloads, all have WHT)
```

## Lessons Learned (From Session 18 Failures)

### `__managed__` memory kills SM86
Session 17-18 used `__managed__` for sink state. Works on SM120 by luck. Crashes on SM86 because `__managed__` triggers implicit page faults during CUDA graph replay. NEVER use `__managed__` in any kernel-accessible path.

### LUT scoring paths are dead code — need restoration with fixes
Session 17 showed +7.2% from turbo3 LUT. Session 18 disabled it (bank conflicts) and deleted the scoring loop. Session 19 confirmed turbo2 LUT is ALSO dead code — table built in shared memory but nobody reads it. **All LUT scoring paths need to be rebuilt in Session 20** with:
1. `[D][9]` padding (fixes bank conflicts — stride 9 coprime to 32 banks)
2. q8_1 Q dequant in LUT construction (compatible with q8_1 Q path)
3. New scoring branch in VEC kernel loop that reads `turbo_lut[elem][idx]`

See vault: `10 Knowledge/CUDA/Shared Memory Bank Conflicts.md`

### V sink was removed but can be restored
Session 18 removed V sinks because `__managed__` reads in hot loop = -14% at 32K. Session 19 replaced `__managed__` with `__device__` + async copy — the perf hit reason is gone. **V sinks should be restored using Session 19's approach.** Code at `git show 399549616`.

### Always verify type properties before writing type-specific code
turbo4 and turbo1.5 use q8_1 Q format (`K_is_unquantized = false`). turbo3 and turbo2 use float Q (`K_is_unquantized = true`). The LUT construction reads Q as float — but q8_1 Q can be dequantized on the fly (one-time cost, D elements). This makes LUT compatible with ALL Q paths.

## Environment Variables

| Variable | Effect | Status |
|----------|--------|--------|
| `TURBO_LAYER_ADAPTIVE=N` | Per-layer KV type (modes 0-11) | Working |
| `TURBO_INNERQ=N` | InnerQ calibration | Working |
| `TURBO_SINK_SIZE=N` | Attention sinks (first N positions at fp16) | FIXED in Session 19 (`__device__` + async). 0% PPL on 27B — needs multi-model test. V sinks removed, need restoration. |

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

## What To Build Next — Session 20 (Priority Order)

1. **q8_1 turbo3/turbo2 vec_dot** — move off float Q, eliminate 4x bandwidth penalty at long ctx
2. **Restore LUT scoring for ALL turbo types** — retrieve from `git show 54d119831`, add `[D][9]` padding + q8_1 Q dequant in construction. Build turbo2/turbo4 LUT scoring too (currently dead code).
3. **Restore V sinks** — retrieve from `git show 399549616`, use Session 19's `__device__` approach
4. **turbo1.5 branchless ternary** — `float(trit) * C * norm` instead of centroid lookup
5. **FP4 tensor core for turbo1.5** — SM120 moonshot: map ternary to FP4 E2M1 `mma.sync`

## Obsidian Vault Maintenance

The project knowledge base lives at `/mnt/c/vaults/forge/`. **Read `10 Knowledge/README.md` for CUDA/hardware reference docs before starting kernel work.**

**After EVERY session**, update these vault files with your results:
1. `00 Dashboard/Project Status.md` — Check off completed items, add new open work
2. `03 Benchmarks/Benchmark Hub.md` — Add your final benchmark numbers
3. `08 Plans/Roadmap.md` — Update session status (DONE/PLANNED), add next session items
4. `01 Sessions/Session N.md` — Create or update your session's tracking note

If you discover a bug, create an issue file in `07 Issues/`. If you make an architecture decision, document it in `02 Architecture/`.

## Remember

- **Session 19 bugs FIXED** — all 36 K×V combos work, sinks graph-compatible, SM86 pending verification
- **LUT scoring is DEAD CODE** for ALL turbo types — table built, nobody reads it. Restore in Session 20.
- **V sinks were removed** — the `__managed__` reason is gone (Session 19 fix). Restore with `__device__` approach.
- **turbo3/turbo2 need q8_1 vec_dot** — float Q = 4x bandwidth penalty at long context
- **turbo1.5 = 113 tok/s on MoE**, 15.26 at 204K — long-context champion
- **Use `git show <SHA>` to retrieve deleted code** — don't rewrite from scratch
- **Search the Obsidian vault** at `/mnt/c/vaults/forge/` for any context you need
- **Read `10 Knowledge/README.md`** for CUDA reference docs before kernel work
- **MEASURE SHORT + 32K + PPL** after every change. Session 18 caught -14% at 32K that short missed.
- **READ THE CODE** before writing code. Every Session 18 bug came from not reading first.
- **DO NOT STOP. DO NOT DEFER. DO NOT SKIP. FINISH THE WORK.**
- **⛔ BENCHMARKS: Foreground Bash call ONLY — it blocks and WAKES YOU UP when done. NO run_in_background. NO timeouts. ONE at a time. See Rule 11. ⛔**
