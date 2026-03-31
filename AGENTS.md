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

**Current state**: Sessions 17-24B complete. All 4 turbo types (1.5/2/3/4) fully optimized with parallel SET_ROWS, native FA vec_dot (q8_1 Q + LUT scoring for turbo3/turbo2), V dequant with sparse skip, `__expf` fast-math softmax, constexpr centroids, nthreads_KQ=8 for all types. 36 K×V asymmetric combos, 5 models validated (D=64/96/128/256), 3 GPUs (SM86/89/120). **We beat AmesianX on ALL types by +10-35%.** AmesianX comp test: `COMP_TEST_RESULTS.md`.

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
    - `07 Issues/` — 0 open bugs, 10 resolved issues with root cause + commit SHA
    - `10 Knowledge/` — CUDA specifics, hardware constraints, quantization theory
    - `01 Sessions/` — all 18 session reports with what worked and what didn't

- **Models**: `/home/erol/ai/turboquant/models/opus-v2-Q6_K.gguf` (27B dense), `Qwen3.5-35B-A3B-Q4_K_M.gguf` (MoE)
- **Hardware**: RTX 5090 32GB (SM120), RTX 3090 Ti 24GB (SM86), RTX 4090M 16GB (SM89), Mac Mini M4 Pro 24GB (Metal)

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

## Current Performance (Session 24B, RTX 5090)

| Type | bpv | Short | 32K | PPL ctx=512 | PPL ctx=2048 | Notes |
|------|----:|------:|----:|:-----------:|:------------:|-------|
| f16 | 16 | 59.52 | 54.11 | — | — | Ceiling |
| q8_0 | 8.5 | 58.58 | 47.99 | 6.759 | 5.674 | Baseline |
| turbo4 | 4.25 | **65.18** | **54.22** | 6.825 (+0.97%) | 5.694 | nthreads_KQ=8 + constexpr centroids (S24B) |
| turbo3 | 3.25 | **65.04** | **54.89** | 6.852 (+1.38%) | **5.674 (=q8_0)** | __expf + constexpr centroids |
| turbo2 | 2.5 | **65.24** | **53.57** | 7.080 (+4.75%) | 5.892 | constexpr centroids, long-ctx champion |
| turbo1.5 | 2.0 | **63.99** | **48.15** | 7.312 (+8.18%) | 6.103 | nthreads_KQ=8 + constexpr (S24B) |

**Q4_K_M (Session 24)**: turbo3=77.25 short, 58.08 32K. **PPL ctx=2048=7.716 (beats q8_0 7.730!)**
**MoE (Qwen 3.5 35B-A3B, S20)**: turbo3=184, turbo1.5=174, turbo4=172, q8_0=191 tok/s
**AmesianX (S24 comp test)**: We beat them on ALL types: turbo3 +15-35%, turbo4 +10-11%

### Cross-GPU Stability (Sessions 20-24B)
- **3090 Ti (SM86)**: 356 iterations, 35/35 PPL checks bit-exact (7.5535), 0 failures
- **4090M (SM89)**: 425 iterations, 14+ PPL checks bit-exact (7.5912), 0 failures
- **RTX 5090 (SM120)**: 340+ iterations, continuous PPL checks, 0 failures
- **Total**: **1,121+ iterations, 49+ bit-exact PPL checks, 0 failures across 3 GPUs**
- **turbo2 beats q8_0 at 32K on ALL tested models** (1B, Phi-4, 8B, 12B, 27B)

### Supported Head Dimensions
Only D∈{64, 128, 256}. VEC kernel requires `D % 64 == 0` (static_assert). D=80, D=96, D=112 fall back to non-FA mul_mat attention (slower but correct, no crash).

## Session 19 Bugs — ALL FIXED

| Bug | Fix | Commit |
|-----|-----|--------|
| Sinks crash SM86 | `__managed__` → `__device__` + `cudaGetSymbolAddress` + `cudaMemcpyAsync` | `01a3b42` |
| Asymmetric K=turbo4/1.5 crash | 20 new VEC template instances for all K×V combos | `cad0533` |
| K=turbo3/V=turbo4 = 6 tok/s | Same fix — missing template caused CPU fallback | `cad0533` |

All 36 K×V combos verified working across all 3 GPUs. S24B verified 25-combo sweep on 4 models (1B/Phi-4/8B/12B).

## Restoration History (Completed)

All deleted code has been restored with fixes in Sessions 20-22B:
- **turbo3 LUT**: Restored S20 with [D][9] padding. Upgraded to 8-wide in S22B. Working.
- **turbo2 LUT**: Built in S20 matching turbo3 pattern. 4-centroid, 8-wide. Working.
- **turbo4 LUT**: Tested in S21, 16-centroid LUT was **net negative** (Dead End #17). Disabled.
- **V sinks**: Attempted S20 with `__device__`. **Dead End #16** — register pressure -12.7% at 32K.

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

### 4. SPEED REJECT THRESHOLDS (Updated S24B)
- turbo3 short: < 60.0 tok/s → **INVESTIGATE** (baseline is 65.04)
- turbo4 short: < 60.0 tok/s → **INVESTIGATE** (baseline is 65.18)
- turbo2 short: < 60.0 tok/s → **INVESTIGATE** (baseline is 65.24)
- turbo1.5 short: < 58.0 tok/s → **INVESTIGATE** (baseline is 63.99)
- ANY type 32K: < 45.0 tok/s → **INVESTIGATE**

### 5. READ THE CODE before writing code.
Session 18 failures all came from not reading existing code:
- Sink ne0 mismatch: didn't read `sink_get_or_alloc` before modifying FA dispatch
- turbo4 LUT with q8_1 Q: didn't check `K_is_unquantized` before writing LUT path
- V sink perf regression: added `__managed__` reads to hottest loop without profiling

**Before touching ANY kernel**: Read the function you're modifying AND the functions that call it.

### 6. Q format and nthreads_KQ for turbo types.
ALL turbo types use q8_1 Q path (since S20). `K_is_unquantized` is only true for f16/bf16.
The `K_is_turbo` flag (S24B) controls nthreads_KQ for ALL 4 turbo types:
- ALL turbo types: `K_is_turbo = true` → nthreads_KQ=8 (4 interleaved dots per warp)
- Standard quants (q4_0, q8_0 etc.): nthreads_KQ=32
LUT scoring (turbo3/turbo2 only) reads Q as float from global memory for one-time table construction.
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

## Key Files (Updated Session 24B)

```
ggml/src/ggml-cuda/fattn-vec.cuh     — THE HOT KERNEL: VEC FA, LUT scoring (8-wide turbo3/turbo2), __expf softmax,
                                        sparse V skip, L2 prefetch, __launch_bounds__(128,3), K_is_turbo nthreads_KQ=8
ggml/src/ggml-cuda/fattn-common.cuh  — vec_dot functions (turbo3 line ~299, turbo2 ~348, turbo4 ~395, turbo1.5 ~500),
                                        V dequant, get_vec_dot_KQ dispatch, get_dequantize_V dispatch
ggml/src/ggml-cuda/turbo-quant.cuh   — static constexpr centroid arrays (S24B), helpers, sign arrays, trit LUT (5×256)
ggml/src/ggml-cuda/fattn.cu          — FA dispatch, supports_op, D check, all 36 K×V combo routing
ggml/src/ggml-cuda/set-rows.cu       — Parallel SET_ROWS encode: all 4 turbo types (128 threads, warp intrinsics)
ggml/src/ggml-cuda/turbo-wht.cu      — GGML_OP_TURBO_WHT CUDA kernel (forward + inverse rotation)
ggml/src/ggml-cuda/turbo-sink.cu     — Attention sinks (__device__ + async, graph-compatible)
ggml/src/ggml-cuda/template-instances/ — 32 VEC template instances (all K×V combos, D=64/128/256)
ggml/src/ggml-cuda/dequantize.cuh    — QR_TURBO4, QR_TURBO1_5, all dequant functions
ggml/src/ggml-cuda/convert.cu        — to_fp16, to_fp32, to_fp16_nc for all 4 turbo types
ggml/src/ggml-turbo-quant.c          — CPU reference quantize/dequant (all 4 types)
ggml/src/ggml-common.h               — Block structs: block_turbo3_0 (16B/32val), block_turbo4_0 (68B/128val),
                                        block_turbo2_0 (10B/32val), block_turbo1_5 (16B/32val)
src/llama-kv-cache.cpp               — LA modes 0-15 (using KV ordinals since S24), turbo type checks, GQA warning
src/llama-context.cpp                — FA auto-enable for turbo types, GQA >8:1 warning (S24)
src/llama-graph.cpp                   — Graph-level WHT rotation (5 build_attn overloads)
COMP_TEST_RESULTS.md                 — AmesianX head-to-head comparison data (S24)
```

## Lessons Learned (Sessions 17-24B)

### `__managed__` memory kills SM86
NEVER use `__managed__` in any kernel-accessible path. Crashes SM86 via page faults during CUDA graph replay. Fixed in S19 with `__device__` + `cudaMemcpyToSymbolAsync`.

### LUT scoring: [D][n+1] padding fixes bank conflicts
turbo3 (8 centroids) and turbo2 (4 centroids) use LUT in shared memory. Stride n_centroids causes systematic bank conflicts. Padding to n+1 makes stride coprime to 32 banks. 8-wide scoring (2 qs bytes per iteration) gives +4.7% at 32K. turbo4 LUT (16 centroids) is Dead End #17 — 8.7KB shmem net negative.

### V sinks: Dead End #16
Register pressure from V sink variables reduces VEC kernel occupancy. -12.7% at 32K. NOT from `__managed__` memory — from register allocation. Would need a separate kernel variant.

### nthreads_KQ must be 8 for ALL turbo types
Dead End #19: nthreads_KQ=32 = -17% at 32K. Fixed for turbo3/turbo2 in S20, fixed for turbo4/turbo1.5 in S24B. With nthreads_KQ=8, each warp processes 4 interleaved KQ dot products instead of 1 — better latency hiding.

### Centroid arrays: constexpr > __constant__ > __device__
S24B moved all centroid arrays from `__constant__`/`__device__` to `static constexpr __device__`. Compiler can place small constexpr arrays in registers (0 latency vs ~30 cycle constant memory). AmesianX uses the same approach.

### `__expf` is safe for attention softmax
S24 replaced all 5 `expf` with `__expf` in VEC kernel. ~2^-21 relative error is irrelevant for attention score normalization. PPL bit-exact at ctx=2048. +3.69% at 32K.

### Norm-out-of-loop: Dead End #32
S24B tested factoring norm multiplication out of centroid lookup (1 multiply instead of 4 per iteration). Compiler was already doing this optimization — manual refactor interfered with instruction scheduling. -0.8% regression. Reverted.

### Sparse V threshold: 5e-3 for turbo3/4, 1e-2 for turbo2/1.5 (S27B validated)
S25 raised sparse V skip threshold from 1e-6 to type-specific values. **S27B control test proved 1e-2 threshold has ZERO PPL impact** — turbo2 ctx=32K PPL is bit-identical at 1e-2 and 1e-6 (10.0536 both). Same for turbo1.5 (11.1275 both). The growing delta at long context is inherent to 2-bit quantization, not the threshold. 1e-2 gives +13% speed at 32K for zero quality cost. turbo3/turbo4 stay at 5e-3 (proven healthy < 3% delta at 32K).

### K type dominates 32K speed, V type barely matters (S25 finding)
With sparse V skip, most V positions are never read. K scoring is the bottleneck at long context. On 8B Llama at 32K: turbo2 K gives 113-119 tok/s regardless of V type (5% spread). turbo3 K: 105-112. turbo4 K: 74-75. turbo1.5 K: 67. **Implication for LA: use turbo2 K for speed, V type for quality only.**

### 2-bit quantization PPL grows with context (S27 finding)
S27 50-chunk wikitext-103 validation showed turbo2/turbo1.5 PPL delta grows with context length (+12%/+24% at 32K vs q8_0). Control test proved this is **inherent to 2-bit quantization** (not threshold-caused): PPL is bit-identical at 1e-2 and 1e-6 thresholds. turbo3/turbo4 stay healthy (< 3% delta at 32K). This is a fundamental limitation of low-bpv types at long context.

### VEC kernel at optimization ceiling (168 regs, 98.4% utilization)
S25 tested 33 micro-optimizations + S27 tested 4 more (context-adaptive threshold, turbo1.5 LUT, asymmetric K/V, sparse K). ALL failed due to register pressure at 168/170. Adding even 1 register causes spills → -2% to -4%. The LUT is essential for BOTH performance AND register pressure (without LUT: 255 regs → massive spilling). The only successful approach was changing CONSTANTS (thresholds), not code STRUCTURE.

## Environment Variables

| Variable | Effect | Status |
|----------|--------|--------|
| `TURBO_LAYER_ADAPTIVE=N` | Per-layer KV type (modes 0-15, KV ordinals since S24) | Working |
| `TURBO_INNERQ=N` | InnerQ calibration | Working |
| `TURBO_SINK_SIZE=N` | Attention sinks (first N positions at fp16) | FIXED on all GPUs (S19 SM86, S22B SM89 alignment fix). 0% PPL benefit. V sinks = dead end (register pressure). |

## Commit Message Format

```
<type>: <short description>

<Detailed explanation>

  Speed: turbo3=XX.XX, turbo4=XX.XX, turbo1.5=XX.XX tok/s
  PPL: turbo3 ctx=512=X.XXXX, ctx=2048=X.XXXX
```

**NEVER add Co-Authored-By lines.**

## Q Format Architecture (Updated Session 24B)

ALL turbo types use q8_1 Q + nthreads_KQ=8. Centroids are `static constexpr` (register-allocated). LUT scoring uses Q from global memory (still float) for one-time table construction.

| Type | Q Path | nthreads_KQ | LUT | Short | 32K | Key Optimization |
|------|:------:|:-----------:|:---:|------:|----:|-----------------|
| turbo3 | q8_1 | 8 | 8 centroids (8-wide) | **65.04** | **54.89** | q8_1 vec_dot + LUT + __expf + constexpr centroids |
| turbo4 | q8_1 | 8 (S24B fix) | disabled | **65.18** | **54.22** | nthreads_KQ=8 (was 32) + constexpr. **AmesianX gap closed: +10-11%** |
| turbo2 | q8_1 | 8 | 4 centroids (8-wide) | **65.24** | **53.57** | Long-context champion + constexpr |
| turbo1.5 | q8_1 | 8 (S24B fix) | none | **63.99** | **48.15** | nthreads_KQ=8 (was 32) + constexpr |

## What To Build Next — Session 25+ (Priority Order)

1. **Kernel autoresearch loop** — 50-100+ micro-optimization iterations (compiler flags, PTX intrinsics, LUT variants, launch config tuning). See `session-25-deep.md` prompt.
2. **Block-256 turbo4** — AmesianX uses 256-element blocks with half the norm overhead. +2-3% at 32K potential. Major refactor: 10+ files.
3. **Float Q for turbo4/turbo1.5** — Remove q8_1 overhead. Risk: may regress at 32K (bandwidth vs compute tradeoff). Needs careful benchmarking.
4. **Discussion #20969 post** — All data collected. Draft in vault `09 Community/DISCUSSION_DRAFT_20969.md`.
5. **PR to TheTom upstream** — Squash Sessions 17-24B into clean commits.
6. **ARKV auto layer-adaptive** — Entropy-based per-layer type selection during prefill.

## Completed Optimizations (Sessions 17-24)

| Session | What | Impact |
|---------|------|--------|
| 17 | signalnine base + LUT attention | turbo3 60.77 (+7.2%) |
| 18 | turbo4, turbo1.5, trit LUT, review pass | 4 types complete |
| 19 | Sinks fix, 36 K×V combos, prefill verified | All bugs fixed |
| 20 | q8_1 turbo3/turbo2 vec_dot, LUT restored | turbo3 32K +7% |
| 21 | `__launch_bounds__(128,3)`, turbo4 LUT removed | ALL types +7-13% at 32K |
| 22 | Multi-model validation (5 models D=64/96/128/256) | Zero crashes, D=96 graceful fallback |
| 22B | SM89 sink fix, L2 prefetch, 8-wide LUT turbo3/turbo2 | turbo3 short +1.8%, 32K +4.7%, turbo2 32K +3.0% |
| 23 | Q4_K_M validation, LA=12 boundary V, fattn.cu D check, README | Q4_K_M+turbo1.5 131K=25.81, LA=12 74.8% gap recovery |
| 24 | __expf fast-math, LA kv_ord hybrid fix, LA=13-15, GQA warning | turbo3 short +0.67%, 32K +3.69%, Q4_K_M ctx=2048 beats q8_0 |
| 24B | nthreads_KQ=8 for turbo4/1.5, constexpr centroids | turbo4 short +10.7%, 32K +17.7%. AmesianX gap closed: +10-11% us |
| 25 | Sparse V threshold 5e-3/1e-2, 43 iterations autoresearch | turbo3 32K +5-11%, turbo2 +13%, 8B +28%. K type dominates 32K speed. |
| 26 | SM120 D=256 LUT fix (NVBUG 5218000), turbo4/1.5 Q_reg fix, block-128 storage | Block-128: turbo3 3.125bpv, turbo2 2.125bpv. All types beat q8_0. |
| 27 | Norm correction verified, 50-chunk wikitext-103, skip rate measurement, quality gate | PPL delta grows with ctx for turbo2/1.5 (inherent to 2-bit quant). |
| 27B | Sparse V threshold VALIDATED + 4 optimization attempts (all dead). NIAH 5090. | 1e-2 proven correct. Context-adaptive, turbo1.5 LUT, sparse K all dead at 168 reg ceiling. |

## Dead Ends (Don't Repeat)

| What | Why It Failed | Session |
|------|--------------|---------|
| V sinks in V accumulation loop | Register pressure -12.7% at 32K (not __managed__) | 20 |
| turbo4 16-centroid LUT | 8.7KB shmem net negative at all contexts | 21 |
| V-specific 64×64 rotation | Inverse WHT hardcodes group_size from tensor ne[0], not from quantization | 20 |
| nthreads_KQ=32 for turbo3 q8_1 | Kills warp-level ILP, -17% at 32K | 20 |
| Sinks for PPL improvement | 0% across 2 models, 5 contexts, 3 sink sizes | 19-20 |
| TURBO_SINK_SIZE on SM89 | Segfault at sizes {1,4,16}; {0,2,8} work. Host-side addressing bug in sink_get_or_alloc | 21 |
| FP4 E2M1 for Q | 99.5% of Q values map to zero (σ=0.088, E2M1 min non-zero=0.5). No mixed fp16×E2M1 MMA | 22 |
| Inner-loop V prefetch hints | RTX 5090 HW prefetcher already handles sequential pattern. No measurable benefit | 23 |
| turbo1.5 sparse V threshold=1e-4 | PPL safe but no speed benefit at 32K or 131K | 23 |
| elect_leader() PTX | VEC kernel warp-leader branches are outside hot loop, no serialization to eliminate | 24 |
| Tawa producer-consumer warp split | VEC ncols=1 is bandwidth-limited, shared memory staging adds overhead, not compute-limited | 24 |
| cp.async K loading | K blocks already loaded directly to registers, no staging benefit | 24 |
| Norm out of loop (simple factor) | Compiler already optimizes norm×centroid multiplication chain. Manual refactor -0.8% regression | 24B |
| --use_fast_math global | -0.77% short, neutral 32K. FTZ changes codegen for ALL kernels | 25 |
| --maxrregcount=128 | Neutral. 128 regs = 4 blocks/SM but spill penalty cancels occupancy gain | 25 |
| launch_bounds(128,2) | -2.57% short. Reduced occupancy hurts | 25 |
| launch_bounds(128,4) | -5.5% 32K. Spilling 40 regs to fit 4 blocks/SM | 25 |
| Float Q for turbo types | -2.62% 32K. Q_reg (half2[8]) adds more pressure than Q_i32/Q_ds saves | 25 |
| nthreads_KQ=16 | -7.2% short, -6.9% 32K. Only 2 interleaved dots/warp vs 4 | 25 |
| Any code restructuring | -2% to -7%. 168 regs = 98.4% utilization, any change causes spills | 25 |
| ptxas flags (opt-level=4, expensive-opts, ftz) | Mixed: help 32K +1.5-2%, hurt short -1-2%. Cannot resolve with global flags | 25 |
| Context-adaptive sparse V threshold | Non-constexpr threshold adds 1 register → spill. -1% at 32K for zero PPL benefit | 27 |
| turbo1.5 3-entry LUT scoring | Zero improvement. Trit multiply already trivial; LUT trades one lookup for another | 27 |
| K=turbo2/V=turbo3 asymmetric config | 55 tok/s 32K with +6.95% PPL delta. Worse than pure turbo3 (53.7 tok/s, +2.84%) | 27 |
| Sparse K (norm early exit) | -1.5% at 32K. Branch + global read cost > rare skip benefit. Most K norms are non-trivial | 27 |

## Obsidian Vault Maintenance

The project knowledge base lives at `/mnt/c/vaults/forge/`. **Read `10 Knowledge/README.md` for CUDA/hardware reference docs before starting kernel work.**

**After EVERY session**, update these vault files with your results:
1. `00 Dashboard/Project Status.md` — Check off completed items, add new open work
2. `03 Benchmarks/Benchmark Hub.md` — Add your final benchmark numbers
3. `08 Plans/Roadmap.md` — Update session status (DONE/PLANNED), add next session items
4. `01 Sessions/Session N.md` — Create or update your session's tracking note

If you discover a bug, create an issue file in `07 Issues/`. If you make an architecture decision, document it in `02 Architecture/`.

## Remember

- **Sessions 17-24B COMPLETE** — all 4 turbo types optimized, 36 K×V combos, 4 GPUs validated
- **ALL types beat q8_0 at short context**: turbo4=65.18, turbo3=65.04, turbo2=65.24, turbo1.5=63.99
- **ALL types beat q8_0 at 32K**: turbo4=54.22, turbo3=54.89, turbo2=53.57, turbo1.5=48.15
- **turbo3 = q8_0 PPL at ctx=2048** (5.674 = 5.674) at 4.6x compression
- **Q4_K_M turbo3 = 77.25 tok/s** (beats vLLM INT4 68.3 on same GB202 die)
- **Q4_K_M turbo3 PPL ctx=2048 = 7.716** (BEATS q8_0 7.730!)
- **MoE: turbo3=184 tok/s** (+96% vs S18), turbo1.5=174
- **AmesianX head-to-head**: we beat them on ALL types — turbo3 +16-35%, turbo4 +10-11%
- **Cross-GPU stability**: 1,121+ iterations (356 SM86 + 425 SM89 + 340 SM120), 49+ PPL checks bit-exact, 0 failures
- **D∈{64, 128, 256} only** — D=96 falls back to non-FA. VEC kernel: `static_assert(D % 64 == 0)`
- **Vault**: `/mnt/c/vaults/forge/` — `06 Models/Validation Target Models.md` for multi-model specs, `09 Infrastructure/Test Machines.md` for GPU fleet
- **Dump folder**: `/mnt/c/vaults/dump/` — incoming test results from 3090 Ti, 4090M, and M4 Pro
- **MEASURE SHORT + 32K + PPL** after every change
- **READ THE CODE** before writing code
- **Search the Obsidian vault** for any context you need
- **DO NOT STOP. DO NOT DEFER. DO NOT SKIP. FINISH THE WORK.**
- **⛔ BENCHMARKS: Foreground Bash call ONLY — it blocks and WAKES YOU UP when done. NO run_in_background. NO timeouts. ONE at a time. See Rule 11. ⛔**
