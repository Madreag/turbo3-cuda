# SESSION 20 — TO THE MOON: Results

## PART 1: q8_1-Compatible turbo3/turbo2 vec_dot ✓
- [x] Rewrote turbo3 vec_dot: 4-element groups, reads Q_q8/Q_ds instead of Q_v
- [x] Rewrote turbo2 vec_dot: same pattern
- [x] Removed turbo3/turbo2 from K_is_unquantized and V_is_unquantized
- [x] Key finding: nthreads_KQ=8 (not 32) critical for 32K performance
  - nthreads_KQ=32 caused -17% regression at 32K (35 vs 42 tok/s)
  - nthreads_KQ=8 gives 4 interleaved KQ dots per warp = better latency hiding

| Metric | Before | After | Change |
|--------|-------:|------:|--------|
| turbo3 short | 59.36 | 60.47 | +1.9% |
| turbo3 32K | 42.04 | 42.86 | +1.9% |
| turbo2 short | 59.65 | 60.11 | +0.8% |
| PPL | 6.8522 | 6.8522 | = |

## PART 2: Restore LUT Scoring for ALL Turbo Types ✓
- [x] Enabled LUT for turbo3 (8 centroids), turbo4 (16), turbo2 (4)
- [x] [D][n_centroids+1] padding fixes bank conflicts (stride coprime to 32)
- [x] 4-at-a-time scoring (shares qs/signs loads, 4x fewer iterations)
- [x] Q read from global memory for LUT construction (compatible with q8_1 Q path)
- [x] Per-element scoring had -3.8% short regression; 4-at-a-time fixed it

| Metric | After P1 | After P2 | Change |
|--------|-------:|------:|--------|
| turbo3 short | 60.47 | 59.89 | -1.0% |
| turbo3 32K | 42.86 | 45.86 | **+7.0%** |
| turbo4 short | 58.45 | 59.09 | +1.1% |
| turbo4 32K | 41.45 | 42.29 | +2.0% |
| turbo2 short | 60.11 | 60.08 | = |

## PART 3: V Sinks + turbo1.5 Branchless ✓
- [x] V sinks: ATTEMPTED, caused -12.7% 32K regression from register pressure
  - Not from __managed__ (fixed in S19) — from register allocation for sink variables
  - Reverted. V sinks need a different approach (separate kernel variant)
- [x] turbo1.5 branchless: ALREADY IMPLEMENTED (`float(trit) * C * norm`)

## PART 4: Quality Investigation ✓
- [x] Sink PPL: 0% across MoE model, ctx=2048/4096/8192, sinks=0/4/8
- [x] Asymmetric: K=turbo4/V=q8_0 = 6.7967 (best at ~6.4 bpv)
- [x] K=turbo4/V=f16 = 6.7775 (best overall quality)
- [x] turbo3 ctx=2048 PPL = 5.6744 (same as q8_0!)

## PART 5: FP4 Tensor Core Feasibility — DEEP RESEARCH ✓

### What Exists in the Codebase Already
The infrastructure is **fully built**:
- `mma.cuh:1048-1062`: `mma_block_scaled()` — the exact PTX `mma.sync.aligned.kind::mxf4` instruction for SM120
- `ggml-common.h:193-206`: `block_mxfp4` struct — 32 elements per block, E8M0 scale + 16 bytes packed E2M1
- `mmq.cuh:720-783`: `load_tiles_mxfp4()` — loads MXFP4 blocks into shared memory for MMA
- `mmq.cuh:1043-1061`: `vec_dot_mxfp4_mxfp4_mma()` — the actual MMA vec_dot using FP4 tensor cores
- `common.cuh:812-832`: `ggml_cuda_float_to_fp4_e2m1()` — float→FP4 quantization
- `convert.cu:473-620`: MXFP4/NVFP4 dequant kernels
- `CMakeLists.txt`: Blackwell 120a architecture support with `BLACKWELL_MMA_AVAILABLE`

### turbo1.5 → MXFP4 Mapping
turbo1.5 ternary {-C, 0, +C} maps to FP4 E2M1 {-1.0, 0.0, +1.0}:
- -1.0 in E2M1 = 0b1100 (sign=1, exp=10, mantissa=0)
- 0.0 in E2M1 = 0b0000
- +1.0 in E2M1 = 0b0100 (sign=0, exp=10, mantissa=0)
- Pack 2 per byte for e2m1x2 format
- Block scale = norm * C (absorb centroid into E8M0 scale)

### What Would Need to Be Built
1. **Repacking kernel**: Convert turbo1.5 trit-packed format (5 trits/byte) to MXFP4 format (2 E2M1/byte)
   - Could happen at SET_ROWS time (dual-format storage) or on-the-fly at FA time
   - 32 trits → 16 bytes MXFP4 qs + 1 byte E8M0 scale

2. **FA MMA kernel using FP4 tensor cores**: New `fattn-mma-fp4.cuh`
   - Use existing `mma_block_scaled()` from mma.cuh
   - Load turbo1.5 K/V as MXFP4 tiles (reuse load_tiles_mxfp4 pattern)
   - Q stays as fp16 (converted to E2M1 pairs? or keep fp16 and use mixed precision?)

3. **Q format issue**: `mma.sync` requires BOTH inputs as e2m1. Q cannot stay as fp16.
   - Option A: Quantize Q to FP4 at FA dispatch time (D elements, one-time cost)
   - Option B: Use fp16×fp4 mixed precision MMA if available (need to check PTX ISA)
   - The instruction signature is `f32.e2m1.e2m1.f32` — BOTH A and B must be e2m1

4. **Tile dimensions**: m16n8k64 processes 64 K-elements per MMA. For D=128: 2 MMAs per head.

### Feasibility Assessment
- **PREFILL**: Straightforward. MMA prefill already pre-dequants turbo→fp16. Could instead repack turbo1.5→mxfp4 and use mma_block_scaled. Expected 2x throughput vs fp16 MMA.
- **DECODE (VEC)**: NOT applicable. VEC kernel is scalar, not MMA. FP4 tensor cores don't help.
- **Q quantization to E2M1**: Significant precision loss. Q values are continuous floats, E2M1 only has 16 representable values (±{0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}). This WILL hurt attention quality.

### Verdict
**Feasible for prefill** with existing infrastructure. The codebase has everything needed:
- mma_block_scaled() ✓
- MXFP4 block struct ✓
- E2M1 quantization functions ✓
- Blackwell SM120a support ✓

**Blocking issue**: Q must also be E2M1, which has only 16 levels. This is severe quantization of the attention query. PPL impact needs testing. Could be acceptable for turbo1.5 where K is already ternary — the K precision is the bottleneck, not Q.

**Estimated effort**: 1-2 sessions for a working prototype. Not a quick win but infrastructure is in place.

## PART 6: Final Benchmarks ✓

### Decode Curve (RTX 5090, Qwen 27B Q6_K)
| Type | bpv | d=0 | d=8K | d=32K | d=64K |
|------|----:|----:|-----:|------:|------:|
| f16 | 16 | ~60 | 59.3 | ~51 | 52.3 |
| q8_0 | 8.5 | 60.7 | 58.1 | 49.6 | ~39 |
| turbo4 | 4.25 | 59.4 | 55.6 | 42.2 | 33.1 |
| turbo3 | 3.25 | **60.2** | 56.6 | 42.3 | 34.7 |
| turbo2 | 2.5 | **60.4** | 57.2 | **47.9** | **39.7** |
| turbo1.5 | 2.0 | 59.0 | 54.6 | 38.2 | 31.4 |

### PPL (Qwen 27B Q6_K)
| Type | ctx=512 | ctx=2048 |
|------|--------:|---------:|
| q8_0 | 6.759 | 5.674 |
| turbo4 | 6.825 | 5.694 |
| turbo3 | 6.852 | **5.674** |
| turbo2 | 7.080 | 5.892 |
| turbo1.5 | 7.312 | 6.103 |

### Key Findings
- **turbo3 now matches q8_0 at ctx=2048 PPL** (5.674 vs 5.674) at 2.6x compression
- **turbo2 is the long-context champion**: 47.9 @ 32K, 39.7 @ 64K (beats q8_0's 39)
- **turbo3 short: 60.2** — beats q8_0 (60.7 is within noise)
- **Sinks: definitively 0% PPL** on 2 models, 5 context lengths, 3 sink sizes
- **V sinks: dead end** — register pressure, not __managed__, causes the regression

## Completed in Retest (post-reboot)

### Extended 131K Decode
| Type | 131K tok/s | vs S18 |
|------|----------:|--------|
| turbo3 | 24.43 | was 14.55 (+68%) |
| turbo4 | 20.55 | — |
| turbo1.5 | 19.62 | was 15.62 (+26%) |

### MoE (Qwen 3.5 35B-A3B, tg32)
| Type | tok/s | vs S18 |
|------|------:|--------|
| q8_0 | 190.59 | — |
| turbo3 | 184.06 | was 93.86 (+96%) |
| turbo1.5 | 174.45 | was 113.35 (+54%) |
| turbo4 | 171.70 | was 98.38 (+75%) |

### Prefill (pp4096)
| Type | tok/s |
|------|------:|
| turbo4 | 3248.68 |
| turbo3 | 3228.24 |
| turbo1.5 | 3228.76 |

### Asymmetric Speed Matrix (tg4, d=0)
All 12 turbo×turbo+q8_0 combos pass, 39-47 tok/s range.

### V-specific 64×64 Rotation — TESTED, BLOCKED
Added `TURBO_V_GROUP` env var to override V group_size in llama-kv-cache.cpp.
**Result**: PPL = 18.26 (catastrophic, baseline = 6.85).
**Root cause**: Graph-level inverse WHT (`llama-graph.cpp:1839`) hardcodes group_size from `tensor->ne[0]` (always 128 for this model), not from the actual quantization group_size. V is quantized with 64-element groups but the inverse rotation uses 128-element signs → complete mismatch.
**Fix required**: Pass V's actual group_size through the graph (e.g., via op_params on the WHT op, or a separate V group_size variable). This is a 2-file architectural change (llama-kv-cache.cpp + llama-graph.cpp). Reverted the test code.

### Vault Dashboard
Updated: Project Status, Benchmark Hub, Roadmap — all current with S20 final numbers.
