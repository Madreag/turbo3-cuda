# SESSION 19 — Bug Fixes: Sinks Crash, Asymmetric Crashes, Cross-Type Dispatch

## Status: COMPLETE

### PART 1: Fix attention sinks SM86 crash ✓
- [x] Replace `__managed__` → `__device__` + `cudaGetSymbolAddress` + `cudaMemcpyAsync`
- [x] Remove `cudaStreamSynchronize` from dispatch
- [x] Extend sink capture to all turbo types (turbo2/turbo4/turbo1.5)
- [x] Extend kernel-side sink check to all turbo types (was turbo2-only)
- [x] Test: sinks=4 with CUDA graphs enabled — 59.31 tok/s, no crash
- [x] Test: sinks=0 no regression — 59.42 tok/s

### PART 2: Fix ALL asymmetric K/V crashes + 6 tok/s regression ✓
- [x] Created 20 new template instance files (turbo×turbo, turbo×f16, f16×turbo)
- [x] Fixed turbo4_0 and q8_0-turbo4_0 missing D=64
- [x] Added extern declarations for all cross-type combos
- [x] Added FATTN_VEC_CASES_ALL_D dispatch for all cross-type combos
- [x] Fixed turbo_q8_mix guard → allows all turbo cross-type combos
- [x] Fixed turbo4 D constraint: %128 → %64
- [x] All 36 K×V combinations verified OK on SM120
- [x] K=turbo3/V=turbo4: 59.20 tok/s (was 6 tok/s CPU fallback)

### PART 3: LUT Bank Conflict — DEFERRED TO SESSION 20 (documented)

**Current state**: LUT infrastructure exists but NO scoring path reads it.

- `fattn-vec.cuh:265`: `n_centroids_lut = (type_K == TURBO2_0) ? 4 : 0` — turbo3 disabled
- `fattn-vec.cuh:268-281`: Shared memory LUT built: `turbo_lut[D][n_centroids_lut]`
- `fattn-common.cuh:358-402`: `vec_dot_fattn_vec_KQ_turbo2_0` reads from `TURBO_CENTROIDS_2BIT` directly, NOT from LUT — **turbo2 LUT is also dead code**

**What Session 20 needs to implement:**
1. New `vec_dot_fattn_vec_KQ_turbo3_0_lut()` in `fattn-common.cuh` (~line 300) that reads from shared memory LUT instead of centroid lookup
2. Pass `turbo_lut` to the scoring loop in `fattn-vec.cuh` (lines 306-321) via constexpr dispatch
3. Bank conflict fix: `turbo_lut[D][8+1]` padding (add 1 float per row to stagger bank addresses)
4. Wire turbo2 LUT scoring too (currently dead code despite n_centroids_lut=4)
5. Measure: turbo3 LUT should recover +7% (56.68→60.77 in Session 17)

### PART 4: Verify trit LUT in all turbo1.5 paths ✓

**All dequant paths verified — LUT used everywhere:**
| Path | File:Line | Chain |
|------|-----------|-------|
| dequantize.cuh | 107-111 | `dequantize_turbo1_5` → `turbo1_5_dequant_element` → `turbo1_5_unpack_trit` → `TURBO1_5_TRIT_LUT` |
| fattn-common.cuh | 542-543, 572-575, 592-599 | Direct `turbo1_5_dequant_element` calls |
| convert.cu | 767, 828, 859, 909 | Via `dequantize_turbo1_5` |
| getrows.cu | 215 | Via `dequantize_turbo1_5` |
| set-rows.cu | 1253, 1379 | `pow3[]` — ENCODING only (trit packing), correct |

No integer division dequant path (`packed / pow3[pos] % 3`) remains in any hot path.

### MMA/TILE Pre-Dequant Verification ✓

Both MMA (`fattn-mma-f16.cuh:1775`) and TILE (`fattn-tile.cuh:1108`) pass `need_f16_K=true, need_f16_V=true` to `launch_fattn`, which calls `ggml_get_to_fp16_cuda()`.

All turbo types verified in convert.cu:
- `ggml_get_to_fp16_cuda`: TURBO4_0 (764), TURBO1_5 (766), TURBO3_0 (761), TURBO2_0 (762)
- `ggml_get_to_fp16_nc_cuda`: TURBO4_0 (856), TURBO1_5 (858)
- `ggml_get_to_fp32_cuda`: TURBO4_0 (825), TURBO1_5 (827)

Prefill tested (pp512) for all cross-type combos — all OK at 2700-3300 tok/s.

### GET_ROWS and Convert Dispatch Verification ✓
- `QR_TURBO4` defined (dequantize.cuh:116), `QR_TURBO1_5` (dequantize.cuh:114)
- `getrows.cu` TURBO4_0 (210), TURBO1_5 (214)
- `convert.cu`: to_fp16, to_fp16_nc, to_fp32 all handle all 4 turbo types

### PART 5: Final benchmarks ✓

#### Symmetric Short Decode (RTX 5090)
| Type | tok/s | vs S18 |
|------|------:|--------|
| f16 | 60.72 | 59.09 (+2.8%) |
| turbo4 | 58.19 | 59.06 (-1.5%) |
| turbo3 | 59.36 | 57.99 (+2.4%) |
| turbo2 | 59.65 | 58.88 (+1.3%) |
| turbo1.5 | 57.83 | 57.36 (+0.8%) |

#### 32K Decode
| Type | tok/s | vs S18 |
|------|------:|--------|
| q8_0 | 48.64 | 46.42 (+4.8%) |
| turbo4 | 41.45 | 45.96 (-9.8%) |
| turbo3 | 42.04 | 40.90 (+2.8%) |
| turbo1.5 | 40.46 | 44.57 (-9.2%) |

#### PPL (ctx=512, 8 chunks)
| Type | PPL | vs S18 |
|------|----:|--------|
| turbo3 | 6.8522 | 6.852 (=) |
| turbo4 | 6.8249 | 6.825 (=) |
| turbo1.5 | 7.3120 | 7.312 (=) |

#### Sink PPL Sweep (turbo3, chunks=4)
| Context | Sinks=0 | Sinks=4 | Delta |
|---------|---------|---------|-------|
| 512 | 6.1335 | 6.1335 | 0% |
| 2048 | 7.3726 | 7.3726 | 0% |
| 4096 | 6.2622 | 6.2622 | 0% |

#### Cross-Type Prefill (pp512, MMA/TILE path)
| K | V | tok/s |
|---|---|------:|
| turbo4 | turbo3 | 3184 |
| turbo4 | f16 | 3012 |
| turbo4 | q8_0 | 3109 |
| turbo1.5 | turbo3 | 3140 |
| turbo1.5 | f16 | 3130 |
| turbo1.5 | q8_0 | 3324 |
| f16 | turbo4 | 3258 |
| f16 | turbo1.5 | 3234 |
| q8_0 | turbo4 | 3200 |
| q8_0 | turbo1.5 | 3238 |

**Conclusion**: Sinks provide 0% PPL improvement for turbo3 across all context lengths.
Sinks speed overhead: <0.2% (59.31 vs 59.42 tok/s). Feature kept for future investigation with other models.
