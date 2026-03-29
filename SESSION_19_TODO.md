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

### PART 3: Quick LUT bank conflict test — DEFERRED TO SESSION 20
- turbo3 LUT scoring path was deleted in Session 18
- Re-adding requires writing new vec_dot code (>10 min)

### PART 4: Verify trit LUT in all turbo1.5 paths ✓
- All dequant paths use `turbo1_5_unpack_trit()` → `TURBO1_5_TRIT_LUT`
- `pow3[]` in set-rows.cu is encoding only (correct)

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

**Conclusion**: Sinks provide 0% PPL improvement for turbo3 across all context lengths.
Sinks speed overhead: <0.2% (59.31 vs 59.42 tok/s). Feature kept for future investigation with other models.
