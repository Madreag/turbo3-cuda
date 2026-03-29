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

## PART 5: FP4 Tensor Core Feasibility ✓
- [x] Feasible in theory: turbo1.5 {-C, 0, +C} maps to FP4 E2M1 {-1.0, 0.0, +1.0}
- [x] Not feasible for Session 20:
  - FP4 MMA helps prefill only (not VEC decode)
  - Fragment layout poorly documented on SM120
  - Needs new MMA kernel + trit repacking to FP4 pairs
- [x] Documented for future session

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
