# Session 25 Deep Autoresearch Log

## Summary
- Iterations: 33/100
- Wins: 3 (committed)
- Dead ends: 27 (reverted), 3 info/baseline
- Best turbo3 short: 65.26 (+0.23% vs baseline 65.11)
- Best turbo3 32K: 56.97 (+4.9% vs baseline 54.31)
- Best turbo2 32K: 60.40 (+12.8% vs baseline 53.57)
- Best turbo1.5 32K: 49.80 (+3.4% vs baseline 48.15)
- All PPL bit-exact at tested contexts

## Key Insights

1. **Register budget: 168/170 = 98.4% utilized** — adding even 1 register causes spills and -2% to -4% regression.
2. **LUT is essential** — both for performance (-11% without) and REGISTER PRESSURE (LUT offloads data to shared memory, reducing register count from 255 to 168).
3. **Sparse V threshold was the only successful optimization** — constant changes that skip more V work at long context.
4. **VEC kernel is at optimization ceiling** — compiler flags, launch bounds, code restructuring, ILP changes, memory hints all neutral or regressive.
5. **SM120 specifics**: L2 prefetch IS needed (-2.3% without), HW prefetcher handles sequential but not strided patterns, shared memory bank conflicts are steeper.

## Baselines (Session 24B, RTX 5090, 27B Q6_K)

| Type | Short (tg128) | 32K (tg32) | PPL ctx=512 |
|------|:------------:|:----------:|:-----------:|
| turbo4 | 65.18 | 54.22 | 6.825 |
| turbo3 | 65.04 | 54.89 | 6.852 |
| turbo2 | 65.24 | 53.57 | 7.080 |
| turbo1.5 | 63.99 | 48.15 | 7.312 |
| q8_0 | 58.58 | 47.99 | 6.759 |

## Iteration Log

| # | Candidate | Change | Short | 32K | PPL | Result | Commit |
|---|-----------|--------|:-----:|:---:|:---:|--------|--------|
| 1 | --use_fast_math | cmake flag | 64.61 (-0.77%) | 54.67 (+0.66%) | — | DEAD | — |
| 2 | --maxrregcount=128 | cmake flag | 64.95 (-0.25%) | 54.74 (+0.79%) | — | DEAD | — |
| 3 | --ptxas-options=-v | diagnostic | — | — | — | INFO | — |
| 3a | Register count | D=128 ncols=1: 167-168 regs, 6912B smem. 98.4% reg util. | — | — | — | INFO | — |
| 4 | -dlcm=cg (L1 bypass) | cmake flag | 63.43 (-2.58%) | — | — | DEAD | — |
| 5 | launch_bounds(128,2) | less occupancy, more regs | 63.44 (-2.57%) | — | — | DEAD | — |
| 6 | Hoist sink vars to regs | 3 extra regs → spills | 62.57 (-3.9%) | — | — | DEAD | — |
| 7 | Binary tree LUT reduce | 4 temp regs → spills | 63.20 (-2.93%) | — | — | DEAD | — |
| 8 | Serial FMA LUT accum | ILP destroyed (depth 8 vs 5) | 62.85 (-3.47%) | — | — | DEAD | — |
| 9 | Disable turbo3 LUT | Force q8_1 vec_dot | 62.52 (-3.97%) | 48.28 (-11.1%) | — | DEAD | — |
| 9a | Register count no-LUT | D128 ncols1: 128 regs → MASSIVE spilling. LUT essential for reg pressure | — | — | — | INFO | — |
| 10 | Sparse V threshold 1e-4 | constant change | 65.17 (+0.09%) | 54.91 (+1.1%) | 6.8522 | TEST | — |
| 10b | Sparse V threshold 1e-3 | constant change | 65.26 (+0.23%) | 56.95 (+4.86%) | 6.8522 | **WIN** | dfa84f6b6 |
| 10c | Sparse V threshold 1e-2 | too aggressive | 65.38 (+0.41%) | 58.74 (+8.2%) | 6.8522 | RISKY | — |
| 11 | Prefetch 2 chunks ahead | 2 extra asm prefetch | 65.22 (=) | 57.02 (+0.12%) | — | DEAD | — |
| 12 | Remove L2 prefetch | HW-only test | — | 55.63 (-2.32%) | — | DEAD | — |
| 13 | __ldg for K reads | texture cache | 65.30 (=) | 57.03 (=) | — | DEAD | — |
| 14 | Float Q for turbo types | remove q8_1 overhead | 64.81 (-0.69%) | 55.46 (-2.62%) | — | DEAD | — |
| 15 | #pragma unroll 1 LUT loop | less code, less ILP | 64.05 (-1.86%) | — | — | DEAD | — |
| — | turbo2 32K verify | 1e-3 threshold benefit | — | 58.60 (+9.4%) | — | WIN | dfa84f6b6 |
| — | turbo1.5 32K verify | 1e-3 threshold benefit | — | 49.80 (+3.4%) | — | WIN | dfa84f6b6 |
| — | turbo4 32K verify | neutral | — | 54.15 (=) | — | OK | dfa84f6b6 |
| 16 | #pragma unroll 4 outer KQ | smaller code | 65.33 (=) | 57.33 (noise) | — | DEAD | — |
| 17 | Type-specific sparse V 1e-2 | turbo2/1.5 low bpv | — | turbo2 60.40 (+12.8%) | 7.0797 | **WIN** | 3d609e224 |
| 18 | Warp-level V tile skip | __shfl max + goto | 65.27 (=) | 56.97 (=) | — | DEAD | — |
| 19 | nthreads_KQ=16 | -10 regs but only 2 dots/warp | 60.54 (-7.2%) | 53.07 (-6.9%) | — | DEAD | — |
| 20 | launch_bounds(128,4) | 33% occupancy, 40 spills | — | 53.82 (-5.5%) | — | DEAD | — |
| 21 | Symmetric LUT construction | if constexpr ruins reg alloc | 60.79 (-6.8%) | — | — | DEAD | — |
| 22 | turbo4 V at 1e-2 | rename caused codegen change | — | regression | — | DEAD | — |
| 23 | turbo3/4 V at 5e-3 | constant change | — | 54.80 (+6.4%) | 6.8522 | **WIN** | 8e27f54bc |
| 24 | turbo3 V at 1e-2 | saturated at 5e-3 | — | 54.30 (=5e-3) | — | DEAD | — |
| 25 | turbo2 V at 5e-2 | marginal (+1.4%), noisy | — | 56.26 (noise) | 7.0797 | DEAD | — |
| 26 | turbo3 V at 1e-6 clean build | baseline for clean build | — | 49.45 | — | INFO | — |
| 27 | turbo4 32K clean build | clean baseline | — | 48.29 | — | INFO | — |
| 28 | turbo1.5 threshold=0 | ternary V too cheap to benefit | — | 44.66 (=1e-2) | — | DEAD | — |
| 29 | --Xptxas --opt-level=4 | slightly better 32K, worse short | 59.15 (-0.9%) | 52.28 (+1.5%) | — | DEAD | — |
| 30 | --allow-expensive-optimizations | mixed: -2.2% short, +2.1% 32K | 58.36 (-2.2%) | 52.56 (+2.1%) | — | DEAD | — |
| 31 | --ftz=true | mixed: -1.9% short, +1.8% 32K | 58.56 (-1.9%) | 52.43 (+1.8%) | — | DEAD | — |

| 32 | --prec-div=false | -1.6% short, -0.8% 32K | 58.76 (-1.6%) | 51.08 (-0.8%) | — | DEAD | — |
| 33 | 2×4 LUT split (no temps) | neutral in noise | 59.29 (=) | 52.34 (=) | — | DEAD | — |

### Pattern: ptxas flags help 32K but hurt short

Iterations 29-32 all show the same pattern: ptxas optimization flags that change codegen hurt short context by 1-2% but help 32K by 1.5-2%. The default compiler favors latency (good for short), while aggressive flags favor throughput (good for 32K). Cannot resolve with global flags — would need per-kernel optimization.

### Pattern: Code restructuring is register-sensitive

Iterations 5-9, 13-15, 18-21, 33 show that ANY code change to the VEC kernel that adds or rearranges registers causes regression. At 168/170 registers (98.4%), the compiler has already found a near-optimal allocation. Any perturbation tips the balance.
