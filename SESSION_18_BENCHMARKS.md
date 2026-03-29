# Session 18 Final Benchmarks

**Model**: Qwen 3.5 27B Q6_K (opus-v2-Q6_K.gguf, 20.56 GiB)
**Hardware**: RTX 5090 32GB GDDR7, SM120, CUDA 12.8, WSL2
**Date**: 2026-03-29

## Dense Model — Short Decode (d=0, tg128) — FINAL (post-regression-fix)

| Type     | bpv  | tok/s      | vs f16 |
|----------|-----:|------------|--------|
| f16      | 16.0 | 59.00      | 100%   |
| q8_0     |  8.5 | 59.53      | 100.9% |
| turbo4   | 4.25 | 59.06      | 100.1% |
| turbo2   | 2.50 | 58.88      |  99.8% |
| turbo3   | 3.25 | 57.99      |  98.3% |
| turbo1.5 | 2.00 | 57.36      |  97.2% |

## Dense Model — 32K Decode (d=32768, tg32) — FINAL

| Type     | bpv  | tok/s      | vs q8_0 |
|----------|-----:|------------|---------|
| q8_0     |  8.5 | 48.03      | 100%    |
| turbo4   | 4.25 | 46.51      |  96.8%  |
| turbo1.5 | 2.00 | 44.49      |  92.6%  |
| turbo3   | 3.25 | 40.90      |  85.2%  |

## Dense Model — 64K Decode (d=65536, tg16)

| Type     | tok/s |
|----------|-------|
| turbo4   | 35.93 |
| turbo1.5 | 35.29 |
| turbo3   | 27.20 |

## Dense Model — 128K Decode (d=131072, tg8)

| Type     | tok/s |
|----------|-------|
| turbo1.5 | 15.62 |
| turbo3   | 14.55 |

## Dense Model — 204K Decode (d=204800, tg4)

| Type     | tok/s |
|----------|-------|
| turbo1.5 | 15.26 |
| turbo3   |  8.42 |

## Dense Model — 8K Decode (d=8192, tg128)

| Type     | tok/s |
|----------|-------|
| q8_0     | 56.69 |
| turbo4   | 55.31 |
| turbo1.5 | 54.99 |
| turbo3   | 50.75 |

## Perplexity (Qwen 3.5 27B, wikitext-2)

| Type     | bpv  | PPL ctx=512 | vs q8_0 | PPL ctx=2048 |
|----------|-----:|-------------|---------|-------------|
| q8_0     |  8.5 | 6.7590      | —       | —           |
| turbo4   | 4.25 | 6.8249      | +0.97%  | 5.6937      |
| turbo3   | 3.25 | 6.8522      | +1.38%  | 5.6744      |
| turbo2   | 2.50 | 7.0797      | +4.75%  | —           |
| turbo1.5 | 2.00 | 7.3120      | +8.18%  | 6.1028      |

## MoE Model — Qwen 3.5 35B-A3B Q4_K_M (tg32)

| Type     | tok/s |
|----------|-------|
| turbo1.5 | 113.35 |
| turbo4   |  98.38 |
| turbo3   |  93.86 |

## Prefill (pp4096)

| Type     | tok/s   |
|----------|---------|
| turbo1.5 | 3219.69 |
| turbo2   | 3196.37 |
| turbo3   | 3176.07 |
| turbo4   | 3168.22 |

## Asymmetric (K turbo, V q8_0, tg128)

| K/V          | tok/s |
|--------------|-------|
| turbo3/q8_0  | 57.49 |
| turbo4/q8_0  | 57.70 |

## Key Insight: Q Format Dominates Long-Context Performance

| Q Format   | Types          | Short d=0 | Long d=32K | Bottleneck |
|------------|----------------|-----------|------------|------------|
| float Q    | turbo3, turbo2 | Best (LUT) | Worst | 4x Q bandwidth + 32KB shared mem |
| q8_1 Q     | turbo4, turbo1.5 | Good | Best | Minimal overhead |

turbo1.5 at 204K: 15.26 tok/s (1.8x turbo3's 8.42) — the q8_1 Q path
and 2.0 bpv makes turbo1.5 the best choice for extreme-context workloads.

## Regression Fix: V Sink Managed Memory + turbo3 LUT

Two performance regressions were found and fixed during review:

1. **V sink managed memory reads in V loop** (-3% short, -14% 32K)
   - The `use_v_sink` check read `d_fattn_sink_n` (__managed__) on every V iteration
   - UVM page migration overhead scaled with context length
   - Fix: removed V sink from V accumulation loop (sinks show 0% PPL benefit)

2. **turbo3 LUT attention bank conflicts** (-1.8% short decode)
   - 8-centroid LUT had 2-way shared memory bank conflicts (stride aliasing)
   - Fix: disabled turbo3 LUT, kept turbo2 LUT (4 centroids, no bank conflicts)

## Session 18 Changes Summary

1. **V attention sinks**: Fixed fundamental ne0 mismatch bug in sink lookup.
   Sinks now correctly read captured data with head offset indexing.
   PPL improvement negligible — turbo3 precision already sufficient.

2. **Trit LUT for turbo1.5**: +6.8% decode speed (54.91→58.66 tok/s).
   Precomputed 5×256 table eliminates integer division (~100 cycles → ~4 cycles).

3. **LUT for turbo4/turbo1.5**: Investigated, found incompatible with q8_1 Q format.
   LUT attention requires float Q path; turbo4/turbo1.5 use q8_1 for bandwidth efficiency.

4. **ISWA WHT audit**: All 5 build_attn overloads verified. WHT rotation PRESENT
   in kv (L2095), k/MLA (L2189), and kv_iswa (L2265).

5. **rsqrtf optimization**: Applied to all 4 turbo SET_ROWS kernels + sink capture.
   Single SFU instruction replaces sqrtf+fdiv. PPL unchanged within noise.

6. **MoE validation**: turbo1.5 at 113 tok/s on 35B-A3B model.
