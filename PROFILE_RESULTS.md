# Session 17 Results — TurboQuant CUDA on RTX 5090
Date: 2026-03-28/29
Hardware: RTX 5090 32GB GDDR7, CUDA 12.8, SM120
Model: Qwen 3.5 27B Q6_K (opus-v2-Q6_K.gguf)
Build: session/17-beat-signalnine

## Decode Speed (tok/s, tg128 unless noted)

| Config | bpv | Short | 8K | 32K | 64K | 128K | 204K | Compression |
|--------|----:|------:|---:|----:|----:|-----:|-----:|------------:|
| f16    | 16  | 65.44 | —  | —   | —   | —    | —    | 1.0x        |
| q8_0   | 8.5 | 57.92 | 55.02 | 46.88 | 39.96 | OOM | OOM | 1.9x |
| **turbo4** | **4.25** | **57.57** | **54.15** | **45.49** | **37.63** | — | — | **3.8x** |
| turbo3 | 3.25 | 56.32 | 58.81* | 41.41 | 35.03 | 21.95 | 16.22 | 4.6x |
| turbo2 | 2.5  | 64.48* | 53.89 | 48.12* | 35.95 | — | — | 6.4x |

*High variance runs (±10+), first-run warmup effect

## Perplexity (wikitext-2-raw, 8 chunks)

| Config | bpv | PPL ctx=512 | PPL ctx=2048 | Delta vs q8_0 |
|--------|----:|:-----------:|:------------:|:-------------:|
| **turbo4** | **4.25** | **6.8145** | **5.6825** | ~+0.5% |
| turbo3 | 3.25 | 6.8380 | 5.6997 | ~+0.9% |
| turbo2 | 2.5 | 7.0841 | 5.9012 | ~+4.5% |

PPL reject thresholds: turbo3 ctx512 < 6.89 ✓, ctx2048 < 5.80 ✓

## Prefill (tok/s)

| Config | pp4096 |
|--------|-------:|
| turbo3 | 3088   |
| turbo4 | 3091   |

## Kernel Time Breakdown (CUDA events, graphs disabled)

### Short Context (d=0, tg32)
| Kernel       | turbo3 % | q8_0 % |
|-------------|----------|--------|
| SET_ROWS    | 25.9%    | —      |
| WHT_forward | 12.1%    | —      |
| FA_VEC      | 48.7%    | 100%   |
| WHT_inverse | 13.2%    | —      |

### 32K Context (d=32768, tg16)
| Kernel       | turbo3 % |
|-------------|----------|
| FA_MMA      | 84.0%    |
| FA_VEC      | 8.1%     |
| SET_ROWS    | 2.4%     |
| WHT         | 5.5%     |

## Features Implemented

1. **turbo4 Full CUDA** — 4-bit PolarQuant (16 centroids, 3.8x compression)
   - Parallel SET_ROWS (128 threads/group, warp-cooperative nibble packing)
   - Native vec_dot + V dequant in FA VEC kernel
   - MMA prefill via launch_fattn pre-dequant to fp16
   - Quality: PPL +0.5% vs q8_0, BETTER than turbo3
   - Speed: matches q8_0 at all depths

2. **LUT Attention Kernel** — precomputed Q×centroid table in shared memory
   - turbo3 decode short: +7% speedup (56.68 → 60.62 tok/s)
   - 8 KB shared memory for D=256, 4 KB for D=128
   - Zero PPL change (mathematically equivalent)

3. **Layer-Adaptive Modes 3-11** — 9 additional KV type promotion strategies
   - Modes 6-8: asymmetric K/V promotion (V-only or K-only to q8_0)
   - LA-3 (last 4 q8_0): PPL -0.35% improvement

4. **Attention Sinks** (TURBO_SINK_SIZE) — infrastructure for fp16 sink positions
   - K-only sinks captured during SET_ROWS, used in FA VEC decode
   - Requires GGML_CUDA_DISABLE_GRAPHS=1

5. **Bug Fixes** from FIXESWEMUSTLOOKFOR audit:
   - turbo3/2/4 added to getrows.cu (was missing — latent crash)
   - turbo4 added to convert.cu fp16/fp32/nc dequant
   - turbo2/3/4 same-type copy added to cpy.cu
   - CPU turbo4 quantizer: corrected_norm no longer overwritten
   - TURBO4_USE_4BIT=1 default on all platforms
   - QR_TURBO4=1 (was incorrectly 2)
