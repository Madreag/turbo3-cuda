# Session 17 Final Results — TurboQuant CUDA on RTX 5090
Date: 2026-03-28/29
Hardware: RTX 5090 32GB GDDR7, CUDA 12.8, SM120
Model: Qwen 3.5 27B Q6_K (opus-v2-Q6_K.gguf)
Branch: session/17-beat-signalnine

## Decode Speed (tok/s)

| Config | bpv | Short tg128 | 8K tg128 | 32K tg128 | 128K tg8 | 204K tg4 | Compression |
|--------|----:|:-----------:|:--------:|:---------:|:--------:|:--------:|:-----------:|
| f16    | 16  | 65.44       | —        | —         | —        | —        | 1.0x        |
| q8_0   | 8.5 | 57.15       | 55.21    | 46.99     | OOM      | OOM      | 1.9x        |
| **turbo4** | **4.25** | **64.15** | **54.20** | **49.64** | —    | —        | **3.8x**    |
| turbo3+LUT | 3.25 | **60.77** | 58.65  | 41.26     | 21.95    | 16.22    | 4.6x        |
| turbo2 | 2.5  | 56.44       | 53.69    | 44.29     | —        | —        | 6.4x        |
| turbo1.5 | 2.0 | 54.03      | 52.84    | 36.19     | 15.14    | —        | **8.0x**    |

## Perplexity (wikitext-2-raw, 8 chunks)

| Config | bpv | PPL ctx=512 | PPL ctx=2048 | Delta vs q8_0 |
|--------|----:|:-----------:|:------------:|:-------------:|
| **turbo4** | **4.25** | **6.8145** | **5.6825** | ~+0.5% |
| turbo3 | 3.25 | 6.8380 | 5.6997 | ~+0.9% |
| turbo2 | 2.5 | 7.0841 | 5.9012 | ~+4.5% |
| turbo1.5 | 2.0 | 7.3218 | 6.0910 | ~+8.6% |

## Key Achievements

1. **turbo4 beats q8_0 speed** at 32K: 49.64 vs 46.99 tok/s (+5.6%) with 3.8x compression
2. **turbo3 LUT kernel**: +7% decode speed (56.68 → 60.77 tok/s) via shared memory Q×centroid table
3. **turbo1.5 is live**: Nobody else has sub-2.5 bpv. 8x compression, 54 tok/s decode, fits 128K in 32GB
4. **All turbo types**: 1.5/2/3/4 with full CUDA (SET_ROWS, vec_dot, V dequant, MMA prefill)
5. **Zero PPL regressions** across all changes

## Features

- turbo4 full CUDA (parallel SET_ROWS, 16-centroid, nibble packing)
- turbo1.5 full CUDA (parallel SET_ROWS, ternary, trit packing)
- LUT attention kernel (shared memory Q×centroid table for turbo3)
- Layer-adaptive modes 3-11 (asymmetric K/V promotion)
- Attention sinks infrastructure (TURBO_SINK_SIZE, K-only)
- Bug fixes: getrows, cpy, convert for all turbo types

## Kernel Profile (CUDA events, short context)

| Kernel | turbo3 % |
|--------|----------|
| FA_VEC | 48.7% |
| SET_ROWS | 25.9% |
| WHT_forward | 12.1% |
| WHT_inverse | 13.2% |
