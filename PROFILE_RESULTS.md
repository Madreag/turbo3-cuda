# Profiling Results — Session 17 Baseline
Date: 2026-03-28
Hardware: RTX 5090 32GB GDDR7, CUDA 12.8, SM120
Model: Qwen 3.5 27B Q6_K (opus-v2-Q6_K.gguf)
Build: 3380d3c72

## Baseline Numbers

| Config | Short tok/s | 8K tok/s | 32K tok/s | 64K tok/s | 128K tok/s | 204K tok/s | PPL 512 | PPL 2048 |
|--------|------------|---------|----------|----------|-----------|-----------|---------|----------|
| f16    | 65.44      | —       | —        | —        | —         | —         | —       | —        |
| q8_0   | 57.77      | —       | 46.51    | 39.24    | —         | —         | —       | —        |
| turbo3 | 56.68      | 52.60   | 26.86*   | 31.38    | 20.44     | 14.72     | 6.8380  | 5.6997   |
| turbo2 | 57.07      | —       | 44.56    | —        | —         | —         | —       | —        |

*turbo3 32K had ±23.27 variance (1 outlier run). Re-profiling with graphs disabled showed 31.90 (stable).

Prefill: turbo3 pp512=3156 t/s, pp4096=3121 t/s

## Kernel Time Breakdown — turbo3 decode (CUDA events, graphs disabled)

### Short Context (d=0, tg32)
| Kernel       | turbo3 ms | turbo3 % | q8_0 ms | q8_0 % | Delta     |
|-------------|-----------|----------|---------|--------|-----------|
| SET_ROWS    | 10.5      | 25.9%    | —       | —      | +10.5 ms  |
| WHT_forward | 4.9       | 12.1%    | —       | —      | +4.9 ms   |
| FA_VEC      | 19.6      | 48.7%    | 20.1    | 100%   | −0.5 ms   |
| WHT_inverse | 5.3       | 13.2%    | —       | —      | +5.3 ms   |
| **TOTAL**   | **40.3**  |          | **20.1**|        | **+20.2** |

### 8K Context (d=8192, tg32)
| Kernel       | turbo3 ms | turbo3 % | q8_0 ms | q8_0 % | Delta     |
|-------------|-----------|----------|---------|--------|-----------|
| SET_ROWS    | 17.6      | 8.6%     | —       | —      | +17.6 ms  |
| WHT_forward | 14.8      | 7.3%     | —       | —      | +14.8 ms  |
| FA_VEC      | 70.3      | 34.6%    | 52.3    | 37.3%  | +18.0 ms  |
| WHT_inverse | 14.0      | 6.9%     | —       | —      | +14.0 ms  |
| FA_MMA      | 86.5      | 42.6%    | 87.9    | 62.7%  | −1.4 ms   |
| **TOTAL**   | **203.0** |          | **140.2**|        | **+62.8** |

### 32K Context (d=32768, tg16)
| Kernel       | turbo3 ms | turbo3 % |
|-------------|-----------|----------|
| SET_ROWS    | 35.9      | 2.4%     |
| WHT_forward | 42.2      | 2.8%     |
| FA_VEC      | 121.4     | 8.1%     |
| WHT_inverse | 40.2      | 2.7%     |
| FA_MMA      | 1256.4    | 84.0%    |
| **TOTAL**   | **1496.1**|          |

## Analysis

- **Short context bottleneck**: SET_ROWS (25.9%) + WHT (25.3%) = 51.2% overhead. FA_VEC time identical to q8_0 (~20ms) — turbo3 compression isn't helping speed because model weights dominate bandwidth, not KV cache.
- **8K context**: FA dominates (77.2%). turbo3 FA_VEC is 34% SLOWER than q8_0 FA_VEC (70.3ms vs 52.3ms) — centroid dequant more expensive per-element. MMA prefill time identical.
- **32K context**: FA_MMA dominates (84%). SET_ROWS + WHT negligible. Decode-specific FA_VEC is only 8.1%.
- **turbo3 vs q8_0 FA_VEC gap**: turbo3 dequant (8-centroid lookup + norm multiply) costs ~34% more per-element than q8_0 dequant, but reads ~2.4x less data. Net effect depends on whether the kernel is compute-bound or memory-bound at each depth.

## Optimization Priorities (data-driven)

1. **LUT attention kernel** — Target: FA_VEC (48.7% at short, 34.6% at 8K). Replace per-element centroid×Q multiply with table lookup. Reduces ALU cost of turbo3 dequant. High impact at short-to-medium context.
2. **Attention sinks** — Free PPL win (−1.9% at ctx=2048), no speed impact. Highest ROI feature.
3. **turbo4 CUDA** — New capability (4.25 bpv, 16 centroids). Uses same kernel framework.
4. **turbo1.5 CUDA** — Unique type (2.0 bpv). Ternary dequant is SIMPLER than turbo3 — fewer ALU ops in FA_VEC.
5. **Layer-adaptive modes 3-11** — Configuration-only, no kernel changes needed.
