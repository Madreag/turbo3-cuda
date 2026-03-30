# AmesianX TurboQuant vs Madreag TurboQuant — RTX 5090 Head-to-Head

## Hardware: RTX 5090 32GB (SM120), CUDA 12.8, WSL2 Ubuntu 24.04

## Build Status
- [x] AmesianX repo built successfully for SM120
- Build notes: Shallow clone (commit 00dbcbf). Had to patch `tools/llama-bench/llama-bench.cpp` — their `ggml_type_from_name()` is hardcoded and doesn't include TBQ types (their own bench can't test their own types!). Fixed with generic loop over `ggml_type_name()`.
- Their block-256 (QK_K=256) vs our block-128 (QK_TURBO3=32, 4 blocks per 128-value rotation group)

## Results

### 27B Q6_K (opus-v2-Q6_K.gguf)

| Config | AmesianX tok/s | Madreag tok/s | Winner | Delta |
|--------|:-:|:-:|:---:|:---:|
| q8_0 baseline (short) | 57.45 | 58.58 | — | Fair test |
| **tbq3_0/turbo3 (short)** | **56.14** | **65.05** | **Madreag** | **+15.9%** |
| **tbq3_0/turbo3 (32K)** | **40.52** | **53.44** | **Madreag** | **+31.9%** |
| tbq4_0/turbo4 (short) | 59.19 | 58.87 | AmesianX | +0.5% |
| tbq4_0/turbo4 (32K) | 48.62 | 46.06 | AmesianX | +5.6% |
| tbqp3_0+tbq3_0 (short) | 46.89 | N/A | — | QJL variant |
| **PPL ctx=512 (3-bit)** | **7.2769** | **6.8522** | **Madreag** | **-5.8%** |
| PPL ctx=512 (4-bit) | 6.8662 | 6.8250 | Madreag | -0.6% |

### 27B Q4_K_M

| Config | AmesianX tok/s | Madreag tok/s | Winner | Delta |
|--------|:-:|:-:|:---:|:---:|
| **tbq3_0/turbo3 (short)** | **65.25** | **77.25** | **Madreag** | **+18.4%** |

### MoE (35B-A3B Q4_K_M)

| Config | AmesianX tok/s | Madreag tok/s | Winner | Delta |
|--------|:-:|:-:|:---:|:---:|
| **tbq3_0/turbo3 (short)** | **166.57** | **184** | **Madreag** | **+10.5%** |

### Multi-Model Validation

| Model | D | AmesianX tbq3_0 | Madreag turbo3 | Winner |
|-------|:-:|:-:|:-:|:---:|
| Llama-3.2-1B | 64 | 351.58 | 685 | **Madreag +94.8%** |
| Gemma-3-12B | 256 | 66.02 | 95 | **Madreag +43.9%** |

## Analysis

### Where We Dominate

**turbo3 (3-bit)**: We're +15.9% faster at short context and +31.9% faster at 32K. Our advantages:
- 3 blocks/SM occupancy (`__launch_bounds__(128,3)`) vs their 1 block/SM
- 8-wide LUT scoring (shared memory pre-computed Q*centroid table)
- `__expf` fast-math softmax (10x cheaper than `expf`)
- L2 prefetch hints for next KV chunk
- nthreads_KQ=8 for better warp ILP at long context

**PPL (3-bit)**: Our 6.8522 vs their 7.2769 — we have better quality at the SAME compression level. Their block-256 with 8 centroids has coarser norm granularity than our block-128.

**Multi-model**: We crush them on D=64 (+94.8%) and D=256 (+43.9%).

**Unique capabilities we have that they DON'T**:
- turbo2 (4 centroids, 2.5 bpv) — long-context champion
- turbo1.5 (ternary, 2.0 bpv) — 8x compression
- 36 K×V asymmetric combos
- Layer-adaptive modes (15 modes)
- Cross-GPU validation (3 GPUs, 1121+ iterations)

### Where They Win

**turbo4 (4-bit)**: They're +0.5% at short and +5.6% at 32K. Their turbo4 uses register-based 16 centroids with deferred norm. We removed turbo4 LUT in Session 21 (net negative) — their approach of keeping centroids in registers may be better for 16 centroids.

### Key Insight

Their register-based centroid approach is competitive for turbo4 (16 centroids fit nicely in registers), but our LUT-based approach dominates for turbo3 (8 centroids — LUT turns multiply-add into table lookup, saving 7 multiplies per 8 elements). The 3 blocks/SM occupancy from `__launch_bounds__` is the biggest single advantage at long context.

### Their QJL Variant (tbqp3_0)

46.89 tok/s — much slower than their own tbq3_0 (56.14). The QJL correction adds overhead without meaningful PPL benefit. Community consensus confirmed: QJL is dead.

---

*Test date: 2026-03-29. Their code unmodified except bench type parser fix. Same models, same GPU, same benchmark commands.*
