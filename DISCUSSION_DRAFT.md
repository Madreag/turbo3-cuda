# TurboQuant CUDA: turbo3 matches q8_0 perplexity at 4.6x KV compression

**Branch**: [`Madreag/turbo3-cuda` `session/22-polish-and-ship`](https://github.com/Madreag/turbo3-cuda/tree/session/22-polish-and-ship)

## Summary

Full CUDA implementation of all 4 TurboQuant types (turbo4/turbo3/turbo2/turbo1.5) for llama.cpp with native Flash Attention decode, parallel SET_ROWS encode, and MMA/TILE prefill. Built on top of signalnine's turbo3 pre-rotate-queries architecture.

**Key results on RTX 5090, Qwen 3.5 27B Q6_K:**

| Type | bpv | Compression | Short tok/s | 32K tok/s | PPL ctx=2048 |
|------|----:|------------:|------:|----:|:---:|
| q8_0 | 8.5 | 1.9x | 58.58 | 47.99 | 5.674 |
| **turbo3** | **3.25** | **4.6x** | **60.18** | **48.44** | **5.674 (=q8_0)** |
| **turbo2** | **2.5** | **6.4x** | **60.67** | **51.29** | 5.892 |
| turbo1.5 | 2.0 | 8.0x | 58.74 | 45.06 | 6.103 |

- **turbo3 = q8_0 perplexity at ctx=2048** (5.674 = 5.674) while using 4.6x less KV memory
- **turbo2 beats q8_0 decode speed at 32K** (51.29 vs 47.99 tok/s) while using 6.4x less memory
- **MoE**: turbo3 = 184 tok/s on Qwen 3.5 35B-A3B (+96% vs signalnine's turbo3 base)

## What We Added Over signalnine's turbo3 Base

1. **turbo4** (4.25 bpv, 16 centroids) — full CUDA: SET_ROWS, vec_dot, V dequant, template instances
2. **turbo1.5** (2.0 bpv, ternary + trit LUT) — 8x compression, 174 tok/s on MoE
3. **turbo2** (2.5 bpv, 4 centroids) — long-context champion, beats q8_0 at 32K on all 3 GPUs
4. **q8_1 vec_dot for turbo3/turbo2** — eliminated 4x float Q bandwidth penalty
5. **LUT scoring with bank conflict fix** — `[D][n+1]` padding (stride coprime to 32 banks), 4-at-a-time scoring. turbo3 32K +7%
6. **`__launch_bounds__(128, 3)`** — forced 3 blocks/SM on SM120, registers 180-194→166-168, +7-13% at 32K
7. **20 cross-type VEC template instances** — all 36 K×V asymmetric combinations work
8. **Attention sinks** — `__device__` + async copy, graph-compatible (though 0% PPL benefit in practice)

## Performance Table

### RTX 5090, Qwen 3.5 27B Q6_K

| Type | bpv | Compression | Short | 32K | PPL ctx=512 | PPL ctx=2048 |
|------|----:|------------:|------:|----:|:-----------:|:------------:|
| f16 | 16 | 1x | 59.52 | 54.11 | — | — |
| q8_0 | 8.5 | 1.9x | 58.58 | 47.99 | 6.759 | 5.674 |
| turbo4 | 4.25 | 3.8x | 58.87 | 46.06 | 6.825 (+0.97%) | 5.694 |
| turbo3 | 3.25 | 4.6x | 60.18 | 48.44 | 6.852 (+1.38%) | **5.674 (=q8_0)** |
| turbo2 | 2.5 | 6.4x | 60.67 | **51.29** | 7.080 (+4.75%) | 5.892 |
| turbo1.5 | 2.0 | 8.0x | 58.74 | 45.06 | 7.312 (+8.18%) | 6.103 |

### MoE (Qwen 3.5 35B-A3B Q4_K_M, tg32)

| Type | tok/s |
|------|------:|
| q8_0 | 190.59 |
| turbo3 | **184.06** |
| turbo1.5 | 174.45 |
| turbo4 | 171.70 |

## Multi-Model Validation (5 models, 3 head dimensions)

| Model | Params | D | GQA | All 4 turbo types? | Asymmetric | Prefill |
|-------|-------:|:-:|:---:|:---:|:---:|:---:|
| Llama-3.2-1B | 1.24B | 64 | 4:1 | PASS | PASS | PASS |
| Phi-3.5-mini | 3.82B | 96 | 1:1 | FALLBACK* | N/A | N/A |
| Phi-4-mini | 3.84B | 128 | 3:1 | PASS | PASS | PASS |
| Llama-3.3-8B | 8.03B | 128 | 4:1 | PASS | PASS | PASS |
| Gemma-3-12B | 12.2B | 256 | 2:1 | PASS | PASS | PASS |

\* D=96: VEC FA kernel requires `D % 64 == 0`. D=96 gracefully falls back to non-FA attention (~89% of f16 speed). No crash.

## Cross-GPU Validation

| GPU | SM | Stability Iterations | Failures | PPL Drift |
|-----|:--:|:---:|:---:|:---:|
| RTX 5090 | SM120 | Continuous | 0 | None |
| RTX 3090 Ti | SM86 | 8+ | 0 | Bit-exact (7.5535) |
| RTX 4090M | SM89 | 10+ | 0 | Bit-exact (7.5912) |

## Known Limitations

1. **Head dimension**: Native FA decode requires D∈{64, 128, 256}. Other D values fall back to mul_mat attention.
2. **Attention sinks**: Implemented but 0% PPL benefit across 2 models, 5 context lengths, 3 sink sizes.
3. **FP4 tensor core**: Not viable — Q values are too small for E2M1 (99.5% map to zero), no mixed fp16×E2M1 MMA on SM120.
4. **Gemma 3**: Upstream llama.cpp bugs (gibberish after context shift, slow quantized KV) are not TurboQuant-specific.

## Usage

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="120"
cmake --build build -j$(nproc)

./build/bin/llama-cli -hf your-model-GGUF -ctk turbo3 -ctv turbo3 -fa -ngl 99
```
