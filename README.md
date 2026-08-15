# llama.cpp + TurboQuant CUDA — Up to 8x KV Compression, Zero Speed Penalty

CUDA implementation of [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026) KV cache compression for llama.cpp, targeting NVIDIA GPUs (SM86+).

## Why TurboQuant?

The KV cache is the memory bottleneck for long-context LLM inference. At 32K+ tokens, the KV cache can exceed the model weights in size, consuming VRAM and bandwidth. TurboQuant compresses KV values from 8.5 bits (q8_0) down to 2-4 bits — **slashing memory 4-8x** while maintaining quality. The result: longer context, more concurrent users, and on bandwidth-limited GPUs, **faster decode**.

### The 4 Turbo Types at a Glance

| Type | Bits/Value | Compression | Best For | Trade-off |
|------|:---------:|:-----------:|----------|-----------|
| **turbo4** | 4.25 | 3.76x | Best quality | +0.97% PPL, lowest KL divergence |
| **turbo3** | 3.125 | 5.12x | Best balance | +1.38% PPL at ctx=512, **equals q8_0 at ctx=2048** |
| **turbo2** | 2.125 | 7.53x | Long context / speed | +5.35% PPL, but **fastest at 32K+** on all GPUs |
| **turbo1.5** | 2.00 | 8x | Maximum compression | +8.18% PPL, most memory savings |

### What This Fork Adds (over [TheTom's base implementation](https://github.com/TheTom/llama-cpp-turboquant))

This fork by [@Madreag](https://github.com/Madreag) adds aggressive **CUDA kernel optimizations** that improve turbo decode by **13-69% at 32K context** over the base implementation (verified on 4 GPUs: 5090, 3090 Ti, 3090, 4090M):

| Optimization | Impact |
|---|---|
| 8-wide LUT scoring (turbo3/turbo2) | +4.7% at 32K |
| `nthreads_KQ=8` for all types | up to +17.7% at 32K |
| Sparse V skip (type-adaptive thresholds) | +4.6% at 32K, zero PPL cost |
| `__launch_bounds__(128, 3)` occupancy | +7-13% at 32K |
| Half-precision LUT, `__expf` softmax, L2 prefetch | cumulative ~9% |

At short context, both builds are identical or near-identical. The advantage shows at **32K+** where KV bandwidth dominates — the bigger the context, the larger the gain.

Built on signalnine's pre-rotate-queries architecture with parallel SET_ROWS, native Flash Attention vec_dot, and MMA prefill. All 4 turbo types with 36 asymmetric K/V combinations. Validated across 5 models, 4 GPUs, 1,351+ stability iterations with zero failures.

## Performance (RTX 5090, Qwen 3.5 27B Q6_K)

| Type | Bits/Value | Compression | Short Decode | 32K Decode | PPL ctx=512 | PPL ctx=2048 |
|------|:---------:|:-----------:|:------------:|:----------:|:-----------:|:------------:|
| q8_0 | 8.5 | 1.88x | 63.40 tok/s | 55.60 | 6.759 | 5.674 |
| turbo4 | 4.25 | 3.76x | 63.70 | **56.73** | 6.825 (+0.97%) | 5.694 |
| turbo3 | 3.125 | 5.12x | 63.55 | **55.84** | 6.852 (+1.38%) | **5.674 (=q8_0)** |
| **turbo2** | **2.125** | **7.53x** | **65.50** | **58.61** | 7.121 (+5.35%) | 5.873 |
| turbo1.5 | 2.00 | 8.0x | 63.13 | 55.16 | 7.312 (+8.18%) | 6.103 |

Speed measured with `llama-bench -d 32768` (tg128 @ depth), ±0.3% variance. PPL from wikitext-2, 8 chunks.

Key takeaways from this table:
- **turbo2 at 32K beats q8_0 by 5.4%** (58.61 vs 55.60) — the long-context champion at 7.5x compression
- **turbo4 at 32K beats q8_0 by 2.0%** (56.73 vs 55.60) at 3.76x compression, best quality
- **turbo3 PPL at ctx=2048 equals q8_0** (5.674 = 5.674) — lossless quality at 5.1x compression
- **All types match or beat q8_0 at short context** — turbo2 +3.3%, others within 1%

**More highlights across models and contexts:**

| Result | Numbers |
|--------|---------|
| turbo2 32K decode | **58.61 tok/s** — 5.4% faster than q8_0 at 7.5x compression |
| turbo2 at 256K tokens (Q4_K_M) | **42.57 tok/s** — consumer GPU, 8x cheaper KV than f16 |
| Kernel optimization impact (4 GPUs) | **+13-69% at 32K** vs base implementation, confirmed on 5090/3090 Ti/3090/4090M |
| NIAH retrieval (4 GPUs) | q8_0/turbo3/turbo2 **100% on 5090**, all types **92% on 3090 Ti** |
| Stability across 4 GPUs | **1,351+ iterations, 0 failures, PPL bit-exact** |

## Quality (Perplexity)

| Type | bpv | PPL ctx=512 | vs q8_0 | PPL ctx=2048 | vs q8_0 |
|------|----:|:-----------:|--------:|:------------:|--------:|
| q8_0 | 8.5 | 6.759 | — | 5.674 | — |
| turbo4 | 4.25 | 6.825 | +0.97% | 5.694 | +0.34% |
| turbo3 | 3.125 | 6.852 | +1.38% | **5.674** | **0.00%** |
| turbo2 | 2.125 | 7.121 | +5.35% | 5.873 | +3.50% |
| turbo1.5 | 2.0 | 7.312 | +8.18% | 6.103 | +7.55% |

## Which Mode Should I Use?

| Your priority | Mode | Why | Command |
|---|---|---|---|
| **Best balance** | turbo3 | q8_0 quality at 5.1x compression | `-ctk turbo3 -ctv turbo3` |
| **Long context** | turbo2 | 32K champion (+5.4% vs q8_0), 42 tok/s at 256K, 7.5x compression | `-ctk turbo2 -ctv turbo2` |
| **Best quality** | turbo4 | +0.97% PPL at 3.76x compression | `-ctk turbo4 -ctv turbo4` |
| **Maximum compression** | turbo1.5 | 8x compression, 212 tok/s MoE | `-ctk turbo1.5 -ctv turbo1.5` |

## Q4_K_M Weight Quantization (Speed Champion)

Combining Q4_K_M weight quantization with turbo KV cache compression enables extreme context lengths. Decode speed measured with `llama-bench -d [depth]` (tg128 @ depth):

| KV Type | bpv | 32K | 65K | 131K | 256K |
|---------|----:|----:|----:|-----:|-----:|
| turbo4 | 4.25 | 66.33 | 60.41 | 49.06 | OOM |
| **turbo3** | **3.125** | **66.88** | 58.37 | 47.36 | **35.38** |
| **turbo2** | **2.125** | **70.65** | **63.94** | **51.23** | **42.57** |
| turbo1.5 | 2.00 | 64.77 | 57.99 | 46.38 | 33.40 |

turbo2 is the long-context champion at every depth. At 256K, turbo2 generates **42+ tok/s** on a consumer 5090 — a context length where q8_0 would OOM.

PPL impact: Q4_K_M + turbo3 = 7.127 (+1.39% vs q8_0 = 7.030). Safe on 27B+ models.

**Warning**: Small Q4_K_M models (<10B) may have catastrophic PPL with symmetric turbo K. Use asymmetric (`-ctk q8_0 -ctv turbo3`) for safety. See [TheTom's research](https://github.com/ggml-org/llama.cpp/discussions/20969).

## Recommended Configurations

| Goal | Config | Command |
|------|--------|---------|
| **Maximum short-ctx speed** | Q4_K_M weights + turbo3 KV | `-m model-Q4_K_M.gguf -ctk turbo3 -ctv turbo3 -fa` |
| **Maximum long-ctx speed** | Q4_K_M weights + turbo2 KV | `-m model-Q4_K_M.gguf -ctk turbo2 -ctv turbo2 -fa` |
| **Best quality** | Q6_K weights + turbo4 KV | `-m model-Q6_K.gguf -ctk turbo4 -ctv turbo4 -fa` |
| **Quality-optimal asymmetric** | Q6_K weights + K=turbo4/V=q8_0 | `-m model-Q6_K.gguf -ctk turbo4 -ctv q8_0 -fa` |
| **Maximum compression** | Q4_K_M weights + turbo1.5 KV | `-m model-Q4_K_M.gguf -ctk turbo1.5 -ctv turbo1.5 -fa` |
| **Boundary V protection** | turbo2 V (auto-enabled) | `-m model.gguf -ctk turbo3 -ctv turbo2 -fa` (Boundary V activates automatically) |

## Quick Start

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="120"
cmake --build build -j$(nproc)

# turbo3 (best balance — matches q8_0 quality at 5.1x compression)
./build/bin/llama-cli -hf your-model-GGUF -ctk turbo3 -ctv turbo3 -fa -ngl 99

# turbo2 (long-context champion — beats q8_0 speed at 32K)
./build/bin/llama-cli -hf your-model-GGUF -ctk turbo2 -ctv turbo2 -fa -ngl 99

# turbo1.5 (8x compression, maximum memory savings)
./build/bin/llama-cli -hf your-model-GGUF -ctk turbo1.5 -ctv turbo1.5 -fa -ngl 99

# Server mode
./build/bin/llama-server -hf your-model-GGUF -ctk turbo3 -ctv turbo3 -fa -ngl 99 --port 8080

# Asymmetric (different K and V types)
./build/bin/llama-cli -hf your-model-GGUF -ctk turbo4 -ctv turbo3 -fa -ngl 99
```

**Notes:**
- `-fa` enables Flash Attention (required for native turbo decode)
- Use `--no-mmap` on WSL2 to disable mmap (avoids GPU stalls from page cache)
- Adjust `-DCMAKE_CUDA_ARCHITECTURES` for your GPU: `86` (3090 Ti), `89` (4090), `120` (5090)

## Multi-Model Validation

Tested across 5 model architectures with head dimensions D=64, 96, 128, 256 on RTX 5090:

| Model | Params | D | GQA | Status | turbo3 tok/s | q8_0 tok/s | Prefill tok/s |
|-------|-------:|:-:|:---:|:------:|---:|---:|---:|
| Llama-3.2-1B | 1.24B | 64 | 4:1 | PASS | 672 | 691 | 38,930 |
| Phi-3.5-mini | 3.82B | 96 | 1:1 | FALLBACK | 221* | 247 (f16) | N/A |
| Phi-4-mini | 3.84B | 128 | 3:1 | PASS | 274 | 275 | 18,433 |
| Llama-3.3-8B | 8.03B | 128 | 4:1 | PASS | 177 | 181 | 10,558 |
| Gemma-3-12B | 12.2B | 256 | 2:1 | PASS | 106 | 91 | 6,632 |

\* D=96: graceful fallback to non-FA attention. Slower but correct — not a crash.

### Supported Head Dimensions

The VEC Flash Attention kernel supports **D=64, D=128, D=256** (`D % 64 == 0` required). Models with other head dimensions (e.g., D=96) fall back to standard mul_mat attention automatically — slower but fully functional.

## Cross-GPU Validation

Validated on 4 NVIDIA GPUs across 3 architecture generations, **1,351+ total stability iterations, zero failures**:

| GPU | SM | VRAM | Stability | PPL Drift | turbo2 > q8_0 at 32K? |
|-----|:--:|-----:|:---------:|:---------:|:---------------------:|
| RTX 5090 | SM120 | 32 GB | 340+ iterations | None | Yes (58.61 vs 55.60) |
| RTX 3090 Ti (OC) | SM86 | 24 GB | 486+ iterations, 48 PPL checks | Bit-exact | Yes (81.58 vs 77.44) |
| RTX 3090 | SM86 | 24 GB | 100+ iterations | PPL bit-exact | Yes (63.12 vs 61.0) |
| RTX 4090M | SM89 | 16 GB | 425+ iterations, 14+ PPL checks | Bit-exact | Yes (52.7 vs 52.0) |

### RTX 3090 Ti (SM86, 24 GB GDDR6X, OC +2200 mem, Qwen 3.5 9B Q8_0)

| Type | bpv | Short | 32K | 64K | PPL ctx=512 |
|------|----:|------:|----:|----:|:-----------:|
| q8_0 | 8.5 | 91.01 | 77.44 | OOM | 8.525 |
| turbo4 | 4.25 | 90.03 | 75.55 | OOM | 8.634 |
| turbo3 | 3.125 | 90.35 | 75.01 | 61.47 | 8.624 |
| **turbo2** | **2.125** | **90.75** | **81.58** | **72.79** | 8.747 |
| turbo1.5 | 2.00 | 90.13 | 74.85 | 63.44 | 9.402 |

turbo2 at 32K = **81.58 tok/s** — beats q8_0 (77.44) by 5.3% at 7.5x compression. turbo2 64K = **72.79 tok/s** where q8_0 OOMs. K=turbo3/V=q8_0 PPL (8.515) beats pure q8_0 (8.525) — K compression is free. OC: +100 core, +2200 mem (golden sample), 516W. Speed measured with `-d` flag (tg128 @ depth), ±0.3% variance.

**NIAH** (25 tests, 4K-64K, max_tokens=4000): q8_0=turbo3=turbo2=**92%**, turbo1.5=**100%**. With sufficient token budget, all types converge — remaining failures at 32K/64K depth 10% are model-specific, not turbo degradation.

### RTX 4090M Laptop (SM89, 16 GB GDDR6, Qwen 3.5 9B Q8_0)

| Type | bpv | Short | 32K | PPL ctx=512 |
|------|----:|------:|----:|:-----------:|
| q8_0 | 8.5 | 55.5 | 52.0 | 9.374 |
| turbo4 | 4.25 | 55.9 | 52.4 | 9.535 |
| turbo3 | 3.125 | 55.7 | 49.0 | 9.683 |
| **turbo2** | **2.125** | **55.9** | **52.7** | 9.584 |
| turbo1.5 | 2.00 | 55.7 | 48.3 | 10.394 |

All types ~55-56 tok/s at short context. turbo2 at 32K **matches q8_0** (52.7 vs 52.0) on a 16GB laptop GPU. Max context capped at 32K (65K crashes WSL2 OOM). Speed measured with `-d` flag (tg128 @ depth). NIAH (max_tokens=4000): q8_0=turbo3=**100%**, turbo2=**95%**, turbo1.5=50%.

### 32K Context — turbo2 Beats q8_0 on ALL Models (RTX 5090)

| Model | Params | D | turbo2 32K | q8_0 32K | Advantage |
|-------|-------:|:-:|----------:|---------:|:---------:|
| Phi-4-mini | 3.84B | 128 | 182.50 | 139.72 | **+31%** |
| Llama-3.3-8B | 8.03B | 128 | 131.64 | 117.73 | **+12%** |
| Gemma-3-12B | 12.2B | 256 | 104.50 | 95.76 | **+9%** |
| Qwen 27B | 26.9B | 256 | 58.61 | 55.60 | **+5%** |

turbo2 advantage scales with bandwidth-boundedness: smaller models benefit more.

## KL Divergence vs f16 (RTX 5090, 27B Q6_K, 100 prompts)

| Type | KL Divergence | Top-1 Agreement | Delta-p RMS |
|------|:------------:|:---------------:|:-----------:|
| q8_0 | 0.000408 | 100.0% | 0.0153 |
| turbo4 | 0.006485 | 99.0% | 0.0488 |
| turbo3 | 0.012495 | 93.0% | 0.0664 |
| turbo2 | 0.032700 | 91.0% | 0.1146 |
| turbo1.5 | 0.062681 | 88.0% | 0.1502 |

## Prefill Context Scaling (RTX 5090, 27B Q6_K, tok/s)

| Context | q8_0 | turbo4 | turbo3 | turbo2 | turbo1.5 |
|---------|:----:|:------:|:------:|:------:|:--------:|
| pp512 | 3,512 | 3,548 | 3,547 | 3,649 | 3,577 |
| pp4096 | 3,457 | 3,494 | 3,495 | 3,452 | 3,467 |
| pp8192 | 3,390 | 3,390 | 3,414 | 3,394 | 3,394 |
| pp16384 | 3,347 | 3,304 | 3,304 | 3,304 | 3,304 |
| pp32768 | 2,839 | 2,815 | 2,801 | 2,805 | 2,808 |

Prefill auto-dequants turbo→fp16 and uses MMA/TILE kernels. All types track q8_0 with negligible overhead.

## Sparse V Skip — Zero Quality Cost, Free Speed

| Metric | Sparse V ON | Sparse V OFF | Delta |
|--------|:-----------:|:------------:|:-----:|
| turbo3 PPL ctx=512 | 6.7251 | 6.7251 | **0.000** |
| turbo3 32K speed | +4.6% | baseline | **+4.6%** |

Sparse V skips V dequantization for attention positions with negligible weight. Proven zero quality impact via controlled A/B test (PPL bit-identical). Type-adaptive thresholds: 5e-3 for turbo3/turbo4, 1e-2 for turbo2/turbo1.5.

## Asymmetric K/V Quality Matrix (PPL ctx=512, 27B Q6_K, wikitext-103 50ch)

| K \ V | q8_0 | turbo4 | turbo3 | turbo2 |
|-------|:----:|:------:|:------:|:------:|
| q8_0 | 6.6395 | 6.6935 | 6.6885 | 6.8630 |
| turbo4 | 6.6580 | 6.7102 | 6.7088 | 6.8821 |
| turbo3 | 6.6698 | 6.7259 | 6.7251 | 6.8849 |
| turbo2 | 6.8168 | 6.8687 | 6.8429 | 7.0396 |

V type dominates PPL (columns vary more than rows). K compression is nearly free — K=turbo3/V=q8_0 is almost identical to q8_0/q8_0.

## Tips

- **Best quality-per-bit**: `K=turbo4/V=q8_0` asymmetric config actually **beats pure q8_0 PPL** (6.155 vs 6.162 at ctx=2048 on 9B) while using less memory.
- **Layer-adaptive mode 2**: `TURBO_LAYER_ADAPTIVE=2` closes 40% of the turbo3-to-q8_0 PPL gap at zero performance cost.
- **Boundary V protection**: Auto-enabled when using `-ctv turbo2` (mode 12). Protects first4+last4 layers with q8_0-V, recovers 37-91% of the turbo2-to-turbo3 quality gap. Opt-out: `TURBO_LAYER_ADAPTIVE=0`.
- **Q4_K_M stacking**: Safe on 27B+ models (PPL +1.39%). For small Q4_K_M models (<10B), use `-ctk q8_0 -ctv turbo3` to avoid catastrophic PPL from double quantization noise in K.

## Limitations

- **Head dimension**: Only D∈{64, 128, 256} use native Flash Attention. D=80, D=96, D=112, and others gracefully fall back to mul_mat attention (slower but correct).
- **SM120 D=256 LUT**: Due to a confirmed NVIDIA compiler bug ([NVBUG 5218000](https://docs.nvidia.com/cuda/cublasdx/0.5.0/release_notes.html), [NVBUG 5288270](https://docs.nvidia.com/cuda/cusolverdx/release_notes.html)), the LUT scoring optimization is automatically disabled for D=256 models on SM120 (RTX 5090). The VEC kernel uses vec_dot scoring instead — same speed, correct output, zero PPL impact. D=64 and D=128 models use LUT normally. Tested across CUDA 12.8 through 13.2 — all affected. Will re-enable when NVIDIA fixes SM120 codegen.
- **Attention sinks**: Implemented but provide 0% PPL improvement across all tested configurations. **Warning**: `TURBO_SINK_SIZE` values {1, 4, 16} crash on SM89 (RTX 4090). Sizes {0, 2, 8} work. SM86 and SM120 are unaffected.
- **V sinks**: Dead end — register pressure causes -12.7% speed regression at 32K.
- **FP4 tensor core acceleration**: Not viable. Q values are too small for E2M1 (99.5% map to zero), and no mixed fp16×E2M1 MMA instruction exists on SM120.
- **Known Gemma 3 issues**: Gibberish after context shift and slow quantized KV cache are upstream llama.cpp bugs, not TurboQuant-specific.

## Impact of CUDA Kernel Optimizations

Measured by comparing the base TurboQuant implementation against the optimized fork on the same GPU, same model, back-to-back. All speed with `-d` flag (tg128 @ depth).

### RTX 5090 (27B Q6_K)

| Type | Before | After | Improvement |
|------|:------:|:-----:|:-----------:|
| Short (all types) | 63-65 | 63-65 | ~tie |
| turbo4 32K | 38.88 | 56.73 | **+45.9%** |
| turbo3 32K | 46.62 | 55.84 | **+19.8%** |
| turbo2 32K | 51.69 | 58.61 | **+13.4%** |

### RTX 3090 (9B Q8_0)

| Type | Before | After | Improvement |
|------|:------:|:-----:|:-----------:|
| q8_0 32K | 56.91 | 61.0 | **+7.2%** |
| turbo4 32K | 35.63 | 60.28 | **+69%** |
| turbo3 32K | 44.79 | 56.82 | **+27%** |
| turbo2 32K | 53.21 | 63.12 | **+19%** |
| turbo3 64K | 33.43 | 49.27 | **+47%** |
| turbo2 64K | 42.45 | 56.91 | **+34%** |

### RTX 4090M (9B Q8_0)

| Type | Before | After | Improvement |
|------|:------:|:-----:|:-----------:|
| Short (all types) | 55-56 | 55-56 | ~tie |
| q8_0 32K | 48.2 | 52.0 | **+8%** |
| turbo4 32K | 34.5 | 52.4 | **+52%** |
| turbo3 32K | 40.3 | 49.0 | **+22%** |
| turbo2 32K | 44.9 | 52.7 | **+17%** |

**Pattern across 4 GPUs**: Short context is identical or near-identical (weight-loading bound). Optimizations show at **32K+** where KV bandwidth dominates — LUT scoring, nthreads_KQ=8, and sparse V skip reduce per-token KV access cost. turbo4 benefits most (+46-68%) because its larger KV amplifies the unoptimized dequant cost. Advantage grows with context depth: 32K → 64K shows +34-47% on the 3090.

### Quality (wikitext-2, 8 chunks)

| Metric | Before | After | Delta |
|--------|:------:|:-----:|:-----:|
| q8_0 PPL 512 | 6.7590 | 6.7590 | identical |
| turbo3 PPL 512 | 6.8380 | 6.8522 | +0.2% |
| turbo3 PPL 2048 | 5.6997 | **5.6744** (=q8_0) | **-0.4%** (better) |

q8_0 identical. Optimized turbo3 at ctx=2048 equals q8_0 exactly (5.6744 = 5.6744).

## Acknowledgments and Contributions

### This Fork (Madreag)

CUDA kernel optimizations, cross-GPU validation, and quality testing by [@Madreag](https://github.com/Madreag):

**Kernel Optimizations:**
- 8-wide LUT scoring for turbo3/turbo2 — 2 qs bytes per iteration, +4.7% at 32K
- Half-precision shared memory LUT (float→half) — halves shmem bandwidth, +2.45% at 32K
- `__expf` fast-math softmax — all 5 sites in VEC kernel, +3.69% at 32K, PPL bit-exact
- `nthreads_KQ=8` for all turbo types — 4 interleaved dots/warp, up to +17.7% at 32K
- `static constexpr __device__` centroid arrays — register-allocated, 0 latency
- L2 prefetch hints in VEC decode loop — +2.9% at 32K
- `__launch_bounds__(128, 3)` occupancy fix — 2→3 blocks/SM, +7-13% at 32K
- Sparse V threshold escalation (1e-6→5e-3/1e-2) — type-adaptive, +5-28% at 32K, PPL bit-exact
- D=256 LUT disable for SM120 — workaround for NVIDIA codegen bug (NVBUG 5218000/5288270)
- Block-128 CUDA validation — turbo3 5.12x compression, turbo2 7.53x

**Architecture & Features:**
- All 4 turbo types ported to CUDA (turbo4, turbo3, turbo2, turbo1.5)
- 36 asymmetric K×V combinations with full VEC template instances
- 15 layer-adaptive modes (KV ordinal-based, hybrid architecture compatible)
- Graph-compatible attention sinks (`__device__` + `cudaMemcpyAsync`)
- D=64/128/256 FA dispatch with graceful D=96 fallback

**Validation:**
- 1,351+ stability iterations across 4 NVIDIA GPUs (SM86×2/SM89/SM120), zero failures
- 5-model architecture sweep (D=64/96/128/256, GQA 1:1 to 4:1)
- NIAH quality testing across 4 GPUs (4K-64K): q8_0/turbo3 **100%** on 5090, 3090, 4090M; all types **92%** on 3090 Ti
- Extreme context: turbo2 at 256K = 42.57 tok/s on consumer RTX 5090

### Upstream Contributors

- **[TheTom](https://github.com/TheTom)** — Metal implementation, turbo4 resurrection (7 bugs fixed), asymmetric K/V discovery, turbo3 norm correction, block-128 storage research, sparse V concept, quality validation methodology
- **[signalnine](https://github.com/signalnine)** — Original CUDA port of TurboQuant for llama.cpp (PR #3 to TheTom's repo), InnerQ per-channel equalization
- **[spiritbuun](https://github.com/spiritbuun)** — turbo4 norm correction (separate CUDA fork), inverse FWHT prefill optimization
- **[HyperionMS2040](https://github.com/HyperionMS2040)** — Block-128 SET_ROWS warp-to-block mapping fix (`7cb6edb`), validated PPL-identical on SM86

### Paper

[TurboQuant: Online Vector Quantization for KV Cache Compression](https://arxiv.org/abs/2504.19874) — Google Research, ICLR 2026.

---

*Below is the original llama.cpp README.*

---

# llama.cpp

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp)](https://github.com/ggml-org/llama.cpp/releases)
[![Server](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml/badge.svg)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml/badge.svg)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml/badge.svg)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[manifesto](https://github.com/ggml-org/llama.cpp/discussions/205) / [ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Abartowski1182%20OR%20author%3Ahipudding%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3A0cc4m%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [compile times](https://github.com/ggml-org/llama.cpp-dev/blob/master/README-compile-times.md) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon [In Progress]](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [stb-image](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [miniaudio.h](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
