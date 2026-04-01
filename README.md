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

![llama](https://user-images.githubusercontent.com/1991296/230134379-7181e485-c521-4d23-a0d6-f7b3b61ba524.png)

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp)](https://github.com/ggml-org/llama.cpp/releases)
[![Server](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml/badge.svg)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)

[Manifesto](https://github.com/ggml-org/llama.cpp/discussions/205) / [ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md)

LLM inference in C/C++

## Recent API changes

- [Changelog for `libllama` API](https://github.com/ggml-org/llama.cpp/issues/9289)
- [Changelog for `llama-server` REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

## Hot topics

- **Hugging Face cache migration: models downloaded with `-hf` are now stored in the standard Hugging Face cache directory, enabling sharing with other HF tools.**
- **[guide : using the new WebUI of llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/16938)**
- [guide : running gpt-oss with llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/15396)
- [[FEEDBACK] Better packaging for llama.cpp to support downstream consumers 🤗](https://github.com/ggml-org/llama.cpp/discussions/15313)
- Support for the `gpt-oss` model with native MXFP4 format has been added | [PR](https://github.com/ggml-org/llama.cpp/pull/15091) | [Collaboration with NVIDIA](https://blogs.nvidia.com/blog/rtx-ai-garage-openai-oss) | [Comment](https://github.com/ggml-org/llama.cpp/discussions/15095)
- Multimodal support arrived in `llama-server`: [#12898](https://github.com/ggml-org/llama.cpp/pull/12898) | [documentation](./docs/multimodal.md)
- VS Code extension for FIM completions: https://github.com/ggml-org/llama.vscode
- Vim/Neovim plugin for FIM completions: https://github.com/ggml-org/llama.vim
- Hugging Face Inference Endpoints now support GGUF out of the box! https://github.com/ggml-org/llama.cpp/discussions/9669
- Hugging Face GGUF editor: [discussion](https://github.com/ggml-org/llama.cpp/discussions/9268) | [tool](https://huggingface.co/spaces/CISCai/gguf-editor)

----

## Quick start

Getting started with llama.cpp is straightforward. Here are several ways to install it on your machine:

- Install `llama.cpp` using [brew, nix or winget](docs/install.md)
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed, you'll need a model to work with. Head to the [Obtaining and quantizing models](#obtaining-and-quantizing-models) section to learn more.

Example command:

```sh
# Use a local model file
llama-cli -m my_model.gguf

# Or download and run a model directly from Hugging Face
llama-cli -hf ggml-org/gemma-3-1b-it-GGUF

# Launch OpenAI-compatible API server
llama-server -hf ggml-org/gemma-3-1b-it-GGUF
```

## Description

The main goal of `llama.cpp` is to enable LLM inference with minimal setup and state-of-the-art performance on a wide
range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is the main playground for developing new features for the [ggml](https://github.com/ggml-org/ggml) library.

<details>
<summary>Models</summary>

Typically finetunes of the base models below are supported as well.

Instructions for adding support for new models: [HOWTO-add-model.md](docs/development/HOWTO-add-model.md)

#### Text-only

- [X] LLaMA 🦙
- [x] LLaMA 2 🦙🦙
- [x] LLaMA 3 🦙🦙🦙
- [X] [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [x] [Mixtral MoE](https://huggingface.co/models?search=mistral-ai/Mixtral)
- [x] [DBRX](https://huggingface.co/databricks/dbrx-instruct)
- [x] [Jamba](https://huggingface.co/ai21labs)
- [X] [Falcon](https://huggingface.co/models?search=tiiuae/falcon)
- [X] [Chinese LLaMA / Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca) and [Chinese LLaMA-2 / Alpaca-2](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2)
- [X] [Vigogne (French)](https://github.com/bofenghuang/vigogne)
- [X] [BERT](https://github.com/ggml-org/llama.cpp/pull/5423)
- [X] [Koala](https://bair.berkeley.edu/blog/2023/04/03/koala/)
- [X] [Baichuan 1 & 2](https://huggingface.co/models?search=baichuan-inc/Baichuan) + [derivations](https://huggingface.co/hiyouga/baichuan-7b-sft)
- [X] [Aquila 1 & 2](https://huggingface.co/models?search=BAAI/Aquila)
- [X] [Starcoder models](https://github.com/ggml-org/llama.cpp/pull/3187)
- [X] [Refact](https://huggingface.co/smallcloudai/Refact-1_6B-fim)
- [X] [MPT](https://github.com/ggml-org/llama.cpp/pull/3417)
- [X] [Bloom](https://github.com/ggml-org/llama.cpp/pull/3553)
- [x] [Yi models](https://huggingface.co/models?search=01-ai/Yi)
- [X] [StableLM models](https://huggingface.co/stabilityai)
- [x] [Deepseek models](https://huggingface.co/models?search=deepseek-ai/deepseek)
- [x] [Qwen models](https://huggingface.co/models?search=Qwen/Qwen)
- [x] [PLaMo-13B](https://github.com/ggml-org/llama.cpp/pull/3557)
- [x] [Phi models](https://huggingface.co/models?search=microsoft/phi)
- [x] [PhiMoE](https://github.com/ggml-org/llama.cpp/pull/11003)
- [x] [GPT-2](https://huggingface.co/gpt2)
- [x] [Orion 14B](https://github.com/ggml-org/llama.cpp/pull/5118)
- [x] [InternLM2](https://huggingface.co/models?search=internlm2)
- [x] [CodeShell](https://github.com/WisdomShell/codeshell)
- [x] [Gemma](https://ai.google.dev/gemma)
- [x] [Mamba](https://github.com/state-spaces/mamba)
- [x] [Grok-1](https://huggingface.co/keyfan/grok-1-hf)
- [x] [Xverse](https://huggingface.co/models?search=xverse)
- [x] [Command-R models](https://huggingface.co/models?search=CohereForAI/c4ai-command-r)
- [x] [SEA-LION](https://huggingface.co/models?search=sea-lion)
- [x] [GritLM-7B](https://huggingface.co/GritLM/GritLM-7B) + [GritLM-8x7B](https://huggingface.co/GritLM/GritLM-8x7B)
- [x] [OLMo](https://allenai.org/olmo)
- [x] [OLMo 2](https://allenai.org/olmo)
- [x] [OLMoE](https://huggingface.co/allenai/OLMoE-1B-7B-0924)
- [x] [Granite models](https://huggingface.co/collections/ibm-granite/granite-code-models-6624c5cec322e4c148c8b330)
- [x] [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) + [Pythia](https://github.com/EleutherAI/pythia)
- [x] [Snowflake-Arctic MoE](https://huggingface.co/collections/Snowflake/arctic-66290090abe542894a5ac520)
- [x] [Smaug](https://huggingface.co/models?search=Smaug)
- [x] [Poro 34B](https://huggingface.co/LumiOpen/Poro-34B)
- [x] [Bitnet b1.58 models](https://huggingface.co/1bitLLM)
- [x] [Flan T5](https://huggingface.co/models?search=flan-t5)
- [x] [Open Elm models](https://huggingface.co/collections/apple/openelm-instruct-models-6619ad295d7ae9f868b759ca)
- [x] [ChatGLM3-6b](https://huggingface.co/THUDM/chatglm3-6b) + [ChatGLM4-9b](https://huggingface.co/THUDM/glm-4-9b) + [GLMEdge-1.5b](https://huggingface.co/THUDM/glm-edge-1.5b-chat) + [GLMEdge-4b](https://huggingface.co/THUDM/glm-edge-4b-chat)
- [x] [GLM-4-0414](https://huggingface.co/collections/THUDM/glm-4-0414-67f3cbcb34dd9d252707cb2e)
- [x] [SmolLM](https://huggingface.co/collections/HuggingFaceTB/smollm-6695016cad7167254ce15966)
- [x] [EXAONE-3.0-7.8B-Instruct](https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct)
- [x] [FalconMamba Models](https://huggingface.co/collections/tiiuae/falconmamba-7b-66b9a580324dd1598b0f6d4a)
- [x] [Jais](https://huggingface.co/inceptionai/jais-13b-chat)
- [x] [Bielik-11B-v2.3](https://huggingface.co/collections/speakleash/bielik-11b-v23-66ee813238d9b526a072408a)
- [x] [RWKV-7](https://huggingface.co/collections/shoumenchougou/rwkv7-gxx-gguf)
- [x] [RWKV-6](https://github.com/BlinkDL/RWKV-LM)
- [x] [QRWKV-6](https://huggingface.co/recursal/QRWKV6-32B-Instruct-Preview-v0.1)
- [x] [GigaChat-20B-A3B](https://huggingface.co/ai-sage/GigaChat-20B-A3B-instruct)
- [X] [Trillion-7B-preview](https://huggingface.co/trillionlabs/Trillion-7B-preview)
- [x] [Ling models](https://huggingface.co/collections/inclusionAI/ling-67c51c85b34a7ea0aba94c32)
- [x] [LFM2 models](https://huggingface.co/collections/LiquidAI/lfm2-686d721927015b2ad73eaa38)
- [x] [Hunyuan models](https://huggingface.co/collections/tencent/hunyuan-dense-model-6890632cda26b19119c9c5e7)
- [x] [BailingMoeV2 (Ring/Ling 2.0) models](https://huggingface.co/collections/inclusionAI/ling-v2-68bf1dd2fc34c306c1fa6f86)

#### Multimodal

- [x] [LLaVA 1.5 models](https://huggingface.co/collections/liuhaotian/llava-15-653aac15d994e992e2677a7e), [LLaVA 1.6 models](https://huggingface.co/collections/liuhaotian/llava-16-65b9e40155f60fd046a5ccf2)
- [x] [BakLLaVA](https://huggingface.co/models?search=SkunkworksAI/Bakllava)
- [x] [Obsidian](https://huggingface.co/NousResearch/Obsidian-3B-V0.5)
- [x] [ShareGPT4V](https://huggingface.co/models?search=Lin-Chen/ShareGPT4V)
- [x] [MobileVLM 1.7B/3B models](https://huggingface.co/models?search=mobileVLM)
- [x] [Yi-VL](https://huggingface.co/models?search=Yi-VL)
- [x] [Mini CPM](https://huggingface.co/models?search=MiniCPM)
- [x] [Moondream](https://huggingface.co/vikhyatk/moondream2)
- [x] [Bunny](https://github.com/BAAI-DCAI/Bunny)
- [x] [GLM-EDGE](https://huggingface.co/models?search=glm-edge)
- [x] [Qwen2-VL](https://huggingface.co/collections/Qwen/qwen2-vl-66cee7455501d7126940800d)
- [x] [LFM2-VL](https://huggingface.co/collections/LiquidAI/lfm2-vl-68963bbc84a610f7638d5ffa)

</details>

<details>
<summary>Bindings</summary>

- Python: [ddh0/easy-llama](https://github.com/ddh0/easy-llama)
- Python: [abetlen/llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- Go: [go-skynet/go-llama.cpp](https://github.com/go-skynet/go-llama.cpp)
- Node.js: [withcatai/node-llama-cpp](https://github.com/withcatai/node-llama-cpp)
- JS/TS (llama.cpp server client): [lgrammel/modelfusion](https://modelfusion.dev/integration/model-provider/llamacpp)
- JS/TS (Programmable Prompt Engine CLI): [offline-ai/cli](https://github.com/offline-ai/cli)
- JavaScript/Wasm (works in browser): [tangledgroup/llama-cpp-wasm](https://github.com/tangledgroup/llama-cpp-wasm)
- Typescript/Wasm (nicer API, available on npm): [ngxson/wllama](https://github.com/ngxson/wllama)
- Ruby: [yoshoku/llama_cpp.rb](https://github.com/yoshoku/llama_cpp.rb)
- Rust (more features): [edgenai/llama_cpp-rs](https://github.com/edgenai/llama_cpp-rs)
- Rust (nicer API): [mdrokz/rust-llama.cpp](https://github.com/mdrokz/rust-llama.cpp)
- Rust (more direct bindings): [utilityai/llama-cpp-rs](https://github.com/utilityai/llama-cpp-rs)
- Rust (automated build from crates.io): [ShelbyJenkins/llm_client](https://github.com/ShelbyJenkins/llm_client)
- C#/.NET: [SciSharp/LLamaSharp](https://github.com/SciSharp/LLamaSharp)
- C#/VB.NET (more features - community license): [LM-Kit.NET](https://docs.lm-kit.com/lm-kit-net/index.html)
- Scala 3: [donderom/llm4s](https://github.com/donderom/llm4s)
- Clojure: [phronmophobic/llama.clj](https://github.com/phronmophobic/llama.clj)
- React Native: [mybigday/llama.rn](https://github.com/mybigday/llama.rn)
- Java: [kherud/java-llama.cpp](https://github.com/kherud/java-llama.cpp)
- Java: [QuasarByte/llama-cpp-jna](https://github.com/QuasarByte/llama-cpp-jna)
- Zig: [deins/llama.cpp.zig](https://github.com/Deins/llama.cpp.zig)
- Flutter/Dart: [netdur/llama_cpp_dart](https://github.com/netdur/llama_cpp_dart)
- Flutter: [xuegao-tzx/Fllama](https://github.com/xuegao-tzx/Fllama)
- PHP (API bindings and features built on top of llama.cpp): [distantmagic/resonance](https://github.com/distantmagic/resonance) [(more info)](https://github.com/ggml-org/llama.cpp/pull/6326)
- Guile Scheme: [guile_llama_cpp](https://savannah.nongnu.org/projects/guile-llama-cpp)
- Swift [srgtuszy/llama-cpp-swift](https://github.com/srgtuszy/llama-cpp-swift)
- Swift [ShenghaiWang/SwiftLlama](https://github.com/ShenghaiWang/SwiftLlama)
- Delphi [Embarcadero/llama-cpp-delphi](https://github.com/Embarcadero/llama-cpp-delphi)
- Go (no CGo needed): [hybridgroup/yzma](https://github.com/hybridgroup/yzma)
- Android: [llama.android](/examples/llama.android)

</details>

<details>
<summary>UIs</summary>

*(to have a project listed here, it should clearly state that it depends on `llama.cpp`)*

- [AI Sublime Text plugin](https://github.com/yaroslavyaroslav/OpenAI-sublime-text) (MIT)
- [BonzAI App](https://apps.apple.com/us/app/bonzai-your-local-ai-agent/id6752847988) (proprietary)
- [cztomsik/ava](https://github.com/cztomsik/ava) (MIT)
- [Dot](https://github.com/alexpinel/Dot) (GPL)
- [eva](https://github.com/ylsdamxssjxxdd/eva) (MIT)
- [iohub/collama](https://github.com/iohub/coLLaMA) (Apache-2.0)
- [janhq/jan](https://github.com/janhq/jan) (AGPL)
- [johnbean393/Sidekick](https://github.com/johnbean393/Sidekick) (MIT)
- [KanTV](https://github.com/zhouwg/kantv?tab=readme-ov-file) (Apache-2.0)
- [KodiBot](https://github.com/firatkiral/kodibot) (GPL)
- [llama.vim](https://github.com/ggml-org/llama.vim) (MIT)
- [LARS](https://github.com/abgulati/LARS) (AGPL)
- [Llama Assistant](https://github.com/vietanhdev/llama-assistant) (GPL)
- [LlamaLib](https://github.com/undreamai/LlamaLib) (Apache-2.0)
- [LLMFarm](https://github.com/guinmoon/LLMFarm?tab=readme-ov-file) (MIT)
- [LLMUnity](https://github.com/undreamai/LLMUnity) (MIT)
- [LMStudio](https://lmstudio.ai/) (proprietary)
- [LocalAI](https://github.com/mudler/LocalAI) (MIT)
- [LostRuins/koboldcpp](https://github.com/LostRuins/koboldcpp) (AGPL)
- [MindMac](https://mindmac.app) (proprietary)
- [MindWorkAI/AI-Studio](https://github.com/MindWorkAI/AI-Studio) (FSL-1.1-MIT)
- [Mobile-Artificial-Intelligence/maid](https://github.com/Mobile-Artificial-Intelligence/maid) (MIT)
- [Mozilla-Ocho/llamafile](https://github.com/Mozilla-Ocho/llamafile) (Apache-2.0)
- [nat/openplayground](https://github.com/nat/openplayground) (MIT)
- [nomic-ai/gpt4all](https://github.com/nomic-ai/gpt4all) (MIT)
- [ollama/ollama](https://github.com/ollama/ollama) (MIT)
- [oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui) (AGPL)
- [PocketPal AI](https://github.com/a-ghorbani/pocketpal-ai) (MIT)
- [psugihara/FreeChat](https://github.com/psugihara/FreeChat) (MIT)
- [ptsochantaris/emeltal](https://github.com/ptsochantaris/emeltal) (MIT)
- [pythops/tenere](https://github.com/pythops/tenere) (AGPL)
- [ramalama](https://github.com/containers/ramalama) (MIT)
- [semperai/amica](https://github.com/semperai/amica) (MIT)
- [withcatai/catai](https://github.com/withcatai/catai) (MIT)
- [Autopen](https://github.com/blackhole89/autopen) (GPL)

</details>

<details>
<summary>Tools</summary>

- [akx/ggify](https://github.com/akx/ggify) – download PyTorch models from Hugging Face Hub and convert them to GGML
- [akx/ollama-dl](https://github.com/akx/ollama-dl) – download models from the Ollama library to be used directly with llama.cpp
- [crashr/gppm](https://github.com/crashr/gppm) – launch llama.cpp instances utilizing NVIDIA Tesla P40 or P100 GPUs with reduced idle power consumption
- [gpustack/gguf-parser](https://github.com/gpustack/gguf-parser-go/tree/main/cmd/gguf-parser) - review/check the GGUF file and estimate the memory usage
- [Styled Lines](https://marketplace.unity.com/packages/tools/generative-ai/styled-lines-llama-cpp-model-292902) (proprietary licensed, async wrapper of inference part for game development in Unity3d with pre-built Mobile and Web platform wrappers and a model example)
- [unslothai/unsloth](https://github.com/unslothai/unsloth) – 🦥 exports/saves fine-tuned and trained models to GGUF (Apache-2.0)

</details>

<details>
<summary>Infrastructure</summary>

- [Paddler](https://github.com/intentee/paddler) - Open-source LLMOps platform for hosting and scaling AI in your own infrastructure
- [GPUStack](https://github.com/gpustack/gpustack) - Manage GPU clusters for running LLMs
- [llama_cpp_canister](https://github.com/onicai/llama_cpp_canister) - llama.cpp as a smart contract on the Internet Computer, using WebAssembly
- [llama-swap](https://github.com/mostlygeek/llama-swap) - transparent proxy that adds automatic model switching with llama-server
- [Kalavai](https://github.com/kalavai-net/kalavai-client) - Crowdsource end to end LLM deployment at any scale
- [llmaz](https://github.com/InftyAI/llmaz) - ☸️ Easy, advanced inference platform for large language models on Kubernetes.
- [LLMKube](https://github.com/defilantech/llmkube) - Kubernetes operator for llama.cpp with multi-GPU and Apple Silicon Metal
  support"
</details>

<details>
<summary>Games</summary>

- [Lucy's Labyrinth](https://github.com/MorganRO8/Lucys_Labyrinth) - A simple maze game where agents controlled by an AI model will try to trick you.

</details>


## Supported backends

| Backend | Target devices |
| --- | --- |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [SYCL](docs/backend/SYCL.md) | Intel and Nvidia GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [WebGPU [In Progress]](docs/build.md#webgpu) | All |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [Hexagon [In Progress]](docs/backend/snapdragon/README.md) | Snapdragon |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |

## Obtaining and quantizing models

The [Hugging Face](https://huggingface.co) platform hosts a [number of LLMs](https://huggingface.co/models?library=gguf&sort=trending) compatible with `llama.cpp`:

- [Trending](https://huggingface.co/models?library=gguf&sort=trending)
- [LLaMA](https://huggingface.co/models?sort=trending&search=llama+gguf)

You can either manually download the GGUF file or directly use any `llama.cpp`-compatible models from [Hugging Face](https://huggingface.co/) or other model hosting sites, by using this CLI argument: `-hf <user>/<model>[:quant]`. For example:

```sh
llama-cli -hf ggml-org/gemma-3-1b-it-GGUF
```

By default, the CLI would download from Hugging Face, you can switch to other options with the environment variable `MODEL_ENDPOINT`. The `MODEL_ENDPOINT` must point to a Hugging Face compatible API endpoint.

After downloading a model, use the CLI tools to run it locally - see below.

`llama.cpp` requires the model to be stored in the [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) file format. Models in other data formats can be converted to GGUF using the `convert_*.py` Python scripts in this repo.

The Hugging Face platform provides a variety of online tools for converting, quantizing and hosting models with `llama.cpp`:

- Use the [GGUF-my-repo space](https://huggingface.co/spaces/ggml-org/gguf-my-repo) to convert to GGUF format and quantize model weights to smaller sizes
- Use the [GGUF-my-LoRA space](https://huggingface.co/spaces/ggml-org/gguf-my-lora) to convert LoRA adapters to GGUF format (more info: https://github.com/ggml-org/llama.cpp/discussions/10123)
- Use the [GGUF-editor space](https://huggingface.co/spaces/CISCai/gguf-editor) to edit GGUF meta data in the browser (more info: https://github.com/ggml-org/llama.cpp/discussions/9268)
- Use the [Inference Endpoints](https://ui.endpoints.huggingface.co/) to directly host `llama.cpp` in the cloud (more info: https://github.com/ggml-org/llama.cpp/discussions/9669)

To learn more about model quantization, [read this documentation](tools/quantize/README.md)

## [`llama-cli`](tools/cli)

#### A CLI tool for accessing and experimenting with most of `llama.cpp`'s functionality.

- <details open>
    <summary>Run in conversation mode</summary>

    Models with a built-in chat template will automatically activate conversation mode. If this doesn't occur, you can manually enable it by adding `-cnv` and specifying a suitable chat template with `--chat-template NAME`

    ```bash
    llama-cli -m model.gguf

    # > hi, who are you?
    # Hi there! I'm your helpful assistant! I'm an AI-powered chatbot designed to assist and provide information to users like you. I'm here to help answer your questions, provide guidance, and offer support on a wide range of topics. I'm a friendly and knowledgeable AI, and I'm always happy to help with anything you need. What's on your mind, and how can I assist you today?
    #
    # > what is 1+1?
    # Easy peasy! The answer to 1+1 is... 2!
    ```

    </details>

- <details>
    <summary>Run in conversation mode with custom chat template</summary>

    ```bash
    # use the "chatml" template (use -h to see the list of supported templates)
    llama-cli -m model.gguf -cnv --chat-template chatml

    # use a custom template
    llama-cli -m model.gguf -cnv --in-prefix 'User: ' --reverse-prompt 'User:'
    ```

    </details>

- <details>
    <summary>Constrain the output with a custom grammar</summary>

    ```bash
    llama-cli -m model.gguf -n 256 --grammar-file grammars/json.gbnf -p 'Request: schedule a call at 8pm; Command:'

    # {"appointmentTime": "8pm", "appointmentDetails": "schedule a a call"}
    ```

    The [grammars/](grammars/) folder contains a handful of sample grammars. To write your own, check out the [GBNF Guide](grammars/README.md).

    For authoring more complex JSON grammars, check out https://grammar.intrinsiclabs.ai/

    </details>


## [`llama-server`](tools/server)

#### A lightweight, [OpenAI API](https://github.com/openai/openai-openapi) compatible, HTTP server for serving LLMs.

- <details open>
    <summary>Start a local HTTP server with default configuration on port 8080</summary>

    ```bash
    llama-server -m model.gguf --port 8080

    # Basic web UI can be accessed via browser: http://localhost:8080
    # Chat completion endpoint: http://localhost:8080/v1/chat/completions
    ```

    </details>

- <details>
    <summary>Support multiple-users and parallel decoding</summary>

    ```bash
    # up to 4 concurrent requests, each with 4096 max context
    llama-server -m model.gguf -c 16384 -np 4
    ```

    </details>

- <details>
    <summary>Enable speculative decoding</summary>

    ```bash
    # the draft.gguf model should be a small variant of the target model.gguf
    llama-server -m model.gguf -md draft.gguf
    ```

    </details>

- <details>
    <summary>Serve an embedding model</summary>

    ```bash
    # use the /embedding endpoint
    llama-server -m model.gguf --embedding --pooling cls -ub 8192
    ```

    </details>

- <details>
    <summary>Serve a reranking model</summary>

    ```bash
    # use the /reranking endpoint
    llama-server -m model.gguf --reranking
    ```

    </details>

- <details>
    <summary>Constrain all outputs with a grammar</summary>

    ```bash
    # custom grammar
    llama-server -m model.gguf --grammar-file grammar.gbnf

    # JSON
    llama-server -m model.gguf --grammar-file grammars/json.gbnf
    ```

    </details>


## [`llama-perplexity`](tools/perplexity)

#### A tool for measuring the [perplexity](tools/perplexity/README.md) [^1] (and other quality metrics) of a model over a given text.

- <details open>
    <summary>Measure the perplexity over a text file</summary>

    ```bash
    llama-perplexity -m model.gguf -f file.txt

    # [1]15.2701,[2]5.4007,[3]5.3073,[4]6.2965,[5]5.8940,[6]5.6096,[7]5.7942,[8]4.9297, ...
    # Final estimate: PPL = 5.4007 +/- 0.67339
    ```

    </details>

- <details>
    <summary>Measure KL divergence</summary>

    ```bash
    # TODO
    ```

    </details>

[^1]: [https://huggingface.co/docs/transformers/perplexity](https://huggingface.co/docs/transformers/perplexity)

## [`llama-bench`](tools/llama-bench)

#### Benchmark the performance of the inference for various parameters.

- <details open>
    <summary>Run default benchmark</summary>

    ```bash
    llama-bench -m model.gguf

    # Output:
    # | model               |       size |     params | backend    | threads |          test |                  t/s |
    # | ------------------- | ---------: | ---------: | ---------- | ------: | ------------: | -------------------: |
    # | qwen2 1.5B Q4_0     | 885.97 MiB |     1.54 B | Metal,BLAS |      16 |         pp512 |      5765.41 ± 20.55 |
    # | qwen2 1.5B Q4_0     | 885.97 MiB |     1.54 B | Metal,BLAS |      16 |         tg128 |        197.71 ± 0.81 |
    #
    # build: 3e0ba0e60 (4229)
    ```

    </details>

## [`llama-simple`](examples/simple)

#### A minimal example for implementing apps with `llama.cpp`. Useful for developers.

- <details>
    <summary>Basic text completion</summary>

    ```bash
    llama-simple -m model.gguf

    # Hello my name is Kaitlyn and I am a 16 year old girl. I am a junior in high school and I am currently taking a class called "The Art of
    ```

    </details>


## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- See [good first issues](https://github.com/ggml-org/llama.cpp/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) for tasks suitable for first contributions
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information
- Make sure to read this: [Inference at the edge](https://github.com/ggml-org/llama.cpp/discussions/205)
- A bit of backstory for those who are interested: [Changelog podcast](https://changelog.com/podcast/532)

## Other documentation

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development documentation

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)

#### Seminal papers and background on the models

If your issue is with model generation quality, then please at least scan the following links and papers to understand the limitations of LLaMA models. This is especially important when choosing an appropriate model size and appreciating both the significant and subtle differences between LLaMA models and ChatGPT:
- LLaMA:
    - [Introducing LLaMA: A foundational, 65-billion-parameter large language model](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/)
    - [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- GPT-3
    - [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- GPT-3.5 / InstructGPT / ChatGPT:
    - [Aligning language models to follow instructions](https://openai.com/research/instruction-following)
    - [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)

## XCFramework
The XCFramework is a precompiled version of the library for iOS, visionOS, tvOS,
and macOS. It can be used in Swift projects without the need to compile the
library from source. For example:
```swift
// swift-tools-version: 5.10
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "MyLlamaPackage",
    targets: [
        .executableTarget(
            name: "MyLlamaPackage",
            dependencies: [
                "LlamaFramework"
            ]),
        .binaryTarget(
            name: "LlamaFramework",
            url: "https://github.com/ggml-org/llama.cpp/releases/download/b5046/llama-b5046-xcframework.zip",
            checksum: "c19be78b5f00d8d29a25da41042cb7afa094cbf6280a225abe614b03b20029ab"
        )
    ]
)
```
The above example is using an intermediate build `b5046` of the library. This can be modified
to use a different version by changing the URL and checksum.

## Completions
Command-line completion is available for some environments.

#### Bash Completion
```bash
$ build/bin/llama-cli --completion-bash > ~/.llama-completion.bash
$ source ~/.llama-completion.bash
```
Optionally this can be added to your `.bashrc` or `.bash_profile` to load it
automatically. For example:
```console
$ echo "source ~/.llama-completion.bash" >> ~/.bashrc
```

## Dependencies

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [stb-image](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [miniaudio.h](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
