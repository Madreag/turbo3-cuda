# Session 27B — Sparse V Threshold Tuning: Speed vs Quality Tradeoffs

## READ FIRST (MANDATORY)

1. **Read `AGENTS.md`** — follow ALL rules exactly.
2. **Read this entire prompt** before starting any work.
3. You are on branch **`session/27-quality`**.
4. GPU is an RTX 5090 32GB (SM120). CUDA 12.8. WSL2 Ubuntu 24.04.

---

## CONTEXT — WHAT S27 FOUND

### The Problem
Session 27's 50-chunk wikitext-103 validation revealed that **turbo2 and turbo1.5 PPL degrades significantly at long context**:

| Type | Threshold | ctx=512 | ctx=2048 | ctx=8192 | ctx=32K |
|------|:---------:|:-------:|:--------:|:--------:|:------:|
| q8_0 | — | 6.1825 | 6.4595 | 7.5932 | 8.9819 |
| turbo4 | 5e-3 | +0.77% | +0.53% | +1.22% | +2.35% |
| turbo3 | 5e-3 | +0.77% | +1.39% | +1.81% | +2.84% |
| **turbo2** | **1e-2** | **+3.37%** | **+4.77%** | **+8.06%** | **+11.93%** |
| **turbo1.5** | **1e-2** | **+6.34%** | **+9.72%** | **+13.42%** | **+23.88%** |

turbo3/turbo4 at 5e-3 threshold: healthy (< 3% delta even at 32K).
turbo2/turbo1.5 at 1e-2 threshold: **degrading badly** (+12% and +24% at 32K).

### The Skip Rate Data
| Threshold | ctx=512 | ctx=2048 | ctx=4096 | ctx=8192 |
|:---------:|:-------:|:--------:|:--------:|:--------:|
| 1e-6 | 9.1% | 20.7% | 28.4% | 26.9% |
| **5e-3** | **96.8%** | **99.3%** | **99.7%** | **99.8%** |
| **1e-2** | **98.4%** | **99.7%** | **99.9%** | **99.9%** |

At 5e-3 and 1e-2, **97-99.9%** of V positions are skipped. The speed gain comes from not reading V data — but at 1e-2, too much signal is lost for turbo2/turbo1.5.

### The Question
**Is the growing PPL delta from the threshold, or from inherent 2-bit quantization compounding over longer sequences?**

If threshold → lowering it will fix quality at the cost of some speed.
If inherent → nothing we can do about it.

---

## TASK 1: Control Test — turbo2/turbo1.5 with Threshold Disabled

Run turbo2 and turbo1.5 at 32K context with threshold=1e-6 (effectively disabled). Compare against the 1e-2 results.

### Implementation
In `fattn-vec.cuh`, temporarily change:
```cuda
constexpr float sparse_v_threshold_f = V_is_low_bpv ? 1e-2f : 5e-3f;
```
To:
```cuda
constexpr float sparse_v_threshold_f = 1e-6f;  // TEMP: control test, all types conservative
```

### Build and Test
```bash
# Rebuild
/home/erol/miniconda3/envs/tq/bin/cmake --build build -j$(nproc)

# Run turbo2 PPL at 512 and 32K
MODEL=/home/erol/ai/turboquant/models/opus-v2-Q6_K.gguf
WIKI103=/home/erol/ai/turboquant/wikitext-103-raw-v1/wiki.test.raw

./build/bin/llama-perplexity -m $MODEL -f $WIKI103 -c 512 -ctk turbo2 -ctv turbo2 -fa on --chunks 50 -ngl 99 --no-mmap
./build/bin/llama-perplexity -m $MODEL -f $WIKI103 -c 32768 -ctk turbo2 -ctv turbo2 -fa on --chunks 50 -ngl 99 --no-mmap

# Same for turbo1.5
./build/bin/llama-perplexity -m $MODEL -f $WIKI103 -c 512 -ctk turbo1.5 -ctv turbo1.5 -fa on --chunks 50 -ngl 99 --no-mmap
./build/bin/llama-perplexity -m $MODEL -f $WIKI103 -c 32768 -ctk turbo1.5 -ctv turbo1.5 -fa on --chunks 50 -ngl 99 --no-mmap
```

### Expected Results

**If threshold is the cause:**
- turbo2 ctx=32K at 1e-6 will have MUCH lower delta than +11.93%
- turbo1.5 ctx=32K at 1e-6 will have MUCH lower delta than +23.88%
- → Fix: lower the threshold for these types

**If inherent to quantization:**
- Deltas will be similar even at 1e-6
- → Nothing we can do. Document the limitation.

---

## TASK 2: Threshold Sweep (if threshold IS the cause)

If Task 1 shows the threshold matters, sweep to find the optimal value:

```cuda
// Test these thresholds for turbo2:
// 1e-6, 1e-4, 1e-3, 5e-3, 1e-2
```

For each threshold, measure:
1. PPL at ctx=512 and ctx=32K (quality)
2. Speed at 32K (performance)

Build a tradeoff table:

| Threshold | turbo2 PPL 512 | turbo2 PPL 32K | turbo2 32K tok/s | Quality vs Speed |
|:---------:|:-:|:-:|:-:|:-:|
| 1e-6 | | | | baseline quality |
| 1e-4 | | | | |
| 1e-3 | | | | |
| 5e-3 | | | | |
| 1e-2 | 6.3908 | 10.0536 | ~58.48 | current |

Same sweep for turbo1.5.

### Finding the Sweet Spot
The optimal threshold is where:
- PPL delta at 32K stays under 5% vs q8_0 (acceptable quality)
- Speed gain is maximized

---

## TASK 3: Speed Impact of Threshold Changes

After finding the right threshold, measure the speed impact:

```bash
MODEL=/home/erol/ai/turboquant/models/opus-v2-Q6_K.gguf

# Warmup + measure for each type at the new threshold
./build/bin/llama-bench -m $MODEL -fa 1 -ctk turbo2 -ctv turbo2 -d 0 -ngl 99 -t 1 -r 1 -p 0 -n 128 -mmp 0 > /dev/null 2>&1
./build/bin/llama-bench -m $MODEL -fa 1 -ctk turbo2 -ctv turbo2 -d 0 -ngl 99 -t 1 -r 5 -p 0 -n 128 -mmp 0

./build/bin/llama-bench -m $MODEL -fa 1 -ctk turbo2 -ctv turbo2 -d 32768 -ngl 99 -t 1 -r 1 -p 0 -n 32 -mmp 0 > /dev/null 2>&1
./build/bin/llama-bench -m $MODEL -fa 1 -ctk turbo2 -ctv turbo2 -d 32768 -ngl 99 -t 1 -r 3 -p 0 -n 32 -mmp 0
```

### The Tradeoff Decision
Cross-reference the speed and quality data:

| Config | turbo2 32K Speed | turbo2 32K PPL delta | Acceptable? |
|--------|:----------------:|:-------------------:|:-----------:|
| 1e-2 (current) | ~58 tok/s | +11.93% | NO — too much quality loss |
| 5e-3 | ? | ? | ? |
| 1e-3 | ? | ? | ? |
| 1e-6 | ? | ? | quality baseline but slow |

---

## TASK 4: Implement the Optimal Thresholds

Based on the sweep data, update `fattn-vec.cuh` with the right thresholds:

```cuda
// Current:
constexpr bool V_is_low_bpv = (type_V == GGML_TYPE_TURBO2_0 || type_V == GGML_TYPE_TURBO1_5);
constexpr float sparse_v_threshold_f = V_is_low_bpv ? 1e-2f : 5e-3f;

// Maybe change to per-type thresholds:
constexpr float sparse_v_threshold_f =
    (type_V == GGML_TYPE_TURBO1_5) ? X :    // turbo1.5 needs lowest threshold (most quality-sensitive)
    (type_V == GGML_TYPE_TURBO2_0) ? Y :     // turbo2 needs moderate threshold
    5e-3f;                                     // turbo3/turbo4 stay at 5e-3 (proven healthy)
```

### After Committing
Run the full quality gate:
```bash
bash quality-tests/quality-gate.sh
```

---

## TASK 5: Re-run Wikitext-103 Validation with New Thresholds

After the threshold change, re-run the critical long-context PPL to verify the fix:

```bash
MODEL=/home/erol/ai/turboquant/models/opus-v2-Q6_K.gguf
WIKI103=/home/erol/ai/turboquant/wikitext-103-raw-v1/wiki.test.raw

for TYPE in turbo2 turbo1.5; do
  for CTX in 512 2048 8192 32768; do
    echo "=== $TYPE ctx=$CTX ==="
    ./build/bin/llama-perplexity -m $MODEL -f $WIKI103 -c $CTX \
      -ctk $TYPE -ctv $TYPE -fa on --chunks 50 -ngl 99 --no-mmap 2>&1 | grep "Final"
  done
done
```

The delta at 32K should now be under 5% for turbo2 and under 10% for turbo1.5.

---

## TASK 6: Cross-GPU Speed Validation

### Context from Other GPUs
S26 data just arrived from 3090 Ti and 4090M. Block-128 is a pure win everywhere:

**3090 Ti (SM86, stock):**
| Type | Short | 32K | vs S24 32K |
|------|:-----:|:---:|:----------:|
| q8_0 | 84.25 | 71.55 | 0.0% |
| turbo3 | 83.43 | 67.61 | **+6.4%** |
| turbo2 | 83.48 | 75.06 | **+4.1%** |
| turbo1.5 | 83.13 | 67.28 | **+4.6%** |
| turbo4 | 83.52 | 69.52 | +0.9% |

**3090 Ti OC (+2200 mem):** turbo2 32K = **80.18 tok/s** (+11.2% vs S24 stock). Linear memory bandwidth scaling confirmed.

**4090M (SM89):**
| Type | 32K | vs S24 32K |
|------|:---:|:----------:|
| turbo3 | 49.32 | **+9.7%** |
| turbo2 | 52.46 | **+9.7%** |
| turbo4 | 52.08 | **+17.1%** |
| turbo1.5 | 49.90 | **+12.5%** |

**All GPUs show zero regressions, all types generate correctly, PPL bit-exact.** Block-128 is confirmed across SM86/SM89/SM120.

Any threshold changes in this session should be re-validated on the 3090 Ti and 4090M afterward (update the dump folder prompts).

---

## TASK 7: Update Documentation

### README.md
- If thresholds change, update the Limitations section
- Add a quality validation section citing the 50-chunk wikitext-103 data
- Note the speed vs quality tradeoff for turbo2/turbo1.5

### AGENTS.md
- Update the sparse V threshold section with the new per-type values
- Add the wikitext-103 PPL table
- Add the skip rate data

### Vault
- `01 Sessions/Session 27.md` — create session note with all data
- `03 Benchmarks/Benchmark Hub.md` — add wikitext-103 PPL table, skip rates, threshold sweep
- `00 Dashboard/Project Status.md` — update

---

## MODELS

| Model | Path | Use For |
|-------|------|---------|
| Qwen 3.5 27B Q6_K | `models/opus-v2-Q6_K.gguf` | PPL, speed |
| Wikitext-103 raw | `wikitext-103-raw-v1/wiki.test.raw` | 50-chunk PPL validation |

---

## KEY INSIGHT: WHY THIS MATTERS

The S25 sparse V optimization gave us huge speed gains:
- turbo3 32K: +5-11%
- turbo2 32K: +13%
- 8B Llama: +28%

But S27 revealed that turbo2/turbo1.5 pay a **hidden quality tax at long context** that our short-context PPL tests didn't catch. The threshold that's optimal for turbo3 (5e-3) may be too aggressive for lower-bpv types that carry less information per V position.

The goal of this session is to find the right threshold for each type — maximizing speed while keeping quality degradation under 5% at 32K.

---

## SUCCESS CRITERIA

- [ ] Control test done: turbo2/turbo1.5 at 1e-6 threshold, PPL at 512 and 32K
- [ ] Root cause confirmed (threshold vs inherent)
- [ ] If threshold: optimal value found for turbo2 and turbo1.5
- [ ] Speed impact measured for the new threshold
- [ ] Wikitext-103 re-validated with new thresholds
- [ ] Quality gate passes
- [ ] Docs and vault updated
