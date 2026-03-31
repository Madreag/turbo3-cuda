# Session 27 — QUALITY: Norm Correction, Validation Infrastructure, Feature Parity

## READ FIRST (MANDATORY)

1. **Read `AGENTS.md`** — contains ALL rules, architecture, dead ends, file locations, benchmarking protocol. Follow it EXACTLY.
2. **Read this entire prompt** before starting any work.
3. Create branch **`session/27-quality`** from `session/26-blast` (or `release/cuda-optimized` if S26 is merged).
4. The Obsidian vault is at `/mnt/c/vaults/forge/` — search it for any context you need.
5. GPU is an RTX 5090 32GB (SM120). CUDA 12.8. WSL2 Ubuntu 24.04.

---

## CONTEXT — WHERE WE ARE AFTER S26

### What S26 Should Have Done
- Fixed SM120 D=256 generation bug (turbo types producing empty output on Qwen 3.5)
- Validated block-128 storage on CUDA (QK_TURBO3=128, QK_TURBO2=128)
- Corrected bpv numbers in README (turbo3: 3.50 → 3.125 at block-128)
- Verify these are done before starting S27. If S26 items are incomplete, finish them first.

### The Feature Parity Gap
TheTom's Metal implementation (`turboquant_plus`) has several features we lack. His repo is at:
```
/home/erol/ai/turboquant/research/llama-cpp-turboquant/.trash/research/repos/TheTom-turboquant_plus/
```

**Session 27 closes the QUALITY gap.** Session 28 will close the METRICS and COMMUNITY gaps.

---

## TASK 1: Norm Correction (spiritbuun) — THE BIG QUALITY WIN

### Background
Norm correction is a shared effort: **spiritbuun** implemented it for turbo4, **TheTom** independently implemented it for turbo3 (different repos). TheTom reports **-1.17% PPL on CUDA** with norm correction applied. We don't have it.

### What Norm Correction Does
After quantizing a 128-element rotation group to centroids, the reconstructed vector has a slightly different L2 norm than the original:

```
original:       x_rotated = WHT(x)
quantized:      x_quant = centroids[indices]
reconstruction: ||x_quant|| ≠ ||x_rotated|| (quantization changes the norm)
```

Norm correction rescales the reconstruction:
```
grp_norm  = ||x_rotated||        (the original rotated vector norm)
recon_norm = ||x_quant||          (the reconstructed centroid vector norm)
corrected_norm = grp_norm / recon_norm
```

The `corrected_norm` replaces the raw group norm stored in the block struct. During dequant, multiplying centroids by `corrected_norm` exactly matches the original vector's magnitude.

### Where To Implement
The norm correction happens in **SET_ROWS** (the quantization kernel), NOT in the FA decode kernel. The decode kernel just reads `block.norm` and multiplies — it doesn't need to change.

**File**: `ggml/src/ggml-cuda/set-rows.cu`

For each turbo type's SET_ROWS kernel, after quantizing the group to centroid indices:
1. Compute `recon_norm = ||centroids[indices]||` (L2 norm of the quantized reconstruction)
2. Store `corrected_norm = grp_norm / recon_norm` instead of `grp_norm` in the block's norm field

### How To Verify
```
grp_norm = sqrt(sum(x_rotated[i]^2 for i in group))
recon = [centroids[idx[i]] for i in group]
recon_norm = sqrt(sum(recon[i]^2 for i in group))
corrected = grp_norm / recon_norm
# Store `corrected` as the block norm
```

### Testing Protocol
1. Run PPL ctx=512 and ctx=2048 for turbo3 BEFORE the change (establish baseline on current build)
2. Implement norm correction in SET_ROWS for turbo3
3. Run PPL ctx=512 and ctx=2048 — expect **improvement** (lower PPL than before, possibly lower than q8_0)
4. If PPL improves: implement for turbo2, turbo4, turbo1.5
5. Run full regression suite (speed should be unchanged — SET_ROWS is not in the decode path)
6. Commit with PPL data

### Reference & Attribution
- **TheTom**: norm correction for turbo3 (his repo `docs/quality-benchmarks.md`)
- **spiritbuun**: norm correction for turbo4 (separate CUDA fork)
- The corrected norm is stored in the existing `norm` field — no struct changes needed
- HyperionMS2040's block-128 SET_ROWS fix (S26 commit) may already include norm correction — check first

---

## TASK 2: Download Wikitext-103 and Run 50-Chunk Validation

### Why
TheTom validated sparse V with 50 chunks of wikitext-103 at 32K context (CI ±0.021). Our validation used 8 chunks of wikitext-2 at up to 16K. His methodology is the gold standard.

### Steps
1. Download wikitext-103-raw:
   ```bash
   cd /home/erol/ai/turboquant
   wget https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1/wiki.test.raw
   # Or via Python:
   # python3 -c "from datasets import load_dataset; d=load_dataset('wikitext','wikitext-103-raw-v1',split='test'); open('wikitext-103-raw/wiki.test.raw','w').write('\n'.join(d['text']))"
   ```

2. Run PPL at multiple contexts with 50 chunks:
   ```bash
   MODEL=/home/erol/ai/turboquant/models/opus-v2-Q6_K.gguf
   WIKI103=/path/to/wikitext-103-raw/wiki.test.raw

   # For each context length:
   for CTX in 512 2048 8192 32768; do
     for TYPE in turbo3 turbo2 turbo1.5 q8_0; do
       ./build/bin/llama-perplexity -m $MODEL -f $WIKI103 -c $CTX \
         -ctk $TYPE -ctv $TYPE -fa on --chunks 50 -ngl 99 --no-mmap
     done
   done
   ```

3. Record results in a table:
   | Type | ctx=512 | ctx=2048 | ctx=8192 | ctx=32768 |
   |------|---------|----------|----------|-----------|
   | q8_0 | | | | |
   | turbo3 | | | | |
   | turbo2 | | | | |
   | turbo1.5 | | | | |

4. **Critical check**: Do turbo types with sparse V at 5e-3/1e-2 show ANY PPL degradation vs q8_0 at 32K that wasn't present at 512? If yes, the threshold may be too aggressive for very long context.

### Runtime Estimate
50 chunks at 32K = ~1.6M tokens per type. At ~150 tok/s PPL processing: ~3 hours per type, ~12 hours total for 4 types. Run overnight.

---

## TASK 3: Port Skip Rate Measurement

### What
TheTom's `measure_skip_rate.py` directly measures what percentage of V positions are skipped by the sparse V threshold, per layer. This tells us whether our aggressive threshold (5e-3 for turbo3, 1e-2 for turbo2) actually skips more positions than his conservative 1e-6.

### How
His script uses PyTorch with `output_attentions=True` on a small model (Qwen3-1.7B). We have transformers + torch installed in the tq conda env.

1. Copy and adapt his script:
   ```
   cp /home/erol/ai/turboquant/research/llama-cpp-turboquant/.trash/research/repos/TheTom-turboquant_plus/scripts/measure_skip_rate.py quality-tests/
   ```

2. Run at multiple thresholds:
   ```bash
   # Measure skip rates at our thresholds AND TheTom's
   for THRESHOLD in 1e-6 5e-3 1e-2; do
     python3 quality-tests/measure_skip_rate.py --threshold $THRESHOLD --contexts 512,2048,4096,8192,32768
   done
   ```

3. Record per-layer skip rates. Expected:
   - 1e-6 (TheTom's): low skip rate, ~9% at 512, ~28% at 4K
   - 5e-3 (our turbo3): higher skip rate
   - 1e-2 (our turbo2): highest skip rate

4. The skip rate data tells us exactly how much V compute we're saving and validates the speed improvements.

---

## TASK 4: Create Automated Quality Gate

### What
A single script that runs before any commit/merge to verify no quality or speed regressions. TheTom has `turbo-quality-gate.sh`. We need our own CUDA version.

### Create `quality-tests/quality-gate.sh`:
```bash
#!/bin/bash
# TurboQuant CUDA Quality Gate
# Run before any merge to release/cuda-optimized
# Exit 0 = PASS, Exit 1 = FAIL

set -e
MODEL=${MODEL:-/home/erol/ai/turboquant/models/opus-v2-Q6_K.gguf}
WIKI=$(find /home/erol/ai/turboquant -name "wiki.test.raw" 2>/dev/null | head -1)

echo "=== TurboQuant Quality Gate ==="

# 1. PPL check (turbo3 ctx=512)
PPL=$(./build/bin/llama-perplexity -m $MODEL -f $WIKI -c 512 -ctk turbo3 -ctv turbo3 -fa on --chunks 8 -ngl 99 --no-mmap 2>&1 | grep "Final estimate" | awk '{print $4}')
echo "turbo3 PPL ctx=512: $PPL"
if (( $(echo "$PPL > 6.89" | bc -l) )); then
  echo "FAIL: turbo3 PPL $PPL > 6.89 reject threshold"
  exit 1
fi

# 2. Speed check (turbo3 short)
SPEED=$(./build/bin/llama-bench -m $MODEL -fa 1 -ctk turbo3 -ctv turbo3 -d 0 -ngl 99 -t 1 -r 3 -p 0 -n 128 -mmp 0 2>&1 | grep "tg128" | awk '{print $(NF-2)}')
echo "turbo3 short: $SPEED tok/s"
if (( $(echo "$SPEED < 55.0" | bc -l) )); then
  echo "FAIL: turbo3 short $SPEED < 55.0 tok/s"
  exit 1
fi

# 3. Speed check (turbo3 32K)
SPEED32=$(./build/bin/llama-bench -m $MODEL -fa 1 -ctk turbo3 -ctv turbo3 -d 32768 -ngl 99 -t 1 -r 3 -p 0 -n 32 -mmp 0 2>&1 | grep "tg32" | awk '{print $(NF-2)}')
echo "turbo3 32K: $SPEED32 tok/s"
if (( $(echo "$SPEED32 < 45.0" | bc -l) )); then
  echo "FAIL: turbo3 32K $SPEED32 < 45.0 tok/s"
  exit 1
fi

echo ""
echo "=== QUALITY GATE: PASS ==="
echo "PPL=$PPL, Short=$SPEED, 32K=$SPEED32"
```

The gate checks 3 things: PPL within threshold, short speed above minimum, 32K speed above minimum. Takes ~5 minutes to run.

---

## TASK 5: Enhance NIAH Test Suite

### What
Our `quality-tests/niah_test.py` is basic (one needle, 5 depths, simple check). TheTom's is 44KB with multi-key, distractors, and depth×context matrices.

### Enhancements
1. **Add multi-key retrieval** (RULER MK-NIAH): Insert 3 target needles + 3 distractor needles. Model must retrieve all 3 targets and ignore distractors.
2. **Add N=10 needle stability**: Run 10 different needles at each depth×context position. Report N/10 accuracy.
3. **Add comparison mode**: Run q8_0 baseline first, then turbo types, output side-by-side matrix.
4. **Fix max_tokens**: Set to 500 for Qwen 3.5 thinking models (the S25 bug).

### Reference
TheTom's NIAH scripts: `scripts/niah_test.py` and `proof/niah/` in his repo.

---

## TASK 6: Update Documentation and Vault

### AGENTS.md
- Add S27 results (norm correction PPL improvement, skip rate data, quality gate)
- Update sparse V section with 50-chunk wikitext-103 validation results

### README.md
- Add norm correction to the quality story if PPL improves
- Add quality validation methodology section (mention 50-chunk wikitext-103, NIAH)

### Vault
- `01 Sessions/Session 27.md` — create session note
- `03 Benchmarks/Benchmark Hub.md` — add 50-chunk PPL data, skip rate data
- `00 Dashboard/Project Status.md` — update

---

## MODELS

| Model | Path | Use For |
|-------|------|---------|
| Qwen 3.5 27B Q6_K | `models/opus-v2-Q6_K.gguf` | PPL, speed benchmarks |
| Qwen 3.5 9B Q8_0 | `models/Qwen3.5-9B-Q8_0.gguf` | NIAH tests |
| Llama-3.3-8B Q6_K | `models/allura-forge_Llama-3.3-8B-Instruct-Q6_K.gguf` | NIAH D=128 control |

Model paths relative to `/home/erol/ai/turboquant/`.

---

## SUCCESS CRITERIA

- [ ] Norm correction implemented in SET_ROWS for all 4 turbo types
- [ ] PPL improved (or at minimum unchanged) for all types
- [ ] 50-chunk wikitext-103 PPL at 32K recorded for all types + q8_0
- [ ] Skip rate measured at 1e-6, 5e-3, 1e-2 thresholds
- [ ] Quality gate script created and passing
- [ ] NIAH test enhanced with multi-key and stability tests
- [ ] All docs updated

---

## ESTIMATED EFFORT

| Task | Time |
|------|------|
| Norm correction implementation + testing | 3-4 hours |
| Wikitext-103 download + 50-chunk validation | 1 hour setup + 12 hours runtime (overnight) |
| Skip rate measurement | 1-2 hours |
| Quality gate script | 1 hour |
| NIAH enhancement | 2-3 hours |
| Documentation | 1 hour |
| **Total** | **~10 hours active + 12 hours overnight** |
