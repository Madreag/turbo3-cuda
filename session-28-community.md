# Session 28 — RELEASE: Full Metrics, Clean Repo, Community Post, PR

## READ FIRST (MANDATORY)

1. **Read `AGENTS.md`** — follow ALL rules.
2. **Read this entire prompt** before starting any work.
3. Create branch **`session/28-community`** from `release/cuda-optimized`.
4. GPU: RTX 5090 32GB (SM120). CUDA 12.8. WSL2.

---

## CONTEXT

Sessions 25-27 are complete. Code is stable. S27C added KV serialization fix + auto Boundary V.

**This session has two goals:**
1. Collect EVERY metric TheTom reports so we have complete parity (see checklist below)
2. Clean the repo, write the Discussion post, prepare the upstream PR

**TheTom's repos for reference:**
- Python/docs: `/home/erol/ai/thetom/turboquant_plus/`
- C++ llama.cpp: `/home/erol/ai/thetom/llama-cpp-turboquant/` (branch `feature/turboquant-kv-cache`)
- His benchmark script: `/home/erol/ai/thetom/turboquant_plus/scripts/turbo-quick-bench.sh`

---

## THE METRICS CHECKLIST

TheTom reports all of these in his README. We need every one.

| # | Metric | Status | What To Run (5090) |
|:-:|--------|:------:|-------------------|
| 1 | PPL ctx=512 (all types) | HAVE | Already collected |
| 2 | PPL ctx=2048 (all types) | HAVE | Already collected |
| 3 | PPL ctx=8K/32K (50-chunk wikitext-103) | HAVE | S27 data |
| 4 | **KL divergence vs f16** | **NEED** | Task 2 below |
| 5 | Decode short tg128 (all types) | HAVE | S26 data |
| 6 | Decode 32K (all types) | HAVE | S26 data |
| 7 | **Prefill context scaling (pp512→pp32K)** | **NEED** | Task 3 below |
| 8 | **Sparse V ON/OFF delta** | **NEED** | Task 4 below |
| 9 | **Norm correction impact** | **NEED** | Task 5 below |
| 10 | **Block-size ablation (32 vs 128)** | **NEED** | Task 6 below |
| 11 | **Asymmetric K/V quality matrix** | **NEED** | Task 7 below |
| 12 | NIAH single needle (depth × context) | HAVE | S27 5090 + S26 3090 Ti |
| 13 | NIAH multi-key (RULER) | PARTIAL | S26 3090 Ti only |
| 14 | Skip rate per layer | HAVE | S27 data |
| 15 | **TheTom bench comparison (our build vs his)** | **NEED** | Task 8 below |

Items 4-11 and 15 are NEW tests to run on the 5090. Items 12-13 also need 3090 Ti/4090M runs (separate prompts).

---

## TASK 1: Clean the Release Branch

Remove all dev files before any public-facing work.

```bash
git checkout release/cuda-optimized

# Remove session prompts
git rm -f session-*.md

# Remove test result JSON (keep test scripts)
git rm -f quality-tests/*_results_*.json quality-tests/skip_rate_*.json quality-tests/passkey_results_*.json

# Remove dev files
git rm -f AGENTS.md

git commit -m "chore: clean release branch for public release"
git push myfork release/cuda-optimized
```

**Keep**: README.md, quality-tests/*.py, quality-tests/*.sh, all source code.

---

## TASK 2: KL Divergence vs f16

TheTom reports KLD for all types. We need the same.

Create `quality-tests/kl_divergence.py`. For 100 wikitext-2 prompts (256 tokens each):
1. Start server with f16 KV → request logprobs (`/v1/completions`, `logprobs: 10`, `max_tokens: 1`)
2. Restart with each turbo type → request same logprobs
3. Compute: KLD, top-p agreement %, delta-p RMS

**Use `max_tokens: 1`** — avoids Qwen 3.5 thinking issue entirely.

TheTom's reference numbers (M5 Max, MoE):
| Type | KLD | Same top-p |
|------|-----|-----------|
| q8_0 | 0.001549 | 98.43% |
| turbo4 | 0.009633 | 95.98% |
| turbo3 | 0.016145 | 94.31% |

Run on 27B Q6_K. If time permits, also MoE (35B-A3B Q4_K_M).

---

## TASK 3: Prefill Context Scaling

TheTom shows prefill tok/s at 2K, 4K, 8K, 16K, 32K. We don't have this.

```bash
MODEL=/home/erol/ai/turboquant/models/opus-v2-Q6_K.gguf

for CTX in 512 4096 8192 16384 32768; do
  for TYPE in q8_0 turbo4 turbo3 turbo2 turbo1.5; do
    echo "=== $TYPE pp$CTX ==="
    ./build/bin/llama-bench -m $MODEL -fa 1 -ctk $TYPE -ctv $TYPE -ngl 99 -t 1 -r 3 -p $CTX -n 0 -mmp 0 2>&1 | grep "pp$CTX"
  done
done
```

Build table:
| Context | q8_0 | turbo4 | turbo3 | turbo2 | turbo1.5 |
|---------|------|--------|--------|--------|----------|
| pp512 | | | | | |
| pp4096 | | | | | |
| pp8192 | | | | | |
| pp16384 | | | | | |
| pp32768 | | | | | |

---

## TASK 4: Sparse V ON/OFF Delta

TheTom claims "Sparse V ON/OFF delta = 0.000" on PPL. We need to prove the same for CUDA.

Temporarily disable sparse V (set threshold to 0 or very small), rebuild, run PPL:

```cuda
// In fattn-vec.cuh, temporarily change:
constexpr float sparse_v_threshold_f = V_is_low_bpv ? 1e-2f : 5e-3f;
// To:
constexpr float sparse_v_threshold_f = 0.0f;  // TEMP: sparse V disabled
```

Run:
```bash
# turbo3 PPL at ctx=512 and ctx=2048 — should be IDENTICAL to sparse V enabled
# turbo3 speed at 32K — should be SLOWER (this measures the speed gain)
```

Record:
| Metric | Sparse V ON | Sparse V OFF | Delta |
|--------|:-----------:|:------------:|:-----:|
| turbo3 PPL 512 | 6.852 | ? | should be 0.000 |
| turbo3 PPL 2048 | 5.674 | ? | should be 0.000 |
| turbo3 32K tok/s | 56.28 | ? | should be -X% |

Revert after testing.

---

## TASK 5: Norm Correction Impact

TheTom claims -1.17% PPL on CUDA. We have norm correction (from HyperionMS2040's S26 block-128 fix) but never measured the delta.

Check if norm correction can be toggled. In `set-rows.cu`, look for:
```cuda
const float corrected_norm = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
```

Temporarily change to `const float corrected_norm = grp_norm;` (disable correction), rebuild, run PPL. Compare.

| Metric | With Correction | Without | Delta |
|--------|:-:|:-:|:-:|
| turbo3 PPL 512 | 6.852 | ? | should be worse without |

Revert after testing.

---

## TASK 6: Block-Size Ablation (32 vs 128)

TheTom documents 4.6x vs 5.12x compression. We standardized on 128 but should show the comparison.

Temporarily revert `QK_TURBO3=32` and `QK_TURBO2=32` in `ggml-common.h`. Also revert the SET_ROWS warp-to-block changes (restore `blk = blk_base + warp_id`). Rebuild, run PPL + speed.

```bash
# Block-32: PPL + speed
./build/bin/llama-perplexity -m $MODEL -f $WIKI -c 512 -ctk turbo3 -ctv turbo3 -fa on --chunks 8 -ngl 99 --no-mmap
./build/bin/llama-bench -m $MODEL -fa 1 -ctk turbo3 -ctv turbo3 -d 32768 -ngl 99 -t 1 -r 3 -p 0 -n 32 -mmp 0
```

Record:
| Block Size | bpv | Compression | PPL 512 | 32K tok/s |
|:----------:|:---:|:-----------:|:-------:|:---------:|
| 32 | 3.50 | 4.57x | ? | ? |
| 128 | 3.125 | 5.12x | 6.852 | 56.28 |

Revert to block-128 after testing. This is reference data for the README.

---

## TASK 7: Asymmetric K/V Quality Matrix

TheTom documents which asymmetric combos work and which are catastrophic. We need the same.

```bash
MODEL=/home/erol/ai/turboquant/models/opus-v2-Q6_K.gguf
WIKI=$(find /home/erol/ai/turboquant -name "wiki.test.raw" 2>/dev/null | head -1)

# Key combos to test:
for K in q8_0 turbo4 turbo3 turbo2; do
  for V in q8_0 turbo4 turbo3 turbo2; do
    echo "=== K=$K V=$V ==="
    ./build/bin/llama-perplexity -m $MODEL -f $WIKI -c 512 -ctk $K -ctv $V -fa on --chunks 8 -ngl 99 --no-mmap 2>&1 | grep "Final"
  done
done
```

Build matrix:
| K \ V | q8_0 | turbo4 | turbo3 | turbo2 |
|-------|:----:|:------:|:------:|:------:|
| q8_0 | baseline | | | |
| turbo4 | | | | |
| turbo3 | | | | |
| turbo2 | | | | |

This takes ~16 PPL runs × 5 min = ~1.5 hours.

---

## TASK 8: TheTom's Benchmark Comparison (Our Build vs His)

Run his standardized script on both builds with the same model.

```bash
# Our build (already built)
# Adapt his script for our build paths
cp /home/erol/ai/thetom/turboquant_plus/scripts/turbo-quick-bench.sh quality-tests/

# His build
cd /home/erol/ai/thetom/llama-cpp-turboquant
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j$(nproc)
```

Run on same model (9B Q8_0 or 27B Q6_K), record side-by-side.

---

## TASK 9: Write Discussion #20969 Post

With ALL metrics collected, write the post. Include every table from the checklist.

**Structure**:
1. Headline: ALL types beat q8_0 at both short and 32K. 5.12x turbo3, 7.53x turbo2.
2. Performance table (S26 final)
3. Prefill context scaling (Task 3)
4. Quality: PPL + KLD + wikitext-103 50-chunk
5. Sparse V validation (ON/OFF delta = 0, speed gain quantified)
6. NIAH matrices (5090 + 3090 Ti)
7. Cross-GPU: 1,351+ iterations, 3090 Ti stock + OC, 4090M
8. Block-128 ablation (Task 6)
9. Asymmetric matrix (Task 7)
10. TheTom comparison (Task 8)
11. Limitations (2-bit long-ctx degradation, SM120 D=256 LUT)
12. Configuration recommendations
13. Attribution

Save to `/mnt/c/vaults/forge/09 Community/discussion_20969_final.md`.

---

## TASK 10: Prepare Upstream PR

Same as before — squash logical features, include speed+PPL data per commit, save draft.

Save to `/mnt/c/vaults/forge/09 Community/upstream_pr_draft.md`.

---

## TASK 11: Update README with All New Metrics

Add sections for:
- KL divergence table
- Prefill context scaling table
- Sparse V ON/OFF delta
- Block-size ablation
- Asymmetric K/V matrix (or key combos)
- Norm correction impact note

---

## TASK 12: Check Dump Folder + Update Vault + Push

```bash
ls /mnt/c/vaults/dump/*s28* /mnt/c/vaults/dump/*comparison* 2>/dev/null
```

Update vault: Session 28 note, Benchmark Hub, Dashboard, Roadmap S28 DONE. Push to release.

---

## EXECUTION ORDER

1. Clean release branch (Task 1) — 15 min
2. Prefill context scaling (Task 3) — 30 min
3. Sparse V ON/OFF delta (Task 4) — 30 min
4. Norm correction impact (Task 5) — 30 min
5. Block-size ablation (Task 6) — 45 min (requires rebuild twice)
6. Asymmetric K/V matrix (Task 7) — 1.5 hours
7. KL divergence (Task 2) — 3 hours
8. TheTom build comparison (Task 8) — 30 min
9. Write Discussion post (Task 9) — 2 hours
10. PR draft (Task 10) — 2 hours
11. README update (Task 11) — 1 hour
12. Vault + push (Task 12) — 30 min

**Total: ~13 hours**

---

## MODELS

| Model | Path | Use For |
|-------|------|---------|
| Qwen 3.5 27B Q6_K | `models/opus-v2-Q6_K.gguf` | PPL, speed, KLD, prefill, all metrics |
| Qwen 3.5 9B Q8_0 | `models/Qwen3.5-9B-Q8_0.gguf` | Generation, NIAH, TheTom bench comparison |
| MoE 35B-A3B Q4_K_M | `models/Qwen3.5-35B-A3B-Q4_K_M.gguf` | MoE KLD (if time) |

---

## SUCCESS CRITERIA

- [ ] Release branch clean
- [ ] ALL 15 metrics from checklist collected
- [ ] KL divergence measured
- [ ] Prefill context scaling table
- [ ] Sparse V ON/OFF delta proven (0.000 PPL, +X% speed)
- [ ] Norm correction impact measured
- [ ] Block-size ablation documented
- [ ] Asymmetric K/V matrix
- [ ] TheTom build comparison
- [ ] Discussion post drafted
- [ ] PR drafted
- [ ] README has all new sections
- [ ] Vault updated, pushed to release
