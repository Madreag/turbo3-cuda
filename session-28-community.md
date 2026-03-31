# Session 28 — RELEASE: Clean Repo, Standardized Benchmarks, Community Post, PR

## READ FIRST (MANDATORY)

1. **Read `AGENTS.md`** — follow ALL rules.
2. **Read this entire prompt** before starting any work.
3. Create branch **`session/28-community`** from `release/cuda-optimized`.
4. GPU: RTX 5090 32GB (SM120). CUDA 12.8. WSL2.

---

## CONTEXT

Sessions 25-27 are complete. All bugs fixed, block-128 shipped, quality validated, 37 kernel optimizations attempted (all dead — 168-reg ceiling confirmed). The code is stable and release-ready.

**Now we need to**:
1. Clean the release branch (remove all dev files)
2. Run TheTom's standardized benchmark script on OUR build and HIS latest build (apples-to-apples)
3. Get standardized data from 3090 Ti and 4090M machines
4. Post to Discussion #20969 with the data
5. Prepare the upstream PR

---

## TASK 1: Clean the Release Branch (FIRST — before anything else)

The release branch has dev files that shouldn't be public. Remove them ALL.

### Files to Remove from Git Tracking

```bash
git checkout release/cuda-optimized

# Session prompts
git rm session-26-blast.md session-26-part2-bugfix.md session-27-quality.md session-27b-threshold.md session-28-community.md

# Quality test results (raw data — keep scripts, remove results)
git rm quality-tests/niah_results_*.json
git rm quality-tests/passkey_results_*.json
git rm quality-tests/skip_rate_*.json

# Dev files
git rm AGENTS.md

git commit -m "chore: clean release branch — remove session prompts, test results, dev files"
```

### Files to KEEP on Release
- `README.md` — the public face
- `quality-tests/niah_test.py` — useful for users to run their own tests
- `quality-tests/passkey_retrieval.py` — same
- `quality-tests/quality-gate.sh` — same
- `quality-tests/measure_skip_rate.py` — same
- `quality-tests/run_quality_suite.sh` — same
- All source code in `ggml/`, `src/`, `common/`, etc.

### Verify .gitignore Blocks Re-addition
Check that `.gitignore` has:
```
/[Ss]ession*
/SESSION*
/AGENTS.md
/CLAUDE.md
```

### After Cleaning
```bash
git push myfork release/cuda-optimized
```

---

## TASK 2: Run TheTom's Standardized Benchmark Script

### Why
TheTom's community uses `turbo-quick-bench.sh` as the standard. Running it on our build gives directly comparable data. Running it on HIS latest build gives an apples-to-apples comparison.

### Step A: Run on Our Build

```bash
# Copy TheTom's benchmark script to our repo
cp /home/erol/ai/turboquant/research/llama-cpp-turboquant/.trash/research/repos/TheTom-turboquant_plus/scripts/turbo-quick-bench.sh quality-tests/

# Adapt paths for our build
# The script expects: turbo-quick-bench.sh <model.gguf> [llama_dir]
# Our llama_dir is the repo root, build is at build/bin/

# Run on 27B Q6_K
bash quality-tests/turbo-quick-bench.sh --no-ref \
  /home/erol/ai/turboquant/models/opus-v2-Q6_K.gguf \
  /home/erol/ai/turboquant/turboquant-kv-cache

# Run on 9B Q8_0
bash quality-tests/turbo-quick-bench.sh --no-ref \
  /home/erol/ai/turboquant/models/Qwen3.5-9B-Q8_0.gguf \
  /home/erol/ai/turboquant/turboquant-kv-cache
```

**Note**: The script expects `build-turbo/bin/` — you may need to edit it to use `build/bin/` or create a symlink.

### Step B: Clone and Build TheTom's Latest, Run Same Script

```bash
cd /home/erol/ai/turboquant
git clone https://github.com/TheTom/llama-cpp-turboquant.git thetom-latest
cd thetom-latest
git checkout feature/turboquant-kv-cache

cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j$(nproc)

# Run same benchmark on same models
bash scripts/turbo-quick-bench.sh --no-ref \
  /home/erol/ai/turboquant/models/opus-v2-Q6_K.gguf \
  /home/erol/ai/turboquant/thetom-latest
```

### Step C: Side-by-Side Comparison Table

| Metric | Madreag (ours) | TheTom (latest) | Delta |
|--------|:-:|:-:|:-:|
| turbo3 PPL ctx=512 | | | |
| turbo3 decode tg128 | | | |
| turbo3 NIAH 3/3 | | | |
| turbo4 PPL ctx=512 | | | |
| turbo4 decode tg128 | | | |

This is the data that goes in the Discussion post.

---

## TASK 3: Prepare 3090 Ti and 4090M Benchmark Prompts

Create updated prompts in `/mnt/c/vaults/dump/` for remote machines to run TheTom's benchmark script.

### 3090 Ti Prompt (`/mnt/c/vaults/dump/3090ti_s28_bench.md`)

```markdown
# 3090 Ti — TheTom Standardized Benchmark

Clone BOTH repos, build both, run turbo-quick-bench.sh on same model.

## Our build:
git clone https://github.com/Madreag/turbo3-cuda.git madreag-build
cd madreag-build && git checkout release/cuda-optimized
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j$(nproc)

## TheTom's build:
git clone https://github.com/TheTom/llama-cpp-turboquant.git thetom-build
cd thetom-build && git checkout feature/turboquant-kv-cache
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j$(nproc)

## Run on 9B Q8_0:
bash scripts/turbo-quick-bench.sh --no-ref $MODEL madreag-build
bash scripts/turbo-quick-bench.sh --no-ref $MODEL thetom-build

Upload results to /mnt/c/vaults/dump/3090ti_s28_comparison.md
```

Same for 4090M with SM89.

---

## TASK 4: KL Divergence Measurement

Same as before — create `quality-tests/kl_divergence.py`, measure all types vs f16. Use `max_tokens: 1` to avoid thinking model issues. This data goes in the Discussion post.

---

## TASK 5: Cross-Format Sparse V

Apply sparse V to q8_0/q4_0 types. Must stay constexpr (S27 proved non-constexpr spills registers). Measure q8_0 speed at 32K before/after. If +3-5% with bit-exact PPL, this is an upstream llama.cpp contribution.

---

## TASK 6: Write Discussion #20969 Post

### Data Sources for the Post
1. **Speed**: S26 final table (all types, short + 32K) + TheTom comparison from Task 2
2. **Quality**: S27 wikitext-103 PPL table + KLD from Task 4
3. **NIAH**: 3090 Ti data (turbo3 86.4% beats q8_0) + 5090 data
4. **Cross-GPU**: 3090 Ti stock + OC, 4090M, 5090 (from dump folder)
5. **Compression**: Block-128 bpv (turbo3 5.12x, turbo2 7.53x)
6. **TheTom comparison**: Apples-to-apples from Task 2

### Structure
1. Headline numbers
2. Performance table (standardized via TheTom's bench script)
3. Quality table (PPL + KLD)
4. NIAH matrices
5. Cross-GPU data
6. What optimizations we did (brief, link to repo for details)
7. Honest limitations (2-bit degradation at long ctx, SM120 D=256 LUT disabled)
8. Configuration recommendations (reference TheTom's guide)
9. Attribution (TheTom, signalnine, spiritbuun, HyperionMS2040)
10. Link to repo + invite testing

### Save To
`/mnt/c/vaults/forge/09 Community/discussion_20969_final.md`

---

## TASK 7: Prepare Upstream PR

### What Goes in the PR to TheTom/llama-cpp-turboquant

**Include** (squashed into logical commits):
1. Sparse V type-adaptive thresholds (5e-3/1e-2)
2. Half-precision LUT (D≤128)
3. Block-128 storage + SET_ROWS fix (credit HyperionMS2040)
4. nthreads_KQ=8 all types
5. constexpr centroids
6. turbo4/turbo1.5 vec_dot Q_reg fix
7. D=256 LUT disable on SM120
8. L2 prefetch hints
9. `__expf` softmax
10. `__launch_bounds__(128, 3)`
11. 8-wide LUT scoring

**Do NOT include**: session files, vault, AGENTS.md, dead code removal hack, quality test JSON results

### PR Description
Title: `feat: CUDA optimizations — all types beat q8_0 at 32K, block-128 5.12x compression`

Body: link to Discussion #20969 post, summary table, attribution, test methodology.

### Save To
`/mnt/c/vaults/forge/09 Community/upstream_pr_draft.md`

---

## TASK 8: Update README for Release

### Add
- KLD section (Task 4 data)
- NIAH depth×context matrices (existing data — format for README)
- Community hardware: 3090 Ti (stock + OC), 4090M (from dump folder)
- Configuration recommendations (reference TheTom's guide)

### Remove/Clean
- Any references to session numbers in the main text (keep in Contributions section commits)
- Any stale numbers (verify all match S26 final)

---

## TASK 9: Update Vault and Push

- `01 Sessions/Session 28.md`
- `03 Benchmarks/Benchmark Hub.md` — KLD data, TheTom comparison
- `09 Community/discussion_20969_final.md`
- `09 Community/upstream_pr_draft.md`
- `00 Dashboard/Project Status.md`
- `08 Plans/Roadmap.md` — S28 DONE
- Push to `release/cuda-optimized`

---

## EXECUTION ORDER

1. **Clean release branch** (Task 1) — 15 min
2. **Run TheTom's bench on our build** (Task 2A) — 15 min
3. **Clone + build TheTom's latest, run bench** (Task 2B) — 30 min
4. **Write 3090 Ti / 4090M prompts** (Task 3) — 15 min
5. **KL divergence** (Task 4) — 3-4 hours
6. **Cross-format sparse V** (Task 5) — 2 hours
7. **Discussion post draft** (Task 6) — 2 hours
8. **PR draft** (Task 7) — 2 hours
9. **README update** (Task 8) — 1 hour
10. **Vault + push** (Task 9) — 30 min

**Total: ~12-14 hours**

---

## SUCCESS CRITERIA

- [ ] Release branch clean (no session prompts, test results JSON, AGENTS.md)
- [ ] TheTom's bench script run on our build + his build (comparison table)
- [ ] 3090 Ti + 4090M prompts written and placed in dump folder
- [ ] KLD measured for all types
- [ ] Cross-format sparse V tested on q8_0
- [ ] Discussion #20969 post drafted (Erol reviews before posting)
- [ ] Upstream PR drafted (Erol reviews before submitting)
- [ ] README fully updated for release
- [ ] Vault updated, release/cuda-optimized pushed
