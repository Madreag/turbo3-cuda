# Session 26 — BLAST: Block-128, Quality Validation, and Release Prep

## READ FIRST

1. Read `AGENTS.md` in this repo — it contains ALL rules, architecture, dead ends, file locations, and benchmarking protocol. Follow it exactly.
2. Read this entire prompt before starting any work.
3. You are on branch `session/26-blast`. The release branch is `release/cuda-optimized`.

---

## WHERE WE ARE

### Sessions 17-24B (Complete)
All 4 turbo types (turbo4/turbo3/turbo2/turbo1.5) are fully optimized with:
- Parallel SET_ROWS encoding (128 threads, warp intrinsics, WHT butterfly)
- Native VEC Flash Attention with LUT scoring (turbo3 8-centroid, turbo2 4-centroid)
- `__expf` fast-math softmax, constexpr centroids, nthreads_KQ=8 all types
- 36 asymmetric K×V combinations, 5 models validated (D=64/96/128/256)
- 3 GPUs validated (SM86/SM89/SM120), 1,121+ stability iterations, 0 failures

### Session 25 (Complete — `session/25-deep`)
60 micro-optimization iterations. 5 committed wins:
1. Sparse V threshold 1e-6 → 5e-3 (turbo3/4) and 1e-2 (turbo2/1.5) — turbo3 32K +5-11%, turbo2 +13%, 8B Llama +28%
2. Half-precision LUT (float → half in shmem) — +2.45% at 32K
3. Dead turbo4 LUT code removal — +4.4% (NVCC codegen sensitivity)
4. PPL bit-exact at all thresholds through ctx=16K
5. VEC kernel confirmed at optimization ceiling: 168 registers, 98.4% utilization, 33 code changes ALL failed

**Key S25 findings:**
- K type determines 32K speed, V type barely matters (sparse V skip makes V nearly free)
- turbo2 at 256K: 36.62 tok/s on consumer 5090
- turbo3 beats q8_0 at 32K (55.08 vs 53.06)
- turbo2 and turbo3 beat f16 at 65K+ (bandwidth savings from compressed KV)

### Quality Tests (In Progress)
Phase 1 (Passkey Retrieval) and Phase 2 (NIAH) running on both 5090 and 3090 Ti. Scripts at `quality-tests/`. Using Qwen3.5-9B Q8_0 model with `max_tokens: 500` (Qwen 3.5 is a thinking model — needs token budget for `<think>` before answering). Results in `quality-tests/*.json` when complete. **Check these results first — if turbo1.5 or turbo2 show retrieval failures, we need to lower the sparse V threshold.**

---

## WHAT SESSION 26 MUST DO

### Task 1: Check Quality Test Results (FIRST)

Before any code changes, check if the quality test results are in:

```bash
ls quality-tests/*.json
```

If results exist, analyze them:
- Compare q8_0 baseline accuracy vs turbo3/turbo2/turbo1.5 at each context × depth
- Any failures unique to a turbo type (not present in q8_0) = quality regression
- If turbo1.5 or turbo2 fail where q8_0 passes: lower the sparse V threshold for that type
- Report the full results matrix in the session log

Also check `/mnt/c/vaults/dump/` for 3090 Ti results.

### Task 2: Block-128 Storage Size (TheTom's Research — CUDA Validation)

**Background**: TheTom proved that changing `QK_TURBO3` from 32 to 128 eliminates 3 redundant norm values per rotation group. On Metal (Apple Silicon): PPL identical to 4 decimal places across 3 models, 2 hardware platforms, 7 cache configs. Compression improves from 4.57x to 5.12x for turbo3. Free lunch.

**What to do**: Validate on CUDA (SM120).

1. Edit `ggml/src/ggml-common.h`:
   ```c
   #define QK_TURBO3 128  // was 32
   #define QK_TURBO2 128  // was 32
   ```

2. Rebuild and run the full regression suite:
   - Short decode (tg128) for ALL turbo types
   - 32K decode (tg32) for ALL turbo types
   - PPL ctx=512 and ctx=2048 for turbo3
   - PPL ctx=512 for turbo2 and turbo1.5

3. Compare vs block_size=32 baseline:
   - PPL must be identical (to 4 decimal places, matching TheTom's Metal results)
   - Speed should be flat or better (TheTom saw +3-7% decode on bandwidth-constrained M2 Pro)

4. If PPL matches and speed is >= baseline: **commit the change**
5. If speed regresses: investigate NVCC codegen (our S25 experience shows NVCC is sensitive to source changes)

**IMPORTANT**: The VEC kernel LUT scoring loop iterates `D/QK_TURBO3` times. At block_size=32: 128/32=4 iterations. At block_size=128: 128/128=1 iteration. This changes the loop structure — NVCC may generate different code. Benchmark carefully.

**Also fix the bpv numbers**: Our README currently says turbo3 = 3.25 bpv. TheTom's paper shows the correct value at block_size=32 is 3.50 bpv (14 bytes per 32 elements). At block_size=128: 3.125 bpv. Update README and AGENTS.md with correct numbers after the change.

### Task 3: Fix README bpv Numbers

Regardless of block-128 outcome, the current bpv numbers in README.md are wrong. Verify each type's actual bpv by calculating from the block struct in `ggml-common.h`:

```
bpv = (sizeof(block_struct) * 8) / QK_TURBO_N
```

Update the main performance table, quality table, and all references.

### Task 4: Update AGENTS.md

Update `AGENTS.md` with:
- Session 25 results (sparse V threshold, half LUT, dead code removal)
- Session 26 results (block-128, quality validation)
- Any new dead ends discovered
- Correct bpv numbers
- Quality test methodology and results

### Task 5: Update Vault

After all tasks, update:
- `01 Sessions/Session 26.md` — create session note
- `03 Benchmarks/Benchmark Hub.md` — add S26 data
- `00 Dashboard/Project Status.md` — update status
- `08 Plans/Roadmap.md` — mark S26 items done, update next steps

### Task 6: Push to Release Branch

After all changes validated:
```bash
git checkout release/cuda-optimized
git merge session/26-blast
git push myfork release/cuda-optimized
```

---

## WHAT NOT TO DO

- Do NOT repeat S25 dead ends (33 code restructuring attempts all failed — see AGENTS.md Dead Ends)
- Do NOT change the VEC kernel code structure (168 registers = ceiling, any change causes spills)
- Do NOT use `__managed__` memory (kills SM86)
- Do NOT run benchmarks in background (Rule 11 — causes 176GB memory stacking → system crash)
- Do NOT use the Qwen 3.5 27B model (`opus-v2-Q6_K.gguf`) for quality tests — it generates `????` with FA enabled on our fork. Use Qwen3.5-9B Q8_0 or Llama-3.3-8B
- Do NOT skip PPL checks. EVER.

---

## MODELS

| Model | Path | Use For |
|-------|------|---------|
| Qwen 3.5 27B Q6_K | `/home/erol/ai/turboquant/models/opus-v2-Q6_K.gguf` | Speed benchmarks, PPL |
| Qwen 3.5 27B Q4_K_M | `/home/erol/ai/turboquant/models/Qwen3.5-27B-Q4_K_M.gguf` | Long-context speed |
| Qwen 3.5 9B Q8_0 | `/home/erol/ai/turboquant/models/Qwen3.5-9B-Q8_0.gguf` | Quality tests (passkey, NIAH) |
| Llama-3.3-8B Q6_K | `/home/erol/ai/turboquant/models/allura-forge_Llama-3.3-8B-Instruct-Q6_K.gguf` | Quality tests (backup) |
| MoE 35B-A3B Q4_K_M | `/home/erol/ai/turboquant/models/Qwen3.5-35B-A3B-Q4_K_M.gguf` | MoE validation |

---

## SUCCESS CRITERIA

Session 26 is DONE when:
- [ ] Quality test results analyzed (passkey + NIAH accuracy for all types)
- [ ] Block-128 validated on CUDA (PPL identical, speed >= baseline)
- [ ] README bpv numbers corrected
- [ ] AGENTS.md updated with S25+S26 results
- [ ] Vault updated (session note, benchmarks, roadmap)
- [ ] All changes pushed to `release/cuda-optimized`
- [ ] Zero PPL regressions, zero speed regressions

---

## ESTIMATED EFFORT

| Task | Time |
|------|------|
| Quality results analysis | 30 min |
| Block-128 implementation + benchmarks | 2-3 hours |
| README + AGENTS.md updates | 1 hour |
| Vault updates | 30 min |
| Release branch merge + push | 15 min |
| **Total** | **~5 hours** |
