# Session 26 — BLAST: Block-128, Quality Validation, Bug Fix, Release Prep

## READ FIRST (MANDATORY)

1. **Read `AGENTS.md`** — contains ALL rules, architecture, dead ends, file locations, benchmarking protocol. Follow it EXACTLY.
2. **Read this entire prompt** before starting any work.
3. You are on branch **`session/26-blast`**. The release branch is `release/cuda-optimized`.
4. The Obsidian vault is at `/mnt/c/vaults/forge/` — search it for any context you need.
5. GPU is an RTX 5090 32GB (SM120). CUDA 12.8. WSL2 Ubuntu 24.04.

---

## SITUATIONAL AWARENESS — WHERE WE ARE

### The Project
CUDA implementation of TurboQuant (ICLR 2026) KV cache compression for llama.cpp. 4 turbo types: turbo4 (4.25 bpv), turbo3 (3.50 bpv), turbo2 (2.50 bpv), turbo1.5 (2.00 bpv). All beat q8_0 at 32K context. turbo3 matches q8_0 perplexity at ctx=2048. 1,121+ stability iterations across 3 GPUs, zero failures.

### Sessions 17-24B (Complete)
All 4 types fully optimized: parallel SET_ROWS, native VEC FA with LUT scoring (turbo3/turbo2), `__expf` softmax, constexpr centroids, nthreads_KQ=8 all types, 36 K×V asymmetric combos, 5 models validated (D=64/96/128/256), 3 GPUs (SM86/89/120).

### Session 25 (Complete — `session/25-deep`)
60 micro-optimization iterations. 5 committed wins:
1. **Sparse V threshold** 1e-6 → 5e-3 (turbo3/4) and 1e-2 (turbo2/1.5) → turbo3 32K +5-11%, turbo2 +13%, 8B Llama +28%
2. **Half-precision LUT** (float → half in shmem) → +2.45% at 32K
3. **Dead turbo4 LUT code removal** → +4.4% (NVCC codegen sensitivity)
4. PPL bit-exact at all thresholds through ctx=16K
5. VEC kernel at optimization ceiling: 168 registers, 98.4% utilization

**Key S25 findings:**
- K type determines 32K speed, V type barely matters (sparse V skip makes V nearly free)
- turbo2 at 256K: 36.62 tok/s. turbo3 beats q8_0 at 32K (55.08 vs 53.06)
- turbo2 and turbo3 beat f16 at 65K+ (bandwidth savings from compressed KV)
- 33 code restructuring attempts ALL failed due to register pressure — do NOT repeat

### Current S25 Numbers (RTX 5090, Qwen 3.5 27B Q6_K)

| Type | Short | 32K | PPL 512 | PPL 2048 |
|------|:-----:|:---:|:-------:|:--------:|
| f16 | 62.30 | 56.55 | — | — |
| q8_0 | 61.21 | 53.06 | 6.759 | 5.674 |
| turbo4 | 60.92 | 50.63 | 6.825 | 5.694 |
| turbo3 | 61.46 | 55.08 | 6.852 | 5.674 |
| turbo2 | 62.07 | 56.02 | 7.080 | 5.892 |
| turbo1.5 | 59.22 | 47.89 | 7.312 | 6.103 |

---

## BUG REPORT: SM120 + Qwen 3.5 Generation Failure (CRITICAL)

### Symptoms
On RTX 5090 (SM120), Qwen 3.5 models (both 9B and 27B) generate **empty output** when using turbo KV types with Flash Attention enabled. The q8_0 FA path works correctly on the same model.

| Config | Result |
|--------|--------|
| `q8_0 -fa on` | Correct answers (upstream q8_0 VEC kernel) |
| `turbo3 -fa on` | Empty `content`, `????` reasoning (our turbo VEC kernel) |
| Any type `-fa off` | Correct answers (mul_mat attention, no VEC kernel) |
| `turbo3 -fa on` on SM86 (3090 Ti) | **Works correctly** |
| `turbo3 -fa on` on SM120 with Llama-3.3-8B (D=128) | **Works correctly** |

### Key Observations
- **PPL is bit-exact** — the MMA prefill path is correct
- **llama-bench reports normal tok/s** — the kernel runs, it just produces wrong attention values
- **SM86 works, SM120 doesn't** — SM120-specific codegen issue
- **D=128 works (Llama-3.3-8B), D=256 doesn't (Qwen 3.5)** — likely D=256-specific

### Partial Quality Test Data (in `quality-tests/passkey_results_*.json`)
- `q8_0`: 43.8% overall — generates real answers (model limitation at long context)
- `turbo3`: 0% — ALL 105 responses are empty strings

### Investigation Hints
The bug is in the VEC FA decode path for D=256 on SM120. Possible causes:
1. **LUT construction at D=256**: The LUT is `half turbo_lut[D][lut_stride]` = `half turbo_lut[256][9]` = 4,608 bytes. Check if shmem allocation is correct.
2. **LUT indexing at D=256**: `d_base` ranges 0-255. Check that `d_base / QK_TURBO3` and `d_base % QK_TURBO3` compute correct block/offset for D=256 (there are 8 blocks of 32 = 256 elements).
3. **Half-precision LUT accumulation**: 32 LUT lookups per iteration (256/8=32) × norm multiplication. The fp16 accumulation might overflow or underflow at D=256.
4. **Sparse V threshold**: At D=256, attention scores have different magnitude distribution. The 5e-3 threshold might be too aggressive for D=256.
5. **nthreads_KQ computation**: For D=256, `K_is_turbo ? 128/cpy_nb : nthreads_KQ_q`. Verify `nthreads_KQ` value is correct for D=256 turbo types.
6. **Q quantization at D=256**: Q is quantized to q8_1 in shared memory. Check that D=256 Q fits in shmem and is indexed correctly.

### How to Bisect
If the cause isn't obvious from code reading:
1. Disable sparse V threshold (set to 1e-6) → test
2. Revert half LUT (use float) → test
3. Disable LUT entirely (force vec_dot path) → test
4. Use nthreads_KQ=32 → test

Each test: start llama-server with `-ctk turbo3 -ctv turbo3 -fa on`, send a simple chat completion, check if content is non-empty.

**Test command** (after starting server on port 8090):
```bash
curl -s http://localhost:8090/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":200,"temperature":0}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('Content:', repr(d['choices'][0]['message']['content'][:100]))"
```
- **PASS**: Content contains "4" or any real text
- **FAIL**: Content is empty string `''`

**IMPORTANT**: Qwen 3.5 is a thinking model. It needs `max_tokens: 200+` to finish `<think>` reasoning before producing the answer. With `max_tokens: 50` it runs out of tokens during thinking and returns empty content. This is NORMAL for thinking models — not a bug.

---

## TASK 1: Fix SM120 D=256 Generation Bug (PRIORITY)

**Before any other work**, fix this bug. The repo is useless for Qwen 3.5 users if turbo types produce garbage on SM120.

1. Start llama-server with 27B Qwen model + turbo3 + FA on
2. Verify the bug reproduces (empty content)
3. Bisect using the steps above
4. Fix the root cause
5. Verify fix: turbo3 + FA on → correct "4" answer
6. Run full regression suite (short + 32K + PPL) to confirm no speed/quality regression

**Model for testing generation**: Use `Qwen3.5-9B-Q8_0.gguf` (smaller, faster to load). The bug reproduces on both 9B and 27B.
**Model for benchmarking speed/PPL**: Use `opus-v2-Q6_K.gguf` (27B, matches all existing data).

---

## TASK 2: Block-128 Storage (TheTom's Research — CUDA Validation)

### Background
TheTom proved that `QK_TURBO3=32` stores 3 redundant norm values per 128-element rotation group. Changing to `QK_TURBO3=128` eliminates this redundancy:

| Block Size | Layout per 128 elements | bpv | Compression |
|:----------:|------------------------|:---:|:-----------:|
| 32 (current) | 4 × (norm+qs+signs) = 4 × 14B = 56B | 3.50 | 4.57x |
| 128 (proposed) | 1 × (norm+qs+signs) = 50B | 3.125 | 5.12x |

On Metal (Apple Silicon): PPL identical to 4 decimal places across 3 models, 2 hardware platforms, 7 cache configs. Free 12% compression improvement.

### What To Do

1. Edit `ggml/src/ggml-common.h`:
   ```c
   #define QK_TURBO3 128  // was 32
   #define QK_TURBO2 128  // was 32
   ```

2. **Check all code that uses QK_TURBO3/QK_TURBO2** — the change propagates through:
   - `block_turbo3_0` struct (qs array size, signs array size)
   - `block_turbo2_0` struct (qs array size)
   - `fattn-vec.cuh` LUT scoring inner loop: `D/QK_TURBO3` iterations → 128/128=1 (was 4)
   - `fattn-common.cuh` vec_dot functions: `D/QK_TURBO3` blocks → 1 block
   - `set-rows.cu` encoding kernels
   - All dequant functions
   - Template instantiations

3. Rebuild and run **full regression suite**:
   - Short + 32K for ALL turbo types
   - PPL ctx=512 and ctx=2048 for turbo3
   - PPL ctx=512 for turbo2, turbo1.5, turbo4

4. **Critical**: The VEC kernel LUT loop changes from 4 iterations to 1. This WILL change NVCC codegen (S25 proved NVCC is extremely sensitive). Benchmark carefully — expect possible speed changes.

5. If PPL matches (to 4 decimal places) and speed is >= baseline: **commit**
6. If speed regresses: try reverting only QK_TURBO3 or only QK_TURBO2

### Also Fix bpv Numbers
Our README says turbo3 = 3.25 bpv. That's WRONG. Actual:
- Block=32: `sizeof(block_turbo3_0) * 8 / 32` = `14 * 8 / 32` = **3.50 bpv**
- Block=128: `50 * 8 / 128` = **3.125 bpv**

Verify each type's actual bpv from the struct in `ggml-common.h` and update README + AGENTS.md.

---

## TASK 3: Quality Test Results Analysis

Check if quality test results exist from the 3090 Ti:

```bash
ls /mnt/c/vaults/dump/*niah* /mnt/c/vaults/dump/*passkey* /mnt/c/vaults/dump/*quality* 2>/dev/null
```

If results exist, analyze:
- Compare q8_0 accuracy vs turbo3/turbo2/turbo1.5 at each context × depth
- Any turbo-unique failures = quality regression from sparse V threshold
- Report results matrix

**Note**: The 5090 quality tests are INVALID (turbo types produced empty output due to the D=256 bug). Only 3090 Ti results are trustworthy.

---

## TASK 4: Update Documentation

### AGENTS.md
- Add S25 results (sparse V threshold, half LUT, dead code removal, K-type-dominates finding)
- Add S26 results (bug fix, block-128)
- Correct bpv numbers
- Add any new dead ends

### README.md
- Correct bpv and compression ratio numbers
- Update performance table if block-128 changes speed
- Add quality test results if available

### Vault
- `01 Sessions/Session 26.md` — create session note
- `03 Benchmarks/Benchmark Hub.md` — add S26 data
- `00 Dashboard/Project Status.md` — update status
- `08 Plans/Roadmap.md` — mark S26 items done

---

## TASK 5: Merge to Release Branch

After ALL tasks validated:
```bash
git checkout release/cuda-optimized
git merge session/26-blast
git push myfork release/cuda-optimized
```

---

## THETOM'S REFERENCE IMPLEMENTATION (Metal, block-128)

TheTom's research repo with block-128 implementation, NIAH tests, threshold ablation, and quality gate scripts:
```
/home/erol/ai/turboquant/research/llama-cpp-turboquant/.trash/research/repos/TheTom-turboquant_plus/
```

**Key files to reference when implementing block-128 on CUDA:**

| File | What It Contains |
|------|-----------------|
| `docs/papers/` | Block-size optimization paper (full research with PPL tables) |
| `docs/threshold-ablation.md` | Sparse V threshold sweep data |
| `docs/long-context-sparse-v-validation.md` | Long-context quality validation |
| `docs/sparse-v-upstream-validation.md` | Upstream sparse V compatibility |
| `scripts/niah_test.py` | TheTom's NIAH test script (works with Qwen 3.5) |
| `scripts/turbo-quality-gate.sh` | Quality gate script (PPL + speed checks) |
| `scripts/turbo-quick-bench.sh` | Quick benchmark runner |
| `scripts/measure_skip_rate.py` | Measures sparse V skip percentage |
| `proof/niah/` | NIAH proof-of-concept results |
| `docs/turboquant-recommendations.md` | TheTom's config recommendations |

**IMPORTANT**: TheTom's code is for **Metal (Apple Silicon)**. The block structs, dequant functions, and FA templates are Metal-specific. You cannot copy-paste his kernel code. Use his implementation as a REFERENCE for the logic, then apply the equivalent changes to our CUDA kernels. The key insight is that `QK_TURBO3=128` is a one-line define change — all downstream code uses the define symbolically.

## KEY FILES (Read These First)

```
ggml/src/ggml-cuda/fattn-vec.cuh      — THE VEC FA KERNEL (bug is here)
                                         Line 94: nthreads_KQ computation
                                         Lines 143-146: sparse V threshold
                                         Lines 270-294: LUT construction
                                         Lines 344-398: LUT scoring inner loop (turbo3 + turbo2)
                                         Lines 452-500: V accumulation with sparse skip

ggml/src/ggml-common.h                 — Block structs and QK_TURBO defines
                                         Lines 275-282: QK_TURBO3=32, block_turbo3_0 (14B)
                                         Lines 322-328: QK_TURBO2=32, block_turbo2_0 (10B)
                                         Lines 291-316: QK_TURBO4=128, block_turbo4_0 (68B)
                                         Lines 333-338: QK_TURBO1_5=32, block_turbo1_5 (16B)

ggml/src/ggml-cuda/fattn-common.cuh    — vec_dot functions for all turbo types
                                         Lines 298-344: turbo3 vec_dot
                                         Lines 348-391: turbo2 vec_dot
                                         Lines 397-443: turbo4 vec_dot
                                         Lines 506-520: turbo1.5 vec_dot

ggml/src/ggml-cuda/turbo-quant.cuh     — Centroid arrays (constexpr), WHT signs
ggml/src/ggml-cuda/set-rows.cu         — SET_ROWS encoding for all 4 types
ggml/src/ggml-cuda/fattn.cu            — FA dispatch, template routing
```

---

## MODELS

| Model | Path | Use For |
|-------|------|---------|
| Qwen 3.5 27B Q6_K | `models/opus-v2-Q6_K.gguf` | Speed benchmarks, PPL (primary) |
| Qwen 3.5 9B Q8_0 | `models/Qwen3.5-9B-Q8_0.gguf` | Generation bug testing (faster to load) |
| Llama-3.3-8B Q6_K | `models/allura-forge_Llama-3.3-8B-Instruct-Q6_K.gguf` | D=128 control test |
| Qwen 3.5 27B Q4_K_M | `models/Qwen3.5-27B-Q4_K_M.gguf` | Long-context speed |

Model paths are relative to `/home/erol/ai/turboquant/`.

---

## WHAT NOT TO DO

- Do NOT repeat S25 dead ends (33 code restructuring attempts, all compiler flag combos — see AGENTS.md)
- Do NOT change VEC kernel code structure beyond the block-128 change (168 regs = ceiling)
- Do NOT use `__managed__` memory (kills SM86 via page faults in CUDA graph replay)
- Do NOT run benchmarks in background (Rule 11 — causes 176GB memory stacking → swap death)
- Do NOT use `max_tokens < 200` with Qwen 3.5 — it's a thinking model that needs token budget
- Do NOT skip PPL checks after ANY code change

---

## SUCCESS CRITERIA

Session 26 is DONE when:
- [ ] SM120 D=256 generation bug identified and fixed
- [ ] Fix verified: Qwen 3.5 + turbo3 + FA on → correct answers on SM120
- [ ] Block-128 validated on CUDA (PPL identical to 4 decimals, speed >= baseline)
- [ ] bpv numbers corrected in README and AGENTS.md
- [ ] Quality test results analyzed (3090 Ti data)
- [ ] AGENTS.md updated with S25+S26 results
- [ ] Vault updated (session note, benchmarks, roadmap, dashboard)
- [ ] All changes pushed to `release/cuda-optimized`
- [ ] Zero PPL regressions, zero speed regressions, generation works on D=128 AND D=256

---

## ESTIMATED EFFORT

| Task | Time |
|------|------|
| Bug investigation + fix | 2-4 hours |
| Block-128 implementation + benchmarks | 2-3 hours |
| Documentation updates | 1 hour |
| Vault + release merge | 30 min |
| **Total** | **~6-8 hours** |
