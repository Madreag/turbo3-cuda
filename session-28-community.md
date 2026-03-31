# Session 28 — COMMUNITY: KL Divergence, Cross-Format Sparse V, Upstream PR Prep

## READ FIRST (MANDATORY)

1. **Read `AGENTS.md`** — follow ALL rules exactly.
2. **Read this entire prompt** before starting any work.
3. Create branch **`session/28-community`** from `release/cuda-optimized`.
4. Vault: `/mnt/c/vaults/forge/`. GPU: RTX 5090 32GB (SM120). CUDA 12.8. WSL2.

---

## CONTEXT — WHERE WE ARE AFTER S26+S27

### Sessions 26-27 Delivered
- **S26**: SM120 D=256 bug fixed (LUT disabled at D=256), turbo4/turbo1.5 Q_reg fix, block-128 storage (turbo3 5.12x, turbo2 7.53x), all 4 types beat q8_0 at both short and 32K
- **S27**: 50-chunk wikitext-103 PPL (all types × 4 contexts), skip rate data (3 thresholds), NIAH on 5090, quality gate script, threshold validated (1e-2 = bit-identical PPL to 1e-6 for turbo2/turbo1.5, +13% free speed)
- **S27 key findings**: 2-bit PPL delta is inherent to quantization (not threshold), VEC kernel at absolute ceiling (37 failed optimization attempts across S25+S27), 1e-2 threshold correct

### Current Performance (RTX 5090, 27B Q6_K, block-128)

| Type | bpv | Compression | Short | 32K | PPL 512 |
|------|:---:|:-----------:|:-----:|:---:|:-------:|
| q8_0 | 8.5 | 1.9x | 64.06 | 54.01 | 6.759 |
| turbo4 | 4.25 | 3.8x | 65.33 | 58.03 | 6.825 |
| turbo3 | 3.125 | 5.12x | 65.14 | 56.28 | 6.852 |
| turbo2 | 2.125 | 7.53x | 64.13 | 58.48 | 7.080 |
| turbo1.5 | 2.00 | 8.0x | 64.00 | 55.62 | 7.312 |

### S27 Quality Data

**50-chunk wikitext-103 PPL (vs q8_0)**:
| Type | ctx=512 | ctx=2048 | ctx=8192 | ctx=32K |
|------|:-------:|:--------:|:--------:|:------:|
| turbo4 | +0.77% | +0.53% | +1.22% | +2.35% |
| turbo3 | +0.77% | +1.39% | +1.81% | +2.84% |
| turbo2 | +3.37% | +4.77% | +8.06% | +11.93% |
| turbo1.5 | +6.34% | +9.72% | +13.42% | +23.88% |

**NIAH (5090)**: q8_0=85%, turbo3=70%, turbo2=75%, turbo1.5=35%
**NIAH (3090 Ti)**: q8_0=84.8%, turbo3=86.4%, turbo2=78.8%

### Cross-GPU S26 Validation
- **3090 Ti (SM86)**: Block-128 = +4-6% at 32K. Zero regressions. All 5 types generate. turbo2 OC = 80.18 tok/s.
- **4090M (SM89)**: Block-128 = +10-17% at 32K. Fastest session ever on SM89.
- **5090 (SM120)**: All types beat q8_0 at both short and 32K.

### What Competitors Are Doing
- **signalnine's CUDA port** just landed publicly — 46.7 tok/s on RTX 4090 (98% of f16). Our optimized fork is significantly faster.
- **TheTom** has comprehensive Metal implementation with KLD, NIAH, configuration recommendations, 70B stress test
- Community interest is high (193 upvotes on r/LocalLLaMA)

### What This Session Does
Close the remaining gaps vs TheTom and prepare for public community engagement:
1. KL divergence metrics (finer-grained quality measure)
2. Cross-format sparse V (q8_0/q4_0 — upstream contribution story)
3. Discussion #20969 post (share our data)
4. Upstream PR preparation
5. README parity with TheTom

---

## TASK 1: KL Divergence Measurement

### Why
PPL is a single number. KLD shows how the full token probability distribution shifts from the f16 reference. TheTom's data:
- q8_0: KLD 0.001549, 98.43% same top-p
- turbo3: KLD 0.016145, 94.31% same top-p
- turbo4: KLD 0.009633, 95.98% same top-p

### Implementation
Create `quality-tests/kl_divergence.py`:

1. Start llama-server with f16 KV (baseline)
2. For 100 wikitext-2 prompts (256 tokens each), request logprobs via `/v1/completions` with `logprobs: 10`
3. Restart server with each turbo type
4. Request same logprobs
5. Compute: KLD, top-p agreement %, delta-p RMS

**Important**: Use `max_tokens: 1` per prompt (we only need the next-token distribution, not generation). This avoids the Qwen 3.5 thinking model issue entirely.

```bash
curl -s http://localhost:8090/v1/completions -H "Content-Type: application/json" -d '{
  "prompt": "The capital of France is",
  "max_tokens": 1,
  "temperature": 0,
  "logprobs": 10
}'
```

### Output
```
| Type | Mean KLD | Delta-p RMS | Same top-p % |
|------|----------|-------------|-------------|
| q8_0 | | | |
| turbo4 | | | |
| turbo3 | | | |
| turbo2 | | | |
| turbo1.5 | | | |
```

---

## TASK 2: Cross-Format Sparse V Validation

### Why
TheTom validated sparse V on q8_0 and q4_0 — it's format-agnostic, not TurboQuant-specific. If we confirm on CUDA, sparse V becomes an **upstream llama.cpp contribution** independent of TurboQuant. Much bigger story.

### Current Code
```cuda
constexpr bool V_is_low_bpv = (type_V == GGML_TYPE_TURBO2_0 || type_V == GGML_TYPE_TURBO1_5);
constexpr float sparse_v_threshold_f = V_is_low_bpv ? 1e-2f : 5e-3f;
```

Non-turbo types effectively get threshold=0 (the constexpr evaluates to 5e-3 but the sparse V check is inside the turbo-specific code path).

### Change
Apply a conservative threshold (1e-6) to ALL V types by moving the sparse V check outside the turbo-specific block:

**IMPORTANT**: This is a constexpr change — do NOT make it runtime (S27 proved non-constexpr causes register spill at 168 regs). Keep it as:
```cuda
constexpr float sparse_v_threshold_f = V_is_low_bpv ? 1e-2f : 5e-3f;
// Apply to ALL types, not just turbo
```

The sparse V check itself may need to be moved outside the `if constexpr (type_K is turbo)` block to apply to q8_0/f16 V types too.

### Testing
1. q8_0 KV: speed at short + 32K BEFORE
2. Apply change
3. q8_0 KV: speed at short + 32K AFTER — expect +3-5% at 32K
4. PPL for q8_0 — must be bit-exact
5. If confirmed: huge upstream story

---

## TASK 3: Discussion #20969 Post

### Content (updated with S26+S27 data)
1. **Headline**: ALL 4 turbo types beat q8_0 at both short and 32K on SM120. turbo2 at 256K = 36.62 tok/s.
2. **Performance table**: S26 final numbers with block-128 bpv
3. **Quality**: 50-chunk wikitext-103 PPL table (S27), KLD data (this session)
4. **NIAH**: turbo3 86.4% beats q8_0 84.8% on 3090 Ti (sparse V denoising)
5. **Cross-GPU**: 3 GPUs, 1,351+ iterations, zero failures. 3090 Ti OC data.
6. **Block-128**: 5.12x turbo3, 7.53x turbo2. HyperionMS2040 SET_ROWS fix.
7. **SM120 D=256 bug**: NVIDIA NVBUG 5218000/5288270 documented. Workaround in place.
8. **Configuration recommendations**: reference TheTom's guide, add our CUDA-specific data
9. **Honest limitations**: 2-bit types degrade at long context (inherent, not threshold). D=256 LUT disabled on SM120.

### Attribution (per TheTom's corrections)
- **TheTom**: Metal implementation, turbo4 resurrection (7 bugs), asymmetric K/V discovery, turbo3 norm correction, block-128 storage research, sparse V concept, quality validation
- **signalnine**: Original CUDA port (PR #3 to TheTom's repo), InnerQ equalization
- **spiritbuun**: turbo4 norm correction (separate fork), inverse FWHT prefill
- **HyperionMS2040**: Block-128 SET_ROWS fix (commit `7cb6edb`)

### Tone
Technical, data-driven, honest about limitations. Show numbers, not claims. Invite testing. Link to repo.

### Save To
`/mnt/c/vaults/forge/09 Community/discussion_20969_final.md` — Erol reviews and posts manually.

---

## TASK 4: Upstream PR Preparation

### PR to TheTom/llama-cpp-turboquant

**Include**:
1. Sparse V skip (type-adaptive thresholds: 5e-3 turbo3/4, 1e-2 turbo2/1.5)
2. Half-precision LUT (turbo3/turbo2, D≤128 only)
3. Block-128 storage (QK_TURBO3=128, QK_TURBO2=128) + HyperionMS2040 SET_ROWS fix
4. nthreads_KQ=8 for all turbo types
5. constexpr centroids
6. turbo4/turbo1.5 vec_dot Q_reg fix (use q8_1 Q correctly)
7. D=256 LUT disable on SM120 (NVIDIA codegen workaround)
8. L2 prefetch hints
9. `__expf` fast-math softmax

**Do NOT include**:
- Dead turbo4 LUT code removal (NVCC-specific codegen hack)
- Session/test files, vault references
- AGENTS.md, CLAUDE.md

**Format**: One squashed commit per logical feature. Each includes speed + PPL data. PR description references Discussion #20969.

### Save To
`/mnt/c/vaults/forge/09 Community/upstream_pr_draft.md` — Erol reviews and submits.

---

## TASK 5: README Parity with TheTom

### What TheTom Has That We Don't
- KL divergence data → add from Task 1
- NIAH depth×context matrices → add from S27 5090 data + 3090 Ti dump data
- Community hardware section → add 3090 Ti (stock + OC), 4090M data from dump folder
- Prefill context scaling → we have pp512 data, need pp4096/pp8192/pp32K
- Configuration recommendations by model type → reference TheTom's guide + our data
- Real-world server benchmark → would need to run a long-document test

### What We Have That He Doesn't
- ALL types beat q8_0 at both short AND 32K (he's at ~90% of q8_0)
- 256K context data (36.62 tok/s turbo2)
- 3-GPU cross-validation (1,351+ iterations)
- Full extreme context table (32K through 256K, all types, Q4_K_M)
- 50-chunk wikitext-103 PPL across 4 context lengths
- SM120 NVIDIA bug investigation and workaround documentation
- Contributions section with full attribution

### Priority
1. KLD section (new data from this session)
2. NIAH section (existing data — just format it)
3. Community hardware section (dump folder data)
4. Configuration recommendations

---

## TASK 6: Update Documentation and Vault

### AGENTS.md
- Add S28 results (KLD, cross-format sparse V, post/PR status)
- Mark project as "release-ready"

### Vault
- `01 Sessions/Session 28.md`
- `03 Benchmarks/Benchmark Hub.md` — KLD data
- `09 Community/discussion_20969_final.md`
- `09 Community/upstream_pr_draft.md`
- `00 Dashboard/Project Status.md`
- `08 Plans/Roadmap.md` — S28 DONE

---

## THETOM'S REFERENCE REPO
```
/home/erol/ai/turboquant/research/llama-cpp-turboquant/.trash/research/repos/TheTom-turboquant_plus/
```
| File | Relevance |
|------|-----------|
| `docs/kl-divergence-results.md` | KLD methodology and reference numbers |
| `docs/sparse-v-upstream-validation.md` | Cross-format sparse V on q8_0/q4_0 |
| `docs/upstream-pr-plan.md` | His PR checklist |
| `docs/turboquant-recommendations.md` | Config recommendations by model |

---

## MODELS

| Model | Path | Use For |
|-------|------|---------|
| Qwen 3.5 27B Q6_K | `models/opus-v2-Q6_K.gguf` | KLD, PPL, speed |
| MoE 35B-A3B Q4_K_M | `models/Qwen3.5-35B-A3B-Q4_K_M.gguf` | MoE KLD |
| Qwen 3.5 9B Q8_0 | `models/Qwen3.5-9B-Q8_0.gguf` | Generation tests |

---

## WHAT NOT TO DO
- Do NOT change VEC kernel code structure (37 failed attempts, ceiling confirmed)
- Do NOT make thresholds non-constexpr (causes register spill, S27 Dead End #1)
- Do NOT run benchmarks in background (Rule 11)
- Do NOT post to Discussion #20969 directly — save drafts for Erol to review
- Do NOT create PRs directly — save descriptions for Erol to submit
- Do NOT use max_tokens < 200 with Qwen 3.5 for generation (thinking model)
- For KLD measurement, use max_tokens=1 (next-token logprobs only — avoids thinking issue)

---

## SUCCESS CRITERIA

- [ ] KL divergence measured for all types vs f16
- [ ] Cross-format sparse V validated on q8_0 (speed + PPL)
- [ ] Discussion #20969 post drafted and saved
- [ ] Upstream PR description drafted and saved
- [ ] README matches or exceeds TheTom's coverage
- [ ] All docs and vault updated
- [ ] Pushed to release/cuda-optimized

---

## ESTIMATED EFFORT

| Task | Time |
|------|------|
| KL divergence implementation + measurement | 3-4 hours |
| Cross-format sparse V | 2 hours |
| Discussion post draft | 2 hours |
| Upstream PR description | 2-3 hours |
| README parity | 2 hours |
| Documentation | 1 hour |
| **Total** | **~12-14 hours** |
