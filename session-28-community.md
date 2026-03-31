# Session 28 — COMMUNITY: KL Divergence, Cross-Format Sparse V, Upstream PR Prep

## READ FIRST (MANDATORY)

1. **Read `AGENTS.md`** — contains ALL rules, architecture, dead ends, file locations, benchmarking protocol. Follow it EXACTLY.
2. **Read this entire prompt** before starting any work.
3. Create branch **`session/28-community`** from `session/27-quality` (or `release/cuda-optimized` if S27 is merged).
4. The Obsidian vault is at `/mnt/c/vaults/forge/` — search it for any context you need.
5. GPU is an RTX 5090 32GB (SM120). CUDA 12.8. WSL2 Ubuntu 24.04.

---

## CONTEXT — WHERE WE ARE AFTER S27

### What S27 Should Have Done
- Norm correction implemented (PPL improvement expected)
- 50-chunk wikitext-103 validation at 32K for all types
- Skip rate measurement at multiple thresholds
- Quality gate script created
- NIAH test suite enhanced

Verify these are done. If S27 items are incomplete, finish them first.

### The Remaining Gap vs TheTom
After S26 (bug fix, block-128) and S27 (norm correction, validation infrastructure), the remaining gaps are:
- **KL divergence metrics** — he measures logit divergence vs f16, we only have PPL
- **Cross-format sparse V** — he validated sparse V on q8_0/q4_0, we only tested turbo types
- **Upstream PR preparation** — both he and we need clean PRs for llama.cpp mainline
- **Discussion #20969 post** — our data is better than anyone's, it's time to share it

Session 28 closes these gaps and prepares for public release.

---

## TASK 1: KL Divergence Measurement

### Why
PPL measures next-token prediction accuracy. KL divergence measures how much the full probability distribution shifts from the f16 reference. TheTom's data:
- q8_0: KLD 0.001549, 98.43% same top-p token
- turbo3: KLD 0.016145, 94.31% same top-p token
- turbo4: KLD 0.009633, 95.98% same top-p token

This is a finer-grained quality metric that catches distribution shifts PPL misses.

### Implementation
Create `quality-tests/kl_divergence.py`:

1. Start llama-server with `-ctk f16 -ctv f16` (baseline)
2. For each test prompt (from wikitext-2, 100 samples), request full logprobs
3. Restart server with `-ctk turbo3 -ctv turbo3` (or each type)
4. Request logprobs for same prompts
5. Compute KL divergence: `KLD = sum(p_f16 * log(p_f16 / p_turbo))`
6. Compute top-p agreement: what % of tokens have the same argmax?
7. Compute delta-p RMS: `sqrt(mean((p_f16_top - p_turbo_top)^2))`

### Using llama-server logprobs
The `/v1/completions` endpoint supports `logprobs: N` parameter which returns top-N log probabilities per token. Use `logprobs: 10` for KLD computation.

```bash
# Example API call with logprobs
curl -s http://localhost:8090/v1/completions -H "Content-Type: application/json" -d '{
  "prompt": "The capital of France is",
  "max_tokens": 1,
  "temperature": 0,
  "logprobs": 10
}'
```

### Output Format
```
=== KL Divergence vs f16 ===
| Type | Mean KLD | Delta-p RMS | Same top-p % |
|------|----------|-------------|-------------|
| q8_0 | 0.001X | X.XX% | 98.X% |
| turbo4 | 0.00XX | X.XX% | 9X.X% |
| turbo3 | 0.0XXX | X.XX% | 9X.X% |
| turbo2 | 0.0XXX | X.XX% | 9X.X% |
| turbo1.5 | 0.0XXX | X.XX% | 9X.X% |
```

### Testing Protocol
- 100 wikitext-2 samples, each 256 tokens
- Compare turbo types vs f16 baseline
- Also compare WITH vs WITHOUT sparse V for the same type (should show delta ≈ 0)
- Record both MoE model (Qwen3.5-35B-A3B) and dense model (27B Q6_K) if time permits

---

## TASK 2: Cross-Format Sparse V Validation

### Why
TheTom validated that sparse V skip works on q8_0 and q4_0 KV caches (not just turbo types), giving +5% decode on q8_0 with zero PPL impact. If this holds on CUDA, sparse V becomes a **general llama.cpp optimization** — not TurboQuant-specific. This is a much bigger story for upstream.

### Implementation
Our sparse V threshold is in `fattn-vec.cuh`:
```cuda
constexpr bool V_is_low_bpv = (type_V == GGML_TYPE_TURBO2_0 || type_V == GGML_TYPE_TURBO1_5);
constexpr float sparse_v_threshold_f = V_is_low_bpv ? 1e-2f : 5e-3f;
```

Currently only turbo types use the aggressive threshold. For q8_0/q4_0/f16, the threshold is effectively 0 (the `V_is_low_bpv` check and the `constexpr` means non-turbo types get no threshold).

**Change needed**: Apply a conservative threshold (1e-6, matching TheTom) to ALL V types:
```cuda
// Conservative sparse V for all types (TheTom validated on q8_0, q4_0)
constexpr float sparse_v_threshold_f =
    V_is_low_bpv ? 1e-2f :        // turbo2/turbo1.5: aggressive
    (type_V == GGML_TYPE_TURBO3_0 || type_V == GGML_TYPE_TURBO4_0) ? 5e-3f :  // turbo3/turbo4
    1e-6f;                          // q8_0, q4_0, f16: conservative (TheTom's threshold)
```

### Testing
1. Benchmark q8_0 KV at short and 32K BEFORE the change
2. Apply the change
3. Benchmark q8_0 at short and 32K AFTER — expect +3-5% at 32K
4. Run PPL for q8_0 at ctx=512 and ctx=2048 — should be bit-exact
5. If confirmed: this is a contribution to upstream llama.cpp independent of TurboQuant

---

## TASK 3: Discussion #20969 Post

### Background
llama.cpp Discussion #20969 is where the community discusses TurboQuant KV cache compression. We have the best CUDA numbers in the ecosystem. It's time to share them.

### Draft Location
A draft exists in the vault: `09 Community/DISCUSSION_DRAFT_20969.md` (if present) or needs to be written.

### Content
The post should include:
1. **Headline numbers**: turbo3 beats q8_0 at 32K (55.08 vs 53.06 tok/s), turbo2 at 256K (36.62 tok/s)
2. **Full performance table**: all types, short + 32K + 64K + 131K + 256K
3. **Quality data**: PPL at 512/2048, 50-chunk wikitext-103 at 32K (from S27), KLD (from this session)
4. **NIAH results**: passkey retrieval + needle-in-haystack accuracy matrices
5. **Cross-GPU validation**: 1,121+ iterations across SM86/89/120
6. **Key optimizations**: sparse V skip, LUT scoring, block-128 storage
7. **Configuration recommendations**: which mode for which use case
8. **Acknowledgments**: TheTom (Metal implementation, block-128 research, quality validation), signalnine (CUDA port base), spiritbuun (norm correction)

### Tone
Technical but accessible. Show data, not claims. Link to the repo. Invite testing.

### Post After Review
Draft the post, save as `/mnt/c/vaults/forge/09 Community/discussion_20969_final.md`. Do NOT post it — Erol will review and post manually.

---

## TASK 4: Upstream PR Preparation

### What
Prepare clean commits for a PR to TheTom's upstream repo (`TheTom/llama-cpp-turboquant`). This PR should contain our CUDA-specific optimizations that benefit the project:

### What To Include in PR
1. **Sparse V skip** (all types, with type-adaptive thresholds)
2. **Half-precision LUT** (turbo3/turbo2)
3. **Block-128 storage** (if validated in S26)
4. **Norm correction** (if implemented in S27)
5. **nthreads_KQ=8 for all turbo types** (S24B)
6. **constexpr centroids** (S24B)

### What NOT To Include
- Dead turbo4 LUT code removal (NVCC-specific codegen hack)
- Any NVCC-specific compiler flags
- Session files, test scripts, vault references

### PR Format
- Clean squashed commits (one per logical feature)
- Each commit includes speed + PPL data in the message
- Tests pass on SM120 at minimum
- README updates for CUDA numbers

### Save PR Draft
Create the PR description as `/mnt/c/vaults/forge/09 Community/upstream_pr_draft.md`. Do NOT create the PR — Erol will review and submit.

---

## TASK 5: README Parity with TheTom

### Gap Analysis
TheTom's README covers things ours doesn't:
- KL divergence data
- NIAH retrieval matrices (depth × context)
- Community hardware results section (RTX 3090, M1 Max, etc.)
- Prefill context scaling table
- Real-world server benchmark (70-page PDF)
- Configuration recommendations with specific model guidance
- Compression quality table (cosine similarity, MSE)

### Updates Needed
1. Add KLD section (from Task 1 data)
2. Add NIAH section with accuracy matrices
3. Add community hardware section (3090 Ti, 4090M data already exists)
4. Add prefill context scaling table (we have this data)
5. Add recommended configuration matrix by model type
6. Ensure every metric in his README has our equivalent (or better)

---

## TASK 6: Update Documentation and Vault

### AGENTS.md
- Add S28 results (KLD metrics, cross-format sparse V, upstream PR status)
- Add NIAH data to the quality section
- Update sparse V section with cross-format validation

### Vault
- `01 Sessions/Session 28.md` — create session note
- `03 Benchmarks/Benchmark Hub.md` — add KLD data, NIAH matrices
- `09 Community/discussion_20969_final.md` — discussion post draft
- `09 Community/upstream_pr_draft.md` — PR description draft
- `00 Dashboard/Project Status.md` — update
- `08 Plans/Roadmap.md` — mark S28 items done

---

## THETOM'S REFERENCE REPO

```
/home/erol/ai/turboquant/research/llama-cpp-turboquant/.trash/research/repos/TheTom-turboquant_plus/
```

Key files for this session:
| File | Relevance |
|------|-----------|
| `docs/kl-divergence-results.md` | KLD methodology and reference numbers |
| `docs/sparse-v-upstream-validation.md` | Cross-format sparse V on q8_0/q4_0 |
| `docs/upstream-pr-plan.md` | His PR preparation checklist |
| `docs/turboquant-recommendations.md` | Configuration matrix by model |
| `docs/quality-benchmarks.md` | Top-of-tree quality data |
| `docs/cross-model-validation.md` | Multi-architecture testing |

---

## MODELS

| Model | Path | Use For |
|-------|------|---------|
| Qwen 3.5 27B Q6_K | `models/opus-v2-Q6_K.gguf` | KLD, PPL, speed |
| MoE 35B-A3B Q4_K_M | `models/Qwen3.5-35B-A3B-Q4_K_M.gguf` | MoE KLD comparison |
| Qwen 3.5 9B Q8_0 | `models/Qwen3.5-9B-Q8_0.gguf` | NIAH tests |

Model paths relative to `/home/erol/ai/turboquant/`.

---

## SUCCESS CRITERIA

- [ ] KL divergence measured for all types vs f16 (both dense and MoE if time)
- [ ] Cross-format sparse V validated on q8_0 (speed + PPL)
- [ ] Discussion #20969 post drafted and saved for review
- [ ] Upstream PR description drafted and saved for review
- [ ] README matches or exceeds TheTom's coverage on every metric
- [ ] All docs and vault updated

---

## ESTIMATED EFFORT

| Task | Time |
|------|------|
| KL divergence implementation + measurement | 3-4 hours |
| Cross-format sparse V validation | 2 hours |
| Discussion post draft | 2 hours |
| Upstream PR preparation | 2-3 hours |
| README parity updates | 2 hours |
| Documentation | 1 hour |
| **Total** | **~12-14 hours** |
