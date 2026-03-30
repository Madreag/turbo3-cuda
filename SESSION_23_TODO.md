# Session 23 — Maximum Performance TODO

## P0: Safety + High Impact

- [ ] **Part 4A**: Q4_K_M + turbo3 PPL validation (ctx=512)
- [ ] **Part 4B**: Q4_K_M + q8_0 baseline PPL (ctx=512)
- [ ] **Part 4C**: Q4_K_M + turbo2 extended context (65K)
- [ ] **Part 4D**: Q4_K_M + turbo1.5 ultra-long context (128K+)
- [ ] **Part 1A**: Read VEC kernel main loop (fattn-vec.cuh ~310-440)
- [ ] **Part 1B**: Research FlashAttention-3 / CUTLASS pipelining
- [ ] **Part 1C**: Plan double-buffer implementation (shmem budget check)
- [ ] **Part 1D**: Implement double-buffer for turbo3
- [ ] **Part 1E**: Benchmark short + 32K + PPL
- [ ] **Part 1F**: If regressed — try extra prefetch hints fallback

## P1: Features + Polish

- [ ] **Part 6A**: Implement LA=12 (first2+last2 q8_0 V) in llama-kv-cache.cpp
- [ ] **Part 6B**: Implement LA=13 (first4+last4 q8_0 V)
- [ ] **Part 6C**: Check model layer count (27B Qwen)
- [ ] **Part 6D**: Benchmark LA=13 PPL vs uniform turbo3/turbo2
- [ ] **Part 2A**: Instrument VEC kernel with skip counters
- [ ] **Part 2B**: Measure skip frequency at 32K
- [ ] **Part 2C**: Try threshold=1e-4 for turbo1.5
- [ ] **Part 2D**: Try threshold=1e-3 if 1e-4 safe
- [ ] **Part 2E**: Remove instrumentation, commit best threshold
- [ ] **Part 3A**: Find D=96 rejection point in fattn.cu / supports_op
- [ ] **Part 3B**: Add one-time warning for unsupported D
- [ ] **Part 3C**: Test warning with Phi-3.5-mini
- [ ] **Part 5A**: Add Q4_K_M section to README
- [ ] **Part 5B**: Add Recommended Configurations section
- [ ] **Part 5C**: Update cross-GPU section with S22 data (751+ iterations)
- [ ] **Part 5D**: Add 3090 Ti multi-model table
- [ ] **Part 5E**: Update DISCUSSION_DRAFT_20969.md with Q4_K_M data
- [ ] **Part 5F**: Verify README numbers match AGENTS.md

## P2: If Time

- [ ] **Part 7A**: KV norm measurement in set-rows.cu
- [ ] **Part 7B**: Auto-asymmetric recommendation log
- [ ] **Part 7C**: Document K=turbo4/V=q8_0 finding in README

## P3: Housekeeping

- [ ] **Part 8A**: Update AGENTS.md with S23 results
- [ ] **Part 8B**: Update vault (Dashboard, Benchmark Hub, Session 23, Roadmap)
- [ ] **Part 8C**: Push to myfork
