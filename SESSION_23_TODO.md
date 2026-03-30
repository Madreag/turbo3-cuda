# Session 23 — Maximum Performance TODO

## P0: Safety + High Impact

- [x] **Part 4A**: Q4_K_M + turbo3 PPL validation → 7.1274 (+1.39% vs q8_0=7.0297) SAFE
- [x] **Part 4B**: Q4_K_M + q8_0 baseline PPL → 7.0297
- [x] **Part 4C**: Q4_K_M + turbo2 extended context → 52.79 tok/s at 65K
- [x] **Part 4D**: Q4_K_M + turbo1.5 ultra-long → 25.81 tok/s at 131K (27B on 32GB!)
- [x] **Part 1A**: Read VEC kernel main loop — understood structure
- [x] **Part 1B-F**: Inner-loop V prefetch tried → no benefit (HW prefetcher sufficient). Double-buffer not worth pursuing. Added to Dead Ends.

## P1: Features + Polish

- [x] **Part 6A-B**: LA=8 already covers narrow (first2+last2). Added LA=12 for wide (first4+last4)
- [x] **Part 6C**: 27B Qwen = 64 layers (ideal for boundary V)
- [x] **Part 6D**: LA=12 PPL=6.899, LA=8 PPL=6.918, uniform turbo2 V=7.038 → 74.8% gap recovery
- [x] **Part 2C-E**: turbo1.5 threshold=1e-4 tested. PPL safe but no speed benefit. Reverted. Dead End.
- [x] **Part 3A-C**: D=96 warning already exists in llama-kv-cache.cpp. Consolidated fattn.cu D check as safety net.
- [x] **Part 5A**: Added Q4_K_M section to README
- [x] **Part 5B**: Added Recommended Configurations table
- [x] **Part 5C**: Updated cross-GPU to 751+ iterations
- [x] **Part 5D**: Added 3090 Ti multi-model table + 32K all-models table
- [ ] **Part 5E**: Update DISCUSSION_DRAFT_20969.md with Q4_K_M data
- [x] **Part 5F**: README numbers verified against AGENTS.md

## P2: If Time

- [ ] **Part 7A**: KV norm measurement in set-rows.cu
- [ ] **Part 7B**: Auto-asymmetric recommendation log
- [x] **Part 7C**: K=turbo4/V=q8_0 documented in README Tips section

## P3: Housekeeping

- [x] **Part 8A**: Update AGENTS.md with S23 results + dead ends
- [ ] **Part 8B**: Update vault (Dashboard, Benchmark Hub, Session 23, Roadmap)
- [x] **Part 8C**: Push to myfork (2 pushes: initial + after LA=12)
