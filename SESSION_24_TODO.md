# Session 24 — Iterate and Innovate

## P0 — Critical

- [x] **Fast-math softmax** — `__expf` in VEC softmax: short +0.67%, 32K +3.69%, PPL bit-exact
- [x] **Fix boundary V for hybrid architectures** — kv_ord instead of raw il, correct for all archs
- [x] **Create LA=13/14/15** — Narrow V boundary, combined K+V, stacked LA=2+LA=12
- [x] **Test LA=12 on MoE** — PPL 5.2259 (31% gap recovery vs uniform 5.2633)
- [x] **Test LA=15 stacked** — PPL 7.0408 (=LA=12, K promotion adds no benefit)
- [x] **Q4_K_M PPL ctx=2048** — 7.7164 (**beats q8_0 7.7304!**)
- [x] **Q4_K_M extended context** — turbo2 64K=52.92, turbo1.5 64K=42.62
- [x] **Q4_K_M asymmetric rescue** — q8_0-K/turbo3-V=76.34 tok/s

## P1 — Important

- [x] **turbo3-K/turbo2-V asymmetric** — short=52.94, 32K=51.52, PPL=7.038, 5.33x compression
- [x] **GQA >8:1 warning** — Added, fires for n_head/n_head_kv > 8
- [x] **FA auto-enable** — Already existed (confirmed)
- [ ] **elect_leader()** — SKIP: targets outside hot loop, no benefit
- [ ] **Tawa producer-consumer** — SKIP: kernel is bandwidth-limited, staging adds overhead
- [ ] **ARKV auto layer-adaptive** — TODO if time permits
- [ ] **turbo2 short for validation models** — TODO
- [ ] **Gemma-3-12B warm 32K** — TODO

## P2 — Quality Assurance

- [x] **turbo3 ctx=2048 PPL** — 5.6744 (=q8_0 5.6744, bit-exact match preserved)
- [x] **All types ctx=2048 PPL** — turbo4=5.6937, turbo2=5.8922, turbo1.5=6.1028
- [x] **turbo2 64K Q6_K speed** — 47.13 tok/s
- [x] **turbo1.5 131K Q6_K speed** — 17.95 tok/s
- [ ] **README update** — IN PROGRESS
- [ ] **AGENTS.md update** — IN PROGRESS
- [ ] **DISCUSSION_DRAFT update** — TODO

## P3 — Housekeeping

- [ ] **Vault: Session 24 note**
- [ ] **Vault: Benchmark Hub update**
- [ ] **Vault: Dashboard update**
- [ ] **Vault: Dead Ends update**
- [ ] **Push to myfork**

## Autoresearch Log

| # | Optimization | Result | Status |
|---|-------------|--------|--------|
| 1 | Fast-math softmax (__expf) | short +0.67%, 32K +3.69%, PPL identical | **COMMIT** |
| 2 | elect_leader() | Targets outside hot loop, no perf benefit | **SKIP** |
| 3 | Tawa producer-consumer | Bandwidth-limited kernel, cp.async staging adds overhead | **SKIP** |
| 4 | cp.async K loading | K loads directly to registers, staging adds overhead | **SKIP** |
| 5 | __frcp_rn reciprocal | Single division at end, not in hot loop | **SKIP** |

## Key S24 Results

| Metric | Value | Notes |
|--------|-------|-------|
| turbo3 short (Q6_K) | **65.05** | +0.67% from __expf (was 64.62 baseline) |
| turbo3 32K (Q6_K) | **53.44** | +3.69% from __expf (was 51.54 baseline) |
| turbo3 ctx=2048 PPL | **5.6744** | =q8_0 (bit-exact) |
| Q4_K_M turbo3 ctx=2048 PPL | **7.7164** | Beats q8_0 (7.7304)! |
| Q4_K_M turbo2 64K | **52.92** | Faster than Q6_K turbo3 at short! |
| Q4_K_M turbo1.5 64K | **42.62** | 8x compression at 64K |
| Q4_K_M q8_0-K/turbo3-V rescue | **76.34** | Safe asymmetric for small models |
| turbo2 64K Q6_K | **47.13** | Long-context champion confirmed |
| LA=12 on MoE | **5.2259** | 31% gap recovery (uniform=5.2633) |
