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
- [x] **turbo2 short for validation models** — Phi-4=285, 8B=191, 12B=114 (beats q8_0 everywhere)
- [x] **Gemma-3-12B warm 32K** — 90.50 ± 7.24 (vs S22B 82.87 ± 18.40, much tighter)
- [x] **Llama-3.3-8B 64K turbo2** — 60.41 tok/s (faster than 27B at SHORT context!)
- [x] **elect_leader()** — DEAD END: targets outside hot loop
- [x] **Tawa producer-consumer** — DEAD END: bandwidth-limited kernel
- [ ] **ARKV auto layer-adaptive** — Deferred to S25 (complex)

## P2 — Quality Assurance

- [x] **turbo3 ctx=2048 PPL** — 5.6744 (=q8_0 5.6744, bit-exact match preserved with __expf)
- [x] **All types ctx=2048 PPL** — turbo4=5.6937, turbo2=5.8922, turbo1.5=6.1028
- [x] **turbo2 64K Q6_K speed** — 47.13 tok/s
- [x] **turbo1.5 131K Q6_K speed** — 17.95 tok/s
- [x] **AGENTS.md update** — S24 results, 3 new dead ends, performance table updated
- [ ] **DISCUSSION_DRAFT update** — Deferred to S25

## P3 — Housekeeping

- [x] **Vault: Session 24 note** — Created
- [x] **Vault: Benchmark Hub update** — S24 evolution row added
- [x] **Vault: Dashboard update** — Headline numbers updated
- [x] **Vault: Dead Ends update** — 3 new entries (#29-31)
- [x] **Push to myfork** — 4 commits pushed

## Autoresearch Log

| # | Optimization | Result | Status |
|---|-------------|--------|--------|
| 1 | Fast-math softmax (__expf) | short +0.67%, 32K +3.69%, PPL identical | **COMMIT** |
| 2 | elect_leader() | Targets outside hot loop, no perf benefit | **DEAD END #29** |
| 3 | Tawa producer-consumer | Bandwidth-limited kernel, staging adds overhead | **DEAD END #30** |
| 4 | cp.async K loading | K loads directly to registers, staging adds overhead | **DEAD END #31** |
| 5 | __frcp_rn reciprocal | Single division at end, not in hot loop | **SKIP** |

## Complete S24 Benchmark Data

### Q6_K (27B Dense, RTX 5090)

| Type | Short | 32K | 64K | 131K | PPL ctx=512 | PPL ctx=2048 |
|------|------:|----:|----:|-----:|:-----------:|:------------:|
| q8_0 | 58.58 | 47.99 | — | — | 6.759 | 5.674 |
| turbo4 | 58.87 | 46.06 | — | — | 6.825 | 5.694 |
| turbo3 | **65.05** | **53.44** | — | — | 6.852 | **5.674 (=q8_0)** |
| turbo2 | 60.67 | 52.82 | **47.13** | — | 7.080 | 5.892 |
| turbo1.5 | 58.74 | 45.06 | — | **17.95** | 7.312 | 6.103 |

### Q4_K_M (27B, RTX 5090)

| Config | Short | 32K | 64K | PPL ctx=2048 |
|--------|------:|----:|----:|:---:|
| turbo3 | 77.25 | 58.08 | — | **7.716 (beats q8_0!)** |
| turbo2 | 74.94 | 61.49 | **52.92** | — |
| turbo1.5 | — | — | **42.62** | — |
| q8_0 | 73.40 | 57.24 | — | 7.730 |
| q8_0-K/turbo3-V | **76.34** | — | — | — |

### Validation Models turbo2 Short (RTX 5090)

| Model | D | turbo3 | turbo2 | q8_0 | turbo2 32K |
|-------|:-:|-------:|-------:|-----:|-----------:|
| Phi-4-mini | 128 | 274 | **285** | 275 | 119.79 |
| Llama-3.3-8B | 128 | 179 | **191** | 181 | 87.26 |
| Gemma-3-12B | 256 | 95 | **114** | 91 | **90.50** |
| Llama-3.3-8B 64K | — | — | **60.41** | — | — |

### Asymmetric & Layer-Adaptive

| Config | PPL (Q6_K ctx=512) | Compression |
|--------|:--:|:-:|
| turbo3-K / turbo2-V | 7.038 | 5.33x |
| LA=12 (MoE turbo3/turbo2) | 5.226 | ~5x |
| LA=13 (narrow, 27B turbo3/turbo2) | 7.040 | ~5x |
| LA=15 (stacked, 27B turbo3/turbo2) | 7.041 | ~5x |
