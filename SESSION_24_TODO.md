# Session 24 — Iterate and Innovate

## P0 — Critical

- [ ] **Fast-math softmax** — Replace `expf` with `__expf` in VEC kernel softmax path (+5-15% expected)
- [ ] **Fix boundary V for hybrid architectures** — LA=12/13 use raw `il` instead of KV layer ordinal
- [ ] **Create LA=14** — Combined K+V boundary mode
- [ ] **Test LA=12 on MoE model** — TheTom showed 91% gap recovery on 64-layer
- [ ] **Test LA=2 + LA=12 stacking** (LA=15)
- [ ] **Q4_K_M PPL ctx=2048** — Missing headline data
- [ ] **Q4_K_M extended context** — 64K turbo2, 128K turbo1.5
- [ ] **Q4_K_M asymmetric rescue speed** — q8_0 K + turbo3 V

## P1 — Important

- [ ] **elect_leader() warp optimization** — Replace `threadIdx.x % 32 == 0` with PTX `elect.sync`
- [ ] **Producer-consumer warp split (Tawa)** — Overlap K load/dequant with Q*K compute
- [ ] **ARKV auto layer-adaptive** — Entropy-based per-layer turbo type assignment
- [ ] **turbo3-K / turbo2-V asymmetric benchmark** — seanrasch reports 5.33x at +0.26 PPL
- [ ] **Multi-model ctx=2048 PPL** — Fill gaps for validation models
- [ ] **turbo2 short decode for validation models** — Only have turbo3 and q8_0
- [ ] **Gemma-3-12B at 32K turbo2 (warm)** — S22B had cold-start variance
- [ ] **FA warning for non-FA users** — Warn when turbo types without `-fa`
- [ ] **GQA >8:1 vulnerability warning** — HyperionMS2040 finding

## P2 — Quality Assurance

- [ ] **turbo3 at 131K PPL on Q6_K** — Long-context stability check
- [ ] **turbo2 at 64K PPL on Q6_K** — Confirm long-context champion
- [ ] **All types at 32K PPL ctx=2048** — Verify turbo4, turbo2, turbo1.5
- [ ] **README update** — New headline numbers
- [ ] **DISCUSSION_DRAFT update** — Latest data for #20969 post
- [ ] **AGENTS.md update** — S24 results

## P3 — Housekeeping

- [ ] **Vault: Session 24 note**
- [ ] **Vault: Roadmap update**
- [ ] **Vault: Benchmark Hub update**
- [ ] **Vault: Dashboard update**
- [ ] **Vault: Dead Ends update**
- [ ] **Push to myfork**

## Autoresearch Log

| # | Optimization | Result | Status |
|---|-------------|--------|--------|
| 1 | Fast-math softmax (__expf) | | |
| 2 | elect_leader() | | |
| 3 | | | |
