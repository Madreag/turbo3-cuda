# Session 24B — Complete Everything, Close turbo4 Gap

## P0 — turbo4 Gap Closure

- [ ] **2A: nthreads_KQ=8 for turbo4/turbo1.5** — 1-line fix, expected +3-5% at 32K
- [ ] **2B: Float Q for turbo4/turbo1.5** — remove q8_1 overhead, REVERT if 32K regresses
- [ ] **2C: Centroids to registers** — static constexpr for all types
- [ ] **2D: Norm out of loop** — factor norm to end of accumulation
- [ ] **2E: Full regression + AmesianX re-test** — verify gap closed

## P0 — Remaining S24 Data

- [ ] **1A: Q4_K_M asymmetric rescue (q8_0-K/turbo4-V)** — speed test
- [ ] **1B: Verify new LA modes if code changed**

## P1 — Code Review + Quality

- [ ] **3A: GQA warning** — DONE S24
- [ ] **3B: FA auto-enable** — DONE S24
- [ ] **4: Code review + bug hunt**

## P2 — Documentation

- [ ] **5: Long-context PPL verification**
- [ ] **6: AGENTS.md Rule 6 fix + turbo4 updates**
- [ ] **7: Vault update + push**
