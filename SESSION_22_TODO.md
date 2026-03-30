# Session 22 — Multi-Model Validation, Polish, Ship

## Part 1: Download Models
- [x] Llama-3.2-1B-Instruct Q6_K (975 MB, D=64)
- [x] Phi-3.5-mini-instruct Q6_K (3.0 GB, D=96)
- [x] Phi-4-mini-instruct Q6_K (3.0 GB, D=128)
- [x] Llama-3.3-8B-Instruct Q6_K (6.2 GB, D=128)
- [x] Gemma-3-12B-it Q6_K (9.0 GB, D=256)

## Part 2: Multi-Model Validation
- [x] Llama-3.2-1B (D=64) — ALL 4 types pass, asymmetric pass, prefill 38930 tok/s
- [x] Phi-3.5-mini (D=96) — graceful fallback ~89% f16, zero crashes
- [x] Phi-4-mini (D=128) — ALL 4 types pass, PPL turbo3=10.75 q8_0=10.17
- [x] Llama-3.3-8B (D=128) — ALL 4 types pass, PPL turbo3=10.72 q8_0=10.22
- [x] Gemma-3-12B (D=256) — ALL 4 types pass, PPL turbo3=10.74 q8_0=10.81
- [x] SESSION_22_MULTIMODEL.md created

## Part 3: Cross-GPU Data
- [x] 3090 Ti: 103 iterations, 10 PPL checks, 0 failures, PPL bit-exact (7.5535)
- [x] 4090M: 125 iterations (25 S20 + 100 S21), 0 failures, PPL bit-exact (7.5912)
- [x] SM89 sink bug found: TURBO_SINK_SIZE {1,4,16} segfault (documented)

## Part 4: FP4 E2M1 Q Precision Test
- [x] Python test: 99.5% map to zero, 3 values used — DEAD
- [x] PTX ISA search: no mixed fp16xE2M1 MMA — DEAD
- [x] Documented findings

## Part 5: Ship
- [x] README.md rewrite (S21 finals + multi-model + cross-GPU + limitations + tips)
- [x] DISCUSSION_DRAFT.md for #20969
- [x] Update Obsidian vault (Dashboard, Benchmark Hub, Roadmap, Session 22 note)
- [x] Push to myfork (2 commits: 77bac16 + 6e891da)

## Review Pass Fixes (from cross-GPU data review)
- [x] Fixed iteration counts: 3090 Ti 103, 4090M 125 (were wrong in all docs)
- [x] Documented SM89 sink crash bug in AGENTS.md, README.md, DISCUSSION_DRAFT.md
- [x] Added K=turbo4/V=q8_0 finding and LA=2 tip to README + DISCUSSION_DRAFT
- [x] Added D=80, D=112 to fallback dimension list in README
- [x] Added FP4 E2M1 to Dead Ends table in AGENTS.md
- [x] Fixed 4090M turbo2 vs q8_0 comparison number
