# WORKPLAN: Best Qwen3.8-27B App (2026-08-15 →)

User directive: implement EVERYTHING on the research board that increases
quality/speed or decreases VRAM, starting with the MMA-turbo port. Re-sweep
upstream + TheTom + spiritbuun for missed items. Do not stop until each item
is implemented or tested-not-needed. Single RTX 32GB SM120, WSL2.

RULES: every change → build → gate (smokes + KLD-quick or NIAH-quick where
relevant + depth-decode probe) → deploy → record here. One restart window per
deploy batch. Slot files archived on config change. Rollback binaries kept.

## Queue (execute in order; update STATE as work proceeds)

1. [IN PROGRESS] **MMA-turbo decode port** — native turbo MMA FA kernels
   (skip per-step F16 dequant; beats measured 41.4 tok/s @38K; VEC path 9.8
   is the depth bottleneck). Source: TheTom fork (origin remote) MMA branch +
   dead PR #234 "OSCAR2" for reference. Steps: locate impl → study → port to
   sync tree (types 80-85 renumber!) → build → depth A/B vs 41.4 baseline →
   KLD parity gate → deploy.
2. [PENDING] **Missed-items re-sweep** (2 agents launched) → fold findings
   into this queue.
3. [PENDING] **Micro-sync to master b10447** (11 commits; watch --load-mode
   rename, yield_to_queue redesign; re-verify our 4 carried patches; check
   #27140 vectorized dequant relevance to turbo converters).
4. [PENDING] **llguidance rebuild** (-DLLAMA_LLGUIDANCE=ON, bump bundled
   version, gate %llguidance DoS #25960, measure PEG vs llg on captured tool
   grammars).
5. [PENDING] **Ops round 3** (proxy/scripts): SSE resume tokens
   (Last-Event-ID), owner-lock circuit breaker during restarts, WSL VRAM
   soft-margin + CUDA_ENABLE_COREDUMP_ON_EXCEPTION, MTP+CUDA-graph crash
   tripwire (LLAMA_GRAPH_REUSE_DISABLE fallback), proxy tool-call salvage.
6. [PENDING] **Trajectory/multi-hop battery** (quality gate for agentic axis;
   REFRACT-style; prerequisite for any bit reduction) + fold Phase-F coding
   battery.
7. [PENDING] **Quest-class sparse decode** (biggest new build: page metadata
   over rotated K + top-k gather via upstream MSA plumbing; design doc first;
   compounds with MMA port).
8. [PENDING] **Bit-allocation experiments** (GATED on #6): q8K/turbo4V
   (check #24403 V-type validation), per-layer adaptive modes, TCQ A/B at
   fixed bits.
9. [PENDING] **Draft-ctx experiments**: -ctkd/-ctvd types, draft warmup
   after restore (C7).
10. [PENDING] **Upstream PR submissions** (karma batch: crash hardening,
    ctx-cap, parse-degrade, checkpoint evidence).

## STATE LOG
- 2026-08-15 eve: Tier-0 done (see RESEARCH-2026-08.md header). Production:
  checkpoint binary, draft-mtp n_max=2, VRAM 32039. Baselines for gates:
  shallow decode ~84-97 tok/s, 38K-depth decode 41.4, prefill@38K 1156 tok/s,
  restore-reuse 49 tok. KLD tool: quality-tests/kl_divergence.py (2048-tok
  prompts, cache_prompt false). Depth probe: scratchpad/depth_decode_test.py.
- MMA port: starting — locating TheTom impl.
