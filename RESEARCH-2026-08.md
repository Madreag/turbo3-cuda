# Research Campaign — 2026-08-15

> **TIER 0 EXECUTED same evening — measured outcomes:**
> - Checkpoint cherry-picks (#26885, #25592×2, #24891, #26004) applied,
>   built, deployed. **Restore-reuse hole CLOSED and verified**: post-restore
>   turn now processes 49 tokens (was 4,166). Owner swaps at depth: minutes →
>   ~1s. Conflicts resolved: 24891's n_past_common clamp kept; its
>   invalidation condition subsumed by 25592's.
> - **ngram-mod cascade: WASH on our model** (same-day same-prompt: 85.5 vs
>   83.8 pure) — ngram drafts don't beat the trained MTP head on fresh code.
>   Not adopted. `--spec-draft-backend-sampling`: also wash (83.0). Not
>   adopted. Yesterday's 89-98 band was prompt-set variance.
> - **Verify-batch cliff hypothesis INVERTED by measurement** at 38K depth:
>   n_max=2 (batch 3 → MMA+F16-dequant) = 41.4 tok/s; n_max=1 (batch 2 →
>   VEC) = **9.8 tok/s**. The F16-dequant MMA path is the FAST path at depth;
>   VEC in-kernel dequant is the bottleneck. Implication: MTP-off (batch 1 =
>   VEC) ≈ ≤10 tok/s at depth → MTP is ~4× at depth, and the **MMA-turbo
>   kernel port (T1-6) is promoted to top strategic priority** (native turbo
>   MMA skips the per-step F16 conversion). n_max=2 kept.
> - Production final: checkpoint binary + unchanged validated flags, smoke
>   96.7 tok/s finish=stop, VRAM 32039 MiB (+~500MB vs pre-checkpoint —
>   in-memory checkpoints; monitor, reduce --ctx-checkpoints if WSL dxg
>   squeezes reappear).

Five parallel research tracks (fork ecosystem, upstream movement, KV-compression
frontier, decode-speed frontier, serving/reliability) + local verification
probes against the production tree. Full agent evidence in session; this file
is the decision board. Production context: Qwen3.8-27B hybrid, turbo4 KV,
409K YaRN, MTP n_max=2, vision, coding-agent primary. Pin = b10435 (Aug 14),
11 commits behind master.

## Verified locally during the campaign

- **Prefix reuse WORKS in-lifetime** (turn-2 probe: 58 of 4,120 tokens
  processed). The "hybrid models full-reprefill every turn" static-analysis
  claim is REFUTED for our build (Aug-14 recurrent-rollback merges are in).
- **Prefix reuse DIES across slot restore** (probe: save→erase→restore→turn
  reprocessed 4,166 of 4,166). Hybrid context-checkpoints don't serialize
  into slot files (upstream #26004, open). **Every owner swap pays full
  re-prefill today** — restore is currently decorative for reuse.
- **`return_progress` is ALIVE in our tree** (server-task.h:55) — the
  bughunt-era "removed upstream, accepted degradation" record was wrong;
  proxy progress-comments are presumably functional.
- `-ctkd/-ctvd` (draft KV types) and `--spec-draft-backend-sampling` exist
  in our binary, both unused.

## TIER 0 — verified need, hours-to-days each

1. **Checkpoint cherry-pick cluster**: #26004 (checkpoints into slot files,
   +195) + #25592 (hybrid seq_pos_min gate) + #24891 (checkpoint invalidation
   after tool-call turns — our exact workload). Closes the verified
   restore-reuse hole; #25592 was validated upstream on Qwen3.8-27B + MTP on
   2026-08-14. Payoff: owner swaps and restarts stop costing minutes of
   re-prefill at depth. [speed+reliability]
2. **#26885 grammar speedup** — 6 lines, 1.2-1.3× on grammar evaluation.
   Trivial cherry-pick. [speed]
3. **`--spec-type ngram-mod,draft-mtp` cascade A/B** — config-only; impls
   cascade in order (speculative.cpp:2633-2670). Sibling-model measurement:
   42 → 56.6 tok/s over MTP alone; n-gram drafting is strongest exactly on
   code (SuffixDecoding 5.3× AgenticSQL; beats EAGLE-3 on code-editing
   benches). One restart + coding battery. [speed]
4. **FA verify-batch cliff measurement** — MTP n_max=2 makes the verify batch
   3 rows; fattn dispatch sends >2 rows to MMA, which has NO turbo instances
   → need_f16_K/V dequants the cache per step (fattn-common.cuh:1673,1701).
   Suspect this is the REAL art-turn/depth MTP penalty (not just grammar).
   A/B decode at ~50-100K ctx: n_max=2 vs n_max=1 (batch 2 = stays VEC).
   If confirmed: either run n_max=1 or fast-track the MMA-turbo port (T1-6).
   [speed at depth + closes an open mystery]
5. **Small flags**: `--spec-draft-backend-sampling` (+~8% claimed at n_max=3,
   less at 2); `-ctkd f16/q8_0` for the draft context; evaluate ik_llama
   commit 7642ac3ec (CUDA Q↔f16 copy inefficiency, ~1 day). [speed, small]

## TIER 1 — high value, medium effort

6. **MMA-turbo decode path** (the standing TheTom debt; ecosystem-validated).
   Now doubly motivated: also structurally fixes T0-4's verify-batch cliff
   (no F16 fallback for batch>2). Read TheTom PR #234 ("OSCAR2 quantized
   KV-cache", closed unmerged — CUDA MMA-turbo paths validated on Qwen3.6)
   before writing anything. [speed, biggest kernel lever]
7. **Quest-class sparse decode retrofit** — the strategic play at 400K.
   Frontier consensus: at 4.25bpv we're at the quantization ceiling ("stop
   optimizing bits, start optimizing reads"); SAW-INT4 (2604.19157) directly
   validates turbo4's design. Upstream merged reusable block-sparse plumbing
   (MSA #24908: block scoring → top-k → get_rows → FA over selected KV).
   Open design problem: per-page min/max metadata over rotated+quantized K.
   Kascade (2512.16391) anchor-layer reuse fits a 16-attn-layer model well.
   [speed at depth; multi-week]
8. **Trajectory/multi-hop quality battery** — highest-signal warning of the
   scan: third-party REFRACT data (discussion #20969) shows ctv=turbo3 holds
   PPL/KLD while trajectory score collapses (84→58 on 27B) — V-cache damage
   that PPL/KLD/NIAH cannot see, aligned with our own "multi-hop is the
   cliff" note. Unreplicated, but it means: (a) build a multi-hop/agentic
   battery (Phase-F++) BEFORE any V-bit reduction; (b) it also gates T2-11.
   We run turbo4-V (empirically clean renders/coding), but our gates are
   blind on this axis. [quality measurement backbone]
9. **Draft-context warmup after restore** + slot-restore draft-KV note
   (upstream C7). [speed, small]

## TIER 2 — strategic options

10. **Grammar stack**: rebuild with `-DLLAMA_LLGUIDANCE=ON` (bundled version
    in pin is v1.0.1 vs upstream ≥1.7.6 — bump), gate the %llguidance
    non-llg-build DoS (#25960), measure vs native PEG on our captured tool
    grammars. Mine #26551 (xgrammar-verified MTP drafts, 3156 lines, open)
    for the grammar-masked-drafting design — recovers speculation on tool
    turns (TRT-LLM ablation +6-12% accept length). Backend-sampling-under-
    grammar remains disabled upstream (sampling.cpp:416) — no fix inbound.
11. **Bit-allocation search** (free, config-only, GATED on T1-8 battery):
    K-heavy splits validated by three papers (KQV>QKQV KLD at every budget;
    K4/V2 75.2% vs K2/V4 54.7%). Our q8-K option: +8.5 KiB/tok → ~280-320K
    ctx; NOTE #24403 (merged Jul 6) extended V-type FA validation — check
    asymmetric configs against it. Per-layer via the (now-fixed) adaptive
    modes. Block-GTQ (2606.24033, built on TurboQuant-MSE) only pays at
    turbo2/3 (K2V2 NIAH 70.6→97.4; K3V3 +0.7). InnerQ calibration: honest
    verdict — no published gains above 3 bits; park until a turbo2/3 push
    (then OSCAR-style attention-aware rotations, arXiv 2605.17757, CUDA
    port needed).
12. **Upstream micro-sync to b10447** (11 commits) — cheap; watch
    `--load-mode` flag rename and the yield_to_queue server thread redesign
    (#27133). Also monitor #27109/#27140 (prefill collapse on 4-bit KV
    qwen35 — our shape; check whether turbo converters are vectorized in
    the MMA/prefill to_fp16 path).
13. **Ops hardening round 3**: SSE resume tokens (id: + Last-Event-ID replay
    from capture buffer); owner-lock circuit breaker (fail-fast during
    server restart instead of queueing); WSL VRAM soft-margin (~256MB) +
    `CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1`; MTP+CUDA-graph crash tripwire
    (#26558: KV-saturation → cuBLAS-7; fallback LLAMA_GRAPH_REUSE_DISABLE=1);
    proxy tool-call salvage (repair malformed blocks instead of dropping).
    `--cache-ram` (merged, host-RAM LRU prompt cache) as a complement to
    slot files once checkpoints serialize.
14. **Upstream PR karma**: our 4 hardening patches + the checkpoint-gap
    verification evidence.

## Explicit SKIPS (reasons on record)

Token eviction (SnapKV/H2O/PyramidKV: NIAH −56pts, breaks agent prefix
reuse); low-rank KV/Palu (quant beats rank by 4-364 PPL at matched budget,
gap grows with GQA); draft trees (10-20% at bs=1, pathological on 48
DeltaNet layers); layer-skip self-spec (0.77-1.28× < our 1.55×); CUDA-graph
tuning (≤1.2× ceiling, disabled under MTP at batch>1); FP4-KV tensor cores
(D=256 exceeds SM120 SMEM budget for tcgen05-less mma); batch-invariant
kernels (bs=1); exllamav3 H32 Hadamard rotate (their own numbers: flat on
GQA models); NSA/MLA/CLA retrofits (need training); craftogrammer (dead);
eager per-request spec upstream fix (confirmed: none coming — fork patch
remains the only path, currently SHELVED by user).

## Record corrections from this campaign
- return_progress: ALIVE in tree (bughunt record corrected).
- "Full re-prefill every turn": refuted in-lifetime; CONFIRMED post-restore.
- Pin freshness: b10435 = Aug 14 release; sync was near-tip, not "early Aug".
