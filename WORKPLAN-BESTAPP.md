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

## ✅ ITEM 1 DONE 2026-08-15 (post-reboot): MMA-TURBO PORT SHIPPED, ALL GATES GREEN
- Reboot exonerated the kernels (user's VRAM-thrash call was right): shallow
  119 tok/s (was 84-97), 38K depth 74.1 (was 41.4 = +79%, VEC 9.8 eliminated).
- KLD gate: 0.004728 / top1 98.3% vs 0.0150 / 96.7% baseline = **−68% KLD**
  (corrected centroids; old table was mis-scaled). Beats q8K-hybrid arm at
  pure turbo4 bits. Needle FOUND @39K. First gate run was FALSE-catastrophic
  (stale Qwen3.6 f16 reference — quarantined; fresh 3.8-YaRN reference
  captured and ARCHIVED in kld38/).
- VRAM lesson: 32.1GB is fine when clean; Windows-side fragmentation after
  long uptime causes uniform 4-8x collapse — reboot clears; consider VRAM
  canary/margin later.
- Production: MMA binary + proxy v6.4 live. q8K option now LESS attractive
  (pure turbo4 beats old hybrid arm) — bit-allocation item deprioritized.

## SOAK + ACCEPTANCE VERDICTS (2026-08-15 late)
- B/C picks deployed (26651/26426/26771/24565). Soak on fixed binary CLEAN:
  echoes 107-128 through burst AND 39K big-ctx episode (decode 84.7 @39K —
  new depth best). In-session decay not reproduced post-fix; healthy shallow
  band = 110-125. Echo singles have ±20% noise — never diagnose on n=1.
- Acceptance: 67.5% @temp1, 72.9% @greedy (metrics-delta method). Regression
  window = 2 commits; suspect 77918caf3 (metrics-during-decode, 400L queue
  rework, in our pin) — could also be counter-accounting artifact. EXPERIMENT
  PARKED behind GDN: revert-77918caf3 A/B (~+14% decode IF real regression
  and IF 90% reference is real). 
- 27106 bisect: CLOSED as parked-experiment above (not a blind bisect).

## VRAM AUTOPSY (2026-08-15 night) — RETRACTS THE 'DRIFT' FINDING BELOW
USER WAS RIGHT: it was the VRAM ceiling, not environmental drift. Allocation
table (-lv 4 probe): weights 20819 + KV 5808 + 3× compute 1828 (target/draft/
vision scratch!) + RS 449 + draft-KV 363 + meta 248 ≈ 33.2GB vs 32.6 physical
— overcommitted and WDDM-paged SINCE MTP ADOPTION (draft ctx f16 KV 1.5GB +
triple scratch were never budgeted; 'zero VRAM cost' claim was the residency
cap masking overcommit). Paging throttled prefill from day one: yesterday's
1156 'baseline' was already paged.
FITTED CONFIG (live): ctx 294912 (288K, YaRN 1.125), -b 512 -ub 512,
--ctx-checkpoints 2, -ctkd/-ctvd turbo4. Result: 31364/32607 (1.24GB free,
counter finally moves), prefill @38K = 2605 tok/s (3.5x yesterday's best),
decode @38K = 78.6-81.9. Full 38K turn = 24s wall.
CONSEQUENCES: (a) every A/B measured today ran under paging — ngram-wash,
GDN-regression(acquitted), 24565 verdicts are all RE-ELIGIBLE for paired
re-test on the fitted config; (b) checkpoint sizes scale ~8KiB/token of
depth — cap 2 is mandatory; (c) OPEN OFFER to user: vision→CPU
(--no-mmproj-offload) reclaims ~1.8GB scratch = context back to ~352K,
cost = slow image ingestion only (awaiting user choice).

## ENVIRONMENTAL-DRIFT FINDING (2026-08-15 night) — RETRACTED, see autopsy above
- Prefill@38K across today: morning binaries 622-752 tok/s; night binaries
  (3 DIFFERENT builds: +GDN, GDN-reverted, no-24565) all 494-498. Decode@38K
  stable 71-74 THROUGHOUT. Verdict: ~35% prefill drift is ENVIRONMENTAL
  (Windows/dxg paging pressure accumulating with box uptime), selectively
  hitting large-batch kernels; NOT the GDN pick, NOT 24565 — both were
  wrongly suspected on stale baselines.
- RULE CHANGE: kernel A/Bs under ±20% require paired back-to-back runs
  alternating binaries within minutes, or reboot-fresh points. Single-arm
  vs historical baseline = invalid on this box.
- 24565: left reverted (neutral-measured, unproven either way; 15-line
  re-apply anytime). GDN 26001: left reverted (unproven, confounded).
- Ops implication: periodic reboot hygiene / VRAM-margin watchdog is the
  real fix; decode (user-facing) is drift-immune so far.

## GDN KERNELS VERDICT (2026-08-15 late) [AMENDED by drift finding above]
- PR 26001 chunked GDN prefill: TESTED-REGRESSIVE on SM120/Qwen3.8 — prefill
  494 tok/s vs 752 same-shape baseline (−35%), decode unchanged. REVERTED
  (revert of 84a01770a). Open PR, likely tuned for other HW; recheck if it
  merges upstream with scheduler companions.
- PR 22587 row-per-warp GDN decode: PARKED — rewrites the same kernel file
  incompatibly with 26001-era layout; dedicated merge session if pursued.
- Net: GDN prefill remains a future lever, not via these PRs as-is.

## (superseded pause block below, kept for history)
## ⏸ PAUSED 2026-08-15 — USER REBOOTING BOX (VRAM at ceiling 32162 MiB)

RESUME EXACTLY HERE AFTER REBOOT:
1. Stack is DOWN (reboot). Start: `bash ~/.config/llama-tcq/start-long-38.sh`
   — this launches the NEW MMA binary (staged in build-g1/bin, commit
   92df9fb24) + proxy v6.4. MMA gate is DEFAULT ON.
2. OPEN QUESTION AT PAUSE: post-MMA-deploy smoke was content-correct but
   **17.5 tok/s at SHALLOW ctx** (expected ~84-97). Depth test COMPLETED
   just before reboot: decode 14.3 tok/s @38K AND **prefill 169 tok/s
   (vs 1156)** — prefill does NOT use the MMA-decode gate, so the uniform
   ~4-8x collapse across prefill+decode+shallow+deep points at
   **VRAM-ceiling/dxg thrash (user's theory), not the kernel**. VRAM was
   32162/32.6GB. Reboot should clear it; if post-reboot numbers are healthy
   the kernel is exonerated — still run the kill-switch A/B for a clean
   MMA-vs-VEC comparison at depth before declaring the port a win.
   Secondary suspect if slowness persists: SM120 VGPR spill (#294/#295).
3. POST-REBOOT SEQUENCE: fresh smoke (shallow tok/s?) → depth_decode_test
   40000 → if still slow: restart with GGML_TURBO_MMA_FUSED=0 in env →
   re-smoke (expect VEC-speed ~84-97 shallow / 41.4 deep) → decide:
   depth-threshold gate patch vs kill-switch-off deploy. EITHER WAY the
   corrected centroids stay (independent of MMA path) → then KLD gate
   (expect ~-33% vs 0.0087 baseline; alphas tuned on OLD centroids — 3-pt
   recheck if off) + NIAH 130K needle.
4. VRAM watch: 32162 near 32.6 ceiling. Levers if pressure persists:
   --ctx-checkpoints (default 8 → 4), ctx trim, checkpoint count.
5. Then continue queue: B-list crash picks (26651 first), C-list perf picks
   (24565 SM120 FA config, GPU top-k 26812/25575), issue 27106 acceptance
   bisect (possibly biggest decode win), GDN kernels (26001/22587), then
   remaining board items per queue above.

## STATE LOG
- MMA PORT APPLIED (5 cherry-picks, conflicts resolved): b3e51cf3d (MMA
  turbo4) + 77ab7e988 (corrected 4-bit centroids + 66B block — kept OUR
  binary-search encode with THEIR values; deduped double table) + 4e223ee9a
  (turbo3/2 extend) + 545092c36 (3-bit centroids, kept constexpr) +
  539ce5de9 (gate: DEFAULT ON, GGML_TURBO_MMA_FUSED=0 kill-switch).
  CMake glob covers instances. Active 4-bit struct 66B asserted; legacy
  #else branch untouched. CPU tables corrected (6 value-hits verified).
  BUILD IN FLIGHT. Gate battery for deploy: (1) smoke; (2) KLD-quick vs
  baseline 0.0087@2048/alpha1.00 — expect ~-33%; alphas were tuned on OLD
  centroids → if KLD off, 3-point alpha recheck queued; (3) depth A/B 40K:
  MMA-on vs GGML_TURBO_MMA_FUSED=0 (env, same binary) vs old 41.4 baseline;
  (4) NIAH 130K single-needle; (5) deploy w/ proxy v6.4 (circuit breaker)
  + archive slot files (66B format change invalidates).
- 2026-08-15 eve: Tier-0 done (see RESEARCH-2026-08.md header). Production:
  checkpoint binary, draft-mtp n_max=2, VRAM 32039. Baselines for gates:
  shallow decode ~84-97 tok/s, 38K-depth decode 41.4, prefill@38K 1156 tok/s,
  restore-reuse 49 tok. KLD tool: quality-tests/kl_divergence.py (2048-tok
  prompts, cache_prompt false). Depth probe: scratchpad/depth_decode_test.py.
- MMA port MAP (recon done): source origin/feature/turboquant-kv-cache,
  commits b3e51cf3d^..539ce5de9 span (5 commits: MMA turbo4 +445, extend
  turbo3/2 +147, turbo2 gate D128, corrected 4-bit centroids + rnorm drop
  68→66B, turbo3 centroids). Files: fattn-mma-f16.cuh loaders,
  fattn-mma-turbo.cuh (118L), fattn.cu gate, 14 template instances
  (dkq128/256). Coverage: turbo4/3 D{128,256}, turbo2 D128, decode-only
  Q<=4, K==V, turing_mma. Type remap THEIRS default-branch 43/44/47
  (t2/t3/t4) → OURS 82/80/81. CRITICAL: our 4-bit centroid table is
  MIS-SCALED (0.174 max vs corrected 0.242; PR#197: KLD −33%) — adopt
  their tables + 66B block IN the port; update VEC+set_rows+convert tables
  same commit (runtime-only format: archive slot files, no model requant).
  Keep graph-level inverse WHT (skip OSCAR2 in-loader variant). nstages=0,
  raw byte pitch, need_f16=false — the 3 gotchas. TCQ/T1.5 stay VEC.
  Gates after: KLD quick (expect improvement), depth A/B vs 41.4@38K,
  NIAH quick, smokes.
- NEW QUEUE ITEMS from crawl: TheTom PR#267 LUT register hoist (+9.9%
  turbo4 decode, take after base port); TheTom PR#271 batched checkpoint-
  restore prefill (40→1115 t/s, UNVERIFIED — evaluate); spiritbuun
  0461612ba TCQ codebook in smem for MMA (+32% tg@32k — only if we adopt
  TCQ in prod); spiritbuun 7a7dfc7e2 cooperative FWHT128 encode
  (set_rows 131→9.9ms — prefill lever, evaluate after MMA); OSCAR2
  6ea6a2cd0 D512+cp.async, 4622556dc nbatch_fa long-ctx scaling (mine
  later if depth perf still lacking).
- Item 5 ops round 3: DONE in commit 74cc9b578 (circuit breaker both paths,
  coredump env, graph-reuse fallback doc; resume-tokens closed not-needed —
  checkpoint fix makes retry ~1s; tool-salvage deferred to battery arc).
  Proxy 49/49. Deploys at next restart window (batch with MMA).
- Re-sweep DONE (both agents). Upstream sweep verdict: merged-side clean
  (12-commit gap touches nothing of ours) — misses are OPEN PRs/issues.
  QUEUE INSERTIONS (priority order after MMA batch):
  A. **issue 27106 acceptance regression** — draft acceptance collapses on
     Qwen3.8 from b10430; WE ARE b10435. Vulkan 92% vs CUDA ~36-50%
     (also 26750). Our 59-63% may be regressed. Bisect b10428..b10430,
     fix/revert → potentially biggest decode win. DO AFTER MMA BATCH.
  B. Cheap crash/correctness picks: PR 26651 (sampler abort w/ draft-mtp
     30K+ — likely bites us), 26426 (dry sampler heap), 26771 (CUDA graph
     recapture after pool flush), 25636 (K-shift null rotation buffer).
  C. Cheap perf picks: PR 24565 (SM120 FA MMA config, 15 lines), 26812+
     25575 (GPU top-k/argmax — we pay full-logits D2H per token at
     top-k 20), 25635 (XOR-swizzle FA smem tiles), 26079 (mvq→MMQ
     crossover tunable).
  D. GDN kernels: PR 26001 chunked prefill (+939L, DeltaNet prefill =
     bottleneck), 22587 row-per-warp GDN decode (bench'd on 5090-class).
  E. Watchlist issues (our shape): 27090 (520K prefill death), 27102
     (sustained-decode XID8 lockup), 26609 (FA IMA on qwen35), 25717
     (vision mmproj IMA w/ FA), 23606 (NaN >80K on 3.6-hybrid, stale).
  F. VRAM smalls: PR 25465 (spec ctx overalloc under fit-params), 26574,
     26487 (blocking sync kills 100% CPU busy-wait), 26130 (VRAM in
     /metrics). Plus 21551 (warn-only unsupported KV types — helps our
     upstreaming). NOT running --cache-ram (27148 cross-restore bug).
