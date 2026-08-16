# TurboQuant Serving Stack — Handoff (2026-08-16, post-consolidation)

Supersedes the 2026-08-09 Failure-B edition (git history holds it; its
grammar-bomb forensics remain valid record).

## BRANCH MAP (ONE git repo, four worktrees, ONE GitHub remote:
## github.com/Madreag/turbo3-cuda — consolidated 2026-08-16, 8→6 branches)

| branch | worktree | what it is |
|---|---|---|
| `release/cuda-optimized` | turboquant-kv-cache | PUBLIC MAIN: README/showcase, kernel-optimization history, TCQ |
| `sync/2026-08` | turboquant-sync | **THE FORK / PROD SOURCE** — upstream b10448 + all carries + sparse-P0 tooling + fused-default + get_rows_keep + GDN #22587. Tip == the shipped prod binary. (feature/sparse-decode and feature/gdn-22587 were linear ancestors — fast-forwarded in and deleted.) |
| `hermes/server-foundation` | turboquant-g1 | ops docs, quality-tests, deploy mirrors, boards/ledgers (this file) |
| `pr/ctx-cap-rope-scaling` ·  `pr/parse-degrade-safety` · `pr/state-restore-hardening` | tq-prstage | staged upstream-ready PRs — USER opens them; assistant never submits to external repos |

Everything is pushed; all six tips verified == GitHub 2026-08-16.

## CURRENT STATE — one screen

- **Model:** Qwen3.8-27B Q6_K (hybrid: 48 DeltaNet + 16 attention layers,
  head_dim 256, GQA-4), native MTP head. Vision via mmproj **on CPU**
  (~21 s/image encode, once per image; text speed unaffected).
- **Context:** TWO PROFILES (2026-08-16, ctx push executed). Default =
  SPEED: 327,680 (320K), YaRN 1.25, MTP n3 — at its VRAM ceiling (~334K
  max per vram_law.py). Opt-in = MAX-CTX: 409,600, YaRN 1.5625, MTP OFF
  (spec-off frees ~1.9 GB of draft/spec compute — measured, not the old
  0.8 GB estimate). Both fill-ladder-verified VRAM-static. Theoretical
  MTP-off ceiling ~484K but that needs YaRN ~1.85 — beyond validated
  territory; 1.5625 is the NIAH-validated boundary.
- **KV cache:** turbo4 (66-byte blocks, 4.125 bpv, corrected Lloyd-Max
  centroids) K+V, target AND draft (`-ctkd/-ctvd turbo4`). Alphas 1.00.
  MMA-turbo fused decode kernels (kill-switch `GGML_TURBO_MMA_FUSED=0`).
- **Speculative:** `--spec-type draft-mtp --spec-draft-n-max 3` (n3 ADOPTED
  2026-08-16: code decode 96→113 +17%, copy/edit-loop 115→140 +22%, prose
  −6% — coding-primary trade; 5-seed ledger gate passed; p-min gate and
  ngram cascades measured WORSE on our fused stack — board P9/P1).
  Acceptance is temp/content-dependent: ~0.88/0.77 greedy, ~0.75/0.41
  code/prose at temp 1.0 (the old "67%" was a blend; entropy-explained,
  TEMP-STUDY). Spec-vs-off greedy output differs by fp-tie class at ANY
  n_max (accepted; P10).
- **Fork:** branch `sync/2026-08` (consolidated), upstream-tip b10448 +
  carries. Binary: `build-g1/bin/llama-server`
  (= gdn22587 build: 24565 + fused-MMA-default + GDN row-per-warp #22587;
  rollback chain: `.pre-gdn22587` → `.mainline` → `.pre-bughunt`). Fused
  MMA-turbo decode is ON via launcher env (2026-08-15 adoption: +2.2% @38K,
  +8.7% @121K decode); GDN #22587 adopted same day (+2.8% decode @38K,
  +1.7-1.8% prefill both depths, +0.4% @121K).
- **Measured (fitted, unpaged, post-adoptions):** at temp 1.0 @38K: code
  ~113 decode (n3), prose ~79, copy/edit-loop ~140; greedy @38K ~108-113;
  121K ~77 (n2-era, pre-n3); prefill ~2,700 @38K / ~1,715 @121K. VRAM
  31.77/32.61 GB (~840 MB headroom post-n3 — the +1 recurrent copy;
  fill-ladder P3 required before ANY ctx increase).
  (Older "84-89 @38K" figures were VEC-path temp-1.0 probes — superseded.)
  KLD vs archived f16 ref: 0.0050-0.0059 / top-1 96.7-98.3% (same-day binary
  pair; the archived 0.00473 predates b10448+24565 binary evolution).
  Battery @64K on the shipped binary: 6/6, ledger 8/8 (traj_gdn64.json).
  VRAM 31.1-31.7 of 32.6 GB.

## OPERATE

```bash
bash ~/.config/llama-tcq/start-long-38.sh   # SPEED profile (default): 320K,
                                            # MTP n3 — code ~113 t/s @38K
bash ~/.config/llama-tcq/start-max-38.sh    # MAX-CTX profile (opt-in): 409,600
                                            # ctx, YaRN 1.5625, MTP OFF —
                                            # ~28-31 t/s deep, 329K fill-proven
bash ~/.config/llama-tcq/stop.sh            # verified kill, port check
bash ~/.config/llama-tcq/status.sh          # ports + /health + VRAM + swap line
```
One profile at a time (same ports; double-start guard enforces). Slot dirs
are per-profile (slots-long/ vs slots-max/) — switching profiles never
poisons the other's slot files. Max-profile gates (2026-08-16): fill-ladder
flat (+32 MiB to 329K cached), KLD@1.5625 mean 0.0052 / p99 0.028 / top-1
95.5% (quant transparency holds at the higher YaRN), battery @64K 2-of-3
(seed-42 spiral = the known trajectory-luck mode), NIAH 5/5 @380K
(pre-validated at this scale). VRAM ~30.7 GB used, ~1.9 GB free.
Clients: `http://192.168.50.130:8130/v1` (OpenAI) or `/v1/messages`
(Anthropic), keys in `keys.json`. After a Windows reboot re-add the portproxy
if clients can't reach 8130 (WSL IP rotation).

Rollback: stop → `cp build-g1/bin/llama-server.pre-gdn22587 build-g1/bin/llama-server`
→ start (drops only the GDN #22587 kernel). Deeper: `.mainline` (also drops
24565), `.pre-bughunt`. Fused-MMA kill-switch: GGML_TURBO_MMA_FUSED=0 env
(no binary swap needed).
**Slot files are config-specific** — archive `slots-long/*.bin` on any config
change (server refuses stale ones gracefully; proxy erases and re-prefills).

## HERMES CLIENT SETTINGS (3.8-era, 2026-08-16 — supersedes 3.5 tuning)

- **Context to declare**: 320,000 (speed profile). Max profile: 400,000.
- **Compaction: trigger ~120K tokens, compact down to ~40K.** Rationale:
  exact state-tracking (ledger axis) is clean through the ~128K tier and
  spirals at the 256K tier — an agent's edit-loops NEED the state axis, so
  keep working context inside it; decode is also ~65-77 t/s at 128K vs ~40
  at 260K. Post-compaction re-prefill of a ~40K prefix ≈ 15-25 s.
- **Sampler: DO NOT change.** temp 1.0 / top-p 0.95 / top-k 20 (client
  already sends 1.0 ✓). Lower temps measured: no shallow benefit on our
  instruments, think-spirals at depth (0.6 @128K-tier, 0.4 @64K).
- **No reasoning_effort override** — template default (xhigh) won the
  ladder; "low" scored 0/2 on renders.
- **max_tokens: 49,152 recommended** (was 131,072). Bounds a runaway
  think's blast radius (spiral at 40 t/s: 131K cap = ~55 min burn, 49K =
  ~20 min) while clearing every healthy deep think we've measured (battery
  budgets 32K). Ceiling is graceful (finish_reason: length).
- **Streaming ON, per-request client timeout ≥ 20 min** (deep 32K-token
  thinks at depth take ~13 min; proxy heartbeats keep streams alive).
- **Prefix stability**: never mutate the system prompt mid-session; keep
  tool outputs append-only. Per-turn TTFT tracks the DELTA (~0.8 ms/token)
  as long as the prefix is stable — compaction is the ONE intentional
  prefix rewrite per cycle.
- **One request at a time** — the proxy serializes per-slot; parallel
  fan-out just queues.
- **Images**: ~21 s first-encode each (mmproj on CPU), then cached in
  context — batch/crop judiciously.
- **Max profile** (`start-max-38.sh`, 400K): for reference-heavy work that
  genuinely needs >280K of un-compactable material; decode ~28-31 t/s deep
  and NO MTP; recall axes (hops/needle) hold at depth but exact
  state-tracking does not — treat it as read-heavy, reason-shallow mode,
  and raise the compaction trigger to ~250K there.

## THE VRAM LAW (hard-won 2026-08-15)

Budget = weights (20.8 GB) + KV (17.5 KiB/token incl. draft) + recurrent ×(1+n_max)
+ **THREE compute scratches** (target/draft/vision — scale with ctx×batch)
+ ~250 MB meta. The stack ran silently WDDM-paged for two days because draft-KV
(f16 default!) and the triple scratch were never budgeted — `nvidia-smi` pins at
the residency cap and HIDES overcommit ("config changes don't move the number" =
you are paged). Keep ≥1 GB free. Context-fill does NOT grow VRAM (static
prealloc; checkpoints live in host RAM, ~8 KiB/token of depth, cap 2) —
**verified to 256K fill** (battery); the 294K (0.92×n_ctx) fill-ladder is
port-board P3 and MANDATORY before any ctx increase (club-3090's
"boots ≠ fills" FA-scratch-at-fill failure class).
Ceiling behavior is graceful: `finish_reason: length`, over-cap prompts → 400.

## QUALITY GATES (all in quality-tests/)

- `kl_divergence.py` — 2048-token prompts, cache_prompt=false, vs the ARCHIVED
  same-model reference `kld38/kld_logprobs_f16_qwen38_yarn.json` (a stale
  cross-model reference produces false-catastrophic numbers — 3.6-era file is
  quarantined). CURRENT gate baseline (2026-08-16, 157-prompt tail set
  `kld38/kld_logprobs_f16_qwen38_yarn_157tail.json`): mean 0.0060 / p99
  0.062 / max 0.064 / top-1 96.2% at YaRN 1.25 (max profile @1.5625:
  0.0052 / 0.028 / 0.064 / 95.5% vs its own matched ref). The old
  0.00473/98.3% is a HISTORICAL datum from a pre-b10448 binary — comparing
  against it reads a healthy run as a fake ~27% regression.
- `trajectory_battery.py` — the agentic axis. Supports `--temp` (default 0
  = greedy, the historical baseline mode). SERVING-TEMP baseline
  (2026-08-15, temp 1.0, seeds 42/43, fused+GDN binary): hops-2/3/4,
  correction, code-traj PASS at 64K/128K/256K; **ledger 8/8 clean through
  128K (envelope doubled vs the stale greedy record "spiral@128K"), full
  spiral at 256K**. Historical greedy baseline
  (`trajbase/traj_b10448-320k-baseline2.json`) kept for greedy-mode
  comparisons. Gate rule unchanged: changes must hold the perfect columns
  and not lower ledger at matched temp/seed. Temp policy: 1.0 LOCKED (see
  TEMP-STUDY.md — lower temps spiral at depth; 0.4 already at 64K).
- Depth probe: `scratchpad/depth_decode_test.py` (recreate from WORKPLAN if
  /tmp wiped). **Paired-run law:** single-arm vs historical baseline is
  invalid on this box (echo noise ±20%, environmental confounds); A/B =
  alternating binaries, minutes apart, ordering must repeat.

## PROXY (deploy/proxy.py, v6.4, 49/49 tests via `python3 test_proxy.py`)

Per-user keys → slot pinning with save/restore (now actually effective:
checkpoint fix made post-restore reuse ~49 tokens instead of full re-prefill);
think-tag extraction (literal tags after content start are preserved); stream
tripwires WITH client-facing repair (truncation → error frame + [DONE] /
error event — never a silent "finished" corpse); heartbeats both protocols;
circuit breaker (fast 503 while server restarts); passthrough allowlisted
(/slots and raw completion endpoints blocked); Anthropic images translated
to image_url. Known accepted gaps: Anthropic-path thinking-memory;
lock-acquire has no timeout (by design: two-user serialization).

## DOC MAP (authority order for "what happened / what's next")

1. `WORKPLAN-BESTAPP.md` — the marathon ledger: every verdict with numbers
   (adopted / tested-not-needed / closed), VRAM autopsy, paired-run rule.
2. `SPARSE-DECODE-BUILD.md` — the sparse-decode P0 execution ledger and
   CLOSED verdict (tested-not-viable + the fused-gate discovery);
   `SPARSE-DECODE-DESIGN.md` is its superseded design record.
3. `BUGHUNT.md` — 2026-08-15 audit ledger (proxy/scripts/C++ fixes, all shipped).
4. `RESEARCH-2026-08.md` — the research campaign board (several speed verdicts
   superseded by later paired re-tests — WORKPLAN wins on conflict).
5. `FUTUREPLAN.md` / `UPSTREAMSYNC.md` — historical (phases done; sync done).
6. `CLUB3090-PORT-BOARD.md` — the standing NEXT queue: ports/tests/skips
   from the club-3090 review, with per-item validation gates and waves.
7. `pr-package/` — three upstream-ready PR branches staged on the user's fork
   (state-restore hardening, ctx-cap rope scaling, parse-degrade). User opens
   PRs; assistant never submits to external repos.

## OPEN ARCS

- **Sparse decode: CLOSED 2026-08-15 — tested, not viable** (P0 offline
  validator: even oracle selection reads 30-50% of cache on this hybrid;
  evidence in SPARSE-DECODE-BUILD.md). Collateral fix ADOPTED: prod was
  silently running fused-MMA OFF — launcher now exports
  GGML_TURBO_MMA_FUSED=1 (in-tree default also flipped on branch
  feature/sparse-decode). Re-run the P0 probe (~30 min) before believing
  the sparse verdict for any future model swap.
- **TEMP-STUDY: CLOSED 2026-08-15 — KEEP temp 1.0** (community 0.6 claim
  real at 64K, INVERTS at 128K — 0.6 spirals; 0.4 spirals at 64K; the
  MTP-acceptance "gap" closed as entropy, +12-18 pts greedy vs 1.0, no
  anomaly). Serving-temp envelope re-baselined: ledger clean through 128K
  now (was spiral), 256K spiral. Zero config change. Full data:
  TEMP-STUDY.md.
- **NEXT: CLUB3090-PORT-BOARD.md** waves. Wave-1 progress: P5 rollback
  audit DONE-clean (triple-guarded); P2 tail-KLD three-way IN PROGRESS;
  remaining: P1 ngram+MTP depth sweep (with the fused-Q≤4 × draft-depth
  interaction) + P4 n_max=3, P6 launcher hardening, F-doc fixes. Wave 2 =
  fill-ladder P3 before any ctx push, VRAM calculator P8, agentic-turns
  probe P7.
- Post-testing-phase: context push + optional MTP-off profile (gated on P3).
- **GDN #22587: ADOPTED 2026-08-15** (un-parked, merged with b10448
  snapshot-slot semantics, all gates green — see WORKPLAN GDN section).
- MTP acceptance note (reframed by the club-3090 review): greedy inflates
  spec acceptance ~2× cross-engine; our 82.5% greedy / 67% temp-1.0 is
  expected, the "Vulkan 92%" reference is presumed greedy. Lever = tuning
  (board P1/P4), not a mystery.
- Watchlist: upstream issues 27090/27102/26609/25717 (our shapes); Vulkan
  92%-acceptance reference gap.

## OPS LAWS (unchanged, blood-signed)

USER STOP overrides goals. No pkill patterns matching your own cmdline; kill
by pidfile. No background llama-bench/perplexity. One GPU workload at a time,
babysat. Never run llama-cli non-interactively. Keys never enter the repo.
120s Bash guillotine: long jobs → run_in_background. Loads ≠ hangs (cold disk
124 MB/s). Render truth = pixels + console only. Multi-step background
scripts: ABSOLUTE paths only (relative-after-cd killed two probes) and
bounded health waits with process-death checks (never `until curl` alone).
Before EVER raising --parallel>1: run the distinct-answers gate (N identical
greedy prompts concurrently must return identical answers — hybrid
graph-reuse state-crossover class, ik#2260) and confirm MTP isn't silently
dropped; today's parallel-1 + proxy serialization is a validated design.
