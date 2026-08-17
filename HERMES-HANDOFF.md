# TurboQuant Serving Stack — Handoff (2026-08-16 LATE — post GPU-lost fix ship)

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

- **Model roster (2026-08-16 late):** TWO weight sets, ONE serving at a time,
  BOTH advertised under the FIXED ids qwen3.8-27b-320k/409k (Hermes breaks on
  any other name — law in repo CLAUDE.md). (a) Original Qwen3.8-27B Q6_K
  (hybrid: 48 DeltaNet + 16 attention layers, head_dim 256, GQA-4, native MTP
  head) via start-long-38.sh / start-max-38.sh. (b) TRIAL:
  JonathanColetti/Qwen3.8-27B-Uncensored Q6_K (same arch, with-MTP variant,
  models/qwen38-uncensored/) via start-long-38u.sh — quality UNGATED (no
  KLD/battery run on it), user-requested serving choice. CURRENTLY SERVING:
  the uncensored trial. Which weights answer = which launcher ran; check
  status.sh/log, never the API label. Vision via ORIGINAL mmproj **on CPU**
  (~21-60 s/image encode, once per image; text speed unaffected) — all three
  launchers use the gate-validated original mmproj.
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
- **Fork:** prod source = branch `fix/vision-hybrid` @ 72fa4ca4e (on top of
  `sync/2026-08`, which sits 4 commits behind upstream master — verified
  2026-08-16, nothing relevant missing). Binary: `build-g1/bin/llama-server`
  = fix/vision-hybrid build (gdn22587 + GPU-lost fix trio: recurrent mrope
  handling, GDN kernel state-write bounds, MTP draft auto-resync; gates all
  green incl. byte-identical text path). Rollback chain: `.pre-visionfix` →
  `.pre-gdn22587` → `.mainline` → `.pre-bughunt`. Fused
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
~/.config/llama-tcq/start-long-38.sh   # SPEED profile (default): 320K,
                                       # MTP n3 — code ~113 t/s @38K, ORIGINAL weights
~/.config/llama-tcq/start-max-38.sh    # MAX-CTX profile (opt-in): 409,600
                                       # ctx, YaRN 1.5625, MTP OFF —
                                       # ~28-31 t/s deep, 329K fill-proven
~/.config/llama-tcq/start-long-38u.sh  # TRIAL: uncensored weights, SPEED params,
                                       # own slot dir (slots-long-unc/), SAME
                                       # advertised id qwen3.8-27b-320k
~/.config/llama-tcq/stop.sh            # the ONLY sanctioned stop
~/.config/llama-tcq/status.sh          # ports + /health + VRAM + swap line
```
PROCESS LIFECYCLE LAW (2026-08-16, double outage): launchers use
`setsid nohup ... </dev/null` = fully detached — nothing kills the stack
except stop.sh, closing the LAST WSL window (that ends WSL itself on this
box), `wsl --shutdown`, or reboot. USER'S CHOICE: no auto-start on boot —
after any reboot the stack is DOWN until a launcher runs (first load
cold-disk slow = minutes, not hung). If you ever touch launch lines:
verify PGID=SID=PID and TT=? on the server pid before claiming detachment.
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

Rollback: stop → `cp build-g1/bin/llama-server.pre-visionfix build-g1/bin/llama-server`
→ start (drops the GPU-lost fix trio — ONLY do this with vision re-disabled!).
Deeper: `.pre-gdn22587` (also drops the row-per-warp kernel), `.mainline`,
`.pre-bughunt`. Fused-MMA kill-switch: GGML_TURBO_MMA_FUSED=0 env
(no binary swap needed).
**Slot files are config-specific** — archive `slots-long/*.bin` on any config
change (server refuses stale ones gracefully; proxy erases and re-prefills).

## HERMES CLIENT SETTINGS (3.8-era, 2026-08-16 — supersedes 3.5 tuning)

- **Provider**: `5090PC`, OpenAI-compatible, base `http://192.168.50.130:8130/v1`,
  key from keys.json. **Two model ids** (served one-at-a-time, launcher
  `--alias`): `qwen3.8-27b-320k` (speed profile) · `qwen3.8-27b-409k`
  (max-ctx profile). Wrong-id requests still serve the running model.
- **ctx-checkpoints stays 2** (tested 2026-08-16: 2 vs 4 token-identical
  reprocess at 25/50/75% edit depths — 11586/11876/13432 both — while 4
  costs +700 MB host RSS; the cap is NOT a stale limitation).

- **Context to declare**: 320,000 (speed profile). Max profile: 400,000.
- **Compaction trigger — user's choice on a measured spectrum, compact
  down to ~40K whenever it fires:**
  - **~250K trigger (max-window mode)**: recall, correction, and executable
    code-trajectories are measured CLEAN through the 256K tier — only exact
    state-tracking (ledger) degrades past ~128K, and decode slows toward
    ~40 t/s. Maximum usable window.
  - **~120K trigger (all-axes-clean mode)**: every measured quality axis
    including exact state-tracking stays perfect, decode stays 65-77 t/s.
  Pick per workload; nothing breaks either way. Post-compaction re-prefill
  of a ~40K prefix ≈ 15-25 s.
- **Sampler: DO NOT change.** temp 1.0 / top-p 0.95 / top-k 20 (client
  already sends 1.0 ✓). Lower temps measured: no shallow benefit on our
  instruments, think-spirals at depth (0.6 @128K-tier, 0.4 @64K).
- **No reasoning_effort override** — template default (xhigh) won the
  ladder; "low" scored 0/2 on renders.
- **max_tokens: MAXED OUT — 131,072 (or omit the cap entirely; the only
  real bound is remaining context).** USER DIRECTIVE 2026-08-16: never cap
  output capability. Ceiling is graceful (finish_reason: length). If a
  depth-spiral ever burns a long generation, kill the request client-side —
  do NOT cap the model.
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
- **Max output tokens: MAXED (131,072 / uncapped) — think-inclusive** on
  this stack (preserve_thinking streams reasoning as output). Server sets
  no cap; the ceiling is remaining context and it degrades gracefully.
  Per user directive: output is never capped on this rig.
- **Broom / pruning rules (prefix-break physics)**: kept old tool outputs
  cost NOTHING per turn (already prefilled; TTFT tracks only the delta) —
  but retroactively trimming/deleting an old message REWRITES THE PREFIX at
  that point → full re-prefill of everything after it (~40-60 s mid-depth).
  Therefore: (1) insertion-time truncation is purely a context-BUDGET
  choice, not a stability need — single inserts to at least ~120K tokens
  are proven on this stack; cap tool dumps only if you want to stretch the
  window, never for safety;
  (2) **rolling retro-trim NO** — an every-turn broom converts 0.8 ms/token
  delta-TTFT into a fresh-prefill per turn, the single worst client setting
  possible here; (3) **batch all cleanup into the compaction event** — one
  intentional prefix rewrite per cycle at the ~120K trigger. A full-clear
  broom (new session) is always fine.
- **cache_prompt must stay enabled** (server-side prefix reuse is the
  foundation of all of the above; verified live — do not send
  cache_prompt:false).
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

## RESUME HERE (2026-08-16 LATE — post GPU-lost fix, second handoff point)

**WHAT WE DID (2026-08-16, chronological):**
1. Morning arcs (see prior sections): consolidation, README showcase,
   mega-bughunt, two-profile ship, n3 adoption, ctx-checkpoints keep-2,
   auto-tier design, QUICK yarn-tax suite (identity gate PASSED; shallow
   unification increment ≈0.5% top-1; deep divergence needs the overnight
   battery as arbiter — data in quality-tests/yarntax/, summary in
   AUTO-TIER-DESIGN.md).
2. Process-lifecycle hardening after a DOUBLE outage: plain nohup died with
   the launching session → all launchers now `setsid nohup </dev/null`
   (verified live-fire). Discovered: closing the LAST WSL window terminates
   WSL itself on this box (VM teardown, graceful SIGTERM) — accepted
   semantics, no auto-start by user's explicit choice. User closes windows
   to reclaim VRAM (terminals cost GPU memory) — that's WHY windows close.
3. Fixed advertised-id LAW: ONLY qwen3.8-27b-320k / qwen3.8-27b-409k.
   Hermes = NousResearch/hermes-agent; its quirks we solved: (a) it sent
   reasoning_effort "max" (Kimi value) → 500s; user's agent scoped qwen ids
   to xhigh (model's legal ladder: xhigh/medium/low). (b) images route via
   its vision_analyze tool unless the provider's model entries are marked
   vision-capable → told user's agent to mark them; images then flow inline.
4. Uncensored trial: downloaded JonathanColetti Q6_K with-MTP (+ its vision
   file, unused — we use original mmproj), built start-long-38u.sh, serving
   under the canonical 320k id. Quality ungated; user's call.
5. Live MTP report on real Hermes traffic: 93K-token pagoda turn at 97 t/s,
   70% acceptance (3.10 tok/verify-pass); tool-loop phase 61%; verdict KEEP.
   Session monitoring: 56 turns clean, ctx-checkpoints restored 53× (feature
   validated in prod), Hermes compaction at ~120K→40K worked as configured;
   known cost = ~60-75s full re-prefill when Hermes rewrites history after
   big thinks (thinking-drop is vendor-CORRECT for Qwen; do NOT "fix").
6. Inference research blast → INFERENCE-RESEARCH-2026-08.md: T1 FP4/NVFP4
   weights on Blackwell (~7GB VRAM free + speed; gate decides), T2 upstream
   sync (we're 4 commits behind — current!), T3 per-layer mixed KV (q8_0 on
   the 16 attn layers), T4 WSL ops wins (Defender vhdx exclusion for the
   124MB/s cold loads, HAGS A/B), T5 grammar-turn spec toggle. EAGLE-3:
   skip — our MTP already beats the 3.6 head's tau (3.1 vs 2.4).
7. **THE BIG ONE — GPU-lost bug, root-caused → fixed → gated → SHIPPED.**
   Two incidents (GPU off the PCIe bus, reboot required) whenever a photo +
   stop/resume hit the stack. Root cause: Qwen-VL mrope image chunks
   (N tokens over max(t,h,w) positions) meet llama-memory-recurrent's
   explicit "special-casing isn't done" warn-and-proceed → DeltaNet
   bookkeeping corrupts → new GDN row-per-warp kernel writes OOB → device
   lost. Upstream master HAS THE SAME DEFECT (verified). Fix trio on branch
   fix/vision-hybrid @72fa4ca4e (pushed to myfork): recurrent mrope
   handling (forward-legal/regression-rejects), kernel state-write hard
   bounds (fused + non-fused, extent from view root), MTP draft auto-resync
   (also fixes draft-never-cleared-on-reset). Gates ALL GREEN: op-tests
   36/36, CPU killer-sequence repro, GPU killer-sequence repro
   (crash-identical fingerprint survived), text path byte-identical, live
   prod image smoke. PROMOTED to prod binary; vision RESTORED in all three
   launchers (original mmproj); rollback .pre-visionfix. Full story:
   GPULOST-BUGREPORT.md + crash-forensics/.

**WHAT IS RUNNING NOW:** uncensored trial weights + vision + fix binary,
SPEED profile params, advertised as qwen3.8-27b-320k, healthy. Hermes may
send images (first image per convo ~30-60s CPU encode).

**WHAT WE WILL DO (priority order):**
1. OVERNIGHT yarn A'+B' battery — ON USER'S WORD ("run the extended
   battery"). Spec in AUTO-TIER-DESIGN.md + session notes: A' depth-station
   KLD (stations 2K/32K/128K/200K/256K; llama-perplexity --kl-divergence for
   shallow full-vocab; yarn_stations.py branch probes deep; --parallel 1
   MANDATORY; no-yarn arm ≤262144) + B' matched-config battery (ctx=262144,
   scales {none,1.25,1.5625} × tiers {64K,128K} × 3 seeds + 256K finalists).
   Delivers the rope-unification verdict.
2. Rope verdict → auto-tier proxy build per AUTO-TIER-DESIGN.md (~1 day:
   unified id, TIER_UP_TOKENS≈180K, hold-and-swap 20-50s if unified at
   1.5625, else ~2-min re-prefill variant).
3. Vision perf re-baseline on the fix build (old 21s/img predates it) before
   quoting numbers to the user.
4. Research board tiers in order (T4 ops wins are cheap; T2 sync is 4
   commits; T1 FP4 gate ladder is the big swing; T3 per-layer KV; T5
   grammar toggle).
5. pr-package: the vision-hybrid fix is an upstream-PR candidate — USER
   OPENS PRs to external repos, never the agent (hard boundary).
6. Uncensored trial: if it becomes permanent, run the quality gates on it
   (tail-KLD 157 + battery) — currently ungated.

**Standing doctrine:** QUALITY AND FEATURES OVER SPEED · never cap output ·
only 2 advertised model ids · no upstream folklore without on-stack A/B ·
options-before-execution on NEW test arcs · TLDR replies · state the plan
BEFORE acting (2026-08-16 lesson) · claim-scoping: name what a test covered
AND what it didn't · verify every claim with a command first.

## OPEN ARCS

- **GPU-lost / vision-on-hybrid: CLOSED+SHIPPED 2026-08-16** (fix trio gated
  and promoted; see RESUME HERE item 7 + GPULOST-BUGREPORT.md). Residual:
  vision perf re-baseline pending; upstream PR candidate (user opens).
- **Inference research board: OPEN** — INFERENCE-RESEARCH-2026-08.md, five
  tiers, FP4 weights = top strategic lever (quality gate decides).
- **Yarn A'+B' overnight battery: ARMED, waiting on user's word** → rope
  unification verdict → auto-tier build (AUTO-TIER-DESIGN.md).
- **Uncensored trial weights: SERVING, quality UNGATED** — gate before
  calling permanent.
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
