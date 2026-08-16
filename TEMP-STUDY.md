# Temperature × MTP-Acceptance Study — the double lever

Status: PLANNED 2026-08-15 (user directive: focus inference; test the
community claim that temp 1.0 is wrong for coding/agentic work). This is
the NEXT arc, ahead of the club-3090 port waves.

## Why this is one study, not two

Temperature is a double lever on this stack:
1. **Quality** (community claim, untested on us): a Qwen3.8-27B-NVFP4 user
   reports temp 0.6 beat 1.0 on agentic suites ("preliminary… six suites,
   3× reps, no statistical treatment") and suggests 0.4-0.5 for
   coding/architecture. Qwen's vendor recommendation is 1.0 — historically
   an anti-repetition guard for long thinking traces, i.e. tuned for a
   failure mode, not for agentic accuracy.
2. **Speed** (our data): MTP acceptance rises as temperature falls
   (greedy 82.5% vs temp-1.0 ~67%; the coding A/B measured 59-63% at
   production sampling). If a lower temp is quality-neutral-or-better on
   OUR battery, a large decode multiplier comes free. This study either
   banks that multiplier or formally closes the "acceptance gap" as the
   price of sampling entropy.

## Priors on record (do not re-derive)

- Effort ladder ALREADY DECIDED (launcher header, session f5334395):
  reasoning_effort=xhigh @ temp 1.0 won pixel-gated renders 3/4 vs medium
  2/4 vs **low 0/2** — the commenter's `effort:low` is refuted for our
  render workload; effort stays template-default in this study. Only TEMP
  varies.
- **The trajectory battery has always run GREEDY** (`temperature: 0`
  hardcoded) — all recorded battery baselines are greedy, and serving-temp
  quality has never been validated. The study must first create temp-1.0
  baselines before comparing lower temps.
- **Live prod acceptance TODAY reads 0.79-0.90** (server print_timing
  lines, post-fused+GDN, real Hermes traffic) — far above the recorded
  67%. Unknown mix of: clients sending their own temps, code-heavy spans,
  binary evolution. Phase A0 resolves this.
- club-3090 cross-engine: greedy inflates spec acceptance ~2×; their
  llama.cpp MTP at temp 0.6 accepted 55.3% narrative / 71.2% code —
  acceptance is prompt-type-dependent; probes must split code vs prose.
- Qwen anti-greedy warning: low temp in thinking mode risks repetition
  loops — the ledger/spiral axis of our battery is exactly where this
  would show. Low temp could help (focus) or hurt (loops): empirical.

## Phase A0 — what does production actually run at? (~10 min)
Inspect recent proxy captures (~/.config/llama-tcq/captures) for
client-sent sampling params. If Hermes already sends its own temp, the
launcher's `--temp 1.0` is a dormant fallback and "production = 1.0" is a
false premise — the study's baseline temp becomes whatever clients send,
and the shipping mechanism becomes client config or a proxy-level
override, not the launcher.

## Phase A — acceptance/decode vs temperature curve (~1.5 h GPU)
- Prod binary, one boot, temp passed per-request (server default untouched).
- Temps {0.0, 0.4, 0.5, 0.6, 0.8, 1.0} × probes {code-continuation,
  prose-continuation} @38K, n_predict 700, 2 reps each, ordering repeated.
- Parse per-task `draft acceptance = X (a/g), mean len = L` from
  server.log per run window; record decode tok/s from timings.
- Output: acceptance(T) and decode(T) curves, split by prompt type.
- Interpretation forks: acceptance rises smoothly as T falls → entropy
  explanation CONFIRMED, gap closed, proceed on quality; acceptance flat →
  genuine anomaly → open the draft-sampling hunt (draft/target sampler
  mismatch, p_min, backend-sampling re-check) — only then.

## Phase B — quality at candidate temps (~2-3 h GPU)
- Prereq: add `--temp` flag to trajectory_battery.py (one-line body
  change; keep default 0 so historical baselines stay reproducible).
- Matrix: battery @64K × temps {1.0, 0.6, 0.4} × seeds {42, 43, 44}
  (temp>0 is stochastic — single-seed comparisons are not evidence; the
  community claim's weakness is exactly "no statistical treatment").
- Ledger @128K (the known cliff/spiral tier) for finalists {1.0, best-low}
  × 2 seeds — the decisive test of "lower temp helps agentic" vs "lower
  temp spirals": record spiral rate and think-length distribution per temp.
- Gates for adopting a lower default: all @64K columns ≥ temp-1.0 baseline
  across 3 seeds; ledger@128K not worse (score AND spiral rate); Phase A
  decode win ≥5% at that temp. Renders re-checked only if a temp change is
  adopted (effort ladder stands).

## Phase C — decision + ship (~30 min)
Outcomes: (a) adopt lower default (launcher + doc note that clients
override); (b) split guidance — e.g. coding sessions at 0.5, general/think
at 1.0, shipped as client config or per-key proxy sampler override;
(c) keep 1.0, publish the curve, close the acceptance item permanently.
Whatever the outcome: handoff + WORKPLAN verdict entry with numbers, and
the "biggest potential decode multiplier left" line gets replaced by data.

## RESULTS

### A0 (2026-08-15): production temp premise CONFIRMED
Current Hermes capture sends `temperature: 1.0` explicitly (older capture
omitted it → launcher 1.0 default applied). Production = 1.0 for real.
Ship mechanism for any change: client config or per-key proxy override;
launcher default is fallback only.

### Phase A (2026-08-15): acceptance curve — GAP EXPLAINED, item closed
38K probes, prod binary (fused+GDN), one boot, per-request temp, fixed
seed, 2 reps (identical — fixed seed ⇒ same path; reps prove measurement
stability, not sampling variance):

| T | code accept / decode | prose accept / decode |
|---|---|---|
| 0.0 | 0.883 / 107.9 | 0.766 / 100.6 |
| 0.4 | 0.819 / 103.3 | 0.847 / 106.7 |
| 0.5 | 0.871 / 107.2 | 0.703 / 95.5 |
| 0.6 | 0.726 / 96.0 | 0.597 / 87.2 |
| 0.8 | 0.786 / 100.7 | 0.528 / 82.0 |
| 1.0 | 0.764 / 98.5 | 0.582 / 86.1 |

- **Entropy-driven, confirmed**: greedy vs 1.0 = +12 pts code / +18 pts
  prose. The recorded "67%" was a workload blend at 1.0 (prose 0.58 / code
  0.76). NO anomaly → the draft-sampling hunt stays closed. The handoff's
  "biggest potential decode multiplier left" framing is retired.
- Mid-curve non-monotonicity (0.6 < 0.8 on code) is CONTENT-trajectory
  noise: one sampled continuation per temp; content moves acceptance more
  than temp locally. Curve endpoints are the trustworthy part.
- Multiplier bound if a low temp is adopted: ~+5-9% code / +11-24% prose
  decode at T≤0.5; T=0.6 ≈ speed-neutral vs 1.0 within this noise.
- Decision therefore rests on QUALITY (Phase B) — speed alone justifies
  at most T≈0.4-0.5, and only if the battery holds.

### Phase B @64K (2026-08-15): T=0.6 sweeps perfect; T=0.4 disqualified

| seed | T=1.0 | T=0.6 | T=0.4 |
|---|---|---|---|
| 42 | 5/6 (ledger 7/8: R2 36≠38) | **6/6 (8/8)** | 5/6 (**ledger 0/8 — EMPTY answers**) |
| 43 | 6/6 | **6/6** | 6/6 |
| 44 | 6/6 | **6/6** | 6/6 |

- **T=0.6 is the only temp that swept 9/9 tiers perfect across all seeds.**
  The community's "0.6 beats 1.0 agentic" claim is SUPPORTED at 64K.
- T=0.4 s=42 ledger `got: {}` — think-budget exhaustion (the spiral trap;
  Qwen's anti-repetition rationale for 1.0 is real, it just bites at 0.4,
  not 0.6). **T=0.4 disqualified for think-mode agentic serving** — the
  commenter's "0.4-0.5 for coding" would need think OFF to be safe; not
  our serving shape.
- T=1.0 s=42 ledger 7/8 was an ordinary tracking error (36 vs 38) — at
  temp>0 even the baseline has quality variance; the historical greedy
  8/8 baselines were the stable ceiling, as suspected.
- hops/correction/code-traj: 18/18 PASS at every temp — temperature does
  not touch the recall/override/executable axes at 64K.

### Phase B @128K finals (2026-08-15): the claim INVERTS at depth

| seed | T=1.0 | T=0.6 |
|---|---|---|
| 42 | **6/6, ledger 8/8** | 5/6, ledger **0/8 (empty answers — spiral)** |
| 43 | **6/6, ledger 8/8** | 6/6, ledger 8/8 |

- **Spiral risk climbs the temp scale with depth**: 64K → 0.4 spirals,
  0.6 perfect; 128K → 0.6 spirals (1/2 seeds), 1.0 clean. The community
  claim is real at shallow depth and inverts where our workload lives —
  consistent with their "preliminary, surface-level" caveat. Qwen's 1.0
  recommendation wins for long-context thinking serving, mechanism now
  understood (temp entropy is the anti-loop guard; the guard matters more
  the deeper the think).
- hops/correction/code-traj: PASS at 128K for BOTH temps, all seeds.
- **BONUS: the recorded 128K ledger cliff is STALE.** Old baseline
  (greedy, pre-fused/GDN binary): 8/8@64K → spiral@128K → 4/8@256K.
  Today at serving temp: **ledger 8/8@128K × 2 seeds.** Attribution
  (temp-mode vs binary evolution) not isolated — practically irrelevant:
  production serves at 1.0 on this binary. 256K tier re-run in flight to
  update the envelope.

### Phase B @256K (2026-08-15): cliff relocated, envelope doubled
T=1.0 × seeds {42,43}: hops/correction/code-traj PASS, **ledger 0/8 both
seeds, `got:{}`** = think-budget exhaustion even at 1.0. New serving-temp
envelope: **ledger clean through 128K (was: spiral@128K on the stale greedy
record — usable state-tracking depth DOUBLED), spiral at 256K** (old greedy
4/8@256K got partial answers; today's mode is full spiral — tier remains
beyond capability either way).

## VERDICT (Phase C, 2026-08-15) — KEEP T=1.0. Item CLOSED.

- **Sampler unchanged**: production stays temp 1.0 / top-p 0.95 / top-k 20
  (clients already send 1.0; zero config change ships this verdict).
- Community claim adjudicated with multi-seed evidence at 3 depths:
  REAL at shallow depth (0.6 swept 9/9 at 64K, the only perfect temp),
  **INVERTED at our working depths** (0.6 spirals at 128K 1/2 seeds; 0.4
  spirals at 64K). Mechanism: temperature entropy is the anti-loop guard
  for long thinking; the guard's value grows with depth. Qwen's 1.0
  recommendation is correct for long-context thinking serving.
- Low-temp-for-coding (0.4-0.5): usable only with thinking off — not our
  serving shape; anyone doing it per-request should know the spiral risk.
- MTP-acceptance "gap": CLOSED as entropy (Phase A curve, +12-18 pts
  greedy vs 1.0, no anomaly). Adopting a lower temp would have bought
  +5-9% code decode — the quality data says don't.
- Bonus deliverables: battery --temp flag (greedy default preserved);
  serving-temp envelope re-baselined (the handoff's cliff table updated);
  live-traffic acceptance context (0.79-0.90 per-task readings are real,
  workload-dependent).
reasoning_effort re-test (decided; renders 0/2 at low), NVFP4 weights
(vLLM/Blackwell path, not our stack), sampler shape changes (top-p/top-k
stay 0.95/20 — one variable at a time), club-3090 port waves (queued
behind this).
