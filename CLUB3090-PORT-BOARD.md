# club-3090 Port Board — what we copy, what we test, what we skip

Source: review of https://github.com/noonghunna/club-3090 (clone at
~/ai/review/club-3090), 2026-08-15. Community serving/bench corpus for
3090-class rigs on OUR model family (Qwen3.6/3.8-27B hybrid, 48 GDN + 16
attn, MTP). Rule: every adoption passes our gates (paired-run law, battery
seed-42 columns, KLD, ops laws). Their numbers are THEIR rigs/engines —
each claim gets validated on our stack before any adoption.

Board status legend: [ ] open · [T] tested · [A] adopted · [R] rejected
· [D] documented-only.

────────────────────────────────────────────────────────────────────────
## THE KEY ORIGINAL INSIGHT (read before P1)

Their best speed claim — ik_llama two-stage ngram+MTP, code decode
59.7→97.8 tok/s with a sharp inverted-U in ngram depth (2→78.8, 4→97.8,
8→90.4, 16→81.5) — interacts with OUR kernel gate in a way theirs doesn't:

Our fused MMA-turbo FA covers verify batches of **Q ≤ 4 rows** (dispatch
gate; (8,1) template instances EXIST but the gate stops at 4). Verify batch
= n_draft+1. So:
- MTP n_max=2 → batch 3 ✓ fused
- MTP n_max=3 → batch 4 ✓ fused (newly viable — the old verify-batch cliff
  was the reason for n_max=2; the fused adoption changed the calculus)
- ngram depth 4 → batch 5 ✗ → those steps fall to the MMA_F16 full-cache
  dequant path (the 41 tok/s-class path at 38K) — at depth this can eat the
  cascade's win. **This plausibly explains why OUR earlier cascade A/B
  washed (85.5 vs 83.8) while ik's gained +64%: ik's kernels have no Q≤4
  window.**
Consequences for P1's design: sweep must include depth 3 (batch 4, stays
fused), must measure BOTH shallow and 38K+, and if depth-4 wins shallow
but loses deep, the stretch move is P1b: raise the fused dispatch gate to
Q≤8 (instances already compiled; gate + validation only).

────────────────────────────────────────────────────────────────────────
## PORTS (claim → port → validation test → gate)

### P1 [T] SWEPT 2026-08-15 — depth 2 is OUR optimum; verdict pending gate.
RESULT (temp 1.0, code+prose @38K, arms restarted, base repeated clean):
base 95.9-96.5 code / 84.1-84.4 prose (accept .760/.582, rep-stable).
casc2: cold ≈ base; WARM (ngram index has seen the text): code 108.5
(+12%, accept .846), prose **132.3 (+57%, accept .981)**. Never worse than
base in any cell; combined draft ≤3 → verify batch ≤4 → STAYS FUSED.
casc3/casc4: code warm +5-10%, but prose REGRESSES (76.1/73.5, accept
.48/.45 — wasted deep drafts + batches spill past the fused Q≤4 window).
**The board's kernel-window prediction confirmed: their depth-4 peak does
not transfer; ours is 2.** Their +64% decoded: their bench measures the
WARM regime (5 repeats after warmups) — real for agentic loops (files
re-read, edit retries) but the honest claim is "up to +12-57% warm,
neutral cold." Gate in flight: battery @64K on casc2 + P4 n_max=3 arm.

### (original P1 spec)
- Their claim: chained ngram(n=4)+MTP(n=3) = +64% code decode, lossless
  quality; inverted-U in ngram depth.
- Port: likely config-only — pre-checked 2026-08-15: `--spec-ngram-mod-n-min`
  exists and speculative.cpp respects per-drafter `dp.n_max` in the ngram
  path (:349) → step 0 = find the exact per-stage n_max syntax; tiny patch
  only if truly absent.
- Test: paired A/B per our law (restarts between arms, ordering repeated):
  arms = MTP-only(prod) vs cascade depth {2,3,4}; probes = code-continuation
  AND prose-continuation, shallow + 38K, **temp 1.0 primary** (their
  greedy-inflation lesson: greedy acceptance ≈2× inflated — one greedy pass
  only as a comparison datum); record acceptance AND fire-rate telemetry
  where logs allow. Battery @64K on any winner.
- Gate: ≥ +10% code decode at temp 1.0 without >2% prose loss, battery
  columns hold, no VRAM delta (ngram drafts are model-free — expect none).
- Effort: 0.5-1 day incl. P1b if triggered. NOTE: our previous "wash"
  verdict used default depth — this supersedes it only if a sweep was
  actually absent (verify in RESEARCH-2026-08 first).

### P2 [T→A] DONE 2026-08-15 — tail metric shipped; five-type ladder run.
RESULT (157 prompts, fresh f16 ref archived as
kld38/kld_logprobs_f16_qwen38_yarn_157tail.json; summaries kld38/*_tail):
q8_0 mean .0019 p99 .056 max .089 top1 99.4% | **turbo4 .0060 / .062 /
.064 / 96.2%** | q4_0 .0156 / .069 / **1.708** / 96.2% | turbo3_tcq .0154 /
**.179 / .185** / 93.6% | turbo3 .0182 / .124 / .143 / 88.5%.
VERDICTS: (1) turbo4 extreme tail = q8_0-class (max even lower) → the
critique DOES NOT transfer to turbo4; community floor matched at half the
bytes. (2) q4_0 = the mean-hides-tail cautionary tale (catastrophic 1.71
single-position blowup). (3) turbo3_tcq independently CONFIRMS the
critique's shape (better mean than turbo3, fattest extreme tail). (4)
turbo3 is NOT lossless-class (88.5% top-1). Tail percentiles are now a
standing gate metric in kl_divergence.py.

### (original P2 spec, for the record)
- Their claim (sharpest external critique of our tech family): TQ-class KV
  quality claims are mean-level, not tail-level — Anbeeld's data puts
  turbo3_tcq at ~82% precision on the worst 0.1% of positions (JSON keys,
  braces, tool tokens); their policy floor is q8_0 KV ("sub-q8 never
  depth-validated").
- Port: extend quality-tests/kl_divergence.py to report per-position KLD
  percentiles (p99 / p99.9 / max) + where top-1 flips concentrate.
- Test: three arms on OUR server, config-only: turbo4 (prod) vs q8_0 KV vs
  q4_0 KV — mean AND tail, same archived f16 reference, same prompts.
- Gate/read: if turbo4 tail ≈ q8_0 tail at half the bytes → their floor
  policy is beaten with data, and the turbo3_tcq critique measurably does
  not transfer to turbo4. If turbo4 tail is bad → feeds bit-allocation
  discussion (K-heavy splits). Either result is valuable. Record tail
  baseline as a standing gate metric.
- Effort: ~1-2 h total.

### P3 [ ] Fill-ladder probe — PREREQUISITE for the context push
- Their claim ("boots ≠ fills", most transferable ops finding): FA
  transient scratch grows with FILL (~7.9 MB/1K tok their config); a 262K
  alloc filled only to ~125K before OOM; fixed-depth probes give false
  all-clears → the probe must scale to ~0.92 × n_ctx.
- Port: quality-tests/fill_ladder.py — grow ONE slot via cache_prompt
  continuation in 16K rungs to 0.92×n_ctx (294K today); per rung: VRAM,
  /health, 20-token decode sanity.
- Test: run at current 320K prod config.
- Gate/read: VRAM static to 294K → our "ctx-fill VRAM-static" claim
  (currently proven to 256K via battery) extends to spec; if growth
  appears, measure OUR MB/1K-tok rate (fused nstages=0 path has a
  different scratch profile than their mainline). MANDATORY before any
  ctx>320K change; the ladder becomes the ctx-push acceptance test.
- Effort: ~1 h build+run.

### P4 [T] MEASURED 2026-08-15 — n3 ungated: code 112.7/113.3 (+17% vs
base 96!), prose 79.3 (−6%), accept .75/.41, no clamp (head supports ≥3),
batch 4 stays fused. The cross-rig shape reproduces. Decision folded into
P9 (p-min gate may keep the code win and erase the prose tax). Original:
### MTP n_max=3 re-test (fused changed the calculus)
- Their data: vLLM n=3 AL 3.3-4.0; llama.cpp-family sweeps show shallow
  optima (n=2 on one model, n=5 knee on another) — model-specific.
- Ours: n_max=2 was chosen when batch>2 fell off the fast path. Post-fused,
  batch 4 stays fused (see insight above).
- Step 0: speculative.cpp clamps MTP n_max to the head's TRAINED block size
  (:973-976, logs a clamp warning) — if our nextn head is block-size 2,
  n_max=3 silently clamps and P4 is moot; check the server log first.
- Test: paired A/B n_max 2 vs 3, temp 1.0, both probes, shallow+38K; VRAM
  check first (+1 recurrent-state copy ≈ ~100 MB — fits ~900 MB headroom).
- Gate: net decode win on the code probe without prose loss; battery @64K.
- Effort: ~1 h. Fold into P1's session (shares harness).

### P5 [T] DONE 2026-08-15 — CLEAN (triple-guarded: bounded rollback + explicit false-return contract, set_rs_idx clamp, kernel slot bounds). Original spec:
### Rollback-clamp audit (vllm#50021 pattern insurance)
- Their claim: vLLM's MTP×GDN spec path indexes state blocks by an
  UNBOUNDED accepted-token count → wild write → Xid 13/31 dead worker on
  5090-class cards; depth-independent, probabilistic.
- Ours: kernel side bounds-checked (verified during #22587 merge:
  `target_slot >= 0 && < K`). Audit the CONSUMER side — wherever
  llama-context/speculative picks a snapshot slot from n_accepted, verify
  clamping; add an assert if absent.
- Test: code audit + existing soak/money-test pass on prod binary.
- Effort: ~30 min. Cheap insurance on a crash class hitting our GPU class
  in the wild.

### P6 [ ] Launcher hardening: VRAM-settle + early-crash detect + swap line
- Their patterns: (a) poll until used-VRAM stops falling before next boot
  (compose down returns before CUDA releases — same class as our WSL
  transient restart OOMs); (b) health-wait loop that detects server death
  immediately (dump last 30 log lines, exit fast) instead of burning the
  timeout; (c) swap check — serving-process pages in swap invalidate all
  perf numbers (our "swap death" history!).
- Port: start-long-38.sh gets vram-settle before launch + PID-death check
  in the health wait; status.sh gets a `VmSwap` line for the server PID.
- Test: 5× rapid stop/start cycles all boot clean; kill server mid-wait →
  script exits fast with logs.
- Effort: ~45 min.

### P7 [ ] Agentic-turns speed probe (their fixture idea, our gap)
- Their bench-agentic: 15 turns of real tool-results ramping to ~53K,
  per-turn TTFT + decode, growth anchored to turn 2 (turn 1 = cold-start),
  verdict bands (≤1.5× stable … O(n)-like).
- Our gap: depth_ab is single-shot; nothing continuously measures the
  TURN-LOOP shape — per-turn TTFT is exactly what our checkpoint/restore
  work improves and we have no standing probe for it.
- Port: quality-tests/agentic_turns_probe.py with a captured fixture
  (their turn-2 anchoring + bands copied).
- Test/gate: baseline the curve on prod; standing regression gate for any
  cache/checkpoint/proxy change.
- Effort: ~2 h.

### P8 [ ] VRAM-law calculator (fold into context push)
- Their kv-calc: calibrated predictor, PASS/TIGHT/FAIL verdicts, ±1.5 GB
  band — but their own docs: it CANNOT model llama.cpp (no elastic pool;
  "measured boot ladder only").
- Port: small tools/vram_law.py encoding OUR law (weights + KV/token incl
  draft + recurrent×(1+n_max) + 3 scratches + meta) with our measured rows
  as calibration; used to plan ctx targets + the MTP-off max-ctx profile.
  P3's ladder data feeds it.
- Effort: ~1-2 h, wave 2.

### P9 [ ] p-min confidence gate sweep (source: sudoingX/qwen38-mtp review)
- Their claim (cross-rig, RX9070 sweep is the clean one): `--spec-draft-p-min
  0.60` makes deep drafts "nearly free" — gated n-max 4 beat ungated n-max 2
  (+3% mixed, **+21% copy-heavy**, acceptance 0.86 vs 0.73, ~3.3 tok/round);
  code climbs with depth on EVERY rig (A6000: 84.3 tok/s at n-max 6!), prose
  pays ungated, the gate rescues it. We have NEVER swept p-min.
- Test: arms {n2 (prod), n2+pmin.6, n3+pmin.6, n4+pmin.6} × probes {code,
  prose, NEW copy-heavy probe (rename-and-echo over a 4K file ≈ agentic
  edit-loop shape)} @38K temp 1.0, reps+ordering per our law. Interacts with
  the fused Q≤4 window: gated deep drafts have VARIABLE batch — steps >4
  fall off the fused path; measure, don't assume.
- Gate: overall ≥ prod-n2 with prose ≥ −2%; battery on winner.

### P10 [ ] Cascade/spec determinism spot-check (pre-adoption insurance)
- One qwen38-mtp host's greedy code-completion hash gate flagged
  "chaining ngram-mod made n-max 2 unstable" → ships without ngram. Spec
  decode should be distribution-preserving; before adopting ANY cascade/
  p-min config: greedy same-prompt ×3 must produce IDENTICAL output
  (cascade on vs off), plus battery. Cheap; catches implementation-level
  divergence their gate may have seen.

### B1 [D] Blog review (veladan.org Qwen3.8 FP8 benchmarks) — recorded intel
- Their HermesAgent-20: 91 @temp0.6 → 79 @1.0 (−12) on SHALLOW agent work —
  reconciles with our TEMP-STUDY (0.6 wins shallow, spirals at 128K; their
  suites never go deep; their 0.6 pass was n=1). OPEN IDEA: depth-aware
  per-request temp override in proxy (0.6 while ctx < ~96K, 1.0 beyond) —
  needs a shallow tool-call-shaped suite on OUR stack first (our battery
  under-weights that axis).
- YaRN tax warning for the ctx push: at YaRN 4.0 their BugFind dropped −12;
  "'slightly impact short-context quality'… undersells it." Any YaRN
  increase (409K profile) must re-gate KLD+battery vs the 1.25 baseline.
- NVFP4 ≈ FP8 within noise on Blackwell/vLLM (weights axis, background).
- Their gen numbers: 3.6→3.8 HermesAgent +29pts — matches our pilot.

### U1 [ ] USER-SIDE (Windows) — two .wslconfig/registry changes
- `networkingMode=mirrored` in .wslconfig (Win11 22H2+): permanently ends
  the portproxy-after-reboot chore (their WSL doc; matches our handoff
  annoyance).
- Registry TdrDelay=60: Windows force-resets the GPU after 2 s kernels
  (their 156K-prompt repro). Our llama.cpp kernels are short (we prefill
  315K fine) so this is prophylactic — but it removes a whole class of
  "transient device not ready".
- Validation: after reboot — clients reach 8130 with no portproxy; note in
  handoff which mode is active.

────────────────────────────────────────────────────────────────────────
## DOC FIXES FROM THE REVIEW (do with wave 1)

- F1 [ ] Handoff watchlist: reframe "Vulkan 92% acceptance gap" — their
  cross-engine data shows greedy inflates spec acceptance ~2× (0.466
  canonical vs 0.96 greedy); the 92% reference is almost certainly greedy.
  Our 82.5% greedy / 67% temp-1.0 is expected physics, not a bug. The
  remaining lever is tuning (P1/P4), not a mystery hunt.
- F2 [ ] Handoff VRAM law: scope "ctx-fill VRAM-static" as "verified to
  256K fill; 294K ladder = P3" until P3 runs.
- F3 [ ] Ops laws: add "before ever raising --parallel>1: run the
  distinct-answers gate (N identical greedy prompts concurrently must give
  identical answers) — ik#2260 class: hybrid graph-reuse can cross
  recurrent state between slots; also mainline may silently drop MTP at
  np>1." We are immune today (parallel 1 + proxy serialization — a design
  their np=4 aider 1/30 measurement independently validates).

────────────────────────────────────────────────────────────────────────
## SKIPS (with reasons, so nobody re-litigates)

- DFlash-family external drafts: greedy-only (DDTree silently off at
  temp>0 → ~20 tok/s), beellama path retired upstream, gibberish on newer
  arches, drafter-pairing fragility (0.25% acceptance on a mispaired
  target). Dead on arrival for a temp-1.0 serving stack.
- LMCache / Genesis patches / kv-calc wholesale / Marlin W4A8: vLLM-side.
  Our checkpoint cluster already beats LMCache's warm path (their own
  breakdown: warm-load is 4.8 s GDN-state RECOMPUTE-bound — we restore
  state directly).
- Multi-GPU tooling (UUID pinning, placement asserts, -ts splits, TP
  advice), c3 TUI, patch drift-guard framework: single-card single-config
  reality; our status.sh + docs cover the need.
- IQ4_KS/IQ5_KS weight quants: only relevant to a future "max-context over
  quality" profile; we deliberately run Q6_K. Park.
- Power-cap tuning: we run stock; their datum (GDN compute-bound → −34-42%
  at 230 W on llama.cpp) recorded as a DON'T (never undervolt without a
  paired A/B).
- Concurrency >1: their N=2 aggregate datum (+19%) noted, but np>1 drops
  MTP + state-crossover risk → stays closed until upstream fixes; proxy
  serialization stands.

────────────────────────────────────────────────────────────────────────
## VALIDATION OF OUR STACK (no action — evidence bank)

- vLLM-land fights our solved wars: SSM state not prefix-cacheable (TTFT
  36× by turn 10 despite cache hits) vs our 49-token restore-reuse; GDN
  prefill OOM cliffs vs our streaming path; MTP×prefix-cache recurrent
  corruption (vllm#43559) vs our battery-verified restore; np>1 crossover
  vs our serialization.
- Their q8_0 KV floor ("sub-q8 never depth-validated") vs our
  battery-validated 4.125 bpv to 256K: we exceed their bar at ~half the
  bytes — P2 adds the tail data to make it airtight.
- TurboQuant landed in upstream vLLM WITHOUT hybrid support; Genesis P67
  may upstream. Our niche (hybrid + KV-quant + MTP, single card, depth-
  validated) remains unique. Watch both.
- Their measured Qwen3.8 ladder (24 GB: 131K ✓ / 196K ✗ compute / 262K ✗
  rs-cache) is consistent with our law at 32.6 GB → 320K.

────────────────────────────────────────────────────────────────────────
## EXECUTION WAVES

- Wave 1 (one session): P1 (+P4 same harness, +P1b if triggered) · P2 ·
  P5 · P6 · F1-F3. Expected outcome: possible double-digit code-decode
  win, tail-KLD baseline + three-way comparison, hardened launcher.
- Wave 2 (context-push prerequisites): P3 → P8 → ctx targets + MTP-off
  profile; P7 agentic-turns baseline.
- Wave 3 (opportunistic): soak-gate upgrades, needle-pattern diversity,
  IQ-quant max-ctx profile if ever wanted.
