# Auto-Tier Design — one Hermes model, silent 320K→409K profile handoff

Status: DESIGN (2026-08-16). User ask: Hermes sees ONE model; we serve the
320K speed profile and silently switch to the 409K max profile when a
session crosses ~170K tokens — seamless except an acceptable switch delay.

## Hardware feasibility (5090 32GB / 48GB RAM) — YES

- Only ONE server can hold weights at a time (2×21.3 GB > 32 GB → no warm
  standby process). Irrelevant in practice: with 48 GB RAM the 22.9 GB
  model file stays fully page-cached → measured warm full-stack restart
  ≈ 7-40 s. A standby process would save only seconds.
- **The dominant switch cost is NOT the model reload — it is re-prefilling
  the session** into the new server: ~1.7K tok/s at depth → 170K ≈ 100-115 s,
  200K ≈ 2 min, 300K ≈ 3.2 min. (Why unavoidable today: the profiles use
  different YaRN scales, and rope is baked into cached K — the old cache is
  mathematically wrong for the new scale; slot files are also
  config-validated. See Phase 2/3.)
- Total switch experienced by the client: ~2-2.5 min inside ONE held
  streaming turn, with keepalive heartbeats flowing. Matches the user's
  "seamless except for maybe a delay".

## Architecture: proxy-orchestrated auto-tiering (Phase 1)

The proxy already sees every request, owns slot save/restore, has a
circuit breaker for restart windows, and heartbeats both protocols — it is
the natural orchestrator. Additions:

1. **Unified model id**: both launchers started with `ALIAS=qwen3.8-27b`
   (env knob already exists). Hermes gets ONE model entry. (The per-profile
   ids remain available for manual mode.)
2. **Exact token tracking, free**: the previous response's `usage`
   (prompt+completion totals) is stored per user/slot; incoming delta
   estimated at chars/3 + 8K margin. No /tokenize calls needed.
3. **Trigger**: projected total > `TIER_UP_TOKENS` (default **180K**,
   env-tunable; user floated 170K — anything in 150-250K is sane: past the
   state-tracking envelope both profiles are quality-equal, and it leaves
   ≥70K headroom before the 320K ceiling could hard-stop a turn).
4. **The switch (hold-and-swap)**: hold the triggering request; drain any
   in-flight stream; `stop.sh`; `start-max-38.sh` (health-gated); forward
   the held request (first turn pays the re-prefill); keepalive frames the
   whole time. Failure → boot speed profile back + proper SSE error frame
   (never a silent dead stream).
5. **Tier-down**: after `TIER_DOWN_IDLE_MIN` (default 15) with no traffic,
   revert to the speed profile. A returning big session simply re-triggers
   tier-up (its 409K slot file restores — slot files are per-profile dirs).
6. Manual escape hatches unchanged: the two launchers + per-profile ids.

## Alternatives ruled out (with numbers)

- **One config that does both (409K + MTP)**: does not fit — vram_law says
  −858 MiB at the 700-floor; closing it needs ub-cuts + weight-quant drops
  (quality-first veto).
- **Always-409K**: user-rejected; costs 113→30-80 t/s on all daily work.
- **Warm standby process**: pointless (page cache already provides it);
  impossible to hold both on GPU anyway.

## Phase 2 (optional, later): fast handoff via matched YaRN

Run BOTH profiles at YaRN 1.5625 (quality at 1.5625 measured
equivalent-class: KLD 0.0052 / p99 0.028 / top-1 95.5% vs matched ref;
NIAH 5/5 @380K) → cached-K semantics match → the only remaining barrier is
server-side config validation of slot files (ctx size / spec flags). Relax
that check in OUR fork for the blessed pair → switch becomes save-state
(~2.8 GB) + restart + restore ≈ **30-60 s, no re-prefill**. Deferred until
Phase 1 experience says the 2-min switch matters. Trade to re-gate: speed
profile would carry the higher rope scale at short context (the yarn-tax
axis — needs its own battery pass before adoption).

## Phase 3 (research): re-rope the cache across scales

The fork already has a rope-shift rebuild path (dequant → un-rotate →
re-rope → re-quant, in-place) for position shifts. A SCALE-change variant
(rotate each cached K by the Δangle between yarn scales) is mathematically
straightforward and would make cross-scale handoff exact without Phase 2's
unification. Real kernel work; only if auto-tiering becomes central.

## Effort (Phase 1)

Proxy patch ~150-250 lines + ~10 tests (hold-and-swap, trigger math,
tier-down, failure paths), launcher alias unification via env, docs, and a
live drill: synthetic session crossing the threshold → verify switch,
continuity, and timing. ≈ half a day with gates.

## Search-blast findings (3 agents, 2026-08-16) — DESIGN REVISED UP

**1. KV-state agent (the game-changer).** Verified in OUR code: slot files
validate KV type / v_trans / layer count / capacity — but NOT n_ctx (larger
-c restore explicitly works), NOT --spec-type (draft ctx is separate; cold
draft after restore = warm-up only), and NOT rope/YaRN (no field exists —
cross-scale restore silently corrupts; HF's identical bug measured PPL 10.2
vs 4.3). **The ONLY barrier between our two profiles is --rope-scale 1.25
vs 1.5625.** Unify both at 1.5625 → slot files become PORTABLE long→max
unconditionally (max→long when ≤320K). NO fork surgery needed. The
mmproj-blocks-slot-save gate exists only in the old public tree — prod
builds from turboquant-sync where it's removed (and proxy slot save runs
daily in prod). Checkpoints persist in slot files in our tree (#26004 pick).
→ **Phase 2 collapses into Phase 1: switch = save slot (~2.8 GB, seconds) +
restart (~10-40 s warm) + restore + ~49-token reuse ≈ 20-50 s total, NO
re-prefill.** Plain re-prefill remains the automatic fallback.
Quality gate required first — and CORRECTED 2026-08-16: our existing KLD
runs compared each scale against a SAME-SCALE f16 reference, which isolates
QUANT error and cancels the yarn tax out entirely. "0.0052 at 1.5625" means
quant stays transparent there — it says NOTHING about what yarn itself
costs. The yarn tax (none vs 1.25 vs 1.5625) has never been measured on
this stack; YARN-TAX-STUDY (menu presented to user) must run before rope
unification is adopted.

**2. Tooling agent.** Nothing off-the-shelf routes on token count locally
(llama-swap + llama.cpp native router are name-driven; token triggers live
in client-side routers — claude-code-router's longContextThreshold=200K
validates the pattern). llama.cpp's new router mode holds connections with
ZERO bytes during loads (worse than our heartbeating proxy) and its sleep
mode has a compounding-fit bug class specifically with mmproj+draft configs
(#24475). Community latency data confirms ours: warm weight load ≈ 3 s
PCIe; "the cost nobody counts is the destroyed prompt cache" — which the
slot-portability finding eliminates. VERDICT: build in our proxy.

**3. Client-UX agent.** Timeout physics: idle watchdogs are beaten only by
PROTOCOL-SHAPED no-op frames (bare SSE comments are invisible to several
agent parsers — 4 documented bug reports); several clients carry 300 s
TOTAL deadlines (configurable — Hermes is ours to configure high). Commit
200+headers immediately, heartbeat every 10-15 s with comment + empty-delta
chunk reusing the stream id (OpenAI) / event:ping (Anthropic), in-band
shaped error frames after headers, 503 only pre-headers. Drain in-flight
before swap (llama-swap pattern), health-gate on /health 200 with 500 =
fail-fast, mutex-fenced writer handoff, heartbeat stays armed through the
post-swap first turn, is_disconnected() aborts a swap nobody awaits.

## REVISED PLAN (recommended)

1. **Rope unification gate**: speed profile SCALE 1.25→1.5625; battery
   2-3 seeds @64K on it (tail-KLD evidence already favorable). If it gates
   clean → proceed; if not → Phase-1 re-prefill design stands at 1.25.
2. **Proxy auto-tier**: unified alias, usage-based token tracking, trigger
   TIER_UP_TOKENS (default 180K), hold-and-swap with save→restart→restore
   (re-prefill fallback), shaped heartbeats per protocol, drain + health
   gate + fenced handoff, idle tier-down (default 15 min), env-config all.
3. **Drill**: synthetic session crossing the threshold → verify switch
   time (~20-50 s), continuity (answer references pre-switch context),
   fallback path, tier-down, and the proxy test suite extended (~12 tests).
Effort: ~1 day including gates.

## QUICK YARN-TAX RESULTS (2026-08-16, 1-hour suite; full A'+B' overnight pending)
- Identity gate PASSED (yarn@1.0 ≡ none at noise floor, 99.991% same-top).
- SHALLOW (full-vocab KLD, 32K samples): none→1.25 = 98.49% same-top
  (-1.5%); none→1.5625 = 97.98% (-2.0%); **unification increment
  1.25→1.5625 ≈ 0.5% top-1 at short context**, mean-Δp -0.03%. All far
  below the KV-quant effect (96.2%).
- DEEP (256K station, n=25 branch probes): the scales genuinely diverge —
  1.25-vs-1.5625 top-1 agreement 80%, heavy tail. CANNOT be adjudicated by
  distribution tools (no-yarn sits at its native cliff there; divergence ≠
  worse). Deep-quality attribution = the overnight matched-config battery
  (B') at 64K/128K/256K tiers × 3 scales.
- Data: quality-tests/yarntax/*.json; analyzers yarn_stations.py /
  yarn_tax_analyze.py.
