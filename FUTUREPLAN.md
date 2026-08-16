> **HISTORICAL (closed 2026-08-15).** All phases executed or superseded.
> Outcomes and every later verdict live in WORKPLAN-BESTAPP.md (authoritative)
> and HERMES-HANDOFF.md (current state). Notable supersessions: MTP adopted
> then re-based on MMA-turbo kernels; q8-K option deprioritized (pure turbo4
> with corrected centroids beats the old hybrid arm); art-turn 'grammar
> penalty' was VRAM paging.

# FUTUREPLAN — Quality & Speed Roadmap (drafted 2026-08-15)

## OUTCOMES (updated 2026-08-15, A+B evening executed)
- Phase 0: DONE (branch pushed; deploy/ versioned; scorer fixed+validated).
- Phase A: DONE, ADOPTED — alpha 1.00/1.00 in production (was 1.10/1.12);
  KLD 0.0087 vs 0.0200, confirmed under YaRN; gated by preverify + render
  PASS. Methodology finding: the 3.6-era KLD runs never actually measured
  quantized-cache readback (single-ubatch prefill) and hid a 503 cascade;
  tool fixed (2048-tok prompts, cache_prompt off, skip-count surfaced,
  noise floor). Full data + caveats: quality-tests/kld38/README.md.
- Phase B: DONE, NOT ADOPTED — q8_0 K gives a real-but-modest gain (KLD
  0.0054 vs 0.0087) that doesn't justify −70K ctx or CPU-vision today;
  data preserved for future re-evaluation. Long-ctx q8K spots skipped
  (only needed for adoption).
- Phase C (MTP): ADOPTED 2026-08-15 (verdict REVERSED by coding A/B after
  user challenge — the first rejection sampled only art turns). At production
  sampling: coding 89-98 tok/s vs 57.4 baseline (1.55-1.71x, acceptance
  59-63%), general 95.4 vs 57.5 (1.66x); art/tool-grammar turns 0.6-0.85x
  (n=2, grammar-draft interaction suspected — investigation lever). Enabled
  globally (no per-request toggle exists) per coding-primary usage. Zero VRAM
  cost at full profile. Bonus fix: stale slot-state files from config changes
  crashed the server via GGML_ASSERT in state_seq_load_file — now refuses
  gracefully (upstreamable). Slot saves archived on config change as ops rule.
- Phases D/E: pending (Hermes-side).

Planning document. Nothing here is executed until the user says go.
Production baseline this plan measures against: Qwen3.8-27B Q6_K, 409600 ctx
YaRN 1.5625, turbo4/turbo4 KV, alphas 1.10/1.12, temp 1.0, reasoning_effort
xhigh, vision (mmproj F16) live. Validated 2026-08-14: ladder xhigh 3/4 clean
renders; NIAH eff. 5/5 @130K+380K; acceptance 1/1, 0 tripwire alerts.

Ordering: A+B share one box window (~5h). C depends on UPSTREAMSYNC.md.
D and E are Hermes-side and need no server window.

## Phase 0 — safety pre-flight (before ANY phase; no GPU; ~15 min)
1. Push `hermes/server-foundation` to myfork (Madreag/turbo3-cuda) — two
   weeks of fixes and these plans currently exist on one disk only.
2. Version the production config: create repo `deploy/` with proxy.py,
   test_proxy.py, start*.sh, watch.sh — KEYS EXCLUDED (api.key, keys.json
   never enter the repo). Deployed copies in ~/.config/llama-tcq/ remain the
   live ones; deploy/ is the versioned source of truth going forward.
3. Note the newest real-traffic capture's date; battery replays must use the
   newest capture, refreshed by one live Hermes turn if the tool set changed
   since (also recorded in UPSTREAMSYNC Layer 4).

─────────────────────────────────────────────────────────────────────────────
## Phase A — turbo4 alpha KLD sweep on Qwen3.8            (~3-4h, one window)

WHY: TURBO_NORM_ALPHA_V=1.10 / TURBO4_NORM_ALPHA_V=1.12 are reconstruction
gain corrections tuned on Qwen3.6's activation statistics. 3.8 shares KV
geometry but not necessarily V-norm distributions. Outcome is either written
confirmation or a small free quality gain on every generated token.

PROCEDURE
0. DRY RUN FIRST (15 min): validate the 3.6-era kl_divergence.py pipeline
   against 3.8 on 2 chunks — the vocab grew to 248,320, so confirm the
   logprob-capture format and check the reference file's disk size scales
   sanely before committing to 32 chunks. Fix-forward here, not mid-sweep.
1. Corpus: wiki.test.raw, 32 chunks × 2048 tokens (same recipe as the 3.6-era
   kld_logprobs_*.json runs in quality-tests/).
2. Reference pass: server at `-ctk f16 -ctv f16`, ctx 16384, vision off,
   NO YaRN (short-context purity, matching how the 3.6 alphas were tuned)
   → logprobs → kld_logprobs_38_f16.json. REUSED BY PHASE B — run once.
3. Sweep passes: same server config but turbo4/turbo4, one pass per alpha in
   {1.00, 1.05, 1.10 (current), 1.12, 1.15} — alphas set via env pair; keep
   the V-alpha pair moving together first; only split-tune if the best value
   disagrees with current by ≥0.05.
4. Compute mean KLD per alpha vs reference (quality-tests/kl_divergence.py).
5. Decision rule: adopt new alpha only if KLD improves ≥3% vs 1.10/1.12
   (below that = noise; keep current).
5b. CONFIRMATION PASS: one extra KLD pass at the winning alpha with the
   PRODUCTION config (YaRN 1.5625, ctx 409600 server settings) — the sweep
   runs YaRN-off for method consistency with 3.6, so confirm the winner holds
   under production RoPE conditions before adopting. If adopted → edit
   start-long-38.sh → one battery preverify + render gate before declaring
   done.
6. OPTIONAL (+1h, novel data): vision-token probe — 5 image prompts, compare
   answer distributions f16-KV vs turbo4-KV. No one has this data anywhere.

GATES: every server restart follows kill-then-launch-in-separate-calls; one
corpus pass at a time; production restored via start-long-38.sh at window end.
ROLLBACK: alphas are env values in one script line.

─────────────────────────────────────────────────────────────────────────────
## Phase B — q8_0 K-cache evaluation                       (+1.5-2h, same window)

WHY: keys do attention addressing; K-cache quantization damages retrieval
more than V. Old 3.6 daily profile ran q8_0 K for exactly this reason.

PROCEDURE
1. Reuse Phase A reference logprobs.
2. KLD pass: `-ctk q8_0 -ctv turbo4` (best alphas from A), same corpus →
   ΔKLD(q8K vs turbo4K).
3. Long-context spot: NIAH 130K point on q8K config (ctx 200000, vision off
   for the test server — fits easily).
4. One artifact preverify + render gate on q8K config.
5. Production trade decision (user call, data in hand):
   - q8_0 K costs +4.25 KiB/token → +1.7GB at 409600. Does NOT fit with
     GPU-resident vision at current ctx.
   - Fit options, best first: (a) ctx 340000 + vision moved to CPU via
     `--no-mmproj-offload` (reclaims ~0.9GB; headroom ≈ 1.8GB — healthy;
     image encode gets a few seconds slower); (b) ctx 340000 + GPU vision →
     headroom ≈ 0.9GB — BELOW our volatility comfort line, not recommended;
     (c) ctx 350000 + vision-off; (d) stay full-turbo4 at 409600.
   - Adopt only if ΔKLD is decisively better AND the artifact/NIAH spots
     show no regression; otherwise document and close.

─────────────────────────────────────────────────────────────────────────────
## Phase C — MTP speculative decode                        (rides UPSTREAMSYNC)

WHY: the MTP head already ships in our GGUF (blk.64, loaded-and-skipped).
Upstream has mature runtime (60 MTP/NextN commits since our base era,
auto-detect draft support). Expected 1.4-2× decode → xhigh art turns drop
from ~30 min toward ~15-20. Single biggest felt improvement available.

PLAN: do NOT hand-port. Absorb via the upstream sync (UPSTREAMSYNC.md), then:
1. Enable spec-decode MTP flags on a test profile (upstream flag names to be
   confirmed at sync time; community guidance: draft-n 2, acceptance drops
   hard at 4+).
2. Verify: tok/s ladder (0K/30K/100K ctx); correctness via GREEDY identity —
   temp 0 + fixed seed on N=5 prompts, MTP-on output must match MTP-off
   byte-for-byte (spec decode is distribution-preserving; at temp 1.0 outputs
   differ by sampling, so greedy is the only honest identity test); then one
   battery preverify + render.
3. VRAM: +~1GB working memory reported by community — headroom math BEFORE
   enabling with vision resident; may require ctx 380K or vision-off choice.

─────────────────────────────────────────────────────────────────────────────
## Phase D — Vision self-repair loop                       (Hermes-side design)

WHY: residual art failure is now ~25% visual/logic misses (ladder: fails were
monochrome washes and silent dead scenes, not JS crashes). The model can now
SEE screenshots of its own artifacts (proven: read its own UI text verbatim).
Post-hoc self-correction — the sanctioned no-caging direction.

SKETCH (needs Hermes-side skill work, no server changes):
1. After write_file of an artifact, Hermes (skill/rule) renders headless,
   screenshots, and sends the image back: "this is your page as rendered —
   fix what's wrong" with the repair patch as a follow-up write.
2. Server side is READY: image_url parts work through the proxy end-to-end.
3. Success metric: ladder clean-rate with one repair round vs without —
   target 75% → 90%+.
4. Open questions for the user: where the render step runs (Hermes host has
   Chrome? or reuse our render_arm plumbing?); loop budget (1 repair max?);
   per-screenshot context cost — roughly 1-3K tokens per image at typical
   patching (measure exactly from usage stats on the first real run), plus
   the repair turn's own thinking budget.

─────────────────────────────────────────────────────────────────────────────
## Phase E — reasoning_effort routing                      (Hermes-side, free)

Verified: per-request `chat_template_kwargs.reasoning_effort` works through
the proxy (xhigh/medium/low render correctly via /apply-template).
Recommendation to encode in Hermes: art/complex turns = xhigh (default);
quick tool loops (ls/bash/todo) = medium. NEVER low (ladder: 0/2 clean,
no token savings, indecisive tool retries). This is a client-side choice —
does not touch the no-caging law. Latency win on the majority of turns.

─────────────────────────────────────────────────────────────────────────────
## Standing rules for every phase

- One GPU job at a time, babysat via Monitor events; no work while Hermes may
  be active; production restored at every window end.
- Kill and relaunch in separate calls; pgrep -x / bracket patterns only.
- Render truth = pixels + console (render_arm), never file size.
- Every adopted change lands in: start script → handoff → memory → commit.
- Rollback pairs preserved before any adoption.
