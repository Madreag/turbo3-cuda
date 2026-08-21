# Deep-context decode benchmark — Qwen3.8-27B, turbo4 KV (2026-08-20)

Measured on the production stack (RTX 5090, Gen4, new 12VHPWR adapter, stock
clocks) to answer: **does our quality-first config hold up at the deep context
we actually serve (262K+)?** Short answer: yes — graceful decay, no cliff.

## Results (pure decode `tg`, server-reported, temp 1.0 production sampling)

| depth | profile | decode t/s | MTP accept | workload |
|------:|---------|-----------:|-----------:|----------|
| ~0    | 320K · YaRN 1.25 · MTP n3 | ~88 | 0.35 | prose essay |
| ~32K  | 320K · YaRN 1.25 · MTP n3 | ~80 | 0.40 | prose essay |
| ~64K  | 320K · YaRN 1.25 · MTP n3 | ~70 | 0.44 | prose essay |
| ~314K | 320K · YaRN 1.25 · MTP n3 | **~45** | 0.55 | corpus synthesis |
| ~350K | 409K · YaRN 1.5625 · MTP OFF | **~27** | n/a  | corpus synthesis |

Prefill at depth ≈ 850–870 tok/s effective → a ~300K prompt is ~6 min to first
token (one-time; ctx-checkpoints reuse it on follow-up turns). VRAM: 320K
profile ~31.5 GB; MAX profile 30.8/32.6 GB (1.8 GB free — MTP-off frees draft
state). Model: uncensored Q6_K for Leg 1 (320K), original Q6_K for Leg 2 (MAX);
same arch/quant → speed representative of the profile, not the weights.

## Analysis

1. **No cliff — graceful ~2× decay over 0→314K** (88→45) on the speed profile.
   This is the whole point of turbo4 + fused-MMA: naive Q6_K KV craters to ~16
   t/s at 262K (community reports); our fused path holds 45 t/s at 314K.
2. **45 t/s at 314K is genuinely usable** — faster than reading speed, at full
   near-transparent Q6_K quality (KLD ~0.005–0.006 vs f16). This is the deep-
   context number that matters for our actual usage.
3. **MAX profile cost is MTP-off, not depth per se.** 27 t/s @350K vs ~40 t/s
   extrapolated-if-MTP-on. MTP is disabled to fit 409K KV; the extra ~95K of
   context costs ~40% decode. Use MAX only when >314K is truly needed.
4. **Acceptance is workload-dependent** — rose to 0.55 on real-document
   synthesis (representative of long-context use) vs ~0.4 on free prose. So the
   deep-context real-world case is *better* than the shallow prose points, which
   is why the 314K number isn't as low as a naive prose extrapolation predicts.
5. **Prefill, not decode, is the deep-context latency driver** — ~6 min to first
   token on a fresh 300K prompt. ctx-checkpoints/prompt-cache amortize this
   across a conversation; the first turn on a huge doc is the wait.

## vs the community speed configs (honest)
At 262–314K the NVFP4/Q4 speed-configs report ~120 t/s vs our ~45. That gap is
(a) lower weight quality (NVFP4/Q4 < Q6_K), (b) their headline numbers are
usually greedy while ours are real temp-1.0, (c) they spend VRAM we spend on
quality. We trade ~2.5× peak deep-speed for top quality that never degrades and
no cliff. For a "cheap/private/does-anything, not primary coder" role this is
the right trade.

## Methodology notes (so this is reproducible & the numbers are trustworthy)
- **Measure pure decode (`tg=` from server log), NOT wall-clock.** Wall-clock at
  depth is prefill-dominated — a first bad probe showed a misleading "18 t/s at
  32K" that was really 32K-prefill + 250 gen. Pure decode at 32K is ~80 t/s.
- **Fill context with varied real text (calib corpus), NOT repeated filler.**
  Repeated filler is trivially predictable → fake-high MTP acceptance.
- Corpus char/token ≈ 3.0 for calib_v2 (a 3.6 estimate overshot: a "262K"
  request actually hit 314,540 tokens; a "300K" request 400'd for exceeding the
  327,680 ctx cap — which is why the top speed-profile point is reported at 314K).
- Harness: /tmp deep.py/deep2.py (spot probes); repro pattern = fill to depth,
  gen 600 tok, read server tg/pp/acceptance.

## Follow-ups
- **DFlash2 + turbo4 (headline post-trip experiment):** DFlash2 block-drafter
  could lift deep-decode above MTP; turbo4's KV compression reclaims the VRAM
  the drafter costs — a combination nobody's published (the two live in separate
  forks). Target: 45 → 70+ t/s @314K at unchanged quality. See notes in
  OVERNIGHT-REPORT / this session.
- Controlled single-workload sweep (same prompt at all depths) would tighten the
  curve; these are real-workload spot measurements, indicative not lab-grade.
