# GOAL v3 — RENDER PARITY: find out WHY, then fix OURS. Non-stop.

**THE QUESTION (the entire goal):** Others reportedly run this same model and get
working voxel scenes. Ours fails ~50% of freeform one-shots. Something differs —
config, runtime, sampling, quantization, context shape — and it is FINDABLE by
controlled comparison. Find it. Fix ours. Prove it with pixels.

**LAWS (absolute):**
1. The MODEL is never modified, constrained, templated, or caged. Config/settings/
   runtime changes are the only levers (sampling, KV cache type, rope, context shape
   are config — allowed; prompts and weights are the model's — untouchable).
2. The GPU never idles while a gate is open. Test batches queue back-to-back;
   CPU verification runs in parallel with the next GPU batch, always.
3. Evidence = headless-Chrome pixels + JS console only. n≥8 runs per config —
   single runs prove nothing at a ~50% base rate.
4. Non-stop: a finished batch immediately triggers analysis AND the next batch.
   No idle turns, no waiting states, no "done" until the user says pagodas render.
5. DIAGNOSTIC ARMS ARE NOT SHIP CANDIDATES. P1 flips variables (f16 KV, YaRN off,
   etc.) only to LOCATE the cause. Production keeps the TurboQuant identity —
   turbo KV compression and the long-context window stay, period. If a diagnostic
   implicates one of our optimizations, the deliverable is a FIX INSIDE it
   (kernel/calibration work — the repo's actual mission), never its removal.

## P1 — Variable isolation on OUR stack (the suspects, each n≥8, fatal-rate table)
Same prompt, same model file, one variable flipped at a time vs current baseline
(temp 0.6, top-k 20, turbo4/turbo4 KV, YaRN 1.5625, 33K agent context):
  a. Sampling: temp 1.0 / top-p 0.95 / top-k 20 / min-p 0 (Qwen's creative preset —
     others' UIs pull THIS from GGUF metadata; we run 0.6)
  b. KV cache: q8_0/q8_0, and f16/f16 — CACHE precision only, at diagnostic ctx
     (~36K → f16 cache ≈ 2.3GB, total ~24GB, fits 32GB fine). WEIGHTS stay Q6_K in
     every arm; f16/bf16 weights (~54GB) do not fit this card and are never used.
     (Others run f16 cache — our turbo KV is a real differing variable; PPL cleared
     it, but 800-line codegen ≠ PPL.)
  c. YaRN: OFF at native ctx (others don't run rope scaling)
  d. Context shape: bare 1-turn prompt (~200 tokens, like a chat UI) vs 33K agent shape
Output: one table — config → fatal-JS rate → verdict per variable.

## P2 — External runtime ground truth (the "other apps" the user demanded)
Install and run at least ONE independent runtime on this box (ollama first choice —
scriptable; LM Studio if needed), same model family+quant class, same prompt, n≥8,
identical pixel verification. Their real failure rate, measured — not reddit's word.

## P3 — VERDICT with the table
Name the variable(s) that move the rate, from data. Outcomes both count as answers:
  - A config difference explains it → that's the fix.
  - Every runtime/config shows the same rate → "others succeed" was survivorship,
    proven with numbers nobody else has ever published.

## P4 — OURS FIXED and re-verified
Deploy the winning config to production (model untouched). ≥8/10 clean pixel-verified
renders through the REAL proxy path at production settings. Then the user runs the
pagoda in Hermes and confirms working scenes in their own Chrome — their words close
the goal, nothing else.

## Standing context (do not re-discover)
Phase-1 transport: DONE (streaming/stalls/preserve_thinking all fixed and live-proven;
branch hermes/server-foundation). R1 forensics: fatal JS slips reproduced (`dy` undefined,
unbounded loop); artifact byte-capture live in proxy; headless render harness proven
(Windows Chrome from WSL + timeout 45 + pixel analysis; venv+pillow in scratchpad).
Baseline measured so far: freeform ≈ 3-4 clean of 6 draws at temp 0.6. Prior user laws:
scaffold method BANNED; benchmarks babysat; socket-owner process management only.
