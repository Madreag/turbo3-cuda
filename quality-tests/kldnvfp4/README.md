# NVFP4-MTP-Q8attn first-slice gate — 2026-08-17 — VERDICT: ARTIFACT REJECTED

Artifact: utautako/Qwen3.8-27B-NVFP4-MTP-Q8attn-GGUF (17.8 GiB, community
quant; local models/qwen38-nvfp4/). Server: prod binary, production-matched
config (yarn 1.25/262144, c=327680, turbo4 KV K+V+draft, alphas 1.00, fused
MMA, draft-mtp n3), port 8233.

- LOAD/MTP: PASS — full 320K window fits, MTP draft context engages
  (acceptance 0.78 smoke, 3.35 mean draft len).
- VRAM: 28,423 MiB vs prod 31,255 → **2.8 GiB freed** (headroom 4.2 GiB).
- SPEED (nmax harness vs same-day Q6_K n3 baseline): shallow code +20-25%
  (161 t/s), shallow prose +15%, 38K code +6-10%, 38K prose +9-10%.
- **QUALITY: FAIL** — tail-KLD 157 vs archived f16-KV ref (same instrument
  as every shipped gate): mean 0.0693 / p50 0.0409 / p99 0.6133 / top-1
  82.8% / Δp-RMS 0.165 — vs prod baseline 0.0060 / p99 0.062 / 96.2%.
  **11.5x KLD, −13.4 pts top-1.** (Calibration: the alpha-1.12 mistake we
  hunted down was 0.024.) High shallow-code self-acceptance (0.93 vs 0.61-
  0.77) = lower-entropy degraded distribution, corroborating.

SCOPE: convicts THIS artifact, not the NVFP4 format. T1 stays OPEN pending a
properly calibrated candidate (unsloth dynamic / imatrix). Speed+VRAM
expectations are now pre-measured for any future candidate: expect ~+10-25%
decode and ~3 GiB freed at Q8attn-class size; gate bar unchanged
(mean ≤ ~0.012-class, top-1 ≥ ~95%).
