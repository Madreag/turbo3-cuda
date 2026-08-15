# Qwen3.8 turbo4 alpha sweep — 2026-08-15 (corrected methodology)

Method: kl_divergence.py with 2048-token prompts (forces cross-ubatch
quantized-cache readback — 256-token single-ubatch prompts NEVER read the
quantized cache; the measured token's logits come from in-graph f16 prefill),
cache_prompt=false (prevents the 503 cascade that silently voided 52/100
prompts in earlier runs), f16-vs-f16 noise floor = exactly 0.0 (single-stream
determinism), n=60.

Result: alpha curve is a clean parabola with MINIMUM AT 1.00 — Qwen3.8 wants
no turbo4 V-norm correction. Old 3.6-inherited 1.10/1.12 cost 2.3x KLD.
Confirmed under production YaRN (1.00: kld 0.0150 / top1 96.7 vs 1.12:
0.0243 / 90.0). Adopted in start-long-38.sh, gated by battery preverify +
pixel render (PASS).

q8_0 K + turbo4 V @ alpha 1.00: kld 0.0054 vs 0.0087 (real but modest gain);
NOT adopted — costs ~70K ctx or CPU-vision at production scale. Data here if
that trade ever looks different.

CAUTION on old data: the 3.6-era kld_*.json files in quality-tests/ were
measured with the 256-token methodology and are unreliable — top-1 deltas
there likely reflect nondeterminism + silent exclusions, not quantization.
The turbo4_cury/a100y pairs here are the YaRN-condition measurements.

## 2026-08-15 addendum — corrected centroids + MMA port gate
turbo4_ccmma_yarn (corrected Lloyd-Max 4-bit table, 66B block, MMA decode):
kld 0.004728 / top1 98.3 / dprms 0.040 — vs old-centroid baseline 0.0150 /
96.7 (−68% KLD; the old table was mis-scaled ~0.72x → ~3x excess KLD).
Beats even the q8K+turbo4V hybrid arm (0.0054) at pure turbo4 bits.
REFERENCE ARCHIVED: kld_logprobs_f16_qwen38_yarn.json (f16-KV arm, YaRN
1.5625 rope, 2048-tok prompts, n=60). LESSON: the Aug-15 alpha sweep archived
only summaries; its f16 reference was lost and the stale Qwen3.6-era file in
quality-tests/ produced a false-catastrophic gate (KLD 0.47, top1 6.7%) when
first used against 3.8. Always archive the reference DISTRIBUTIONS with the
summaries; the 3.6-era file is quarantined as *-DO-NOT-USE.
