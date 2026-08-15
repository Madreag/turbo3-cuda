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
