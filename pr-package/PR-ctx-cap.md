# server: respect configured rope scaling in the slot-context cap

The server unconditionally caps `n_ctx_slot` to `n_ctx_train`, silently
clamping deployments that configure rope scaling (YaRN etc.) to run past the
training context. A 409K-context YaRN deployment (Qwen3.8-27B,
`--rope-scaling yarn --rope-scale 1.5625`) started rejecting >262144-token
requests with HTTP 400 after this cap landed — caught by a 380K NIAH battery
going 0/5 with no error in the logs beyond the WRN.

This keeps the cap for unscaled models (where it protects users from silent
quality collapse) but allows the configured case:
- explicit CLI rope scaling (`--rope-scaling` != none), or
- GGUF-native scaling (`rope_scaling` in metadata; the library resolves
  UNSPECIFIED to the model's trained scaling, so the CLI field alone is not
  sufficient — checked via `llama_model_rope_freq_scale_train()`).

Tested: YaRN 409600/1.5625 deployment accepts >n_ctx_train requests again
(NIAH 5/5 at 380K); unscaled models still cap with the original warning.
