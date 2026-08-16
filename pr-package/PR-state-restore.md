# llama: make sequence-state restore abort-proof on malformed state files

State files saved under one configuration and restored under another
(changed KV types, context size, speculative settings, or simply a different
build) currently walk into raw GGML_ASSERTs and abort the whole server.
Production forensics: enabling MTP changed the recurrent-state layout
(n_rs_seq 0→2); restoring a slot file saved pre-MTP aborted llama-server via
an assert in the restore-meta path. A hybrid model with vision (mrope) also
changes the per-cell serialization (llama_kv_cell_ext), so stale files can
MISALIGN every subsequent read — producing garbage that passes the weak
n_seq_id check and detonates the "DEBUG CHECK" asserts.

The attention and recurrent caches both had three gaps, fixed symmetrically:
1. `state_read_meta` ran OUTSIDE the try that guards `state_read_data`, so an
   EOF throw mid-meta skipped the seq_rm cleanup and left cells partially
   applied.
2. The single-sequence restore path had no bound on `cell_count` before
   `ubatch_reserve` (the whole-cache branch has one) — a corrupt count aborts
   or over-allocates.
3. The post-apply consistency checks were GGML_ASSERTs reachable from
   malformed input; they are now soft failures that log and return false,
   taking the existing seq_rm + "failed to restore kv cache" path, which the
   server already converts to a clean 400 for the client.

After this change a stale/corrupt/incompatible state file cannot abort the
process on either cache: verified by restoring the exact file that previously
crashed a production server — now a clean HTTP 400 with the server healthy,
and a same-config save/restore round-trip is unaffected (byte-identical
restore, same tokens).
