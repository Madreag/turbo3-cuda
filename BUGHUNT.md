# Bug Hunt — 2026-08-15

Four parallel read-only audits over the production stack (ops scripts, proxy.py, carried
C++ patches in turboquant-sync, server flag/speculative semantics), findings verified
against primary sources before recording. Production config for triage: Qwen3.8-27B
(qwen35 hybrid, 16 attn layers, head_dim 256), turbo4/turbo4, FA on, MTP n_max=2,
ctx 409600 YaRN, --parallel 1, vision on, SM120, TURBO_LAYER_ADAPTIVE unset.

Legend: **[LIVE]** = affects production behavior today. **[DORMANT]** = real defect,
not reachable in current config. **[HYGIENE]** = correctness-neutral debt.

---

## A. Production-impacting now

### A1. [LIVE] Slot save/restore is effectively dead — proxy checks the wrong directory
`proxy.py:906` hardcodes `slots/` but the long-38 profile saves to `slots-long/`
(`--slot-save-path`). Verified on disk: `slots/` holds only a stale Jul-28 Qwen3.6-era
`erol.bin` (acts as an accidental existence-sentinel), `slots-long/` currently has no
`.bin`. Effect: `brother` NEVER gets a restore on owner swap (existence check always
false → slot_erase); `erol` gets a restore *attempted* only because of the stale
sentinel. Every owner swap = full re-prefill (minutes at long ctx).
**Fix:** drop the existence check; always attempt restore and rely on the server's
graceful 400 → slot_erase fallback (that path is verified correct). Derive any needed
dir from config, never hardcode.

### A2. [LIVE] The morning's slot-restore crash was NOT fixed — patch targets an unreachable assert
Commit `1ad957c8e` hardened `state_seq_load_file`'s size check, but both disjuncts are
provably unreachable (short reads throw in `llama_file::read_raw` first; the tell()
identity holds by construction). The actual abort: Qwen3.8 is hybrid, and the
**recurrent-memory** restore path uses raw `GGML_ASSERT`s
(`llama-memory-recurrent.cpp:799,805,987-991`) where the attention path throws
(→ graceful 400). Enabling MTP changed the recurrent state layout (`n_rs_seq` 0→2,
rows ×3 — see D5), making the pre-MTP file trip a recurrent-section assert → abort.
**Still live**: any future config change + a stale slot file reproduces it. Currently
shielded only by bug A1 (restores rarely happen). The `slots/erol.bin` stale sentinel
means fixing A1 *unshields* A2 — fix both together.
**Fix:** convert recurrent state_read GGML_ASSERTs to `throw std::runtime_error`
(mirrors attention-side contract; the catch → seq_rm → 400 machinery already exists).
Keep 1ad957c8e as defence-in-depth. This is the real upstreamable fix.

### A3. [LIVE] Proxy delivers truncated upstream streams as *successful* completions
OpenAI path (`proxy.py:1216-1226`): on end-without-[DONE] it logs [ALERT] then
`write_eof()` — client sees clean EOF, no error frame, no [DONE]. Anthropic path is
worse: no tripwire at all, and `builder.close()` unconditionally synthesizes
`end_turn` + `message_stop` — the exact 2026-08-08 corpse signature presented as a
normal finished turn. Detection exists for *us*; repair for the *client* does not.
Also: upstream death mid-stream (`ClientPayloadError`) propagates uncaught → raw
transport drop (no error frame); unmatched `<think>` buffer is never flushed at
stream end (silent tail loss); literal `<thinking>` in generated content (e.g. inside
an artifact) reroutes the remainder of the response into reasoning.
**Fix:** on both paths, when the stream ends without its terminator: emit
`data: {"error":...}` + `data: [DONE]` (OpenAI) / `event: error` before
`message_stop` (Anthropic); catch ClientPayloadError in the pump; flush think-buffer
as content at end; only honor think-tag openers in the leading segment.

### A4. [LIVE] Passthrough forwards ANY path upstream with the server's key, outside the owner lock
`proxy.py:1250-1268`: any valid client key reaches every llama-server endpoint —
including `POST /slots/0?action=save|restore|erase` — no lock, no owner-state update.
A stray client call can erase/overwrite the other user's cache and desync
`current_owner` (next swap then saves the wrong user's KV into the stale owner's
file). `/completion`, `/v1/completions`, `/infill` also bypass slot pinning, and the
passthrough buffers whole bodies with `total=120` (breaks streaming users of those
endpoints).
**Fix:** allowlist (`/v1/models`, `/props`, `/health`, `/tokenize`, `/metrics`), 403
the rest, `/slots*` above all.

### A5. [LIVE] stop.sh/start scripts: no guards, false green, log destruction, key exposure
Verified against sources:
- `stop.sh` sends one SIGTERM, immediately rm's pidfiles, then **unconditionally
  pkills the same processes** (second SIGTERM lands mid-graceful-shutdown → aborts
  the proxy's on_cleanup slot_save; "killed extra" prints on every healthy stop),
  exits 0 without verifying death or port release. `stop.sh && start` can
  deterministically reproduce the bind-conflict incident.
- `start-long-38.sh`: no double-start guard; if run while up, new server dies on
  bind, pidfile clobbered, and the health loop gets 200 **from the old server** —
  success banner naming a dead PID. Health-timeout is not an error: after 180s it
  launches the proxy against a possibly-still-loading server and reports success
  (cold-cache model load alone can exceed the window at this box's 124 MB/s).
- `>` log redirects truncate the LIVE processes' logs on double-start. Verified
  casualty: current proxy.log contains only the dead duplicate's traceback; the live
  proxy's fd sits at offset 3898 past a NUL hole (~3.9 KB of production log
  destroyed).
- API key passed via `--api-key "$TCQ_KEY"` → readable in `/proc/*/cmdline` by any
  local process. Server supports `--api-key-file` (arg.cpp:3455). `keys.json` is 644.
- Header comment still claims alphas 1.10/1.12 (live exports: 1.00) and pins binary
  provenance to a superseded commit. Documented rollback line is broken (relative
  paths not on PATH; `cp` onto a running binary → ETXTBSY; `.pre-nextn` target is
  older than `.pre-sync` and dynamically linked against another tree). No backup of
  the current static binary exists.
- Latent stale systemd units (Apr 12: Qwopus model, no top-k 20, no slot-save-path,
  `Restart=on-failure` — would fight manual scripts if ever enabled). Not in version
  control.
- `status.sh` checks PID liveness only (pidfile-missing-but-alive → "Not running" →
  operator starts → bind conflict; wedged-but-alive → "Running").
**Fix:** rewrite stop.sh (verified kill: poll `kill -0`, escalate to -9, confirm
ports free; pkill net only after pidfile PIDs are gone); start guard (`ss` port
check → abort); health-timeout → abort before proxy launch; append+rotate logs;
`--api-key-file`; `chmod 600 keys.json`; fix comments + rollback text; back up the
live binary; delete or regenerate systemd units; add port+health probe to status.sh.

### A6. [LIVE] `python3 test_proxy.py` silently under-runs the suite: 37/37 reported, 42 exist
The `TESTS` list + `__main__` block sit *before* five tripwire/capture tests defined
later in the file; script mode exits early. Verified by running it. Two of the five
need pytest fixtures. The "42 tests" claim we've been operating under was never
exercised end-to-end in script mode.
**Fix:** move `TESTS`/`__main__` to end; port the two fixture tests to script mode.
Coverage gaps worth closing while there: zero handler-level (aiohttp test client)
coverage — client-disconnect, upstream-death, lock, swap, heartbeat all untested;
`maybe_swap_slot` has no test (a one-line assertion on the slots dir would have
caught A1).

---

## B. Correct-but-notable semantics (no action required to stay correct)

- **No per-request speculative control exists.** The request fields are `#if 0`-ed
  out of `server-schema.cpp:197-227` AND dead-ended (task params never read by the
  drafting decision — `server-context.cpp:2915-2957` consults only the server-wide
  object). `"speculative.n_max": 0` in a request is silently ignored. Proxy-side
  per-request MTP disable is impossible without a code change (see Improvements #1).
- **Drafts are grammar-blind** (zero grammar references in common/speculative.cpp;
  draft = bare top-k argmax) — rejection happens at the grammar-constrained target
  verifier. With `tools` + auto choice the grammar is LAZY (active only inside the
  tool-call JSON block); `tool_choice:"required"`/`json_schema` = active from token 0.
- **Grammar disables GPU backend sampling on the target**
  (`common/sampling.cpp:415-416`) — a per-request throughput tax on grammar turns
  that applies WITH OR WITHOUT MTP. Part of the observed art-turn 0.6-0.85× is
  likely this, not draft rejection. Measurable via `spec_decode_*` Prometheus
  counters — requires `--metrics` at launch (not currently enabled).
- **`--swa-full` is structurally dead** on qwen35 (sole consumer is the iSWA cache
  ctor; qwen35 asserts `!is_swa_any()`). Safe to drop.
- **`--no-context-shift` is clean** (overflow → `finish_reason:"length"`, never a
  crash) and likely redundant: mmproj presence force-disables ctx_shift anyway.
- **Sampling precedence is correct** (per-field request-over-CLI, null = server
  default). Gotcha: the server's native `/v1/messages` endpoint injects
  `max_tokens=4096` when the client omits it, and whitelists only
  temp/top_p/top_k/stream/chat_template_kwargs.
- **MTP n_max=2 geometry**: verification batch 3 tokens vs 2048 budget (~680×
  headroom); `--parallel 1` optimal; n_max=2 sits exactly on the no-checkpoint
  rollback fast path. BUT n_max sets `n_rs_seq` → recurrent-state rows ×(1+n_max):
  raising n_max costs DeltaNet state VRAM linearly across 48 layers (D5) and was
  the layout change that made pre-MTP slot files fatal (A2).

## C. Dormant C++ defects (real, not reachable in today's config)

- **C1 (was 4.1):** hand-port introduced shadowed `n_embd_head = k_cur->ne[0]` after
  the merged-2D reassignment (`llama-kv-cache.cpp:1496,1539`) — reads n_embd_gqa
  (1024) not head_dim (256). Same group size (128) selected either way today; the
  64-group branch is dead and write/read sides derive groups from different
  quantities. Fix: delete both inner declarations (2 lines, restores fork behavior).
- **C2 (was 2.2):** ctx-cap fix `dc6f94ef2` tests only the CLI rope field; a GGUF
  with native YaRN metadata run without `--rope-scaling` still gets capped — the
  exact failure the commit meant to prevent. Fix: mirror library resolution
  (CLI value, else hparams train type). Also doesn't check scale > 1×.
- **C3 (was 3.1/3.3):** parse-degrade try/catch rethrows on the partial/streaming
  path (`server-task.cpp:179`) — a partial parse failing at offset 0 still throws
  (SSE lambda converts it to an in-stream error frame, so outcome ≠ bare EOF, but
  the stated goal is only half-met). And the degraded final path uses the last good
  incremental parse — can silently drop the un-parsed tail of generated_text with a
  normal finish_reason, or (continuation+echo case) return prefill only. Fix: raw-
  text fallback whenever the parse doesn't span the full text.
- **C4 (was 5.1):** the D=512 turbo carve-out in fattn dispatch routes
  MHA/batch>2 cases into an MMA path that `GGML_ABORT`s (`fattn.cu:499` →
  `switch_ncols2<512,512>` tail). head_dim-512 turbo models + prefill = abort.
  head_dim 256 unaffected. Fix: require `gqa_opt_applies || n_tokens<=2` for turbo
  at D=512.
- **C5 (was 4.2-4.4):** layer-adaptive machinery: kv-layer count diverges from old
  `n_layer_kv()` semantics on MTP/filtered models (boundary modes shift); head-dim
  fallback mutates ctor params for all later layers + warns only on il==0; adaptive
  mode env is read into a function-local `static` → leaks across target/draft
  contexts. All inert at mode 0/turbo4. Fix before ever using
  TURBO_LAYER_ADAPTIVE in production.
- **C6 (was 6.1-6.4):** CPU backend claims support for 4 turbo types with NULL
  vec_dot/from_float (abort or infinite loop if a turbo op ever lands on CPU —
  e.g. `--no-kv-offload` or the one reachable combo `-ctk turbo3_tcq -ctv f16
  -fa off`); TCQ types missing from FA auto-enable + GQA-warn lists; SET_ROWS
  supports_op tests %64 where turbo3/turbo2 blocks are 128 (wrong invariant, masked
  by kv-cache's %128 fallback); TURBO1_5 missing from CUDA same-type CPY list.
- **C7 (was 1.3):** upstream: file-based slot save/restore touches only the target
  context — with MTP, a restored slot has no draft KV → silent perf cliff (drafts
  re-verify against cold draft state) until re-warmed. Not correctness.

## D. Hygiene / debt (fix before any upstream PR)

- **D1 (was 4.5):** `turbo_rotation`/`turbo_rotation_inv` tensors: allocated,
  filled, re-filled on every clear, exposed via accessors — zero consumers
  (graph uses sign-table WHT + innerq scale only). 128 KiB VRAM per cache + labels
  inverted (R vs R^T comments swapped). Delete or wire.
- **D2 (was 4.7):** InnerQ update hook is compiled against `GGML_USE_CUDA` which is
  (almost certainly) not defined for src/llama-kv-cache.cpp in current llama.cpp →
  stubs selected → scale tensor stays identity forever. Confirms InnerQ inert at a
  deeper level than "uncalibrated". Needs one `nm`/preprocessor check, then: wire
  properly or delete the machinery.
- **D3 (was 4.6):** `#include "turbo-rotation-data.h"` inside two function bodies;
  header has no include guards; 128 KiB .rodata duplicated; adding a guard would
  break clear(). Hoist to file scope.
- **D4 (was 6.5-6.7):** `ggml_turbo_wht` writes op_params via
  `op_params + sizeof(int)` (pointer arithmetic lands on [4] not [1] — all three
  sites agree so it works, but it silently burns slots 1-3); CPU WHT kernel misses
  the group-size assert the CUDA path has; stale "block size 32 / 3.5 bpv" comments
  in ggml-common.h + turbo-quant.cuh (plausible origin of the %64 bug in C6).
- **D5 (was Q6a):** `--spec-draft-n-max N` allocates recurrent state ×(1+N) across
  all 48 DeltaNet layers (`n_rs_seq` plumbing) — document as the hidden VRAM cost
  of raising n_max; 2 is the sweet spot (fast-path rollback preserved).
- **D6:** HANDOFF.md/FUTUREPLAN records saying "slot crash fixed (graceful refusal)"
  are wrong per A2 — corrected in those docs when the fix batch lands. Memory
  updated 2026-08-15.

## E. Verified-clean (worth knowing what held up)

- Q-rotation coverage exact: 5 cached build_attn overloads rotated, no-cache and
  cross-attention correctly bare; inverse-WHT keys on actual v->type (FA + non-FA);
  per-layer Q8_0 promotion stays consistent on both sides.
- op_params slot agreement write↔read (byte 0, defensive clamp in kernels).
- supports_op ↔ dispatch are the same predicate by construction (C4/dispatch-hole
  aside); all FATTN_VEC turbo cases have matching declarations; MMA/TILE prefill
  fallback safe (to_fp16 converters exist for all six types; WHT linearity keeps
  graph-level inverse valid).
- ggml registration: 41/41 traits initialized, block sizes static_assert-backed,
  op tables all at 102, GGML_OP_TURBO_WHT consistent at index 87.
- Type-traits/type-size cross-checks; no zero-init holes in slots 80-85.
- Proxy: owner_lock release is cancellation-safe (async with; no leak path);
  auth ordering correct on both chat paths (key check before body read; exact-match
  public paths = fail-closed); capture failures can't crash the request path;
  client-disconnect cancels upstream and generation stops within a token; heartbeat
  can't split a frame mid-write; no unsynchronized shared state (single-loop
  asyncio); artifacts/reasoning-memory bounded (but see A3's persist-twice: every
  artifact written twice → 20-cap holds ~10 real).
- Sampling precedence + per-field fallback correct on /v1/chat/completions.
- rope_scaling_type cannot be set by GGUF into the CLI field (C2 is about the
  *effective* type, not an injection).
- Drift check: all 8 deployed files byte-identical between ~/.config/llama-tcq and
  turboquant-g1/deploy. Zero drift. (Gap: HANDOFF.md, SYSTEMD.md, systemd units,
  start.sh.bak-qwopus not under version control.)

---

## Improvement plan (proposed, not started)

**Batch 1 — proxy + scripts (no rebuild; needs one restart to deploy):**
A1 slots-dir fix, A3 client-facing error frames + think-buffer flush, A4 passthrough
allowlist, A5 script rewrite (guards, verified stop, log rotation, --api-key-file,
600 keys.json, comment/rollback corrections, binary backup), A6 test suite fix +
maybe_swap_slot test + handler-level smoke tests. Add `--metrics` to launch line
(enables spec_decode_* counters for the grammar/MTP split measurement).

**Batch 2 — C++ (rebuild + battery gate + restart):**
A2 real crash fix (recurrent asserts → throws; upstreamable), C1 shadow deletion,
C2 effective-rope resolution, C3 parse-degrade completion, C4 D=512 guard, C6 CPU
claims minimally corrected (supports_op honesty), D1-D4 hygiene sweep. Then PR
candidates upstream: A2, C2, C3 (+ existing parse-degrade), slot-restore refusal.

**Batch 3 — capability:**
1. Per-request speculative wiring (un-#if0 schema + honor task params in the
   drafting decision) + proxy injects `speculative.n_max: 0` on tools-bearing
   requests → keeps coding 1.7×, removes art-turn penalty automatically.
2. Hermes-side: prefer lazy grammar (tools+auto) over required/json_schema where
   possible — recovers GPU backend sampling for most of the turn regardless of MTP.
3. Phase F coding battery (primary-use quality gate).
4. Nightly canary cron (health + 1-token gen + VRAM + render-tally append).
5. InnerQ: build-check → wire calibration or delete.
6. Slot-restore draft-context warmup after restore (C7) if restore latency matters
   post-A1.
