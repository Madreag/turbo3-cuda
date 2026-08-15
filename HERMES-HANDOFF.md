# Hermes × Qwen3.6-27B — Handoff (2026-08-09, post-fix)

Supersedes the 2026-08-08 version of this file (see git history for it) and the
diagnostic threads in HERMES-INTEGRATION-GOAL.md. Branch: `hermes/server-foundation`.

## STATUS: Failure B is SOLVED — root-caused, fixed, deployed, verified.

The "Response remained truncated after 4 continuation attempts" failure was
neither a token cap (our side's early claim — wrong), nor slot preemption or
proxy restarts (the Hermes agent's claim — also wrong for the failing session).
It was a server-side grammar/parser failure chain in the G1 binary:

### The confirmed mechanism (byte-exact evidence in scratchpad pre-fix/ captures
### and ~/.config/llama-tcq/server.log.20260808-grammar-corpse-evidence)

1. **Grammar bomb.** The live Hermes tool suite includes `computer_use` with
   1 required + 48 optional parameters. G1's auto-parser (the #24624 port)
   emitted the optional-args grammar as `(space (arg1|...|arg48)){0,48}` —
   GBNF repetition rewrite multiplies the 48-way choice by 48 ≈ 2304 rules,
   tripping `MAX_REPETITION_THRESHOLD` (2000) in src/llama-grammar.cpp:494 →
   `failed to parse grammar` on EVERY live Hermes request (4/4 that session;
   verified reproducible with a minimal 2-tool request). The quality-test
   battery's 38-tool replica lacks the 48-param monster — its grammar compiled,
   which is why batteries were clean while the user bled. NOT preemption:
   trace analysis shows all 4 failing streams ran with zero concurrent traffic,
   zero restarts, owner_lock held; llama-server released each slot normally.

2. **Unprotected generation.** Grammar dead → sampler chain built with no
   grammar stage → nothing constrains the model's tool-call syntax. The 33K
   Hermes system prompt teaches JSON-style `<tool_call>{"name":...}` calls;
   the template/parser expect XML `<function=...><parameter=...` style, with
   required args in strict definition order. Unconstrained, the model follows
   the prompt (JSON args, or reordered parameters — our captured corpse wrote
   `content` before `path`) → the PEG incremental parser consumes the tool-call
   open, then JAMS: one tool_calls delta, then silence (heartbeats while the
   GPU generates at full speed), everything accumulating unconsumed.

3. **The corpse.** At end of generation the final parse (is_partial=false,
   server-task.cpp update_chat_msg → common_chat_parse) THROWS
   `Failed to parse input at pos N: ...` (chat.cpp:1771). The streaming path
   emits that exception as a terminal in-stream error frame —
   `data: {"error":{"code":500,"message":"Failed to parse input at pos 402:
   <tool_call>...<full raw generation embedded>...","type":"server_error"}}`
   — then closes with **no [DONE], no finish_reason**. That is the giant
   "keyless" 20-23KB terminal chunk in stream-trace.log, and it is the exact
   answer to the Hermes agent's question about terminal frames. Hermes
   correctly flags truncation, retries with its ~8K write-splitting nudge, the
   model dies the same way (7,826-token corpse = task 54273, eval'd at full
   48 tok/s and released `truncated = 0` — a parse corpse, not a cut), ×4 →
   the user-facing error. The same throw is the known non-streaming 500
   ("Failed to parse input at pos 22: <think>") — one bug, two faces.

### The fix (deployed 2026-08-09 ~00:20, commit on this branch)

- **Root fix** — common/chat-auto-parser-generator.cpp: optional tool args now
  emit `(space (arg1|...|argN))*` via `p.zero_or_more` instead of
  `p.repeat(..., 0, N)`. `*` renders as a single recursive GBNF rule — no
  repetition rewrite, no threshold, identical accepted language ({0,N} never
  enforced per-arg uniqueness). The 48-param tool now compiles; grammar stays
  ALIVE on live Hermes requests, which forces correct XML syntax and arg order
  (grammar-constrained decoding masks divergent tokens), which keeps the
  incremental parser streaming tool deltas, which makes proper
  `finish_reason` + `[DONE]` termination structurally guaranteed.
- **Safety net** — tools/server/server-task.cpp update_chat_msg: a final-parse
  throw can no longer corpse a response. is_partial=false failures degrade to
  the last good incrementally-parsed message (streams: terminate properly with
  finish_reason + [DONE]; non-stream: raw text as content instead of a 500),
  with a loud SRV_WRN. Streaming partial-parse behavior unchanged.

### Verification (all on the patched binary, direct :8131 captures)

- RED (pre-fix, preserved in scratchpad pre-fix/): minimal 2-tool repro trips
  the grammar guard; JSON-bait sysprompt run reproduces the user's exact
  signature — 84 content frames, 1 tool frame, 30s silence, 4.7KB terminal
  error frame, EOF without [DONE].
- GREEN (post-fix): same JSON-bait request — grammar compiles (0 failures in
  server.log), 1753 streaming tool_calls deltas, no silence, terminal
  `finish_reason:"tool_calls"` + `[DONE]`, zero error frames.
- Battery regression: hermes_shaped_battery preverify via proxy — result
  recorded in quality-tests/niah_results (see latest preverify entry).
- Stop-string mid-call cutoff (`stop: ["</parameter>"]`) verified graceful
  BEFORE the fix (finish_reason:"tool_calls" + [DONE]) — plain truncation was
  never the corpse trigger; syntax divergence was.

### For the Hermes-side agent (their question, answered with bytes)

Terminal frame of the truncated streams = in-stream `data: {"error":{...
"type":"server_error"}}` then bare EOF. No [DONE], no finish_reason, no cap
(max_tokens 131072 confirmed arriving; generations died at 9767/2153/7826/1016
tokens, all `truncated = 0`, all slot-released normally). Their preemption /
proxy-restart theory: disproven for the failing session (no concurrent
traffic, no restarts in window, owner_lock covers the full stream lifetime) —
but their insistence on the terminal-frame evidence was correct methodology
and is what cracked it. Their two upstream nits stand: the "~8K" continuation
nudge teaches write-splitting for a failure class where it can't help, and
continuations re-pay full prefill.

## STACK STATE (live now — updated 2026-08-14, Qwen3.8 cutover)

- **PRODUCTION IS NOW Qwen3.8-27B** (`models/qwen38/Qwen3.8-27B-Q6_K.gguf`,
  arch qwen35, embedded NextN/MTP block — needs binary ≥ commit 815b14d61).
  Profile: `start-long-38.sh` (409600 / YaRN 1.5625 / turbo4 / alphas
  1.10-1.12 / temp 1.0 / reasoning_effort xhigh by default). Validated
  2026-08-14: effort ladder xhigh 3/4 clean renders (medium 2/4, low 0/2,
  xhigh@t0.6 1/2 with one 131K-cap truncation); NIAH effective 5/5 at 130K
  and 380K (scorer's CTRL "FABRICATED" = false-positive, model refuses
  correctly while quoting the Meridian code — fix the scorer someday);
  proxy-path acceptance 1/1 clean render, 0 tripwire alerts. Costs to know:
  xhigh art turns ≈ 40-100K tokens / 20-40 min; ~1/12 runs brushes Hermes's
  131072 max_tokens mid-think (finish=length, properly terminated — knob is
  Hermes-side); 3.8 detours to skill_view/bash before write_file (real
  Hermes loops handle this; single-shot harnesses must be loop-tolerant).
  Battery harness max_tokens raised 30000→131072 to match the real wire.
  Rollback: `stop.sh && cp build-g1/bin/llama-server.pre-nextn
  build-g1/bin/llama-server && start-long.sh` (3.6 slot saves archived in
  slots-long/pre-38-backup/).
- Previous model (rollback pair): `/home/erol/ai/turboquant/models/Qwen3.6-27B-Q6_K.gguf` (unsloth, qwen35).
- Server: `build-g1/bin/llama-server` — G1 + tonight's two patches. Built from
  worktree `/home/erol/ai/turboquant/turboquant-g1` (branch checkout), staged
  into `turboquant-kv-cache/build-g1/bin/`. Rollbacks:
  `build-g1/bin/llama-server.pre-grammar-fix` (last night's G1), `build-tcq/`
  (pre-G1, untouched).
- Live profile: `~/.config/llama-tcq/start-long.sh` (409600 ctx, turbo4/turbo4,
  YaRN 1.5625, alphas 1.10/1.12, --parallel 1). Daily: `start.sh`.
- Proxy: v6.1 (owner_lock verified to hold for the entire request lifetime —
  proxy-mediated preemption is impossible). 41/41 unit tests. New in v6.1:
  (a) TRIPWIRES — any relayed in-stream error frame or stream ending without
  [DONE] logs `[ALERT] ...` to proxy.log + `ALERT` trace lines; validated
  against the real 2026-08-08 corpse bytes (fires both) and a healthy stream
  (fires neither). A future corpse announces itself instead of hiding.
  (b) REAL-TRAFFIC CAPTURE — every tools-bearing request body is persisted
  pre-mutation to `~/.config/llama-tcq/captures/tools_<user>_<suitehash>.json`;
  regression gates must replay the live Hermes capture, never a hand-built
  replica (the replica gap is what hid the grammar bomb). Soak on the patched
  stack: 20/20 turns PASS, 0 alerts.
  KNOWN INSTR BUG unchanged: `_partial` artifact files also fire on clean runs;
  cut signal is "no END frame" — WHICH MEANS handler exception (client-leg
  reset or proxy death), NOT upstream EOF: an upstream bare-EOF still writes
  END. Trace END + client-visible missing [DONE] = in-stream error frame.
- Client: `http://192.168.50.130:8130/v1`, keys in keys.json (users: erol,
  brother). Windows portproxy for :8130 (WSL 172.17.154.123; re-add on rotate).

## KNOWN OPEN ITEMS (none block Hermes art sessions)

- PEG required-args parsing is definition-order-strict; only the (now working)
  grammar makes order safe. If grammar is ever disabled again, reordering jams
  return. Upstream-worthy: order-flexible required-arg parsing, and a visible
  per-request warning when a tool grammar fails to build (today it's one log
  line and silent unconstrained generation).
- Between-instance quality variance (Failure A territory): 50-75% clean rate
  spread, one sick instance cured by restart. Discriminating test still to run:
  cache-purge vs restart. Mitigation: restart server before art sessions.
- Non-streaming think+fenced 500: now degrades to raw-content 200 via the
  safety net; the true parser fix still rides the next upstream sync.
- HERMES-REPAIR-RULE.md: written, NOT deployed. USER LAW unchanged: no
  model-caging; config-only levers.
- 4591828-class no-END traces (client-leg resets, e.g. ~22:49 that evening):
  consistent with user-side aborts; not implicated in the solved failure. If
  they recur without a user abort, suspect the Windows portproxy leg.

## VERIFICATION TOOLING

- Everything from the previous handoff (hermes_shaped_battery.py,
  longctx_battery.py, render_arm.sh, pw_verify.py, bare_arm.py, ollama_arm.py)
  plus this session's corpse harnesses in the session scratchpad:
  `repro_corpse.py` (minimal grammar-bomb), `corpse_force.py` (stop-string
  cutoff), `corpse_json.py` (JSON-bait, the deterministic user-condition
  reproducer) — pre-fix captures under `pre-fix/`.

## OPERATING LAWS (unchanged, hard-won)

- USER STOP overrides everything. Kill test work, never production; park; quiet.
- Never `pkill -f`/`pgrep -f` a pattern present in your own command line;
  pgrep -x / socket-owner / bracket-classes; kill and relaunch in SEPARATE calls.
- Render evidence = pixels + JS console only; `timeout 45` headless Chrome.
- Bash hard-kills ~120s: setsid + background waiters for anything long.
- Don't run GPU work while the user may be using Hermes (queueing delays them
  even though decapitation is disproven). Single slot.
- Warm page cache before timing; Windows VRAM volatile; CUDA 12.8; --no-mmap.
- Model quality is proven good. The model was never the bug. Don't cage it.
