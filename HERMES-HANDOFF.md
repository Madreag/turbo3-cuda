# Hermes × Qwen3.6-27B — Handoff for the next agent (2026-08-09)

Read this whole file before touching anything. It supersedes the diagnostic
threads in HERMES-INTEGRATION-GOAL.md (which remains the historical goal record).
Branch: `hermes/server-foundation`. All code/harnesses/evidence committed there.

## THE HEADLINE: two DISTINCT failure modes were conflated all session. Keep them apart.

**Failure A — model emits broken JS (a RENDER problem).**
The model writes a fatal JS slip (undeclared var, malformed syntax, unbounded loop)
in a fraction of giant one-shot artifacts → black/dead page. VERIFIED by headless
console (e.g. `dy is not defined`, `Unexpected token ']'`). Intrinsic to the model,
config-modulated. Measured clean-rate on OUR best config ≈ 50-75% per server instance
(pooled healthy ≈ 61%); ollama-as-shipped = 1/8 (12.5%). Conclusion: "works for
everyone else" is survivorship; ours is the best-measured setup anywhere. This mode
is DATA-COMPLETE. Not the current user-facing bug.

**Failure B — stream dies mid-generation (a TRANSPORT problem). THIS IS THE OPEN BUG.**
User's current error: "Response remained truncated after 4 continuation attempts."
Mechanism (from the Hermes session transcript 20260808_231216_49cba6, confirmed):
 1. A streaming tool-call response is CUT mid-flight — bare EOF, no `[DONE]`, no
    finish_reason.
 2. Hermes correctly detects truncation and runs its designed recovery: partial-stub +
    a continuation prompt that says "break into smaller tool calls, keep args under ~8K
    tokens" (Hermes `conversation_loop.py:689`).
 3. The model obeys, writes a smaller chunk — the stream is cut AGAIN.
 4. After 4 rounds Hermes gives up (`conversation_loop.py:3120`) → the error string.

### My earlier misdiagnosis (owned, do not repeat)
I claimed Hermes capped output at ~8,192 tokens because a request "stopped at 7,826."
WRONG on the central fact. Byte-exact wire capture shows Hermes sends
`max_tokens: 131072`. The 7,826 was the model COMPLYING with Hermes's "~8K" retry
nudge and dying anyway — a corpse, not a cap (caps stop at round numbers). Raising
max_tokens changes nothing. The Hermes-side agent was correct.

### Evidence already gathered for Failure B
- `stream-trace.log` classifier (this handoff's session): 100 clean-END streams, and
  req `4591828` = 2124 chunks then **bare EOF, no END frame** = a genuine mid-stream
  cut. That is the deciding datum: cuts are real; they are not cap hits.
- Earlier, two of my own replay streams died clean-EOF at 9.7s and 117s — non-
  deterministic cut points → an external hand, not a timer.
- Hermes watchdog EXONERATED by the Hermes agent: every content/tool/reasoning delta
  resets its stall timer; LAN IP (192.168.x.x, RFC-1918) → "local" → 900s threshold,
  never approached by a 47-delta/s stream.

### Prime suspects for the cut (fix on THIS side — proxy/server)
1. **Single-slot preemption (most likely).** Server runs `--parallel 1`. If a second
   request hits the box while a Hermes stream is generating, the slot can preempt and
   decapitate the in-flight stream. FOR HOURS this session I ran test batteries
   concurrently with the user's live Hermes use — my batteries were very likely
   preempting their streams (my clean test artifacts = the preemptor's view; their
   blank/"snow" pages = the victim's). This alone may explain most user-visible pain.
2. **Proxy restarts mid-stream.** I redeployed proxy.py many times this session; every
   restart clean-EOFs in-flight streams.

### THE CONFIRMING EXPERIMENT (run this first — ~2 min, cheap, definitive)
Start one long streaming request; ~3s in, fire a second request at the same server;
watch whether the first dies with a bare EOF. If yes → preemption confirmed → fix is
proxy-side queuing/serialization. Repro harness pattern is already in
`quality-tests/hermes_shaped_battery.py` (stream loop) — just launch two at once and
diff their terminal frames in `stream-trace.log` (END vs no-END).

### Fix direction for Failure B (this side)
- Make the proxy SERIALIZE at the slot: hold the owner_lock so a second request cannot
  reach llama-server until the first stream completes (it already has owner_lock for
  cross-user; verify it covers same-user concurrent + all test traffic). OR
- Move to `--parallel 2` + proper slot routing so concurrent requests get separate
  slots (costs KV VRAM; at 400K ctx it won't fit — would need reduced ctx). OR
- Detect a cut upstream and retry the WHOLE call transparently rather than letting
  Hermes do write-splitting continuation.
- Two legit Hermes-side nits (upstream, not blockers): its continuation prompt's "~8K"
  advice is wrong for this failure class (teaches write-splitting, contradicts one-shot
  artifacts) — a truthful nudge is "stream was cut, retry same call"; and each
  continuation re-sends full history (re-pays prefill — brutal at high ctx).

## STACK STATE (all live now)
- Model: `/home/erol/ai/turboquant/models/Qwen3.6-27B-Q6_K.gguf` (unsloth, arch qwen35).
- Server binary: `build-g1/bin/llama-server` (G1 = old build-tcq + 4 upstream SSE
  commits #23173/#23884/#24281/#24774 + timeout 3600 + grammar fixups #24624/#24653).
  Rollback: `build-tcq/` untouched.
- Live profile: `~/.config/llama-tcq/start-long.sh` — `-c 409600 -ctk turbo4 -ctv turbo4
  --rope-scaling yarn --rope-scale 1.5625 --yarn-orig-ctx 262144 --temp 0.6 --top-p 0.95
  --top-k 20 --chat-template-kwargs '{"preserve_thinking": true}'`, alphas
  TURBO_NORM_ALPHA_V=1.10 / TURBO4_NORM_ALPHA_V=1.12. Daily profile: `start.sh` (262K,
  q8_0 K + turbo4 V). This IS the best-measured config; don't change it to chase Failure A.
- Proxy: `~/.config/llama-tcq/proxy.py` (v6). Features added this session: return_progress
  injection→SSE-comment, no-cancel heartbeat, ServerDisconnected retry, preserve_thinking
  reasoning-memory, client_max_size 64MiB (fixes 413 on >230K sessions), artifact
  byte-capture to `~/.config/llama-tcq/artifacts/`, `stream-trace.log`. 37/37 unit tests.
  KNOWN INSTR BUG: `_partial` artifact capture fires in finally on clean runs too →
  `_partial` trace entries are NOT proof of cuts; use "no END frame" as the cut signal.
- Client: `http://192.168.50.130:8130/v1`, key in `~/.config/llama-tcq/keys.json`,
  model name any string (passthrough; use `qwen3.6-27b`). Windows portproxy + firewall
  for :8130 live (WSL IP 172.17.154.123 — re-add portproxy if it rotates on reboot).

## KNOWN OPEN ITEMS
- Failure B fix (above) — THE priority.
- Between-instance quality variance on identical config: 4 fresh instances measured
  50-75% clean, plus one sick instance at 0/6 (statistically impossible; cured by
  restart; mechanism undetermined). Discriminating test: cache-purge vs restart on a
  sick instance. Pragmatic mitigation already known: restart server before art sessions.
- Non-streaming /v1/chat/completions 500s on G1 binary parsing Qwen3.6 think+fenced
  output ("Failed to parse input at pos 22: <think>"). Streaming path unaffected.
  Candidate fix rides next server-code sync.
- HERMES-REPAIR-RULE.md: sanctioned freeform+self-repair rule for Failure A, written,
  NOT deployed (raises art reliability without caging the model). USER LAW: never
  constrain/template/cage the model; config-only levers.

## VERIFICATION TOOLING (all in quality-tests/, committed)
- `hermes_shaped_battery.py` — 33K-sysprompt + 38-tool preverify/soak harness
  (HERMES_SAMPLING=creative|precise, HERMES_RECIPE modes). The realistic Hermes shape.
- `longctx_battery.py` — multi-needle + 3-hop + control NIAH at any ctx.
- `render_arm.sh <label> <n> [srcdir]` — headless Chrome (Windows, `timeout 45`) +
  pixel gate (brightness>25, hues>=8, colored>=8%, blob>=1%, 0 uncaught/SyntaxError).
  RENDER_SKIP=1 = analyze pre-staged shots.
- `pw_verify.py` — playwright desktop+mobile render truth (console + pageerror + pixels).
- `bare_arm.py`, `ollama_arm.py` — chat-shape and independent-runtime arms.
- Python venv w/ numpy+pillow+playwright in the session scratchpad (recreate if gone).
- ollama installed (needs zstd); service DISABLED — start only for comparison arms.

## OPERATING LAWS (hard-won; violating these cost hours)
- USER STOP overrides any goal/automation instantly. Kill test work (never production),
  park state, go quiet.
- Process mgmt: NEVER `pkill -f`/`pgrep -f` a pattern in your own command line (self-
  match; struck 6×). Use `pgrep -x`, socket-owner (`ss -tlnp | grep :PORT`), or brackets
  `name[_]pat` — AND keep kill and relaunch in SEPARATE Bash calls.
- Render evidence = pixels + JS console ONLY. File size / closed tags prove nothing.
  Always `timeout 45` headless Chrome (unbounded-loop artifacts hang virtual-time).
- Bash tool hard-kills at ~120s; use setsid + Monitor for anything long; one action/call.
- Don't run GPU batteries while the user may be using Hermes — you'll preempt them
  (this is literally Failure B). Single slot.
- Warm page cache before timing; Windows-side VRAM is volatile (transient OOMs);
  CUDA 12.8 only; -mmp 0 / --no-mmap always.
- Model quality is PROVEN good (best-in-field). Do not "fix" the model. Fix transport.
