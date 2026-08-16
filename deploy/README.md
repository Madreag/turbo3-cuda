# deploy/ — versioned source of truth for the serving stack

Live copies run from ~/.config/llama-tcq/ (NOT from here). Discipline:
any change to the live proxy or start scripts is copied back here and
committed in the same session ("proxy v6.x" in HERMES-HANDOFF.md tracks it).

Contents: proxy.py (v6.2: owner-lock serialization, thinking reinjection,
corpse tripwires, real-traffic capture w/ image stripping), test_proxy.py
(49 tests; run: pytest test_proxy.py), start.sh (3.6 daily 262K),
start-long.sh (3.6 long 409K), start-long-38.sh (PRODUCTION: Qwen3.8 409K
+ vision), stop.sh, status.sh, watch.sh.

SECRETS ARE NEVER HERE: api.key and keys.json stay in ~/.config/llama-tcq/
only. Scripts read them at runtime via $(cat .../api.key).
