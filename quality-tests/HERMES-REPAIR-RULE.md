# Hermes self-repair rule (R3 deliverable under USER LAW — ready to paste, NOT deployed)

USER LAW 2026-08-08: the model is never constrained, templated, or caged. Full freeform
creativity. Reliability comes from the agent verifying its own work — the behavior
Hermes already exhibited unprompted (it caught and patched its own syntax error).

Add to Hermes's system prompt (verification section):

---
AFTER WRITING ANY RUNNABLE ARTIFACT (HTML page, script, app): verify it before
reporting done. For HTML: open it headlessly or parse it; if the page throws any
JavaScript error or renders nothing, read the exact error, fix your own code, and
rewrite the file. Repeat until it runs clean (max 3 repair rounds). Only then report
completion, mentioning any repairs made.
---

R2 re-measurement under this law: 5 freeform Hermes-shaped pagoda runs; each artifact
headless-rendered; on failure, ONE simulated repair turn (feed the exact console error
back, model rewrites); gate = 5/5 final artifacts pass pixels (brightness >25, ≥8 hues,
0 uncaught). Measures the model doing what it does naturally: create freely, fix its
own bugs when shown them.

R3 deploy: hand the rule to the Hermes agent → one real Hermes run → wire-captured
artifact renders clean. R4: user's three confirmation runs, their words only.
