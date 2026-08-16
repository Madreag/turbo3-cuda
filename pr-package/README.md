# Upstream PR package — staged, NOT submitted (user opens PRs)

Three clean branches off upstream/master (ece963f41, b10448-era), each
compile-gated against pristine upstream (CPU build, llama + server targets).
Pushed to the user's fork (myfork). To submit: fork ggml-org/llama.cpp on
GitHub, push the branch there, open PR with the matching .md as description.

| Branch | Description file | What |
|---|---|---|
| pr/state-restore-hardening | PR-state-restore.md | abort-proof state restore, both caches (production crash fix) |
| pr/ctx-cap-rope-scaling | PR-ctx-cap.md | respect CLI/GGUF rope scaling in slot ctx cap (380K NIAH regression fix) |
| pr/parse-degrade-safety | PR-parse-degrade.md | graceful degrade on chat-parse throw (stream-corpse fix) |

Boundary (user directive 2026-08-15): pushes to user's GitHub allowed; PRs to
other repos are user-initiated only.
