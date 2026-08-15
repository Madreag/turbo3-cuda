# UPSTREAMSYNC — Full-Accuracy Plan for Rebasing onto Current llama.cpp
(investigated & drafted 2026-08-15; all numbers measured, not assumed)

## Executive summary

We re-base by **fresh clone + layered re-application**, not by git merge —
because merge is structurally impossible (see Ground Truth), and because
re-application is the only strategy where every carried change is an explicit,
reviewable decision. The user's worry — "we'll miss things or it'll be hard
to combine" — is answered three ways:
1. The carry-list is generated **mechanically** (tree-diff vs a pinned base),
   classified file-by-file, so nothing can be silently forgotten.
2. Two pre-identified traps are already neutralized in this plan (GGML type
   collision; superseded patches).
3. Acceptance is **empirical, not diff-reading**: the full regression battery
   we built in Aug 2026 must go green on the synced build before it touches
   production. If we missed something, the battery — which replays the REAL
   captured Hermes traffic — is designed to catch exactly that.

─────────────────────────────────────────────────────────────────────────────
## 1. Ground truth (measured 2026-08-15)

REPO TOPOLOGY
- `origin`  = TheTom/llama-cpp-turboquant (our foundation project)
- `myfork`  = Madreag/turbo3-cuda (our push target)
- `upstream`= ggml-org/llama.cpp
- Our history begins at an "Initial release" **rewritten import**: upstream
  authors' commits are present (Gerganov 1,682 …) but hashes were rewritten →
  **no common ancestor with either origin/master or upstream/master**
  (verified: `git merge-base` fails). Git merge/rebase against upstream is
  therefore IMPOSSIBLE, not merely inadvisable.
- Our own layer: **~164 commits** (Madreag 144 + Erol 20), individually
  enumerable. TheTom's ~165-commit turbo foundation is baked into the import
  invisibly (no Tom-authored commits in our graph).

FRESHNESS
- TheTom's public repo is **stale everywhere**: master 2026-04-20; experiment
  branches (fused-centroid-decode, decode-speed-parity, layer-adaptive,
  asymmetric-kv) all 2026-03-25 … 04-03. His July "MMA decode" results are
  not in his public repo. ⇒ **sync target is ggml-org directly**; TheTom's
  branches are intel for Phase-C-style kernel ideas, not a sync vehicle.
- TheTom's upstream base: `b3d758750` (2026-04-15). Upstream has moved
  **1,631+ commits** since. Raw tree-delta of our HEAD vs that base:
  **723 files, +57,073 / −87,786** (inflated by import-era stripping — CI,
  docs; the functional surface is concentrated: ggml/src 281 files,
  tools/server 141, tools/mtmd 25, src/models 8, gguf-py 4).

WHAT UPSTREAM GAINED SINCE OUR BASE ERA (grep-verified counts)
- MTP/NextN: **60 commits** (mature spec-decode runtime, auto-detect,
  multi-arch) — this is FUTUREPLAN Phase C's payoff.
- qwen3-family: 64 commits (incl. day-0 Qwen3.8 support).
- Gated DeltaNet / GDN kernels: 17 commits (community measured ~30% kernel
  gap at 3.6 launch — free prefill/decode speed for our hybrid models).
- Grammar: 11 commits — **incl. `cd0fa6051` "grammar: degrade max repetition
  >= 2000 to unbounded (#26613)"** (see §3).
- SSM: 21 commits. Plus months of server/parser/mtmd hardening.

─────────────────────────────────────────────────────────────────────────────
## 2. TRAP #1 — GGML type-ID collision (found in planning; would have been
##            a silent-corruption incident mid-sync)

Our custom types occupy enum slots **41–46**:
  41 TURBO3_0 · 42 TURBO4_0 · 43 TURBO2_0 · 44 TURBO1_5 · 45 TURBO3_TCQ ·
  46 TURBO2_TCQ  (GGML_TYPE_COUNT=47)
Upstream has SINCE ASSIGNED **41 = Q1_0, 42 = Q2_0** (their COUNT=43).
A naive re-application keeps our numbers → a Q1_0 GGUF would decode as
TURBO3 KV and vice versa. RESOLUTION (mandatory, mechanical):
- Renumber all six turbo/TCQ types to a **high private range (e.g., 80–85)**
  far above upstream's growth path; single header change + wherever the IDs
  are switch-cased (extract via grep at execution).
- Cost of renumbering: archived KV slot-save files embed type IDs → all
  pre-sync slot saves invalid. Acceptable: slots-long/ is already segregated
  per model era (pre-38-backup/); post-sync starts with fresh slots.
  **No model GGUF ever contains turbo types** (KV-only) — zero file-format
  exposure.
- Execution check: re-verify upstream's enum tail AT SYNC TIME (they keep
  adding types); pick the range after looking, and leave a comment reserving
  it.

─────────────────────────────────────────────────────────────────────────────
## 3. Classification of everything we carry (the three buckets)

BUCKET A — SUPERSEDED BY UPSTREAM → **drop ours, take theirs** (verified):
- The five G1 cherry-picks (SSE #23884/#24281/#24774, grammar fixups
  #24624/#24653) — upstream originals.
- Our auto-parser `zero_or_more` optional-args fix (0b3ce5e42) — upstream
  #26613 degrades ≥2000 repetitions to unbounded at the grammar-parse layer,
  which covers our case AND non-auto-parser grammars. Drop ours.
- Our qwen35 NextN preserve-and-skip port (815b14d61) — upstream has full
  qwen3.8 + MTP-runtime support. Take theirs (it's a superset: theirs RUNS
  the head we merely tolerate).
- VERIFY-AT-EXECUTION list (30 min): whether upstream now has an equivalent
  of our final-parse graceful degrade (server-task update_chat_msg
  try/catch); whether --timeout-style server knobs we set still exist under
  the same names; mmproj/mtmd flag parity for our launch scripts.

BUCKET B — OURS, GENERAL-INTEREST → **PR upstream first, then sync brings
them back officially** (kills long-term carry cost):
- Final-parse graceful degrade (streams must never corpse on a parser throw)
  — IF execution-check shows upstream still lacks it. Clean, small, sellable
  with our corpse forensics as motivation.
- (Nothing else currently qualifies; the grammar fix was pre-empted.)

BUCKET C — OURS FOREVER (the product) → **re-apply deliberately, layered**:
- ggml/src: turbo1.5/2/3/4 + TCQ quant/dequant kernels (CUDA + CPU ref),
  type traits, FA integration points. THE bulk of the 281-file surface.
- Type plumbing: ggml.h enum (RENUMBERED per §2), gguf-py mirrors (4 files),
  ggml type-traits tables.
- KV-cache type wiring in llama.cpp core (-ctk/-ctv acceptance of turbo
  names, llama-kv-cache glue).
- Alpha env-var hooks (TURBO_NORM_ALPHA_V / TURBO4_NORM_ALPHA_V) — these
  live IN the dequant kernels, so they ride Layer 1 with the kernel port,
  not Layer 2.
- Ours-only server files (carry verbatim, no conflict possible):
  tools/server/server-cors-proxy.h, tools/server/webui/.../cors-proxy.ts,
  tools/server/tests/unit/test_proxy.py.
- Other server-side bits that are ours and not upstreamable: (audit at
  execution — likely: nothing beyond the parse-degrade candidate; timeout
  tweak likely now a flag).
- NOT in the repo (immune to the sync, listed so nobody worries): proxy.py
  v6.2 + its 42 tests, start scripts, keys, harnesses in quality-tests/*
  (repo-tracked but ours-only paths — reapply trivially), HERMES docs.

MECHANICAL EXTRACTION (how nothing gets missed):
1. Pin the TRUE base: find upstream commit whose tree best matches our
   import (candidate b3d758750; verify by diff-size minimization over
   upstream commits ±3 weeks around it; 30-min scripted step).
2. `git diff <base> HEAD --name-status` → full 723-file manifest.
3. Auto-classify each file: ours-only path (carry), upstream-changed-too
   (three-way review), deleted-by-import (ignore), generated (ignore).
   Output: CARRY-MANIFEST.md checked into the sync branch — the auditable
   contract. Nothing merges without appearing there with a disposition.

─────────────────────────────────────────────────────────────────────────────
## 4. Execution plan (layered, build-gated)

Layer 0 — Prep (no box time)
  - Fresh clone ggml-org at a pinned recent tag/commit → new worktree
    `turboquant-sync/`, branch `sync/2026-08`.
  - Generate CARRY-MANIFEST.md (§3 mechanical extraction).
  - Bucket-B PR(s) opened upstream (optional but recommended; don't block).

Layer 1 — Types & kernels (the product)
  - Apply renumbered type enum + gguf-py mirrors + traits.
  - Port turbo/TCQ kernels; expect friction ONLY where upstream refactored
    ggml-cuda scaffolding (FA API, type-traits tables — the 281-file area's
    real risk). Build after every sub-step; `test-quantize-fns`-style unit
    coverage if present for custom types (add minimal roundtrip test if not).
  - Gate: build green + a 1-layer roundtrip quant/dequant numerical check
    against the OLD binary's output on identical input (scripted, exact).

Layer 2 — Model/server glue
  - -ctk/-ctv turbo name acceptance and any Bucket B/C server patches that
    survived the §3 audit (alpha hooks already landed with Layer 1 kernels).
  - Gate: server starts, loads BOTH models (3.6 GGUF and 3.8 GGUF — 3.8 needs
    no port anymore, upstream has it), text smoke each.

Layer 3 — Feature parity & config
  - Verify flags used by start scripts all exist (mmproj, chat-template-kwargs,
    reasoning-format, slot-save, parallel, swa-full, yarn args) — rename map
    if upstream renamed anything.
  - MTP spec-decode smoke (FUTUREPLAN Phase C step 1 happens HERE).

Layer 4 — The battery (the actual answer to the worry)
  On the synced binary, full sequence, production untouched:
  1. 42 proxy unit tests (env unchanged — sanity).
  2. REAL-CAPTURE replay (tools_erol_8e811fec) — grammar compiles, tool
     deltas stream, finish+[DONE]. (This exact test caught the grammar bomb.)
  3. Effort-ladder arm: xhigh ×3 + render gate — accept ≥ 2/3 clean
     (small-n guard band vs the 3/4 baseline).
  4. NIAH 130K + 380K — accept effective 5/5 both (known scorer
     false-positive on CTRL documented).
  5. Vision smoke (own-artifact screenshot read-back).
  6. 10-turn soak through proxy — 0 tripwire alerts.
  7. Non-stream degrade check: one non-streaming request with think+fenced
     output — must return 200 with content, never a 500 (this behavior is
     ours via the parse-degrade patch OR upstream's equivalent; whichever
     survived §3, the BEHAVIOR is the acceptance criterion).
  8. Effort-kwarg check: /apply-template renders reasoning_effort
     xhigh/medium/low correctly per-request (Phase E dependency).
  9. 3.6 rollback-pair check: old binary + 3.6 still boots (untouched files —
     this is a 2-minute paranoia check, not a build).
  Gate: ALL green → stage binary, cut over in an idle window exactly like the
  2026-08-14 cutover (backup binary → swap → acceptance run → docs).

DEFINITION OF DONE (one screen, checked off in the sync branch's final commit)
  □ CARRY-MANIFEST.md: every differing file has a written disposition
  □ Type IDs renumbered; upstream enum tail re-verified at sync time
  □ Buckets A dropped / B resolved (PR or carried) / C applied
  □ Layer gates 1-3 green (build, roundtrip numerics, dual-model load, flags)
  □ Layer 4 battery items 1-9 green on the synced binary
  □ MTP spec-decode smoke done (flags confirmed, greedy-identity per
    FUTUREPLAN C)
  □ Production cutover done with acceptance run; rollback inventory intact
  □ Handoff + memory + FUTUREPLAN updated; sync branch pushed to myfork

Rollback at ANY layer: the sync lives in its own worktree + branch; production
binaries and scripts are never touched until Layer 4 passes. Current rollback
inventory stays intact: llama-server.pre-nextn, llama-server.pre-grammar-fix,
build-tcq/, start-long.sh (3.6), slots pre-38-backup.

─────────────────────────────────────────────────────────────────────────────
## 5. Effort & sequencing estimate

- Layer 0: half a day, no GPU.
- Layer 1: the real work — 1-2 focused sessions (kernel-scaffold friction is
  the unknown; everything else is mechanical).
- Layers 2-3: half a session.
- Layer 4: one ~4h babysat box window (same shape as the 2026-08-14 evening).
- Recommended order vs FUTUREPLAN: run FUTUREPLAN A+B first (they tune the
  CURRENT stack and their results — alphas — carry through the sync
  unchanged); sync next; MTP (Phase C) immediately after as the payoff.

## 6. Standing risks & mitigations (explicit)

- R1 ggml-cuda refactor friction (Layer 1): mitigated by sub-step builds +
  the roundtrip numerical gate; worst case = kernel API adaptation work, not
  silent breakage.
- R2 type collision: neutralized by §2 renumbering; verified again at
  execution.
- R3 behavioral drift from upstream parser/sampler changes: caught by Layer 4
  battery (the ladder + replay are behavior tests, not unit tests).
- R4 "we forgot a file": impossible-by-construction — CARRY-MANIFEST covers
  every differing file with a written disposition.
- R5 upstream moves during the sync: pin the target commit at Layer 0; do not
  chase head mid-sync.
