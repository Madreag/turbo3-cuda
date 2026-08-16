# TurboQuant Serving Stack — Handoff (2026-08-15, board-complete)

Supersedes the 2026-08-09 Failure-B edition (git history holds it; its
grammar-bomb forensics remain valid record). Branch: `hermes/server-foundation`
(this repo) + `sync/2026-08` (the llama.cpp fork, remote `myfork`).

## CURRENT STATE — one screen

- **Model:** Qwen3.8-27B Q6_K (hybrid: 48 DeltaNet + 16 attention layers,
  head_dim 256, GQA-4), native MTP head. Vision via mmproj **on CPU**
  (~21 s/image encode, once per image; text speed unaffected).
- **Context:** 327,680 (320K), YaRN 1.25. LOCKED for the testing phase (user
  decision): guarantees ~1 GB VRAM headroom so no measurement is paging-
  poisoned. Post-testing options: push ctx up; optional MTP-off profile
  (~0.8 GB back) as a user-selectable trade.
- **KV cache:** turbo4 (66-byte blocks, 4.125 bpv, corrected Lloyd-Max
  centroids) K+V, target AND draft (`-ctkd/-ctvd turbo4`). Alphas 1.00.
  MMA-turbo fused decode kernels (kill-switch `GGML_TURBO_MMA_FUSED=0`).
- **Speculative:** `--spec-type draft-mtp --spec-draft-n-max 2`. Acceptance
  82.5% greedy / ~67% at temp 1.0 (post-#27133 queue redesign).
- **Fork:** upstream-tip b10448 + carries. Binary: `build-g1/bin/llama-server`
  (= gdn22587 build: 24565 + fused-MMA-default + GDN row-per-warp #22587;
  rollback chain: `.pre-gdn22587` → `.mainline` → `.pre-bughunt`). Fused
  MMA-turbo decode is ON via launcher env (2026-08-15 adoption: +2.2% @38K,
  +8.7% @121K decode); GDN #22587 adopted same day (+2.8% decode @38K,
  +1.7-1.8% prefill both depths, +0.4% @121K).
- **Measured (fitted, unpaged, post-adoptions, greedy probes):** 38K depth
  ~108 decode / ~2,720 prefill; 121K depth ~77 decode / ~1,715 prefill.
  (Older "84-89 @38K" figures were VEC-path temp-1.0 probes — superseded.)
  KLD vs archived f16 ref: 0.0050-0.0059 / top-1 96.7-98.3% (same-day binary
  pair; the archived 0.00473 predates b10448+24565 binary evolution).
  Battery @64K on the shipped binary: 6/6, ledger 8/8 (traj_gdn64.json).
  VRAM 31.1-31.7 of 32.6 GB.

## OPERATE

```bash
bash ~/.config/llama-tcq/start-long-38.sh   # guarded: refuses double-start,
                                            # rotates logs, health-gates both layers
bash ~/.config/llama-tcq/stop.sh            # verified kill, port check
bash ~/.config/llama-tcq/status.sh          # ports + /health + VRAM (not pidfiles)
```
Clients: `http://192.168.50.130:8130/v1` (OpenAI) or `/v1/messages`
(Anthropic), keys in `keys.json`. After a Windows reboot re-add the portproxy
if clients can't reach 8130 (WSL IP rotation).

Rollback: stop → `cp build-g1/bin/llama-server.pre-gdn22587 build-g1/bin/llama-server`
→ start (drops only the GDN #22587 kernel). Deeper: `.mainline` (also drops
24565), `.pre-bughunt`. Fused-MMA kill-switch: GGML_TURBO_MMA_FUSED=0 env
(no binary swap needed).
**Slot files are config-specific** — archive `slots-long/*.bin` on any config
change (server refuses stale ones gracefully; proxy erases and re-prefills).

## THE VRAM LAW (hard-won 2026-08-15)

Budget = weights (20.8 GB) + KV (17.5 KiB/token incl. draft) + recurrent ×(1+n_max)
+ **THREE compute scratches** (target/draft/vision — scale with ctx×batch)
+ ~250 MB meta. The stack ran silently WDDM-paged for two days because draft-KV
(f16 default!) and the triple scratch were never budgeted — `nvidia-smi` pins at
the residency cap and HIDES overcommit ("config changes don't move the number" =
you are paged). Keep ≥1 GB free. Context-fill does NOT grow VRAM (static
prealloc; checkpoints live in host RAM, ~8 KiB/token of depth, cap 2).
Ceiling behavior is graceful: `finish_reason: length`, over-cap prompts → 400.

## QUALITY GATES (all in quality-tests/)

- `kl_divergence.py` — 2048-token prompts, cache_prompt=false, vs the ARCHIVED
  same-model reference `kld38/kld_logprobs_f16_qwen38_yarn.json` (a stale
  cross-model reference produces false-catastrophic numbers — 3.6-era file is
  quarantined). Baseline: 0.00473 / 98.3%.
- `trajectory_battery.py` — the agentic axis. Baseline
  (`trajbase/traj_b10448-320k-baseline2.json`, seed 42): hops-2/3/4,
  correction, executable code-traj = PASS at 64K/128K/256K; **ledger
  (state-tracking) is the discriminative cliff: 8/8@64K → spiral@128K →
  4/8@256K**. Any quant/kernel change must hold the perfect columns and not
  lower ledger.
- Depth probe: `scratchpad/depth_decode_test.py` (recreate from WORKPLAN if
  /tmp wiped). **Paired-run law:** single-arm vs historical baseline is
  invalid on this box (echo noise ±20%, environmental confounds); A/B =
  alternating binaries, minutes apart, ordering must repeat.

## PROXY (deploy/proxy.py, v6.4, 49/49 tests via `python3 test_proxy.py`)

Per-user keys → slot pinning with save/restore (now actually effective:
checkpoint fix made post-restore reuse ~49 tokens instead of full re-prefill);
think-tag extraction (literal tags after content start are preserved); stream
tripwires WITH client-facing repair (truncation → error frame + [DONE] /
error event — never a silent "finished" corpse); heartbeats both protocols;
circuit breaker (fast 503 while server restarts); passthrough allowlisted
(/slots and raw completion endpoints blocked); Anthropic images translated
to image_url. Known accepted gaps: Anthropic-path thinking-memory;
lock-acquire has no timeout (by design: two-user serialization).

## DOC MAP (authority order for "what happened / what's next")

1. `WORKPLAN-BESTAPP.md` — the marathon ledger: every verdict with numbers
   (adopted / tested-not-needed / closed), VRAM autopsy, paired-run rule.
2. `SPARSE-DECODE-BUILD.md` — the sparse-decode P0 execution ledger and
   CLOSED verdict (tested-not-viable + the fused-gate discovery);
   `SPARSE-DECODE-DESIGN.md` is its superseded design record.
3. `BUGHUNT.md` — 2026-08-15 audit ledger (proxy/scripts/C++ fixes, all shipped).
4. `RESEARCH-2026-08.md` — the research campaign board (several speed verdicts
   superseded by later paired re-tests — WORKPLAN wins on conflict).
5. `FUTUREPLAN.md` / `UPSTREAMSYNC.md` — historical (phases done; sync done).
6. `CLUB3090-PORT-BOARD.md` — the standing NEXT queue: ports/tests/skips
   from the club-3090 review, with per-item validation gates and waves.
7. `pr-package/` — three upstream-ready PR branches staged on the user's fork
   (state-restore hardening, ctx-cap rope scaling, parse-degrade). User opens
   PRs; assistant never submits to external repos.

## OPEN ARCS

- **Sparse decode: CLOSED 2026-08-15 — tested, not viable** (P0 offline
  validator: even oracle selection reads 30-50% of cache on this hybrid;
  evidence in SPARSE-DECODE-BUILD.md). Collateral fix ADOPTED: prod was
  silently running fused-MMA OFF — launcher now exports
  GGML_TURBO_MMA_FUSED=1 (in-tree default also flipped on branch
  feature/sparse-decode). Re-run the P0 probe (~30 min) before believing
  the sparse verdict for any future model swap.
- **NEXT QUEUE = CLUB3090-PORT-BOARD.md** (2026-08-15 review of the
  club-3090 community repo): Wave 1 = two-stage ngram+MTP depth sweep
  (P1, top lever — includes the fused-Q≤4 × draft-depth interaction
  analysis), tail-KLD metric + turbo4-vs-q8_0-vs-q4_0 three-way (P2),
  rollback-clamp audit (P5), launcher hardening (P6). Wave 2 = fill-ladder
  probe (P3, MANDATORY before any ctx push) + VRAM-law calculator (P8) +
  agentic-turns probe (P7).
- Post-testing-phase: context push + optional MTP-off profile (gated on P3).
- **GDN #22587: ADOPTED 2026-08-15** (un-parked, merged with b10448
  snapshot-slot semantics, all gates green — see WORKPLAN GDN section).
- MTP acceptance note (reframed by the club-3090 review): greedy inflates
  spec acceptance ~2× cross-engine; our 82.5% greedy / 67% temp-1.0 is
  expected, the "Vulkan 92%" reference is presumed greedy. Lever = tuning
  (board P1/P4), not a mystery.
- Watchlist: upstream issues 27090/27102/26609/25717 (our shapes); Vulkan
  92%-acceptance reference gap.

## OPS LAWS (unchanged, blood-signed)

USER STOP overrides goals. No pkill patterns matching your own cmdline; kill
by pidfile. No background llama-bench/perplexity. One GPU workload at a time,
babysat. Never run llama-cli non-interactively. Keys never enter the repo.
120s Bash guillotine: long jobs → run_in_background. Loads ≠ hangs (cold disk
124 MB/s). Render truth = pixels + console only.
