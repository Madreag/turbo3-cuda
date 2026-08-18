# OVERNIGHT REPORT — 2026-08-17/18 (omega run, stock clocks)

LIVING DOCUMENT — updated at each milestone through the night; mega-analysis
lands at the top by morning. Raw data: quality-tests/yarnB/ (ledger.txt =
ground truth of progress), quality-tests/kldnvfp4/ (quant gates),
quality-tests/quant-lab/ (imatrices, recipes).

## ROOT CAUSE — FINAL FORM (~04:30, machine-code-verified) — READ THIS FIRST

**The crash class = PDL launch overlap x restrict-licensed memory semantics in
the recurrent hot path.** Two concrete surfaces, both verified IN THE COMPILED
SASS of the crashing binary:

1. **Non-coherent (stale-cache) loads on the tightest producer-consumer chain
   in the graph.** The racy binary's GDN kernel loads ALL inputs via
   `LDG.E.128.CONSTANT` (non-coherent/texture path — licensed by
   const+__restrict__). Under PDL, the kernel starts BEFORE its predecessor
   finishes; NC loads may serve stale lines for data written microseconds
   earlier. The GDN state tensor is re-read/re-written by consecutive kernels
   at the same addresses every step, x48 layers x every batch = millions of
   razor-thin windows/hour. Upstream's own rule (#24030: "Avoid PDL race
   conditions by disabling __restrict__ when PDL is used") bans exactly this
   combination.
2. **Hoist/wild-address surface on index-driven kernels.** `k_get_rows_raw`
   (embedding gather, every token) carried entry restricts on its INDEX
   pointer under PDL — a hoisted stale index = wild gather address = the
   device-fault class. Present upstream too (user's PR candidate).

**Why every prior theory fell**: deterministic replays pass (razor-thin
non-deterministic window), op-tests pass (single-op tests never overlap
kernels), memtest passed (no PDL chains), OC removal didn't help (mechanism
is clock-independent), thermal theory dead (Aug-14-15 sustained loads were
stable — the GDN kernel joined the PDL party Aug 15, day one of instability).

**THE FIX STACK (all shipped tonight, in the binary + launchers):**
- `GGML_CUDA_PDL=0` in every launcher/gate/orchestrator = the COMPLETE
  mitigation (kills both surfaces at the launch level; serialized kernel
  boundaries restore normal flush/invalidate semantics). THE load-bearing fix.
- Binary `.restrictfix`: GDN + k_get_rows_raw entry restricts removed
  (closes the hoist/wild-address class structurally; upstream's pattern;
  also makes 4 of 10 GDN loads coherent). Tree-wide entry-grain audit: ZERO
  remaining violators (machine-checked, script committed).
- A "zero-cost restrict-locals" variant was BUILT, SASS-DISSECTED, found to
  still emit 10/10 NC loads, and REVERTED — the paper trail is in git.
  SASS verifier committed: quality-tests/yarnB/verify-sass.sh.
- Fallback binary C `.gdnmainline` (kernel revert) + vacation watchdog
  (disarmed) complete the depth.

**PERFORMANCE (user directive: lose none):** PDL=0 costs ~1-3% class launch
overlap; entry-restrict removal ~0-1% on one kernel. BOTH get measured in the
morning battery vs our recorded baselines; any real regression gets hunted
with the bigger levers queued (fused-verify window ~5-15%, ub sweep). The
crash cost was 100%.

**FINAL FORM (~05:40) — ZERO-COMPROMISE, MACHINE-VERIFIED:** the residual 6
NC loads traced to ggml_cuda_memcpy_1's INTERNAL __restrict__ (upstream
helper). Replaced with a kernel-local coherent load path for predecessor-
written data. Result, verified in SASS of the PROMOTED prod binary:
- NC loads: **0 of 10** in every GDN variant; k_get_rows_raw 0; FA path 0.
- Instruction count: **256 = 256** (old vs fixed) — byte-identical size,
  the ONLY change is load cache policy on single-use data = zero cost.
- PDL plan: env stays 0 for phase 1; phase 1b certifies PDL back ON
  (safe now: sync-first + fully coherent fresh-data loads) => the 1-3%
  overlap returns => NET PERFORMANCE LOSS: ZERO. User directive honored.

**GUARANTEE LADDER:** [done] mechanism identified in machine code ->
[done] fixes shipped in binary+launchers -> [PENDING REBOOT] execution proof:
morning-protocol.sh phase 1 (killer workload x2 on the fixed stack) ->
phase 3 (promote + full battery soak). The GPU is physically absent until
the reboot; the proof fires the moment it returns.

## EXECUTION DAY LOG (2026-08-18, user at work)

- ~05:30 GPU RECOVERED on warm reboot (5090, PCIe5, clean idle). Day begins.
- PHASE 0 op-tests: first attempt showed 1/2+FAIL but tail-only capture
  discarded the detail (instrument lesson: gates keep FULL logs). Re-runs:
  GDN 36/36, GET_ROWS 111/111, FULL SUITE 13,253/13,253 GREEN. Classified:
  post-reboot first-touch transient; watch for recurrence. PHASE 0: PASS.
- PHASE 1: **MACHINE DIED 05:59 (~2 min into run 1) + residual 06:03 boot
  bugcheck — both 0x116. FIX-AS-ROOT-CAUSE REFUTED at the execution gate.**
  Three-way elimination now: clocks (OC removed->died), our software (PDL
  off + restrict-clean + 13253 green->died), and the user's counter-evidence:
  GAMES RUN STABLE AT HIGHER POWER (graphics path) — every death is on the
  CUDA-compute + WSL dxg path games never touch. Surviving suspects:
  WSL2 dxg layer / driver compute mode / b10448-era base beyond our fixes.
- DISCRIMINATION TESTS (2026-08-18 ~11:30):
  TEST A = OFFICIAL native Windows b10488 build, same killer workload, zero
  WSL, zero fork code. Native survives -> WSL/dxg or fork-base convicted.
  TEST B = .pre-sync binary (pre-b10448, no PDL infra, no GDN/MMA) + Qwen3.6
  under WSL. Survives -> new-base convicted -> bisect.
  DEADMAN: if this machine is found dead, the ledger line below names the
  arm that was executing.

## KERNEL AUDIT LEDGER (goal v2, all-night pass — grows as audits complete)

| kernel / path | audit | SASS | verdict |
|---|---|---|---|
| gated_delta_net (row-per-warp) | full line audit x2 | nc=0, wait-first, instr parity 256=256 | FIXED (restrict race) + verified |
| k_get_rows_raw | entry audit | nc=0 | FIXED + verified |
| fattn-mma-turbo.cuh (host launcher) | full | n/a (host) | CLEAN — shmem sizing matches upstream formula; attr flags per-instantiation |
| turbo4/3/2 FA tile loaders | full bounds/race audit | — | CLEAN — 66B/128-elem mapping exact; D=256 two-block span within pitch; turbo3 sign +1 shift safe by parity; per-cell disjoint writes |
| turbo-typed flash_attn_ext_f16 instantiations | — | nc=0 of 69/129/81 loads; ACQBULK@2441 < firstLDG@3159 | CLEAN (both PDL surfaces) |
| k_set_rows_turbo3 (+tail) | full | plain-launch (PDL-exempt) | CLEAN — butterfly barriers correct, single-writer packing, full-mask warp ops divergence-safe, shared reuse barriered |
| fwht_cuda | full | wrapper/PDL; sync-before-loads; no entry restrict | CLEAN — early-exit is warp-uniform (shuffle masks legal); butterfly reg indexing bounded |
| k_set_rows_turbo4 | full (pattern-delta vs turbo3) | plain-launch | CLEAN — note: is_v via tensor-name prefix is fragile (zero-risk at alpha 1.00) |
| TCQ + VEC turbo paths | spot-check | device-inner callbacks in entry-clean kernels | CLEAN |
| ssm-scan (20 raw restricts) | reachability | — | EXCLUDED — zero refs in all qwen3* graphs (DeltaNet uses gated_delta_net + ssm_conv) |
| **CHUNKED PREFILL (goal item 3)** | **MERGED: 26001x22587 dedicated merge done** | 3 kernels compiled; plain launches = PDL-exempt; all-NC loads safe by construction | branch feature/gdn-chunked-prefill; artifacts .chunked staged; op-test suite installed (PR-26001 + boundary killers); validation = protocol phase 4 |
| tree-wide entry-grain re-audit (post all merges) | machine | ZERO violations | no regressions |
| chunked kernel deep audit — NOW 100% LINE COVERAGE (fwdsub + state + preqk all full) | full | smem accounting EXACT (fwdsub match; state 30KB<48KB as commented); pool scratch exactly-sized; state->dst-tail contract matches ours; plain launches; per-stage CUDA_CHECK | CLEAN — every overlay barriered, tails guarded (valid_cs), exp clamped, single-warp preqk exact; arithmetic = morning 37-case GPU suite (phase 4) |

## OPEN-ITEMS LIST (live — the /goal ledger; strike items as they close)

- [GATE: reboot] Phase 1 killer x2 on fixed stack -> phase 1b PDL-on
  certification -> phase 3 battery soak + serving restore. THE guarantee rung.
- [ready] .gdn-prefetch variant (side branch): post-stability A/B for extra
  GDN speed (beta+v carried-register prefetch; SASS-checked, op-tests needed).
- [ready] v3 imatrix A/B gates (both quants built, gate slots in omega).
- [ready] DRY anti-spiral arm (in omega).
- [closed tonight] ssm-scan raw restricts: UNREACHABLE in qwen3.5 graphs
  (zero refs in all qwen3* model files — DeltaNet uses gated_delta_net +
  ssm_conv, both verified clean). uid fast-path audit: sound (uid regenerates
  per scheduler rebuild; pool-flush separately guarded); residual risk noted,
  belt available (always-compare) if ever implicated.
- [user, when convenient] vacation-mode arming; compaction; PL-cap optional.

## (superseded first-form analysis below)
## OLD: ROOT CAUSE FOUND (~03:00)

**The GDN row-per-warp kernel (fork-only, shipped Aug 15 = instability day 1)
violated upstream's documented PDL race rule.** Upstream commit 9e58d4d69:
"Avoid PDL race conditions by disabling __restrict__ when PDL is used"
(#24030). Under PDL, kernels launch with early-start overlap; __restrict__ on
kernel-entry pointers licenses the compiler to hoist loads ACROSS
cudaGridDependencySynchronize() into the window where the PREDECESSOR kernel
still executes. Our hand-merged GDN kernel had raw __restrict__ on all 8
entry pointers and fires ~millions of PDL launches/hour (48 layers x every
batch). Per-kernel-entry audit of the whole tree: it was the ONLY violator in
our active path (set_rows/ssm_conv/fattn-turbo entries all clean).
Explains: non-determinism, sustained-load dose-response, ramp/warmup
clustering, Aug-14-15 stability (kernel not yet shipped), community silence
(nobody else runs this kernel), stock-clocks death (clocks irrelevant), and
why op-tests pass (single-op tests never race the PDL boundary).
OC and thermals: aggravators at most, both refuted as root.

**Shipped tonight (defense in depth):**
1. Binary D `llama-server.restrictfix` — entry restricts removed (upstream's
   pattern; arch-conditional macro cannot appear in __global__ signatures —
   first build attempt taught that). Committed+pushed on fix/vision-hybrid.
2. `GGML_CUDA_PDL=0` in ALL launchers + gate + omega (independent layer;
   class-wide; ~0-2% cost; re-enable = one env after post-vacation soak).
3. Binary C `llama-server.gdnmainline` — full kernel revert fallback
   (branch debug/gdn-mainline, pushed).
4. Vacation watchdog drafted (vacation-mode/, DISARMED — user arms):
   GPU-wedge -> auto-reboot -> auto-serve -> alert.

**GUARANTEE STATUS: root-caused + double-mitigated, execution-proof pending
GPU.** The proof ceremony = morning-protocol.sh: Phase 1 (D + PDL0, killer
workload x2) -> Phase 3 (promote + full battery as soak). Phase 1b isolates
whether the restrict fix alone suffices. Binary C + fault tree stand by if
Phase 1 ever dies.

## MEGA-ANALYSIS (as of ~02:00 — investigation log; superseded by ROOT CAUSE above)

**USER'S CALL VINDICATED SO FAR: the software trail got hot.** After the
stock-clocks death (00:27) refuted the OC theory, a from-scratch audit found
what changed on Aug 15 when instability began — and it isn't clocks:

### The two prime suspects (both fork-specific, both entered Aug 15)

**S1 — PDL (Programmatic Dependent Launch) on the hot path.** The b10448 base
carries upstream's PDL infra: kernels launch with
cudaLaunchAttributeProgrammaticStreamSerialization=1 → each kernel may START
BEFORE its predecessor finishes, correctness resting on device-side sync
intrinsics. ~24 kernel files participate. DEFAULT: ON in our tree
(env_pdl_enabled: unset -> true). This runs on consumer SM120 + WDDM + WSL2 +
CUDA graphs — a scheduling path with near-zero field mileage (PDL targets
datacenter Linux). A rare dependent-launch bookkeeping fault = device loss,
non-deterministic, worst at graph capture/warmup and under sustained launch
storms. imatrix fires ~3K GDN launches/min for 30 min = exactly the death dose.
Why Aug 14-15 was stable-ish: the heaviest PDL consumers we run today — the
row-per-warp GDN kernel and fused-MMA — joined the party Aug 15.

**S2 — the row-per-warp GDN kernel itself (#22587-merge).** CONFIRMED NEVER
MERGED UPSTREAM — our hand-integrated variant (181 lines diverged, snapshot
slots + fused-cache + separate state ptr) runs on exactly one machine: this
one. Zero community soak-hours. It already produced one proven device-killing
OOB (the vision bug). Full line audit tonight found the state-write guards
sound and offsets correct — but a subtle race or OOB READ under rare shapes
cannot be excluded by reading; op-tests provably miss this class (they passed
while the first OOB existed).

### The discriminating instruments (built tonight, zero rebuild for S1)

- **S1 test = `GGML_CUDA_PDL=0`** (runtime env; falls back to classic
  serialized launches; cost ~0-2%). No rebuild — it was in the tree all along.
- **S2 test = binary C** `build-g1/bin/llama-server.gdnmainline` (built
  tonight: row-per-warp reverted to mainline GDN, CPU-side mrope/MTP fixes
  kept, branch debug/gdn-mainline).
- **The assay**: the exact workload that killed the GPU at 30 min (imatrix
  1200-chunk run) — repeatable, sustained, needs no serving.
  quality-tests/yarnB/morning-protocol.sh phases 1/2.

### Verdict matrix (morning, after reboot)

| PDL-off survives 2x? | binary-C survives? | Verdict |
|---|---|---|
| YES | (skip) | **PDL convicted** -> GGML_CUDA_PDL=0 in all launchers forever; battery+serving resume same day; vacation viable |
| NO | YES | **row-per-warp GDN convicted** -> serve .gdnmainline (−2.8% decode), hunt the kernel bug offline |
| NO | NO | both software suspects eliminated WITH DATA -> driver reinstall, then hardware track (PL cap, HWiNFO mem-junction, RMA) — with 2 days to react |

### Also delivered tonight (CPU salvage)
- v3 quants BUILT: tqmix-v3-imx2 (session-calibrated) + tqmix-v3-imx1
  (synthetic-calibrated), 19G each — the imatrix A/B pair, gates queued
  post-stability.
- imatrix_v2.dat survived the crash (saved at 00:27, ~310 chunks of real
  session data).
- yarn-B' battery: still queued in omega.sh (resumable) — runs as soon as a
  stable configuration is identified (and doubles as its soak).

## THE PLAN (what tonight answers)

1. **Rope-unification verdict (yarn-B' matrix)** — 3 scales {none, 1.25,
   1.5625} x 3 seeds @64K+128K depth, matched config (ctx=262144, MTP n3,
   turbo4, temp 1.0), + 256K finalists (1.25 vs 1.5625). Decides whether the
   speed profile can run 1.5625 permanently → slot-portable 20-50s auto-tier
   switches instead of ~2-min re-prefills (AUTO-TIER-DESIGN).
2. **Stability soak (the OC conviction test)** — the matrix IS the gauntlet:
   ~20 server restarts + hours of sustained load, the exact pattern that
   killed the OC'd box 3x on Aug 17. Stock surviving the night = OC convicted.
   Any GPU event → SOAKEVENT line in ledger + forensics in the morning.
3. **Quant-lab v3 + imatrix A/B (novel science)** — recipe: Q6_K base +
   ffn gate/up/down=q5_k (~20GB), built TWICE: once with imatrix_v2 (5MB
   real-session corpus: 6,551 msgs from 4 harnesses) and once with imatrix_v1
   (1.4MB synthetic local corpus). Same recipe, different calibration →
   isolates what workload-calibration is actually worth. Gates: KLD-157 +
   speed probe each. Bar: mean ≤~0.012 / top-1 ≥95% (prod Q6_K = 0.0060/96.2%).
4. **DRY anti-spiral probe** — 250K-depth battery arm at --dry-multiplier 0.8
   vs the no-DRY 1.25 finalist (both seed 42): does sequence-repetition
   penalty tame the known 256K ledger spiral without hurting code?
5. Conditional: if v3 passes, a 1.5625-scale KLD of the winner (unified-scale
   ship combo check) runs in the morning window.

## !! NIGHT EVENT — GPU LOST AT STOCK CLOCKS (00:27:56) — OC THEORY REFUTED

The card died 30 minutes into the imatrix-v2 run: single nvlddmkm event
00:27:56, quiet adapter loss (no bugcheck, machine alive, WSL alive), every
later CUDA init = "invalid argument", NVML dead. STOCK CLOCKS. The OC removal
did NOT fix the instability — the OC was at most an aggravator.

**Revised fault tree (evidence-ordered):**
1. **Memory-junction / VRM thermal accumulation** (TOP): GDDR7 junction runs
   20-30C above the core reading and is INVISIBLE to nvidia-smi on consumer
   cards — "temps fine" measured the wrong sensor. Fits the duration pattern
   perfectly: 60s burn OK, bursty 16h serving OK, ~13-30 min SUSTAINED load =
   death (15:00 ~13min benchmark; 00:27 exactly 30min imatrix; 23:21 after a
   full day + warmup ramp). MORNING CHECK: HWiNFO64 (user-run, Windows) shows
   GDDR7 memory-junction + VRM temps — run it during a load test.
2. **PSU heat-soak / sustained-draw marginality**: same duration signature.
   Mitigation test: power-limit 80% (user-run, 1 min) — if deaths stop at
   PL80, strongly implicates power/thermal envelope.
3. Board/VRAM defect (RMA track) — if 1-2 mitigations fail.
4. Driver 610.47 + WSL pathology under sustained load — clean reinstall is
   cheap to try; lowest prior (3 months stable... but 3.8-era load is new).

**Vacation impact (user leaves ~Aug 19-20):** normal Hermes serving (bursty)
has survived 16h stretches — the killer is SUSTAINED load (benchmarks,
batteries, quant jobs). Interim doctrine: NO sustained GPU jobs unattended;
serving itself is probably survivable, but vacation-mode must assume deaths:
auto-relauncher becomes ESSENTIAL, plus remote reboot ability or acceptance
of downtime-until-return. Decision menu in morning analysis.

## TIMELINE / DATA (appended as milestones land)

- 23:42 first overnight launched (simple version); user interrupted; replaced
  by omega at ~00:2x with expanded scope. Ledger preserved (was empty).
- 00:27:56 GPU adapter lost during imatrix v2 (see NIGHT EVENT above).
  imatrix_v2.dat SAVED at 00:27 (~310 chunks / ~159K session tokens) = usable.
  Night converted to CPU salvage: convert -> v3 quants x2 (both imatrices)
  built overnight; GPU gates queued for post-reboot morning window.
- yarn-B' battery: ZERO arms ran (GPU died before block 1; the 6 SOAKEVENT
  ledger lines are launcher-race artifacts EXCEPT the root cause = dead GPU).
  Battery re-queued for morning IF stability path chosen; spec unchanged.
- Stock clocks verified before launch ([Startup] zeros, 14001 MHz mem max).
  Amplifier env gone from launchers; TdrDelay=60 armed. Serving DOWN for the
  night by design; restored automatically at omega completion.

## CONTINGENCY NOTES (for morning-me or a successor agent)

- Machine death overnight = OC theory REFUTED at stock → deeper hw/driver
  track; forensics per repo CLAUDE.md 3.5/3.6 (Windows events + nvcudmp);
  ledger+logs survive on disk. Vacation decision changes accordingly.
- Ledger line format: DONE/TIMEOUT/FAIL <label>, SOAKEVENT <where>,
  "DONE gate_<q> mean=X top1=Y", COMPLETE <time>. omega.sh is resumable:
  relaunch skips DONE lines.
- Battery arms that FAIL individually don't kill the night (isolation);
  three FAILs in one block usually = server died = check server-cur.log.
- Morning queue after mega-analysis: vacation-mode build+test (relauncher
  DISARMED tonight by design), compaction (user-run), v3@1.5625 conditional,
  serving verification, handoff final update.
