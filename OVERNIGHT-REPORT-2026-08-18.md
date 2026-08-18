# OVERNIGHT REPORT — 2026-08-17/18 (omega run, stock clocks)

LIVING DOCUMENT — updated at each milestone through the night; mega-analysis
lands at the top by morning. Raw data: quality-tests/yarnB/ (ledger.txt =
ground truth of progress), quality-tests/kldnvfp4/ (quant gates),
quality-tests/quant-lab/ (imatrices, recipes).

## MEGA-ANALYSIS (as of ~02:00 — night investigation phase complete)

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
