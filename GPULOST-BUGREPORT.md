# GPU-LOST bug — root cause + corrective action plan (2026-08-16)

STATUS: root cause identified with high confidence. Vision DISABLED in all three
launchers as mitigation. Fix not yet implemented. Forensic logs:
crash-forensics/gpulost1-server.log, gpulost2-server.log.

## Symptom

Twice on 2026-08-16 (~16:3x and ~17:45): RTX 5090 falls off the PCIe bus
(nvidia-smi: "GPU is lost. Reboot the system"), display dead, WSL survives,
server becomes a zombie (HTTP health 200, all inference hangs, client cancels
every ~10 min). Reboot required. Both times the user had just sent a photo to
Hermes, then stopped and resumed the chat.

## Root cause chain (evidence-backed)

1. Hermes (fixed by user's agent earlier that day) sends the image INLINE ->
   it reaches llama-server's mtmd path. NOTE: log shows no "image" strings at
   this verbosity — detection was via the position-warning fingerprint below.
2. Qwen-VL uses MTMD_POS_TYPE_MROPE: "each image takes max(t,h,w) position
   indexes" (tools/mtmd/mtmd.cpp:199). ~1450 image-embedding tokens occupy
   only ~53 llama_pos values. Confirmed in logs: 512+512+426-token embd
   ubatches whose positions sit at ~136, then text resumes at 189.
3. llama-memory-recurrent::find_slot (src/llama-memory-recurrent.cpp:609)
   receives these non-monotonic positions. The code path explicitly does not
   support them: "What should happen when the pos backtracks or skips a
   value? Clearing the state mid-batch would require special-casing which
   isn't done." It WARNS AND PROCEEDS -> DeltaNet cell bookkeeping
   (pos/tail/src/src0) corrupts. Warnings print twice per ubatch (main ctx +
   MTP draft ctx recurrent memories).
4. Corrupted recurrent-state metadata feeds the GDN CUDA kernel path
   (row-per-warp kernel, adopted this week, WITH MTP rs slots x(1+n_max)) ->
   out-of-range state access -> wild VRAM write -> WSL dxg escalates to
   device-lost (not a catchable CUDA error).
5. Death timing: crash 1 died during the resumed re-prefill (which re-encoded
   the image: 40s silent CPU-encode gap then warnings then death). Crash 2's
   state corrupted at image time (69.07), died on the next forced full
   re-prefill (69.33).

## Exonerated

- The uncensored finetune: same code path any weights (it was merely loaded
  both times because vision requests only started today).
- stop/resume alone: text-only forced re-prefills ran 3x same morning, clean.
- The uploader's vision file specifically: plausibly irrelevant (mrope layout
  is model-family behavior), unproven either way.
- Power delivery: pattern too correlated with images to be transients.

## Why the pilot's vision tests didn't kill it

Pilot vision smoke tests (2026-08-1x, "vision live", 21s/img) ran on the OLD
GDN kernel. The row-per-warp kernel (#22587, adopted 2026-08-16) had never
seen an image until today. Old kernel: tolerated the corrupt bookkeeping
(warn-only era). New kernel: fatal. Secondary confirmation available via
.pre-gdn22587 rollback binary under the repro (Phase 2).

## Corrective action plan

### Phase 0 — mitigation (DONE 2026-08-16)
- --mmproj removed from start-long-38.sh, start-max-38.sh, start-long-38u.sh
  (restoration note above the launch command in each). Server now cleanly
  rejects image content. No image can enter the death path.

### Phase 1 — fixes (fork, in order)
1. SAFETY FLOOR: llama-memory-recurrent::find_slot must REJECT (return false)
   non-monotonic-position batches instead of warn-and-proceed -> llama_decode
   fails -> server returns a clean error. Converts GPU-death into a 4xx/5xx.
   Small, surgical, ship first.
2. UPSTREAM CHECK: Qwen3.5-VL hybrids exist upstream — check master's
   handling of mrope-on-recurrent (mtmd + memory-recurrent + hybrid memory).
   If upstream solved it, cherry-pick semantics (snapshot slots — same merge
   discipline as #22587). If not, implement: decouple the recurrent branch's
   position domain from mrope (recurrent state needs a per-token monotonic
   counter, not rope positions; hybrid memory should feed each sub-memory its
   own pos domain).
3. KERNEL GUARDS: bounds-check rs slot indices / s_copy values at GDN launch
   (cheap in release). Any future bookkeeping bug fails the request, never
   the GPU.
4. MTP HOOK DESYNC: process() skips embd batches -> draft ctx position desync
   after any image (degraded acceptance). Fix alongside (2).

### Phase 2 — verification (reboot-safe protocol)
1. CPU-only build: replay the mrope position pattern (tiny image, small ctx)
   -> confirm bookkeeping corruption pre-fix, absence post-fix. Zero GPU risk.
2. GPU, old-kernel binary (.pre-gdn22587): one babysat image test — validates
   the kernel-differential theory. Accept one reboot risk.
3. GPU, fixed build: image -> stop -> resume -> re-prefill (the exact killer
   sequence), babysat.
4. Standard gates: test-backend-ops, text KLD unchanged, vision smoke on
   ORIGINAL model + original mmproj, battery spot-check.

### Phase 3 — policy
- Vision stays OFF in every launcher until Phase 1+2 complete.
- CLAUDE.md ops laws updated: images on hybrid = known GPU-killer until fixed.
- After fix: re-run BestApp-era vision measurements (21s/img baseline) before
  telling Hermes vision is available again.

## Open questions

- Exact faulting kernel/instruction (GDN rs slots vs s_copy gather vs other):
  Phase 2.1 CPU repro + Phase 2.2 kernel differential will localize. Not
  required for the fix set above (defense in depth covers all candidates).
- Whether crash 1's PC-wide death vs crash 2's GPU-only death is severity
  variance of the same fault (assumed yes).

## STATUS UPDATE 2026-08-16 late — FIX IMPLEMENTED, CPU-VALIDATED

Branch: turboquant-sync fix/vision-hybrid @ 72fa4ca4e (F1 kernel bounds, F2
recurrent mrope handling, F3 MTP draft resync). Build clean (CUDA compiled).

CPU-mode live repro (crash-forensics/cpu-repro-PASS.log), fix build, ORIGINAL
model + mmproj, MTP n3:
- Phase 1 image request: clean 200; no recurrent warnings (F2 silent-forward
  path); F3 fired as designed ("draft ctx desync ... clearing draft state").
- Phase 2 EXACT killer sequence (abort mid-gen -> resume w/ rewritten history
  -> forced full re-prefill incl. image re-encode): clean 200; identical log
  fingerprint to both crashes (cancel/forcing-full/erased-checkpoints) with a
  healthy server at the end.

REMAINING GATES (post-reboot, in order): CUDA test-backend-ops GDN 36/36;
same two-phase repro on GPU babysat (accepts one reboot risk); text KLD spot
vs prod baseline; battery spot; THEN restore --mmproj in launchers + rebuild
prod binary from the fix branch + re-baseline vision (21s/img). Prod stays on
the unmodified binary + vision-off until all gates pass.
