# Inference Improvement Research Board — 2026-08-16 search blast

Rig: RTX 5090 32GB (SM120 Blackwell, FP4 tensor cores) / 48GB RAM / WSL2.
Stack: fork w/ turbo4 KV (4.125bpv WHT+centroid), MTP n3, fused MMA verify,
GDN row-per-warp, ctx-checkpoints, two-profile serving (320K/409K).
Doctrine: QUALITY AND FEATURES OVER SPEED; nothing adopts without on-stack gates.

## TIER 1 — FP4 weight track (biggest single lever found)

FP4 (NVFP4/MXFP4) landed in llama.cpp upstream with native Blackwell tensor-core
paths (PR #17906 era; correctness discussion #22042). Evidence:
- Qwen3.6-27B NVFP4 GGUFs shipped (LibertAIDAI incl. MTP variant; williamliao);
  community report: 24 t/s (Q4_K_M) -> 45 t/s @160K ctx with NVFP4+MTP.
- Unsloth ships Qwen3.8 NVFP4/UD quants; claims 1.5x vs BF16, "92-97% top-1
  accuracy" — that claim is BELOW our KV-gate bar; weight quality gate decides.
- MTP verify batches "nearly free on FP4 tensor cores" (community note).
Payoff for us: Q6_K 21.4GB -> ~14GB weights = ~7GB VRAM freed
  => 409K + MTP in ONE profile (kills the two-profile trade), or 500K+ ctx,
  PLUS prefill/decode speedup on FP4 cores.
Cost/risk: (a) requires upstream sync (our fork predates FP4 kernels);
  (b) Q6_K->4bit weight quality is a REAL drop class — full gate ladder
  mandatory (tail-KLD 157, battery multi-seed, NIAH, fill-ladder);
  (c) Q5_K_M (~15.5GB, -5.9GB) is the fallback middle rung if FP4 fails gates.
Verdict: RESEARCH NOW, ADOPT ONLY THROUGH GATES.

## TIER 2 — Upstream sync 2.0 (unlocks T1 + cheap A/Bs)

Fork is months behind master. Landed upstream since our snapshot:
- FP4 weight kernels (T1 dependency)
- Unified spec framework: comma-COMBINED spec types (e.g. draft-mtp,ngram-mod),
  new ngram-map-k / ngram-map-k4v, DFlash/DSpark block-diffusion drafts,
  EAGLE-3 (incl. hybrid/recurrent support PR #21437 — mirrors our snapshot-slot
  work), upstream MTP (#22673, different design than ours)
- GGML_CUDA_MMVQ_MAX runtime crossover knob (PR #26079) — cheap decode A/B
- SWA/hybrid checkpoint+MTP acceptance fixes (issue #23322 — matches bugs we
  solved independently; compare approaches)
Known trap (from #22587 merge): upstream semantic evolution — snapshot slots.
Re-A/B after sync: combined ngram-mod+MTP (our ngram wash was pre-framework).

## TIER 3 — Per-layer mixed-precision KV (quality-per-byte, our home turf)

2026 literature: KVTuner (layer-wise pairs, ~3.25bpv near-lossless), RateQuant
(rate-distortion per-head bit allocation), PM-KVQ (progressive for long CoT),
KVSink, SKVQ (recent-window high precision).
Our angle: P0 probe proved the 16 full-attn layers carry ALL global routing on
this hybrid. Testable forks of the idea, using existing KLD/battery harness:
  (a) q8_0 on 16 attn layers + turbo4 elsewhere  -> quality up, +~0.9GB
  (b) turbo4 attn + turbo3_tcq on DeltaNet-adjacent -> bytes down, ctx up
  (c) age-based: recent window turbo4/q8, old tokens turbo3_tcq
Needs fork feature: per-layer -ctk/-ctv typing. Moderate effort, high fit.

## TIER 4 — Ops quick wins (cheap, do soon)

- WSL cold-load 124MB/s: add Defender exclusions (ext4.vhdx + vmmem), sparse
  VHD (wsl --manage --set-sparse), fstrim + compact, wsl --update, verify vhdx
  on fastest NVMe. Could turn 3-8min cold restarts into ~30s. USER-SIDE mostly.
- HAGS (Windows hardware GPU scheduling) A/B: reports of notable gains
  specifically WITH CUDA graphs (we use them). Toggle + measure decode.
- Windows: networkingMode=mirrored still pending (U1, user-side).

## TIER 5 — Spec-decode refinements

- EAGLE-3: NO Qwen3.8-27B head exists yet. Qwen3.6-27B head (Ex0bit) reaches
  tau=2.4 chain / 3.35 tree (tree throughput-neutral on hybrid GDN — matches
  our arch). Our MTP n3 measured 2.98-3.10 tok/cycle TODAY => already ahead.
  WATCH for a 3.8 head; training our own (SpecForge/TRT-MO) = real project.
- Grammar-turn spec toggle: tool/grammar turns run 0.6-0.85x with MTP.
  Small patch: proxy or server disables spec per-request when grammar present.
  Recovers tool-loop tax; zero quality risk (spec is lossless either way).
- Draft-model spec (repo's draft-Q8_0) vs MTP: A/B only out of curiosity; MTP
  shares weights/cache (VRAM-free) — draft model unlikely to win here.

## TIER 6 — Bigger swings (research-grade)

- SageAttention2++/3: INT8/FP4 quantized attention. SA3 = 1038 TOPS on RTX5090
  (5x FlashAttention); prebuilt sm_120 wheels exist (PyTorch world).
  Port to our CUDA FA = heavy kernel work + interacts with turbo4 KV format
  (K stored rotated+quantized). Prefill-first port is the plausible slice
  (attention share of prefill grows with depth). Park unless prefill at depth
  becomes the pain point.
- Second GPU (heterogeneous): llama.cpp layer-split pools VRAM but SLOWS what
  already fits on the 5090; auto-fit exists (#18049); ik_llama '-sm graph'
  proves multi-GPU can scale (3-4x claims) but is a different fork.
  Sensible uses if a card appears: +ctx overflow layers, second model, vision.
  NOT recommended to buy for speed of the current model.
- vLLM/TRT-LLM NVFP4 (5090 fixes exist): benchmark reference only — would
  abandon turbo4/fork features. Use as external baseline for T1 expectations.

## Examined & closed (do not re-tread without new evidence)

- Sparse/paged attention decode: CLOSED at P0 (oracle needs 30-50% of cache;
  hybrid arch concentrates routing in 16 layers). Same finding kills
  token-eviction (H2O/SnapKV class) here.
- ngram spec (our fork impl): wash. p-min gate, cascades: WORSE (fused-verify
  economics invert community tuning).
- Backend sampling: wash at np=1 (issue #27050 confirms it's a multi-slot win;
  we serve single-slot by quality choice).
- FWHT butterfly port: slower on SM120 (bank conflicts).
- iGPU assist: WSL2 iGPU compute path (Dozen) is poor; CPU mmproj stays.
- KV->RAM decode offload: decode is bandwidth-bound; only slot/checkpoint
  storage belongs in RAM (already done).

## Suggested sequence

1. Tonight: yarn A'+B' battery (already queued) — unchanged.
2. T4 ops wins (hours, mostly user-side clicks + one A/B).
3. T2 upstream sync on a branch (snapshot slots; op-tests -> A/B -> KLD gates).
4. T1 FP4 weight gate ladder on synced branch (NVFP4 vs Q5_K_M vs Q6_K).
5. T3 per-layer KV mixed precision prototype + KLD.
6. T5 grammar-turn spec toggle (small, anytime).

Sources: llama.cpp docs/speculative.md; PRs #17906 #22673 #21437 #26079 #18039;
discussions #22042 #18049 #15013; issues #23322 #27050; arXiv 2505.11594 (SA3),
2505.21136 (SA2++), 2505.18610 (PM-KVQ), 2502.04420 (KVTuner), 2605.06675
(RateQuant); HF: LibertAIDAI/williamliao NVFP4, Ex0bit PRISM-EAGLE3, unsloth
qwen3.8 docs; InsiderLLM FP4 guide; Bored Consultant NVFP4 45t/s writeup;
ceos3c WSL2 perf 2026.
