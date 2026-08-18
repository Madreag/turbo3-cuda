# Optimization blast — 2026-08-17 evening (speed/quality/reliability/features/multi-GPU)

Hardware: 9950X3D (Zen5 AVX512, 3D V-cache) / 48GB / RTX 5090 32GB / fast SSD (main);
RTX 3090 24GB + RTX 3090 Ti 24GB in two other PCs (slower CPU, 32GB) on LAN.
Method: web verification (llama.cpp upstream, community tooling) x our measured
bottleneck profile (depth decode 55-90 t/s = attention/KV-bound; prefill 1.7-2.7K t/s;
40-75s history-rewrite stalls; 256K ledger spiral; near-cap VRAM era just ended-ish).

## TIER 1 — high value, near-term, on-stack testable

**R1. Custom workload-calibrated quant (imatrix from OUR Hermes traffic).**
NVFP4 (type 40) + MXFP4 (39) confirmed in-tree at our base; llama-imatrix +
llama-quantize --imatrix + per-tensor overrides (--tensor-type) = roll our own:
(a) calibrated NVFP4 or NVFP4/Q-mix targeting ~18-19GB with near-Q6 quality —
the community Q8attn artifact FAILED our gate (11.5x KLD) but was uncalibrated;
imatrix typically buys 10-30% quality back, and OUR calibration text = real
agent traffic (code/tool/think), not wikitext. (b) Even a Q6_K-with-imatrix
re-quant may inch quality up free. Needs: BF16 safetensors (~54GB dl, disk ok),
imatrix via partial offload (-ngl ~55%, 9950X3D AVX512 carries the rest,
overnight-class, babysat), quantize, then the standard gate ladder. Expectation
pre-measured from the failed artifact: ~+10-25% decode, ~3GB VRAM freed —
IF the quality gate passes. This is the highest EV item on the board.

**R2. DRY sampler vs the 256K ledger spiral.** --dry-multiplier is IN our prod
binary (default 0). DRY penalizes repeated SEQUENCES (not blunt token bans —
grammar-safe). Test: battery 256K tier + the known spiral seeds, arms
{off, 0.6, 0.8} x dry-allowed-length {2,4}; gate = spiral tamed with ZERO
regression on clean seeds + code-traj (code repeats legitimate tokens — watch
closely; if code regresses at all, scope DRY as a >128K-depth-only profile
knob via client). Cheap (~1h), could extend the all-axes-clean envelope past
128K. XTC: NO for us (removes top tokens = anti-coding by design).

**R3. Failover/overflow serving tier on the 3090/3090Ti.** Layer-split RPC =
verified pointless for a model that fits (and slows to the weakest card).
The RIGHT multi-PC play: each 24GB box serves its own full Qwen3.8-27B
(Q4-class, turbo KV, ~131-200K ctx — community proved turbo3+200K on 24GB;
turboquant 3090 row: 61 t/s) as an OVERFLOW/FAILOVER upstream. Options:
extend our proxy with a fallback upstream list (~100-200 lines, keeps auth/
slot-pinning/tripwires) or insert Paddler (slot-aware llama.cpp LB, Rust,
production-grade) behind our proxy. Solves the REAL pain: Hermes goes dark
every gate window/battery/crash. USER DECISION REQUIRED: fallback serves the
same advertised id at lower quant = silent quality drop vs honest 503.

**R4. Auxiliary-model mesh on the spare cards (featureset).** 3090 hosts
lightweight LAN services: fast VLM for Hermes's vision_analyze tool (kills
our 21-60s CPU encode path for casual image asks), embeddings/reranker for
agent memory, a 4-8B utility model for cheap subtasks. Zero risk to main.

## TIER 2 — real but behind Tier 1

**R5. Fused-verify window Qle4 -> Qle6** (from n4 rejection analysis + vLLM
climbing where llama.cpp peaks): would flip n4 economics at depth; re-test n4
after. MMA kernel work on fattn-mma-turbo. Do AFTER R1 (weights change resets
baselines).
**R6. Prefill ubatch sweep** (-ub 768/1024) when VRAM headroom allows (R1
would fund it): prefill +10-30% class; helps the 40-75s history-rewrite stalls.
**R7. mmproj CPU encode tuning**: Zen5 AVX512 thread/affinity sweep for the
vision tower (V-cache CCD pinning); possibly GPU mmproj on MAX profile (1.9GB
free there). Fold into the pending vision re-baseline + --image-min-tokens
1024 grounding A/B.
**R8. Uncensored-trial quality gate** (STILL UNGATED in prod): tail-KLD 157 +
battery vs original — ~1h, honesty demands it if the trial stays.
**R9. Reliability autos**: crash-recovery relauncher (pid-dead AND no stop.sh
sentinel -> relaunch + notify; does NOT violate the no-boot-autostart choice),
metrics scraper -> Discord alerts (health, VmSwap>0, decode-collapse tripwire,
VRAM near-cap), weekly idle VRAM mtest cron (GDDR7 has no ECC; catch
degradation early). USER DECISION on the relauncher (lifecycle law territory).

## WATCHLIST (not actionable yet)

- **Disaggregated prefill/decode — upstream issue #21266**: prefill on remote
  RPC devices, KV shipped to the decode box. THE future role for the 3090s
  (read-ahead prefill of next-session context while 5090 decodes). Not merged;
  LAN math today: 100K-tok KV ~1.7GB ~14s @1GbE + 3090 prefills 3x slower —
  only pays when it overlaps. Revisit when merged; a 2.5/10GbE NIC pair would
  change the math.
- SM120 attention ceiling: consumer Blackwell has NO TMEM -> FA4-class kernels
  are datacenter-only; our fused-MMA lane + KV-bpv is the right (only) lane.
- Upstream TurboQuant integration discussion (#20969) alive; Windows prebuilts
  of MTP+TurboQuant+sm120 circulating (Andgihat repo) — ecosystem adoption.

## CLOSED WITH REASONS (so nobody re-litigates)

- RPC layer-split of the 27B across boxes: model fits on 5090; split = slower
  than weakest card + per-layer LAN hop. Confirmed by community reports.
- LAN draft server for spec decode: per-step RTT kills it; MTP is in-model.
- Remote KV/slot store on other PCs' RAM: local SSD is faster than 1GbE.
- Cross-arch paired A/Bs (5090 vs 3090 arms): different kernels = invalid
  pairing; 3090s may pre-screen (NIAH/battery-class) and serve as build+op-test
  CI mules, never as gate arms.
- XTC sampler: anti-top-token by design = anti-coding.
- 3090s as "more VRAM" for the main model: see RPC line.

Sources: llama.cpp quantize/imatrix docs, discussion #23853 (NVFP4 tooling),
rpc README + community distributed-inference reports, DRY/XTC server docs,
Paddler (distantmagic/paddler) + llama-swap, issue #21266 (P/D disagg),
FA4/Blackwell TMEM coverage (lambda.ai), CUDA-toolkit-trap article (zenn),
qwen38-mtp community sweeps (already reviewed).
