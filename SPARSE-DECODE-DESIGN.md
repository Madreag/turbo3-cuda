# Sparse Decode Design — Quest-class page selection over TurboQuant KV

Status: DESIGN — CURRENT as of 2026-08-15 board-complete. Prereqs shipped:
MMA-turbo decode path (+24565 tune: 84-89 tok/s @38K), trajectory battery
WITH recorded baseline (traj_b10448-320k-baseline2.json — the gate).

## Problem
Decode at depth is bandwidth-bound: every token reads the full attention KV
(~5.3 GB at the current 320K profile). Measured decode (fitted, unpaged,
24565 adopted): ~110-122 tok/s shallow → 84-89 @38K → falls further with
depth. Frontier consensus (SAW-INT4 etc.): at 4.125 bpv we sit at the
quantization ceiling — the remaining lever is reading FEWER tokens, not
smaller ones.

## Key insight: rotation is a non-issue
Quest-class selection needs per-page, per-dimension K extremes to upper-bound
`max(q·k)` per page. Our K is stored WHT-rotated; Q is rotated at graph level.
WHT is orthonormal → inner products are basis-invariant → bounds computed in
ROTATED coordinates are valid bounds on the same scalar q·k. No de-rotation,
no extra transforms: score directly in the stored domain. (The bound quality
may even improve: rotated dims are variance-flattened, tightening min/max.)

## Sketch
- Page = 64 tokens. Metadata per page/layer/kv-head: f16 min[256] + max[256]
  of rotated K → 1 KiB; ×4 heads ×16 layers = 64 KiB per page ≈ **+1 KiB/token
  ≈ +6% of KV** (f8 later → +3%).
- Selection per decode step, per layer: score pages via Σ_d max(q_d·min_d,
  q_d·max_d); keep {first S sink pages} ∪ {last W recent pages} ∪ {top-N
  scored}. Target effective window 8-16K tokens → ~10-40x fewer KV reads at
  320K (current profile) and more at any future larger ctx.
- Gather → FA: upstream MiniMax-MSA plumbing (merged #24908: block score →
  top-k → get_rows → FA over selected KV) is reusable; our fork already ships
  MMA-turbo FA over arbitrary KV views.
- Metadata maintenance: set_rows encode kernel touches each token exactly once
  → update page min/max in the same kernel (running max within the launch;
  page boundaries align with 64/128 blocks).
- Layer reuse (Kascade): exact top-k only in anchor layers (e.g. every 4th of
  the 16), adjacent layers reuse the anchor's page list → scoring cost ÷4.

## Phases + gates
- P0 harness: offline validator — for captured contexts, compare selected
  pages vs true attention mass (from full FA) → recall@N curves per layer.
  No kernel work; python + one debug endpoint.
- P1 kernels behind env flag (TURBO_SPARSE_DECODE=N): metadata buffers +
  set_rows update + scoring kernel + gather+FA wiring. Decode-only (Q≤4),
  full attention during prefill.
- P2 gates (CONCRETE, baselines recorded): trajectory battery seed-42 —
  hops/correction/code-traj must stay PASS at 64K/128K/256K and ledger must
  not drop below 8/8 / spiral / 4/8; KLD ≤ 0.0052 (1.1x of 0.00473); needle
  multi-position; depth-decode curve 16K→256K — win must be ≥1.5x over the
  84-89 tok/s @38K baseline at 128K+ to justify the complexity.
- P3 tune: page size {64,128}, N schedule per depth, anchor-layer count,
  sink/recent sizes; f8 metadata.

## Risks
- Multi-hop regression (mitigation: battery gate + generous N floor).
- Draft/MTP interplay: draft context stays FULL attention (tiny anyway).
- VRAM +6% metadata: ~320 MB at 320K vs ~900 MB current headroom — fits but
  tightens; f8 metadata (P3) halves it. Re-verify the budget table (VRAM law
  in HERMES-HANDOFF.md) before P1 lands.

Effort: ~1-2 weeks of kernel work. NOT STARTED — the single remaining board
item; start in a fresh session with this doc + WORKPLAN-BESTAPP.md + the
battery baseline as the working set.
