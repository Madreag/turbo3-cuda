# Sparse Decode Design — Quest-class page selection over TurboQuant KV

Status: DESIGN (2026-08-15). Prereqs shipped: MMA-turbo decode path,
trajectory battery (the safety gate this feature must pass).

## Problem
Decode at depth is bandwidth-bound: every token reads the full attention KV
(5.8 GB at 320K). Measured decode: ~110-122 tok/s shallow → ~75-84 @38K →
falls with depth. Frontier consensus (SAW-INT4 etc.): at 4.125 bpv we sit at
the quantization ceiling — the remaining lever is reading FEWER tokens, not
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
  scored}. Target effective window 8-16K tokens → ~10-40× fewer KV reads at
  320K.
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
- P2 gates: trajectory battery (multi-hop is EXACTLY what sparsity threatens)
  ≥ baseline; KLD ≤ 1.1× baseline; needle multi-position; depth-decode curve
  16K→256K (win must be ≥1.5× at 128K+ to justify).
- P3 tune: page size {64,128}, N schedule per depth, anchor-layer count,
  sink/recent sizes; f8 metadata.

## Risks
- Multi-hop regression (mitigation: battery gate + generous N floor).
- Draft/MTP interplay: draft context stays FULL attention (tiny anyway).
- VRAM +6% metadata (budget exists: 1.5 GB headroom holds ~350 MB at 320K).

Effort: ~1-2 weeks of kernel work. Not started — next big arc after current
queue closes.
