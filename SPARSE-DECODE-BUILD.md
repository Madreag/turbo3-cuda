# Sparse Decode — BUILD LEDGER (2026-08-15 →)

Executing SPARSE-DECODE-DESIGN.md ("go sparse" authorized). This file is the
live build ledger: verified code facts, locked decisions, phase results.
Resume rule: read this + SPARSE-DECODE-DESIGN.md + WORKPLAN-BESTAPP.md.

## Working set
- Branch: `feature/sparse-decode` off `sync/2026-08` tip 4858318a2 (= prod
  binary source), worktree /home/erol/ai/turboquant/turboquant-sync.
- Build dir: turboquant-sync/build (ccache, incremental). Prod binary safe at
  build-g1/bin/llama-server (deploy-by-copy; candidates go to
  llama-server.sparse, promote ONLY after P2 gates).
- Model: models/qwen38/Qwen3.8-27B-Q6_K.gguf. Prod flags: see
  ~/.config/llama-tcq/start-long-38.sh (320K YaRN 1.25, turbo4 K/V+draft,
  MTP n_max 2, vision CPU).
- GPU discipline: server stopped during any GPU experiment; restart after.
  One GPU workload at a time. Long jobs run_in_background (120s guillotine).

## Verified code facts (recon 2026-08-15, three agents, file:line in session)
- A1 ROTATION IS TWO-LAYER. (i) learned Hadamard (`llama_mul_mat_hadamard`,
  self_k_rot/self_v_rot, active for quantized KV, kill LLAMA_ATTN_ROT_DISABLE)
  applied to q,k,v at graph level (llama-graph.cpp:2803-2810) + un-rot on
  output :2848. (ii) turbo WHT: K/V INSIDE the set_rows quantize kernel
  (set-rows.cu k_set_rows_turbo4:1111: innerq→L2-norm→WHT(signs,1/√128)→
  centroid+pack); Q via graph op ggml_turbo_wht fwd (llama-graph.cpp:2836-43);
  inverse WHT applied to FA OUTPUT keyed on v->type (llama-graph.cpp:2564-77).
  CUDA dequant = centroid×norm ONLY → STORED (WHT) BASIS. FA q̃ and dequant-K
  are the consistent basis pair → bounds must be built from DEQUANTIZED CACHE
  BYTES, not from k_cur (k_cur is pre-WHT/pre-norm).
- A1b CPU codec (ggml-turbo-quant.c) uses a DIFFERENT dense QR rotation —
  NOT bit-compatible with CUDA. Never dequant GPU-written blocks on CPU.
- A2 KV write = ggml_set_rows(full-cache-tensor, k_cur_2d f32, k_idxs I64
  [n_tokens]) via kv_cache cpy_k/cpy_v (llama-kv-cache.cpp:1454/1501); wht
  group smuggled in op_params. No cpy path for turbo.
- A3 MMA-turbo fused: instances D∈{128,256}×{turbo4,3,2}(no turbo2@256),
  Q≤4, K.type==V.type, raw BYTE strides from nb[] (no contiguity req; only
  nb[0]==type_size), dequant into SMEM (no inverse WHT), nstages=0. Mask:
  f16 [n_kv, n_batch] contiguous, ne[2]==1; with GQA packing (ncols2>1)
  K->ne[1] % 256 == 0 REQUIRED (else falls to (8,1) instance). Gathered
  contiguous turbo4 is consumable. FATTN_KQ_STRIDE=256.
- A3b ⚠ GGML_TURBO_MMA_FUSED is DEFAULT OFF in-tree (fattn.cu:190-196,
  opt-in =1; comment: fused faster at every depth + KLD-equal but not
  token-identical to VEC — kept OFF for A/B hygiene) and PROD SERVER ENV
  DOES NOT SET IT → live prod runs VEC(Q≤2)/MMA_F16-dequant(Q=3,4), NOT the
  fused port. Handoff doc claims default ON. ACTION: sparse branch flips
  default ON (=0 kill-switch); P2 measures dense fused-ON as the baseline.
- A4 get_rows: dst ALWAYS F32 at API level (ggml.c:3941); CUDA supports
  turbo src→F32 (dequant); NO quantized→quantized raw path — must add
  (seams identified: ggml.c dst type entry, getrows.cu k_get_rows_float_vec
  16B-copy kernel shape, dst switch :443, asserts :479; CPU must DENY).
  MSA's decode gather does turbo→F32→cast F16 = full materialization (would
  move MORE bytes than dense at 320K → raw gather is REQUIRED for the win).
- A5 ggml_top_k real op (I32, contiguous), CUDA argsort bitonic ≤1024 cols
  else CUB; ggml_cast I32↔F32 exists on CUDA (cpy.cu). MSA does index
  arithmetic via cast→scale→add→cast (proven pattern).
- A6 insertion point: build_attn(llm_graph_input_attn_kv) llama-graph.cpp
  :2788-2850 between q-WHT and build_attn_mha. Draft/MTP uses the SAME
  build_attn → exclude via params.gtype == LLM_GRAPH_TYPE_DECODER_MTP.
  Decode signal: ubatch n_tokens (reserve-time tg graph uses n_kv=kv_size →
  sparse topology gets reserved iff condition uses n_kv threshold ✓).
- A7 mask [n_kv, n_tokens/stream, 1, n_stream] f16 (FA), 0/-inf; n_kv
  GGML_PAD to max(n_pad,256); empty cells = -inf (pad rows safely masked).
- A8 restore (state_read_data) writes cache via ggml_backend_tensor_set —
  BYPASSES graph → meta stale ⇒ rebuild flag needed. seq_rm touches only
  host cell metadata (ghost bytes remain → meta only WIDENS = still-valid
  bounds ✓). No defrag exists. Cell reuse goes through set_rows ✓.
- A9 cells.pos host-side (llama_kv_cells). RoPE-shift rebuild does
  cast→hadamard→rope→hadamard→cpy in place (prod uses --no-context-shift).
- A10 block_turbo4_0 = 66B {ggml_half norm; uint8 qs[64]} = 128 vals,
  4.125bpv; nibble: elem j → qs[j/2], low=even. Row (n_embd_gqa=1024) =
  8 blocks = 528B. Centroids TURBO_CENTROIDS_4BIT (turbo-quant.cuh:385):
  symmetric 16-table [-0.241529..-0.011349, +0.011349..+0.241529].
  norm stored WITH alpha correction (write-time).
- MSA (MiniMax-M3) full plumbing in-tree: pos↔cell maps, +1e30 force-bias
  before top_k, decode=gather(get_rows→cast f16→FA folded GQA-on-channel),
  prefill=mask-scatter via set_rows-of-zeros; llama-kv-cache-msa dual-cache
  precedent. DSA per-token top-k mask path also in-tree (deepseek32 etc.).
- New-op registry checklist (fork precedent TURBO_WHT, op count 102):
  ggml.h enum+API, ggml.c names+2×static_assert+ctor, ggml-cpu ops.h/.cpp+
  ggml-cpu.c fwd+n_tasks, ggml-cuda <op>.cu/.cuh+dispatch+supports_op,
  ggml-backend-meta.cpp split handler (EASY TO MISS).
- kv_layer extra tensors: precedent turbo_innerq_scale_inv; bump ctx
  mem_size at llama-kv-cache.cpp:135; state save iterates ONLY k/v layers →
  slot format unchanged; extra tensors NOT persisted (⇒ rebuild on restore).

## Locked design decisions (from design doc + this session)
- Page = 64 cells (cache rows), metadata per (layer, kv-head, page):
  f16 min[256]+max[256] ≈ +1 KiB/token ≈ 320 MB @320K. Meta = separate
  per-layer tensors beside k_l (NOT serialized → slot format unchanged;
  restore ⇒ full meta rebuild from quantized cache).
- Bound validity under churn: meta only WIDENS on erase/ghosts (bounds stay
  valid, just less tight); cell REUSE goes through the write path → updated.
  Restore bypasses graph → rebuild flag. [verify A8/A9]
- Scoring: score(page) = Σ_d max(q_d·min_d, q_d·max_d) = relu(q)·max + neg-
  relu(q)·min. GQA: aggregate over the 4 q-heads/group; P0 decides per-kv-head
  vs per-layer page set (P1 default: ONE page set per layer — enables whole-row
  gather; per-head only if P0 shows a big recall gap).
- Selection = {sink pages} ∪ {recent pages incl. current partial} ∪ {top-N}.
  Force-list built host-side per step (host knows cell→pos), uploaded as tiny
  I32 input; scores biased +inf for forced pages inside the kernel.
- Custom CUDA ops (fork precedent: turbo types): GGML_OP_TURBO_META_UPDATE
  (mode a: fold new rotated K rows by cell idx; mode b: full rebuild reading
  quantized cache) + GGML_OP_TURBO_SPARSE_SELECT (q, meta_min, meta_max,
  force-list → row indices I32 [N*64], fused score+bias+topk+expand).
  Gather: extend GET_ROWS with same-type raw row copy (turbo4→turbo4,
  f16→f16 for mask). FA: existing MMA-turbo kernels over gathered contiguous
  turbo4 + gathered mask.
- Sparse ONLY when: decode-shaped batch (n_tokens ≤ 4), n_kv ≥ threshold
  (else dense), attention layers only, target model only (draft ctx small →
  threshold naturally excludes). Env: TURBO_SPARSE_DECODE=N tokens effective
  window (0/unset = off). Kascade anchor layers: P0 decides.
- Shapes static per graph (fixed N at a given depth band) → CUDA-graph safe;
  pos arrives via input tensor, not op_params.

## P1 concrete spec (locked from recon; P0 may adjust aggregation/anchors)
- Op A `GGML_OP_TURBO_META_UPDATE` — ggml_turbo_meta_update(ctx, meta,
  k_after_setrows, idxs). Returns view of meta (set_rows in-place pattern);
  src[0]=k set_rows RESULT (dependency ordering after cache write), src[1]=
  idxs, src[2]=meta. Modes in op_params: fold (dequant just-written rows
  centroid×norm → f16 CAS min/max into page meta) | rebuild (grid over all
  pages, non-atomic). Meta reads the QUANTIZED CACHE (byte-faithful with FA).
  Inserted whenever enabled (prefill too — maintenance).
- Op B `GGML_OP_TURBO_SPARSE_SELECT` — (q̃ [256,n_head,n_tok] f32, meta f16,
  forced I32 [n_forced_max, -1 pad]) → I32 [n_sel_rows]. Two-phase kernel:
  scores[n_pages] = Σ_heads Σ_dims max(q⁺·M, q⁻·m) summed over tokens (one
  page set per verify batch); +inf bias for forced; top n_sel_pages; bitmap→
  ascending compact → expand ×64. n_sel_pages multiple of 4 (⇒ rows %256=0 ⇒
  keeps GQA-packed MMA instances). n_pages/n_sel baked in op_params (graphs
  rebuild when n_kv pad-step changes anyway).
- Gather: `ggml_get_rows_keep` — same GGML_OP_GET_ROWS, dst type = src type;
  CUDA raw row-copy branch (int4 16B chunks; 528B rows = 33 chunks; also
  f16→f16 for mask); CPU supports_op DENY non-F32 dst. Mask gather: reshape
  mask [1, n_kv·n_tok] + flat idx built via cast/scale/repeat/add/cast
  (MSA-proven idiom) → [n_sel, n_tok] f16 contiguous.
- Meta tensors: per kv_layer `cache_meta_l%d` F16 [2·n_embd_k_gqa,
  kv_size/64] (4 KiB/page/layer = 1 KiB/token total @16 layers = 320 MB
  @320K); ctor gated on env TURBO_SPARSE_DECODE>0 && turbo4 K; mem_size
  ctx bump; NOT serialized; meta_dirty set by state_read_data → next graph
  inserts rebuild; zeroed meta benign (scores 0, masked if selected).
- build_attn (attn_kv overload) insertion between q-WHT and mha:
  active iff env N>0 && gtype != DECODER_MTP && n_stream==1 && K turbo4 &&
  n_tokens ≤ 4 && n_kv ≥ 2·N && head_dim ∈ {128,256}. Gathered k/v viewed
  [256, n_sel, n_kvh] (nb2 = 132B within-row head chunks) → build_attn_mha
  (inverse-WHT on output + hadamard un-rot unchanged). Reserve-time tg graph
  sees n_kv = kv_size → sparse topology pre-reserved ✓.
- Kascade anchors: llm_graph_context caches sel tensor per graph; layer il
  reuses anchor's sel if TURBO_SPARSE_ANCHOR_EVERY>1 (default from P0).
- FUSED GATE: flip GGML_TURBO_MMA_FUSED default ON in sparse branch
  (=0 kill-switch) — sparse NEEDS fused FA over gathered turbo4; prod env
  never set it (live prod runs VEC path today — see A3b).
- Per-kv-head selection fallback (if P0 demands): gather 132B head-chunks
  via view [256, kv_size·4] rows, idx j=h·n_sel+s → gathered viewable
  [256, n_sel, 4] ✓ same machinery, 4× idx count.

## Phase status
- P0 offline validator: IN PROGRESS — recon running; tool = eval-callback
  hook on FLASH_ATTN_EXT (dump q per probed step + dequantized K once per
  depth checkpoint + positions); python analysis → recall@N curves per layer,
  aggregation mode, page size, anchor sharing, sink/recent sizes, N schedule.
  Corpus: battery-style agentic text + code. Depth checkpoints 16K/32K/64K.
- P1 kernels: NOT STARTED
- P2 gates: NOT STARTED (battery seed-42 columns hold; KLD ≤ 0.0052; needle;
  depth curve ≥1.5× vs dense at same depth at 128K+ — note: design doc line
  "over the 84-89 @38K baseline" is ambiguous; the implementable gate is
  sparse ≥ 1.5× dense-at-same-depth for 128K+, plus no regression ≤64K)
- P3 tune: NOT STARTED

## Results log

### P0 round 1 (Quest min/max bounds) — probe 127K, corpus = code+docs+agent
Tool: examples/sparse-probe (q̃ at FA + raw turbo4 K dumps, ckpts 16K/32K/64K/
127K × 4 steps). Analysis: sparse-p0/analyze.py → sparse-p0-data/recall.json.
- Quest bound recall@N256(=16K tok) @127K: l00-07 0.77-0.93, l08-15 **0.45-0.71**.
- Oracle (perfect mass ranking) @N256 @127K: 0.86-0.97 (l15: 0.74) → problem
  is mostly SCORER, partly diffuseness.
- Root cause of bound failure: double rotation (learned Hadamard + WHT)
  flattens per-dim outlier structure → min/max envelopes non-discriminative.
  Design-doc assumption "rotated bounds may even tighten" is INVERTED.
- Aggregation per-layer vs per-kv-head: wash (≤0.05). Anchor sharing (Kascade):
  UNSAFE (l07←l04 = 0.44 vs 0.77 own). recent=64pg helps (+0.05-0.08).

### P0 round 2 (scorer bake-off) — analyze2.py → recall2.json
- **mean scorer (q·μ_page) is the winner**: rotation-immune (mean commutes
  with rotation). @N256 @127K late layers 0.75-0.81 (vs quest 0.44-0.56);
  @N512 0.75-0.93. mean+β·std: no gain (≤0.005) at any β → μ-only meta
  (0.5 KiB/token) would suffice.
- MTP verify-batch joint selection (one page set per forward, scores summed
  over rows): costs ≈ one N-doubling (joint@N256 ≈ single@N128).

### P0 round 3 (min-N sweep, mean scorer, sink 512 + recent 4K forced) —
analyze3.py → recall3.json. Min tokens for worst-step recall (joint/MTP):
- @65K depth: ≥0.85 needs 16-32K sel; ≥0.90 needs 16-48K — i.e. 25-75% of
  the cache. @127K: ≥0.85 needs 32-64K; ≥0.90 mostly NOT REACHABLE at 64K
  (=50% of cache). Page 32 ≈ page 64 (no gain). Oracle needs ~32-48K for
  0.90 → **even perfect selection reads 30-50%**.
- Ceiling math: attention ≈ 50-60% of step time at depth → oracle-recall-0.90
  speedup ≤ ~1.2× @128K, ≤ ~1.5× @320K — BELOW the ≥1.5×@128K+ gate, before
  any quality loss (and ledger already spirals at 128K dense).
- Architectural read: 16 attention layers among 48 DeltaNet carry ALL global
  routing → their mass is intrinsically spread on agentic/code content. The
  Quest/MInference concentration premise does not hold for this hybrid.

### P0 round 4 (320K-depth probe — the production-context kill shot)
Probe at c=327680 (prod YaRN config), ckpts 196608/315392, 2 steps; dumps
sparse-p0-data-320k/ (2.5 GB). Same analysis (mean scorer, joint/MTP):
- @197K: recall ≥0.85 needs 48-128K tokens (25-65% of cache); ≥0.90 needs
  64-128K, several layers unreachable at 128K.
- @315K: SIX of 16 layers (l02-l04, l13-l15) cannot reach even 0.85 while
  selecting 128K tokens = 40% of the cache. Fraction required stays ~40-60%
  at EVERY depth (65K/127K/197K/315K) — required-N scales ~linearly with
  depth. The design premise "8-16K effective window → 10-40× fewer reads at
  320K" is measured FALSE for this model.

### VERDICT — P0 kills the kernel build (2026-08-15)
Sparse decode (Quest-class page selection over the KV cache) is
architecturally unsuited to Qwen3.8-27B hybrid: its 16 global-attention
layers spread mass too widely on real agentic/code content. Even ORACLE
selection at recall 0.90 reads 30-50% of the cache → ceiling ≈1.2× @128K,
≈1.5× @320K BEFORE quality loss — under the ≥1.5×@128K+ adoption gate, with
high risk to the already-marginal ledger axis. P1-P3 will not be built.
Board rule satisfied: "implemented or tested to not be needed" — this is
tested-not-viable, with 4 rounds of measured evidence and the offline
validator + dumps retained for reproduction.

What survives the arc:
- GGML_TURBO_MMA_FUSED default flip + launcher env (the live-prod VEC
  discovery) — dense depth-decode win, validated separately (see below).
- examples/sparse-probe + sparse-p0/ analyzers — reusable attention-mass
  instrumentation (any future model swap should re-run P0 in ~30 min;
  a DENSE-attention model on this hardware may well pass where the
  hybrid fails).
- ggml_get_rows_keep (raw same-type gather) — harmless infra, kept on the
  feature branch.
Successor depth levers for the best-app goal (out of this arc's scope):
parked #22587 GDN decode rewrite (48 DeltaNet layers), MTP acceptance work,
ctx push now that VRAM law is fitted.

### FUSED-GATE ADOPTION (the arc's banked win) — paired A/B 2026-08-15
Same binary, env-only arms, full restarts between arms, ordering repeated,
depth probe = 38K/121K-token prefill + 700 greedy decode (MTP active,
ignore_eos), direct :8131, cache_prompt=false:
- @38K decode: OFF 101.6 / 101.5 vs ON 104.1 / 103.3 → **+2.2%**
- @121K decode: OFF 70.3 / 70.3 vs ON 76.5 / 76.3 → **+8.7%**
- prefill unchanged both depths (2618-2651 / 1675-1681) — decode-only path ✓
- win grows with n_kv (the removed cost = full-cache F16-dequant per verify
  step, linear in depth) → expect larger still at 200-320K.
ADOPTED: `export GGML_TURBO_MMA_FUSED=${GGML_TURBO_MMA_FUSED:-1}` in
start-long-38.sh (override-friendly; comment documents the A/B); in-tree
default flipped ON in commit 605e59b4d (branch feature/sparse-decode).
KLD gate: fused-ON = 0.005865 / top-1 98.3%; fused-OFF same binary =
**bit-identical 0.005865** — expected, KLD measures PROMPT logprobs =
prefill path; the gate is decode-only (Q≤4), so this methodology cannot see
it (and proves the gate's scope). The 0.00473→0.00586 drift vs the archived
ccmma baseline is binary evolution (b10448 sync + 24565) on the prefill
path — top-1 unchanged 98.3%, still excellent; noted, not gate-blocking
(the 0.0052 threshold was written against the same-binary premise that no
longer holds; behavioral gate = battery). Decode-path behavioral check =
trajectory battery @64K on fused-ON (below). Caveat on record: fused is NOT
token-identical to VEC (f16 reduction order, ~1-in-25 hard-tie flips) — set
=0 for strict-identity A/Bs; the 84-89 tok/s @38K in older docs are
VEC-path temp-1.0 probes; this A/B's greedy probes ran 101-104 @38K.
