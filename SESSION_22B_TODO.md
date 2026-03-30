# Session 22B — Breakthrough: Bug Fixes, Novel Optimizations

## Part 1: SM89 Sink Bug Fix (P0)
- [ ] Read turbo-sink.cu and fattn-vec.cuh sink dispatch
- [ ] Apply alignment fix to cudaMemcpyAsync source variables
- [ ] Rebuild and test SINK_SIZE={0,1,2,4,8,16} on SM120
- [ ] Regression test primary model (speed + PPL)
- [ ] Commit with root cause documentation

## Part 2: Long-Context Benchmarks on Validation Models (P0)
- [ ] Phi-4-mini (D=128) at 32K: turbo3, turbo2, q8_0
- [ ] Llama-3.3-8B (D=128) at 32K: turbo3, turbo2, q8_0
- [ ] Gemma-3-12B (D=256) at 32K or 16K: turbo3, turbo2, q8_0
- [ ] Update SESSION_22_MULTIMODEL.md with results

## Part 3: VEC Kernel Micro-Optimizations (P1)
- [ ] 3A: Sparse V branch divergence fix (__ballot_sync)
- [ ] 3D: Partial V dequant unroll (#pragma unroll 4)
- [ ] 3E: KQ_max reduction optimization
- [ ] 3B: 8-at-a-time LUT scoring
- [ ] 3C: Software pipeline K/V double-buffer (if time)

## Part 4: L2 Cache KV Prefetch (P1)
- [ ] Step 1: L2 persistence for centroid tables
- [ ] Step 2: Tile-level K prefetch with cp.async.bulk.prefetch.L2
- [ ] Benchmark impact at short and 32K

## Part 5: Adaptive LUT for D=256 (P1)
- [ ] Benchmark Gemma-3-12B turbo3 vs q8_0 at 32K
- [ ] If LUT is net negative: disable for D=256
- [ ] Benchmark and commit

## Part 6-7: V Sparse Skip + KV Norm + Auto-Asymmetric (P2)
- [ ] V sparse threshold tuning for turbo1.5
- [ ] KV norm measurement or document K=turbo4/V=q8_0 finding

## Part 8-9: Final Benchmarks + Vault + Push
- [ ] Full regression suite after all changes
- [ ] Update AGENTS.md, vault files
- [ ] Push to myfork
