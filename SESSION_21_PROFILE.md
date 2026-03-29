# Session 21 Profiling Results

## Method
nsight-compute ERR_NVGPUCTRPERM on WSL2 (Windows GPU driver, not Linux modprobe).
Used `cuobjdump --dump-resource-usage` for static register/shmem analysis instead.

## VEC Kernel Resource Usage (D=128, ncols=1, softcap=0, V_DOT2 half path)

| Type | REG | SHARED | Blocks/SM | Warps/SM | Occupancy |
|------|----:|-------:|:---------:|:--------:|:---------:|
| f16 | 168 | 9472 | 3 | 12 | 25% |
| q8_0 | 178 | 5888 | 2 | 8 | 17% |
| turbo3 | 180 | 5888 | 2 | 8 | 17% |
| turbo4 | 180 | 5888 | 2 | 8 | 17% |
| turbo2 | 168 | 5888 | 3 | 12 | 25% |
| turbo1.5 | 194 | 5888 | 2 | 8 | 17% |

SM120: 64K registers/SM, 128 threads/block, 48 max warps/SM.
Threshold: 170 regs = 21760/block → 65536/21760 = 3.01 → 3 blocks.
Above 170 regs: only 2 blocks. Below 170: 3 blocks (50% more occupancy).

## KEY FINDING: Register Pressure Is the #1 Bottleneck

turbo2 = 168 regs → 3 blocks → 25% occupancy → long-context champion.
turbo3 = 180 regs → 2 blocks → 17% occupancy → 12 registers over the cliff.
turbo4 = 180 regs → same as turbo3.
turbo1.5 = 194 regs → worst occupancy, furthest from cliff.

**If turbo3/turbo4 can drop 12 registers (180→168), they get 50% more warps.**

## Where Are the Extra Registers?

turbo3 has 180 regs vs turbo2's 168. The 12-register difference comes from:
1. turbo3 decodes 3-bit indices (qs + signs byte) vs turbo2's 2-bit (qs only)
   - Extra signs byte read + shift/mask = ~2-3 registers
2. turbo3 LUT has 8 centroids vs turbo2's 4
   - Larger LUT means more loop iterations = more live registers in unrolled loops = ~4-5 regs
3. The 4-at-a-time scoring decodes 4 turbo3 indices (idx0-idx3) from qs+signs
   - 4 idx vars + 4 intermediate shifts = ~4 regs

turbo4 at 180: 4-bit nibble extraction is simpler than turbo3 3-bit, but 16 centroids → more LUT entries.

turbo1.5 at 194: trit extraction (LUT read from __device__ array) + norm*C multiplication chain.

## Optimization Targets

### Priority 1: Get turbo3/turbo4 to ≤170 regs
- Remove turbo3 LUT: saves ~12 regs from LUT array and scoring unroll → but kills +7% at 32K
- Reduce unroll: `#pragma unroll 1` on LUT scoring loop → fewer live regs but more loop overhead
- Share computations: factor norm out of 4-element accumulation
- Use `__launch_bounds__(128, 3)` to force compiler to target 3 blocks

### Priority 2: Get turbo1.5 to ≤170 regs
- Currently at 194 — needs 24 register reduction
- Much harder — may need to restructure trit extraction

### Priority 3: Verify LUT shared memory isn't also limiting
- turbo3/turbo4 SHARED=5888 fits easily in 99KB
- NOT the bottleneck — registers are the bottleneck
