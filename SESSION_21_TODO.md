# SESSION 21 — Profile, Optimize, Ship

## PART 1: nsight-compute Profiling
- [ ] Profile turbo3 short (VEC kernel)
- [ ] Profile turbo3 32K
- [ ] Profile turbo4 short
- [ ] Profile turbo2 short
- [ ] Profile turbo1.5 short
- [ ] Profile f16 short (baseline)
- [ ] Profile q8_0 short (baseline)
- [ ] Extract: occupancy, registers, shmem, bank conflicts, warp stalls, throughput
- [ ] Document in SESSION_21_PROFILE.md

## PART 2: Adaptive LUT
- [ ] Implement runtime LUT disable at short context
- [ ] Test turbo3 short recovery (target: 60.47)
- [ ] Test turbo4 short recovery (target: 59.06)
- [ ] Verify 32K not regressed
- [ ] Commit

## PART 3: Profile-Driven Micro-Optimizations
- [ ] Implement fixes based on profiling data
- [ ] Regression test each fix
- [ ] Commit each independently

## PART 4: FP4 Tensor Core Prototype
- [ ] Test E2M1 Q quantization precision
- [ ] If acceptable: build turbo1.5 FP4 prefill kernel
- [ ] If not: document and move on

## PART 5: Multi-Model Validation
- [ ] Find available models
- [ ] Test decode speed on new model
- [ ] Test PPL on new model
- [ ] Verify D=64 heads work

## PART 6: Ship Preparation
- [ ] Squash commits
- [ ] README.md rewrite
- [ ] Discussion post draft
- [ ] PR preparation
