# Session 26 Part 2 — BUGFIX: Fix Turbo Generation Failures Before Anything Else

## READ FIRST (MANDATORY)

1. **Read `AGENTS.md`** — contains ALL rules, architecture, dead ends, file locations, benchmarking protocol. Follow it EXACTLY.
2. **Read this entire prompt** before starting any work.
3. You are on branch **`session/26-blast`**.
4. **DO NOT** work on block-128, bpv corrections, or any optimization until BOTH bugs are fixed and verified.
5. GPU is an RTX 5090 32GB (SM120). CUDA 12.8. WSL2 Ubuntu 24.04.

---

## THE SITUATION

We have TWO generation bugs that make the repo broken for real users. Everything else (block-128, optimizations, community posts) is pointless until these are fixed.

### Bug 1: turbo1.5 Generates Garbage on SM86 (and Likely All GPUs)

**Evidence**: 3090 Ti (SM86) NIAH test results from `/mnt/c/vaults/dump/3090ti_niah_single.md`:
- q8_0: 84.8% accuracy (56/66 pass) — working baseline
- turbo3: 86.4% (57/66) — working, actually BEATS q8_0
- turbo2: 78.8% (52/66) — working, minor degradation at 64K+
- **turbo1.5: 0.0% (0/66) — EVERY test failed at EVERY depth and EVERY context**

This is not a quality issue. 0/66 means the model output is fundamentally wrong. Something in the turbo1.5 VEC decode path produces incorrect attention values.

**Note**: turbo1.5 PPL is fine (7.312 at ctx=512 = baseline). PPL uses the MMA prefill path, not the VEC decode path. The bug is in VEC decode only.

### Bug 2: SM120 D=256 Turbo Types Generate Empty Output

**Evidence**: 5090 (SM120) quality testing:
- q8_0 with FA on: generates correct answers on Qwen 3.5 (D=256)
- ANY turbo type with FA on: generates empty `content` field on Qwen 3.5 (D=256)
- ANY turbo type with FA OFF: generates correctly (falls back to mul_mat attention)
- ALL turbo types on Llama-3.3-8B (D=128): generate correctly with FA on
- ALL turbo types on SM86 (3090 Ti): generate correctly (turbo3/turbo2 at least)

**Conclusion**: The bug is specific to SM120 + D=256 + turbo VEC FA kernel.

**Note on Qwen 3.5 thinking**: Qwen 3.5 is a thinking model. It needs `max_tokens: 200+` to finish `<think>` reasoning before producing content. With `max_tokens: 50` content is empty — this is NORMAL, not a bug. The actual bug was confirmed with `max_tokens: 500` and FA OFF as control.

---

## BUG 1 INVESTIGATION: turbo1.5

### What turbo1.5 Is
- 2.0 bpv ternary quantization: each value is {-C, 0, +C} where C = 0.107632
- Encoded as trits (5 per byte, 3^5=243 ≤ 255)
- Block struct: `block_turbo1_5` = 16 bytes per 32 values (norm + 7 trit bytes + 7 pad bytes)
- NO LUT scoring — uses `vec_dot_fattn_vec_KQ_turbo1_5` (direct centroid multiply)
- Uses `TURBO1_5_TRIT_LUT` for trit decoding (constant memory, 5×256 lookup table)
- Sparse V threshold: 1e-2 (most aggressive)

### Files To Read
```
ggml/src/ggml-cuda/fattn-common.cuh    — vec_dot_fattn_vec_KQ_turbo1_5() (line ~506)
                                        — dequantize_V_turbo1_5()
ggml/src/ggml-cuda/fattn-vec.cuh       — turbo1.5 path in the main KQ scoring section (line ~396, the `else` branch)
                                        — V dequant path (sparse V threshold at 1e-2)
ggml/src/ggml-cuda/turbo-quant.cuh     — TURBO1_5_TRIT_LUT, trit decode functions
ggml/src/ggml-common.h                 — block_turbo1_5 struct (line ~333)
ggml/src/ggml-cuda/set-rows.cu         — turbo1.5 SET_ROWS encoding kernel
```

### Investigation Steps

**Step 1**: Verify the bug reproduces on the 5090 (SM120).
```bash
# Start server with turbo1.5
./build/bin/llama-server -m /home/erol/ai/turboquant/models/Qwen3.5-9B-Q8_0.gguf \
  -ctk turbo1.5 -ctv turbo1.5 -fa on -ngl 99 -c 4096 --port 8090 --no-mmap --log-disable &

# Wait for server, then test
curl -s http://localhost:8090/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is 2+2? Just the number."}],"max_tokens":200,"temperature":0}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); c=d['choices'][0]['message']; print('Content:', repr(c.get('content','')[:100]))"
```
- If content is empty or garbage → Bug confirmed on SM120 too
- If content is correct → Bug is SM86-specific (different codegen)

**Step 2**: Test turbo1.5 K only (asymmetric — isolates K path vs V path):
```bash
# turbo1.5 K + q8_0 V (tests K scoring only)
pkill -f llama-server; sleep 3
./build/bin/llama-server -m /home/erol/ai/turboquant/models/Qwen3.5-9B-Q8_0.gguf \
  -ctk turbo1.5 -ctv q8_0 -fa on -ngl 99 -c 4096 --port 8090 --no-mmap --log-disable &
# Same curl test...

# q8_0 K + turbo1.5 V (tests V dequant only)
pkill -f llama-server; sleep 3
./build/bin/llama-server -m /home/erol/ai/turboquant/models/Qwen3.5-9B-Q8_0.gguf \
  -ctk q8_0 -ctv turbo1.5 -fa on -ngl 99 -c 4096 --port 8090 --no-mmap --log-disable &
# Same curl test...
```
This tells you if the bug is in K scoring (vec_dot_turbo1_5) or V dequant (dequantize_V_turbo1_5).

**Step 3**: If K path fails — examine `vec_dot_fattn_vec_KQ_turbo1_5`:
- Check trit decoding: does `TURBO1_5_TRIT_LUT[byte_val][trit_pos]` return correct {-1, 0, +1}?
- Check centroid multiply: `float(trit) * C * norm` where C = 0.107632
- Check Q format: turbo1.5 vec_dot reads `Q_v` (float/half2), NOT `Q_q8` (q8_1). Is `Q_v` initialized correctly for the turbo1.5 template?
- **IMPORTANT**: turbo4 and turbo1.5 vec_dot both read `Q_v` directly (not q8_1). The `GGML_UNUSED(Q_q8)` at the top of their vec_dot functions confirms this. But the VEC kernel quantizes Q to q8_1 for ALL `Q_q8_1` types. Does the Q_v (Q_reg) array get populated correctly when `Q_q8_1 = true`?

**Step 4**: If V path fails — examine `dequantize_V_turbo1_5`:
- Check trit decoding from V blocks
- Check norm multiply
- Check sparse V threshold (1e-2 is very aggressive — try 1e-6 as control)

**Step 5**: Check if the 1e-2 sparse V threshold is the problem:
```cuda
// In fattn-vec.cuh, temporarily change:
constexpr float sparse_v_threshold_f = V_is_low_bpv ? 1e-2f : 5e-3f;
// To:
constexpr float sparse_v_threshold_f = V_is_low_bpv ? 1e-6f : 5e-3f;
```
Rebuild and retest. If turbo1.5 generates correctly with 1e-6, the threshold was too aggressive.

### Likely Root Causes (Most Probable First)

1. **Q_v uninitialized for turbo1.5 vec_dot**: The VEC kernel's Q preparation fills `Q_i32` and `Q_ds` (for q8_1 types) but turbo1.5's vec_dot reads `Q_v` (Q_reg). If `Q_q8_1 = true` for turbo types, Q_reg may not be filled on the non-f16 path. Check lines 156-262 of fattn-vec.cuh — does Q_reg get initialized when `Q_q8_1 = true`?

2. **Sparse V at 1e-2 skipping too much**: At 1e-2 threshold, turbo1.5 V skips all positions where attention weight < 0.01. On a 9B model at short context, this might skip positions that matter. BUT the 3090 Ti agent tested at ALL contexts (4K through 128K) and ALL depths — 0/66 pass. If threshold were the issue, at least short context would work.

3. **Trit LUT indexing bug**: The `TURBO1_5_TRIT_LUT` is a 5×256 constant memory table. If the table is corrupted or the byte indexing is wrong, every trit decode produces garbage.

4. **Block padding issue**: `block_turbo1_5` has 7 bytes of `_pad`. If the padding is read as data or the struct alignment is wrong, the trit bytes are misread.

---

## BUG 2 INVESTIGATION: SM120 D=256 Empty Output

### What We Know
- Works: SM86 + D=256, SM120 + D=128, FA off + any config
- Fails: SM120 + D=256 + FA on + any turbo type
- PPL is correct (MMA prefill path works)
- llama-bench tok/s is normal (kernel runs, just wrong output)

### Files To Read
```
ggml/src/ggml-cuda/fattn-vec.cuh       — THE kernel. D is a template parameter.
                                        — LUT: turbo_lut[D][lut_stride] = turbo_lut[256][9] at D=256
                                        — Q quantization: D/sizeof(int) iterations
                                        — KQ scoring: D/QK_TURBO3 = 256/32 = 8 blocks per position
ggml/src/ggml-cuda/fattn.cu            — Template dispatch: verify D=256 turbo templates exist
ggml/src/ggml-cuda/template-instances/ — Check for D=256 turbo3/turbo2/turbo1.5 template files
```

### Investigation Steps

**Step 1**: Verify the bug still reproduces (it may have been fixed by S26 Part 1 work).
```bash
./build/bin/llama-server -m /home/erol/ai/turboquant/models/Qwen3.5-9B-Q8_0.gguf \
  -ctk turbo3 -ctv turbo3 -fa on -ngl 99 -c 4096 --port 8090 --no-mmap --log-disable &
# Wait, then test with max_tokens:200
```

**Step 2**: Test D=128 as control (should work):
```bash
./build/bin/llama-server -m /home/erol/ai/turboquant/models/allura-forge_Llama-3.3-8B-Instruct-Q6_K.gguf \
  -ctk turbo3 -ctv turbo3 -fa on -ngl 99 -c 4096 --port 8090 --no-mmap --log-disable &
# Same test — should produce correct content
```

**Step 3**: If D=256 still fails, check the LUT:
- LUT size: `half turbo_lut[256][9]` = 256 × 9 × 2 = 4,608 bytes shared memory
- Total shmem per block: 4,608 (LUT) + KQ storage. Is this exceeding SM120's per-block shmem limit?
- SM120 has 228 KB shared memory per SM, but per-block limit is 101 KB. 4,608 bytes is fine.

**Step 4**: Check LUT construction loop at D=256:
```cuda
for (int d = tid; d < D; d += nthreads) {  // D=256, nthreads=128 → 2 iterations per thread
    const float q_val = Q_f[d] * scale;
    for (int c = 0; c < n_centroids_lut; c++) {
        turbo_lut[d][c] = __float2half(q_val * centroids_ptr[c]);
    }
}
```
At D=256: each thread writes 2 LUT rows (d and d+128). At D=128: each thread writes 1 row. The D=256 path is fine mathematically but check for bank conflicts or race conditions.

**Step 5**: Check KQ scoring at D=256:
```cuda
for (int d0 = 0; d0 < D; d0 += 8 * nthreads_KQ) {
    const int d_base = d0 + (threadIdx.x % nthreads_KQ) * 8;
    const int ib = d_base / QK_TURBO3;    // 256/32 = 8 blocks
    const int jj = d_base % QK_TURBO3;
```
At D=256 with nthreads_KQ=8: the loop runs `256 / (8*8) = 4` iterations. Each thread processes d_base values: 0,8,16,24 then 64,72,80,88 then 128,136,144,152 then 192,200,208,216. Check that `ib` correctly maps to blocks 0-7 and `jj` maps within each block.

**Step 6**: Check if q8_1 Q quantization is correct at D=256:
```cuda
for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += nthreads_quantize) {
    quantize_q8_1_to_shared<float2, nthreads_quantize>(...)
}
```
At D=256: `D/sizeof(int) = 64` iterations. With `nthreads_quantize = min(64, 32) = 32`: 2 iterations. This should be correct but verify the Q buffer `KQ[j*D]` has enough space for D=256.

**Step 7**: Check `KQ` shared memory size:
```cuda
constexpr int ne_KQ = ncols * D;  // 1 * 256 = 256 floats = 1024 bytes
```
Plus the LUT (4,608 bytes). Total shmem: ~5,632 bytes. Should be fine.

**Step 8**: Nuclear option — disable half-precision LUT for D=256 and test:
```cuda
// In LUT construction, change:
turbo_lut[d][c] = __float2half(q_val * centroids_ptr[c]);
// To:
// Use float LUT for D=256 as diagnostic
```
But this requires changing the LUT type from `half` to `float`, which is a larger change. Instead, try disabling LUT entirely for D=256:
```cuda
constexpr int n_centroids_lut = (D <= 128 && type_K == GGML_TYPE_TURBO3_0) ? 8 :
                                (D <= 128 && type_K == GGML_TYPE_TURBO2_0) ? 4 : 0;
```
If D=256 works without LUT (falls through to vec_dot), the bug is in the LUT path at D=256.

---

## TESTING PROTOCOL

For each fix attempt:

1. **Generation test** (PASS = non-empty correct content):
```bash
curl -s http://localhost:8090/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":200,"temperature":0}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('Content:', repr(d['choices'][0]['message']['content'][:100]))"
```

2. **PPL regression** (must match baseline):
```bash
MODEL=/home/erol/ai/turboquant/models/opus-v2-Q6_K.gguf
WIKI=$(find /home/erol/ai/turboquant -name "wiki.test.raw" 2>/dev/null | head -1)
./build/bin/llama-perplexity -m $MODEL -f $WIKI -c 512 -ctk turbo3 -ctv turbo3 -fa on --chunks 8 -ngl 99 --no-mmap
```

3. **Speed regression** (must match baseline):
```bash
./build/bin/llama-bench -m $MODEL -fa 1 -ctk turbo3 -ctv turbo3 -d 0 -ngl 99 -t 1 -r 3 -p 0 -n 128 -mmp 0
```

---

## VERIFICATION MATRIX

After BOTH bugs are fixed, run this full verification:

| Test | turbo3 | turbo2 | turbo1.5 | q8_0 |
|------|:------:|:------:|:--------:|:----:|
| Generation D=128 (Llama 8B) | | | | |
| Generation D=256 (Qwen 9B) | | | | |
| PPL ctx=512 (27B Q6_K) | | | | |
| Speed short (27B Q6_K) | | | | |
| Speed 32K (27B Q6_K) | | | | |

Every cell must be PASS before moving to block-128 or any other work.

---

## MODELS

| Model | Path | Use For |
|-------|------|---------|
| Qwen 3.5 9B Q8_0 | `models/Qwen3.5-9B-Q8_0.gguf` | Generation bug testing (D=256, fast load) |
| Llama-3.3-8B Q6_K | `models/allura-forge_Llama-3.3-8B-Instruct-Q6_K.gguf` | D=128 control test |
| Qwen 3.5 27B Q6_K | `models/opus-v2-Q6_K.gguf` | PPL + speed regression |

Model paths relative to `/home/erol/ai/turboquant/`.

---

## SUCCESS CRITERIA

Session 26 Part 2 is DONE when:
- [ ] Bug 1 root cause identified
- [ ] Bug 1 fixed — turbo1.5 generates correct output on Qwen 3.5 9B
- [ ] Bug 2 root cause identified
- [ ] Bug 2 fixed — ALL turbo types generate correct output on SM120 with D=256
- [ ] Full verification matrix passes (all cells PASS)
- [ ] PPL unchanged for all types
- [ ] Speed unchanged for all types
- [ ] Fix committed with root cause explanation

**DO NOT proceed to block-128, bpv corrections, or any S27/S28 work until this matrix is green.**
