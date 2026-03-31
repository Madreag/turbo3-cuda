# Session 27C — Feature Parity: Adopt TheTom's Bug Fix + Auto Boundary V

## READ FIRST

1. **Read `AGENTS.md`** — follow ALL rules.
2. You are on branch **`session/27-quality`** (or create from `release/cuda-optimized`).
3. GPU: RTX 5090 32GB (SM120). CUDA 12.8. WSL2.
4. These are TWO small changes — should take ~1 hour total.

---

## CONTEXT

We compared TheTom's latest repo (`/home/erol/ai/thetom/llama-cpp-turboquant/` branch `feature/turboquant-kv-cache`) against ours. Most features we already have (block-128, turbo4 port, asymmetric K/V, D=640, 36 K×V combos). Two things we're missing:

1. A bug fix for KV state serialization (crash on llama-server slot reuse)
2. Auto Boundary V when turbo2-V is used (free quality recovery)

Both are small, proven changes from TheTom's repo. Adopt them before S28 (community post + PR).

---

## TASK 1: KV State Serialization Fix

### The Bug
`state_write_data` and `state_read_data` in `llama-kv-cache.cpp` use `hparams.n_embd_k_gqa(il)` for `ggml_row_size`. But turbo types with zero-padded heads (e.g., GLM-4.7: D=576→640) have a PADDED tensor width. For turbo4 (QK=128), `576 % 128 != 0` → `ggml_row_size` assertion failure during prompt cache save on llama-server slot reuse.

### The Fix
Use `k->ne[0]` / `v->ne[0]` (actual padded tensor width) instead of `hparams` values. Four locations in `llama-kv-cache.cpp`.

### Reference
TheTom's commit `89d267c0b`. The exact diff is at:
```
/home/erol/ai/thetom/llama-cpp-turboquant/
```
Run:
```bash
cd /home/erol/ai/thetom/llama-cpp-turboquant && git show 89d267c0b -- src/llama-kv-cache.cpp
```

### What To Change

In `src/llama-kv-cache.cpp`, find the 4 serialization functions (K write, K read, V write, V read). In each one:

```cpp
// BEFORE (crashes on padded turbo types):
const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

// AFTER (uses actual tensor width):
auto * k = layer.k_stream[cr.strm];
const uint32_t n_embd_k_gqa = (uint32_t) k->ne[0];
```

Same pattern for V:
```cpp
// BEFORE:
const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

// AFTER:
auto * v = layer.v_stream[cr.strm];
const uint32_t n_embd_v_gqa = (uint32_t) v->ne[0];
```

### Testing
This bug only triggers with:
- llama-server (not llama-cli)
- Slot reuse / prompt cache save
- Models with non-128-aligned head dims (GLM-4.7)
- turbo4 KV type

Since we don't have a GLM-4.7 model, verify the change compiles and doesn't break PPL:
```bash
./build/bin/llama-perplexity -m $MODEL -f $WIKI -c 512 -ctk turbo3 -ctv turbo3 -fa on --chunks 8 -ngl 99 --no-mmap
```
PPL must still be 6.8522.

### Commit
```
fix: KV state serialization uses padded tensor width

Turbo types with zero-padded heads (e.g. GLM-4.7 D=576→640) crash
ggml_row_size assertion during llama-server prompt cache save.
Use k->ne[0] / v->ne[0] instead of hparams values.

Credit: signalnine (TheTom's repo commit 89d267c0b)
```

---

## TASK 2: Auto Boundary V for turbo2

### What
When user sets `-ctv turbo2`, auto-enable layer-adaptive mode 7 (Boundary V). This protects the first 2 + last 2 layers with q8_0-V while compressing the rest with turbo2-V. TheTom's data shows 37-91% quality recovery across 4 models with zero speed penalty.

User can opt-out with `TURBO_LAYER_ADAPTIVE=0`.

### Reference
TheTom's commit `5364f8a1d`. Run:
```bash
cd /home/erol/ai/thetom/llama-cpp-turboquant && git show 5364f8a1d -- src/llama-kv-cache.cpp
```

### What To Change

In `src/llama-kv-cache.cpp`, in the KV cache constructor where `TURBO_LAYER_ADAPTIVE` is read:

```cpp
// Current code:
static const int adaptive_mode = []() {
    const char * env = getenv("TURBO_LAYER_ADAPTIVE");
    int mode = env ? atoi(env) : 0;
    if (mode > 0) {
        LLAMA_LOG_INFO("llama_kv_cache: layer-adaptive mode %d enabled\n", mode);
    }
    return mode;
}();

// New code:
static const int adaptive_mode = [&]() {
    const char * env = getenv("TURBO_LAYER_ADAPTIVE");
    if (env) {
        int mode = atoi(env);
        if (mode > 0) {
            LLAMA_LOG_INFO("llama_kv_cache: layer-adaptive mode %d enabled (env)\n", mode);
        }
        return mode;
    }
    // Auto-enable Boundary V (mode 7) when V is turbo2 and model has enough layers
    if (type_v == GGML_TYPE_TURBO2_0 && hparams.n_layer >= 8) {
        LLAMA_LOG_INFO("llama_kv_cache: Boundary V auto-enabled for turbo2-V (opt-out: TURBO_LAYER_ADAPTIVE=0)\n");
        return 7;
    }
    return 0;
}();
```

**Key change**: The lambda captures `[&]` instead of `[]` so it can access `type_v` and `hparams`. The env var check comes first (explicit always wins). Auto-enable only if `type_v == TURBO2_0` and the model has 8+ layers.

**IMPORTANT**: Check what mode 7 does in our LA implementation. TheTom uses mode 7 = "first 2 + last 2 layers q8_0-V". Verify this matches our mode numbering. Our modes are in `AGENTS.md`:
- Mode 7: `last8 K, none V` — that's K-only tail, NOT Boundary V!
- Mode 12: `none K, first4+last4 V` — THIS is our Boundary V equivalent

**If our mode 12 = TheTom's mode 7 concept**, use mode 12 instead:
```cpp
if (type_v == GGML_TYPE_TURBO2_0 && hparams.n_layer >= 8) {
    LLAMA_LOG_INFO("llama_kv_cache: Boundary V auto-enabled for turbo2-V (opt-out: TURBO_LAYER_ADAPTIVE=0)\n");
    return 12;  // Our mode 12 = first4+last4 V protection
}
```

### Testing
```bash
# Test turbo2 with auto Boundary V
./build/bin/llama-perplexity -m $MODEL -f $WIKI -c 512 -ctk turbo2 -ctv turbo2 -fa on --chunks 8 -ngl 99 --no-mmap
```

Should show the "Boundary V auto-enabled" log message. PPL should be between pure turbo2 (7.080) and turbo3 (6.852) — closer to turbo3 means Boundary V is working.

Also verify opt-out works:
```bash
TURBO_LAYER_ADAPTIVE=0 ./build/bin/llama-perplexity -m $MODEL -f $WIKI -c 512 -ctk turbo2 -ctv turbo2 -fa on --chunks 8 -ngl 99 --no-mmap
```
Should NOT show the auto-enable message. PPL should be pure turbo2 (7.080).

### Commit
```
feat: auto-enable Boundary V when turbo2-V is used

Protects first 4 + last 4 layers with q8_0-V, rest turbo2-V.
37-91% quality recovery across 4 models (TheTom's data).
Zero speed penalty. Opt-out: TURBO_LAYER_ADAPTIVE=0.

Credit: TheTom (concept + Metal implementation, commit 5364f8a1d)
```

---

## TASK 3: Rebuild, Regression Check, Push

After both changes:

```bash
# Build
/home/erol/miniconda3/envs/tq/bin/cmake --build build -j$(nproc)

# PPL check — turbo3 (should be unchanged, these changes don't affect turbo3)
./build/bin/llama-perplexity -m $MODEL -f $WIKI -c 512 -ctk turbo3 -ctv turbo3 -fa on --chunks 8 -ngl 99 --no-mmap

# PPL check — turbo2 with auto Boundary V (should improve vs pure turbo2)
./build/bin/llama-perplexity -m $MODEL -f $WIKI -c 512 -ctk turbo2 -ctv turbo2 -fa on --chunks 8 -ngl 99 --no-mmap

# Speed check — turbo3 short + 32K (warmup then measure)
./build/bin/llama-bench -m $MODEL -fa 1 -ctk turbo3 -ctv turbo3 -d 0 -ngl 99 -t 1 -r 1 -p 0 -n 128 -mmp 0 > /dev/null 2>&1
./build/bin/llama-bench -m $MODEL -fa 1 -ctk turbo3 -ctv turbo3 -d 0 -ngl 99 -t 1 -r 3 -p 0 -n 128 -mmp 0

# Generation check — turbo2 on Qwen 9B
./build/bin/llama-server -m /home/erol/ai/turboquant/models/Qwen3.5-9B-Q8_0.gguf \
  -ctk turbo2 -ctv turbo2 -fa on -ngl 99 -c 4096 --port 8090 --no-mmap --log-disable &
sleep 30
curl -s http://localhost:8090/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":200,"temperature":0}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'][:80])"
pkill -f "llama-server.*8090"
```

### Push
```bash
git push myfork session/27-quality
git checkout release/cuda-optimized
git merge session/27-quality --no-edit
git push myfork release/cuda-optimized
```

---

## SUCCESS CRITERIA

- [ ] KV state serialization fix applied (4 locations in llama-kv-cache.cpp)
- [ ] Auto Boundary V for turbo2 implemented (correct mode number for our LA system)
- [ ] turbo3 PPL unchanged (6.8522)
- [ ] turbo2 PPL improved with auto Boundary V
- [ ] turbo2 opt-out works (TURBO_LAYER_ADAPTIVE=0)
- [ ] Speed unchanged
- [ ] Generation works
- [ ] Pushed to release/cuda-optimized

---

## ESTIMATED EFFORT

| Task | Time |
|------|------|
| KV serialization fix | 15 min |
| Auto Boundary V | 30 min |
| Regression tests | 15 min |
| **Total** | **~1 hour** |
