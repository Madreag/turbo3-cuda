# Custom-quant lab — iteration ledger (R1 arc)

Goal: ≤~20GB weights at gate-passing quality (bar: tail-KLD mean ≤~0.012-class,
top-1 ≥~95% vs kld38 f16-KV ref @ yarn 1.25). Baseline prod Q6_K+turbo4:
0.0060 / 96.2%. imatrix_v1.dat = 349K tokens of agent-shaped local corpus
(calib_v1.txt builder in this dir). Source: D:\spill\qwen38-bf16-src (52G).
Workflow: convert from D: -> quantize (imatrix) -> gate_server.sh -> KLD-157.
Artifacts deleted after each verdict (recreatable ~10-15 min); summaries kept.

| arm | recipe | size | VRAM@320K | mean KLD | top-1 | MTP acc | verdict |
|---|---|---|---|---|---|---|---|
| (prod) | Q6_K | 21.4G | 31,255 MiB | 0.0060 | 96.2% | ~.75 | SHIPPED baseline |
| community | utautako NVFP4-Q8attn | 17.8G | 28,423 | 0.0693 | 82.8% | .78 | REJECTED |
| tqmix-v1 | Q6_K + ffn_gate/up=nvfp4 + imatrix | 19G | 28,580 | 0.0561 | 91.1% | .79 | REJECTED (FP4 encode immature; attn-protection helped top-1) |
| tqmix-v2 | Q5_K_M + imatrix | 19G | 29,285 | 0.0197 | 88.5% | .75 | REJECTED (K-quant FFN fixed mean; unprotected attn hurt top-1) |
| tqmix-v3 | PROPOSED: Q6_K base + ffn_gate/up/down=q5_k + imatrix (~20G) | — | — | est ~0.010-0.015 | est 92-94% | — | NEXT — combines v1 attn-protection with v2 K-quant FFN |

Honest trajectory note: even v3 may land short of 95% top-1 — the tail
instrument is strict and ANY sub-Q6 weight quant diverges measurably. If v3
misses, the R1 verdict is "Q6_K stays; ~2-3GB VRAM is not purchasable at
gate quality," which is itself a valuable closed answer.
