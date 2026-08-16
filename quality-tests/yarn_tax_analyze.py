#!/usr/bin/env python3
"""Yarn-tax analysis: KLD(reference-scale || arm-scale) per station, same
quant type across arms so quantization error cancels. Reference = the
no-yarn arm of the same type."""
import json, math, sys
from pathlib import Path

D = Path("/home/erol/ai/turboquant/turboquant-g1/quality-tests/yarntax")

def kld(p_lp, q_lp):
    if not p_lp or not q_lp:
        return None, None
    toks = set(p_lp) | set(q_lp)
    MIN = -100.0
    p = {t: math.exp(p_lp.get(t, MIN)) for t in toks}
    q = {t: math.exp(q_lp.get(t, MIN)) for t in toks}
    ps, qs = sum(p.values()), sum(q.values())
    if not ps or not qs:
        return None, None
    k = sum((p[t]/ps) * math.log((p[t]/ps) / (q[t]/qs))
            for t in toks if p[t]/ps > 1e-30 and q[t]/qs > 1e-30)
    top = int(max(p_lp, key=p_lp.get) == max(q_lp, key=q_lp.get))
    return k, top

def load(label):
    return json.load(open(D / f"stations_{label}.json"))["stations"]

def compare(ref_label, arm_label):
    ref, arm = load(ref_label), load(arm_label)
    print(f"\n== {arm_label} vs {ref_label} (yarn tax) ==")
    for st in ref:
        if st not in arm:
            continue
        ks, tops = [], []
        for b in ref[st]:
            k, t = kld(ref[st][b], arm[st].get(b, {}))
            if k is not None:
                ks.append(k); tops.append(t)
        if not ks:
            print(f"  station {st}: no data"); continue
        ks_s = sorted(ks)
        p99 = ks_s[min(len(ks_s)-1, max(0, math.ceil(0.99*len(ks_s))-1))]
        print(f"  station {st:>7}: mean {sum(ks)/len(ks):.4f}  p99 {p99:.4f}  "
              f"max {ks_s[-1]:.4f}  top1 {100*sum(tops)/len(tops):.0f}%  n={len(ks)}")

for ref, arms in [("t4_none", ["t4_125", "t4_156"]),
                  ("f16_none", ["f16_125", "f16_156"])]:
    if not (D / f"stations_{ref}.json").exists():
        continue
    for a in arms:
        if (D / f"stations_{a}.json").exists():
            compare(ref, a)
# control: quant-vs-pure at shallow (t4 tax should ≈ f16 tax if quant cancels)
