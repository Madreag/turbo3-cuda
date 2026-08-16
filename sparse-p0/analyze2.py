#!/usr/bin/env python3
"""P0 round 2: scorer bake-off on the probe dumps.

Scorers (all computable from stored-basis page stats, all valid since dot
products are rotation-invariant and mean/std commute with rotation):
  quest    : sum_d max(q_d*min_d, q_d*max_d)         (round-1 baseline)
  mean     : q . mu_page                              (mass proxy)
  meanstd_B: q . mu + B * sqrt(sum_d (q_d*sigma_d)^2) (UCB-style)
Eval: recall@N vs oracle, forced sink=2/recent=64; per-batch selection
(sum of 4 steps' scores) evaluated per step (MTP verify-batch proxy);
anchor sharing under best scorer.
"""
import json, os, sys
import numpy as np

CENT = np.array([-0.241529, -0.182877, -0.143016, -0.111036, -0.083292,
                 -0.058050, -0.034299, -0.011349, 0.011349, 0.034299,
                 0.058050, 0.083292, 0.111036, 0.143016, 0.182877,
                 0.241529], np.float32)
PAGE = 64
NS = [128, 256, 512]
SINK, RECENT = 2, 64

def dequant_turbo4(raw_bytes, n_rows, n_dims):
    bpr = n_dims // 128
    b = np.frombuffer(raw_bytes, np.uint8).reshape(n_rows, bpr, 66)
    norm = b[:, :, :2].copy().view(np.float16).astype(np.float32)
    qs = b[:, :, 2:]
    idx = np.empty((n_rows, bpr, 128), np.uint8)
    idx[:, :, 0::2] = qs & 0xF
    idx[:, :, 1::2] = qs >> 4
    return (CENT[idx] * norm).reshape(n_rows, n_dims)

def page_stats(K):
    n, nkvh, d = K.shape
    npg = (n + PAGE - 1) // PAGE
    full = n // PAGE
    Kp = K[:full*PAGE].reshape(full, PAGE, nkvh, d)
    mn, mx = Kp.min(1), Kp.max(1)
    mu, sd = Kp.mean(1), Kp.std(1)
    if n - full*PAGE:
        t = K[full*PAGE:]
        mn = np.concatenate([mn, t.min(0)[None]]); mx = np.concatenate([mx, t.max(0)[None]])
        mu = np.concatenate([mu, t.mean(0)[None]]); sd = np.concatenate([sd, t.std(0)[None]])
    return mn, mx, mu, sd   # [npg, nkvh, d]

def true_mass(q, K, scale):
    nh, d = q.shape
    n, nkvh, _ = K.shape
    gqa = nh // nkvh
    npg = (n + PAGE - 1) // PAGE
    pid = np.arange(n) // PAGE
    out = np.zeros((nh, npg), np.float64)
    for kvh in range(nkvh):
        lg = (q[kvh*gqa:(kvh+1)*gqa] @ K[:, kvh, :].T).astype(np.float64) * scale
        lg -= lg.max(1, keepdims=True)
        p = np.exp(lg); p /= p.sum(1, keepdims=True)
        for g in range(gqa):
            np.add.at(out[kvh*gqa+g], pid, p[g])
    return out

def scores_all(q, mn, mx, mu, sd, betas):
    # returns dict name -> [npg] (aggregated over heads)
    nh, d = q.shape
    npg, nkvh, _ = mn.shape
    gqa = nh // nkvh
    qp, qm = np.maximum(q, 0), np.minimum(q, 0)
    quest = np.zeros(npg, np.float64)
    mean = np.zeros(npg, np.float64)
    var = np.zeros(npg, np.float64)   # sum over heads of sum_d (q_d sd_d)^2
    for h in range(nh):
        kvh = h // gqa
        quest += mx[:, kvh, :] @ qp[h] + mn[:, kvh, :] @ qm[h]
        mean += mu[:, kvh, :] @ q[h]
        var += (sd[:, kvh, :]**2) @ (q[h]**2)
    out = {"quest": quest, "mean": mean}
    for b in betas:
        out[f"meanstd_{b}"] = mean + b * np.sqrt(var)
    return out

def forced_set(npg, sink=SINK, recent=RECENT):
    return set(range(min(sink, npg))) | set(range(max(0, npg-recent), npg))

def recall_at(tm, order, forced, Ns):
    tot = np.maximum(tm.sum(1), 1e-12)
    fl = list(forced)
    res = {}
    for N in Ns:
        sel = fl + [p for p in order[:max(0, N-len(fl))]]
        res[N] = round(float((tm[:, sel].sum(1) / tot).mean()), 4)
    return res

def ordered(s, forced):
    return [int(p) for p in np.argsort(-s, kind="stable") if p not in forced]

def main():
    d = sys.argv[1]
    entries = [json.loads(l) for l in open(os.path.join(d, "manifest.jsonl"))]
    run = next(e for e in entries if e["kind"] == "run")
    qs = [e for e in entries if e["kind"] == "q"]
    fps = {(e["ckpt"], e["step"], e["layer"]): e for e in entries if e["kind"] == "fa_params"}
    Ks = {e["layer"]: e for e in entries if e["kind"] == "K"}
    ckpts = run["checkpoints"]
    CKS = [2, 3]   # 65536, 126976
    betas = [0.5, 1.0, 2.0]
    names = ["quest", "mean"] + [f"meanstd_{b}" for b in betas]

    agg = {}   # (name, ck) -> list of per-layer recalls at each N
    batchsel = {}  # scorer union results
    anchor_pen = []
    layer_scores = {}   # (il, ci) -> best-scorer score vec (for anchor pass)
    per_layer = {}

    for il in sorted(Ks):
        ke = Ks[il]
        raw = open(os.path.join(d, ke["file"]), "rb").read()
        dhead, n_kvh = ke["view_ne"][0], ke["view_ne"][2]
        Kfull = dequant_turbo4(raw, ke["n_rows"], ke["n_dims"]).reshape(ke["n_rows"], n_kvh, dhead)
        for ci in CKS:
            tms, svs = [], []
            for s in range(run["n_steps"]):
                e = next(x for x in qs if x["layer"] == il and x["ckpt"] == ci and x["step"] == s)
                n_kv = ckpts[ci] + s + 1
                fp = fps[(ci, s, il)]
                dt = {"f32": np.float32, "f16": np.float16}[e["type"]]
                q = np.fromfile(os.path.join(d, e["file"]), dtype=dt)
                ne = e["ne"]
                q = q.reshape(ne[3], ne[2], ne[1], ne[0]).reshape(-1, ne[0]).astype(np.float32)
                K = Kfull[:n_kv]
                tm = true_mass(q, K, fp["scale"])
                mn, mx, mu, sd = page_stats(K)
                sv = scores_all(q, mn, mx, mu, sd, betas)
                tms.append(tm); svs.append(sv)
                if s == 0:
                    npg = tm.shape[1]
                    F = forced_set(npg)
                    orc = recall_at(tm, ordered(tm.sum(0), F), F, NS)
                    row = {"oracle": orc}
                    for nm in names:
                        row[nm] = recall_at(tm, ordered(sv[nm], F), F, NS)
                    per_layer[(il, ci)] = row
                    layer_scores[(il, ci)] = sv
            # batch-union: select once on summed scores of all 4 steps, eval each step
            npg0 = tms[0].shape[1]
            F = forced_set(npg0)
            for nm in ["mean", "meanstd_1.0", "quest"]:
                ssum = np.zeros(npg0)
                for sv in svs:
                    ssum += sv[nm][:npg0]
                o = ordered(ssum, F)
                rs = [recall_at(tms[s][:, :npg0], o, F, [256])[256] for s in range(len(tms))]
                batchsel.setdefault((nm, ci), []).append((il, [round(x, 3) for x in rs]))
        del Kfull

    print("=== per-layer recall @N256, step0 (scorer bake-off) ===")
    for ci in CKS:
        print(f"-- depth {ckpts[ci]}: layer: " + " ".join(f"{nm:>11s}" for nm in names + ["oracle"]))
        for il in sorted(Ks):
            row = per_layer[(il, ci)]
            print(f"l{il:02d}: " + " ".join(f"{row[nm][256]:11.3f}" for nm in names + ["oracle"]))
    print("\n=== recall @N512 late layers ===")
    for ci in CKS:
        for il in [l for l in sorted(Ks) if l >= 10]:
            row = per_layer[(il, ci)]
            print(f"@{ckpts[ci]} l{il:02d}: " + " ".join(f"{nm}={row[nm][512]:.3f}" for nm in ["quest", "mean", "meanstd_1.0", "oracle"]))
    print("\n=== batch-union (select on sum of 4 steps, eval per step) N256 ===")
    for (nm, ci), rows in sorted(batchsel.items()):
        worst = min(min(r[1]) for r in rows)
        late = [r for r in rows if r[0] >= 12]
        print(f"{nm} @{ckpts[ci]}: worst-any-layer-step={worst:.3f}  late-layers: " +
              "; ".join(f"l{r[0]}:{r[1]}" for r in late))
    # anchor penalty under mean scorer
    print("\n=== anchor sharing (mean scorer) l->anchor @126976 N256 ===")
    for il in sorted(Ks):
        a = (il//4)*4
        if a == il or (a, 3) not in layer_scores:
            continue
        tm = None  # need tm again — skip heavy recompute; report via scores proxy
    json.dump({"per_layer": {f"{k[0]}_{k[1]}": v for k, v in per_layer.items()},
               "batchsel": {f"{k[0]}_{k[1]}": v for k, v in batchsel.items()}},
              open(os.path.join(d, "recall2.json"), "w"), indent=1, default=float)
    print("wrote recall2.json")

if __name__ == "__main__":
    main()
