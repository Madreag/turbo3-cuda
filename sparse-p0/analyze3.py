#!/usr/bin/env python3
"""P0 round 3: lock P1 defaults. Mean scorer only.
- page 32 vs 64
- per-layer min N (tokens) to reach mass-recall targets, single-step and
  MTP-joint (selection on sum of 4 steps' scores, worst step recall)
- resulting read-cost table for candidate configs at 128K/320K
Forced: sink=512 tok, recent=4096 tok.
"""
import json, os, sys
import numpy as np

CENT = np.array([-0.241529, -0.182877, -0.143016, -0.111036, -0.083292,
                 -0.058050, -0.034299, -0.011349, 0.011349, 0.034299,
                 0.058050, 0.083292, 0.111036, 0.143016, 0.182877,
                 0.241529], np.float32)
SINK_T, RECENT_T = 512, 4096
NTOK = [8192, 16384, 24576, 32768, 49152, 65536, 98304, 131072]
TARGETS = [0.85, 0.90, 0.95]

def dequant_turbo4(raw, n_rows, n_dims):
    bpr = n_dims // 128
    b = np.frombuffer(raw, np.uint8).reshape(n_rows, bpr, 66)
    norm = b[:, :, :2].copy().view(np.float16).astype(np.float32)
    qs = b[:, :, 2:]
    idx = np.empty((n_rows, bpr, 128), np.uint8)
    idx[:, :, 0::2] = qs & 0xF
    idx[:, :, 1::2] = qs >> 4
    return (CENT[idx] * norm).reshape(n_rows, n_dims)

def run_layer(Kfull, qlist, scales, page):
    """qlist: list of q [nh,d] per step. Returns per-step true page mass and
    per-step mean scores + npg (at max n_kv)."""
    tms, scs = [], []
    for s, q in enumerate(qlist):
        n_kv = base + s + 1
        K = Kfull[:n_kv]
        n, nkvh, d = K.shape
        nh = q.shape[0]; gqa = nh // nkvh
        npg = (n + page - 1) // page
        pid = np.arange(n) // page
        tm = np.zeros((nh, npg))
        mu = np.zeros((npg, nkvh, d), np.float32)
        full = n // page
        Kp = K[:full*page].reshape(full, page, nkvh, d)
        mu[:full] = Kp.mean(1)
        if n - full*page:
            mu[full] = K[full*page:].mean(0)
        sc = np.zeros(npg)
        for kvh in range(nkvh):
            lg = (q[kvh*gqa:(kvh+1)*gqa] @ K[:, kvh, :].T) * scales[s]
            lg -= lg.max(1, keepdims=True)
            p = np.exp(lg); p /= p.sum(1, keepdims=True)
            for g in range(gqa):
                np.add.at(tm[kvh*gqa+g], pid, p[g])
            sc += mu[:, kvh, :] @ q[kvh*gqa:(kvh+1)*gqa].sum(0)
        tms.append(tm); scs.append(sc)
    return tms, scs

def min_n_for(tms, scs, page, joint, targets, n_kv):
    npg_min = min(t.shape[1] for t in tms)
    sink_p = max(1, SINK_T // page)
    rec_p = max(1, RECENT_T // page)
    forced = set(range(sink_p)) | set(range(max(0, npg_min - rec_p), npg_min))
    if joint:
        ssum = np.zeros(npg_min)
        for sc in scs: ssum += sc[:npg_min]
        orders = [ [p for p in np.argsort(-ssum) if p not in forced] ]*len(tms)
    else:
        orders = [[p for p in np.argsort(-sc[:npg_min]) if p not in forced] for sc in scs]
    res = {}
    for nt in NTOK:
        N = nt // page
        rec = []
        for tm, o in zip(tms, orders):
            sel = list(forced) + o[:max(0, N-len(forced))]
            tot = np.maximum(tm[:, :npg_min].sum(1), 1e-12)
            rec.append(float((tm[:, sel].sum(1)/tot).mean()))
        res[nt] = round(min(rec), 3)
    out = {}
    for t in targets:
        ok = [nt for nt in NTOK if res[nt] >= t]
        out[t] = ok[0] if ok else None
    return res, out

d = sys.argv[1]
entries = [json.loads(l) for l in open(os.path.join(d, "manifest.jsonl"))]
run = next(e for e in entries if e["kind"] == "run")
qs = [e for e in entries if e["kind"] == "q"]
fps = {(e["ckpt"], e["step"], e["layer"]): e for e in entries if e["kind"] == "fa_params"}
Ks = {e["layer"]: e for e in entries if e["kind"] == "K"}
ckpts = run["checkpoints"]

CKS_ARG = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [2, 3]
report = {}
for ci in CKS_ARG:
    base = ckpts[ci]
    print(f"=== depth {base} — min tokens (K) for worst-step recall targets (mean scorer) ===")
    print("layer |    single .85/.90/.95 | joint(MTP) .85/.90/.95 | p32 joint .85/.90/.95")
    for il in sorted(Ks):
        ke = Ks[il]
        raw = open(os.path.join(d, ke["file"]), "rb").read()
        dhead, n_kvh = ke["view_ne"][0], ke["view_ne"][2]
        Kfull = dequant_turbo4(raw, ke["n_rows"], ke["n_dims"]).reshape(ke["n_rows"], n_kvh, dhead)
        qlist, scales = [], []
        for s in range(run["n_steps"]):
            e = next(x for x in qs if x["layer"] == il and x["ckpt"] == ci and x["step"] == s)
            dt = {"f32": np.float32, "f16": np.float16}[e["type"]]
            q = np.fromfile(os.path.join(d, e["file"]), dtype=dt)
            ne = e["ne"]
            qlist.append(q.reshape(ne[3], ne[2], ne[1], ne[0]).reshape(-1, ne[0]).astype(np.float32))
            scales.append(fps[(ci, s, il)]["scale"])
        r64 = run_layer(Kfull, qlist, scales, 64)
        _, m_single = min_n_for(*r64, 64, False, TARGETS, base)
        _, m_joint = min_n_for(*r64, 64, True, TARGETS, base)
        r32 = run_layer(Kfull, qlist, scales, 32)
        _, m32_joint = min_n_for(*r32, 32, True, TARGETS, base)
        fmt = lambda m: "/".join(str(m[t]//1024) if m[t] else "-" for t in TARGETS)
        print(f"l{il:02d}   |  {fmt(m_single):>12s}  |  {fmt(m_joint):>12s}  |  {fmt(m32_joint):>12s}", flush=True)
        report[(ci, il)] = dict(single=m_single, joint=m_joint, p32=m32_joint)
        del Kfull

json.dump({f"{k[0]}_{k[1]}": {str(t): v[t] for t in TARGETS for v in [vv] } if False else
           {kk: {str(t): vv2[t] for t in TARGETS} for kk, vv2 in vv.items()}
           for k, vv in report.items()},
          open(os.path.join(d, "recall3.json"), "w"), indent=1)
print("wrote recall3.json")
