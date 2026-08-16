#!/usr/bin/env python3
"""P0 sparse-decode offline validator (turbo4 raw dumps -> recall@N curves).

Dequantizes raw turbo4 K (CUDA formula: centroid[idx] * block norm — stored/WHT
basis, byte-faithful with what the FA kernel sees) and evaluates Quest-style
page selection against true attention mass, per layer/depth/step.

Outputs: recall@N for {per-layer agg, per-kv-head agg, oracle}, page 64 vs 128,
anchor-layer sharing (every 4th), MTP step-union, sink/recent sensitivity.
"""
import json, os, sys
import numpy as np

CENT = np.array([-0.241529, -0.182877, -0.143016, -0.111036, -0.083292,
                 -0.058050, -0.034299, -0.011349, 0.011349, 0.034299,
                 0.058050, 0.083292, 0.111036, 0.143016, 0.182877,
                 0.241529], np.float32)

PAGE = 64
NS = [32, 64, 128, 256, 512]      # pages -> 2K/4K/8K/16K/32K tokens
SINK, RECENT = 2, 16               # forced pages (128 sink / 1024 recent)

def dequant_turbo4(raw_bytes, n_rows, n_dims):
    bpr = n_dims // 128
    b = np.frombuffer(raw_bytes, np.uint8).reshape(n_rows, bpr, 66)
    norm = b[:, :, :2].copy().view(np.float16).astype(np.float32)      # [r,bpr,1]
    qs = b[:, :, 2:]
    idx = np.empty((n_rows, bpr, 128), np.uint8)
    idx[:, :, 0::2] = qs & 0xF
    idx[:, :, 1::2] = qs >> 4
    return (CENT[idx] * norm).reshape(n_rows, n_dims)                  # f32

def page_minmax(K, page):
    n, nkvh, d = K.shape
    npg = n // page
    Kp = K[:npg*page].reshape(npg, page, nkvh, d)
    mn, mx = Kp.min(axis=1), Kp.max(axis=1)
    rem = n - npg*page
    if rem:
        t = K[npg*page:]
        mn = np.concatenate([mn, t.min(axis=0)[None]], 0)
        mx = np.concatenate([mx, t.max(axis=0)[None]], 0)
    return mn, mx                                                      # [npg,kvh,d]

def true_mass(q, K, scale, softcap, page):
    # q [nh,d], K [n,nkvh,d] -> [nh, npg]
    nh, d = q.shape
    n, nkvh, _ = K.shape
    gqa = nh // nkvh
    npg = (n + page - 1) // page
    pid = np.arange(n) // page
    out = np.zeros((nh, npg), np.float64)
    for kvh in range(nkvh):
        logits = (q[kvh*gqa:(kvh+1)*gqa] @ K[:, kvh, :].T).astype(np.float64) * scale
        if softcap:
            logits = softcap * np.tanh(logits / softcap)
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        for g in range(gqa):
            np.add.at(out[kvh*gqa+g], pid, p[g])
    return out

def bound_scores(q, mn, mx):
    # q [nh,d]; mn/mx [npg,nkvh,d] -> [nh,npg]
    nh, d = q.shape
    npg, nkvh, _ = mn.shape
    gqa = nh // nkvh
    qp, qm = np.maximum(q, 0), np.minimum(q, 0)
    s = np.empty((nh, npg), np.float32)
    for h in range(nh):
        kvh = h // gqa
        s[h] = mx[:, kvh, :] @ qp[h] + mn[:, kvh, :] @ qm[h]
    return s

def recall_at(true_m, order, forced, Ns):
    # true_m [nh,npg]; order: page ids best-first (excluding forced)
    tot = np.maximum(true_m.sum(axis=1), 1e-12)
    res = {}
    fl = list(forced)
    for N in Ns:
        sel = fl + [p for p in order[:max(0, N-len(fl))]]
        m = true_m[:, sel].sum(axis=1) / tot
        res[N] = (round(float(m.mean()), 4), round(float(m.min()), 4))
    return res

def forced_set(npg, sink=SINK, recent=RECENT):
    return set(range(min(sink, npg))) | set(range(max(0, npg-recent), npg))

def ordered(scores_1d, forced):
    o = np.argsort(-scores_1d, kind="stable")
    return [int(p) for p in o if p not in forced]

def main():
    d = sys.argv[1]
    entries = [json.loads(l) for l in open(os.path.join(d, "manifest.jsonl"))]
    run = next(e for e in entries if e["kind"] == "run")
    qs = [e for e in entries if e["kind"] == "q"]
    fps = {(e["ckpt"], e["step"], e["layer"]): e for e in entries if e["kind"] == "fa_params"}
    Ks = {e["layer"]: e for e in entries if e["kind"] == "K"}
    ckpts = run["checkpoints"]
    layers = sorted(Ks)

    store = {}   # (il, ci, s) -> dict(tm[nh,npg], agg[npg], aggkvh[nkvh,npg])
    rows = []
    for il in layers:
        ke = Ks[il]
        n_rows, n_dims = ke["n_rows"], ke["n_dims"]
        dhead, n_kvh = ke["view_ne"][0], ke["view_ne"][2]
        assert dhead * n_kvh == n_dims
        raw = open(os.path.join(d, ke["file"]), "rb").read()
        Kfull = dequant_turbo4(raw, n_rows, n_dims).reshape(n_rows, n_kvh, dhead)

        for e in [e for e in qs if e["layer"] == il]:
            ci, s = e["ckpt"], e["step"]
            n_kv = ckpts[ci] + s + 1
            fp = fps.get((ci, s, il), {})
            scale = fp.get("scale", 1.0/np.sqrt(dhead))
            softcap = fp.get("softcap", 0.0) or 0.0
            dt = {"f32": np.float32, "f16": np.float16}[e["type"]]
            qraw = np.fromfile(os.path.join(d, e["file"]), dtype=dt)
            ne = e["ne"]
            q = qraw.reshape(ne[3], ne[2], ne[1], ne[0]).reshape(-1, ne[0]).astype(np.float32)
            nh = q.shape[0]

            K = Kfull[:n_kv]
            tm = true_mass(q, K, scale, softcap, PAGE)
            mn, mx = page_minmax(K, PAGE)
            bs = bound_scores(q, mn, mx)
            npg = tm.shape[1]
            gqa = nh // n_kvh
            agg = bs.sum(axis=0)
            aggk = bs.reshape(n_kvh, gqa, npg).sum(axis=1)
            store[(il, ci, s)] = dict(tm=tm.astype(np.float32), agg=agg,
                                      aggk=aggk, npg=npg)

            F = forced_set(npg)
            r_layer = recall_at(tm, ordered(agg, F), F, NS)
            r_orc = recall_at(tm, ordered(tm.sum(axis=0), F), F, NS)
            # per-kv-head: each head group ranked by its own kv-head agg
            r_kvh = {}
            for N in NS:
                vals = []
                for kvh in range(n_kvh):
                    o = ordered(aggk[kvh], F)
                    r = recall_at(tm[kvh*gqa:(kvh+1)*gqa], o, F, [N])
                    vals.append(r[N][0])
                r_kvh[N] = (round(float(np.mean(vals)), 4), round(float(np.min(vals)), 4))
            # page 128
            npg2 = npg // 2
            mn2 = np.minimum(mn[:npg2*2:2], mn[1:npg2*2:2])
            mx2 = np.maximum(mx[:npg2*2:2], mx[1:npg2*2:2])
            tm2 = tm[:, :npg2*2].reshape(nh, npg2, 2).sum(axis=2)
            bs2 = bound_scores(q, mn2, mx2).sum(axis=0)
            F2 = forced_set(npg2, max(1, SINK//2), max(1, RECENT//2))
            r_p128 = recall_at(tm2, ordered(bs2, F2), F2, [n//2 for n in NS])

            rows.append(dict(layer=il, ckpt=ckpts[ci], step=s, npg=npg,
                             layer_agg=r_layer, kvh_agg=r_kvh, oracle=r_orc,
                             p128=r_p128))
            if s == 0:
                print(f"l{il:02d} @{ckpts[ci]:6d}: " +
                      " ".join(f"N{N}={r_layer[N][0]:.3f}/orc{r_orc[N][0]:.3f}"
                               for N in [64, 256]), flush=True)
        del Kfull

    # anchor sharing: layer j uses selection ranked by anchor a=(j//4)*4
    anchor_rows = []
    for (il, ci, s), st in store.items():
        a = (layers.index(il)//4)*4
        ail = layers[a]
        if ail == il or (ail, ci, s) not in store:
            continue
        F = forced_set(st["npg"])
        o = ordered(store[(ail, ci, s)]["agg"][:st["npg"]], F)
        r = recall_at(st["tm"], o, F, NS)
        anchor_rows.append(dict(layer=il, anchor=ail, ckpt=ckpts[ci], step=s, recall=r))

    # MTP union: selection from step 0 evaluated on steps 1..3
    union_rows = []
    for (il, ci, s), st in store.items():
        if s == 0 or (il, ci, 0) not in store:
            continue
        st0 = store[(il, ci, 0)]
        npg = min(st["npg"], st0["npg"])
        F = forced_set(st["npg"])
        o = ordered(st0["agg"][:npg], F)
        r = recall_at(st["tm"], o, F, NS)
        union_rows.append(dict(layer=il, ckpt=ckpts[ci], step=s, recall=r))

    # sink/recent sensitivity at N=256, layer agg (bigger recent)
    sens_rows = []
    for (il, ci, s), st in store.items():
        if s != 0:
            continue
        for (sk, rc) in [(0, 0), (2, 16), (2, 64), (8, 64)]:
            F = forced_set(st["npg"], sk, rc)
            r = recall_at(st["tm"], ordered(st["agg"], F), F, [256])
            sens_rows.append(dict(layer=il, ckpt=ckpts[ci], sink=sk, recent=rc,
                                  r256=r[256]))

    out = dict(run=run, Ns=NS, page=PAGE, sink=SINK, recent=RECENT,
               main=rows, anchor=anchor_rows, union=union_rows, sens=sens_rows)
    with open(os.path.join(d, "recall.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("wrote", os.path.join(d, "recall.json"))

if __name__ == "__main__":
    main()
