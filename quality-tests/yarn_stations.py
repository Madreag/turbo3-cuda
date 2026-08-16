#!/usr/bin/env python3
"""Yarn-tax station probe: grow one cached prefix to each depth station, then
branch K short continuations off the cache and collect next-token top-k
logprobs. Across-arm comparison (same quant, different rope scale) isolates
the yarn tax. Writes stations_<label>.json.

Usage: yarn_stations.py --label t4_none --stations 2048,256000 [--branches 25]
"""
import argparse, json, os, urllib.request

CORPUS = "/home/erol/ai/turboquant/turboquant-g1/sparse-p0/corpus.txt"

def req(body):
    key = os.environ.get("TCQ_KEY", "")
    r = urllib.request.Request("http://127.0.0.1:8131/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(r, timeout=1800) as resp:
        return json.load(resp)

def logprobs_of(prompt):
    out = req({"prompt": prompt, "max_tokens": 1, "temperature": 0,
               "logprobs": 40, "cache_prompt": True})
    ch = out["choices"][0]
    lp = ch.get("logprobs") or {}
    if "content" in lp and lp["content"]:
        return {i["token"]: i["logprob"] for i in lp["content"][0].get("top_logprobs", [])}
    if "top_logprobs" in lp and lp["top_logprobs"]:
        return lp["top_logprobs"][0]
    return {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--stations", default="2048,256000")
    ap.add_argument("--branches", type=int, default=25)
    args = ap.parse_args()
    corpus = open(CORPUS, errors="ignore").read()
    # ~0.2975 tok/char measured for prefix slices of this corpus
    TPC = 0.2975
    stations = [int(x) for x in args.stations.split(",")]
    out = {"label": args.label, "stations": {}}
    for st in stations:
        prefix = corpus[: int(st / TPC)]
        # warm the cache once (branch probes then pay only their tails)
        req({"prompt": prefix, "max_tokens": 1, "temperature": 0, "cache_prompt": True})
        branches = {}
        for b in range(args.branches):
            tail = corpus[1_100_000 + b * 400 : 1_100_000 + b * 400 + 320]  # distinct ~95-token tails
            branches[b] = logprobs_of(prefix + "\n" + tail)
        out["stations"][st] = branches
        print(f"[{args.label}] station {st}: {args.branches} branches collected", flush=True)
    path = f"/home/erol/ai/turboquant/turboquant-g1/quality-tests/yarntax/stations_{args.label}.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, "w"))
    print("wrote", path)

if __name__ == "__main__":
    main()
