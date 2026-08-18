#!/usr/bin/env python3
"""n-max sweep probe / 15:00-crash repro instrument.

Raw /completion on 8131, production sampling (temp 1.0 / top-p .95 / top-k 20),
seeds paired across arms, cache_prompt on. IGNORE_EOS=0 env reproduces the v1
instrument (EOS can end a cell early); default is ignore_eos:true (llama-bench
-standard sustained decode; uniform across arms so paired deltas stay honest).
Acceptance from response timings draft_n/draft_n_accepted.

Usage: nmax_probe.py <arm_label> [--deep]
Cells: {shallow,38K} x {code,prose} x seeds {42,43}. --deep: single 121K station.
Results append to results.jsonl beside this script.
"""
import json, os, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
KEY = open("/home/erol/.config/llama-tcq/api.key").read().strip()
BASE = f"http://127.0.0.1:{os.environ.get('PORT', '8131')}"
HDR = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
SEEDS = [42, 43]
N_PREDICT = 700

ARM = sys.argv[1]
DEEP = "--deep" in sys.argv

CODE = open(f"{HERE}/prefix_code.txt").read()
PROSE = open(f"{HERE}/prefix_prose.txt").read()

STATIONS = {"shallow": (8_000, 9_000), "38K": (137_000, 163_000)}
if DEEP:
    STATIONS = {"121K": (None, None)}


def post(path, body, timeout=170):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), HDR)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def one(prompt, seed, n_predict=N_PREDICT):
    body = {
        "prompt": prompt, "n_predict": n_predict, "temperature": 1.0,
        "top_p": 0.95, "top_k": 20, "seed": seed, "cache_prompt": True,
    }
    if os.environ.get("IGNORE_EOS", "1") != "0":
        body["ignore_eos"] = True
    r = post("/completion", body)
    t = r.get("timings", {})
    acc = None
    if t.get("draft_n"):
        acc = t.get("draft_n_accepted", 0) / t["draft_n"]
    return t, acc


def main():
    results = []
    for st_name, (cc, pc) in STATIONS.items():
        for cls, text, chars in (("code", CODE, cc), ("prose", PROSE, pc)):
            if DEEP:
                prompt = open(f"{HERE}/prefix_{cls}_deep.txt").read()
            else:
                prompt = text[:chars]
            t = {}
            for attempt in range(3):
                try:
                    t, _ = one(prompt, 1, n_predict=1)
                    break
                except Exception as e:
                    print(f"  prefill retry {attempt+1}: {e}", flush=True)
                    time.sleep(3)
            depth = t.get("prompt_n", -1)
            for seed in SEEDS:
                t, acc = one(prompt, seed)
                row = {"arm": ARM, "station": st_name, "cls": cls, "seed": seed,
                       "depth": depth, "decode_tps": round(t.get("predicted_per_second", 0), 1),
                       "n_gen": t.get("predicted_n"), "prefill_tps": round(t.get("prompt_per_second", 0), 1),
                       "accept": round(acc, 3) if acc is not None else None,
                       "draft_n": t.get("draft_n"), "draft_acc": t.get("draft_n_accepted"),
                       "ts": time.strftime("%H:%M:%S")}
                results.append(row)
                print(f"[{ARM}] {st_name:7s} {cls:5s} seed{seed} depth={depth:>6} "
                      f"decode={row['decode_tps']:>6} t/s accept={row['accept']}", flush=True)
    with open(f"{HERE}/results.jsonl", "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"[{ARM}] DONE {len(results)} cells", flush=True)


if __name__ == "__main__":
    main()
