#!/usr/bin/env python3
"""
Long-context validation battery — multi-needle NIAH + 3-hop chain + hallucination control.
One full prefill per arm; questions ride the prompt cache as prefix-sharing single turns.

Usage: longctx_battery.py --target-tokens 290000 --label 290k-raw [--port 8091] [--outdir DIR]
Server must already be running with the arm's config.
"""
import argparse, json, os, sys, time
import requests

DEF_WIKI = "/home/erol/ai/turboquant/research/llama-cpp-turboquant/wikitext-2-raw/wiki.train.raw"

NEEDLES = [
    dict(id="N1", depth=0.10, text="The vault access code for the Meridian facility is 7391-ALPHA-2286.",
         q="What is the vault access code for the Meridian facility?",
         checks=["7391", "2286"]),
    dict(id="N2", depth=0.50, text="The migration of the Aurelius database cluster is scheduled for October 9th, 2027, at 03:45 UTC.",
         q="On what date is the migration of the Aurelius database cluster scheduled?",
         checks=["october", "2027"]),
    dict(id="N3", depth=0.90, text="Dr. Selene Varga's experimental compound is catalogued under reference number QX-4417-DELTA.",
         q="What is the catalogue reference number of Dr. Selene Varga's experimental compound?",
         checks=["qx-4417", "delta"]),
]
HOP_FACTS = [
    (0.20, "Project Nightglass operates with an annual budget of 84 million dollars."),
    (0.55, "Project Nightglass is directed by Dr. Tomasz Keller."),
    (0.85, "Dr. Tomasz Keller's laboratory is located in Tromso, Norway."),
]
HOP_Q = ("Which city hosts the laboratory of the person who directs the project with the "
         "84 million dollar annual budget? Name the project, the director, and the city.")
HOP_CHECKS = ["tromso"]  # full pass; partials tracked separately
CONTROL_Q = "What is the vault access code for the Zephyr facility?"  # not in text

def build_prompt(target_tokens, chars_per_tok=4.0):
    with open(DEF_WIKI) as f:
        base = f.read()
    target_chars = int(target_tokens * chars_per_tok)
    hay = (base * (target_chars // len(base) + 1))[:target_chars]
    inserts = [(n["depth"], n["text"]) for n in NEEDLES] + HOP_FACTS
    inserts.sort(key=lambda x: -x[0])  # inject deepest first so earlier offsets stay valid
    for depth, text in inserts:
        pos = hay.find("\n\n", int(len(hay) * depth))
        pos = pos if pos != -1 else int(len(hay) * depth)
        hay = hay[:pos] + f"\n\n{text}\n\n" + hay[pos:]
    return hay

def ask(port, hay, question, max_tokens=3500):
    t0 = time.time()
    key = open("/home/erol/.config/llama-tcq/api.key").read().strip()
    r = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions",
                      headers={"Authorization": f"Bearer {key}"}, json={
        "messages": [
            {"role": "system", "content": "You answer questions using only the provided text. Be concise and exact."},
            {"role": "user", "content": hay + "\n\n---\n\nBased only on the text above: " + question},
        ],
        "max_tokens": max_tokens, "temperature": 0,
    }, timeout=2400)
    dt = time.time() - t0
    if r.status_code != 200:
        return dict(error=f"HTTP {r.status_code}: {r.text[:300]}", secs=round(dt, 1))
    d = r.json()
    c = d["choices"][0]["message"].get("content") or ""
    ans = c.split("</think>")[-1].strip() if "</think>" in c else c.strip()
    u = d.get("usage", {})
    return dict(answer=ans, thinking_len=len(c) - len(ans), secs=round(dt, 1),
                prompt_tokens=u.get("prompt_tokens"), completion_tokens=u.get("completion_tokens"),
                finish=d["choices"][0].get("finish_reason"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-tokens", type=int, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--outdir", default="/home/erol/ai/turboquant/turboquant-kv-cache/quality-tests/niah_results")
    a = ap.parse_args()

    hay = build_prompt(a.target_tokens)
    print(f"[{a.label}] haystack {len(hay):,} chars (~{len(hay)//4:,} est tok)", flush=True)
    results = dict(label=a.label, target_tokens=a.target_tokens, probes={})

    probes = [(n["id"], n["q"], n["checks"]) for n in NEEDLES]
    probes.append(("HOP3", HOP_Q, HOP_CHECKS))
    probes.append(("CTRL", CONTROL_Q, None))

    for pid, q, checks in probes:
        print(f"[{a.label}] asking {pid}...", flush=True)
        res = ask(a.port, hay, q)
        if "error" in res:
            print(f"  {pid}: ERROR {res['error']}", flush=True)
            results["probes"][pid] = res
            continue
        low = res["answer"].lower()
        if checks is not None:
            res["pass"] = all(c in low for c in checks)
            verdict = "PASS" if res["pass"] else "FAIL"
        else:  # control: pass = does NOT fabricate (no digit-groups presented as the code)
            fabricated = any(t in low for t in ["-alpha-", "code is", "code for the zephyr facility is"]) and any(ch.isdigit() for ch in low)
            res["pass"] = not fabricated
            verdict = "OK(no-fab)" if res["pass"] else "FABRICATED"
        if pid == "HOP3":
            res["partials"] = {k: (k in low) for k in ["nightglass", "keller", "tromso"]}
        print(f"  {pid}: {verdict} ({res['secs']}s, ptok={res['prompt_tokens']}, ctok={res['completion_tokens']}, finish={res['finish']})", flush=True)
        print(f"  answer: {res['answer'][:220]}", flush=True)
        results["probes"][pid] = res

    os.makedirs(a.outdir, exist_ok=True)
    out = os.path.join(a.outdir, f"battery_{a.label}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    npass = sum(1 for p in results["probes"].values() if p.get("pass"))
    print(f"[{a.label}] DONE {npass}/{len(results['probes'])} pass → {out}", flush=True)

if __name__ == "__main__":
    main()
