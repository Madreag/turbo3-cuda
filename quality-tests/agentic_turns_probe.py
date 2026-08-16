#!/usr/bin/env python3
"""Agentic-turns probe: per-turn TTFT + decode across a growing tool session.

Models the production shape (club-3090 bench-agentic idea): a conversation
that accumulates large tool results each turn. With cache_prompt=true this
measures the TURN LOOP — prefix reuse + per-turn prefill delta + decode —
the exact path the checkpoint/restore work optimizes. Growth verdict anchors
to turn 2 (turn 1 = cold start).

Usage: agentic_turns_probe.py [--port 8131] [--turns 12] [--label x]
"""
import argparse, json, os, time, urllib.request

CORPUS = "/home/erol/ai/turboquant/turboquant-g1/sparse-p0/corpus.txt"
# per-turn tool-result sizes in chars (shaped like real Claude-Code sessions)
TURN_CHARS = [900, 750, 850, 25000, 27000, 9500, 19500, 7800, 76000, 52000,
              64000, 66000, 34000, 95000, 73000]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8131)
    ap.add_argument("--turns", type=int, default=12)
    ap.add_argument("--label", default="run")
    ap.add_argument("--temp", type=float, default=1.0)
    args = ap.parse_args()
    key = os.environ.get("TCQ_KEY", "")
    corpus = open(CORPUS, errors="ignore").read()

    msgs = [{"role": "system", "content": "You are a coding agent. Analyze the tool output and reply with a one-paragraph plan."}]
    off = 0
    rows = []
    for t in range(1, args.turns + 1):
        n = TURN_CHARS[(t - 1) % len(TURN_CHARS)]
        chunk = corpus[off:off + n]; off += n
        msgs.append({"role": "user", "content": f"Tool output (turn {t}):\n{chunk}\n\nWhat next?"})
        body = {"messages": msgs, "max_tokens": 220, "temperature": args.temp,
                "top_p": 0.95, "top_k": 20, "seed": 42, "cache_prompt": True}
        req = urllib.request.Request(
            f"http://127.0.0.1:{args.port}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=900) as r:
            out = json.load(r)
        wall = time.time() - t0
        tm = out.get("timings", {})
        ttft = tm.get("prompt_ms", 0) / 1000.0
        dec = tm.get("predicted_per_second", 0)
        pn = tm.get("prompt_n", 0)
        rows.append((t, pn, ttft, dec, wall))
        print(f"turn {t:>2}: prefill_delta={pn:>6} ttft={ttft:6.2f}s "
              f"decode={dec:5.1f} t/s wall={wall:6.2f}s", flush=True)
        content = out["choices"][0]["message"]["content"]
        msgs.append({"role": "assistant", "content": content[:400]})

    # growth verdict: TTFT PER DELTA TOKEN, anchored to turn 2 (turn 1 = cold).
    # Raw TTFT scales with each turn's tool-result size by design; the
    # pathology to catch (vLLM Cliff-3 class) is TTFT scaling with TOTAL
    # context despite cache hits — visible as per-delta-token cost growth.
    if len(rows) >= 3:
        c2 = rows[1][2] / max(rows[1][1], 1)
        cN = rows[-1][2] / max(rows[-1][1], 1)
        ratio = cN / max(c2, 1e-9)
        band = ("STABLE" if ratio <= 1.5 else
                "GREW" if ratio <= 3 else "O(n)-LIKE (Cliff-3 class)")
        print(f"VERDICT [{args.label}]: prefill {c2*1000:.2f} -> {cN*1000:.2f} ms/delta-token "
              f"(x{ratio:.2f}) -> {band}; decode turn2={rows[1][3]:.1f} -> {rows[-1][3]:.1f} t/s")

if __name__ == "__main__":
    main()
