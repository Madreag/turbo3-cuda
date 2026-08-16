#!/usr/bin/env python3
"""Fill ladder: verify VRAM stays static as ONE slot fills toward n_ctx.

club-3090 "boots != fills" class: FA transient scratch can grow with FILL,
so a config that boots and passes fixed-depth probes can still OOM at high
fill. Ladder grows a single slot via cache_prompt prefix extension in ~16K
token rungs to ~0.92 * n_ctx, recording VRAM, health, and a 20-token decode
sanity (which doubles as a decode-vs-depth curve).

Usage: fill_ladder.py [--port 8131] [--target-frac 0.92] [--rung 16384]
Env: TCQ_KEY for auth.
"""
import argparse, json, os, subprocess, sys, time, urllib.request

CORPUS = "/home/erol/ai/turboquant/turboquant-g1/sparse-p0/corpus.txt"
TOK_PER_CHAR = 0.3376   # initial guess; AUTO-RECALIBRATED after the first rung
                        # (2026-08-16 fix: the static value overshot chars→tokens
                        # ~12% at depth, so ladders undershot their token targets)

def vram():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return [int(x) for x in out.strip().split(",")]

def health(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8131)
    ap.add_argument("--target-frac", type=float, default=0.92)
    ap.add_argument("--rung", type=int, default=16384)
    ap.add_argument("--n-ctx", type=int, default=327680)
    args = ap.parse_args()

    key = os.environ.get("TCQ_KEY", "")
    corpus = open(CORPUS, errors="ignore").read()
    target_tokens = int(args.n_ctx * args.target_frac)
    rungs = list(range(args.rung, target_tokens + 1, args.rung))
    if rungs[-1] < target_tokens:
        rungs.append(target_tokens)

    used0, total = vram()
    print(f"start: VRAM {used0}/{total} MiB, target {target_tokens} tokens "
          f"({args.target_frac:.0%} of {args.n_ctx}), {len(rungs)} rungs")

    peak = used0
    tok_per_char = TOK_PER_CHAR
    for i, tok_target in enumerate(rungs):
        nchars = min(int(tok_target / tok_per_char), len(corpus))
        body = {"prompt": corpus[:nchars], "n_predict": 20, "temperature": 0.0,
                "ignore_eos": True, "cache_prompt": True}
        req = urllib.request.Request(
            f"http://127.0.0.1:{args.port}/completion",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {key}"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                out = json.load(r)
        except Exception as e:
            u, _ = vram()
            print(f"RUNG {tok_target}: FAILED ({type(e).__name__}: {e}) — VRAM {u} MiB")
            print("LADDER: FAIL — wall found below this rung")
            sys.exit(1)
        wall = time.time() - t0
        t = out.get("timings", {})
        cached = out.get("tokens_cached")
        if isinstance(cached, int) and cached > 4096 and nchars < len(corpus):
            tok_per_char = cached / nchars   # recalibrate so later rungs hit target
        u, _ = vram()
        peak = max(peak, u)
        hp = "ok" if health(args.port) else "UNHEALTHY"
        print(f"rung {tok_target:>7}: cached={out.get('tokens_cached','?'):>7} "
              f"proc={t.get('prompt_n','?'):>6} decode={t.get('predicted_per_second',0):5.1f} t/s "
              f"VRAM={u} MiB (Δ{u-used0:+d}) wall={wall:5.1f}s health={hp}", flush=True)
        if hp != "ok":
            print("LADDER: FAIL — health lost")
            sys.exit(1)

    measured = out.get("tokens_cached", 0)
    short = target_tokens and measured < 0.98 * target_tokens
    tag = "PASS" if not short else "SHORT (corpus/ratio undershoot — measured < 98% of target)"
    print(f"LADDER: {tag} — measured {measured} cached of {target_tokens} target; "
          f"VRAM start {used0} peak {peak} (growth {peak-used0:+d} MiB)")

if __name__ == "__main__":
    main()
