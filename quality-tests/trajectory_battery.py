#!/usr/bin/env python3
"""Trajectory/multi-hop battery — quality on the agentic axis.

PPL/KLD/NIAH measure token-level fidelity and single-fact recall; this battery
measures what agents actually need: chaining facts ACROSS depths, tracking
mutated state, honoring corrections, and holding a coding trajectory. Motivated
by third-party REFRACT data showing V-cache quantization can hold KLD while
multi-hop collapses (llama.cpp discussion #20969).

Tests per depth (one prefill per depth, queries share the cached prefix):
  hops-2/3/4   : chained facts planted at scattered offsets; answer requires
                 following the full chain. Exact-match scoring.
  ledger       : 8 registers, 24 interleaved mutations; report final values.
                 Per-register accuracy.
  correction   : fact planted early, corrected later; corrected value must win.
  code-traj    : module spec + 4 incremental change-requests referencing
                 earlier decisions; final code executed against asserts.

Usage: trajectory_battery.py [--depths 16000,64000] [--label NAME] [--port 8131]
Writes trajbase/traj_<label>.json
"""
import argparse
import json
import random
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

KEY = open("/home/erol/.config/llama-tcq/api.key").read().strip()

FILLER_SENTS = [
    "The archive clerk stamped another routine page and filed it away.",
    "Rain traced idle patterns down the depot windows all afternoon.",
    "A maintenance log noted nominal readings across every gauge.",
    "The shuttle manifest listed cargo of no particular interest.",
    "Somewhere a printer hummed through its unremarkable queue.",
    "The quarterly memo restated policies everyone already knew.",
]

NAMES = ["Maren", "Tobias", "Ilka", "Ravi", "Sole", "Petra", "Anselm", "Yuki"]
PLACES = ["Brindle Bay", "Fort Casker", "Ludlow Spur", "Vetch Hollow",
          "Sarn Break", "Quill Harbor", "Mirren Flats", "Osier Point"]


def words_for_tokens(n_tok):
    return int(n_tok * 0.72)


def build_corpus(depth_tokens, rng):
    """Deterministic filler + planted items. Returns (text, answers)."""
    total_words = words_for_tokens(depth_tokens)
    # --- chained facts (4 links; hop-k uses first k) ---
    code = f"{rng.randint(10,99)}-{rng.choice('KLMNPQ')}{rng.randint(100,999)}"
    person, place = rng.sample(NAMES, 1)[0], rng.sample(PLACES, 1)[0]
    locker = rng.randint(200, 899)
    chain = [
        f"[REGISTRY] The Calder dossier is custodied by {person}.",
        f"[REGISTRY] {person} operates out of {place}.",
        f"[REGISTRY] The {place} office uses storage locker {locker}.",
        f"[REGISTRY] Locker {locker} is sealed with code {code}.",
    ]
    # --- ledger: 8 registers, 24 mutations ---
    regs = {f"R{i}": 0 for i in range(1, 9)}
    ledger_lines = []
    for _ in range(24):
        r = rng.choice(list(regs))
        op = rng.choice(["set", "add", "sub"])
        v = rng.randint(1, 40)
        if op == "set":
            regs[r] = v
        elif op == "add":
            regs[r] += v
        else:
            regs[r] -= v
        ledger_lines.append(f"[LEDGER] {op.upper()} {r} {v}.")
    # --- correction pair ---
    wrong, right = rng.sample(range(1000, 9999), 2)
    corr_early = f"[BULLETIN] The relay frequency is {wrong} kHz."
    corr_late = (f"[BULLETIN] Correction to an earlier bulletin: the relay "
                 f"frequency is {right} kHz, not {wrong}.")

    items = ([("chain", s) for s in chain]
             + [("ledger", s) for s in ledger_lines]
             + [("corr_early", corr_early), ("corr_late", corr_late)])
    # placement: chain scattered 10-90%, ledger in order, correction early/late
    n_slots = len(items)
    text_parts = []
    words_done = 0
    # interleave: emit filler, drop items at spaced offsets (ledger keeps order)
    chain_offsets = sorted(rng.sample(range(10, 91, 5), 4))
    offsets = {}
    ci = 0
    for kind, s in items:
        if kind == "chain":
            offsets[s] = chain_offsets[ci] / 100
            ci += 1
        elif kind == "corr_early":
            offsets[s] = 0.08
        elif kind == "corr_late":
            offsets[s] = 0.85
    ledger_items = [s for k, s in items if k == "ledger"]
    for i, s in enumerate(ledger_items):
        offsets[s] = 0.12 + 0.72 * i / len(ledger_items)
    ordered = sorted(offsets.items(), key=lambda kv: kv[1])
    pos = 0
    for s, frac in ordered:
        target = int(total_words * frac)
        while words_done < target:
            sent = rng.choice(FILLER_SENTS)
            text_parts.append(sent)
            words_done += len(sent.split())
        text_parts.append(s)
        words_done += len(s.split())
    while words_done < total_words:
        sent = rng.choice(FILLER_SENTS)
        text_parts.append(sent)
        words_done += len(sent.split())
    answers = {"person": person, "place": place, "locker": str(locker),
               "code": code, "regs": {k: str(v) for k, v in regs.items()},
               "freq": str(right)}
    return " ".join(text_parts), answers


CODE_SPEC = """[TASK SPEC] Build a python module `acc.py`:
- class Account(owner: str) with balance starting at 0
- deposit(x): adds x, rejects (ValueError) non-positive x
[CHANGE 1] Add withdraw(x): subtracts x; ValueError if x non-positive or balance would go negative.
[CHANGE 2] Every successful deposit/withdraw appends ("dep"|"wd", amount) to self.history.
[CHANGE 3] Rename the class to LedgerAccount but keep `Account = LedgerAccount` alias for compatibility.
[CHANGE 4] Add fee: withdrawals over 100 charge an extra 1 unit fee (also recorded in history as ("fee", 1)); fee counts toward the negative-balance check.
"""

CODE_ASSERTS = """
import acc
a = acc.Account("kim")
assert a.balance == 0
a.deposit(150)
assert a.balance == 150
a.withdraw(120)
assert a.balance == 29, a.balance
assert ("fee", 1) in a.history
assert a.history[0] == ("dep", 150)
try:
    a.withdraw(50)
    raise SystemExit("FAIL negative allowed")
except ValueError:
    pass
assert acc.LedgerAccount is acc.Account
print("CODE-TRAJ PASS")
"""


TEMP = 0.0          # set from --temp; 0 = greedy (historical baseline mode)
SAMPLER_SEED = 42   # set from --seed; sent per-request when TEMP > 0

def ask(port, messages, max_tokens=400):
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": TEMP,
            "cache_prompt": True}
    if TEMP > 0:
        # production sampler shape at temp>0; fixed seed for reproducibility
        body.update({"top_p": 0.95, "top_k": 20, "seed": SAMPLER_SEED})
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {KEY}",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        obj = json.load(r)
    c = obj["choices"][0]["message"]["content"]
    # strip think block if present
    c = re.sub(r"<think>.*?</think>", "", c, flags=re.S).strip()
    t = obj.get("timings", {})
    return c, t


def run_depth(port, depth, rng, results):
    corpus, ans = build_corpus(depth, rng)
    sysmsg = {"role": "system",
              "content": "You are precise. Archive follows.\n" + corpus}
    t0 = time.time()
    hop_qs = [
        ("hops-2", f"Which location does the custodian of the Calder dossier operate out of? Answer with the location name only.", ans["place"]),
        ("hops-3", f"What is the storage locker number used by the office of the Calder dossier custodian? Number only.", ans["locker"]),
        ("hops-4", f"What is the seal code on the locker used by the office of the Calder dossier custodian? Code only.", ans["code"]),
    ]
    for name, q, want in hop_qs:
        c, t = ask(port, [sysmsg, {"role": "user", "content": q}], 2048)
        ok = want.lower() in c.lower()
        results.append({"depth": depth, "test": name, "pass": ok,
                        "want": want, "got": c[:90]})
        print(f"  {name}@{depth}: {'PASS' if ok else 'FAIL'} (want {want!r} got {c[:40]!r})")
    # ledger
    q = ("Compute the final values of registers R1..R8 from all LEDGER lines "
         "(SET replaces, ADD adds, SUB subtracts, starting at 0). "
         "Answer as lines 'R1=<n>' .. 'R8=<n>'.")
    c, _ = ask(port, [sysmsg, {"role": "user", "content": q}], 32768)
    got = dict(re.findall(r"(R[1-8])\s*=\s*(-?\d+)", c))
    n_ok = sum(1 for k, v in ans["regs"].items() if got.get(k) == v)
    results.append({"depth": depth, "test": "ledger", "pass": n_ok == 8,
                    "score": f"{n_ok}/8", "want": ans["regs"], "got": got})
    print(f"  ledger@{depth}: {n_ok}/8")
    # correction
    c, _ = ask(port, [sysmsg, {"role": "user", "content":
               "What is the relay frequency in kHz? Number only."}], 400)
    ok = ans["freq"] in c and str(int(ans["freq"]) != 0)
    ok = ans["freq"] in c
    results.append({"depth": depth, "test": "correction", "pass": ok,
                    "want": ans["freq"], "got": c[:60]})
    print(f"  correction@{depth}: {'PASS' if ok else 'FAIL'}")
    print(f"  [depth {depth} wall {round(time.time()-t0)}s]")


def run_code_traj(port, depth, rng, results):
    corpus, _ = build_corpus(depth, rng)
    sysmsg = {"role": "system", "content":
              "Background archive (irrelevant to the task):\n" + corpus}
    user = (CODE_SPEC + "\nProduce the FINAL acc.py after ALL changes. "
            "Output only python code, no fences, no commentary.")
    c, t = ask(port, [sysmsg, {"role": "user", "content": user}], 8192)
    code = re.sub(r"^```(python)?|```$", "", c.strip(), flags=re.M).strip()
    with tempfile.TemporaryDirectory() as td:
        Path(td, "acc.py").write_text(code)
        Path(td, "t.py").write_text(CODE_ASSERTS)
        try:
            p = subprocess.run([sys.executable, "t.py"], cwd=td, timeout=30,
                               capture_output=True, text=True)
            ok = "CODE-TRAJ PASS" in p.stdout
            err = (p.stderr or p.stdout)[-120:]
        except Exception as e:
            ok, err = False, str(e)[:120]
    results.append({"depth": depth, "test": "code-traj", "pass": ok,
                    "err": None if ok else err})
    print(f"  code-traj@{depth}: {'PASS' if ok else 'FAIL'}"
          + ("" if ok else f" ({err})"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8131)
    ap.add_argument("--depths", default="16000,64000")
    ap.add_argument("--label", default="baseline")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temp", type=float, default=0.0,
                    help="sampling temperature (0 = greedy, the historical baseline mode)")
    args = ap.parse_args()
    global TEMP, SAMPLER_SEED
    TEMP = args.temp
    SAMPLER_SEED = args.seed
    depths = [int(x) for x in args.depths.split(",")]
    results = []
    for d in depths:
        print(f"== depth {d} ==")
        run_depth(args.port, d, random.Random(args.seed + d), results)
        run_code_traj(args.port, d, random.Random(args.seed + d + 1), results)
    outdir = Path(__file__).parent / "trajbase"
    outdir.mkdir(exist_ok=True)
    n_pass = sum(1 for r in results if r["pass"])
    summary = {"label": args.label, "seed": args.seed, "depths": depths,
               "pass": n_pass, "total": len(results), "results": results,
               "ts": time.strftime("%Y-%m-%d %H:%M")}
    out = outdir / f"traj_{args.label}.json"
    out.write_text(json.dumps(summary, indent=1))
    print(f"\nTOTAL: {n_pass}/{len(results)} → {out}")


if __name__ == "__main__":
    main()
