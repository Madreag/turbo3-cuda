#!/usr/bin/env python3
"""Morning mega-analysis summarizer: ledger + battery outputs + gate summaries
-> one markdown block for OVERNIGHT-REPORT. Run any time; idempotent."""
import glob, json, os, re

YB = os.path.dirname(os.path.abspath(__file__))
Q = os.path.dirname(YB)

print("## AUTO-SUMMARY\n")
led = open(f"{YB}/ledger.txt").read().splitlines() if os.path.exists(f"{YB}/ledger.txt") else []
done = [l for l in led if l.startswith("DONE")]
bad = [l for l in led if l.startswith(("FAIL", "TIMEOUT", "SOAKEVENT"))]
print(f"ledger: {len(done)} done, {len(bad)} fail/timeout/soak\n")
if bad:
    print("attention lines:")
    for l in bad:
        print(f"  - {l}")
    print()

# battery arms: each .out ends with a summary the battery prints
print("| arm | verdict tail |")
print("|---|---|")
for f in sorted(glob.glob(f"{YB}/yarnB_*.out")):
    tail = ""
    try:
        lines = [l.strip() for l in open(f, errors="replace").read().splitlines() if l.strip()]
        tail = " / ".join(lines[-2:])[:110]
    except OSError:
        pass
    print(f"| {os.path.basename(f)[:34]} | {tail} |")
print()

# quant gates
print("| quant gate | mean KLD | top-1 |")
print("|---|---|---|")
for f in sorted(glob.glob(f"{Q}/kldnvfp4/kld_summary_*.json")):
    try:
        d = json.load(open(f))
        name = os.path.basename(f).replace("kld_summary_", "").replace(".json", "")
        mean = d.get("kld_mean", d.get("mean", "?"))
        top1 = d.get("top1_agree", d.get("top1", "?"))
        print(f"| {name} | {mean} | {top1} |")
    except Exception:
        pass
print()

# speed probes from gate slots
for f in sorted(glob.glob(f"{YB}/gate-*-speed.out")):
    print(f"### {os.path.basename(f)}")
    for l in open(f, errors="replace").read().splitlines():
        if "decode=" in l:
            print(f"    {l.strip()}")
    print()
