#!/usr/bin/env python3
"""Build deterministic code/prose prefix files for the n-max sweep / 15:00 repro.
Same bytes every run -> paired arms see identical prefixes."""
import glob, os

OUT = os.path.dirname(os.path.abspath(__file__))
REPO = "/home/erol/ai/turboquant/turboquant-sync"
G1 = "/home/erol/ai/turboquant/turboquant-g1"

code_files = sorted(glob.glob(f"{REPO}/ggml/src/ggml-cuda/*.cu"))[:40]
buf, total = [], 0
for f in code_files:
    t = open(f, errors="replace").read()
    buf.append(f"// ===== FILE: {os.path.basename(f)} =====\n" + t)
    total += len(t)
    if total > 160_000:
        break
code = "\n".join(buf)[:160_000]
open(f"{OUT}/prefix_code.txt", "w").write(
    "// Continue this CUDA codebase. Study the style and write more kernels.\n" + code)

prose_files = sorted(glob.glob(f"{G1}/*.md"))
buf, total = [], 0
for f in prose_files:
    t = open(f, errors="replace").read()
    buf.append(t)
    total += len(t)
    if total > 190_000:
        break
prose = "\n\n".join(buf)[:190_000]
open(f"{OUT}/prefix_prose.txt", "w").write(
    "The following are engineering notes. Read them, then continue writing "
    "a detailed narrative essay about this project.\n\n" + prose)

for n in ("prefix_code.txt", "prefix_prose.txt"):
    print(n, os.path.getsize(f"{OUT}/{n}"), "bytes")
