#!/usr/bin/env python3
"""Build the P0 probe corpus: realistic coding-agent long context.

Mix: fork C++/CUDA source + project markdown + synthetic agent-transcript glue
(tool outputs, diffs, prose). Deterministic. Output: corpus.txt (~1.2 MB text
-> comfortably >128K tokens for a code-heavy tokenizer).
"""
import os, random, sys

SYNC = "/home/erol/ai/turboquant/turboquant-sync"
G1 = "/home/erol/ai/turboquant/turboquant-g1"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus.txt")

random.seed(42)

code_files = []
for root, dirs, files in os.walk(os.path.join(SYNC, "src")):
    for f in sorted(files):
        if f.endswith((".cpp", ".h")):
            code_files.append(os.path.join(root, f))
for root, dirs, files in os.walk(os.path.join(SYNC, "ggml", "src", "ggml-cuda")):
    dirs[:] = [d for d in dirs if d not in ("template-instances",)]
    for f in sorted(files):
        if f.endswith((".cu", ".cuh")):
            code_files.append(os.path.join(root, f))

md_files = [os.path.join(G1, f) for f in sorted(os.listdir(G1)) if f.endswith(".md")]

GLUE = [
    "\n\n=== TOOL OUTPUT (tests) ===\nRunning suite... 49 passed, 0 failed. "
    "Coverage 87.3%. Slowest: test_restore_pipeline 4.2s.\n\n",
    "\n\n=== USER ===\nNow check whether the scheduler handles the overflow "
    "case we discussed, and summarize what the guard in the previous file "
    "actually protects against.\n\n=== ASSISTANT ===\nLooking at the guard: it "
    "protects the ring index from wrapping when n_past exceeds the window. "
    "The overflow case is handled by the clamp added two files back.\n\n",
    "\n\n=== TOOL OUTPUT (git diff) ===\n--- a/src/scheduler.cpp\n+++ b/src/"
    "scheduler.cpp\n@@ -142,6 +142,9 @@\n+    if (n_queued > cap) {\n+        "
    "return QUEUE_FULL;\n+    }\n\n",
    "\n\n=== ASSISTANT (analysis) ===\nThe allocation path above pins the "
    "buffer before the copy; if the device pool is exhausted it falls back to "
    "host staging, which explains the latency spike in the trace earlier.\n\n",
]

parts = []
total = 0
gi = 0
budget = 1_400_000  # chars
files = []
# interleave: 3 code files then 1 md, round-robin
ci, mi = 0, 0
while total < budget and (ci < len(code_files) or mi < len(md_files)):
    for _ in range(3):
        if ci < len(code_files):
            files.append(code_files[ci]); ci += 1
    if mi < len(md_files):
        files.append(md_files[mi]); mi += 1

for path in files:
    if total >= budget:
        break
    try:
        with open(path, "r", errors="ignore") as f:
            txt = f.read()
    except OSError:
        continue
    if len(txt) > 60_000:
        txt = txt[:60_000]
    header = f"\n\n=== FILE: {os.path.relpath(path, '/home/erol/ai/turboquant')} ===\n"
    parts.append(header + txt)
    total += len(header) + len(txt)
    parts.append(GLUE[gi % len(GLUE)])
    total += len(GLUE[gi % len(GLUE)])
    gi += 1

corpus = "".join(parts)
with open(OUT, "w") as f:
    f.write(corpus)
print(f"corpus: {len(corpus)} chars, {len(files)} files -> {OUT}")
