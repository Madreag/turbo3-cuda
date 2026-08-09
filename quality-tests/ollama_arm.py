#!/usr/bin/env python3
"""P2 independent-runtime arm: same pagoda prompt through ollama AS SHIPPED —
its own quant, its own sampling defaults, zero overrides beyond output budget.
Usage: ollama_arm.py <n>"""
import json, re, sys, time, urllib.request

OUT = "/tmp/claude-1000/-home-erol-ai-turboquant-turboquant-kv-cache/8aa7a7bf-b854-4d9b-aea4-ce4faf0656f1/scratchpad/ollama-artifacts"
PROMPT = ("Design and create a very creative, elaborate, and detailed voxel art scene of a pagoda "
          "in a beautiful garden with trees, including some cherry blossoms. Make the scene "
          "impressive and varied and use colorful voxels. Use whatever libraries to get this done "
          "but make sure I can paste it all into a single HTML file and open it in Chrome.")

import os
os.makedirs(OUT, exist_ok=True)
n = int(sys.argv[1])
for run in range(1, n + 1):
    t0 = time.time()
    req = urllib.request.Request("http://127.0.0.1:11434/v1/chat/completions",
        data=json.dumps({"model": "qwen3.6:27b", "max_tokens": 30000, "stream": True,
                         "messages": [{"role": "user", "content": PROMPT}]}).encode(),
        headers={"Content-Type": "application/json"})
    try:
        c = ""; fin = None
        with urllib.request.urlopen(req, timeout=2400) as r:
            for line in r:
                if not line.startswith(b"data: "): continue
                body = line[6:].strip()
                if body == b"[DONE]": break
                try: ch = json.loads(body)["choices"][0]
                except (json.JSONDecodeError, KeyError, IndexError): continue
                fin = ch.get("finish_reason") or fin
                c += (ch.get("delta") or {}).get("content") or ""
        m = re.search(r"```(?:html)?\s*\n(.*?)```", c, re.S)
        html = m.group(1) if m else (c if "<html" in c.lower() else "")
        open(f"{OUT}/ollama{run}.html", "w").write(html)
        print(f"[ollama run {run}/{n}] {time.time()-t0:.0f}s | {len(html):,}ch | "
              f"closed={'</html>' in html.lower()} | finish={fin}", flush=True)
    except Exception as e:
        print(f"[ollama run {run}/{n}] ERROR: {str(e)[:120]}", flush=True)
print("[ollama] GENERATION DONE", flush=True)
