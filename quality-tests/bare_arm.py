#!/usr/bin/env python3
"""Bare-context diagnostic arm: pagoda prompt as a plain chat request (no system
prompt, no tools) — the chat-UI shape. Extracts fenced HTML from content and saves
to the artifacts dir for the standard render gate. Usage: bare_arm.py <n> [port]"""
import json, re, sys, time, urllib.request

KEY = open("/home/erol/.config/llama-tcq/api.key").read().strip()
OUT = "/home/erol/.config/llama-tcq/artifacts"
PROMPT = ("Design and create a very creative, elaborate, and detailed voxel art scene of a pagoda "
          "in a beautiful garden with trees, including some cherry blossoms. Make the scene "
          "impressive and varied and use colorful voxels. Use whatever libraries to get this done "
          "but make sure I can paste it all into a single HTML file and open it in Chrome.")

n, port = int(sys.argv[1]), (int(sys.argv[2]) if len(sys.argv) > 2 else 8130)
for run in range(1, n + 1):
    t0 = time.time()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps({"model": "qwen3.6-27b", "max_tokens": 30000, "stream": True,
                         "messages": [{"role": "user", "content": PROMPT}]}).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    try:
        c = ""
        fin = None
        with urllib.request.urlopen(req, timeout=1800) as r:
            for line in r:
                if not line.startswith(b"data: "):
                    continue
                body = line[6:].strip()
                if body == b"[DONE]":
                    break
                try:
                    ch = json.loads(body)["choices"][0]
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                fin = ch.get("finish_reason") or fin
                c += (ch.get("delta") or {}).get("content") or ""
        d = {"choices": [{"message": {"content": c}, "finish_reason": fin}]}
        m = re.search(r"```(?:html)?\s*\n(.*?)```", c, re.S)
        html = m.group(1) if m else (c if "<html" in c.lower() else "")
        path = f"{OUT}/bare{run}_{int(time.time())}.html"
        open(path, "w").write(html)
        print(f"[bare run {run}/{n}] {time.time()-t0:.0f}s | {len(html):,}ch | "
              f"closed={'</html>' in html.lower()} | finish={d['choices'][0].get('finish_reason')}", flush=True)
    except Exception as e:
        print(f"[bare run {run}/{n}] ERROR: {e}", flush=True)
print("[bare] GENERATION DONE", flush=True)
