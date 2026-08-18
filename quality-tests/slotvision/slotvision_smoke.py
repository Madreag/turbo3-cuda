#!/usr/bin/env python3
"""Vision x slot-save round-trip gate (upstream #27278/#27274 adoption).

Against a TEST server on 8233 (new binary, original weights+mmproj, small ctx,
--slot-save-path set, no spec): image chat -> slot save -> erase -> restore ->
continue about the image. PASS = all steps 200, color answered correctly both
times, restore continuation reuses the cache (small prompt_n), no error lines.

Usage: slotvision_smoke.py [port]
"""
import base64, json, sys, urllib.request

HERE = "/home/erol/ai/turboquant/turboquant-g1/quality-tests/slotvision"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8233
BASE = f"http://127.0.0.1:{PORT}"

img_b64 = base64.b64encode(open(f"{HERE}/red256.png", "rb").read()).decode()
IMG_URL = f"data:image/png;base64,{img_b64}"

MSGS = [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": IMG_URL}},
    {"type": "text", "text": "What solid color is this image? Answer with just the color name."},
]}]


def post(path, body):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


fails = []

# 1. image turn
r1 = post("/v1/chat/completions", {"messages": MSGS, "max_tokens": 300,
                                   "temperature": 0, "cache_prompt": True})
a1 = r1["choices"][0]["message"]["content"]
ok1 = "red" in a1.lower()
print(f"1. image turn: {'PASS' if ok1 else 'FAIL'} -> {a1.strip()[:80]!r}")
ok1 or fails.append("image-turn")

# 2. slot save
r2 = post("/slots/0?action=save", {"filename": "visiontest.bin"})
print(f"2. slot save: n_saved={r2.get('n_saved')} ({r2.get('filename')})")
r2.get("n_saved", 0) > 0 or fails.append("save")

# 3. erase
r3 = post("/slots/0?action=erase", {})
print(f"3. erase: {r3}")

# 4. restore
r4 = post("/slots/0?action=restore", {"filename": "visiontest.bin"})
print(f"4. restore: n_restored={r4.get('n_restored')}")
r4.get("n_restored", 0) > 0 or fails.append("restore")

# 5. continuation about the image on the restored cache
msgs2 = MSGS + [{"role": "assistant", "content": a1},
                {"role": "user", "content": "What color was the image I showed you? One word."}]
r5 = post("/v1/chat/completions", {"messages": msgs2, "max_tokens": 300,
                                   "temperature": 0, "cache_prompt": True})
a2 = r5["choices"][0]["message"]["content"]
pn = r5.get("usage", {}).get("prompt_tokens")
t5 = r5.get("timings", {})
ok5 = "red" in a2.lower()
print(f"5. post-restore continuation: {'PASS' if ok5 else 'FAIL'} -> {a2.strip()[:80]!r}")
print(f"   prompt_tokens={pn} prompt_n_processed={t5.get('prompt_n')} "
      f"(small processed = restored cache reused)")
ok5 or fails.append("continuation")
if t5.get("prompt_n") is not None and t5["prompt_n"] > 200:
    print("   WARN: large reprocess — placeholder reuse may not have engaged")
    fails.append("cache-reuse")

print("GATE:", "PASS" if not fails else f"FAIL ({','.join(fails)})")
sys.exit(0 if not fails else 1)
