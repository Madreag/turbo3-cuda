#!/bin/bash
# Render + pixel-verify the N newest wire-captured artifacts for one diagnostic arm.
# Usage: render_arm.sh <arm-label> <n>
set -u
LABEL=${1:?arm label}; N=${2:?count}
S=/tmp/claude-1000/-home-erol-ai-turboquant-turboquant-kv-cache/8aa7a7bf-b854-4d9b-aea4-ce4faf0656f1/scratchpad
P=/mnt/c/Users/Public/render-truth
CHROME="/mnt/c/Program Files/Google/Chrome/Application/chrome.exe"
mkdir -p "$P"
SRC="${3:-/home/erol/.config/llama-tcq/artifacts}"
i=0
[ "${RENDER_SKIP:-0}" = "1" ] && i=$N   # analyze-only mode: use pre-staged shots
for ART in $(ls -t "$SRC"/*.html 2>/dev/null | head -"$N"); do
  [ "$i" -ge "$N" ] && break
  i=$((i+1))
  cp "$ART" "$P/${LABEL}_$i.html"
  timeout 45 "$CHROME" --headless=new --no-first-run --enable-logging=stderr --v=0 \
    --window-size=1280,900 --virtual-time-budget=12000 \
    --screenshot="C:\\Users\\Public\\render-truth\\shot_${LABEL}_$i.png" \
    "file:///C:/Users/Public/render-truth/${LABEL}_$i.html" >/dev/null 2> "$S/chrome_${LABEL}_$i.err"
  [ $? -eq 124 ] && rm -f "$P/shot_${LABEL}_$i.png"
done
"$S/venv/bin/python" - "$LABEL" "$N" <<'EOF'
from PIL import Image
import colorsys, re, os, sys
label, n = sys.argv[1], int(sys.argv[2])
S = "/tmp/claude-1000/-home-erol-ai-turboquant-turboquant-kv-cache/8aa7a7bf-b854-4d9b-aea4-ce4faf0656f1/scratchpad"
passes = 0
for i in range(1, n + 1):
    p = f"/mnt/c/Users/Public/render-truth/shot_{label}_{i}.png"
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        print(f"{label}_{i}: HUNG/no-shot -> FAIL"); continue
    im = Image.open(p).convert("RGB").resize((320, 225))
    px = list(im.getdata())
    mb = sum(sum(q) for q in px) / (len(px) * 3)
    hues = set()
    for r, g, b in px[::7]:
        if max(r, g, b) - min(r, g, b) > 30:
            hues.add(int(colorsys.rgb_to_hsv(r/255, g/255, b/255)[0] * 24))
    try: err = open(f"{S}/chrome_{label}_{i}.err").read()
    except FileNotFoundError: err = ""
    unc = len(re.findall(r"Uncaught|SyntaxError", err))
    # STRUCTURE gate (anti-snow): saturated pixels must cover real area AND form
    # large connected blobs (buildings/trees), not scattered specks.
    w, h = 160, 112
    sm = im.resize((w, h))
    grid = [[0]*w for _ in range(h)]
    sat_count = 0
    data = list(sm.getdata())
    for yy in range(h):
        for xx in range(w):
            r, g, b = data[yy*w + xx]
            if max(r,g,b) - min(r,g,b) > 30 and max(r,g,b) > 50:
                grid[yy][xx] = 1; sat_count += 1
    colored_frac = sat_count / (w*h)
    best_blob = 0
    seen = [[False]*w for _ in range(h)]
    for yy in range(h):
        for xx in range(w):
            if grid[yy][xx] and not seen[yy][xx]:
                stack = [(yy,xx)]; seen[yy][xx] = True; sz = 0
                while stack:
                    cy, cx = stack.pop(); sz += 1
                    for dy2, dx2 in ((1,0),(-1,0),(0,1),(0,-1)):
                        ny, nx = cy+dy2, cx+dx2
                        if 0 <= ny < h and 0 <= nx < w and grid[ny][nx] and not seen[ny][nx]:
                            seen[ny][nx] = True; stack.append((ny,nx))
                best_blob = max(best_blob, sz)
    blob_frac = best_blob / (w*h)
    ok = (mb > 25 and len(hues) >= 8 and unc == 0
          and colored_frac >= 0.08 and blob_frac >= 0.01)
    passes += ok
    print(f"{label}_{i}: brightness={mb:.0f} hues={len(hues)} colored={colored_frac*100:.0f}% "
          f"biggest-blob={blob_frac*100:.1f}% uncaught={unc} -> {'PASS' if ok else 'FAIL'}")
print(f"ARM {label}: {passes}/{n} clean (gates: bright>25, hues>=8, colored>=8%, blob>=1%, 0 uncaught)")
EOF
