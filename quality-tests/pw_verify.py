#!/usr/bin/env python3
"""Playwright artifact verifier — desktop + mobile-viewport render truth.

Usage: pw_verify.py <file.html> [more.html ...]
Per file and per context prints: console errors, page errors, pixel verdict.
(Chromium engine only: emulates phone viewport/touch, not iOS-Safari's missing
importmap support — that gap is documented, not simulatable here.)
"""
import sys, json, pathlib
from playwright.sync_api import sync_playwright

def pixel_verdict(png_path):
    from PIL import Image
    import colorsys
    im = Image.open(png_path).convert("RGB").resize((320, 225))
    px = list(im.getdata())
    mb = sum(sum(p) for p in px) / (len(px) * 3)
    hues = set()
    for r, g, b in px[::7]:
        if max(r, g, b) - min(r, g, b) > 30:
            hues.add(int(colorsys.rgb_to_hsv(r/255, g/255, b/255)[0] * 24))
    sat = sum(1 for r, g, b in px if max(r,g,b)-min(r,g,b) > 30 and max(r,g,b) > 50) / len(px)
    return dict(brightness=round(mb), hues=len(hues), colored_pct=round(sat*100),
                ok=bool(mb > 25 and len(hues) >= 8 and sat >= 0.08))

def verify(path):
    url = pathlib.Path(path).resolve().as_uri()
    out = {}
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        for ctx_name, kwargs in (
            ("desktop", dict(viewport={"width": 1280, "height": 900})),
            ("mobile", dict(viewport={"width": 390, "height": 844}, is_mobile=True,
                            has_touch=True, device_scale_factor=3)),
        ):
            ctx = browser.new_context(**kwargs)
            page = ctx.new_page()
            console, page_errors = [], []
            page.on("console", lambda m: console.append(f"{m.type}: {m.text[:120]}"))
            page.on("pageerror", lambda e: page_errors.append(str(e)[:160]))
            try:
                page.goto(url, timeout=30000)
                page.wait_for_timeout(9000)
                shot = f"/tmp/pw_{ctx_name}_{pathlib.Path(path).stem}.png"
                page.screenshot(path=shot)
                v = pixel_verdict(shot)
            except Exception as e:
                v = dict(ok=False, error=str(e)[:160])
            errs = [c for c in console if c.startswith(("error", "warning: Uncaught"))]
            out[ctx_name] = dict(pixels=v, page_errors=page_errors, console_errors=errs[:4],
                                 verdict="PASS" if (v.get("ok") and not page_errors) else "FAIL")
            ctx.close()
        browser.close()
    return out

if __name__ == "__main__":
    for f in sys.argv[1:]:
        print(f"== {f}")
        print(json.dumps(verify(f), indent=1))
