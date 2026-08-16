#!/usr/bin/env python3
"""VRAM-law calculator for the TurboQuant serving stack (single 32.6 GB card).

Encodes the hard-won budget law with tonight's measured anchor:
  2026-08-16, prod n_max=3 @327,680 ctx, vision-CPU, ub512, ckpt->host:
  nvidia-smi used = 31,768 MiB, of which ~850 MiB is non-server (WSL/desktop
  baseline measured with the stack down). Fill-ladder verified VRAM is
  STATIC vs fill (+80 MiB one-time first-prefill allocation).

Components (MiB):
  weights Q6_K            21,300
  target KV turbo4        ctx * 16.9 KiB / 1024      (K+V, 16 attn layers)
  draft KV turbo4         ctx * 1.056 KiB / 1024     (1 nextn layer; spec only)
  recurrent state         R * (1 + n_max if spec else 1),  R ~= 100
  residual (compute/graph/CUDA ctx/meta)  fitted at the anchor; scales
     weakly with ctx via the mask/graph term — modeled linear in ctx with
     70% fixed / 30% proportional (single-anchor fit; treat as +/-300 MiB).

Verdicts: PASS (>=700 MiB free), TIGHT (300-700), FAIL (<300) against the
32,607 MiB card minus the 850 MiB baseline.
"""
import argparse

TOTAL = 32607
BASELINE = 850
WEIGHTS = 21300
KV_TGT_KIB = 16.9
KV_DFT_KIB = 1.056
R_MIB = 100
ANCHOR_CTX = 327680
ANCHOR_USED = 31768          # nvidia-smi total at the anchor config
BAND = 300

def residual_at_anchor():
    srv = ANCHOR_USED - BASELINE
    kv = ANCHOR_CTX * KV_TGT_KIB / 1024
    dft = ANCHOR_CTX * KV_DFT_KIB / 1024
    rec = R_MIB * 4              # n_max=3 -> 1+3 copies
    return srv - WEIGHTS - kv - dft - rec

RES_ANCHOR = residual_at_anchor()

# Second anchor (2026-08-16, measured): MTP-OFF @376,832 ctx booted at
# 29,998 MiB total → spec-off residual = 29,998-850-21,300-100-6,218 =
# 1,530 MiB. Spec-off drops ~1.9 GB of draft/spec compute-graph overhead —
# far more than the old 350 MiB estimate.
RES_OFF_ANCHOR = 1530
OFF_ANCHOR_CTX = 376832

def predict(ctx, spec=True, n_max=3):
    kv = ctx * KV_TGT_KIB / 1024
    dft = ctx * KV_DFT_KIB / 1024 if spec else 0.0
    rec = R_MIB * ((1 + n_max) if spec else 1)
    if spec:
        res = RES_ANCHOR * (0.7 + 0.3 * ctx / ANCHOR_CTX)
    else:
        res = RES_OFF_ANCHOR * (0.7 + 0.3 * ctx / OFF_ANCHOR_CTX)
    used = BASELINE + WEIGHTS + kv + dft + rec + res
    free = TOTAL - used
    verdict = "PASS" if free >= 700 else ("TIGHT" if free >= 300 else "FAIL")
    return used, free, verdict

def solve_max_ctx(spec=True, n_max=3, floor=700):
    lo, hi = 32768, 1500000
    while hi - lo > 1024:
        mid = (lo + hi) // 2
        _, free, _ = predict(mid, spec, n_max)
        if free >= floor:
            lo = mid
        else:
            hi = mid
    return lo

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int)
    ap.add_argument("--no-spec", action="store_true")
    ap.add_argument("--n-max", type=int, default=3)
    ap.add_argument("--solve-max-ctx", action="store_true")
    ap.add_argument("--floor", type=int, default=700, help="required free MiB")
    ap.add_argument("--table", action="store_true")
    args = ap.parse_args()

    print(f"(residual at anchor = {RES_ANCHOR:.0f} MiB; predictions +/-{BAND} MiB)")
    if args.table:
        print(f"{'config':<28}{'ctx':>9}{'used':>9}{'free':>7}  verdict")
        for spec, nmax, label in [(True, 3, "spec n3 (prod)"),
                                  (True, 2, "spec n2"),
                                  (False, 0, "MTP OFF")]:
            for ctx in (327680, 360448, 393216, 409600, 458752, 524288):
                used, free, v = predict(ctx, spec, nmax)
                print(f"{label:<28}{ctx:>9}{used:>9.0f}{free:>7.0f}  {v}")
            mx = solve_max_ctx(spec, nmax, args.floor)
            print(f"{label:<28}{'max@'+str(args.floor):>9}{mx:>9} tokens\n")
        return
    if args.solve_max_ctx:
        mx = solve_max_ctx(not args.no_spec, args.n_max, args.floor)
        print(f"max ctx at >= {args.floor} MiB free: {mx}")
        return
    ctx = args.ctx or ANCHOR_CTX
    used, free, v = predict(ctx, not args.no_spec, args.n_max)
    print(f"ctx={ctx} spec={'off' if args.no_spec else f'n{args.n_max}'}: "
          f"used~{used:.0f} MiB, free~{free:.0f} MiB -> {v}")

if __name__ == "__main__":
    main()
