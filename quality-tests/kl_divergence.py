#!/usr/bin/env python3
"""
KL Divergence measurement: turbo KV types vs f16 baseline.

For each prompt, request logprobs from the server with max_tokens=1.
Compare the top-k log probability distributions between f16 and turbo types.

Metrics:
  - KL divergence (bits)
  - Top-1 agreement rate (same most-probable token)
  - Delta-p RMS (root mean square of probability differences)

Usage:
  1. Start server: ./build/bin/llama-server -m $MODEL -ctk f16 -ctv f16 -fa on -ngl 99 -c 4096 --port 8090 --no-mmap
  2. Run: python3 kl_divergence.py --port 8090 --type f16
  3. Restart server with turbo type, run again
  4. Compare results

Or use --auto mode which starts/stops servers automatically (requires build path).
"""

import argparse
import os
import json
import math
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: 'requests' not installed. Run: pip install requests")
    sys.exit(1)


def load_prompts(wiki_path: str, n_prompts: int = 100, prompt_tokens: int = 256) -> list:
    """Load n_prompts from wikitext, each ~prompt_tokens tokens long."""
    text = Path(wiki_path).read_text(encoding="utf-8", errors="ignore")
    # Split into chunks of ~prompt_tokens * 4 chars (rough token estimate)
    chunk_size = prompt_tokens * 4
    prompts = []
    for i in range(0, len(text) - chunk_size, chunk_size):
        chunk = text[i:i + chunk_size].strip()
        if len(chunk) > 100:  # skip empty/short chunks
            prompts.append(chunk)
        if len(prompts) >= n_prompts:
            break
    print(f"Loaded {len(prompts)} prompts (~{prompt_tokens} tokens each)")
    return prompts


def get_logprobs(base_url: str, prompt: str, top_logprobs: int = 10) -> dict:
    """Request next-token logprobs from llama-server."""
    url = f"{base_url}/v1/completions"
    payload = {
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0,
        "logprobs": top_logprobs,
        # Never accumulate prompts in the slot cache: with --no-context-shift
        # a filling cache 503s after ~n_ctx/prompt_len requests, and cache
        # reuse lets prompts interfere. Every probe stands alone.
        "cache_prompt": False,
    }
    try:
        _key = os.environ.get("TCQ_KEY", "")
        _hdrs = {"Authorization": f"Bearer {_key}"} if _key else {}
        resp = requests.post(url, json=payload, timeout=240, headers=_hdrs)
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        if "logprobs" in choice and choice["logprobs"]:
            lp = choice["logprobs"]
            # llama-server format: logprobs.content[0].top_logprobs = [{token, logprob}, ...]
            if "content" in lp and lp["content"]:
                top_lps = lp["content"][0].get("top_logprobs", [])
                return {item["token"]: item["logprob"] for item in top_lps}
            # Fallback: old format
            if "top_logprobs" in lp and lp["top_logprobs"]:
                return lp["top_logprobs"][0]
        return {}
    except Exception as e:
        print(f"  ERROR: {e}")
        return {}


def compute_kl_divergence(p_logprobs: dict, q_logprobs: dict) -> tuple:
    """
    Compute KL(P || Q) where P=f16 (reference) and Q=turbo (test).
    Returns (kld, top1_match, delta_p_sq).
    """
    if not p_logprobs or not q_logprobs:
        return None, None, None

    # Get union of tokens
    all_tokens = set(p_logprobs.keys()) | set(q_logprobs.keys())

    # Convert log probs to probs, with smoothing for missing tokens
    MIN_LOG = -100.0  # floor for missing tokens

    p_probs = {}
    q_probs = {}
    for tok in all_tokens:
        p_lp = p_logprobs.get(tok, MIN_LOG)
        q_lp = q_logprobs.get(tok, MIN_LOG)
        p_probs[tok] = math.exp(p_lp)
        q_probs[tok] = math.exp(q_lp)

    # Normalize
    p_sum = sum(p_probs.values())
    q_sum = sum(q_probs.values())
    if p_sum == 0 or q_sum == 0:
        return None, None, None

    for tok in all_tokens:
        p_probs[tok] /= p_sum
        q_probs[tok] /= q_sum

    # KL(P || Q) = sum_x P(x) * log(P(x) / Q(x))
    kld = 0.0
    for tok in all_tokens:
        p = p_probs[tok]
        q = q_probs[tok]
        if p > 1e-30 and q > 1e-30:
            kld += p * math.log(p / q)

    # Top-1 agreement
    p_top = max(p_logprobs, key=p_logprobs.get) if p_logprobs else None
    q_top = max(q_logprobs, key=q_logprobs.get) if q_logprobs else None
    top1_match = 1 if p_top == q_top else 0

    # Delta-p squared (for RMS)
    delta_p_sq = 0.0
    for tok in all_tokens:
        dp = p_probs.get(tok, 0) - q_probs.get(tok, 0)
        delta_p_sq += dp * dp

    return kld, top1_match, delta_p_sq


def check_server(base_url: str) -> bool:
    try:
        resp = requests.get(f"{base_url}/health", timeout=5)
        return resp.status_code == 200
    except:
        return False


def main():
    parser = argparse.ArgumentParser(description="KL Divergence: turbo vs f16")
    parser.add_argument("--port", type=int, default=8090, help="llama-server port")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--type", required=True, help="KV type label (f16, q8_0, turbo3, etc.)")
    parser.add_argument("--wiki", default=None, help="Path to wiki.test.raw")
    parser.add_argument("--n-prompts", type=int, default=100)
    parser.add_argument("--prompt-tokens", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=10, help="Number of top logprobs to request")
    parser.add_argument("--output-dir", default="quality-tests", help="Output directory")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    # Find wiki file
    wiki_path = args.wiki
    if not wiki_path:
        import glob
        candidates = glob.glob("/home/erol/ai/turboquant/**/wiki.test.raw", recursive=True)
        if candidates:
            wiki_path = candidates[0]
        else:
            print("ERROR: wiki.test.raw not found")
            sys.exit(1)

    print(f"Server: {base_url}")
    print(f"Type: {args.type}")
    print(f"Wiki: {wiki_path}")

    if not check_server(base_url):
        print(f"ERROR: Server not responding at {base_url}")
        sys.exit(1)

    prompts = load_prompts(wiki_path, args.n_prompts, args.prompt_tokens)

    print(f"\nCollecting logprobs for {len(prompts)} prompts...")
    results = []
    for i, prompt in enumerate(prompts):
        lp = get_logprobs(base_url, prompt, args.top_k)
        results.append(lp)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(prompts)}")

    # Save raw logprobs
    out_path = Path(args.output_dir) / f"kld_logprobs_{args.type}.json"
    with open(out_path, "w") as f:
        json.dump({"type": args.type, "prompt_tokens": args.prompt_tokens, "n_prompts": len(results), "logprobs": results}, f)
    print(f"\nSaved {len(results)} logprob sets to {out_path}")

    # If f16 logprobs exist, compute KLD
    f16_path = Path(args.output_dir) / "kld_logprobs_f16.json"
    if args.type != "f16" and not f16_path.exists():
        print(f"\nWARNING: no f16 reference at {f16_path} — captured logprobs only, "
              f"NO KLD computed. Capture a reference with --type f16 first.")
    if args.type != "f16" and f16_path.exists():
        print(f"\nComputing KL divergence vs f16...")
        with open(f16_path) as f:
            f16_data = json.load(f)
        f16_lps = f16_data["logprobs"]
        ref_pt = f16_data.get("prompt_tokens")
        if ref_pt is not None and ref_pt != args.prompt_tokens:
            print(f"  WARNING: reference prompt_tokens={ref_pt} != run prompt_tokens="
                  f"{args.prompt_tokens} — DIFFERENT PROMPTS, comparison invalid!")

        n = min(len(f16_lps), len(results))
        if len(f16_lps) != len(results):
            print(f"  NOTE: clamping comparison to {n} prompts "
                  f"(reference has {len(f16_lps)}, run has {len(results)})")
        klds = []
        top1_matches = 0
        delta_p_sqs = []
        n_skipped = 0

        for i in range(n):
            kld, match, dp_sq = compute_kl_divergence(f16_lps[i], results[i])
            if kld is not None:
                klds.append(kld)
                top1_matches += match
                delta_p_sqs.append(dp_sq)
            else:
                n_skipped += 1
        if n_skipped:
            # Silent exclusion once hid a 52-prompt 503 cascade behind a
            # perfect-looking 0.0 summary. Never again.
            print(f"  WARNING: {n_skipped}/{n} prompts skipped (empty/error responses)")

        if klds:
            mean_kld = sum(klds) / len(klds)
            top1_pct = top1_matches / len(klds) * 100
            rms_dp = math.sqrt(sum(delta_p_sqs) / len(delta_p_sqs))
            # tail metrics (club-3090 critique: mean-level claims hide tail
            # damage concentrated on JSON/tool tokens). Percentiles over the
            # per-prompt KLD samples; resolution scales with --n-prompts.
            ks = sorted(klds)
            def pct(p):
                # nearest-rank: ceil(p/100*n)-1 (fix: int(p/100*n) gave p99==max at n=100)
                import math as _m
                return ks[min(len(ks) - 1, max(0, _m.ceil(p / 100.0 * len(ks)) - 1))]
            p50, p90, p99, kmax = pct(50), pct(90), pct(99), ks[-1]
            print(f"\n{'='*50}")
            print(f"KL Divergence: {args.type} vs f16")
            print(f"{'='*50}")
            print(f"  KLD (mean):     {mean_kld:.6f}")
            print(f"  KLD p50/p90:    {p50:.6f} / {p90:.6f}")
            print(f"  KLD p99/max:    {p99:.6f} / {kmax:.6f}")
            print(f"  Top-1 agree:    {top1_pct:.1f}%")
            print(f"  Delta-p RMS:    {rms_dp:.6f}")
            print(f"  N prompts:      {len(klds)}")

            # Save summary
            summary_path = Path(args.output_dir) / f"kld_summary_{args.type}.json"
            with open(summary_path, "w") as f:
                json.dump({
                    "type": args.type,
                    "vs": "f16",
                    "kld_mean": mean_kld,
                    "kld_p50": p50,
                    "kld_p90": p90,
                    "kld_p99": p99,
                    "kld_max": kmax,
                    "top1_agreement_pct": top1_pct,
                    "delta_p_rms": rms_dp,
                    "n_prompts": len(klds),
                    "n_skipped": n_skipped,
                    "prompt_tokens": args.prompt_tokens,
                }, f, indent=2)
            print(f"  Summary: {summary_path}")
    elif args.type == "f16":
        print("f16 baseline collected. Run again with --type turbo3 etc. to compute KLD.")


if __name__ == "__main__":
    main()
