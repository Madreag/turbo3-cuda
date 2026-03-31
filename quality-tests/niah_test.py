#!/usr/bin/env python3
"""
Needle-in-a-Haystack (NIAH) Test for TurboQuant KV Cache Quality

Uses Paul Graham essays as haystack text, inserts a factual "needle" at
various depths, and tests retrieval via llama-server's OpenAI-compatible API.

Prerequisites:
  1. Start llama-server:
     ./build/bin/llama-server -m $MODEL -ctk turbo3 -ctv turbo3 -fa -ngl 99 \
       -c 65536 --port 8080 -mmp 0

  2. Run this script:
     python3 niah_test.py --port 8080 --contexts 4096,8192,16384,32768

Produces a results matrix and optional heatmap data for visualization.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: 'requests' not installed. Run: pip install requests")
    sys.exit(1)


# The "needle" — a specific fact to retrieve
NEEDLE = "The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day."
RETRIEVAL_QUESTION = "What is the best thing to do in San Francisco?"
EXPECTED_ANSWER_FRAGMENTS = ["sandwich", "dolores", "park", "sunny"]


def load_haystack_text(essays_dir: str) -> str:
    """Load all Paul Graham essays into one big string."""
    essays_path = Path(essays_dir)
    if not essays_path.exists():
        # Try the NIAH repo location
        alt = Path(__file__).parent / "LLMTest_NeedleInAHaystack" / "needlehaystack" / "PaulGrahamEssays"
        if alt.exists():
            essays_path = alt
        else:
            print(f"ERROR: Essays directory not found at {essays_dir} or {alt}")
            sys.exit(1)

    texts = []
    for f in sorted(essays_path.glob("*.txt")):
        texts.append(f.read_text(encoding="utf-8", errors="ignore"))

    full_text = "\n\n".join(texts)
    print(f"Loaded {len(texts)} essays, ~{len(full_text.split())} words")
    return full_text


def insert_needle(haystack: str, needle: str, depth_percent: float, context_length_tokens: int) -> str:
    """Insert needle at specified depth within a truncated haystack."""
    # Rough token estimate: 1 token ≈ 4 chars
    target_chars = context_length_tokens * 4
    # Leave room for needle + prompt overhead (~500 tokens)
    hay_chars = target_chars - len(needle) - 2000

    if hay_chars > len(haystack):
        hay_chars = len(haystack)

    # Find insertion point
    insert_char = int(hay_chars * depth_percent / 100.0)
    # Snap to sentence boundary
    period_pos = haystack.rfind(". ", 0, insert_char)
    if period_pos > 0:
        insert_char = period_pos + 2

    # Build: haystack_before + needle + haystack_after
    before = haystack[:insert_char]
    after = haystack[insert_char:hay_chars]

    return before + "\n" + needle + "\n" + after


def query_server(base_url: str, prompt: str, max_tokens: int = 2000) -> str:
    """Query llama-server via OpenAI-compatible chat completions API."""
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": "default",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Answer questions based on the provided context. Be concise."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
    }

    try:
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        return "ERROR: Cannot connect to llama-server. Is it running?"
    except requests.exceptions.Timeout:
        return "ERROR: Timeout (300s)"
    except Exception as e:
        return f"ERROR: {e}"


def evaluate_response(response: str) -> bool:
    """Check if the response contains the expected answer."""
    response_lower = response.lower()
    # Check if at least 2 of the expected fragments are present
    matches = sum(1 for frag in EXPECTED_ANSWER_FRAGMENTS if frag in response_lower)
    return matches >= 2


def check_server(base_url: str) -> bool:
    """Verify llama-server is running and responsive."""
    try:
        resp = requests.get(f"{base_url}/health", timeout=5)
        return resp.status_code == 200
    except:
        return False


def main():
    parser = argparse.ArgumentParser(description="NIAH Test for TurboQuant")
    parser.add_argument("--port", type=int, default=8080, help="llama-server port")
    parser.add_argument("--host", default="localhost", help="llama-server host")
    parser.add_argument("--contexts", default="4096,8192,16384,32768", help="Context lengths to test")
    parser.add_argument("--depths", default="10,25,50,75,90", help="Depth percentages")
    parser.add_argument("--reps", type=int, default=1, help="Repetitions per test point")
    parser.add_argument("--essays-dir", default=None, help="Path to Paul Graham essays")
    parser.add_argument("--output", default=None, help="Output JSON file")
    parser.add_argument("--label", default="turbo3", help="Label for this test run")
    parser.add_argument("--max-tokens", type=int, default=2000, help="Max tokens for generation")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    contexts = [int(x) for x in args.contexts.split(",")]
    depths = [float(x) for x in args.depths.split(",")]

    # Check server
    print(f"Checking llama-server at {base_url}...")
    if not check_server(base_url):
        print(f"ERROR: llama-server not responding at {base_url}")
        print(f"Start it with:")
        print(f"  ./build/bin/llama-server -m $MODEL -ctk turbo3 -ctv turbo3 -fa -ngl 99 -c 65536 --port {args.port} -mmp 0")
        sys.exit(1)
    print("Server OK.")

    # Load haystack
    essays_dir = args.essays_dir or str(Path(__file__).parent / "LLMTest_NeedleInAHaystack" / "needlehaystack" / "PaulGrahamEssays")
    haystack = load_haystack_text(essays_dir)

    print(f"\n=== NIAH Test: {args.label} ===")
    print(f"Contexts: {contexts}")
    print(f"Depths: {depths}")
    print(f"Reps: {args.reps}")
    print(f"Total tests: {len(contexts) * len(depths) * args.reps}")
    print(f"Needle: '{NEEDLE[:60]}...'")
    print(f"Question: '{RETRIEVAL_QUESTION}'")
    print()

    results = []
    total = len(contexts) * len(depths) * args.reps
    done = 0

    for ctx in contexts:
        for depth in depths:
            passes = 0
            for rep in range(args.reps):
                # Build prompt with needle inserted
                context_with_needle = insert_needle(haystack, NEEDLE, depth, ctx)
                prompt = (
                    f"Read the following text carefully:\n\n"
                    f"{context_with_needle}\n\n"
                    f"Based on the text above, {RETRIEVAL_QUESTION}"
                )

                start = time.time()
                response = query_server(base_url, prompt, max_tokens=args.max_tokens)
                elapsed = time.time() - start

                correct = evaluate_response(response)
                if correct:
                    passes += 1
                done += 1

                status = "PASS" if correct else "FAIL"
                print(f"[{done}/{total}] ctx={ctx:>6} depth={depth:>4}% {status} ({elapsed:.1f}s) -> '{response[:80]}'")

                results.append({
                    "context": ctx,
                    "depth": depth,
                    "rep": rep,
                    "response": response[:300],
                    "correct": correct,
                    "elapsed": round(elapsed, 1),
                })

            accuracy = passes / args.reps * 100
            print(f"  --> ctx={ctx} depth={depth}%: {passes}/{args.reps} = {accuracy:.0f}%")
        print()

    # Summary matrix
    print("\n=== RESULTS MATRIX ===")
    print(f"{'Context':>8}", end="")
    for d in depths:
        print(f" {d:>5}%", end="")
    print("   Avg")

    overall_correct = 0
    overall_total = 0

    for ctx in contexts:
        print(f"{ctx:>7}:", end="")
        row_correct = 0
        row_total = 0
        for d in depths:
            matches = [r for r in results if r["context"] == ctx and r["depth"] == d]
            correct = sum(1 for r in matches if r["correct"])
            total_r = len(matches)
            pct = correct / total_r * 100 if total_r > 0 else 0
            row_correct += correct
            row_total += total_r
            overall_correct += correct
            overall_total += total_r

            if pct == 100:
                print(f"   100", end="")
            elif pct > 0:
                print(f"  *{pct:>3.0f}", end="")
            else:
                print(f"  FAIL", end="")
        row_avg = row_correct / row_total * 100 if row_total > 0 else 0
        print(f"  {row_avg:>4.0f}%")

    print(f"\nOverall: {overall_correct}/{overall_total} = {overall_correct/overall_total*100:.1f}%")

    # Save JSON
    out_path = args.output or f"quality-tests/niah_results_{args.label}.json"
    with open(out_path, "w") as f:
        json.dump({
            "label": args.label,
            "needle": NEEDLE,
            "question": RETRIEVAL_QUESTION,
            "contexts": contexts,
            "depths": depths,
            "reps": args.reps,
            "overall_accuracy": overall_correct / overall_total * 100 if overall_total > 0 else 0,
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
