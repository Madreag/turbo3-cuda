#!/usr/bin/env python3
"""
Passkey Retrieval Test for TurboQuant KV Cache Quality Evaluation

Inserts a random passkey (5-digit number) at various depths within filler text,
then asks the model to retrieve it via llama-server's OpenAI-compatible API.

Prerequisites:
  Start llama-server first:
    ./build/bin/llama-server -m $MODEL -ctk turbo3 -ctv turbo3 -fa on -ngl 99 \
      -c 65536 --port 8090 --no-mmap

Usage:
    python3 passkey_retrieval.py --port 8090 --label turbo3 [options]
"""

import argparse
import json
import os
import random
import re
import sys
import time

try:
    import requests
except ImportError:
    print("ERROR: pip install requests")
    sys.exit(1)

FILLER_SENTENCES = [
    "The weather in the city was pleasant with clear skies and mild temperatures throughout the day.",
    "Researchers at the university published new findings about sustainable energy sources last week.",
    "The quarterly financial report showed steady growth across all major business segments.",
    "Local community members gathered at the park to celebrate the annual harvest festival.",
    "Engineers developed an innovative approach to reduce manufacturing costs by thirty percent.",
    "The museum opened a new exhibition featuring contemporary art from emerging international artists.",
    "Transportation infrastructure improvements were announced for the downtown corridor project.",
    "Agricultural experts recommended new irrigation techniques for drought-resistant crop cultivation.",
    "The software development team released a major update with improved security features.",
    "Medical professionals discussed the latest advances in preventive healthcare at the conference.",
    "Environmental monitoring stations recorded normal atmospheric conditions across the region.",
    "The educational program expanded to include online learning options for remote students.",
    "Construction of the new commercial district is expected to be completed by next quarter.",
    "Telecommunications providers upgraded network capacity to support growing data demands.",
    "The scientific expedition discovered previously unknown species in the deep ocean region.",
    "Public libraries introduced digital lending services to improve community access to books.",
    "The manufacturing plant implemented automated quality control systems on all production lines.",
    "Sports analysts predicted strong performances from several teams in the upcoming tournament.",
    "The city council approved new zoning regulations for residential development projects.",
    "International trade agreements were finalized after months of diplomatic negotiations.",
]


def generate_context(n_tokens_approx: int, passkey: str, depth_percent: float) -> str:
    """Generate filler text with a passkey inserted at the specified depth."""
    sentences_needed = (n_tokens_approx // 17) + 10
    needle_pos = max(1, min(int(sentences_needed * depth_percent / 100.0), sentences_needed - 1))
    rng = random.Random(42 + hash(f"{n_tokens_approx}_{depth_percent}_{passkey}"))

    lines = []
    for i in range(sentences_needed):
        if i == needle_pos:
            lines.append(f"The special passkey needed to unlock the system is: {passkey}. Remember this passkey.")
        lines.append(rng.choice(FILLER_SENTENCES))
    return " ".join(lines)


def query_server(base_url: str, context: str, passkey: str) -> tuple:
    """Query llama-server and return (response_text, elapsed_seconds)."""
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": "default",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Read the text carefully and answer precisely."},
            {"role": "user", "content": f"{context}\n\nWhat is the special passkey mentioned in the text above? Respond with ONLY the 5-digit number, nothing else."},
        ],
        "max_tokens": 2000,
        "temperature": 0,
    }
    try:
        start = time.time()
        resp = requests.post(url, json=payload, timeout=300)
        elapsed = time.time() - start
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        return text, elapsed
    except requests.exceptions.ConnectionError:
        return "ERROR: server not running", 0
    except requests.exceptions.Timeout:
        return "ERROR: timeout", 300
    except Exception as e:
        return f"ERROR: {e}", 0


def check_server(base_url: str) -> bool:
    try:
        return requests.get(f"{base_url}/health", timeout=5).status_code == 200
    except:
        return False


def main():
    parser = argparse.ArgumentParser(description="Passkey Retrieval Test")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--contexts", default="2048,4096,8192,16384,32768")
    parser.add_argument("--depths", default="10,25,50,75,90")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--label", default="turbo3")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    contexts = [int(x) for x in args.contexts.split(",")]
    depths = [float(x) for x in args.depths.split(",")]

    print(f"Checking server at {base_url}...")
    if not check_server(base_url):
        print("ERROR: llama-server not responding. Start it first.")
        sys.exit(1)
    print("Server OK.\n")

    print(f"=== Passkey Retrieval: {args.label} ===")
    print(f"Contexts: {contexts}")
    print(f"Depths: {depths}")
    print(f"Reps: {args.reps}")
    total = len(contexts) * len(depths) * args.reps
    print(f"Total tests: {total}\n")

    results = []
    done = 0

    for ctx in contexts:
        for depth in depths:
            passes = 0
            for rep in range(args.reps):
                passkey = f"{random.randint(10000, 99999)}"
                context = generate_context(ctx, passkey, depth)
                response, elapsed = query_server(base_url, context, passkey)

                correct = passkey in response
                if not correct:
                    nums = re.findall(r'\d{5}', response)
                    correct = passkey in nums

                if correct:
                    passes += 1
                done += 1

                status = "PASS" if correct else "FAIL"
                print(f"[{done}/{total}] ctx={ctx:>6} depth={depth:>4}% key={passkey} {status} ({elapsed:.1f}s) -> '{response[:60]}'")

                results.append({
                    "context": ctx, "depth": depth, "rep": rep,
                    "passkey": passkey, "response": response[:200],
                    "correct": correct, "elapsed": round(elapsed, 1),
                })

            print(f"  --> ctx={ctx} depth={depth}%: {passes}/{args.reps} = {passes/args.reps*100:.0f}%")
        print()

    # Summary matrix
    print("\n=== RESULTS MATRIX ===")
    print(f"{'':>8}", end="")
    for d in depths:
        print(f" {d:>5}%", end="")
    print("   Avg")

    for ctx in contexts:
        print(f"{ctx:>7}:", end="")
        row_c, row_t = 0, 0
        for d in depths:
            matches = [r for r in results if r["context"] == ctx and r["depth"] == d]
            c = sum(1 for r in matches if r["correct"])
            t = len(matches)
            row_c += c; row_t += t
            pct = c / t * 100 if t else 0
            print(f"  {pct:>4.0f}", end="")
        print(f"  {row_c/row_t*100:>4.0f}%")

    total_c = sum(1 for r in results if r["correct"])
    print(f"\nOverall: {total_c}/{len(results)} = {total_c/len(results)*100:.1f}%")

    out = args.output or f"quality-tests/passkey_results_{args.label}.json"
    with open(out, "w") as f:
        json.dump({"label": args.label, "contexts": contexts, "depths": depths,
                   "reps": args.reps, "overall_accuracy": total_c/len(results)*100,
                   "results": results}, f, indent=2)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
