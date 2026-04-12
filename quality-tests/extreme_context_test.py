#!/usr/bin/env python3
"""
Extreme context NIAH test — loads ~500K tokens of filler text with a hidden fact,
then asks the model to retrieve it.
"""
import glob
import os
import requests
import sys
import time
import json

SERVER = "http://localhost:8080"
NEEDLE = "The secret project codename for the quantum computing initiative at DARPA is Operation Midnight Falcon, which was approved on July 14th, 2019 with a budget of $3.7 billion."

def wait_for_server(timeout=120):
    print(f"Waiting for server at {SERVER}...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{SERVER}/health", timeout=5)
            if r.status_code == 200:
                print("Server ready!")
                return True
        except:
            pass
        time.sleep(2)
    print("Server timeout!")
    return False

def build_haystack(target_tokens=500000):
    """Build a massive text haystack from wikitext, repeating as needed."""
    if os.environ.get("WIKITEXT_PATHS"):
        wiki_paths = os.environ["WIKITEXT_PATHS"].split(":")
    elif os.environ.get("WIKITEXT_DIR"):
        wiki_paths = sorted(glob.glob(os.path.join(os.environ["WIKITEXT_DIR"], "**", "*.raw"), recursive=True))
    else:
        print("ERROR: Set WIKITEXT_PATHS (colon-separated .raw file paths) or WIKITEXT_DIR environment variable")
        sys.exit(1)
    
    text = ""
    for path in wiki_paths:
        try:
            with open(path, 'r') as f:
                text += f.read() + "\n\n"
        except:
            pass
    
    if not text:
        print("ERROR: No wikitext files found")
        sys.exit(1)
    
    # Rough estimate: 1 token ≈ 4 chars for English text
    target_chars = target_tokens * 4
    
    # Repeat text until we hit target
    base_len = len(text)
    repeats = (target_chars // base_len) + 1
    haystack = (text * repeats)[:target_chars]
    
    print(f"Haystack: {len(haystack):,} chars (~{len(haystack)//4:,} est. tokens)")
    return haystack

def inject_needle(haystack, needle, position_pct=10):
    """Insert needle at position_pct% through the haystack."""
    pos = int(len(haystack) * position_pct / 100)
    # Find a paragraph break near the target position
    newline_pos = haystack.find("\n\n", pos)
    if newline_pos == -1:
        newline_pos = pos
    
    result = haystack[:newline_pos] + f"\n\n{needle}\n\n" + haystack[newline_pos:]
    print(f"Needle injected at ~{position_pct}% ({newline_pos:,} chars)")
    return result

def test_retrieval(prompt_text):
    """Send the haystack + question to the server."""
    question = "What is the secret project codename for the quantum computing initiative at DARPA? What was the budget and approval date? Answer with the exact details."
    
    full_prompt = prompt_text + "\n\n---\n\nBased on ALL the text above, answer this question:\n" + question
    
    print(f"\nSending request ({len(full_prompt):,} chars)...")
    print(f"Question: {question}")
    
    start = time.time()
    try:
        r = requests.post(
            f"{SERVER}/v1/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": "Answer directly and concisely. Do not think step by step. Do not use <think> tags."},
                    {"role": "user", "content": full_prompt}
                ],
                "max_tokens": 200,
                "temperature": 0,
            },
            timeout=1800  # 30 min timeout for huge context
        )
        elapsed = time.time() - start
        
        if r.status_code != 200:
            print(f"ERROR: HTTP {r.status_code}")
            print(r.text[:500])
            return
        
        data = r.json()
        answer = data['choices'][0]['message']['content']
        usage = data.get('usage', {})
        
        print(f"\n{'='*60}")
        print(f"ANSWER: {answer}")
        print(f"{'='*60}")
        print(f"Time: {elapsed:.1f}s")
        print(f"Prompt tokens: {usage.get('prompt_tokens', '?'):,}")
        print(f"Completion tokens: {usage.get('completion_tokens', '?')}")
        
        # Check if needle was found
        found_codename = "midnight falcon" in answer.lower()
        found_budget = "3.7" in answer
        found_date = "july" in answer.lower() and "2019" in answer
        
        print(f"\nRETRIEVAL CHECK:")
        print(f"  Codename 'Midnight Falcon': {'PASS' if found_codename else 'FAIL'}")
        print(f"  Budget '$3.7 billion':       {'PASS' if found_budget else 'FAIL'}")
        print(f"  Date 'July 14, 2019':        {'PASS' if found_date else 'FAIL'}")
        print(f"  Overall: {'PASS' if (found_codename and found_budget and found_date) else 'FAIL'}")
        
    except requests.exceptions.Timeout:
        print(f"TIMEOUT after {time.time()-start:.0f}s")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 500000
    depth_pct = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    print(f"=== Extreme Context NIAH Test ===")
    print(f"Target: ~{target:,} tokens, needle at {depth_pct}% depth")
    
    if not wait_for_server():
        sys.exit(1)
    
    haystack = build_haystack(target)
    prompt = inject_needle(haystack, NEEDLE, depth_pct)
    test_retrieval(prompt)
