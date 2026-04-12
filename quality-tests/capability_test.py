#!/usr/bin/env python3
"""
TurboQuant Capability Comparison Test Suite

Tests tool calling, instruction following, and reasoning across KV cache types
(f16, q8_0, turbo3) to isolate whether turbo compression degrades capabilities.

Prerequisites:
  Start llama-server first:
    ./build/bin/llama-server -m $MODEL -ctk turbo3 -ctv turbo3 -fa on -ngl 99 \
      -c 40000 --port 8090 --jinja

Usage:
    python3 capability_test.py --port 8090 --label turbo3 --context-depth 2048
"""

import argparse
import copy
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime

try:
    import requests
except ImportError:
    print("ERROR: pip install requests")
    sys.exit(1)

# ============================================================================
# Tool Definitions
# ============================================================================

TOOL_WEATHER = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City and country, e.g. 'Tokyo, Japan'"}
            },
            "required": ["location"],
        },
    },
}

TOOL_CALCULATE = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Evaluate a mathematical expression",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "The math expression to evaluate"}
            },
            "required": ["expression"],
        },
    },
}

TOOL_PYTHON = {
    "type": "function",
    "function": {
        "name": "python",
        "description": "Execute Python code in an interpreter and return the output",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"}
            },
            "required": ["code"],
        },
    },
}

TOOL_WEB_SEARCH = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"],
        },
    },
}

TOOL_SEND_EMAIL = {
    "type": "function",
    "function": {
        "name": "send_email",
        "description": "Send an email message",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient email address"},
                "subject": {"type": "string", "description": "Email subject"},
                "body": {"type": "string", "description": "Email body text"},
            },
            "required": ["to", "subject", "body"],
        },
    },
}

TOOL_REMINDER = {
    "type": "function",
    "function": {
        "name": "create_reminder",
        "description": "Create a timed reminder",
        "parameters": {
            "type": "object",
            "properties": {
                "time": {"type": "string", "description": "When to remind, e.g. 'tomorrow 9am'"},
                "message": {"type": "string", "description": "Reminder message"},
            },
            "required": ["time", "message"],
        },
    },
}

TOOL_CURRENCY = {
    "type": "function",
    "function": {
        "name": "convert_currency",
        "description": "Convert an amount from one currency to another",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {"type": "number", "description": "Amount to convert"},
                "from_currency": {"type": "string", "description": "Source currency code"},
                "to_currency": {"type": "string", "description": "Target currency code"},
            },
            "required": ["amount", "from_currency", "to_currency"],
        },
    },
}

TOOL_STOCK = {
    "type": "function",
    "function": {
        "name": "get_stock_price",
        "description": "Get the current stock price for a ticker symbol",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Stock ticker symbol, e.g. 'AAPL'"}
            },
            "required": ["symbol"],
        },
    },
}

TOOL_READ_FILE = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read the contents of a file at the given path",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to read"}
            },
            "required": ["path"],
        },
    },
}

# ============================================================================
# Test Definitions
# ============================================================================

TOOL_CALLING_TESTS = [
    {
        "id": "TC01",
        "prompt": "What is the current weather in Tokyo, Japan?",
        "tools": [TOOL_WEATHER],
        "tool_choice": "required",
        "expected_fn": "get_current_weather",
        "arg_check": lambda a: "location" in a and "tokyo" in a["location"].lower(),
    },
    {
        "id": "TC02",
        "prompt": "Look up the weather in Paris, France",
        "tools": [TOOL_WEATHER],
        "tool_choice": "required",
        "expected_fn": "get_current_weather",
        "arg_check": lambda a: "location" in a and "paris" in a["location"].lower(),
    },
    {
        "id": "TC03",
        "prompt": "Calculate the square root of 144",
        "tools": [TOOL_CALCULATE],
        "tool_choice": "required",
        "expected_fn": "calculate",
        "arg_check": lambda a: "expression" in a and ("144" in a["expression"] or "sqrt" in a["expression"].lower()),
    },
    {
        "id": "TC04",
        "prompt": "Run this python code: print('hello world')",
        "tools": [TOOL_PYTHON],
        "tool_choice": "required",
        "expected_fn": "python",
        "arg_check": lambda a: "code" in a and ("hello" in a["code"].lower() or "print" in a["code"].lower()),
    },
    {
        "id": "TC05",
        "prompt": "Search for recent news about quantum computing",
        "tools": [TOOL_WEB_SEARCH, TOOL_WEATHER],
        "tool_choice": "required",
        "expected_fn": "web_search",
        "arg_check": lambda a: "query" in a and "quantum" in a["query"].lower(),
    },
    {
        "id": "TC06",
        "prompt": "Send an email to bob@example.com with subject 'Meeting' and body 'See you at 3pm'",
        "tools": [TOOL_SEND_EMAIL],
        "tool_choice": "required",
        "expected_fn": "send_email",
        "arg_check": lambda a: a.get("to", "") == "bob@example.com" and "meeting" in a.get("subject", "").lower(),
    },
    {
        "id": "TC07",
        "prompt": "Create a reminder for tomorrow at 9am to buy groceries",
        "tools": [TOOL_REMINDER],
        "tool_choice": "required",
        "expected_fn": "create_reminder",
        "arg_check": lambda a: "time" in a and "message" in a,
    },
    {
        "id": "TC08",
        "prompt": "Convert 100 USD to EUR",
        "tools": [TOOL_CURRENCY],
        "tool_choice": "required",
        "expected_fn": "convert_currency",
        "arg_check": lambda a: (
            a.get("amount") == 100 or a.get("amount") == 100.0 or str(a.get("amount", "")) == "100"
        ) and "usd" in a.get("from_currency", "").lower() and "eur" in a.get("to_currency", "").lower(),
    },
    {
        "id": "TC09",
        "prompt": "Get the stock price of AAPL",
        "tools": [TOOL_STOCK, TOOL_WEATHER],
        "tool_choice": "required",
        "expected_fn": "get_stock_price",
        "arg_check": lambda a: a.get("symbol", "").upper() == "AAPL",
    },
    {
        "id": "TC10",
        "prompt": "Read the file at /tmp/data.csv and summarize it",
        "tools": [TOOL_READ_FILE, TOOL_CALCULATE],
        "tool_choice": "required",
        "expected_fn": "read_file",
        "arg_check": lambda a: a.get("path", "") == "/tmp/data.csv",
    },
    {
        "id": "TC11",
        "prompt": "Just say hello to me, no tools needed.",
        "tools": [TOOL_WEATHER],
        "tool_choice": "auto",
        "expected_fn": None,  # Should NOT call a tool
        "arg_check": lambda a: True,
    },
    {
        "id": "TC12",
        "prompt": "What is the current weather in London and also in Berlin?",
        "tools": [TOOL_WEATHER],
        "tool_choice": "required",
        "expected_fn": "get_current_weather",
        "arg_check": lambda a: "location" in a and ("london" in a["location"].lower() or "berlin" in a["location"].lower()),
    },
]

INSTRUCTION_TESTS = [
    {
        "id": "IF01",
        "prompt": "List exactly 5 prime numbers between 10 and 50. Output ONLY the numbers, one per line, no other text.",
        "check": lambda t: _check_if01(t),
    },
    {
        "id": "IF02",
        "prompt": 'Write a JSON object with keys "name", "age", "city". Use values: Alice, 30, Boston. Output ONLY valid JSON, nothing else.',
        "check": lambda t: _check_if02(t),
    },
    {
        "id": "IF03",
        "prompt": "Respond with ONLY the word 'YES' or 'NO': Is 17 a prime number?",
        "check": lambda t: _check_if03(t),
    },
    {
        "id": "IF04",
        "prompt": "Write a Python function called 'add_two' that takes two integers and returns their sum. Output ONLY the code, no explanation.",
        "check": lambda t: _check_if04(t),
    },
    {
        "id": "IF05",
        "prompt": "Sort these numbers in ascending order, comma-separated: 42, 7, 19, 3, 88, 15. Output ONLY the sorted list.",
        "check": lambda t: _check_if05(t),
    },
    {
        "id": "IF06",
        "prompt": "Translate 'Good morning' to French, German, and Spanish. Format as exactly 3 lines: LANGUAGE: TRANSLATION",
        "check": lambda t: _check_if06(t),
    },
    {
        "id": "IF07",
        "prompt": "Output the path '/home/user/projects/myapp/src/main.py' followed by a colon and the text 'entry point'. Output ONLY that, nothing else.",
        "check": lambda t: _check_if07(t),
    },
    {
        "id": "IF08",
        "prompt": "Count the vowels in the word 'extraordinarily'. Respond with ONLY the number.",
        "check": lambda t: _check_if08(t),
    },
    {
        "id": "IF09",
        "prompt": "Output a markdown table with 3 columns: Name, Age, Role. Include exactly 2 data rows: (Alice, 30, Engineer) and (Bob, 25, Designer).",
        "check": lambda t: _check_if09(t),
    },
    {
        "id": "IF10",
        "prompt": "Write exactly 3 bullet points about why testing software is important. Each bullet must start with '- '.",
        "check": lambda t: _check_if10(t),
    },
    {
        "id": "IF11",
        "prompt": "Reverse the string 'hello world' and output ONLY the reversed string.",
        "check": lambda t: _check_if11(t),
    },
    {
        "id": "IF12",
        "prompt": "Output the numbers 1 through 10, each on its own line. No other text.",
        "check": lambda t: _check_if12(t),
    },
]

REASONING_TESTS = [
    {"id": "RE01", "prompt": "What is 347 + 258? Respond with ONLY the number.", "answer": "605"},
    {"id": "RE02", "prompt": "What is 23 * 17? Respond with ONLY the number.", "answer": "391"},
    {"id": "RE03", "prompt": "If a shirt costs $25 and is on sale for 20% off, what is the sale price in dollars? Respond with ONLY the number.", "answer": "20"},
    {"id": "RE04", "prompt": "How many prime numbers are there between 1 and 20? Respond with ONLY the number.", "answer": "8"},
    {"id": "RE05", "prompt": "What is the next number in the sequence: 2, 6, 12, 20, 30, ? Respond with ONLY the number.", "answer": "42"},
    {"id": "RE06", "prompt": "A train travels 120 km in 2 hours. What is its speed in km/h? Respond with ONLY the number.", "answer": "60"},
    {"id": "RE07", "prompt": "In Python, what does len([1, [2, 3], 4]) return? Respond with ONLY the number.", "answer": "3"},
    {"id": "RE08", "prompt": "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? Respond ONLY 'YES' or 'NO'.", "answer": "NO"},
    {"id": "RE09", "prompt": "What is the binary representation of the decimal number 13? Respond with ONLY the binary number.", "answer": "1101"},
    {"id": "RE10", "prompt": "A function f(x) = 2x + 3. What is f(f(2))? Respond with ONLY the number.", "answer": "17"},
    {"id": "RE11", "prompt": "How many letter 'r' are in the word 'strawberry'? Respond with ONLY the number.", "answer": "3"},
    {"id": "RE12", "prompt": "What is the GCD of 48 and 36? Respond with ONLY the number.", "answer": "12"},
]

# ============================================================================
# Instruction Following Scorers
# ============================================================================

PRIMES_10_50 = {11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47}

def _check_if01(text):
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    fmt = len(lines) == 5
    content = fmt and all(l.isdigit() and int(l) in PRIMES_10_50 for l in lines)
    return {"format_correct": fmt, "content_correct": content}

def _check_if02(text):
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        obj = json.loads(text)
        fmt = isinstance(obj, dict) and set(obj.keys()) >= {"name", "age", "city"}
        content = fmt and obj["name"] == "Alice" and obj["age"] == 30 and obj["city"] == "Boston"
        return {"format_correct": fmt, "content_correct": content}
    except (json.JSONDecodeError, KeyError):
        return {"format_correct": False, "content_correct": False}

def _check_if03(text):
    cleaned = text.strip().rstrip(".").strip().upper()
    fmt = cleaned in ("YES", "NO")
    content = cleaned == "YES"
    return {"format_correct": fmt, "content_correct": content}

def _check_if04(text):
    text = text.strip()
    # Strip markdown code fences
    if "```" in text:
        blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
        if blocks:
            text = blocks[0].strip()
    has_def = "def add_two" in text
    has_return = "return" in text
    no_prose = not any(text.strip().startswith(w) for w in ["Here", "This", "The ", "Below", "Sure"])
    fmt = has_def and no_prose
    content = has_def and has_return
    return {"format_correct": fmt, "content_correct": content}

def _check_if05(text):
    cleaned = text.strip().rstrip(".")
    nums = re.findall(r"\d+", cleaned)
    fmt = len(nums) == 6
    content = nums == ["3", "7", "15", "19", "42", "88"]
    return {"format_correct": fmt, "content_correct": content}

def _check_if06(text):
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    fmt = len(lines) == 3 and all(":" in l for l in lines)
    joined = text.lower()
    content = (
        ("bonjour" in joined or "bon matin" in joined)
        and "guten morgen" in joined
        and ("buenos" in joined or "buen" in joined)
    )
    return {"format_correct": fmt, "content_correct": content}

def _check_if07(text):
    cleaned = text.strip()
    fmt = "/home/user/projects/myapp/src/main.py" in cleaned
    content = fmt and "entry point" in cleaned.lower()
    return {"format_correct": fmt, "content_correct": content}

def _check_if08(text):
    cleaned = text.strip().rstrip(".")
    # 'extraordinarily' vowels: e-x-t-r-a-o-r-d-i-n-a-r-i-l-y -> e,a,o,i,a,i = 6
    nums = re.findall(r"\d+", cleaned)
    fmt = len(nums) >= 1
    content = "7" in nums or "6" in nums  # Accept 6 or 7 (y is debatable)
    return {"format_correct": fmt, "content_correct": content}

def _check_if09(text):
    fmt = "|" in text and "Alice" in text and "Bob" in text
    content = (
        fmt
        and "30" in text
        and "25" in text
        and "Engineer" in text
        and "Designer" in text
    )
    return {"format_correct": fmt, "content_correct": content}

def _check_if10(text):
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    bullet_lines = [l for l in lines if l.startswith("- ")]
    fmt = len(bullet_lines) == 3
    content = fmt  # Content is subjective, format is what matters
    return {"format_correct": fmt, "content_correct": content}

def _check_if11(text):
    cleaned = text.strip()
    fmt = len(cleaned) > 0  # Non-empty response
    content = cleaned == "dlrow olleh"
    return {"format_correct": fmt, "content_correct": content}

def _check_if12(text):
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    fmt = len(lines) == 10
    content = fmt and lines == [str(i) for i in range(1, 11)]
    return {"format_correct": fmt, "content_correct": content}


# ============================================================================
# Filler text for context depth padding
# ============================================================================

FILLER_SENTENCES = [
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "A comprehensive analysis of market trends reveals interesting patterns.",
    "Scientists have discovered new species in the deep ocean trenches.",
    "The architecture of modern buildings reflects cultural values and aspirations.",
    "Renewable energy sources are becoming increasingly cost-effective globally.",
    "Historical records indicate significant climate variations over centuries.",
    "Mathematical models can predict complex system behaviors with accuracy.",
    "The development of artificial intelligence raises important ethical questions.",
    "Musical compositions often reflect the emotional state of their creators.",
    "Agricultural innovations have dramatically increased food production worldwide.",
    "The study of linguistics reveals deep connections between human cultures.",
    "Advances in materials science enable new engineering possibilities daily.",
    "Economic theories attempt to explain the behavior of complex markets.",
    "Philosophical debates about consciousness continue to challenge researchers.",
    "The exploration of space has yielded numerous technological breakthroughs.",
    "Environmental conservation requires coordinated global efforts and policies.",
]


def generate_filler(target_tokens):
    """Generate filler text of approximately target_tokens length."""
    # Rough estimate: 1 token ~ 4 chars
    target_chars = target_tokens * 4
    chunks = []
    total = 0
    while total < target_chars:
        s = random.choice(FILLER_SENTENCES)
        chunks.append(s)
        total += len(s) + 1
    return " ".join(chunks)


def pad_messages(messages, target_depth):
    """Pad system message with filler to reach target context depth."""
    if target_depth <= 2048:
        return messages
    padded = copy.deepcopy(messages)
    filler = generate_filler(target_depth - 1024)  # Reserve tokens for actual prompt
    preamble = f"[Context document follows]\n\n{filler}\n\n[End of context document]\n\n"
    if padded and padded[0]["role"] == "system":
        padded[0]["content"] = preamble + padded[0]["content"]
    else:
        padded.insert(0, {"role": "system", "content": preamble + "You are a helpful assistant."})
    return padded


# ============================================================================
# Server Communication
# ============================================================================

def check_server(base_url):
    """Check if server is healthy."""
    try:
        r = requests.get(f"{base_url}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def query_chat(base_url, messages, tools=None, tool_choice=None, max_tokens=4096,
               temperature=None, top_p=None, top_k=None, presence_penalty=None):
    """Send chat completion request. Returns full response JSON.
    Uses Qwen 3.5 recommended defaults if not overridden."""
    payload = {
        "model": "default",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature if temperature is not None else 0.6,
        "top_k": top_k if top_k is not None else 20,
        "top_p": top_p if top_p is not None else 0.95,
        "seed": 42,
    }
    if presence_penalty is not None:
        payload["presence_penalty"] = presence_penalty
    if tools:
        payload["tools"] = tools
        payload["parallel_tool_calls"] = False
    if tool_choice:
        payload["tool_choice"] = tool_choice

    try:
        start = time.time()
        resp = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=300)
        elapsed = time.time() - start
        resp.raise_for_status()
        return resp.json(), elapsed
    except requests.exceptions.ConnectionError:
        return {"error": "server not running"}, 0
    except requests.exceptions.Timeout:
        return {"error": "timeout"}, 300
    except Exception as e:
        return {"error": str(e)}, 0


# ============================================================================
# Scoring Functions
# ============================================================================

def score_tool_call(test, response_json):
    """Score a tool calling test. Returns dict with json_valid, function_correct, args_correct."""
    if "error" in response_json:
        return {"json_valid": False, "function_correct": False, "args_correct": False}

    choice = response_json.get("choices", [{}])[0]
    message = choice.get("message", {})
    tool_calls = message.get("tool_calls")
    content = message.get("content", "")

    # TC11: should NOT call a tool
    if test["expected_fn"] is None:
        no_call = not tool_calls or len(tool_calls) == 0
        has_content = bool(content and content.strip())
        return {"json_valid": no_call, "function_correct": no_call and has_content, "args_correct": no_call}

    # All other tests: must have at least one tool call
    if not tool_calls or len(tool_calls) == 0:
        return {"json_valid": False, "function_correct": False, "args_correct": False}

    tc = tool_calls[0]
    fn_name = tc.get("function", {}).get("name", "")
    raw_args = tc.get("function", {}).get("arguments", "")

    # 1. JSON validity
    json_valid = False
    parsed_args = {}
    if isinstance(raw_args, dict):
        json_valid = True
        parsed_args = raw_args
    elif isinstance(raw_args, str):
        try:
            parsed_args = json.loads(raw_args)
            json_valid = True
        except (json.JSONDecodeError, TypeError):
            pass

    # 2. Function name
    function_correct = fn_name == test["expected_fn"]

    # 3. Arguments
    args_correct = False
    if json_valid:
        try:
            args_correct = test["arg_check"](parsed_args)
        except Exception:
            args_correct = False

    return {"json_valid": json_valid, "function_correct": function_correct, "args_correct": args_correct}


def score_instruction(test, response_text):
    """Score an instruction following test."""
    if not response_text:
        return {"format_correct": False, "content_correct": False}
    return test["check"](response_text)


def score_reasoning(test, response_text):
    """Score a reasoning test. Returns bool."""
    if not response_text:
        return False
    # Strip markdown formatting, punctuation, whitespace, currency
    cleaned = response_text.strip()
    cleaned = re.sub(r"[*_`#$]", "", cleaned)  # Remove markdown + currency
    cleaned = cleaned.strip().rstrip(".").strip()
    cleaned = re.sub(r"\.00$", "", cleaned)  # Strip trailing .00
    expected = test["answer"]
    # Exact match
    if cleaned == expected:
        return True
    # For YES/NO answers, be flexible
    if expected in ("YES", "NO"):
        return expected in cleaned.upper().split()
    # Extract numbers and check
    numbers = re.findall(r"-?\d+\.?\d*", cleaned)
    if expected in numbers:
        return True
    return False


# ============================================================================
# Test Runners
# ============================================================================

def run_tool_calling_tests(base_url, context_depth):
    """Run all tool calling tests. Returns list of result dicts."""
    results = []
    for test in TOOL_CALLING_TESTS:
        messages = [
            {"role": "system", "content": "You are a helpful assistant with access to tools. Use them when appropriate."},
            {"role": "user", "content": test["prompt"]},
        ]
        messages = pad_messages(messages, context_depth)

        response, elapsed = query_chat(
            base_url, messages,
            tools=test["tools"],
            tool_choice=test.get("tool_choice"),
            presence_penalty=1.5,
        )

        scores = score_tool_call(test, response)
        points = sum(1 for v in scores.values() if v)

        # Extract response preview
        preview = ""
        if "error" not in response:
            choice = response.get("choices", [{}])[0]
            msg = choice.get("message", {})
            tc = msg.get("tool_calls", [])
            if tc:
                preview = f"tool_call: {tc[0].get('function', {}).get('name', '?')}({tc[0].get('function', {}).get('arguments', '')})"
            elif msg.get("content"):
                preview = msg["content"][:100]

        results.append({
            "test_id": test["id"],
            "category": "tool_calling",
            "prompt_summary": test["prompt"][:60],
            "scores": scores,
            "points": points,
            "max_points": 3,
            "response_preview": preview[:200],
            "elapsed": round(elapsed, 2),
        })
        status = "PASS" if points == 3 else "PARTIAL" if points > 0 else "FAIL"
        print(f"  {test['id']}: {status} ({points}/3) [{elapsed:.1f}s] {preview[:60]}")

    return results


def run_instruction_tests(base_url, context_depth):
    """Run all instruction following tests. Returns list of result dicts."""
    results = []
    for test in INSTRUCTION_TESTS:
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Follow instructions precisely."},
            {"role": "user", "content": test["prompt"]},
        ]
        messages = pad_messages(messages, context_depth)

        response, elapsed = query_chat(base_url, messages, presence_penalty=1.5)

        content = ""
        if "error" not in response:
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")

        scores = score_instruction(test, content)
        points = sum(1 for v in scores.values() if v)

        results.append({
            "test_id": test["id"],
            "category": "instruction_following",
            "prompt_summary": test["prompt"][:60],
            "scores": scores,
            "points": points,
            "max_points": 2,
            "response_preview": content[:200] if content else "(empty)",
            "elapsed": round(elapsed, 2),
        })
        status = "PASS" if points == 2 else "PARTIAL" if points > 0 else "FAIL"
        print(f"  {test['id']}: {status} ({points}/2) [{elapsed:.1f}s] {content[:60] if content else '(empty)'}")

    return results


def run_reasoning_tests(base_url, context_depth):
    """Run all reasoning tests. Returns list of result dicts."""
    results = []
    for test in REASONING_TESTS:
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Be concise and precise."},
            {"role": "user", "content": test["prompt"]},
        ]
        messages = pad_messages(messages, context_depth)

        response, elapsed = query_chat(base_url, messages, presence_penalty=1.5)

        content = ""
        if "error" not in response:
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")

        correct = score_reasoning(test, content)

        results.append({
            "test_id": test["id"],
            "category": "reasoning",
            "prompt_summary": test["prompt"][:60],
            "scores": {"correct": correct},
            "points": 1 if correct else 0,
            "max_points": 1,
            "expected": test["answer"],
            "response_preview": content[:200] if content else "(empty)",
            "elapsed": round(elapsed, 2),
        })
        status = "PASS" if correct else "FAIL"
        print(f"  {test['id']}: {status} (expected={test['answer']}) [{elapsed:.1f}s] -> {content[:40] if content else '(empty)'}")

    return results


# ============================================================================
# Main
# ============================================================================

# ============================================================================
# Agentic Coding Tests — Real execution with multi-turn tool loop
# ============================================================================

AGENTIC_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file at the given path. Creates parent directories if needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute file path to write to"},
                    "content": {"type": "string", "description": "File content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command and return stdout/stderr. Use for compiling, running, testing code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file at the given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute file path to read"},
                },
                "required": ["path"],
            },
        },
    },
]

AGENTIC_CODING_TESTS = [
    {
        "id": "AC01",
        "prompt": (
            "Write a C++ function called `get_or_create` that is a thread-safe cache lookup. "
            "It should take a `std::unordered_map<std::string, int>&`, a `std::mutex&`, "
            "a `const std::string& key`, and a `std::function<int()> factory`. "
            "If the key exists, return the value. If not, call factory(), insert it, and return it. "
            "Write it to a .cpp file with a main() that tests it, compile it with g++, and run it to verify."
        ),
        "check_compile": True,
        "check_output": lambda out: any(w in out.lower() for w in ["pass", "ok", "success", "correct", "found", "got"]),
        "check_code": lambda code: "mutex" in code and "unordered_map" in code and "get_or_create" in code,
    },
    {
        "id": "AC02",
        "prompt": (
            "Write a C++ function `parse_csv_line` that parses a single CSV line, correctly handling "
            "quoted fields with embedded commas and escaped quotes. For example, "
            '`parse_csv_line(\'hello,"world,test","say \\"hi\\""\')` should return '
            '`["hello", "world,test", "say \\"hi\\""]`. '
            "Write it to a .cpp file with test cases in main(), compile with g++, and run to verify all tests pass."
        ),
        "check_compile": True,
        "check_output": lambda out: any(w in out.lower() for w in ["pass", "ok", "success", "correct", "all"]),
        "check_code": lambda code: "parse_csv" in code and "main" in code,
    },
    {
        "id": "AC03",
        "prompt": (
            "Here is a buggy C++ program. Find the bug, fix it, write the fixed version to a file, "
            "compile and run it.\n\n"
            "```cpp\n"
            "#include <iostream>\n"
            "#include <vector>\n"
            "#include <numeric>\n\n"
            "double average(const std::vector<int>& nums) {\n"
            "    int sum = std::accumulate(nums.begin(), nums.end(), 0);\n"
            "    return sum / nums.size();\n"
            "}\n\n"
            "int main() {\n"
            "    std::vector<int> v = {1, 2, 3, 4};\n"
            "    double avg = average(v);\n"
            "    if (avg == 2.5) std::cout << \"PASS: avg=\" << avg << std::endl;\n"
            "    else std::cout << \"FAIL: avg=\" << avg << \" expected 2.5\" << std::endl;\n"
            "    return 0;\n"
            "}\n"
            "```"
        ),
        "check_compile": True,
        "check_output": lambda out: "pass" in out.lower() and "2.5" in out,
        "check_code": lambda code: "double" in code or "static_cast" in code or "1.0" in code or ".0" in code,
    },
    {
        "id": "AC04",
        "prompt": (
            "Write a C++ program that implements a simple stack using a linked list (no std::stack). "
            "It should have push(), pop(), top(), and is_empty() methods. "
            "Include a main() with tests that push 3 values, verify top(), pop all, "
            "and verify is_empty(). Print PASS/FAIL for each check. "
            "Write to a file, compile with g++ -std=c++17, and run it."
        ),
        "check_compile": True,
        "check_output": lambda out: out.lower().count("pass") >= 3,
        "check_code": lambda code: "push" in code and "pop" in code and "struct" in code.lower() or "class" in code.lower(),
    },
    {
        "id": "AC05",
        "prompt": (
            "Write a C++ function `fizzbuzz(int n)` that returns a `std::vector<std::string>` "
            "for numbers 1 to n with the classic FizzBuzz rules. "
            "Write a main() that calls fizzbuzz(15) and verifies: "
            "element[2]==\"Fizz\", element[4]==\"Buzz\", element[14]==\"FizzBuzz\", element[0]==\"1\". "
            "Print PASS/FAIL for each check. Write to a file, compile, and run."
        ),
        "check_compile": True,
        "check_output": lambda out: out.lower().count("pass") >= 3,
        "check_code": lambda code: "fizzbuzz" in code.lower() and "vector" in code,
    },
]


def execute_tool_call(fn_name, args, workdir):
    """Actually execute a tool call. Returns result string."""
    if fn_name == "write_file":
        path = args.get("path", "")
        content = args.get("content", "")
        # Security: force all paths into workdir
        if not path:
            return "Error: no path provided"
        basename = os.path.basename(path)
        if not basename:
            basename = "code.cpp"
        real_path = os.path.join(workdir, basename)
        try:
            with open(real_path, "w") as f:
                f.write(content)
            return f"File written to {real_path} ({len(content)} bytes)"
        except Exception as e:
            return f"Error writing file: {e}"

    elif fn_name == "run_command":
        command = args.get("command", "")
        if not command:
            return "Error: no command provided"
        # Rewrite paths to use workdir
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True,
                timeout=30, cwd=workdir,
            )
            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                output += result.stderr
            if not output:
                output = f"(exit code {result.returncode})"
            return output[:2000]
        except subprocess.TimeoutExpired:
            return "Error: command timed out (30s)"
        except Exception as e:
            return f"Error: {e}"

    elif fn_name == "read_file":
        path = args.get("path", "")
        basename = os.path.basename(path) if path else ""
        real_path = os.path.join(workdir, basename) if basename else ""
        if real_path and os.path.exists(real_path):
            try:
                with open(real_path) as f:
                    return f.read()[:4000]
            except Exception as e:
                return f"Error reading: {e}"
        return f"Error: file not found: {path}"

    return f"Error: unknown tool '{fn_name}'"


def run_agentic_loop(base_url, messages, tools, workdir, max_turns=10):
    """Run a multi-turn agentic loop. Returns (messages, tool_log).
    Uses Qwen recommended coding params: temp=1.0, presence_penalty=0.0."""
    tool_log = []

    for turn in range(max_turns):
        response, elapsed = query_chat(base_url, messages, tools=tools, tool_choice="auto",
                                       temperature=1.0, presence_penalty=0.0)

        if "error" in response:
            tool_log.append({"turn": turn, "error": response["error"]})
            break

        choice = response.get("choices", [{}])[0]
        message = choice.get("message", {})
        finish_reason = choice.get("finish_reason", "")

        # Add assistant message to conversation
        assistant_msg = {"role": "assistant"}
        if message.get("content"):
            assistant_msg["content"] = message["content"]
        if message.get("tool_calls"):
            assistant_msg["tool_calls"] = message["tool_calls"]
        if message.get("reasoning_content"):
            assistant_msg["reasoning_content"] = message["reasoning_content"]
        messages.append(assistant_msg)

        tool_calls = message.get("tool_calls")
        if not tool_calls:
            # Model is done (no more tool calls)
            tool_log.append({"turn": turn, "action": "done", "content": (message.get("content") or "")[:200]})
            break

        # Execute each tool call and feed results back
        for tc in tool_calls:
            fn_name = tc.get("function", {}).get("name", "")
            raw_args = tc.get("function", {}).get("arguments", "")
            tc_id = tc.get("id", f"call_{turn}")

            parsed_args = {}
            if isinstance(raw_args, dict):
                parsed_args = raw_args
            elif isinstance(raw_args, str):
                try:
                    parsed_args = json.loads(raw_args)
                except (json.JSONDecodeError, TypeError):
                    pass

            result = execute_tool_call(fn_name, parsed_args, workdir)

            tool_log.append({
                "turn": turn,
                "action": fn_name,
                "args_summary": {k: str(v)[:300] for k, v in parsed_args.items()},
                "result_preview": result[:200],
            })

            # Add tool result to conversation
            messages.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": result,
            })

    return messages, tool_log


def score_agentic_test(test, workdir, tool_log):
    """Score an agentic coding test. Returns dict of scores."""
    scores = {
        "wrote_file": False,
        "valid_path": False,
        "code_quality": False,
        "compiled": False,
        "ran_compile": False,
        "ran_execute": False,
        "correct_output": False,
    }

    # Check tool_log for actions
    actions = [entry.get("action", "") for entry in tool_log]

    # Did it write a file?
    scores["wrote_file"] = "write_file" in actions

    # Did it try to compile?
    scores["ran_compile"] = any(
        entry.get("action") == "run_command" and any(
            comp in entry.get("args_summary", {}).get("command", "")
            for comp in ["g++", "gcc", "clang", "c++", "make", "cmake"]
        )
        for entry in tool_log
    )

    # Did it try to run the binary? Check for any execution pattern
    scores["ran_execute"] = any(
        entry.get("action") == "run_command" and any(
            run in entry.get("args_summary", {}).get("command", "")
            for run in ["./", "a.out", "&& /tmp", "&& ./", "| /tmp"]
        )
        for entry in tool_log
    )
    # Also count chained compile+run as ran_execute (g++ -o X file && X)
    if not scores["ran_execute"]:
        scores["ran_execute"] = any(
            entry.get("action") == "run_command" and "&&" in entry.get("args_summary", {}).get("command", "")
            and any(comp in entry.get("args_summary", {}).get("command", "") for comp in ["g++", "gcc"])
            for entry in tool_log
        )

    # Check written files
    cpp_files = [f for f in os.listdir(workdir) if f.endswith((".cpp", ".cc", ".cxx", ".c"))]
    if cpp_files:
        scores["valid_path"] = True
        # Read the code and check quality
        code_path = os.path.join(workdir, cpp_files[0])
        try:
            with open(code_path) as f:
                code = f.read()
            if test.get("check_code"):
                scores["code_quality"] = test["check_code"](code)
        except Exception:
            pass

    # Check if compilation succeeded — look for binary in workdir OR successful compile in tool_log
    binaries = [f for f in os.listdir(workdir)
                if os.access(os.path.join(workdir, f), os.X_OK) and not f.endswith((".cpp", ".cc", ".py", ".sh"))]
    if binaries:
        scores["compiled"] = True
    else:
        # Check tool_log: a compile command that produced output without "error" in result
        for entry in tool_log:
            cmd = entry.get("args_summary", {}).get("command", "")
            result = entry.get("result_preview", "")
            if entry.get("action") == "run_command" and any(c in cmd for c in ["g++", "gcc"]):
                if "error" not in result.lower() or "PASS" in result or "pass" in result:
                    scores["compiled"] = True
                    break

    # Check output from run_command results
    if test.get("check_output"):
        for entry in tool_log:
            result = entry.get("result_preview", "")
            if result and test["check_output"](result):
                scores["correct_output"] = True
                break

    return scores


def run_agentic_coding_tests(base_url, context_depth):
    """Run agentic coding tests with real execution. Returns list of result dicts."""
    results = []

    for test in AGENTIC_CODING_TESTS:
        # Create temp workdir for this test
        workdir = tempfile.mkdtemp(prefix=f"turbotest_{test['id']}_")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a skilled C++ developer. You have access to tools to write files, "
                    "run shell commands, and read files. When asked to write code:\n"
                    "1. Write the code to a .cpp file in /tmp/\n"
                    "2. Compile it with g++ -std=c++17\n"
                    "3. Run the binary to verify it works\n"
                    "Always compile and test your code. Do not skip verification."
                ),
            },
            {"role": "user", "content": test["prompt"]},
        ]
        messages = pad_messages(messages, context_depth)

        start = time.time()
        messages, tool_log = run_agentic_loop(base_url, messages, AGENTIC_TOOLS, workdir)
        elapsed = time.time() - start

        scores = score_agentic_test(test, workdir, tool_log)
        points = sum(1 for v in scores.values() if v)
        max_points = len(scores)

        # Summarize actions
        action_seq = " -> ".join(e.get("action", "?") for e in tool_log if e.get("action"))

        results.append({
            "test_id": test["id"],
            "category": "agentic_coding",
            "prompt_summary": test["prompt"][:80],
            "scores": scores,
            "points": points,
            "max_points": max_points,
            "action_sequence": action_seq,
            "tool_log": tool_log,
            "elapsed": round(elapsed, 2),
        })

        status = "PASS" if points >= 6 else "PARTIAL" if points >= 3 else "FAIL"
        print(f"  {test['id']}: {status} ({points}/{max_points}) [{elapsed:.1f}s]")
        print(f"         actions: {action_seq}")
        score_str = ", ".join(f"{k}={'Y' if v else 'N'}" for k, v in scores.items())
        print(f"         scores:  {score_str}")

        # Cleanup
        shutil.rmtree(workdir, ignore_errors=True)

    return results


def main():
    parser = argparse.ArgumentParser(description="TurboQuant Capability Comparison Test")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--label", default="turbo3", help="KV type label for output")
    parser.add_argument("--context-depth", type=int, default=2048, help="Context depth in tokens (2048 or 32768)")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--category", default="all", choices=["all", "tool_calling", "instruction", "reasoning", "agentic"])
    parser.add_argument("--reps", type=int, default=3, help="Repetitions per test (default: 3)")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    # Check server
    if not check_server(base_url):
        print(f"ERROR: Server not responding at {base_url}")
        sys.exit(1)
    print(f"Server OK at {base_url}")
    print(f"Label: {args.label} | Context depth: {args.context_depth} | Reps: {args.reps}")
    print(f"Sampling: temp=0.6 (general) / 1.0 (coding), top_p=0.95, top_k=20, seed=42")
    print("=" * 60)

    categories = {}
    reps = args.reps

    def run_with_reps(name, runner_fn, reps_count):
        """Run a category reps_count times, take best score per test."""
        all_runs = []
        for rep in range(reps_count):
            if reps_count > 1:
                print(f"\n--- {name} (rep {rep+1}/{reps_count}) ---")
            else:
                print(f"\n--- {name} ---")
            results = runner_fn(base_url, args.context_depth)
            all_runs.append(results)

        # Take best score per test across reps
        best_results = []
        for i in range(len(all_runs[0])):
            best = max(all_runs, key=lambda run: run[i]["points"])[i]
            # Add rep stats
            all_points = [run[i]["points"] for run in all_runs]
            best["avg_points"] = round(sum(all_points) / len(all_points), 2)
            best["all_points"] = all_points
            best_results.append(best)

        total_pts = sum(r["points"] for r in best_results)
        total_max = sum(r["max_points"] for r in best_results)
        avg_pts = sum(r["avg_points"] for r in best_results)
        acc = round(total_pts / total_max * 100, 1) if total_max else 0
        avg_acc = round(avg_pts / total_max * 100, 1) if total_max else 0

        if reps_count > 1:
            print(f"\n  {name} BEST: {total_pts}/{total_max} ({acc}%) | AVG: {avg_pts}/{total_max} ({avg_acc}%)")
        else:
            print(f"  {name}: {total_pts}/{total_max} ({acc}%)")

        return {
            "accuracy_best": acc,
            "accuracy_avg": avg_acc,
            "points_best": total_pts,
            "points_avg": avg_pts,
            "max_points": total_max,
            "reps": reps_count,
            "tests": best_results,
        }

    # Tool calling
    if args.category in ("all", "tool_calling"):
        categories["tool_calling"] = run_with_reps("Tool Calling", run_tool_calling_tests, reps)

    # Instruction following
    if args.category in ("all", "instruction"):
        categories["instruction_following"] = run_with_reps("Instruction Following", run_instruction_tests, reps)

    # Agentic coding (only at 2k — skip for 32k depth)
    if args.category in ("all", "agentic") and args.context_depth <= 4096:
        categories["agentic_coding"] = run_with_reps("Agentic Coding", run_agentic_coding_tests, reps)

    # Reasoning
    if args.category in ("all", "reasoning"):
        categories["reasoning"] = run_with_reps("Reasoning", run_reasoning_tests, reps)

    # Overall
    total_best = sum(c["points_best"] for c in categories.values())
    total_avg = sum(c["points_avg"] for c in categories.values())
    total_max = sum(c["max_points"] for c in categories.values())
    overall_best = round(total_best / total_max * 100, 1) if total_max else 0
    overall_avg = round(total_avg / total_max * 100, 1) if total_max else 0

    print("\n" + "=" * 60)
    if reps > 1:
        print(f"OVERALL BEST: {total_best}/{total_max} ({overall_best}%) | AVG: {total_avg}/{total_max} ({overall_avg}%)")
    else:
        print(f"OVERALL: {total_best}/{total_max} ({overall_best}%)")

    # Write JSON
    output = {
        "label": args.label,
        "context_depth": args.context_depth,
        "reps": reps,
        "timestamp": datetime.now().isoformat(),
        "sampling": {
            "general": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5},
            "coding": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 0.0},
            "seed": 42,
        },
        "categories": categories,
        "overall_accuracy_best": overall_best,
        "overall_accuracy_avg": overall_avg,
        "overall_points_best": total_best,
        "overall_points_avg": total_avg,
        "overall_max_points": total_max,
    }

    out_path = args.output or f"quality-tests/capability_{args.label}_ctx{args.context_depth}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
