#!/usr/bin/env python3
"""
Hermes-shaped battery — replicates the Hermes agent's request shape against our
stack: ~33K-char system prompt, 38-tool roster with chunky schemas, streamed,
no client sampling params. Two modes:

  --mode preverify   3x voxel-pagoda one-shot: expect ONE write_file tool call
                     with complete HTML (closed </html>). G3 pre-verification.
  --mode soak        10-turn scripted tool session (small calls + one big write
                     + plain turns), asserting: no stalls (>15s gap), no HTTP
                     errors, no finish=length, no empty tool arguments. G4.

Usage: hermes_shaped_battery.py --mode preverify [--port 8130] [--runs 3]
"""
import argparse, json, os, sys, time, urllib.request, urllib.error

KEY = open("/home/erol/.config/llama-tcq/api.key").read().strip()
OUT = "/home/erol/ai/turboquant/turboquant-kv-cache/quality-tests/niah_results"

PAGODA = ("Design and create a very creative, elaborate, and detailed voxel art scene of a pagoda "
          "in a beautiful garden with trees, including some cherry blossoms. Make the scene "
          "impressive and varied and use colorful voxels. Use whatever libraries to get this done "
          "but make sure I can paste it all into a single HTML file and open it in Chrome. "
          "Write the complete file with the write_file tool in ONE call.")


def build_system_prompt(target_chars=33000):
    base = (
        "You are Hermes, a capable autonomous agent operating on the user's workstation.\n"
        "You have access to tools for file operations, shell commands, task tracking and search.\n"
        "RULES:\n"
        "1. Prefer completing artifacts in single tool calls; do not fragment file writes.\n"
        "2. Always verify results after mutating operations.\n"
        "3. Track multi-step work with todo_update.\n"
        "4. Be precise with paths; never overwrite without reading first.\n"
        "5. When a task is ambiguous, choose the interpretation that maximizes user value.\n\n")
    policy = ("POLICY SECTION %d: operational guidance for scenario class %d. When handling this "
              "class, evaluate preconditions, select minimal tool sequence, execute, verify output, "
              "and record status. Escalate only on unrecoverable errors. Timeouts: generous. "
              "Retries: two, with backoff. Logging: structured, terse, complete.\n")
    s = base
    i = 0
    while len(s) < target_chars:
        s += policy % (i, i % 7)
        i += 1
    return s


def build_tools():
    tools = []
    for i in range(30):
        tools.append({"type": "function", "function": {
            "name": f"op_{i}", "description": f"Operation {i} over workspace resources",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "recursive": {"type": "boolean"},
                    "mode": {"type": "string", "enum": ["fast", "safe", "dry_run", "force"]},
                    "filters": {"type": "array", "items": {
                        "type": "object",
                        "properties": {
                            "field": {"type": "string"},
                            "op": {"type": "string", "enum": ["eq", "ne", "gt", "lt", "contains"]},
                            "value": {"type": "string"}},
                        "required": ["field", "op"]}},
                },
                "required": ["path"]}}})
    for name, props, req in [
        ("write_file", {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
        ("read_file", {"path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}}, ["path"]),
        ("edit_file", {"path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}}, ["path", "old", "new"]),
        ("ls", {"path": {"type": "string"}, "all": {"type": "boolean"}}, ["path"]),
        ("bash", {"command": {"type": "string"}, "timeout_s": {"type": "integer"}}, ["command"]),
        ("search", {"query": {"type": "string"}, "max_results": {"type": "integer"}}, ["query"]),
        ("memory_store", {"key": {"type": "string"}, "value": {"type": "string"}}, ["key", "value"]),
    ]:
        tools.append({"type": "function", "function": {"name": name, "description": name.replace("_", " "),
                      "parameters": {"type": "object", "properties": props, "required": req}}})
    tools.append({"type": "function", "function": {"name": "todo_update", "description": "Update todos",
        "parameters": {"type": "object", "properties": {"todos": {"type": "array", "items": {"type": "object",
            "properties": {"id": {"type": "string"}, "title": {"type": "string"},
                           "status": {"type": "string", "enum": ["pending", "in_progress", "completed", "cancelled"]},
                           "subtasks": {"type": "array", "items": {"type": "object", "properties": {
                               "title": {"type": "string"}, "done": {"type": "boolean"}}, "required": ["title"]}}},
            "required": ["id", "title", "status"]}}}, "required": ["todos"]}}})
    return tools  # 38 total


def stream_request(port, messages, tools, max_tokens):
    payload = {"model": "qwen3.6-27b", "stream": True, "max_tokens": max_tokens,
               "messages": messages, "tools": tools}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    t0 = time.time(); last = t0
    stats = dict(maxgap=0.0, comments=0, r_deltas=0, c_deltas=0, t_deltas=0, http=None)
    content = ""; tool_calls = {}; finish = None
    try:
        with urllib.request.urlopen(req, timeout=2400) as r:
            stats["http"] = r.status
            for raw in r:
                now = time.time(); stats["maxgap"] = max(stats["maxgap"], now - last); last = now
                if raw.startswith(b":"):
                    stats["comments"] += 1; continue
                if not raw.startswith(b"data: "):
                    continue
                body = raw[6:].strip()
                if body == b"[DONE]":
                    break
                try:
                    ch = json.loads(body)["choices"][0]
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                finish = ch.get("finish_reason") or finish
                d = ch.get("delta") or {}
                if d.get("reasoning") or d.get("reasoning_content"): stats["r_deltas"] += 1
                if d.get("content"):
                    content += d["content"]; stats["c_deltas"] += 1
                for tc in d.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    slot = tool_calls.setdefault(idx, {"name": "", "args": "", "id": tc.get("id", "")})
                    fn = tc.get("function") or {}
                    if fn.get("name"): slot["name"] = fn["name"]
                    if fn.get("arguments"): slot["args"] += fn["arguments"]; stats["t_deltas"] += 1
                    if tc.get("id"): slot["id"] = tc["id"]
    except urllib.error.HTTPError as e:
        stats["http"] = e.code
    stats["secs"] = round(time.time() - t0, 1)
    return content, [tool_calls[k] for k in sorted(tool_calls)], finish, stats


def check_turn(label, content, tcs, finish, stats, expect_tool=None):
    problems = []
    if stats["http"] != 200: problems.append(f"HTTP {stats['http']}")
    if stats["maxgap"] > 15: problems.append(f"STALL gap {stats['maxgap']:.1f}s")
    if finish == "length": problems.append("TRUNCATED (finish=length)")
    for tc in tcs:
        try:
            args = json.loads(tc["args"] or "{}")
            if args == {}: problems.append(f"{tc['name']}: EMPTY ARGS (collapse signature)")
        except json.JSONDecodeError: problems.append(f"{tc['name']}: UNPARSEABLE ARGS")
    if expect_tool and not any(t["name"] == expect_tool for t in tcs):
        problems.append(f"expected tool {expect_tool}, got {[t['name'] for t in tcs] or 'none'}")
    ok = not problems
    print(f"  [{label}] {'PASS' if ok else 'FAIL'} finish={finish} secs={stats['secs']} "
          f"maxgap={stats['maxgap']:.1f}s R/C/T={stats['r_deltas']}/{stats['c_deltas']}/{stats['t_deltas']} "
          f"hb/prog={stats['comments']} tools={[t['name'] for t in tcs]}"
          + (f"  PROBLEMS: {problems}" if problems else ""), flush=True)
    return ok, problems


def mode_preverify(port, runs):
    sysp, tools = build_system_prompt(), build_tools()
    passes = 0
    for run in range(1, runs + 1):
        print(f"[preverify run {run}/{runs}] sending pagoda prompt (33K sys + 38 tools)...", flush=True)
        msgs = [{"role": "system", "content": sysp}, {"role": "user", "content": PAGODA}]
        content, tcs, finish, stats = stream_request(port, msgs, tools, 131072)
        ok, problems = check_turn(f"run{run}", content, tcs, finish, stats, expect_tool="write_file")
        wf = next((t for t in tcs if t["name"] == "write_file"), None)
        if wf:
            try:
                html = json.loads(wf["args"]).get("content", "")
                closed = "</html>" in html
                big = len(html) >= 8000
                print(f"    artifact: {len(html):,} chars, closed={closed}, "
                      f"writes={sum(1 for t in tcs if t['name']=='write_file')}", flush=True)
                open(os.path.join(OUT, f"preverify_run{run}.html"), "w").write(html)
                if not closed: ok = False; problems.append("html not closed")
                if not big: ok = False; problems.append(f"html small ({len(html)})")
            except json.JSONDecodeError:
                ok = False
        passes += bool(ok)
    print(f"[preverify] {passes}/{runs} PASS", flush=True)
    return passes == runs


def mode_soak(port, rounds):
    sysp, tools = build_system_prompt(), build_tools()
    all_ok = True
    for rnd in range(1, rounds + 1):
        print(f"[soak round {rnd}/{rounds}]", flush=True)
        msgs = [{"role": "system", "content": sysp},
                {"role": "user", "content": "Set up a small project: list /tmp/hermes-soak, then create it with "
                 "3 files (config.json, main.py, README.md), track progress with todos, finish with a summary. "
                 "For main.py write a complete 200-line voxel-scene generator script in ONE write_file call."}]
        for turn in range(1, 11):
            content, tcs, finish, stats = stream_request(port, msgs, tools, 131072)
            ok, _ = check_turn(f"r{rnd}t{turn}", content, tcs, finish, stats)
            all_ok &= ok
            if tcs:
                msgs.append({"role": "assistant", "content": content or "",
                             "tool_calls": [{"type": "function", "id": t["id"] or f"call_{turn}",
                                             "function": {"name": t["name"], "arguments": t["args"]}} for t in tcs]})
                for t in tcs:
                    msgs.append({"role": "tool", "tool_call_id": t["id"] or f"call_{turn}",
                                 "content": "ok" if t["name"] != "ls" else "(empty directory)"})
            else:
                msgs.append({"role": "assistant", "content": content})
                if turn < 10:
                    msgs.append({"role": "user", "content": "Continue with the next step."})
            if finish == "stop" and turn >= 8 and not tcs:
                print(f"  [r{rnd}] model concluded at turn {turn}", flush=True)
                break
    print(f"[soak] {'PASS' if all_ok else 'FAIL'}", flush=True)
    return all_ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["preverify", "soak"], required=True)
    ap.add_argument("--port", type=int, default=8130)
    ap.add_argument("--runs", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    ok = mode_preverify(a.port, a.runs) if a.mode == "preverify" else mode_soak(a.port, a.runs if a.runs != 3 else 2)
    sys.exit(0 if ok else 1)
