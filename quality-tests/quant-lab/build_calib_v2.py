#!/usr/bin/env python3
"""calib_v2: workload-calibrated corpus from the user's agent-session harvest.

Source: D:\\spill\\sessions\\FULL (REDACTED set ONLY — law). Emits chat-shaped
text approximating the serving distribution. Output stays LOCAL (never commit
corpus or raw sessions; this builder script is the only committed artifact).

Weights: hermes 60% (qwen-local first = on-distribution), claude-subagents 15%,
claude-main 10%, codex 10%, droid 5%. Per-file extraction cap so the 31MB
monster session cannot dominate. Dedup on content-prefix hash kills repeated
system preambles/compaction copies. Base64/data-URI blobs stripped.
"""
import glob, hashlib, json, os, re

SRC = "/mnt/d/spill/sessions/FULL"
OUT = "/home/erol/ai/turboquant/turboquant-g1/quality-tests/quant-lab/calib_v2.txt"
TARGET = 5_000_000
PER_MSG_CAP = 6_000
PER_FILE_CAP = 300_000

BUCKETS = [  # (glob, budget_bytes)
    (f"{SRC}/hermes/qwen-local/*.jsonl", 1_200_000),
    (f"{SRC}/hermes/k3/*.jsonl",         1_000_000),
    (f"{SRC}/hermes/gpt-5.6-sol/*.jsonl",  600_000),
    (f"{SRC}/hermes/fable/*.jsonl",        200_000),
    (f"{SRC}/claude/subagents_opus/*.jsonl", 750_000),
    (f"{SRC}/claude/main/*.jsonl",         500_000),
    (f"{SRC}/codex/gpt-5.6-sol/*.jsonl",   300_000),
    (f"{SRC}/codex/gpt-5.5/*.jsonl",       200_000),
    (f"{SRC}/droid/opus_main/*.jsonl",     200_000),
    (f"{SRC}/droid/opus_btw/*.jsonl",       50_000),
]

B64RE = re.compile(r"[A-Za-z0-9+/=_-]{200,}")
seen = set()


def clean(t):
    if not isinstance(t, str) or not t.strip():
        return None
    t = B64RE.sub("[BLOB]", t)[:PER_MSG_CAP]
    h = hashlib.md5(t[:200].encode()).hexdigest()
    if h in seen:
        return None
    seen.add(h)
    return t


def blocks_text(content):
    """Flatten claude/codex/droid content-block lists to text."""
    if isinstance(content, str):
        return content
    out = []
    if isinstance(content, list):
        for b in content:
            if not isinstance(b, dict):
                continue
            for k in ("text", "input_text", "output_text", "thinking", "content"):
                v = b.get(k)
                if isinstance(v, str):
                    out.append(v)
            if b.get("type") == "tool_use":
                out.append(json.dumps({"tool": b.get("name"), "input": b.get("input")})[:2000])
            if b.get("type") == "tool_result":
                out.append(blocks_text(b.get("content")) or "")
    return "\n".join(x for x in out if x)


def emit(role, text, sink):
    t = clean(text)
    if t:
        sink.append(f"<|im_start|>{role}\n{t}<|im_end|>\n")


def parse_line(obj, sink):
    t = obj.get("type")
    # hermes flat message
    if t == "message" and "role" in obj and "message" not in obj:
        r = obj["role"]
        if obj.get("reasoning_content"):
            emit(r, "<think>\n" + str(obj["reasoning_content"])[:PER_MSG_CAP // 2] + "\n</think>", sink)
        if obj.get("tool_calls"):
            emit(r, json.dumps(obj["tool_calls"])[:2500], sink)
        emit(r if r != "tool" else "tool", obj.get("content") if isinstance(obj.get("content"), str) else blocks_text(obj.get("content")), sink)
        return
    # claude / droid nested message
    if t in ("user", "assistant", "message") and isinstance(obj.get("message"), dict):
        m = obj["message"]
        emit(m.get("role", t), blocks_text(m.get("content")), sink)
        return
    # codex response_item
    if t == "response_item" and isinstance(obj.get("payload"), dict):
        p = obj["payload"]
        pt = p.get("type")
        if pt == "message":
            emit(p.get("role", "assistant"), blocks_text(p.get("content")), sink)
        elif pt == "reasoning":
            emit("assistant", "<think>\n" + blocks_text(p.get("summary") or p.get("content"))[:PER_MSG_CAP // 2] + "\n</think>", sink)
        elif pt in ("function_call", "custom_tool_call"):
            emit("assistant", json.dumps({"call": p.get("name"), "args": p.get("arguments")})[:2500], sink)


total = 0
chunks = []
for pattern, budget in BUCKETS:
    got = 0
    files = sorted(glob.glob(pattern), key=os.path.getsize)  # small first = session variety
    for f in files:
        if got >= budget:
            break
        sink = []
        fsize = 0
        try:
            with open(f, errors="replace") as fh:
                for line in fh:
                    if fsize >= PER_FILE_CAP:
                        break
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    before = sum(len(x) for x in sink)
                    parse_line(obj, sink)
                    fsize = sum(len(x) for x in sink)
        except OSError:
            continue
        text = "".join(sink)[:PER_FILE_CAP]
        if len(text) > 500:
            chunks.append(text)
            got += len(text)
    total += got
    print(f"{pattern.split('FULL/')[1]:35s} -> {got/1e6:.2f} MB")

corpus = "\n".join(chunks)[:TARGET]
open(OUT, "w").write(corpus)
print(f"\ncalib_v2.txt: {len(corpus)/1e6:.2f} MB (~{len(corpus)//4//1000}K tokens est), {len(seen)} unique msgs")
