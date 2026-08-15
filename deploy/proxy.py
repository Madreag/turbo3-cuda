#!/usr/bin/env python3
"""Reasoning-extraction + session-pinning proxy for llama-server.

Sits between clients (e.g. Factory Droid, Claude Code) and llama-server.

Two client-facing endpoints on the same port, sharing auth + slot-pinning:
- `/v1/chat/completions`  — OpenAI Chat Completions (Factory Droid)
- `/v1/messages`          — Anthropic Messages API (Claude Code)

Both route through the same upstream (OpenAI format), same asyncio.Lock,
same per-user slot save/restore.

Other responsibilities:
1. Per-user API keys → user_id mapping (keys.json).
2. Session pinning: each user has their own KV-cache slot file on disk.
3. Reasoning extraction: <think>...</think> → `reasoning` field (OpenAI)
   or `thinking` content block (Anthropic).
4. Request cleanup: strip reasoning from assistant history.
5. Auth rewrite: validates client key, forwards with server's internal key.

Usage:
  python3 proxy.py --upstream http://127.0.0.1:8131 --port 8130
"""

import argparse
import asyncio
import contextlib
import hashlib
import json
import os
import re
import sys
import time
from typing import Optional

from aiohttp import (web, ClientSession, ClientTimeout, ServerDisconnectedError,
                     ClientConnectionError, ClientPayloadError, ServerTimeoutError)


CONFIG_DIR = "/home/erol/.config/llama-tcq"
STREAM_TRACE_PATH = os.path.join(CONFIG_DIR, "stream-trace.log")


def progress_to_comment(line: str) -> Optional[str]:
    """Convert a prefill-progress SSE data line into an SSE comment.

    llama-server (with return_progress) emits chunks carrying a top-level
    `prompt_progress` object and an empty delta during prompt processing.
    Clients' SSE parsers ignore comment lines per spec, so converting keeps
    every stall watchdog fed without exposing non-standard chunks.
    Returns None if the line is not a pure progress chunk (pass through).
    """
    if not line.startswith("data: ") or '"prompt_progress"' not in line:
        return None
    try:
        obj = json.loads(line[6:])
    except json.JSONDecodeError:
        return None
    pp = obj.get("prompt_progress")
    if not isinstance(pp, dict):
        return None
    choices = obj.get("choices") or [{}]
    delta = (choices[0] or {}).get("delta") or {}
    if (delta.get("content") or delta.get("reasoning")
            or delta.get("reasoning_content") or delta.get("tool_calls")):
        return None  # real payload riding along: pass through untouched
    return f": progress {pp.get('processed', 0)}/{pp.get('total', 0)}"


ARTIFACT_DIR = os.path.join(CONFIG_DIR, "artifacts")


def persist_artifacts(tool_acc: dict, req_id: str) -> None:
    """Byte-truth capture: persist write_file contents that crossed the wire,
    so artifact corruption can be located (wire vs client write vs viewer).
    Keeps the 20 newest files. Best-effort — never disturbs the relay."""
    try:
        for slot_acc in tool_acc.values():
            if slot_acc.get("name") != "write_file":
                continue
            args = json.loads("".join(slot_acc["args"]) or "{}")
            content = args.get("content") or ""
            path_l = str(args.get("path") or "").lower()
            # Capture EVERYTHING that looks like a page — including tiny/blank
            # writes: those ARE the failure evidence (user's "blank or nothing").
            if not (path_l.endswith((".html", ".htm")) or "</html>" in content.lower()
                    or len(content) >= 2000):
                continue
            os.makedirs(ARTIFACT_DIR, exist_ok=True)
            base = os.path.basename(str(args.get("path") or "artifact")).replace("/", "_")[:60]
            out = os.path.join(ARTIFACT_DIR, f"{req_id}_{base}")
            with open(out, "w") as f:
                f.write(content)
            trace_write(f"{time.time():.3f} {req_id} ARTIFACT {len(content)}ch -> {out}")
            old = sorted(os.listdir(ARTIFACT_DIR),
                         key=lambda n: os.path.getmtime(os.path.join(ARTIFACT_DIR, n)))
            for n in old[:-20]:
                os.remove(os.path.join(ARTIFACT_DIR, n))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        pass


_trace_fh = None


def trace_write(line: str) -> None:
    """Per-chunk stream trace for stall diagnosis. Best-effort, size-capped.
    Holds one open handle — the previous open/stat/close per SSE chunk was
    ~100k synchronous filesystem round-trips a day on the event loop."""
    global _trace_fh
    try:
        if _trace_fh is None:
            _trace_fh = open(STREAM_TRACE_PATH, "a")
        if _trace_fh.tell() > 50_000_000:
            _trace_fh.close()
            _trace_fh = None
            os.replace(STREAM_TRACE_PATH, STREAM_TRACE_PATH + ".1")
            _trace_fh = open(STREAM_TRACE_PATH, "a")
        _trace_fh.write(line + "\n")
        _trace_fh.flush()
    except (OSError, ValueError):
        _trace_fh = None


CAPTURES_DIR = os.path.join(CONFIG_DIR, "captures")


def classify_terminal_line(line: str) -> Optional[str]:
    """SSE line classifier for stream-termination tripwires.
    'done' = the OpenAI stream terminator. 'error' = an in-stream error frame:
    llama-server serializes exceptions into the stream as data: {"error": ...}
    and then closes WITHOUT [DONE] — the corpse signature of 2026-08-08.
    Tolerates CRLF framing, `data:` without a space, and key-reordered error
    objects (a real error frame must have "error" at the JSON top level; model
    text lives escaped inside "choices" frames and cannot produce one)."""
    line = line.rstrip("\r")
    if not line.startswith("data:"):
        return None
    data = line[5:].strip()
    if data == "[DONE]":
        return "done"
    if not data:
        return None
    if data.startswith('{"error"'):
        return "error"
    if '"error"' in data:
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            return None
        if isinstance(obj, dict) and "error" in obj:
            return "error"
    return None


def capture_tools_request(body_obj: dict, user_id: str) -> None:
    """Persist tools-bearing request bodies (pre-mutation) so regression gates
    replay REAL client traffic instead of hand-built replicas — the battery's
    38-tool 'replica' missing the live 48-param computer_use is what hid the
    grammar bomb. One file per (user, tool-suite hash): battery traffic cannot
    clobber the live suite's capture."""
    try:
        tools_key = hashlib.md5(
            json.dumps(body_obj.get("tools"), sort_keys=True).encode()).hexdigest()[:8]
        os.makedirs(CAPTURES_DIR, exist_ok=True)
        path = os.path.join(CAPTURES_DIR, f"tools_{user_id}_{tools_key}.json")
        # Same suite captured within the last hour → skip: bodies differ every
        # turn but replay value doesn't, and this was a synchronous multi-100KB
        # write on the event loop per request.
        try:
            if time.time() - os.path.getmtime(path) < 3600:
                return
        except OSError:
            pass
        with open(path, "w") as f:
            json.dump({"captured_at": time.time(),
                       "body": strip_image_payloads(body_obj)}, f)
        old = sorted(os.listdir(CAPTURES_DIR),
                     key=lambda n: os.path.getmtime(os.path.join(CAPTURES_DIR, n)))
        for n in old[:-40]:
            os.remove(os.path.join(CAPTURES_DIR, n))
    except Exception as e:
        print(f"[capture] skipped: {e}", flush=True)


_B64_BLOB_RE = re.compile(r"^[A-Za-z0-9+/=\r\n]{4096,}$")


def strip_image_payloads(body_obj: dict) -> dict:
    """Copy of the request with binary blobs replaced by short placeholders —
    captures are for tool-suite/prompt replay, and base64 adds nothing. Walks
    any nesting (OpenAI image_url, Anthropic source.data, audio/file parts)
    and only touches strings that are data: URIs or pure-base64 ≥4 KiB, so
    real prompt text and code are never stripped. Builds a new structure; the
    forwarded body is never mutated (and no deepcopy doubling peak memory)."""
    def walk(node):
        if isinstance(node, dict):
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        if isinstance(node, str) and len(node) >= 4096 and (
                node.startswith("data:") or _B64_BLOB_RE.match(node)):
            return f"<stripped {len(node)} chars>"
        return node
    return walk(body_obj)


KEYS_PATH = os.path.join(CONFIG_DIR, "keys.json")
SERVER_KEY_PATH = os.path.join(CONFIG_DIR, "api.key")
SLOT_ID = 0  # --parallel 1, single slot

# Hop-by-hop headers (RFC 9110 §7.6.1) + auth/framing headers the proxy owns.
# Forwarding Connection/Transfer-Encoding upstream alongside aiohttp's own
# framing produced conflicting semantics (bughunt #20).
_HOP_HEADERS = {"host", "content-length", "authorization", "connection",
                "keep-alive", "proxy-authenticate", "proxy-authorization",
                "te", "trailer", "transfer-encoding", "upgrade", "expect"}

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
# Normalize both tag variants to the canonical <think>:
#   <think> ... </think>       — Qwen style
#   <thinking> ... </thinking> — Claude style (Qwopus v3 distilled from Claude emits this)
_THINK_NORMALIZE_RE = re.compile(r"<(/?)thinking>")


def normalize_think_tags(text: str) -> str:
    """Fold <thinking>/<'/thinking'> into <think>/</think> so the rest of the
    reasoning-extraction pipeline (state machine, regex) only needs to handle
    one tag form."""
    return _THINK_NORMALIZE_RE.sub(r"<\1think>", text)


THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>\s*", re.DOTALL)


# ─── Reasoning extraction ──────────────────────────────────────────────────────


def split_thinking(content: str) -> tuple[str, str]:
    content = normalize_think_tags(content)
    reasoning_parts = []
    def capture(m: re.Match) -> str:
        reasoning_parts.append(m.group(1).strip())
        return ""
    cleaned = THINK_BLOCK_RE.sub(capture, content)
    if cleaned.lstrip().startswith(THINK_OPEN):
        idx = cleaned.find(THINK_OPEN)
        reasoning_parts.append(cleaned[idx + len(THINK_OPEN):].strip())
        cleaned = cleaned[:idx]
    reasoning = "\n\n".join(r for r in reasoning_parts if r).strip()
    return reasoning, cleaned.strip()


def transform_non_stream(payload: dict) -> dict:
    for choice in payload.get("choices", []):
        msg = choice.get("message") or {}
        content = msg.get("content") or ""
        if content:
            reasoning, cleaned = split_thinking(content)
            if reasoning:
                msg["reasoning"] = reasoning
                msg["content"] = cleaned
        if "reasoning_content" in msg:
            rc = msg.pop("reasoning_content")
            if rc:
                existing = msg.get("reasoning") or ""
                msg["reasoning"] = (existing + "\n\n" + rc).strip() if existing else rc
    return payload


class StreamState:
    def __init__(self):
        self.mode = "content"
        self.buffer = ""
        # True once real (non-whitespace) answer content has been emitted.
        # After that point <think>/<thinking> is treated as literal text: the
        # template puts thinking FIRST, so a later tag is model output (e.g.
        # an artifact documenting think tags), not a reasoning block —
        # previously it silently rerouted the rest of the answer (bughunt #10).
        self.content_started = False
        # Accumulated totals for the whole response — used by the
        # preserve_thinking reasoning-memory (Qwen3.6 is post-trained to keep
        # prior-turn thinking in context; clients don't echo it back, so the
        # proxy remembers it and re-inlines on history replay).
        self.full_content = ""
        self.full_reasoning = ""

    def process(self, delta_content: str) -> tuple[str, str]:
        new_content = []
        new_reasoning = []
        # Combine with buffer, THEN normalize <thinking>/</thinking> → <think>/</think>.
        # Normalizing post-concat catches tags split across SSE chunks
        # (e.g. chunk A ends with "<think" and chunk B starts with "ing>").
        # Only normalize while still in the leading segment — once content has
        # started, a literal <thinking> in the answer must survive untouched.
        text = self.buffer + delta_content
        if not self.content_started:
            text = normalize_think_tags(text)
        self.buffer = ""
        while text:
            if self.mode == "content":
                if self.content_started:
                    new_content.append(text)
                    text = ""
                    continue
                idx = text.find(THINK_OPEN)
                if idx == -1:
                    tail_check = min(len(text), len(THINK_OPEN) - 1)
                    for k in range(tail_check, 0, -1):
                        if text[-k:] == THINK_OPEN[:k]:
                            new_content.append(text[:-k])
                            self.buffer = text[-k:]
                            text = ""
                            break
                    else:
                        new_content.append(text)
                        text = ""
                    if "".join(new_content).strip():
                        self.content_started = True
                        # A held tag-prefix after real content is literal text.
                        if self.buffer:
                            new_content.append(self.buffer)
                            self.buffer = ""
                elif text[:idx].strip():
                    # Opener found but real content precedes it in this same
                    # chunk → the tag is literal text, not a thinking block.
                    self.content_started = True
                    new_content.append(text)
                    text = ""
                else:
                    new_content.append(text[:idx])
                    text = text[idx + len(THINK_OPEN):]
                    self.mode = "thinking"
            else:
                idx = text.find(THINK_CLOSE)
                if idx == -1:
                    tail_check = min(len(text), len(THINK_CLOSE) - 1)
                    for k in range(tail_check, 0, -1):
                        if text[-k:] == THINK_CLOSE[:k]:
                            new_reasoning.append(text[:-k])
                            self.buffer = text[-k:]
                            text = ""
                            break
                    else:
                        new_reasoning.append(text)
                        text = ""
                else:
                    new_reasoning.append(text[:idx])
                    text = text[idx + len(THINK_CLOSE):]
                    self.mode = "content"
                    text = text.lstrip("\n")
        out_c, out_r = "".join(new_content), "".join(new_reasoning)
        self.full_content += out_c
        self.full_reasoning += out_r
        return out_c, out_r

    def finish(self) -> str:
        """Flush whatever the tag-matcher still buffers at stream end — a
        trailing partial tag like '<thi' was previously dropped silently.
        Returns residue that belongs in CONTENT (thinking residue only lands
        in the accumulated totals)."""
        tail = self.buffer
        self.buffer = ""
        if not tail:
            return ""
        if self.mode == "content":
            self.full_content += tail
            return tail
        self.full_reasoning += tail
        return ""


def transform_sse_line(line: str, state: StreamState) -> list[str]:
    if not line.startswith("data:"):
        return [line]
    data = line[5:].strip()
    if data == "[DONE]" or not data:
        return [line]
    try:
        obj = json.loads(data)
    except json.JSONDecodeError:
        return [line]
    try:
        choices = obj.get("choices") or []
        if not choices:
            return [line]
        delta = choices[0].get("delta") or {}
        if "reasoning_content" in delta:
            rc = delta.pop("reasoning_content")
            if rc:
                existing = delta.get("reasoning") or ""
                delta["reasoning"] = existing + rc
        content = delta.get("content")
        if content is not None:
            new_content, new_reasoning = state.process(content)
            if new_reasoning:
                existing = delta.get("reasoning") or ""
                delta["reasoning"] = existing + new_reasoning
            if new_content:
                delta["content"] = new_content
            elif delta.get("reasoning"):
                delta.pop("content", None)
            else:
                delta["content"] = new_content
        return [f"data: {json.dumps(obj)}"]
    except Exception:
        return [line]


def remember_reasoning(memory: dict, content: str, reasoning: str) -> None:
    """Store reasoning keyed by the assistant content it accompanied (LRU 50)."""
    if not reasoning or not content or not content.strip():
        return
    import hashlib
    key = hashlib.sha1(content.strip().encode("utf-8", "replace")).hexdigest()
    memory[key] = reasoning
    while len(memory) > 50:
        memory.pop(next(iter(memory)))


def reinject_thinking_in_request(body_obj: dict, memory: Optional[dict] = None) -> dict:
    """preserve_thinking: Qwen3.6 is post-trained expecting prior-turn <think>
    blocks in context. Clients don't echo reasoning back, so re-inline it into
    assistant history — from the client's reasoning field if sent, else from
    the proxy's per-user reasoning memory. Also restores byte-identical history
    vs the server's generated tokens, so the prompt cache actually hits."""
    import hashlib
    msgs = body_obj.get("messages") or []
    for msg in msgs:
        if msg.get("role") != "assistant":
            continue
        r = msg.pop("reasoning", None) or msg.pop("reasoning_content", None)
        msg.pop("reasoning", None)
        msg.pop("reasoning_content", None)
        c = msg.get("content")
        if not isinstance(c, str) or THINK_OPEN in c:
            continue
        if not r and memory is not None and c.strip():
            key = hashlib.sha1(c.strip().encode("utf-8", "replace")).hexdigest()
            r = memory.get(key)
        if r:
            msg["content"] = f"{THINK_OPEN}\n{r}\n{THINK_CLOSE}\n\n{c}"
    return body_obj


# ─── Anthropic Messages API translation ────────────────────────────────────────
#
# Claude Code (and any anthropic SDK client) sends Anthropic-format requests to
# /v1/messages. We translate to OpenAI Chat Completions for upstream llama-server
# and translate the response back.
#
# Covered: text, tool_use, tool_result, thinking. Images skipped (text model).

STOP_REASON_OPENAI_TO_ANTHROPIC = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
    "content_filter": "refusal",
}


def _flatten_text_blocks(blocks: list) -> str:
    """Concatenate 'text'-type blocks from an Anthropic content list."""
    out = []
    for b in blocks:
        if b.get("type") == "text":
            out.append(b.get("text", ""))
    return "".join(out)


def anthropic_request_to_openai(body: dict) -> dict:
    """Translate Anthropic Messages API request → OpenAI Chat Completions."""
    openai_msgs: list = []

    system = body.get("system")
    if system:
        if isinstance(system, str):
            openai_msgs.append({"role": "system", "content": system})
        elif isinstance(system, list):
            text = _flatten_text_blocks(system)
            if text:
                openai_msgs.append({"role": "system", "content": text})

    for msg in body.get("messages") or []:
        role = msg.get("role")
        content = msg.get("content")

        if isinstance(content, str):
            openai_msgs.append({"role": role, "content": content})
            continue

        if not isinstance(content, list):
            continue

        text_parts: list = []
        tool_calls: list = []
        tool_results: list = []
        image_parts: list = []

        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "image" and role == "user":
                # Vision is live upstream (--mmproj): translate Anthropic image
                # blocks to OpenAI image_url parts instead of dropping them
                # (bughunt #14 — silent drop produced confidently blind answers).
                src = block.get("source") or {}
                if src.get("type") == "base64" and src.get("data"):
                    media = src.get("media_type") or "image/png"
                    image_parts.append({"type": "image_url", "image_url": {
                        "url": f"data:{media};base64,{src['data']}"}})
                elif src.get("type") == "url" and src.get("url"):
                    image_parts.append({"type": "image_url",
                                        "image_url": {"url": src["url"]}})
            elif btype == "tool_use" and role == "assistant":
                tool_calls.append({
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {})),
                    },
                })
            elif btype == "tool_result" and role == "user":
                tc_content = block.get("content", "")
                if isinstance(tc_content, list):
                    tc_content = _flatten_text_blocks(tc_content)
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id", ""),
                    "content": tc_content,
                })
            # thinking/image blocks in history: drop

        # Tool results come from user turn but must be separate OpenAI "tool" messages
        if tool_results:
            openai_msgs.extend(tool_results)

        text = "".join(text_parts).strip()
        if role == "assistant":
            out_msg: dict = {"role": "assistant"}
            out_msg["content"] = text or ""
            if tool_calls:
                out_msg["tool_calls"] = tool_calls
            openai_msgs.append(out_msg)
        elif role == "user":
            # User message: any remaining text (tool_results already flushed).
            # Suppress empty user messages that had ONLY tool_result blocks.
            if image_parts:
                parts: list = ([{"type": "text", "text": text}] if text else [])
                openai_msgs.append({"role": "user", "content": parts + image_parts})
            elif text:
                openai_msgs.append({"role": "user", "content": text})

    openai: dict = {
        "model": body.get("model", "qwopus-v3"),
        "messages": openai_msgs,
        "max_tokens": body.get("max_tokens", 4096),
        "stream": bool(body.get("stream", False)),
    }
    for src, dest in (("temperature", "temperature"), ("top_p", "top_p"),
                      ("top_k", "top_k"), ("stop_sequences", "stop")):
        if src in body:
            openai[dest] = body[src]

    if "tools" in body:
        openai["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema") or {"type": "object"},
                },
            }
            for t in body["tools"]
        ]

    if "tool_choice" in body:
        tc = body["tool_choice"] or {}
        ttype = tc.get("type")
        if ttype == "auto":
            openai["tool_choice"] = "auto"
        elif ttype == "any":
            openai["tool_choice"] = "required"
        elif ttype == "tool" and tc.get("name"):
            openai["tool_choice"] = {"type": "function",
                                      "function": {"name": tc["name"]}}
        elif ttype == "none":
            openai["tool_choice"] = "none"

    return openai


def _anthropic_usage_from_openai(usage: dict) -> dict:
    """Map OpenAI usage → Anthropic usage shape.

    llama.cpp reports `prompt_tokens_details.cached_tokens` = tokens reused from
    the slot's KV prefix. We translate:
      - cache_read_input_tokens  = cached_tokens  (reused from existing slot state)
      - cache_creation_input_tokens = prompt_tokens - cached_tokens
                                     (new tokens processed this turn, which DO
                                      become part of the slot cache for next turn)
      - input_tokens = 0  (we don't have an Anthropic-style "non-cacheable" class)
      - output_tokens = completion_tokens
    """
    prompt_tokens = usage.get("prompt_tokens", 0) or 0
    cached_tokens = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
    new_tokens = max(0, prompt_tokens - cached_tokens)
    return {
        "input_tokens": 0,
        "cache_creation_input_tokens": new_tokens,
        "cache_read_input_tokens": cached_tokens,
        "output_tokens": usage.get("completion_tokens", 0) or 0,
    }


def openai_response_to_anthropic(openai_resp: dict, model: str) -> dict:
    """Translate non-streaming OpenAI response → Anthropic Messages response."""
    choices = openai_resp.get("choices") or []
    choice = choices[0] if choices else {}
    msg = choice.get("message") or {}
    finish_reason = choice.get("finish_reason", "stop")
    usage = openai_resp.get("usage") or {}

    content_blocks: list = []

    # Extract <think> from content (same logic as openai responses)
    raw_content = msg.get("content") or ""
    inline_reasoning, cleaned_content = split_thinking(raw_content) if raw_content else ("", "")
    explicit_reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    combined_reasoning = (explicit_reasoning + ("\n\n" + inline_reasoning if inline_reasoning else "")).strip()

    if combined_reasoning:
        content_blocks.append({
            "type": "thinking",
            "thinking": combined_reasoning,
            "signature": "",
        })

    if cleaned_content:
        content_blocks.append({"type": "text", "text": cleaned_content})

    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {}) or {}
        try:
            input_obj = json.loads(fn.get("arguments") or "{}")
        except (json.JSONDecodeError, TypeError):
            input_obj = {"_raw": fn.get("arguments", "")}
        content_blocks.append({
            "type": "tool_use",
            "id": tc.get("id") or f"toolu_{uuid4()}",
            "name": fn.get("name", ""),
            "input": input_obj,
        })

    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    return {
        "id": openai_resp.get("id") or f"msg_{uuid4()}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
        "stop_reason": STOP_REASON_OPENAI_TO_ANTHROPIC.get(finish_reason, "end_turn"),
        "stop_sequence": None,
        "usage": _anthropic_usage_from_openai(usage),
    }


# ─── Streaming: OpenAI SSE → Anthropic SSE ─────────────────────────────────────
#
# Anthropic SSE event sequence for a typical turn with thinking + text + tool use:
#   event: message_start     → envelope with message metadata
#   event: content_block_start (index=0, type=thinking)  [if thinking]
#   event: content_block_delta (thinking_delta)
#   event: content_block_stop
#   event: content_block_start (index=1, type=text)
#   event: content_block_delta (text_delta)
#   event: content_block_stop
#   event: content_block_start (index=2, type=tool_use)
#   event: content_block_delta (input_json_delta)  [partial JSON]
#   event: content_block_stop
#   event: message_delta     → final stop_reason + usage.output_tokens
#   event: message_stop
# `event: ping` can be inserted anywhere as keep-alive.
#
# Index must increase monotonically per content block. Only one block open at a time.


class AnthropicStreamBuilder:
    """Translates an OpenAI SSE chunk stream into Anthropic SSE events.

    Call `.feed(openai_sse_line)` repeatedly with the raw SSE lines from
    llama-server; receive back a list of fully-formed Anthropic SSE frames
    (including trailing blank line) to write to the client.

    On upstream `[DONE]`, call `.close()` to get final `message_delta` +
    `message_stop` events.
    """

    BLOCK_NONE = 0
    BLOCK_THINKING = 1
    BLOCK_TEXT = 2
    BLOCK_TOOL = 3

    def __init__(self, message_id: str, model: str):
        self.message_id = message_id
        self.model = model
        self.started = False
        self.block_index = -1
        self.block_type = self.BLOCK_NONE
        # For streaming <think> extraction from content
        self.think_state = StreamState()
        # Tool use tracking: openai tool_calls[index] → anthropic block_index
        self.tool_oai_to_ant: dict = {}
        # Per-tool cumulative arg buffer (for splitting partial json)
        self.tool_arg_buf: dict = {}
        self.stop_reason = "end_turn"
        # Usage accumulation for final message_delta event
        self.prompt_tokens = 0
        self.cached_tokens = 0
        self.output_tokens = 0
        self.closed = False

    @staticmethod
    def _event(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    def _start_message(self) -> str:
        self.started = True
        return self._event("message_start", {
            "type": "message_start",
            "message": {
                "id": self.message_id,
                "type": "message",
                "role": "assistant",
                "model": self.model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        })

    def _close_current_block(self) -> str:
        if self.block_type == self.BLOCK_NONE:
            return ""
        out = self._event("content_block_stop", {
            "type": "content_block_stop",
            "index": self.block_index,
        })
        self.block_type = self.BLOCK_NONE
        return out

    def _open_block(self, btype: int, start_payload: dict) -> list:
        out = [self._close_current_block()] if self.block_type != self.BLOCK_NONE else []
        self.block_index += 1
        self.block_type = btype
        out.append(self._event("content_block_start", {
            "type": "content_block_start",
            "index": self.block_index,
            "content_block": start_payload,
        }))
        return out

    def _append_thinking(self, delta: str) -> list:
        out: list = []
        if not self.started:
            out.append(self._start_message())
        if self.block_type != self.BLOCK_THINKING:
            out.extend(self._open_block(self.BLOCK_THINKING, {
                "type": "thinking",
                "thinking": "",
            }))
        out.append(self._event("content_block_delta", {
            "type": "content_block_delta",
            "index": self.block_index,
            "delta": {"type": "thinking_delta", "thinking": delta},
        }))
        return out

    def _append_text(self, delta: str) -> list:
        out: list = []
        if not self.started:
            out.append(self._start_message())
        if self.block_type != self.BLOCK_TEXT:
            out.extend(self._open_block(self.BLOCK_TEXT, {
                "type": "text",
                "text": "",
            }))
        out.append(self._event("content_block_delta", {
            "type": "content_block_delta",
            "index": self.block_index,
            "delta": {"type": "text_delta", "text": delta},
        }))
        return out

    def _append_tool_call(self, oai_index: int, tc_delta: dict) -> list:
        """Handle an OpenAI tool_calls delta chunk.

        First chunk for a given oai_index has `id`/`function.name`; later chunks
        append JSON fragments to `function.arguments`.
        """
        out: list = []
        if not self.started:
            out.append(self._start_message())
        if oai_index not in self.tool_oai_to_ant:
            # New tool_use block — open it
            fn = tc_delta.get("function", {}) or {}
            tc_id = tc_delta.get("id") or f"toolu_{uuid4()}"
            tc_name = fn.get("name", "")
            out.extend(self._open_block(self.BLOCK_TOOL, {
                "type": "tool_use",
                "id": tc_id,
                "name": tc_name,
                "input": {},
            }))
            self.tool_oai_to_ant[oai_index] = self.block_index
            self.tool_arg_buf[oai_index] = ""

        # Emit any argument fragment as input_json_delta
        fn = tc_delta.get("function", {}) or {}
        arg_frag = fn.get("arguments", "") or ""
        if arg_frag:
            self.tool_arg_buf[oai_index] += arg_frag
            out.append(self._event("content_block_delta", {
                "type": "content_block_delta",
                "index": self.tool_oai_to_ant[oai_index],
                "delta": {"type": "input_json_delta", "partial_json": arg_frag},
            }))
        return out

    def feed(self, line: str) -> list:
        """Consume one OpenAI SSE line; return list of Anthropic SSE frames."""
        if self.closed:
            return []
        if not line.startswith("data:"):
            return []
        data = line[5:].strip()
        if not data:
            return []
        if data == "[DONE]":
            return self.close()
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            return []

        out: list = []
        usage = obj.get("usage") or {}
        if usage:
            self.prompt_tokens = usage.get("prompt_tokens", self.prompt_tokens)
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
            if cached is not None:
                self.cached_tokens = cached
            self.output_tokens = usage.get("completion_tokens", self.output_tokens)

        choices = obj.get("choices") or []
        if not choices:
            return out

        choice = choices[0]
        delta = choice.get("delta") or {}

        # 1. reasoning_content / reasoning → thinking
        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
        if reasoning:
            out.extend(self._append_thinking(reasoning))

        # 2. inline <think>...</think> inside content → thinking + text
        raw_content = delta.get("content")
        if raw_content:
            new_content, new_reasoning = self.think_state.process(raw_content)
            if new_reasoning:
                out.extend(self._append_thinking(new_reasoning))
            if new_content:
                out.extend(self._append_text(new_content))

        # 3. tool_calls
        for tc in delta.get("tool_calls") or []:
            idx = tc.get("index", 0)
            out.extend(self._append_tool_call(idx, tc))

        # 4. finish_reason
        finish = choice.get("finish_reason")
        if finish:
            self.stop_reason = STOP_REASON_OPENAI_TO_ANTHROPIC.get(finish, "end_turn")

        return out

    def close(self) -> list:
        """Emit closing events. Idempotent."""
        if self.closed:
            return []
        self.closed = True
        out: list = []
        if not self.started:
            out.append(self._start_message())
        if self.block_type != self.BLOCK_NONE:
            out.append(self._close_current_block())
        new_tokens = max(0, self.prompt_tokens - self.cached_tokens)
        out.append(self._event("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": self.stop_reason, "stop_sequence": None},
            "usage": {
                "input_tokens": 0,
                "cache_creation_input_tokens": new_tokens,
                "cache_read_input_tokens": self.cached_tokens,
                "output_tokens": self.output_tokens,
            },
        }))
        out.append(self._event("message_stop", {"type": "message_stop"}))
        return out


def uuid4() -> str:
    import uuid
    return uuid.uuid4().hex[:24]


# ─── Session pinning ──────────────────────────────────────────────────────────


def load_keys() -> dict:
    with open(KEYS_PATH) as f:
        return json.load(f)


def load_server_key() -> str:
    with open(SERVER_KEY_PATH) as f:
        return f.read().strip()


def extract_user_id(auth_header: str, keys: dict) -> Optional[str]:
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
    token = auth_header[len("Bearer "):].strip()
    if not token:
        return None
    return keys.get(token)


async def slot_save(session: ClientSession, upstream: str, server_key: str, user_id: str) -> bool:
    url = f"{upstream}/slots/{SLOT_ID}?action=save"
    payload = {"filename": f"{user_id}.bin"}
    try:
        async with session.post(url, json=payload,
                                headers={"Authorization": f"Bearer {server_key}"},
                                timeout=ClientTimeout(total=30)) as r:
            ok = r.status == 200
            if ok:
                data = await r.json()
                ms = data.get("timings", {}).get("save_ms", 0)
                print(f"[slot] saved {user_id}: {data.get('n_saved', 0)} tokens, "
                      f"{data.get('n_written', 0)//(1024*1024)} MiB, {ms:.0f}ms",
                      flush=True)
            else:
                body = await r.text()
                print(f"[slot] save {user_id} failed: HTTP {r.status} {body[:200]}", flush=True)
            return ok
    except Exception as e:
        print(f"[slot] save {user_id} exception: {e}", flush=True)
        return False


async def slot_restore(session: ClientSession, upstream: str, server_key: str, user_id: str) -> bool:
    url = f"{upstream}/slots/{SLOT_ID}?action=restore"
    payload = {"filename": f"{user_id}.bin"}
    try:
        async with session.post(url, json=payload,
                                headers={"Authorization": f"Bearer {server_key}"},
                                timeout=ClientTimeout(total=30)) as r:
            ok = r.status == 200
            if ok:
                data = await r.json()
                ms = data.get("timings", {}).get("restore_ms", 0)
                print(f"[slot] restored {user_id}: {data.get('n_restored', 0)} tokens, "
                      f"{data.get('n_read', 0)//(1024*1024)} MiB, {ms:.0f}ms",
                      flush=True)
            return ok
    except Exception as e:
        print(f"[slot] restore {user_id} exception: {e}", flush=True)
        return False


async def slot_erase(session: ClientSession, upstream: str, server_key: str) -> None:
    url = f"{upstream}/slots/{SLOT_ID}?action=erase"
    try:
        async with session.post(url,
                                headers={"Authorization": f"Bearer {server_key}"},
                                timeout=ClientTimeout(total=10)) as r:
            if r.status == 200:
                print("[slot] erased", flush=True)
    except Exception as e:
        print(f"[slot] erase exception: {e}", flush=True)


async def maybe_swap_slot(app, session, upstream, server_key, requested_user):
    # If a previous swap was cancelled mid-flight (client disconnect), its
    # shielded task is still running — wait for it before deciding anything.
    prev = app.get("_swap_task")
    if prev is not None and not prev.done():
        await asyncio.shield(prev)
    current = app["current_owner"]
    if current == requested_user:
        return
    print(f"[slot] SWAP {current or '(empty)'} → {requested_user}", flush=True)
    start = time.time()

    async def _do_swap():
        # Owner is unknown for the duration of the swap: a crash/cancel between
        # save and restore must not leave the proxy believing the OLD owner's
        # bytes are current (bughunt #11 — desync persisted the wrong user's KV).
        app["current_owner"] = None
        if current is not None:
            await slot_save(session, upstream, server_key, current)
        # Always ATTEMPT restore — no filesystem existence check. The proxy
        # previously peeked a hardcoded `slots/` dir while the server saved to
        # `slots-long/` (bughunt A1: restore silently dead). The server answers
        # 400 for a missing/incompatible file and we fall back to erase.
        ok = await slot_restore(session, upstream, server_key, requested_user)
        if not ok:
            await slot_erase(session, upstream, server_key)
        app["current_owner"] = requested_user

    task = asyncio.ensure_future(_do_swap())
    app["_swap_task"] = task
    # shield: if THIS request is cancelled, the swap still runs to a consistent
    # end state (owner assigned only after the physical slot matches).
    await asyncio.shield(task)
    print(f"[slot] swap complete in {(time.time()-start)*1000:.0f}ms", flush=True)


# ─── HTTP handlers ─────────────────────────────────────────────────────────────


async def handle_chat_completions(request: web.Request) -> web.StreamResponse:
    upstream = request.app["upstream"].rstrip("/")
    keys = request.app["keys"]
    server_key = request.app["server_key"]
    session: ClientSession = request.app["session"]

    # Auth check: client key → user_id
    user_id = extract_user_id(request.headers.get("Authorization", ""), keys)
    if user_id is None:
        return web.json_response(
            {"error": {"message": "Invalid API key", "type": "invalid_request_error"}},
            status=401,
        )

    reasoning_mem = request.app.setdefault("reasoning_memory", {}).setdefault(user_id, {})
    body_bytes = await request.read()
    is_streaming = False
    try:
        body_obj = json.loads(body_bytes) if body_bytes else {}
        if body_bytes and not isinstance(body_obj, dict):
            # A JSON array/scalar body crashed the handler with AttributeError
            # further down (bughunt #15) — reject it properly.
            return web.json_response(
                {"error": {"message": "request body must be a JSON object",
                           "type": "invalid_request_error"}},
                status=400,
            )
        if isinstance(body_obj, dict) and body_obj.get("tools"):
            capture_tools_request(body_obj, user_id)
        is_streaming = bool(body_obj.get("stream"))
        body_obj = reinject_thinking_in_request(body_obj, reasoning_mem)
        if is_streaming:
            # Ask llama-server for prefill progress chunks (present in our build,
            # off by default). The relay converts them to SSE comments downstream,
            # so clients see continuous activity through long prefills without
            # their parsers ever encountering non-standard chunks.
            body_obj.setdefault("return_progress", True)
        body_bytes = json.dumps(body_obj).encode("utf-8")
    except json.JSONDecodeError:
        pass

    # Hold the lock for the ENTIRE request lifetime (including streaming).
    # This serializes requests across users — strict queued behavior —
    # while requests from the same user still benefit from the cache.
    async with request.app["owner_lock"]:
        await maybe_swap_slot(request.app, session, upstream, server_key, user_id)

        # Rewrite auth to server's internal key
        headers = {k: v for k, v in request.headers.items()
                   if k.lower() not in _HOP_HEADERS}
        headers["Authorization"] = f"Bearer {server_key}"

        url = upstream + request.rel_url.path
        if request.rel_url.query_string:
            url += "?" + request.rel_url.query_string

        timeout = ClientTimeout(total=None, sock_read=600)

        response_prepared = False
        for _attempt in (0, 1):
            try:
                async with session.request(
                    request.method, url, headers=headers, data=body_bytes, timeout=timeout
                ) as upstream_resp:
                    resp_headers = {
                        k: v for k, v in upstream_resp.headers.items()
                        if k.lower() not in ("content-length", "content-encoding", "transfer-encoding")
                    }

                    if not is_streaming:
                        data = await upstream_resp.read()
                        try:
                            obj = json.loads(data)
                            obj = transform_non_stream(obj)
                            try:
                                m = obj["choices"][0]["message"]
                                remember_reasoning(reasoning_mem, m.get("content") or "",
                                                   m.get("reasoning") or "")
                            except (KeyError, IndexError, TypeError):
                                pass
                            out = json.dumps(obj).encode("utf-8")
                        except json.JSONDecodeError:
                            out = data
                        return web.Response(body=out, status=upstream_resp.status, headers=resp_headers)

                    response = web.StreamResponse(status=upstream_resp.status, headers=resp_headers)
                    response.enable_chunked_encoding()
                    await response.prepare(request)
                    response_prepared = True

                    state = StreamState()
                    line_buffer = ""
                    tool_acc: dict = {}
                    saw_done = False
                    saw_error_frame = False
                    req_id = f"{int(time.time() * 1000) % 10_000_000:07d}"
                    trace_write(f"{time.time():.3f} {req_id} START user={user_id}")

                    # Heartbeat WITHOUT cancelling the pending read: wait_for()
                    # cancels readany() on timeout, and a cancel that races an
                    # arriving chunk tears down the aiohttp connection (clean
                    # EOF, no [DONE]). asyncio.wait() leaves the read pending.
                    read_task = None
                    pump_error = None
                    pump_ok = False
                    try:
                        while True:
                            if read_task is None:
                                read_task = asyncio.ensure_future(
                                    upstream_resp.content.readany())
                            done, _ = await asyncio.wait({read_task}, timeout=10.0)
                            if not done:
                                # SSE comment heartbeat: spec-ignored by clients,
                                # keeps stall watchdogs fed through silent
                                # stretches (long prefills, grammar pauses).
                                await response.write(b": hb\n\n")
                                trace_write(f"{time.time():.3f} {req_id} HB upstream-silent>10s")
                                continue
                            chunk = read_task.result()
                            read_task = None
                            if not chunk:
                                pump_ok = True
                                break
                            flags = "".join(
                                t for t, m in ((("T"), b'"tool_calls"'),
                                               (("R"), b'"reasoning'),
                                               (("C"), b'"content"')) if m in chunk)
                            trace_write(f"{time.time():.3f} {req_id} {len(chunk)}B {flags}")
                            line_buffer += chunk.decode("utf-8", errors="replace")
                            parts = line_buffer.split("\n")
                            line_buffer = parts.pop()
                            out_chunk = []
                            for line in parts:
                                term = classify_terminal_line(line)
                                if term == "done":
                                    saw_done = True
                                elif term == "error":
                                    saw_error_frame = True
                                pc = progress_to_comment(line)
                                if pc is not None:
                                    out_chunk.append(pc + "\n")
                                    continue
                                if '"tool_calls"' in line and line.startswith("data: "):
                                    try:
                                        for tc in (json.loads(line[6:])["choices"][0]
                                                   .get("delta", {}).get("tool_calls") or []):
                                            slot_acc = tool_acc.setdefault(tc.get("index", 0),
                                                                           {"name": "", "args": []})
                                            fn = tc.get("function") or {}
                                            if fn.get("name"):
                                                slot_acc["name"] = fn["name"]
                                            if fn.get("arguments"):
                                                slot_acc["args"].append(fn["arguments"])
                                    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                                        pass
                                for out_line in transform_sse_line(line, state):
                                    out_chunk.append(out_line + "\n")
                            if out_chunk:
                                await response.write("".join(out_chunk).encode("utf-8"))
                    except (ClientPayloadError, ClientConnectionError,
                            ServerTimeoutError, asyncio.TimeoutError) as e:
                        # Upstream died mid-stream. Don't let it surface as a raw
                        # transport drop — the client gets a real error frame +
                        # [DONE] below (bughunt A3 / #9).
                        pump_error = f"upstream died mid-stream: {type(e).__name__}"
                    finally:
                        if read_task is not None:
                            read_task.cancel()
                            with contextlib.suppress(BaseException):
                                await read_task
                        if not pump_ok:
                            # Persist tool content from aborted streams only —
                            # incomplete artifacts ARE the failure evidence.
                            # (Guarded: successful streams used to persist twice,
                            # halving the artifact retention window — bughunt #6.)
                            persist_artifacts(tool_acc, req_id + "_partial")
                    if line_buffer.strip():
                        term = classify_terminal_line(line_buffer.strip())
                        if term == "done":
                            saw_done = True
                        elif term == "error":
                            saw_error_frame = True
                        for out_line in transform_sse_line(line_buffer, state):
                            await response.write((out_line + "\n").encode("utf-8"))

                    # Flush residue the tag-matcher still buffers (a trailing
                    # partial tag like "<thi" was previously dropped silently).
                    tail_content = state.finish()
                    if tail_content:
                        await response.write(("data: " + json.dumps(
                            {"choices": [{"index": 0,
                                          "delta": {"content": tail_content}}]}
                        ) + "\n\n").encode("utf-8"))

                    remember_reasoning(reasoning_mem, state.full_content, state.full_reasoning)
                    persist_artifacts(tool_acc, req_id)
                    # Tripwires + CLIENT-FACING repair: a healthy stream ends with
                    # [DONE]. Detection alone is not enough — a truncated stream
                    # must not look like a finished answer to the client
                    # (bughunt A3; the 2026-08-08 corpse rode exactly this gap).
                    if saw_error_frame:
                        trace_write(f"{time.time():.3f} {req_id} ALERT in-stream-error-frame")
                        print(f"[ALERT] {req_id} relayed an in-stream error frame "
                              f"(upstream exception serialized into the stream)", flush=True)
                    if pump_error or not saw_done:
                        trace_write(f"{time.time():.3f} {req_id} ALERT end-without-DONE")
                        print(f"[ALERT] {req_id} truncated stream "
                              f"({pump_error or 'ended without [DONE]'}) — "
                              f"error frame delivered to client", flush=True)
                        if not saw_error_frame:
                            await response.write(("data: " + json.dumps(
                                {"error": {"message": pump_error or
                                           "upstream stream ended unexpectedly (truncated response)",
                                           "type": "upstream_truncated", "code": 502}}
                            ) + "\n\n").encode("utf-8"))
                        await response.write(b"data: [DONE]\n\n")
                    trace_write(f"{time.time():.3f} {req_id} END")
                    await response.write_eof()
                    return response
            except ServerDisconnectedError:
                # llama-server closed the pooled keepalive connection exactly as
                # we reused it. Retry once on a fresh connection — but ONLY while
                # the client response has not started: a second prepare() would
                # write a second HTTP status line into the SSE body (bughunt #8).
                if _attempt or response_prepared:
                    raise
                trace_write(f"{time.time():.3f} retry after ServerDisconnectedError")
                await asyncio.sleep(0.2)


async def handle_anthropic_messages(request: web.Request) -> web.StreamResponse:
    """Anthropic Messages API endpoint. Translates to OpenAI internally."""
    upstream = request.app["upstream"].rstrip("/")
    keys = request.app["keys"]
    server_key = request.app["server_key"]
    session: ClientSession = request.app["session"]

    # Anthropic clients may send either `Authorization: Bearer ...` or
    # `x-api-key: ...` header. Accept both.
    auth_header = request.headers.get("Authorization", "")
    user_id = extract_user_id(auth_header, keys)
    if user_id is None:
        x_key = request.headers.get("x-api-key", "") or request.headers.get("X-Api-Key", "")
        if x_key:
            user_id = keys.get(x_key.strip())
    if user_id is None:
        return web.json_response(
            {"type": "error", "error": {"type": "authentication_error",
                                          "message": "Invalid API key"}},
            status=401,
        )

    try:
        body_bytes = await request.read()
        anthropic_body = json.loads(body_bytes) if body_bytes else {}
    except json.JSONDecodeError:
        return web.json_response(
            {"type": "error", "error": {"type": "invalid_request_error",
                                         "message": "Invalid JSON"}},
            status=400,
        )
    if not isinstance(anthropic_body, dict):
        return web.json_response(
            {"type": "error", "error": {"type": "invalid_request_error",
                                         "message": "request body must be a JSON object"}},
            status=400,
        )

    is_streaming = bool(anthropic_body.get("stream"))
    requested_model = anthropic_body.get("model", "qwopus-v3")

    # Translate Anthropic → OpenAI
    openai_body = anthropic_request_to_openai(anthropic_body)
    # Also strip reasoning from assistant history (same hygiene as native openai path)
    openai_body = reinject_thinking_in_request(openai_body)
    openai_bytes = json.dumps(openai_body).encode("utf-8")

    async with request.app["owner_lock"]:
        await maybe_swap_slot(request.app, session, upstream, server_key, user_id)

        url = f"{upstream}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {server_key}",
            "Content-Type": "application/json",
        }
        timeout = ClientTimeout(total=None, sock_read=600)

        async with session.post(url, headers=headers, data=openai_bytes,
                                timeout=timeout) as upstream_resp:
            resp_headers = {
                "Content-Type": "application/json",
            }
            if is_streaming:
                resp_headers["Content-Type"] = "text/event-stream"
                resp_headers["Cache-Control"] = "no-cache"

            if not is_streaming:
                data = await upstream_resp.read()
                if upstream_resp.status != 200:
                    # Forward error verbatim, wrapped in anthropic shape
                    return web.json_response(
                        {"type": "error",
                         "error": {"type": "api_error",
                                    "message": data.decode("utf-8", "replace")[:500]}},
                        status=upstream_resp.status,
                    )
                try:
                    oai_obj = json.loads(data)
                    ant_obj = openai_response_to_anthropic(oai_obj, requested_model)
                    return web.json_response(ant_obj, status=200)
                except json.JSONDecodeError:
                    return web.json_response(
                        {"type": "error",
                         "error": {"type": "api_error",
                                    "message": "upstream returned non-JSON"}},
                        status=502,
                    )

            # Streaming: translate OpenAI SSE → Anthropic SSE
            if upstream_resp.status != 200:
                body = await upstream_resp.read()
                return web.Response(
                    body=body, status=upstream_resp.status,
                    headers={"Content-Type": "application/json"},
                )

            response = web.StreamResponse(status=200, headers=resp_headers)
            response.enable_chunked_encoding()
            await response.prepare(request)

            builder = AnthropicStreamBuilder(
                message_id=f"msg_{uuid4()}",
                model=requested_model,
            )

            line_buffer = ""
            pump_error = None
            read_task = None
            try:
                while True:
                    if read_task is None:
                        read_task = asyncio.ensure_future(
                            upstream_resp.content.readany())
                    done, _ = await asyncio.wait({read_task}, timeout=10.0)
                    if not done:
                        # Protocol ping: keeps client watchdogs fed through long
                        # prefills — parity with the OpenAI path's ": hb"
                        # comment heartbeat (bughunt #4: this path had none, so
                        # Claude Code saw minutes of dead silence).
                        await response.write(
                            b'event: ping\ndata: {"type": "ping"}\n\n')
                        continue
                    chunk = read_task.result()
                    read_task = None
                    if not chunk:
                        break
                    line_buffer += chunk.decode("utf-8", errors="replace")
                    parts = line_buffer.split("\n")
                    line_buffer = parts.pop()
                    out_frames: list = []
                    for line in parts:
                        out_frames.extend(builder.feed(line.rstrip("\r")))
                    if out_frames:
                        await response.write("".join(out_frames).encode("utf-8"))
            except (ClientPayloadError, ClientConnectionError,
                    ServerTimeoutError, asyncio.TimeoutError) as e:
                pump_error = f"upstream died mid-stream: {type(e).__name__}"
            finally:
                if read_task is not None:
                    read_task.cancel()
                    with contextlib.suppress(BaseException):
                        await read_task
            if line_buffer.strip():
                out_frames = builder.feed(line_buffer.rstrip("\r"))
                if out_frames:
                    await response.write("".join(out_frames).encode("utf-8"))

            # A stream that never delivered [DONE] is a corpse. Previously this
            # path synthesized a normal end_turn + message_stop — the exact
            # 2026-08-08 signature presented as success (bughunt A3). Tell the
            # client before closing the message.
            if pump_error or not builder.closed:
                trace_write(f"{time.time():.3f} anthropic ALERT end-without-DONE")
                print(f"[ALERT] anthropic stream truncated "
                      f"({pump_error or 'ended without [DONE]'}) — "
                      f"error event delivered to client", flush=True)
                await response.write(builder._event("error", {
                    "type": "error",
                    "error": {"type": "api_error",
                              "message": pump_error or
                              "upstream stream ended unexpectedly (truncated response)"},
                }).encode("utf-8"))

            # Close the message
            closing = builder.close()
            if closing:
                await response.write("".join(closing).encode("utf-8"))
            await response.write_eof()
            return response


async def handle_passthrough(request: web.Request) -> web.StreamResponse:
    """Generic passthrough for non-chat-completion endpoints.
    Also validates client key + rewrites auth so clients can use their own key
    for /v1/models etc. Unauthenticated endpoints (/health) skip auth check."""
    upstream = request.app["upstream"].rstrip("/")
    keys = request.app["keys"]
    server_key = request.app["server_key"]
    session: ClientSession = request.app["session"]

    path = request.rel_url.path
    # /health and /metrics are public on llama-server; allow without auth rewrite
    public_paths = {"/health", "/metrics"}
    # Everything else must be authenticated AND allowlisted. The proxy owns
    # /slots (save/restore/erase is slot-pinning state — a client reaching it
    # desyncs ownership and can overwrite the other user's cache: bughunt A4),
    # and raw completion endpoints (/completion, /v1/completions, /infill)
    # bypass the owner lock and slot pinning entirely. None are forwardable.
    allowed_paths = {"/v1/models", "/models", "/props", "/tokenize",
                     "/detokenize", "/apply-template"}

    if path not in public_paths:
        user_id = extract_user_id(request.headers.get("Authorization", ""), keys)
        if user_id is None:
            return web.json_response(
                {"error": {"message": "Invalid API key", "type": "invalid_request_error"}},
                status=401,
            )
        if path not in allowed_paths:
            return web.json_response(
                {"error": {"message": f"endpoint {path} is not proxied "
                           f"(chat: /v1/chat/completions or /v1/messages)",
                           "type": "invalid_request_error"}},
                status=403,
            )

    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in _HOP_HEADERS}
    if path not in public_paths:
        headers["Authorization"] = f"Bearer {server_key}"

    body_bytes = await request.read()
    url = upstream + path
    if request.rel_url.query_string:
        url += "?" + request.rel_url.query_string

    async with session.request(
        request.method, url, headers=headers, data=body_bytes, timeout=ClientTimeout(total=120)
    ) as upstream_resp:
        body = await upstream_resp.read()
        resp_headers = {
            k: v for k, v in upstream_resp.headers.items()
            if k.lower() not in ("content-length", "content-encoding", "transfer-encoding")
        }
        return web.Response(body=body, status=upstream_resp.status, headers=resp_headers)


# ─── App setup ─────────────────────────────────────────────────────────────────


async def on_startup(app: web.Application):
    app["session"] = ClientSession()
    app["keys"] = load_keys()
    app["server_key"] = load_server_key()
    app["current_owner"] = None  # nobody owns the slot until first request
    app["owner_lock"] = asyncio.Lock()
    print(f"[proxy] loaded {len(app['keys'])} client key(s): "
          f"{sorted(set(app['keys'].values()))}", flush=True)


async def on_cleanup(app: web.Application):
    # Save current owner's cache before shutdown so they resume fast next boot
    if app["current_owner"]:
        await slot_save(app["session"], app["upstream"].rstrip("/"),
                        app["server_key"], app["current_owner"])
    await app["session"].close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default="http://127.0.0.1:8131")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8130)
    args = ap.parse_args()

    # 64 MiB request cap: a 409,600-token context arrives as ~2 MiB of JSON;
    # aiohttp's 1 MiB default rejected any session past ~230K tokens with 413.
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app["upstream"] = args.upstream
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    # OpenAI Chat Completions (Factory Droid / OpenAI clients)
    app.router.add_route("*", "/v1/chat/completions", handle_chat_completions)
    app.router.add_route("*", "/chat/completions", handle_chat_completions)
    # Anthropic Messages (Claude Code / anthropic SDK)
    app.router.add_route("POST", "/v1/messages", handle_anthropic_messages)
    # Fallback: passthrough (models list, health, metrics, etc.)
    app.router.add_route("*", "/{path:.*}", handle_passthrough)

    print(f"[proxy] listening on {args.host}:{args.port} -> {args.upstream}", flush=True)
    web.run_app(app, host=args.host, port=args.port, print=None, access_log=None)


if __name__ == "__main__":
    main()
