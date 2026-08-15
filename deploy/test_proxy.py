#!/usr/bin/env python3
"""Unit tests for proxy.py — verify tool_calls/tool_call_id preservation
and reasoning-strip correctness across request + response + SSE paths.

Run: python3 -m pytest /home/erol/.config/llama-tcq/test_proxy.py -v
Or:  python3 /home/erol/.config/llama-tcq/test_proxy.py
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from proxy import (
    reinject_thinking_in_request,
    transform_non_stream,
    transform_sse_line,
    split_thinking,
    StreamState,
    extract_user_id,
    anthropic_request_to_openai,
    openai_response_to_anthropic,
    AnthropicStreamBuilder,
)


def assert_eq(actual, expected, label):
    assert actual == expected, f"{label}: expected {expected!r}, got {actual!r}"


def test_tool_calls_preserved_in_request():
    """Assistant messages with tool_calls must keep tool_calls/tool_call_id
    after reasoning strip."""
    body = {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "what files are here"},
            {
                "role": "assistant",
                "content": "",
                "reasoning": "The user wants a file listing...",
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {"name": "list_files", "arguments": "{\"path\": \".\"}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_abc123",
                "content": "file1.txt\nfile2.txt",
            },
            {
                "role": "assistant",
                "content": "There are 2 files.",
                "reasoning_content": "Tool returned two files, summarizing.",
            },
        ]
    }
    out = reinject_thinking_in_request(body)

    asst1 = out["messages"][2]
    assert "reasoning" not in asst1, "reasoning must be stripped"
    assert "reasoning_content" not in asst1
    assert asst1["tool_calls"][0]["id"] == "call_abc123", "tool_calls.id must survive"
    assert asst1["tool_calls"][0]["function"]["name"] == "list_files"
    assert asst1["tool_calls"][0]["function"]["arguments"] == '{"path": "."}'

    tool_msg = out["messages"][3]
    assert_eq(tool_msg["role"], "tool", "tool role preserved")
    assert_eq(tool_msg["tool_call_id"], "call_abc123", "tool_call_id preserved")
    assert_eq(tool_msg["content"], "file1.txt\nfile2.txt", "tool content preserved")

    asst2 = out["messages"][4]
    assert "reasoning_content" not in asst2, "reasoning_content field removed from last assistant"
    # preserve_thinking contract (Qwen3.6): client-sent reasoning is re-inlined
    # as a <think> block prefix so the template sees prior-turn thinking.
    assert_eq(asst2["content"],
              "<think>\nTool returned two files, summarizing.\n</think>\n\nThere are 2 files.",
              "asst reasoning re-inlined into content (preserve_thinking)")

    print("OK test_tool_calls_preserved_in_request")


def test_user_message_untouched():
    """User messages must never be mutated."""
    body = {
        "messages": [
            {"role": "user", "content": "hi", "reasoning": "should not be stripped from user"},
        ]
    }
    out = reinject_thinking_in_request(body)
    assert_eq(out["messages"][0].get("reasoning"), "should not be stripped from user",
              "user.reasoning stays (proxy only strips from assistant)")
    print("OK test_user_message_untouched")


def test_empty_messages_no_crash():
    body = {}
    out = reinject_thinking_in_request(body)
    assert out == {}
    body2 = {"messages": []}
    out2 = reinject_thinking_in_request(body2)
    assert_eq(out2["messages"], [], "empty list handled")
    print("OK test_empty_messages_no_crash")


def test_split_thinking_basic():
    text = "<think>I should think about this</think>Hello world"
    reasoning, cleaned = split_thinking(text)
    assert_eq(reasoning, "I should think about this", "reasoning extracted")
    assert_eq(cleaned, "Hello world", "content cleaned")
    print("OK test_split_thinking_basic")


def test_split_thinking_no_think():
    text = "Hello world"
    reasoning, cleaned = split_thinking(text)
    assert_eq(reasoning, "", "no reasoning")
    assert_eq(cleaned, "Hello world", "unchanged")
    print("OK test_split_thinking_no_think")


def test_split_thinking_unterminated():
    text = "<think>Still thinking here"
    reasoning, cleaned = split_thinking(text)
    assert_eq(reasoning, "Still thinking here", "unterminated think captured")
    assert_eq(cleaned, "", "content empty")
    print("OK test_split_thinking_unterminated")


def test_split_thinking_multiple_blocks():
    text = "<think>first</think>mid text<think>second</think>end"
    reasoning, cleaned = split_thinking(text)
    assert_eq(reasoning, "first\n\nsecond", "blocks joined")
    assert_eq(cleaned, "mid textend", "mid text preserved")
    print("OK test_split_thinking_multiple_blocks")


def test_split_thinking_claude_style_tag():
    """Qwopus v3 (distilled from Claude) emits <thinking>...</thinking>."""
    text = "<thinking>reasoning here</thinking>The answer."
    reasoning, cleaned = split_thinking(text)
    assert_eq(reasoning, "reasoning here", "claude-style tag recognized")
    assert_eq(cleaned, "The answer.", "content cleaned")
    print("OK test_split_thinking_claude_style_tag")


def test_split_thinking_mixed_styles():
    """Handle both <think> and <thinking> in same message."""
    text = "<think>a</think>mid<thinking>b</thinking>end"
    reasoning, cleaned = split_thinking(text)
    assert_eq(reasoning, "a\n\nb", "both styles captured")
    assert_eq(cleaned, "midend", "content preserved")
    print("OK test_split_thinking_mixed_styles")


def test_stream_claude_style_tag_split_chunks():
    """Simulate <thinking> tag split across SSE chunks."""
    state = StreamState()
    nc, nr = state.process("<think")
    assert_eq(nc, "", "no content yet — partial tag buffered")
    nc, nr = state.process("ing>reason</thin")
    # After normalize, this becomes "reason</thin" with a partial close tag
    assert_eq(nr, "reason", "reasoning emitted")
    nc, nr = state.process("king>final answer")
    # </thinking> completed, then "final answer" is text
    assert_eq(nc, "final answer", "text after close")
    print("OK test_stream_claude_style_tag_split_chunks")


def test_response_rename_reasoning_content():
    """Upstream returns reasoning_content — proxy renames to reasoning."""
    resp = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Answer.",
                    "reasoning_content": "My reasoning here.",
                }
            }
        ]
    }
    out = transform_non_stream(resp)
    msg = out["choices"][0]["message"]
    assert "reasoning_content" not in msg, "reasoning_content removed"
    assert_eq(msg["reasoning"], "My reasoning here.", "renamed to reasoning")
    assert_eq(msg["content"], "Answer.", "content unchanged")
    print("OK test_response_rename_reasoning_content")


def test_response_extract_inline_think():
    """Upstream returns <think> inline in content — proxy extracts to reasoning."""
    resp = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "<think>Let me think</think>The answer is 42.",
                }
            }
        ]
    }
    out = transform_non_stream(resp)
    msg = out["choices"][0]["message"]
    assert_eq(msg["reasoning"], "Let me think", "extracted to reasoning")
    assert_eq(msg["content"], "The answer is 42.", "content cleaned")
    print("OK test_response_extract_inline_think")


def test_response_tool_calls_survive():
    """Response with tool_calls must keep tool_calls intact."""
    resp = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "<think>I need to call a tool</think>",
                    "tool_calls": [
                        {
                            "id": "call_xyz",
                            "type": "function",
                            "function": {"name": "search", "arguments": "{}"},
                        }
                    ],
                }
            }
        ]
    }
    out = transform_non_stream(resp)
    msg = out["choices"][0]["message"]
    assert_eq(msg["reasoning"], "I need to call a tool", "reasoning extracted")
    assert_eq(msg["content"], "", "content empty after extraction")
    assert_eq(msg["tool_calls"][0]["id"], "call_xyz", "tool_calls.id preserved")
    assert_eq(msg["tool_calls"][0]["function"]["name"], "search",
              "tool_calls.function.name preserved")
    print("OK test_response_tool_calls_survive")


def test_sse_rename_reasoning_content():
    """Streaming delta with reasoning_content gets renamed."""
    state = StreamState()
    chunk = {
        "choices": [{"delta": {"reasoning_content": "partial reasoning"}}]
    }
    line = f"data: {json.dumps(chunk)}"
    out = transform_sse_line(line, state)
    assert len(out) == 1
    parsed = json.loads(out[0][5:].strip())
    delta = parsed["choices"][0]["delta"]
    assert "reasoning_content" not in delta, "reasoning_content removed"
    assert_eq(delta["reasoning"], "partial reasoning", "renamed")
    print("OK test_sse_rename_reasoning_content")


def test_sse_tool_calls_delta_passthrough():
    """Streaming delta with tool_calls passes through untouched."""
    state = StreamState()
    chunk = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "function": {"name": "foo", "arguments": "{\"x\":"},
                        }
                    ]
                }
            }
        ]
    }
    line = f"data: {json.dumps(chunk)}"
    out = transform_sse_line(line, state)
    parsed = json.loads(out[0][5:].strip())
    delta = parsed["choices"][0]["delta"]
    assert_eq(delta["tool_calls"][0]["id"], "call_1", "tool_calls id preserved in SSE")
    assert_eq(delta["tool_calls"][0]["function"]["name"], "foo", "function name preserved")
    assert_eq(delta["tool_calls"][0]["function"]["arguments"], '{"x":',
              "partial arguments preserved")
    print("OK test_sse_tool_calls_delta_passthrough")


def test_sse_inline_think_split_across_chunks():
    """<think> opened in chunk 1, closed in chunk 3 — state machine must track."""
    state = StreamState()

    def emit(content):
        chunk = {"choices": [{"delta": {"content": content}}]}
        line = f"data: {json.dumps(chunk)}"
        out = transform_sse_line(line, state)
        return json.loads(out[0][5:].strip())["choices"][0]["delta"]

    d1 = emit("<think>partial")
    assert "reasoning" in d1
    assert_eq(d1["reasoning"], "partial", "first reasoning piece")
    assert "content" not in d1, "no content emitted while thinking"

    d2 = emit(" more thinking")
    assert_eq(d2["reasoning"], " more thinking", "continued reasoning")
    assert "content" not in d2

    d3 = emit("</think>Final answer.")
    assert_eq(d3.get("content"), "Final answer.", "content after close tag")
    print("OK test_sse_inline_think_split_across_chunks")


def test_sse_done_signal_passthrough():
    state = StreamState()
    out = transform_sse_line("data: [DONE]", state)
    assert_eq(out[0], "data: [DONE]", "[DONE] passthrough")
    print("OK test_sse_done_signal_passthrough")


def test_sse_empty_content_dropped_when_reasoning_present():
    """When delta would emit empty content + reasoning, content key should be dropped."""
    state = StreamState()
    chunk = {"choices": [{"delta": {"content": "<think>thinking</think>"}}]}
    line = f"data: {json.dumps(chunk)}"
    out = transform_sse_line(line, state)
    parsed = json.loads(out[0][5:].strip())
    delta = parsed["choices"][0]["delta"]
    assert "content" not in delta, "empty content dropped when reasoning present"
    assert_eq(delta["reasoning"], "thinking", "reasoning extracted")
    print("OK test_sse_empty_content_dropped_when_reasoning_present")


def test_multi_turn_tool_call_roundtrip():
    """Full 3-turn flow: assistant(tool_call) + tool_result + assistant(synth)."""
    body = {
        "messages": [
            {"role": "user", "content": "ls and cat foo.txt"},
            {
                "role": "assistant",
                "content": "",
                "reasoning": "Need to call ls first",
                "tool_calls": [{"id": "c1", "type": "function",
                                "function": {"name": "ls", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "foo.txt"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "Now cat it",
                "tool_calls": [{"id": "c2", "type": "function",
                                "function": {"name": "cat",
                                             "arguments": '{"f":"foo.txt"}'}}],
            },
            {"role": "tool", "tool_call_id": "c2", "content": "hello"},
        ]
    }
    out = reinject_thinking_in_request(body)
    assert len(out["messages"]) == 5, "no messages lost"
    assert_eq(out["messages"][1]["tool_calls"][0]["id"], "c1", "turn1 tool id")
    assert_eq(out["messages"][2]["tool_call_id"], "c1", "turn1 tool result id")
    assert_eq(out["messages"][3]["tool_calls"][0]["id"], "c2", "turn2 tool id")
    assert_eq(out["messages"][4]["tool_call_id"], "c2", "turn2 tool result id")
    assert "reasoning" not in out["messages"][1]
    assert "reasoning_content" not in out["messages"][3]
    print("OK test_multi_turn_tool_call_roundtrip")


def test_extract_user_id_valid():
    keys = {"sk-erol": "erol", "sk-brother": "brother"}
    assert_eq(extract_user_id("Bearer sk-erol", keys), "erol", "erol key maps")
    assert_eq(extract_user_id("Bearer sk-brother", keys), "brother", "brother key maps")
    print("OK test_extract_user_id_valid")


def test_extract_user_id_invalid():
    keys = {"sk-erol": "erol"}
    assert extract_user_id("Bearer sk-evil", keys) is None, "unknown key rejected"
    assert extract_user_id("", keys) is None, "empty header rejected"
    assert extract_user_id("Bearer ", keys) is None, "empty token rejected"
    assert extract_user_id("sk-erol", keys) is None, "missing Bearer prefix rejected"
    print("OK test_extract_user_id_invalid")


def test_extract_user_id_whitespace_tolerance():
    keys = {"sk-erol": "erol"}
    assert_eq(extract_user_id("Bearer sk-erol  ", keys), "erol", "trailing whitespace ok")
    print("OK test_extract_user_id_whitespace_tolerance")


# ─── Anthropic Messages translation ────────────────────────────────────────────


def test_anthropic_request_simple_text():
    body = {
        "model": "claude-3-5-sonnet",
        "system": "You are helpful.",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 100,
    }
    out = anthropic_request_to_openai(body)
    assert_eq(out["messages"][0], {"role": "system", "content": "You are helpful."},
              "system promoted")
    assert_eq(out["messages"][1], {"role": "user", "content": "Hello"}, "user passed")
    assert_eq(out["max_tokens"], 100, "max_tokens forwarded")
    print("OK test_anthropic_request_simple_text")


def test_anthropic_request_system_blocks():
    body = {
        "model": "claude",
        "system": [
            {"type": "text", "text": "Part A."},
            {"type": "text", "text": " Part B."},
        ],
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 10,
    }
    out = anthropic_request_to_openai(body)
    assert_eq(out["messages"][0]["content"], "Part A. Part B.", "system concatenated")
    print("OK test_anthropic_request_system_blocks")


def test_anthropic_request_tool_use_history():
    body = {
        "model": "claude",
        "messages": [
            {"role": "user", "content": "list files"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I'll check."},
                    {"type": "tool_use", "id": "toolu_A", "name": "ls",
                     "input": {"path": "."}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_A",
                     "content": "file1.txt"},
                ],
            },
        ],
        "max_tokens": 100,
    }
    out = anthropic_request_to_openai(body)
    # assistant message with content + tool_calls
    assistant = next(m for m in out["messages"] if m["role"] == "assistant")
    assert_eq(assistant["content"], "I'll check.", "assistant text preserved")
    assert_eq(assistant["tool_calls"][0]["id"], "toolu_A", "tool id preserved")
    assert_eq(assistant["tool_calls"][0]["function"]["name"], "ls", "tool name preserved")
    args = json.loads(assistant["tool_calls"][0]["function"]["arguments"])
    assert_eq(args["path"], ".", "tool input serialized")
    # tool_result converted to separate tool message
    tool_msg = [m for m in out["messages"] if m["role"] == "tool"][0]
    assert_eq(tool_msg["tool_call_id"], "toolu_A", "tool_call_id maps")
    assert_eq(tool_msg["content"], "file1.txt", "tool content preserved")
    print("OK test_anthropic_request_tool_use_history")


def test_anthropic_request_tools_translation():
    body = {
        "model": "claude",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 10,
        "tools": [{
            "name": "ls",
            "description": "list files",
            "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
        }],
        "tool_choice": {"type": "any"},
    }
    out = anthropic_request_to_openai(body)
    t = out["tools"][0]
    assert_eq(t["type"], "function", "tool wrapped as function")
    assert_eq(t["function"]["name"], "ls", "name")
    assert_eq(t["function"]["parameters"]["type"], "object", "input_schema → parameters")
    assert_eq(out["tool_choice"], "required", "any → required")
    print("OK test_anthropic_request_tools_translation")


def test_anthropic_request_tool_choice_specific():
    body = {
        "model": "claude",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 10,
        "tool_choice": {"type": "tool", "name": "ls"},
    }
    out = anthropic_request_to_openai(body)
    assert_eq(out["tool_choice"], {"type": "function", "function": {"name": "ls"}},
              "specific tool → function")
    print("OK test_anthropic_request_tool_choice_specific")


def test_anthropic_request_sampling_params():
    body = {
        "model": "claude",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 10,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "stop_sequences": ["END"],
    }
    out = anthropic_request_to_openai(body)
    assert_eq(out["temperature"], 0.7, "temperature")
    assert_eq(out["top_p"], 0.9, "top_p")
    assert_eq(out["top_k"], 40, "top_k")
    assert_eq(out["stop"], ["END"], "stop_sequences → stop")
    print("OK test_anthropic_request_sampling_params")


def test_anthropic_response_text_only():
    oai = {
        "id": "cc_1",
        "choices": [{
            "message": {"role": "assistant", "content": "Hello back."},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3},
    }
    ant = openai_response_to_anthropic(oai, "qwopus-v3")
    assert_eq(ant["type"], "message", "type")
    assert_eq(ant["role"], "assistant", "role")
    assert_eq(ant["model"], "qwopus-v3", "model")
    assert_eq(ant["content"], [{"type": "text", "text": "Hello back."}], "content")
    assert_eq(ant["stop_reason"], "end_turn", "stop → end_turn")
    # With no cached_tokens, everything goes into cache_creation
    assert_eq(ant["usage"]["input_tokens"], 0, "input_tokens=0 for full-process")
    assert_eq(ant["usage"]["cache_creation_input_tokens"], 10, "all 10 new to cache")
    assert_eq(ant["usage"]["cache_read_input_tokens"], 0, "no cache read")
    assert_eq(ant["usage"]["output_tokens"], 3, "output_tokens")
    print("OK test_anthropic_response_text_only")


def test_anthropic_response_with_cache():
    """When llama.cpp reports cached prefix, usage must split correctly."""
    oai = {
        "choices": [{
            "message": {"role": "assistant", "content": "Done."},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 50,
            "prompt_tokens_details": {"cached_tokens": 900},
        },
    }
    ant = openai_response_to_anthropic(oai, "q")
    u = ant["usage"]
    assert_eq(u["input_tokens"], 0, "input_tokens=0")
    assert_eq(u["cache_read_input_tokens"], 900, "cache hit surfaced")
    assert_eq(u["cache_creation_input_tokens"], 100, "new tokens = prompt - cached")
    assert_eq(u["output_tokens"], 50, "output preserved")
    print("OK test_anthropic_response_with_cache")


def test_anthropic_response_with_thinking_inline():
    oai = {
        "id": "cc_2",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "<think>Let me consider</think>The answer is 42.",
            },
            "finish_reason": "stop",
        }],
    }
    ant = openai_response_to_anthropic(oai, "q")
    blocks = ant["content"]
    assert_eq(blocks[0]["type"], "thinking", "thinking block first")
    assert_eq(blocks[0]["thinking"], "Let me consider", "thinking text")
    assert_eq(blocks[1]["type"], "text", "text second")
    assert_eq(blocks[1]["text"], "The answer is 42.", "text content")
    print("OK test_anthropic_response_with_thinking_inline")


def test_anthropic_response_tool_use():
    oai = {
        "id": "cc_3",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_ls",
                    "type": "function",
                    "function": {"name": "ls", "arguments": '{"path": "/tmp"}'},
                }],
            },
            "finish_reason": "tool_calls",
        }],
    }
    ant = openai_response_to_anthropic(oai, "q")
    tool_block = next(b for b in ant["content"] if b["type"] == "tool_use")
    assert_eq(tool_block["name"], "ls", "tool name")
    assert_eq(tool_block["id"], "call_ls", "tool id")
    assert_eq(tool_block["input"], {"path": "/tmp"}, "tool input parsed")
    assert_eq(ant["stop_reason"], "tool_use", "stop → tool_use")
    print("OK test_anthropic_response_tool_use")


def test_anthropic_response_reasoning_field():
    oai = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "OK",
                "reasoning_content": "Thought about it.",
            },
            "finish_reason": "stop",
        }],
    }
    ant = openai_response_to_anthropic(oai, "q")
    assert_eq(ant["content"][0]["type"], "thinking", "reasoning → thinking block")
    assert_eq(ant["content"][0]["thinking"], "Thought about it.", "reasoning text")
    assert_eq(ant["content"][1]["type"], "text", "text follows")
    print("OK test_anthropic_response_reasoning_field")


def test_anthropic_stream_basic_text():
    """A simple text-only stream: text deltas → text_delta events, one content block."""
    b = AnthropicStreamBuilder("msg_test", "qwopus")

    def oai_chunk(content):
        return "data: " + json.dumps({"choices": [{"delta": {"content": content}}]})

    frames = b.feed(oai_chunk("Hello"))
    # Expect message_start + content_block_start(text) + content_block_delta
    joined = "".join(frames)
    assert "event: message_start" in joined, "message_start emitted"
    assert "content_block_start" in joined and '"type": "text"' in joined, "text block opened"
    assert '"text_delta"' in joined and "Hello" in joined, "text_delta payload"

    frames2 = b.feed(oai_chunk(" world"))
    joined2 = "".join(frames2)
    # Should NOT re-start a block; just another delta
    assert "content_block_start" not in joined2, "no new block"
    assert "text_delta" in joined2 and "world" in joined2, "continued text"

    closing = b.close()
    joined3 = "".join(closing)
    assert "content_block_stop" in joined3, "block closed"
    assert "message_delta" in joined3, "message_delta emitted"
    assert "message_stop" in joined3, "message_stop emitted"
    print("OK test_anthropic_stream_basic_text")


def test_anthropic_stream_thinking_then_text():
    b = AnthropicStreamBuilder("msg_t", "q")

    # Chunk 1: inline <think> start
    f1 = "".join(b.feed("data: " + json.dumps({
        "choices": [{"delta": {"content": "<think>reasoning"}}]
    })))
    assert "message_start" in f1
    assert '"type": "thinking"' in f1, "thinking block opened"
    assert '"thinking_delta"' in f1 and "reasoning" in f1

    # Chunk 2: close think, start text
    f2 = "".join(b.feed("data: " + json.dumps({
        "choices": [{"delta": {"content": "</think>Answer."}}]
    })))
    assert "content_block_stop" in f2, "thinking closed"
    assert '"type": "text"' in f2, "text block opened"
    assert "Answer." in f2

    closing = "".join(b.close())
    assert "message_stop" in closing
    print("OK test_anthropic_stream_thinking_then_text")


def test_anthropic_stream_tool_call():
    b = AnthropicStreamBuilder("msg_tc", "q")

    # First chunk: tool_calls id + name
    f1 = "".join(b.feed("data: " + json.dumps({
        "choices": [{"delta": {"tool_calls": [{
            "index": 0, "id": "call_1",
            "function": {"name": "search", "arguments": ""}
        }]}}]
    })))
    assert "message_start" in f1
    assert '"type": "tool_use"' in f1, "tool_use block opened"
    assert '"name": "search"' in f1

    # Second chunk: partial args
    f2 = "".join(b.feed("data: " + json.dumps({
        "choices": [{"delta": {"tool_calls": [{
            "index": 0, "function": {"arguments": '{"q":'}
        }]}}]
    })))
    assert '"input_json_delta"' in f2, "partial json as input_json_delta"
    assert '{\\"q\\":' in f2 or '\"q\":' in f2, "partial args forwarded"

    # Finish with tool_calls
    f3 = "".join(b.feed("data: " + json.dumps({
        "choices": [{"delta": {"tool_calls": [{
            "index": 0, "function": {"arguments": ' "foo"}'}
        }]}, "finish_reason": "tool_calls"}]
    })))
    assert "input_json_delta" in f3

    closing = "".join(b.close())
    assert '"stop_reason": "tool_use"' in closing, "stop_reason mapped"
    print("OK test_anthropic_stream_tool_call")


def test_anthropic_stream_done_signal():
    b = AnthropicStreamBuilder("msg_d", "q")
    b.feed("data: " + json.dumps({"choices": [{"delta": {"content": "hi"}}]}))
    closing = "".join(b.feed("data: [DONE]"))
    # [DONE] from upstream triggers close()
    assert "content_block_stop" in closing
    assert "message_stop" in closing
    # Further feeds are no-ops
    assert b.feed("data: garbage") == []
    print("OK test_anthropic_stream_done_signal")


# NOTE: the TESTS registry + __main__ runner live at the END of this file.
# They used to sit here — every test defined below this point was silently
# skipped in `python3 test_proxy.py` mode, which reported 37/37 while the
# suite had 42 (2026-08-15 bughunt A6).


# ─── Stream-termination tripwires + real-traffic capture (2026-08-09) ─────────

import proxy as proxy_mod
from proxy import classify_terminal_line, capture_tools_request


def test_classify_done():
    assert_eq(classify_terminal_line("data: [DONE]"), "done", "DONE terminator")


def test_classify_error_frame():
    line = 'data: {"error":{"code":500,"message":"Failed to parse input at pos 402: x","type":"server_error"}}'
    assert_eq(classify_terminal_line(line), "error", "error frame")


def test_classify_normal_delta_not_flagged():
    line = 'data: {"choices":[{"delta":{"content":"data: {\\"error\\" is just text"}}]}'
    assert_eq(classify_terminal_line(line), None, "content delta")
    assert_eq(classify_terminal_line(": hb"), None, "heartbeat comment")


def test_capture_tools_request_writes_per_suite_file(tmp_path, monkeypatch):
    monkeypatch.setattr(proxy_mod, "CAPTURES_DIR", str(tmp_path))
    body = {"tools": [{"type": "function", "function": {"name": "write_file"}}],
            "messages": [{"role": "user", "content": "hi"}]}
    capture_tools_request(body, "erol")
    capture_tools_request(body, "erol")  # same suite → same file, no growth
    files = list(tmp_path.iterdir())
    assert_eq(len(files), 1, "one file per (user, suite)")
    body2 = {"tools": [{"type": "function", "function": {"name": "other"}}]}
    capture_tools_request(body2, "erol")
    assert_eq(len(list(tmp_path.iterdir())), 2, "different suite → new file")


def test_capture_strips_image_payloads(tmp_path, monkeypatch):
    from proxy import strip_image_payloads
    monkeypatch.setattr(proxy_mod, "CAPTURES_DIR", str(tmp_path))
    body = {"tools": [{"type": "function", "function": {"name": "t"}}],
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "what is this"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 5000}}]},
                {"role": "user", "content": "plain text untouched"}]}
    stripped = strip_image_payloads(body)
    assert_eq(stripped["messages"][0]["content"][1]["image_url"]["url"].startswith("<stripped"),
              True, "image payload replaced")
    assert_eq(body["messages"][0]["content"][1]["image_url"]["url"][:15],
              "data:image/png;", "original body untouched")
    assert_eq(stripped["messages"][1]["content"], "plain text untouched", "string content untouched")


# ─── 2026-08-15 bughunt regression tests ──────────────────────────────────────

from proxy import strip_image_payloads, StreamState


def test_classify_crlf_and_nospace():
    assert_eq(classify_terminal_line("data: [DONE]\r"), "done", "CRLF-framed DONE")
    assert_eq(classify_terminal_line('data:{"error":{"message":"x"}}'), "error",
              "data: without space")


def test_classify_reordered_error_object():
    line = 'data: {"code":500,"error":{"message":"boom"}}'
    assert_eq(classify_terminal_line(line), "error", "top-level error key, any order")
    line2 = 'data: {"choices":[{"delta":{"content":"the \\"error\\" word"}}]}'
    assert_eq(classify_terminal_line(line2), None, "content mentioning error not flagged")


def test_stream_literal_think_after_content():
    s = StreamState()
    _, r = s.process("<think>plan</think>Here is the doc: ")
    assert_eq(r, "plan", "leading think extracted")
    c2, r2 = s.process("use <think> tags in your template")
    assert_eq(r2, "", "literal tag after content produces no reasoning")
    assert_eq(c2, "use <think> tags in your template", "literal tag stays in content")
    c3, _ = s.process(" and <thinking> too")
    assert_eq(c3, " and <thinking> too", "thinking-variant untouched after content")


def test_stream_finish_flushes_partial_tag():
    s = StreamState()
    c, _ = s.process("answer ends with <thi")
    assert_eq(c, "answer ends with <thi", "no silent tail hold after real content")
    s2 = StreamState()
    c2, _ = s2.process("<thi")
    assert_eq(c2, "", "pure tag prefix held")
    assert_eq(s2.finish(), "<thi", "finish() flushes the buffered prefix")


def test_swap_has_no_filesystem_existence_check():
    """Bughunt A1: the proxy peeked a hardcoded slots/ dir while the server
    saved to slots-long/ — restore was silently dead. Guard the fix."""
    import inspect
    src = inspect.getsource(proxy_mod.maybe_swap_slot)
    assert_eq("os.path.exists" in src, False, "no filesystem peek in swap")
    assert_eq('join(CONFIG_DIR, "slots")' in src, False, "no hardcoded slots dir")


def test_strip_leaves_long_text_alone():
    long_code = "def f():\n    return 1\n" * 500  # >4 KiB, not base64/data:
    body = {"messages": [{"role": "user", "content": long_code}]}
    out = strip_image_payloads(body)
    assert_eq(out["messages"][0]["content"], long_code, "real text never stripped")
    out2 = strip_image_payloads({"m": [{"source": {"data": "A" * 5000}}]})
    assert_eq(out2["m"][0]["source"]["data"].startswith("<stripped"), True,
              "anthropic-shape base64 stripped")


def test_anthropic_image_blocks_translated():
    body = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "what is this"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                     "data": "AAAA"}}]}]}
    out = anthropic_request_to_openai(body)
    content = out["messages"][0]["content"]
    assert_eq(isinstance(content, list), True, "structured content for vision")
    assert_eq(content[0]["type"], "text", "text part first")
    assert_eq(content[1]["image_url"]["url"], "data:image/png;base64,AAAA",
              "image translated, not dropped")


# ─── Script-mode fixture shim (pytest supplies real fixtures under pytest) ────

class _ScriptMonkeypatch:
    def __init__(self):
        self._saved = []

    def setattr(self, obj, name, value):
        self._saved.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def undo(self):
        for obj, name, value in reversed(self._saved):
            setattr(obj, name, value)


def _run_with_fixtures(test_fn):
    import inspect
    import pathlib
    import tempfile
    kwargs = {}
    mp = _ScriptMonkeypatch()
    tmpdir = None
    try:
        for pname in inspect.signature(test_fn).parameters:
            if pname == "tmp_path":
                tmpdir = tempfile.TemporaryDirectory()
                kwargs["tmp_path"] = pathlib.Path(tmpdir.name)
            elif pname == "monkeypatch":
                kwargs["monkeypatch"] = mp
        test_fn(**kwargs)
    finally:
        mp.undo()
        if tmpdir is not None:
            tmpdir.cleanup()


TESTS = [
    test_tool_calls_preserved_in_request,
    test_user_message_untouched,
    test_empty_messages_no_crash,
    test_split_thinking_basic,
    test_split_thinking_no_think,
    test_split_thinking_unterminated,
    test_split_thinking_multiple_blocks,
    test_split_thinking_claude_style_tag,
    test_split_thinking_mixed_styles,
    test_stream_claude_style_tag_split_chunks,
    test_response_rename_reasoning_content,
    test_response_extract_inline_think,
    test_response_tool_calls_survive,
    test_sse_rename_reasoning_content,
    test_sse_tool_calls_delta_passthrough,
    test_sse_inline_think_split_across_chunks,
    test_sse_done_signal_passthrough,
    test_sse_empty_content_dropped_when_reasoning_present,
    test_multi_turn_tool_call_roundtrip,
    test_extract_user_id_valid,
    test_extract_user_id_invalid,
    test_extract_user_id_whitespace_tolerance,
    # Anthropic translation
    test_anthropic_request_simple_text,
    test_anthropic_request_system_blocks,
    test_anthropic_request_tool_use_history,
    test_anthropic_request_tools_translation,
    test_anthropic_request_tool_choice_specific,
    test_anthropic_request_sampling_params,
    test_anthropic_response_text_only,
    test_anthropic_response_with_cache,
    test_anthropic_response_with_thinking_inline,
    test_anthropic_response_tool_use,
    test_anthropic_response_reasoning_field,
    test_anthropic_stream_basic_text,
    test_anthropic_stream_thinking_then_text,
    test_anthropic_stream_tool_call,
    test_anthropic_stream_done_signal,
    # Tripwires + capture (previously unreachable in script mode: bughunt A6)
    test_classify_done,
    test_classify_error_frame,
    test_classify_normal_delta_not_flagged,
    test_capture_tools_request_writes_per_suite_file,
    test_capture_strips_image_payloads,
    # 2026-08-15 bughunt regressions
    test_classify_crlf_and_nospace,
    test_classify_reordered_error_object,
    test_stream_literal_think_after_content,
    test_stream_finish_flushes_partial_tag,
    test_swap_has_no_filesystem_existence_check,
    test_strip_leaves_long_text_alone,
    test_anthropic_image_blocks_translated,
]


if __name__ == "__main__":
    failed = 0
    for t in TESTS:
        try:
            _run_with_fixtures(t)
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    sys.exit(1 if failed else 0)
