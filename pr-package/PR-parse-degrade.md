# server: degrade gracefully when chat parsing throws instead of corpsing the stream

`common_chat_parse` throws when a strict-format parser cannot consume the
generated text (e.g. a model emitting tool-call syntax without a grammar, or
a parser that fails at offset 0 on every streaming delta). Today that
exception unwinds into the SSE layer and terminates an otherwise-healthy
generation with an in-stream error frame; for non-streaming requests the
whole response is lost.

Production forensics that motivated this: a lazily-grammared model emitted a
malformed tool block ~1/128 tool calls; the final parse threw, the client
received a truncated stream presented as an error, and the actual generated
content (recoverable raw text) was discarded.

Changes:
- wrap both the incremental (partial) and final parse in try/catch;
- on partial failure: keep the last good incremental message (or raw text if
  none) so streaming continues;
- on final failure: fall back to the last good parse, but if it covers only a
  prefix of `generated_text` (and no tool calls were parsed), return the raw
  content instead — never silently drop the tail;
- guard the continuation-seed parse in the ctor the same way.

No behavior change when parsing succeeds.
