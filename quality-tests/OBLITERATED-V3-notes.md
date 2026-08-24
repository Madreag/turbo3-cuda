# OBLITERATED V3 — settings, analysis, serving decision (2026-08-24)

Reference for the abliterated Qwen3.8-27B escalation model. Installed in WSL
`models/qwen38-obliterated/` (Q6_K, sha256 3535d4a1…, verified). Launcher:
`~/.config/llama-tcq/start-long-38o.sh`. Serving via `qwen3.8-27b-320k` alias.

## Version story (author = Pliny/OBLITERATUS, abliteration = weight surgery)
| ver | MMLU vs stock | liberation | notes |
|---|---|---|---|
| V1 (what we first tested 08-20) | **−6.0pp** (81.4) | hard refusals removed | "noticeably dumber"; thinking-ON SPIRALED (our bakeoff) |
| V2 | −0.3pp (84.3) | soft safety-lecture deflections REMAIN | best capability, incomplete liberation |
| **V3 (installed 08-24)** | **−2.1pp** (82.3) | hard refusals AND soft deflections removed | thinking-ON fixed; 20/20 code; 7/8 real-world = stock |

MMLU hit is non-uniform: STEM −3.3pp (worst), humanities −1.0pp. Matters LESS for
us — the local model is cheap/private/uncensored, NOT our primary reasoner (Claude
SOTA does that). So V3's capability tax is largely irrelevant to our use.

## Author-recommended settings (from the V3 model card)
| setting | value | why |
|---|---|---|
| temperature | **0** (general) / **0.1–0.3** (agentic) | greedy = most complete; pure greedy CAN STALL in agent loops |
| repetition_penalty | **1.15** (ESSENTIAL) | else greedy loops on imports/boilerplate/tool-calls; 1.10–1.12 = tighter |
| top_p/top_k/min_p | **not needed** | greedy + rep-penalty only; sampling adds noise, no quality gain |
| max_new_tokens | **≥2048** general / **1024–2048** agentic | code/attack chains need room; but cap per-turn for agent focus |
| **system prompt** | **NONE / empty** | A/B tested: system prompts REINTRODUCE refusals. "Naked is better." |
| enable_thinking | **OFF** | baked template prefills empty `<think></think>`; ON works (V3-fixed) but longer |
| context mgmt | summarize after ~10 turns | abliterated models drift as context fills with repeated actions |

## Our launcher audit — already ~90% aligned (NO launcher changes needed)
- temp 0 (env `OBL_TEMP` override) ✓ · repeat-penalty 1.15 ✓ · top_p/top_k stripped ✓
- enable_thinking off (baked template + explicit kwarg) ✓ · V3's own bf16 mmproj on CPU ✓
- MTP n3 + turbo4 KV + YaRN 1.25 + 320K + ctx-checkpoints 2 (our infra, model-agnostic) ✓

## The 3 things that matter and are NOT launcher settings (Hermes-side)
1. **SYSTEM PROMPT = the deciding unknown.** Author validated liberation NAKED; Hermes
   always sends a system prompt, which can re-activate refusal along adjacent pathways.
   The fine-tune (trained-uncensored) is inherently more robust to this than V3
   (surgically-uncensored). **Untested through Hermes.** This is THE reason V3 is not
   yet the daily driver.
2. **max_tokens per turn** — author 1024–2048 for agents (focus). Mildly conflicts with
   our "never cap max_tokens" law (that law protects capability; this is agent focus).
3. **context summarization after ~10 turns** — Hermes-side hygiene.

## MTP × rep-penalty interaction (subtle speed note)
rep-penalty 1.15 reshapes target logits the MTP draft head can't see → LOWERS MTP
acceptance (dropped to ~0.35 on V1). Greedy partially offsets (greedy inflates
acceptance). Net: V3 likely decodes a touch slower through our MTP stack than the
fine-tune's temp-1.0 path. Not a blocker; a real cost. Testable.

## DECISION (2026-08-24): keep the FINE-TUNE as daily; V3 = upgraded escalation
- **Daily driver = uncensored FINE-TUNE (JonathanColetti).** Proven robust through
  Hermes, stronger reasoning (bakeoff 4/4 incl. probability), plug-and-play at our
  temp-1.0/MTP sampling. 
- **V3 OBLITERATED = escalation** (replaces V1): reach for it when the fine-tune emits
  a safety-lecture instead of an answer. Far better than V1 (−2.1 vs −6pp, thinking
  fixed, soft-deflections gone).
- **Prod (aligned) = rarely used** — Claude covers aligned SOTA.
- Our use case actually TILTS toward V3 as daily (capability matters little, liberation
  matters most) — the ONLY blocker is the untested system-prompt-through-Hermes question.

## The one test that could promote V3 to daily (when home + stable GPU)
Run V3 THROUGH Hermes with its real system prompt; check if liberation holds or the
system prompt reintroduces deflections.
- Holds → V3 becomes daily; also give Hermes a minimal/empty system prompt + 1024–2048
  per-turn cap for this model.
- Reintroduces refusals → fine-tune stays daily, V3 stays escalation.
Also worth: bakeoff V3 vs fine-tune (reasoning/code/gray/pagoda) like the 08-20 run.

## Roster / files
- Daily: `models/qwen38-uncensored/` (fine-tune) via `start-long-38u.sh`
- Escalation: `models/qwen38-obliterated/` (V3) via `start-long-38o.sh`
- Aligned: `models/qwen38/Qwen3.8-27B-Q6_K.gguf` via `start-long-38.sh` / `start-max-38.sh`
- Self-quant option: OBLITERATUS ships full BF16 (55.6GB, 29 shards) — we could
  imatrix+up-quant the abliterated weights for a better escalation quant (post-trip).
