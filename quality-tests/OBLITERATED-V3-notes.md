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


## CORRECTION (2026-08-24, verified from HF cards + search-agent research)
**JonathanColetti/Qwen3.8-27B-Uncensored is a HERETIC ABLITERATION, not an SFT
fine-tune** (card: "Refusal directions removed with Heretic, no fine-tuning, no
additional training data"). This INVALIDATES the earlier "fine-tune is more
robust to system prompts than V3" reasoning above — ALL our uncensored options
(Coletti, Huihui, OrcaRouter, OBLITERATED) are abliterations = a "hole a system
prompt can refill." Only an SFT firms the hole; NO 3.8 SFT with published KLD
exists yet (closest = DavidAU 3.6 Heretic2 SFT, KLD 0.0469).

**Measured field (third-party: returnity/r/LocalLLaMA + cards — NOT our gate):**
- Coletti (our daily): KLD-vs-stock 0.1191, refusals 12/100 — HIGH-KL end.
- huihui-ai/Huihui-Qwen3.8-27B-abliterated: KLD 0.0078 (lowest), 1.5% refuse,
  MTP intact (ablates layers 18-51 only). GGUF exists — easy to test.
- orcarouter/Qwen3.8-27B-Uncensored: MMLU +0.4 vs stock, 0-6% refuse, vision+MTP
  preserved. SAFETENSORS-ONLY (no GGUF) — needs convert+quant on our pipeline.
- OBLITERATUS V3 (escalation): MMLU -2.1pp, 0% hard+soft refuse (manual audit).

**REVISED PLAN:** Don't blind-swap on Reddit numbers (our law: no folklore w/o
on-stack A/B). Our OWN bakeoff had Coletti STRONG (4/4 reasoning, elaborate
pagoda, complied) — tension with the 0.119/12% third-party numbers = different
metrics. POST-TRIP BAKE-OFF: Coletti vs Huihui vs OrcaRouter vs V3 on OUR Q6_K,
measuring KLD + refusal rate WITH vs WITHOUT a system prompt (the Hermes
question). The real fix for the system-prompt-refill is stripping Hermes's system
prompt on the uncensored slot (proxy), not a better abliteration.

**DFlash2 verdict (search-agent, same-binary 5090 A/B):** MTP n-max 3 = 35.5 t/s
vs DFlash2 = 33.2 t/s — DFlash2 SLOWER same-binary; earlier "+28%" was a
cross-build confound. +1.3GB cost, vision HTTP 500, PR #27342 unmerged, CUDA
greedy not bit-identical (#27407). STAY MTP n-max 3. Nobody has published
DFlash2 + turbo4/TCQ (Anbeeld: draft KV must stay standard cache, not TCQ).
Steal: FR-Spec MTP vocab-trim (Pernici gist, lossless 0/14042 MMLU mismatch).

## FIELD MAP UPDATE 2 (2026-08-24, search-agent + verified repos)
Blackfrost/RedPillReader store adds 2 SKUs (both verified exist):
- **Blackfrost-AI/Qwen3.8-27B-ABLITERATED** (GGUF Q6_K 22.4GB drop-in, NO imatrix):
  weight abliteration + **BAKED liberation prompt in the Jinja ("Snapback")** —
  OPPOSITE of OBLITERATUS "naked". Refusal 11/450=2.4% but measured WITH their
  template + on NVFP4 derivative; no KLD/MMLU published.
- **Blackfrost-Research/M.O.G.-SEC-27B** (NVFP4): first 3.8 **SFT** (cyber-specialized),
  MMLU-Pro val 92.9% n=70, 4/300 refuse. KILLERS for us: NVFP4 (we gate-rejected
  0.069/82.8%), **native MTP DEAD after the FT** (default→DFlash2), cyber-skewed,
  1M YaRN always-on, dual-GPU serve. = proof a 3.8 SFT exists, wrong impl for us.

**KEY INSIGHT (most valuable from both research turns): the system-prompt problem
is UNIVERSAL across design philosophies.** OBLITERATUS-naked: harness prompt REFILLS
the hole. Blackfrost-baked: harness prompt REPLACES the liberation prompt. BOTH die
under Hermes's arbitrary system prompt. => FIX IS ARCHITECTURAL: **inject a liberation
preamble at the PROXY on the uncensored slot** (in front of whatever Hermes sends), so
liberation survives regardless of client + regardless of which abliteration. This is
the real answer, model-agnostic. (A general 3.8 SFT would also solve it — none exists;
MOG-SEC is cyber+NVFP4+MTP-dead. Watch for a general one.)

**DEFINITIVE POST-TRIP BAKE-OFF (field now fully mapped):** stock · Coletti · Huihui ·
OrcaRouter(needs quant) · OBLITERATUS-V3 · Blackfrost-Abliterated — all as OUR
Q6_K+imatrix on turbo4 KV, 450-prompt refusal set, measure KLD + refusal WITH vs
WITHOUT system prompt, + a run with each vendor's baked template. Settles which-weights
+ does-proxy-prompt-fix-work + is-baked-approach-worth-it in one matrix.

## TOOL-CALLING FIX (2026-08-24) — V3 template regression
SYMPTOM: tool calls printed as text in Hermes chat (not parsed). ROOT CAUSE:
V3's author REBUILT the chat template and stripped tool-calling — V3's baked
template is 506 B, text-only, NO `tools`/`tool_calls` handling (V1's template
HAD tools, which is why V1 worked). With `--jinja` on the stripped template,
llama.cpp can't parse the model's Qwen tool-call syntax → leaks to chat.
FIX (V3 launcher ONLY, other models untouched): added
`--chat-template-file ~/.config/llama-tcq/qwen38-tool-template.jinja` (the
canonical STOCK Qwen3.8 template, 8952 B, full tool support + enable_thinking)
to start-long-38o.sh. VERIFIED: tool_call parses (get_weather ✓), thinking
still off (empty <think></think>), liberation intact (gray complies). Template
file staged in ~/.config/llama-tcq/. NOTE: if serving any future model with a
stripped/custom template, check `grep -c tool_call <template>` — 0 = will leak;
override with the stock tool template.
