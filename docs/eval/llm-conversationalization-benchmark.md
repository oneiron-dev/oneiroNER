# LLM Conversationalization Benchmark Report

> Date: 2026-02-12
> Purpose: Evaluate free/low-cost LLMs for Task 6 (Phase A Labeling — Conversationalize B2NERD)
> Context: [Task 6 spec](../plan/task-6-labeling.md)

## Objective

Task 6 requires rewriting B2NERD passages as 2–4 turn dialogues while preserving gold entity annotations verbatim. This benchmark evaluates 6 LLMs on their ability to:

1. **Preserve entities** — include every gold entity surface string exactly as written
2. **Follow format** — return valid JSON without markdown fences or preamble text
3. **Respect constraints** — stay within 2–4 turns, same language as input
4. **Avoid hallucination** — not invent entities beyond the gold list

## Models Tested

| Model | CLI Tool | Access | Default Model ID |
|-------|----------|--------|------------------|
| Kimi K2.5 | `opencode` | Free | opencode/kimi-k2.5-free |
| GLM-5 | `kilocode` | Free | kilo/z-ai/glm-5:free |
| Trinity | OpenRouter `curl` | Free | arcee-ai/trinity-large-preview:free |
| Gemini | `gemini` | Free (rate-limited) | Gemini (default) |
| GPT-5.3 Codex | `codex` | Paid (ChatGPT Plus) | gpt-5.3-codex |
| Sonnet 4.5 | `claude` | Paid | claude-sonnet-4-5 |

### CLI Invocations

```bash
# Kimi K2.5
echo "$PROMPT" | opencode run --format json --model opencode/kimi-k2.5-free -

# GLM-5
echo "$PROMPT" | kilocode run --format json --model kilo/z-ai/glm-5:free -

# Trinity (via OpenRouter)
curl -s https://openrouter.ai/api/v1/chat/completions \
  -H "Authorization: Bearer $OPENROUTER_KEY" \
  -H "Content-Type: application/json" \
  -d "$(jq -n --arg p "$PROMPT" \
    '{model:"arcee-ai/trinity-large-preview:free",
      messages:[{role:"user",content:$p}],max_tokens:1024}')" \
  | jq -r '.choices[0].message.content'

# Gemini
echo "$PROMPT" | gemini -p "" --yolo

# Codex
echo "$PROMPT" | codex exec --skip-git-repo-check -

# Sonnet
echo "$PROMPT" | claude -p --model sonnet --allowedTools ""
```

## Test Design

### Prompt Template

```
You are a data augmentation assistant. Given a passage and its gold entity
annotations, rewrite the passage as a natural 2-4 turn dialogue between
speakers A and B.

RULES:
1. You MUST include each entity surface string at least once, exactly as written.
2. Do NOT paraphrase, abbreviate, or modify entity names.
3. Do NOT invent new entities.
4. Return ONLY valid JSON, no markdown fences.
5. The dialogue MUST be in the same language as the input passage. (CJK only)
```

### Test Passages

Four test passages covering all target languages, each with 5 gold entities:

| Lang | Entities | Topic |
|------|----------|-------|
| EN | March 2024, Dr. Yuki Tanaka, University of Tokyo, RIKEN, meditation | Neural plasticity study |
| JA | 2024年3月, 東京大学, 田中優希博士, 理化学研究所, 瞑想 | Same (Japanese) |
| ZH | 2024年3月, 北京大学, 王明教授, 自然, 中国科学院 | Quantum computing paper |
| KO | 2024년 3월, 서울대학교, 김지현 교수, 한국과학기술원, 딥러닝 | AI ethics research |

### Evaluation Criteria

- **No fences**: Raw JSON output, no ` ```json ``` ` wrapping
- **Turn count**: 2–4 turns as specified
- **Entity preservation**: All 5 gold surfaces appear verbatim in dialogue text
- **Location accuracy**: `entity_locations` correctly maps surfaces to turn indices
- **No hallucination**: No entities in `entity_locations` beyond the gold list

## Results

### English (EN)

| Model | No fences | Turns | Entities 5/5 | Locations | Notes |
|-------|-----------|-------|-------------|-----------|-------|
| Kimi K2.5 | ✓ | 4 ✓ | ✓ | ✓ | Clean |
| GLM-5 | ✗ | 4 ✓ | ✓ | ✓ | Fences |
| Trinity | ✗ | 5 ✗ | ✓ | ✓ | Fences + exceeded turn count |
| Gemini | ✓ | 3 ✓ | ✓ | ✓ | Clean |
| Codex | ✓ | 3 ✓ | ✓ | ✓ | Clean |
| Sonnet | ✗ | 4 ✓ | ✓ | ✓ | Fences; confused by project context on first attempt |

### Japanese (JA)

| Model | No fences | Turns | Entities 5/5 | Locations | Notes |
|-------|-----------|-------|-------------|-----------|-------|
| Kimi K2.5 | ✗ | 4 ✓ | ✓ | ✓ | Fences (inconsistent vs EN) |
| GLM-5 | — | — | — | — | **TIMEOUT** (free-tier rate limit) |
| Trinity | ✓ | 4 ✓ | ✓ | ✓ | Clean |
| Gemini | ✓ | 3 ✓ | ✓ | ✓ | Clean |
| Codex | ✓ | 2 ✓ | ✓ | ✓ | Clean |
| Sonnet | ✗ | 3 ✓ | ✓ | ✓ | Fences |

### Chinese (ZH)

| Model | No fences | Turns | Entities 5/5 | Locations | Notes |
|-------|-----------|-------|-------------|-----------|-------|
| Kimi K2.5 | ✓ | 3 ✓ | ✓ | ✓ | Clean |
| GLM-5 | ✓ | 3 ✓ | ✓ | ✓ | Clean |
| Trinity | ✗ | 4 ✓ | ✓ | ✗ | **Hallucinated entity "量子纠错" in entity_locations** |
| Gemini | ✓ | 3 ✓ | ✓ | ✓ | Clean |
| Codex | ✓ | 3 ✓ | ✓ | ✓ | Clean |
| Sonnet | ✗ | 3 ✓ | ✓ | ✓ | Fences + preamble/postamble text |

### Korean (KO)

| Model | No fences | Turns | Entities 5/5 | Locations | Notes |
|-------|-----------|-------|-------------|-----------|-------|
| Kimi K2.5 | ✓ | 3 ✓ | ✓ | ✓ | Clean |
| GLM-5 | ✗ | 3 ✓ | ✓ | ✓ | Fences |
| Trinity | ✓ | 3 ✓ | ✓ | ✓ | Clean |
| Gemini | ✓ | 3 ✓ | ✓ | ✓ | Clean |
| Codex | ✓ | 2 ✓ | ✓ | ✓ | Clean |
| Sonnet | ✗ | 4 ✓ | ✓ | ✓ | Fences + preamble text |

## Aggregate Summary

| Rank | Model | Clean JSON | Entity Preservation | Turn Compliance | Critical Failures | Cost |
|------|-------|------------|---------------------|-----------------|-------------------|------|
| 1 | **Gemini** | 4/4 (100%) | 4/4 | 4/4 | 0 | Free |
| 2 | **Codex** (gpt-5.3) | 4/4 (100%) | 4/4 | 4/4 | 0 | Paid |
| 3 | **Kimi K2.5** | 3/4 (75%) | 4/4 | 4/4 | 0 | Free |
| 4 | **GLM-5** | 2/3 (67%) | 3/3 | 3/3 | 1 timeout | Free |
| 5 | **Sonnet** | 0/4 (0%) | 4/4 | 4/4 | Context confusion | Paid |
| 6 | **Trinity** | 2/4 (50%) | 4/4 | 3/4 | **1 entity hallucination** | Free |

## Failure Analysis

### Trinity: Entity Hallucination (CRITICAL)

In the ZH test, Trinity added `{"surface": "量子纠错", "turn_index": 3}` to `entity_locations`. This term ("quantum error correction") appears in the dialogue text but was **not** in the gold entity list. This violates Rule 3 ("Do NOT invent new entities") and would silently corrupt training data if undetected.

**Mitigation**: A post-processing validation gate checking `output_surfaces ⊆ gold_surfaces` would catch this. However, the fact that Trinity invents entities at all makes it the riskiest model for this task.

### Trinity: Turn Count Violation (EN)

Generated 5 turns when the prompt specifies 2–4. Minor issue but indicates weaker instruction following.

### Sonnet: Project Context Confusion

When run via `claude -p` from within the project directory (`/home/ubuntu/projects/oneiron-ner`), Sonnet read `CLAUDE.md` and project files, then interpreted the data augmentation prompt as a request to *implement the conversationalization pipeline* rather than *perform a conversationalization*. It generated Python scripts and implementation plans instead of JSON dialogue output.

- EN: Tried to implement `span_computer.py` and `phase_a_conversationalize.py`
- ZH: Same — generated implementation code instead of dialogue

Running from `/tmp` (outside the project) fixed this, but Sonnet still wraps output in markdown fences and adds natural-language preamble/postamble text in every case.

### GLM-5: Rate Limit Timeout (JA)

When 3 GLM-5 requests ran concurrently (JA, ZH, KO), the JA request hung indefinitely. The ZH and KO requests completed successfully. This suggests the free tier has a concurrency limit of ~2 concurrent requests.

### Markdown Fences

Models that wrap output in ` ```json ... ``` ` fences: GLM-5 (inconsistent), Sonnet (always), Trinity (inconsistent), Kimi K2.5 (inconsistent). This is trivially fixable with regex stripping in the pipeline but adds post-processing complexity.

## Recommendations for Task 6

### Model Priority

1. **Primary (50%)**: Gemini — best quality, free, reliable, always clean JSON
2. **Primary (50%)**: Kimi K2.5 — free, minor fence-stripping needed, excellent entity preservation
3. **Fallback**: Codex (gpt-5.3) — paid but perfect output, use for retries only
4. **Do not use**: Trinity (hallucination risk), Sonnet via CLI (context confusion + always fences)

### Required Post-Processing

Regardless of model choice, the pipeline should implement:

1. **Fence stripping**: `re.sub(r'^```json\n?|\n?```$', '', output.strip())`
2. **Preamble stripping**: Extract first `{...}` JSON object from output
3. **Entity validation gate**: `set(output_surfaces) ⊆ set(gold_surfaces)` — reject any output containing entities not in the gold list
4. **Turn count validation**: `2 ≤ len(turns) ≤ 4`
5. **JSON parse check**: Reject malformed JSON and retry with fallback model

### Updated CLI Tools Table (for task-6-labeling.md)

| Tool | Model | Cost | Role | Clean JSON | CJK Quality |
|------|-------|------|------|------------|-------------|
| Gemini CLI | Gemini | $0 | Primary (50%) | 4/4 | Excellent |
| OpenCode CLI | Kimi K2.5 | $0 | Primary (50%) | 3/4 | Excellent |
| Codex CLI | gpt-5.3-codex | Paid | Fallback (<5%) | 4/4 | Excellent |
| OpenRouter curl | Trinity | $0 | **Not recommended** | 2/4 | **Hallucination risk** |
| Claude CLI | Sonnet 4.5 | Paid | **Not recommended** | 0/4 | Good (when it works) |
| KiloCode CLI | GLM-5 | $0 | **Not recommended** | 2/3 | Good (reliability issues) |

## Test Artifacts

All raw outputs are preserved at:

```
# Round 1 (EN only: Kimi, GLM-5, Trinity)
/tmp/test_k2.5.txt
/tmp/test_glm5.txt
/tmp/test_trinity.txt

# Round 2 (CJK: Kimi, GLM-5, Trinity)
/tmp/cjk_{k2.5,glm5,trinity}_{ja,zh,ko}.txt

# Round 3 (EN+CJK: Gemini, Codex, Sonnet)
/tmp/test2_gemini_{en,ja,zh,ko}.txt
/tmp/test2_codex_{en,ja,zh,ko}.txt      # Failed (o4-mini not available)
/tmp/test2_sonnet_{en,ja,zh,ko}.txt
/tmp/test3_codex_{en,ja,zh,ko}.txt       # Retried with default model
/tmp/test3_sonnet_{en,zh}.txt            # Retried from /tmp
```

## Methodology Notes

- All tests used the same prompt template with identical rules and output format
- CJK prompts added Rule 5: "The dialogue MUST be in the same language as the input passage"
- Free-tier models were tested at their default settings with no temperature overrides
- Concurrent execution was used for parallel testing (up to 9 requests simultaneously)
- Each model was tested once per language (n=1); production use should validate with larger samples
- Codex initially failed with `-m o4-mini` ("not supported with ChatGPT account"); retested with default model (gpt-5.3-codex)
