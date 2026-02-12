---
task: 6
title: Phase A Labeling (Conversationalize B2NERD)
phase: 3
depends_on: [4]
agent_model: sonnet
outputs:
  - scripts/phase_a_conversationalize.py
  - data/labeled/phase_a/b2nerd_conv_en.jsonl
  - data/labeled/phase_a/b2nerd_conv_zh.jsonl
---

# Task 6: Phase A Labeling (Conversationalize B2NERD)

> Back to [README](README.md) | Prev: [Task 5](task-5-eval.md)

## Key Principle: Preserve Gold Entities, NOT Re-Extract

B2NERD already has entity annotations. The LLM rewrites passages into dialogue while **preserving known entities** — it does NOT rediscover them. This is higher quality, cheaper, and avoids hallucination.

## Three-Step Process

### Step 1: Conversationalize (LLM)

```
Rewrite the following passage as a 2-4 turn dialogue between speakers A and B.
You MUST include these exact entity strings with AT LEAST the specified count:
{entity_surfaces_with_counts}
Do NOT paraphrase entity names. Do NOT invent new entities.
Return JSON: {"turns": [{"speaker": "A", "text": "..."}, ...],
              "entity_locations": [{"surface": "Sarah", "turn_index": 0}, ...]}
```

Where `entity_surfaces_with_counts` is formatted like: `"Sarah" (×2), "Kyoto" (×1), "Tuesday" (×1)`

### Step 2: Compute Spans (programmatic)

- For each gold entity, find its surface in the indicated turn via [`span_computer.py`](task-4-conversion.md#span-computer).
- Entity type comes from the **gold annotation**, NOT from the LLM.
- `confidence: "silver"` (text is new, entities are gold-sourced but may be dropped/mangled).

### Step 3: Validate

- Every gold entity surface must appear in the dialogue with at least the same count as in the source. If missing or under-count, retry or discard.
- New entities the LLM invents are NOT included (unverified).

### Step 4: Post-Process LLM Output

All models except Gemini produce markdown fences or preamble text at least sometimes. Required pipeline in `phase_a_conversationalize.py`:

```python
import re, json

def clean_llm_output(raw: str) -> dict | None:
    """Strip fences, preamble, and validate JSON."""
    # 1. Strip markdown fences
    cleaned = re.sub(r'^```(?:json)?\n?', '', raw.strip())
    cleaned = re.sub(r'\n?```$', '', cleaned)
    # 2. Extract first JSON object (skip preamble/postamble text)
    match = re.search(r'\{[\s\S]*\}', cleaned)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None

def validate_conversationalization(parsed: dict, gold_surfaces: list[str]) -> bool:
    """Validate output against gold annotations."""
    if not parsed or "turns" not in parsed:
        return False
    # Turn count: 2-4
    if not (2 <= len(parsed["turns"]) <= 4):
        return False
    # Entity hallucination gate: reject any entity not in gold list
    if "entity_locations" in parsed:
        output_surfaces = {e["surface"] for e in parsed["entity_locations"]}
        if not output_surfaces.issubset(set(gold_surfaces)):
            return False
    # All gold entities must appear with correct cardinality
    from collections import Counter
    gold_counts = Counter(gold_surfaces)
    dialogue_text = " ".join(t["text"] for t in parsed["turns"])
    for surface, count in gold_counts.items():
        if dialogue_text.count(surface) < count:
            return False
    return True
```

## CLI Tools

Pre-installed at `/home/ubuntu/.bun/bin/`:

| Tool | Model | Cost | Role | Clean JSON | CJK Quality |
|------|-------|------|------|------------|-------------|
| Gemini CLI | Gemini | $0 | **Primary (50%)** | 4/4 | Excellent |
| OpenCode CLI | Kimi K2.5 | $0 | **Primary (50%)** | 3/4 | Excellent |
| Codex CLI | gpt-5.3-codex | Paid | Fallback (<5%) | 4/4 | Excellent |
| KiloCode CLI | GLM-5 | $0 | Tertiary (rate-limited) | 2/3 | Good |
| Trinity | arcee-ai/trinity | $0 | **DO NOT USE** | 2/4 | Hallucination risk |
| Claude CLI | Sonnet 4.5 | Paid | **DO NOT USE** | 0/4 | Context confusion |

### Usage

```bash
# Gemini (primary — highest clean JSON rate)
echo "$PROMPT" | gemini -p "" --yolo -o text

# OpenCode / Kimi K2.5 (primary)
echo "$PROMPT" | opencode run --format json --model opencode/kimi-k2.5-free -

# Codex (fallback only)
echo "$PROMPT" | codex exec --skip-git-repo-check -s read-only -o /tmp/output.txt -

# GLM-5 (tertiary, rate-limited — max 2 concurrent)
echo "$PROMPT" | kilocode run --format json --model kilo/z-ai/glm-5:free -
```

### Excluded Models (Benchmark-Validated)

- **Trinity (arcee-ai/trinity-large-preview)**: Hallucinated entity "量子纠错" in ZH test — silently adds entities not in gold list, corrupting training data. Entity hallucination is undetectable without the validation gate.
- **Claude/Sonnet 4.5**: Context confusion when run via `claude -p` from project directory — reads CLAUDE.md and tries to implement pipeline instead of performing conversationalization. Always produces markdown fences + preamble. Not fixable without isolated environment.

## Script: `scripts/phase_a_conversationalize.py`

1. Reads processed B2NERD JSONL from `data/processed/b2nerd_{en,zh}.jsonl`.
2. Batches 5-10 passages per CLI call (reduces subprocess overhead).
3. Shells out via `subprocess.run()`, piping batched prompt via stdin.
4. Parses JSON response.
5. Computes entity spans via [`span_computer.py`](task-4-conversion.md#span-computer) (occurrence-based).
6. Entity types from **gold annotations**, NOT LLM output.
7. Retries failed examples with fallback chain: gemini → opencode → codex → kilocode (tertiary).
8. Outputs in UniversalNER format with `query_types`.
9. Writes to `data/labeled/phase_a/b2nerd_conv_{en,zh}.jsonl`.

## Parallelization

- **2-way primary split**: gemini ~50%, opencode ~50%.
- Codex NOT in primary split — fallback only for failed examples.
- Fallback chain: gemini fails → retry opencode; opencode fails → retry codex.
- GLM-5 as tertiary overflow when both primaries are at capacity.
- `concurrent.futures.ThreadPoolExecutor`, rate-limit ~2-3 concurrent calls per tool.

## Scale

- **Pilot**: ~500 EN + 200 ZH (validate quality + throughput).
- **This phase**: 5K EN + 2K ZH.
- **Note**: Research doc (Section 10, Phase A) targets 50K total. The 7K here is an intentional pilot scope. Script designed to scale.
