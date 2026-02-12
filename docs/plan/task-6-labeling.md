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
You MUST include these exact entity strings at least once each: {entity_surfaces}.
Do NOT paraphrase entity names. Do NOT invent new entities.
Return JSON: {"turns": [{"speaker": "A", "text": "..."}, ...],
              "entity_locations": [{"surface": "Sarah", "turn_index": 0}, ...]}
```

### Step 2: Compute Spans (programmatic)

- For each gold entity, find its surface in the indicated turn via [`span_computer.py`](task-4-conversion.md#span-computer).
- Entity type comes from the **gold annotation**, NOT from the LLM.
- `confidence: "silver"` (text is new, entities are gold-sourced but may be dropped/mangled).

### Step 3: Validate

- Every gold entity surface must appear in the dialogue. If missing, retry or discard.
- New entities the LLM invents are NOT included (unverified).

## CLI Tools

Pre-installed at `/home/ubuntu/.bun/bin/`:

| Tool | Model | Cost | Role |
|------|-------|------|------|
| OpenCode CLI | opencode/kimi-k2.5-free | $0 (free tier) | Primary (50%) |
| Gemini CLI v0.27.3 | Gemini | $0 (free tier, rate-limited) | Primary (50%) |
| Codex CLI 0.98.0 | gpt-5.3-codex | Paid | Fallback only (<5%) |

### Usage

```bash
# OpenCode (primary)
echo "$PROMPT" | opencode run --format json --model opencode/kimi-k2.5-free -

# Gemini (primary)
echo "$PROMPT" | gemini -p "" --yolo -o text

# Codex (fallback only)
echo "$PROMPT" | codex exec --skip-git-repo-check -s read-only -o /tmp/output.txt -
```

## Script: `scripts/phase_a_conversationalize.py`

1. Reads processed B2NERD JSONL from `data/processed/b2nerd_{en,zh}.jsonl`.
2. Batches 5-10 passages per CLI call (reduces subprocess overhead).
3. Shells out via `subprocess.run()`, piping batched prompt via stdin.
4. Parses JSON response.
5. Computes entity spans via [`span_computer.py`](task-4-conversion.md#span-computer) (occurrence-based).
6. Entity types from **gold annotations**, NOT LLM output.
7. Retries failed examples with fallback chain: opencode → gemini → codex.
8. Outputs in UniversalNER format with `query_types`.
9. Writes to `data/labeled/phase_a/b2nerd_conv_{en,zh}.jsonl`.

## Parallelization

- **2-way primary split**: opencode ~50%, gemini ~50%.
- Codex NOT in primary split — fallback only for failed examples.
- Fallback chain: opencode fails → retry gemini; gemini fails → retry codex.
- `concurrent.futures.ThreadPoolExecutor`, rate-limit ~2-3 concurrent calls per tool.

## Scale

- **Pilot**: ~500 EN + 200 ZH (validate quality + throughput).
- **This phase**: 5K EN + 2K ZH.
- **Note**: Research doc (Section 10, Phase A) targets 50K total. The 7K here is an intentional pilot scope. Script designed to scale.
