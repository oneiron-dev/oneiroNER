---
task: 2
title: Common Schema Design (UniversalNER format)
phase: 1
depends_on: []
agent_model: opus
outputs:
  - configs/schema.json
  - configs/schema.md
---

# Task 2: Common Schema Design

> Back to [README](README.md) | Prev: [Task 1](task-1-inventory.md) | Next: [Task 3](task-3-mapping.md)

Every training example is `(text, query_types) → entities`. The `query_types` field is part of the **input**, not just output.

## Dual-View Training (View A + View B)

To preserve type diversity for open-vocab training AND ensure the model has seen Oneiron canonical type strings enough times:

- **View A (raw)**: `type` = source label (PER, LOC, 人名, etc.), `original_type` = same. Every example gets View A. Preserves diversity for open-vocab generalization.
- **View B (canonical)**: `type` = Oneiron canonical label (PERSON, PLACE, etc.), `original_type` = source label. Generated only for entities whose source type maps to an Oneiron canonical type (via `type_mapping.json` from [Task 3](task-3-mapping.md)). Source suffixed with `_canonical`. `query_types` use canonical strings.
- Long-tail types with no Oneiron mapping: View A only.

### View A Example (passage)

```json
{
  "schema_version": "1.0",
  "source": "b2nerd",
  "source_id": "PileNER_00001",
  "language": "en",
  "split": "train",
  "confidence": "synthetic-gold",
  "provenance": ["b2nerd"],
  "text": "I had coffee with Sarah last Tuesday",
  "query_types": ["PER", "DATE", "WEAPON"],
  "entities": [
    {"surface": "Sarah", "type": "PER", "original_type": "PER", "start": 18, "end": 23},
    {"surface": "last Tuesday", "type": "DATE", "original_type": "DATE", "start": 24, "end": 36}
  ]
}
```

### View B Example (same text, canonical types)

```json
{
  "schema_version": "1.0",
  "source": "b2nerd_canonical",
  "source_id": "PileNER_00001_canonical",
  "language": "en",
  "split": "train",
  "confidence": "synthetic-gold",
  "provenance": ["b2nerd"],
  "text": "I had coffee with Sarah last Tuesday",
  "query_types": ["PERSON", "DATE", "WEAPON"],
  "entities": [
    {"surface": "Sarah", "type": "PERSON", "original_type": "PER", "start": 18, "end": 23},
    {"surface": "last Tuesday", "type": "DATE", "original_type": "DATE", "start": 24, "end": 36}
  ]
}
```

### Conversation Format

```json
{
  "schema_version": "1.0",
  "source": "b2nerd_conv",
  "source_id": "PileNER_00001_conv",
  "language": "en",
  "split": "train",
  "confidence": "silver",
  "provenance": ["b2nerd"],
  "format": "conversation",
  "turns": [
    {"speaker": "A", "text": "Did you hear Sarah went to Kyoto?"},
    {"speaker": "B", "text": "Yeah, last Tuesday!"}
  ],
  "query_types": ["PER", "LOC", "DATE", "ORG"],
  "entities": [
    {"surface": "Sarah", "type": "PER", "original_type": "PER", "turn_index": 0, "start": 14, "end": 19},
    {"surface": "Kyoto", "type": "LOC", "original_type": "LOC", "turn_index": 0, "start": 28, "end": 33},
    {"surface": "last Tuesday", "type": "DATE", "original_type": "DATE", "turn_index": 1, "start": 6, "end": 18}
  ]
}
```

### Conversation Rendering Spec

The canonical string rendering for training/inference input:

```
A: Did you hear Sarah went to Kyoto?
B: Yeah, last Tuesday!
```

The model receives the rendered conversation as the `Text:` field in the per-type Q/A prompt. Entity `start`/`end` offsets remain relative to `turns[turn_index].text` in the schema, but the training prompt uses the rendered form. `span_computer.py` handles conversation format by scoping to `turns[turn_index].text`.

## Schema Fields

| Field | Type | Description |
|-------|------|-------------|
| `query_types` | `string[]` | INPUT to model. Positive types + 2-5 frequency-weighted negative types. |
| `entities[].type` | `string` | View A: source label. View B: Oneiron canonical label. |
| `entities[].original_type` | `string` | Source dataset's verbatim label (always preserved). |
| `entities[].turn_index` | `int` | Required when `format == "conversation"`. `start`/`end` relative to `turns[turn_index].text`. |
| `split` | `string` | `train`, `val`, `eval`, `zero_shot_eval`. |
| `confidence` | `string` | `human-gold`, `synthetic-gold`, `gold`, `silver`. |
| `provenance` | `string[]` | Source dataset(s). Multi-element for merged duplicates (see [Task 4 dedup](task-4-conversion.md#dedup)). |

## Confidence Tiers

| Tier | Definition | Examples |
|------|-----------|----------|
| human-gold | Human-reviewed and adjudicated. Requires actual human audit. | None currently. Upgrade: audit ≥20% of eval set. |
| synthetic-gold | LLM-generated + programmatic validation. | Task 5 eval, B2NERD, fiNERweb, chinese_ner_sft. |
| gold | Human-annotated benchmarks. | stockmark, klue, multiconer dev/test, open-ner-core-types. |
| silver | Auto-converted or lower-confidence. | open-ner-standardized, conversationalized outputs. |

## Negative Type Sampling

`query_types` = positive types + 2-5 frequency-weighted negatives. Per UniversalNER: frequency-based > uniform, +5.7 F1.

**Critical**: Held-out zero-shot types (from `configs/zero_shot_holdout_types.json`) are **excluded from the negative sampling pool entirely**. Even appearing as a negative is leakage — the model learns "RIVER → usually empty."

- v1: Static (fixed at conversion time).
- v2 TODO: Dynamic per-epoch sampling.

See [Task 4 — negative_sampler.py](task-4-conversion.md#negative-sampler) for implementation.

## Training Prompt Template

Each `query_type` becomes a separate user turn (UniversalNER per-type Q/A format):

```
System: Extract entities of the requested type from the given text.
Return one entry per occurrence in the text. If "Paris" appears twice, return ["Paris", "Paris"].
Order by first appearance in the text. If none exist, return [].

User: Text: "I had coffee with Sarah last Tuesday in Kyoto."
Extract all entities of type: PERSON
Return one entry per occurrence. Order by first appearance. If none, return [].
Assistant: ["Sarah"]

User: Text: "I had coffee with Sarah last Tuesday in Kyoto."
Extract all entities of type: WEAPON
Return one entry per occurrence. Order by first appearance. If none, return [].
Assistant: []
```

This makes "empty answer for negative types" unambiguous.

### Chat Format Output

The schema JSONL is converted to OpenAI chat format for training frameworks (MLX-LM, MS-SWIFT) by `convert_all.py`. Each `query_type` becomes a separate chat example with system/user/assistant messages. See [Task 4 — Training Format Conversion](task-4-conversion.md#training-format-conversion) for implementation.

Reference: ONEIRON-RESEARCH-003 for training framework selection.
