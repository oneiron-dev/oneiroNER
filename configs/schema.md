# oneiroNER Common Schema

> Formal documentation for `configs/schema.json` — the JSONL record format for all converted NER training data.

## Overview

Every training example is `(text, query_types) → entities`. The schema supports two text formats:

- **Passage**: flat text string (default)
- **Conversation**: multi-turn dialogue with per-turn entity offsets

And a dual-view typing system:

- **View A (raw)**: `type` = source label, `original_type` = same
- **View B (canonical)**: `type` = Oneiron canonical label, `original_type` = source label

The schema is defined in JSON Schema draft 2020-12, enabling `unevaluatedProperties: false` across `if/then/else` branches.

## Quick Reference

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
  "query_types": ["PER", "DATE", "WEAPON", "PRODUCT", "EVENT"],
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
  "query_types": ["PERSON", "DATE", "WEAPON", "PRODUCT", "EVENT"],
  "entities": [
    {"surface": "Sarah", "type": "PERSON", "original_type": "PER", "start": 18, "end": 23},
    {"surface": "last Tuesday", "type": "DATE", "original_type": "DATE", "start": 24, "end": 36}
  ]
}
```

### Conversation Example

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
  "query_types": ["PER", "LOC", "DATE", "ORG", "EVENT", "PRODUCT"],
  "entities": [
    {"surface": "Sarah", "type": "PER", "original_type": "PER", "turn_index": 0, "start": 13, "end": 18},
    {"surface": "Kyoto", "type": "LOC", "original_type": "LOC", "turn_index": 0, "start": 27, "end": 32},
    {"surface": "last Tuesday", "type": "DATE", "original_type": "DATE", "turn_index": 1, "start": 6, "end": 18}
  ]
}
```

## Field Reference

| Field | Type | Required | Description | Constraints |
|-------|------|----------|-------------|-------------|
| `schema_version` | string | always | Schema version | Must be `"1.0"` |
| `source` | string | always | Source dataset identifier | Non-empty. View B appends `_canonical` |
| `source_id` | string | always | Record ID within source | Non-empty. View B appends `_canonical` |
| `language` | string | always | Language code | Non-empty. Normalization is code-enforced |
| `split` | string | always | Dataset split | Enum: `train`, `val`, `eval`, `zero_shot_eval` |
| `confidence` | string | always | Data quality tier | Enum: `human-gold`, `synthetic-gold`, `gold`, `silver` |
| `provenance` | string[] | always | Source dataset(s) | minItems: 1, uniqueItems |
| `text` | string | passage | Passage text | Non-empty. Forbidden in conversation format |
| `format` | string | conversation | Format discriminator | Must be `"conversation"`. Omitted for passages |
| `turns` | turn[] | conversation | Dialogue turns | minItems: 2. Forbidden in passage format |
| `query_types` | string[] | always | Entity types to extract | minItems: 1, uniqueItems |
| `entities` | entity[] | always | Extracted entities | Empty array allowed (all-negative examples) |

## Entity Fields

| Field | Type | Required | Description | Constraints |
|-------|------|----------|-------------|-------------|
| `surface` | string | always | Matched text span | minLength: 1. Must equal source text at `[start:end]` (code-enforced) |
| `type` | string | always | Entity type label | minLength: 1. View A: source label, View B: canonical |
| `original_type` | string | always | Source dataset's verbatim label | minLength: 1. Always preserved |
| `start` | integer | always | Start character offset (inclusive) | >= 0 |
| `end` | integer | always | End character offset (exclusive) | >= 0. Must be > start (code-enforced) |
| `turn_index` | integer | conversation | Index into `turns[]` | >= 0. Required only in conversation format |

## Confidence Tiers

Ordered from highest to lowest quality. When merging duplicates, use `min()` with this ordering:

| Tier | Rank | Definition | Examples |
|------|------|-----------|----------|
| `human-gold` | 1 (highest) | Human-reviewed and adjudicated | None currently. Upgrade: audit >=20% of eval set |
| `synthetic-gold` | 2 | LLM-generated + programmatic validation | B2NERD, fiNERweb, chinese_ner_sft |
| `gold` | 3 | Human-annotated benchmarks | stockmark, KLUE, MultiCoNER dev/test |
| `silver` | 4 (lowest) | Auto-converted or lower-confidence | open-ner-standardized, conversationalized outputs |

## Split Values

| Split | Purpose |
|-------|---------|
| `train` | Training data |
| `val` | Validation / dev set |
| `eval` | Evaluation benchmark |
| `zero_shot_eval` | Zero-shot evaluation (held-out types not seen during training) |

Rules:
- `eval` and `zero_shot_eval` splits are never used for training
- `zero_shot_eval` contains only types from `configs/zero_shot_holdout_types.json`

## Dual-View System

### View A (raw)
- `type` = source dataset's original label (e.g. `PER`, `人名`, `LOC`)
- `original_type` = same as `type`
- Every record gets a View A
- Preserves label diversity for open-vocab generalization

### View B (canonical)
- `type` = Oneiron canonical label (e.g. `PERSON`, `PLACE`, `ORG`)
- `original_type` = source dataset's original label (preserved)
- Generated only when source type maps to a canonical type via `type_mapping_train.json`
- `source` suffixed with `_canonical` (e.g. `b2nerd_canonical`)
- `source_id` suffixed with `_canonical`
- `query_types` use canonical type strings

### When View B is omitted
- Long-tail types with no Oneiron canonical mapping → View A only

## Query Types

`query_types` is the **input** to the model — it defines what entity types to extract.

- Contains all **positive types** (types that have entities in the text)
- Plus **2-5 frequency-weighted negative types** (types with no entities)
- Negative types produce empty extraction results (`[]`), teaching the model to say "none found"

### Negative type sampling rules
- 2-5 negatives per example (code-enforced in Task 4)
- Schema enforces `minItems: 1` only
- Frequency-weighted sampling (+5.7 F1 over uniform per UniversalNER)
- **Held-out zero-shot types are excluded from the negative pool entirely** — even appearing as a negative is leakage

### After merge
- `query_types` must be recomputed to account for merged entity sets

## Validation Rules

### Schema-enforced (JSON Schema)

| Rule | Mechanism |
|------|-----------|
| Required fields present | `required` |
| Correct types (string, integer, array, object) | `type` |
| Enum values for `split`, `confidence` | `enum` |
| `schema_version` is `"1.0"` | `const` |
| Non-empty strings | `minLength: 1` |
| Non-empty provenance | `minItems: 1` |
| No duplicate query_types | `uniqueItems` |
| No duplicate provenance | `uniqueItems` |
| Passage/conversation discrimination | `if/then/else` on `format` |
| No unknown top-level fields | `unevaluatedProperties: false` |
| No unknown fields in turns/entities | `additionalProperties`/`unevaluatedProperties: false` |
| Conversation turns >= 2 | `minItems: 2` |
| `turn_index` required in conversation entities | `required` in `entity_conversation` |
| `turn_index` forbidden in passage entities | `unevaluatedProperties: false` in `entity_passage` |

### Code-enforced (lib/schema.py in Task 4)

| Rule | Reason |
|------|--------|
| Span matches source text: `text[start:end]` (passage) or `turns[turn_index].text[start:end]` (conversation) | Span validation |
| `end > start` | Forbid zero-length spans |
| `turn_index < len(turns)` | Bounds check |
| `query_types ⊇ positive_types` | All entity types must be queried |
| 2-5 negative types in query_types | Training design constraint |
| Holdout type exclusion from negatives | Prevent zero-shot leakage |
| `(source, source_id)` uniqueness across file | Deduplication |
| Language code normalization | Consistent format |

## Conversation Format

### Turn structure
- Each turn has `speaker` (non-empty string) and `text` (non-empty string)
- Minimum 2 turns per conversation (single-turn should use passage format)
- No additional properties allowed on turn objects

### Rendering spec
Conversations are rendered for training prompts as:
```
{speaker}: {text}
{speaker}: {text}
```
Example:
```
A: Did you hear Sarah went to Kyoto?
B: Yeah, last Tuesday!
```

### Entity offsets
- `start`/`end` are relative to `turns[turn_index].text`, not the rendered string
- `span_computer.py` handles conversation format by scoping to `turns[turn_index].text`

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

Each query_type produces one Q/A pair. Negative types produce `[]` responses.
