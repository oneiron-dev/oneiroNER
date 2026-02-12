---
task: 5
title: Eval Harness
phase: 3
depends_on: [4]
agent_model: opus
outputs:
  - data/labeled/eval/oneiron_ner_eval_v1.jsonl
  - scripts/eval_ner.py
  - configs/zero_shot_holdout_types.json
---

# Task 5: Eval Harness (200+ Gold Messages)

> Back to [README](README.md) | Prev: [Task 4](task-4-conversion.md) | Next: [Task 6](task-6-labeling.md)

## Part A: Evaluation Set

File: `data/labeled/eval/oneiron_ner_eval_v1.jsonl`

- Generate 230+ messages via Claude Opus covering required distribution.
- Label all Tier 1 and Tier 2 entity types with strict span boundaries.
- Include negative examples (no entities).
- Each example includes `query_types` with positive and negative types.
- **`confidence: "synthetic-gold"`** — NOT human-gold. Opus-generated + programmatic validation catches boundary errors but misses wrong types, missing entities, hallucinated entities with valid spans.
- `split: "eval"`
- Validate all spans programmatically before committing.
- **Benchmark all 4 model sizes** (0.6B, 1.7B, 4B, 8B) + quantization variants (int4, int8) for on-device models.

## Part B: Evaluation Script

File: `scripts/eval_ner.py`

- Entity-level F1 (strict span match + relaxed overlap).
- Per-language breakdown (EN, JA, ZH, KO, mixed).
- Per-type breakdown (PERSON, PLACE, DATE, ORG + Tier 2 types).
- **Per-model-size metrics** (0.6B vs 1.7B vs 4B vs 8B).
- Confusion matrix for type misclassification.
- Output: JSON report + human-readable summary.

## Part C: Zero-Shot Evaluation Split

File: `configs/zero_shot_holdout_types.json`

- Hold out 10-20 entity types from training — **enforced across ALL datasets**. If held-out type "river" appears in B2NERD, fiNERweb, AND chinese_ner_sft, it must be excluded from all three.
- Select types that: (a) exist across multiple datasets, (b) have >50 test examples each, (c) map to meaningful Oneiron-adjacent category.
- Converters **strip held-out-type entities** from examples (not move entire examples). See [Task 4 splitter](task-4-conversion.md#zero-shot-split-handling).
- Only create dedicated `zero_shot_eval` examples for texts where ALL entities are held-out types.

### Per-Type-Category Metrics

| Category | Description | Target |
|----------|-------------|--------|
| Trained types | Types seen in training data | 85%+ F1 |
| Zero-shot types | Types held out from training | >60% F1 for 8B, TBD for smaller |
| Oneiron-specific | EMOTION, GOAL, ACTIVITY, RELATIONSHIP_REF, etc. | From Phase 2 synthetic data |
