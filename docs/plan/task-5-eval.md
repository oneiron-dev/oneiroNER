---
task: 5
title: Eval Harness
phase: 3
depends_on: [4-pre, 4]
agent_model: opus
outputs:
  - data/labeled/eval/oneiron_ner_eval_v1.jsonl
  - scripts/eval_ner.py
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
- Eval prompts must specify: "Return one entry per occurrence in the text. If 'Paris' appears twice, return ['Paris', 'Paris']. Order by first appearance." This matters for mention-level F1.

## Part C: Zero-Shot Evaluation Split

File: `configs/zero_shot_holdout_types.json` (produced by Pre-Task 4, read-only for Task 5)

- Task 5 **reads** the holdout config produced by [Pre-Task 4](task-4-pre-holdout.md) — it does not create or modify it.
- 10-20 entity types are held out from training — **enforced across ALL datasets**. If held-out type "river" appears in B2NERD, fiNERweb, AND chinese_ner_sft, it must be excluded from all three.
- Types are selected such that: (a) they exist across multiple datasets, (b) have >50 test examples each, (c) map to meaningful Oneiron-adjacent category.
- Converters **strip held-out-type entities** from examples (not move entire examples). See [Task 4 splitter](task-4-conversion.md#zero-shot-split-handling).
- Only create dedicated `zero_shot_eval` examples for texts where ALL entities are held-out types.

### Zero-Shot Eval Set Construction

For each held-out type, **reserve N texts** (50-100) that contain it → move those entire texts to `zero_shot_eval`. For remaining texts containing held-out types, strip the held-out entities and keep in train. This gives a real zero-shot eval set with enough volume.

**Target**: ≥50 examples per held-out type, ≥500 total zero-shot eval examples.

### Per-Type-Category Metrics

| Category | Description | Target |
|----------|-------------|--------|
| Trained types | Types seen in training data | 85%+ F1 |
| Zero-shot types | Types held out from training | >60% F1 for 8B, TBD for smaller |
| Oneiron-specific | EMOTION, GOAL, ACTIVITY, RELATIONSHIP_REF, etc. | From Phase 2 synthetic data |

### Benchmark Protocol

**Hardware:**
- Local: M4 Max (MLX-LM) — all 4 Qwen3 sizes
- Cloud: A10G via Modal (MS-SWIFT) — report both

**Prompt lengths**: 64/128/256/512 token buckets (matching Luminal static graph strategy from RESEARCH-002 §11)

**Procedure:**
- Warmup: 10 inferences discarded before measurement
- Both zero-shot baseline (no LoRA) AND post-training for each model size

**Metrics:**
- p50/p95 latency per prompt-length bucket
- Peak RSS memory
- Entity-level F1: strict span match + relaxed token overlap
- Per-language breakdown (EN, JA, ZH, KO, mixed)
- Per-type breakdown (Tier 1 + Tier 2 types)
- Confusion matrix for type misclassification
- Per-model-size comparison (0.6B vs 1.7B vs 4B vs 8B)

**Quantization matrix**: fp16, int8, int4 (Q4_K_M) for all 4 model sizes

**LoRA config anchor** (from ONEIRON-RESEARCH-003): rank 32, alpha 64, dropout 0.05, 16 LoRA layers, lr 5e-5, batch 4, grad accumulation 4, 1000 iterations.

**Reproducibility**: Log all hyperparams, random seeds, dataset version hash.

**Output**: JSON report + human-readable summary.
