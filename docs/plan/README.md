---
project: oneiron-ner
type: implementation-plan
version: 4
tasks: 7
phases: 3
---

# Oneiron NER Data Processing & Labeling Pipeline

## Purpose

Improve NER accuracy from ~64% to 90%+ across EN and CJK languages for the Eiri AI companion's Oneiron memory engine. Normalize 11 heterogeneous source datasets into a common schema, build an eval harness, and produce conversationalized training data.

## Architecture

**UniversalNER-style open-vocabulary NER.** Entity types are input parameters at inference time, not hardcoded output classes. The model learns to extract any entity type specified in the prompt, including types never seen during training. This is critical for Oneiron's plugin extensibility — plugins register new entity types, NER extracts them without retraining.

### Model Scaling Ladder

| Tier | Model | Params | fp16 Size | Deployment | Latency | Purpose |
|------|-------|--------|-----------|------------|---------|---------|
| Tier 0 | Qwen3-0.6B | 0.6B | 1.2GB | On-device | <50ms | Ultra-light fallback. Tier 1 types only. May not support open-vocab. |
| Tier 1 | Qwen3-1.7B | 1.7B | 3.4GB | On-device | <100ms | Primary on-device. Best quality/latency for mobile. |
| Tier 1.5 | Qwen3-4B | 4B | 8GB (int4: ~2.5GB) | On-device or server | <200ms | Full type vocabulary. Fallback when 1.7B insufficient. |
| Tier 2 | Qwen3-8B | 8B | 16GB | Server async | <500ms | Batch processing, plugin types, maximum quality. |

- On-device: 0.6B trivially fits. 1.7B fits (int8 ~1.7GB). 4B needs int4 (~2.5GB). 8B server-only.
- All 4 models train on identical data with full type vocabulary. Tier split is inference-time only.
- LoRA fine-tuning for all 4.

### Eval Key Questions

1. Does 0.6B achieve acceptable zero-shot F1 (>50%)? If not, Tier 1 types only with fixed list.
2. Quality cliff between 1.7B and 4B — is the jump worth 2x memory?
3. Can 4B at int4 match 4B at fp16? Quantization quality gap matters for on-device.

(Ref: ONEIRON-RESEARCH-002 §11-§12 cover Luminal/on-device.)

## Task Index

| Task | File | Phase | Agent Model | Description |
|------|------|-------|-------------|-------------|
| Task 1 | [task-1-inventory.md](task-1-inventory.md) | 1 | Sonnet | Dataset inventory |
| Task 2 | [task-2-schema.md](task-2-schema.md) | 1 | Opus | Common schema (UniversalNER) |
| Task 3 | [task-3-mapping.md](task-3-mapping.md) | 1 | Opus | Ontology mapping (eval-only) |
| Pre-Task 4 | [task-4-pretask.md](task-4-pretask.md) | 1.5 | Sonnet | Dataset verification (blocking) |
| Task 4 | [task-4-conversion.md](task-4-conversion.md) | 2 | Opus x3 | Conversion scripts |
| Task 5 | [task-5-eval.md](task-5-eval.md) | 3 | Opus | Eval harness |
| Task 6 | [task-6-labeling.md](task-6-labeling.md) | 3 | Sonnet | Phase A conversationalization |

## Execution Plan

### Phase 1: Configuration (parallel)
Launch 3 agents simultaneously:
- **inventory-agent** (Sonnet): [Task 1](task-1-inventory.md)
- **schema-agent** (Opus): [Task 2](task-2-schema.md)
- **mapping-agent** (Opus): [Task 3](task-3-mapping.md)

### Phase 1.5: Verification (blocking)
1 agent (Sonnet): [Pre-Task 4](task-4-pretask.md) — extract B2NERD, verify all dataset assumptions. Blocks Phase 2.

### Phase 2: Conversion (parallel, after Phase 1.5)
Launch 3 Opus agents:
- **converter-spans**: lib/ + stockmark + fiNERweb + chinese_ner_sft
- **converter-b2nerd**: B2NERD conversion
- **converter-bio**: open-ner + multiconer + klue + orchestrator

### Phase 3: Eval & Labeling (after Phase 2)
- **eval-agent** (Opus): [Task 5](task-5-eval.md)
- **conversationalize-agent** (Sonnet): [Task 6](task-6-labeling.md)

## Dependencies

```
# requirements.txt
datasets>=2.14.0
pandas>=2.0.0
jsonlines>=4.0.0
pyarrow>=14.0.0
datasketch>=1.6.0    # MinHash for near-dedup
```

CLI tools at `/home/ubuntu/.bun/bin/`: codex (paid), gemini (free), opencode (free).

NuNER: Skip. Model checkpoint, not dataset.

## Change Log

See [CHANGELOG.md](CHANGELOG.md) for full history of plan revisions across 4 rounds.
