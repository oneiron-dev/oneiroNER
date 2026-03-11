# oneiroNER

Named entity recognition for conversational text. Extracts 6 sync entity types from multi-turn conversations, optimized for low-latency inference across 75 languages.

## Status

| Component | State | Notes |
|-----------|-------|-------|
| Data pipeline | Complete | 4.9M train / 260K val records, 103K silver, 46-language multilingual seed |
| Model harness | Code complete | Needs smoke run + Modal e2e validation |
| Autoresearch | Code complete | Needs sequential validation before autonomous loop |
| Training | Not started | Blocked on harness validation |

## Entity Types

6 sync types extracted as BIO spans (43 labels total):

| Type | Subtypes |
|------|----------|
| PERSON | — |
| PLACE | — |
| ORG | — |
| DATE | Day, Week, Month, Season, Year, Decade, Relative, Range |
| EVENT | Life, General |
| RELATIONSHIP_REF | Family, Romantic, Friend, Professional, Acquaintance |

3 async types (EMOTION, GOAL, ACTIVITY) are claim-level Dreamer outputs, not span extraction.

## Architecture

- **Backbone**: mGTE 306M (`Alibaba-NLP/gte-multilingual-mlm-base`) — 12 layers, 768 hidden, 8K context, 75 languages
- **Head**: Dropout + Linear(768, 43) token classifier
- **Training**: HF Trainer with MixtureSampler (75% gold / 20% silver-en / 5% silver-ml)
- **Eval**: Exact char-span `(type, start, end)` F1 — token-level is debug only
- **Inference**: BF16 train → CPU INT8 + GPU FP16

## Directory Structure

```
model/              NER model, training, eval (7 files, ~1100 LOC)
research/           Autoresearch harness (7 files, ~770 LOC)
configs/            Schema, type mappings, dataset inventory, training mix
scripts/            Data pipeline: converters, annotation, validation
  lib/              Shared modules (dedup, windower, span tools, schema)
  task8/            JP-RP + ChatHaruhi batch annotation pipeline
  task9/            Silver data QA/validation pipeline
  task9_5/          Silver quality & coverage upgrade
docs/               Design docs, task plans, research analysis
tests/              Test suite
data/               Gitignored — managed locally + HF LFS
```

## Model (`model/`)

| File | Purpose |
|------|---------|
| `config.py` | 43 BIO labels, type normalization, hyperparams, source tiers |
| `ner_model.py` | `NerModel`: AutoModel encoder + dropout + linear head |
| `ner_dataset.py` | JSONL loader, conversation flattening, BIO alignment, collate |
| `train.py` | `NerTrainer` with token accounting, `TokenBudgetCallback`, `AutoresearchCallback` |
| `eval.py` | Char-span F1, token-level callback, REL_REF hard-neg analysis, `run_full_eval()` |
| `verify_split.py` | Train/val split leakage audit |

## Autoresearch (`research/`)

Autonomous hyperparameter search adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch). Runs experiments on Modal H100, tracks via W&B, uses git worktree isolation.

- **Editable surface**: `research/train.py` (EXPERIMENT dict only — nothing else)
- **Composite score**: weighted REL_REF F0.5 (0.40) + macro F1 (0.25) + hard-neg precision (0.20) + multilingual (0.10) + latency (0.05)
- **Budget**: 3.2M tokens/experiment, 15 min wall-clock, ~$0.85/run on H100
- **Stopping**: no improvement for 5 consecutive experiments

See `research/README.md` for the full protocol.

### Running

```bash
# Preparation (once)
python3 -m research.prepare

# Manual single experiment
python3 -m research.launch

# Autonomous loop (requires OpenProse)
prose run research/autoresearch.prose
```

## Data Pipeline

### Sources

| Tier | Sources | Weight |
|------|---------|--------|
| Gold | Existing labeled data (~4.9M train / ~260K val) | 1.0 |
| T1 | mentalchat, therapy, personachat, prosocial | 0.9 |
| T2 | reddit_confessions | 0.7 |
| T3 | pippa, opencharacter, roleplay_hieu, synthetic_persona_chat | 0.5 |
| Multilingual | 46 languages, synthetic seed (4,960 convos) | 0.8 |

### Pipeline

```bash
# Full conversion + dedup + split
python3 scripts/convert_all.py

# Silver annotation (Task 9)
python3 scripts/task9/annotate.py

# Batch annotation (Task 8) — current pipeline
python3 scripts/task8/annotate_batch.py
```

Note: `scripts/task8/label_roleplay.py` is **legacy** (initial `provider="spark"` pipeline). The current Task 8 pipeline uses `annotate_batch.py` with GPT-5.4 and DeepSeek V3.2.

## Configs

| File | Purpose |
|------|---------|
| `schema.json` | JSONL record format (JSON Schema draft 2020-12) |
| `type_mapping_train.json` | Source labels → canonical types for training |
| `type_mapping_eval.json` | Source labels → canonical types for eval |
| `dataset_inventory.json` | All ingested datasets with metadata |
| `training_mix.json` | Dataset mixing configuration |
| `zero_shot_holdout_types.json` | Types excluded from training |

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `transformers`, `accelerate`, `wandb`, `datasets`, `huggingface_hub`

## Compute

- **Training**: Modal (H100 $3.95/hr, B200 $6.25/hr)
- **Tracking**: Weights & Biases (`oneiron-dev/ner-sft`)
- **Inference**: Vast.ai Serverless (3-region: US/EU/Asia)
