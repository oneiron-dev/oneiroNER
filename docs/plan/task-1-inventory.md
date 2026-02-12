---
task: 1
title: Dataset Inventory
phase: 1
depends_on: []
agent_model: sonnet
outputs:
  - configs/dataset_inventory.json
  - configs/dataset_inventory.md
---

# Task 1: Dataset Inventory

> Back to [README](README.md) | Next: [Task 2 — Schema](task-2-schema.md)

Write `configs/dataset_inventory.json` and `configs/dataset_inventory.md` from the exploration findings.

## Datasets

| Dataset | Format | Size | Languages | Entity Types | Span Info |
|---------|--------|------|-----------|-------------|-----------|
| B2NERD | JSON (zipped) | ~160K (unverified — see [Pre-Task 4](task-4-pretask.md)) | EN, ZH | 15K+ (unverified) | EN has `pos`; ZH needs verification |
| open-ner-standardized | Parquet BIO | ~830K | 52 langs | 60 types | Via BIO reconstruction |
| open-ner-core-types | Parquet BIO | ~830K | 52 langs | 3 (PER/LOC/ORG) | Via BIO reconstruction |
| fiNERweb | Parquet spans | ~2-4M (needs verification) | 91 langs | 235K+ | Native char spans |
| NuNER | Model weights | N/A | N/A | N/A | SKIP — model checkpoint |
| chinese_ner_sft | JSONL | ~182K | ZH | 72 types | Native spans (verify char vs token) |
| multiconer_v2 | CoNLL BIO | ~170K | 12 langs | 33 types | Via BIO reconstruction |
| klue | Parquet BIO | ~26K | KO | 6 types | Via BIO reconstruction |
| stockmark-ner-ja | JSON | ~5.3K | JA | 8 types | Native char spans |
| ChatHaruhi-RolePlaying | JSONL | ~11.5K | ZH/EN | None | Needs LLM labeling |
| Japanese-Roleplay-Dialogues | JSONL | ~4.3K | JA | None | Needs LLM labeling |

## Data Locations

All datasets symlinked in `data/raw/` → HF cache snapshots (READ ONLY).

## Notes

- B2NERD data is in `B2NERD_data.zip` — must extract before processing (see [Pre-Task 4](task-4-pretask.md)).
- NuNER is a model checkpoint, not a dataset. Skip it.
- Size and type count for B2NERD and fiNERweb need verification in [Pre-Task 4](task-4-pretask.md).
