# Repo Hygiene Report — 2026-03-11

## Summary

Cleaned ~120 items (~6.7 MB) from the repo root. All items were **untracked by git** — no tracked files were modified or deleted (except `.gitignore`).

## Actions Taken

### 1. Deleted caches
- 11 `__pycache__/` directories (regenerated automatically)
- `.ruff_cache/` (regenerated automatically)

### 2. Archived root scripts → `archive/scratch/root_scripts/` (70 files)
- 16 Task 8 batch annotation scripts (`annotate_*.py`, `do_annotate_*.py`)
- 23 multilingual generation scripts (`gen_*.py`, `generate_*.py`)
- 18 offset debugging scripts (`get_offsets*.py`, `get_indices*.py`)
- 3 one-off batch annotators (`tmp_batch0*_annotate.py`)
- 3 draft multilingual gen scripts (`tmp_lt_gen*.py`, `tmp_sr_gen.py`)
- 7 misc scratch scripts (`fix_bn_offsets.py`, `my_script.py`, `test_ner.py`, etc.)

### 3. Archived scratch data → `archive/scratch/tmp_files/` (41 files)
- 8 pilot annotation partials (`.tmp_annotations_*.json`)
- 6 conversation chunk slices (`tmp_chunk_*.json`)
- 5 individual conversation extracts (`tmp_conv_*.json`)
- 11 JP-RP conversation extracts (`tmp_jp_rp_*.json`, `jp_rp_*.json`)
- 5 text dumps (`*.txt`)
- 6 misc JSON files

### 4. Archived temp directories → `archive/scratch/task8_experiments/`
- `tmp/` — batch splits, candidates, draft annotators
- `tmp_batch069/` — JP-RP extracts + partial annotations
- `tmp_extract_task8/` — 5 JP-RP conversation JSONs
- `.tmp_task8_dump/` — 2 raw conversation text files
- `data/task8_tmp/` — batch 050 + 058 conversation splits

### 5. Archived debug output → `archive/scratch/debug_outputs/`
- `tools/batch071_entities.py` — one-off entity injection script
- Removed empty `tools/` directory

### 6. Archived stale docs → `archive/notes/`
- `HANDOVER.md` — Task 6 handover doc (2026-02-16)
- `RECENT_NER_DATASETS_2024-2025.md` — research notes

### 7. Updated `.gitignore`
Added entries for: `.env`, `tmp/`, `tmp_*/`, `.tmp_*/`, `.ruff_cache/`, `archive/scratch/`, `data/task8_batches/`, `data/task8_output/`, `data/task8_output_gpt/`, `data/task8_merged/`, `data/task8_pilot/`, `data/task8_tmp/`, `data/labeled/`

## Untouched directories
- `model/` — active SFT code
- `research/` — active autoresearch harness
- `scripts/` — active pipeline code
- `configs/`, `docs/`, `tests/` — active/canonical
- `data/processed/`, `data/raw/` — canonical data (already gitignored)
- `.claude/` — local config

## Validation
- All `.py` files in `model/`, `research/`, `scripts/` pass `py_compile`
- No imports reference moved files
- `git status` clean (only `.gitignore` and `reports/` staged)
