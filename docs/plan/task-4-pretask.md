---
task: 4-pre
title: Dataset Verification
phase: 1.5
depends_on: [1, 2, 3]
blocks: [4, 5]
agent_model: sonnet
outputs:
  - configs/dataset_inventory.json
  - configs/zero_shot_holdout_types.json
---

# Pre-Task 4: Dataset Verification (BLOCKING)

> Back to [README](README.md) | Prev: [Task 3](task-3-mapping.md) | Next: [Task 4](task-4-conversion.md)

Before [Task 4](task-4-conversion.md) begins, verify assumptions that affect converter design. This task **blocks all Phase 2 agents** and also produces the holdout types file that Task 5 reads.

## Verification Steps

1. **Extract `B2NERD_data.zip`** on the VPS
2. **Determine which B2NERD version is in `B2NERD_data.zip`** (likely raw 1.4M based on filename). Extract and identify curated vs raw subsets. Verify curated subset has ~52K examples.
3. **Count actual B2NERD examples**: EN total, ZH total
   - Curated: 51,907 (25,403 EN + 26,504 ZH), 341 types
   - Raw: 1,419,161 (838,648 EN + 580,513 ZH), 341 types
   - Test: 20,723 (6,466 EN + 14,257 ZH), 145 types
4. **Count unique B2NERD entity types** across all files — confirm 341 types
5. **Check 20 ZH B2NERD examples for `pos` field**
   - If ZH has offsets → simplify `convert_b2nerd.py` (remove string matching for ZH)
   - If ZH lacks offsets → keep string matching + ambiguous case logging
6. **Verify B2NERD `pos` end semantics** (10 EN samples)
   - Check inclusive vs exclusive end index
   - Our schema uses Python-style exclusive (`text[start:end] == surface`)
   - If B2NERD uses inclusive end, converters add +1
7. **Verify fiNERweb size**: Plan says ~2M rows, review says ~4M. Check actual HF dataset size.
8. **Verify chinese_ner_sft `start_idx`/`end_idx`** are character offsets (not token indices)
   - Check 10 examples: `text[start_idx:end_idx] == entity_text`

### Type Frequency Scan + Holdout Selection

1. **Scan all datasets** — collect type frequency counts per dataset per language (no full conversion needed, just iterate and count).
2. **Apply holdout selection criteria**:
   - Types that exist across >=3 datasets
   - Have >50 examples each
   - Are mid-frequency (not head types like PER/LOC/ORG, not extreme tail)
3. **Build equivalence classes** for cross-dataset enforcement — map the same semantic type across different label namespaces:
   ```json
   {
     "DISEASE": ["disease", "Disease", "medical condition", "疾病", "病気", "질병"],
     "AWARD": ["award", "Award", "prize", "Prize", "奖项", "賞"],
     "FOOD": ["food", "Food", "cuisine", "dish", "食物", "料理", "음식"]
   }
   ```
   Uses Task 3's mapping infrastructure to build these classes.
4. **Output**: `configs/zero_shot_holdout_types.json` with equivalence classes.

Good holdout candidates (pending actual frequency scan): DISEASE/MEDICAL_CONDITION, AWARD, FOOD, ANIMAL, SOFTWARE/PRODUCT, BUILDING/FACILITY, LANGUAGE, CREATIVE_WORK. Actual selection is data-driven — do not hardcode.

The splitter checks: does this entity's type match ANY string in ANY holdout equivalence class? If yes, strip it (or reserve for zero_shot_eval per Fix 9).

## Output

Update `configs/dataset_inventory.json` with verified numbers. Produce `configs/zero_shot_holdout_types.json` with equivalence classes. Report findings to Phase 2 agents.
