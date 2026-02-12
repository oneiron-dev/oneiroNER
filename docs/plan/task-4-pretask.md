---
task: 4-pre
title: Dataset Verification
phase: 1.5
depends_on: [1, 2, 3]
blocks: [4]
agent_model: sonnet
---

# Pre-Task 4: Dataset Verification (BLOCKING)

> Back to [README](README.md) | Prev: [Task 3](task-3-mapping.md) | Next: [Task 4](task-4-conversion.md)

Before [Task 4](task-4-conversion.md) begins, verify assumptions that affect converter design. This task **blocks all Phase 2 agents**.

## Verification Steps

1. **Extract `B2NERD_data.zip`** on the VPS
2. **Count actual B2NERD examples**: EN total, ZH total
   - Plan says ~160K; one review cites ~52K for recommended subset
3. **Count unique B2NERD entity types** across all files
   - Plan says 15K+; one review cites 341
4. **Check 20 ZH B2NERD examples for `pos` field**
   - If ZH has offsets → simplify `convert_b2nerd.py` (remove string matching for ZH)
   - If ZH lacks offsets → keep string matching + ambiguous case logging
5. **Verify B2NERD `pos` end semantics** (10 EN samples)
   - Check inclusive vs exclusive end index
   - Our schema uses Python-style exclusive (`text[start:end] == surface`)
   - If B2NERD uses inclusive end, converters add +1
6. **Verify fiNERweb size**: Plan says ~2M rows, review says ~4M. Check actual HF dataset size.
7. **Verify chinese_ner_sft `start_idx`/`end_idx`** are character offsets (not token indices)
   - Check 10 examples: `text[start_idx:end_idx] == entity_text`

## Output

Update `configs/dataset_inventory.json` with verified numbers. Report findings to Phase 2 agents.
