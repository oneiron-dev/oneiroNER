# Autoresearch: NER Baseline Optimization

## Goal
Maximize composite NER score on mGTE-306M via autonomous hyperparameter search.

## Composite Score
- 0.40 × REL_REF exact F0.5
- 0.25 × sync macro F1 (6 base types)
- 0.20 × REL hard-negative precision
- 0.10 × multilingual F1 (disabled until holdout)
- 0.05 × latency bonus (disabled until benchmark)

## Hard Gates
- macro_f1 ≥ 0.40, rel_ref_precision ≥ 0.30, model_size ≤ 1500MB

## Editable Surface
- `research/train.py` — EXPERIMENT dict + MODEL_OVERRIDES dict. Nothing else.

## Fixed Files (NEVER edit during autoresearch)
- model/*, research/score.py, research/launch.py, research/prepare.py, configs/*

## Budget
- Fixed token budget (3.2M tokens/experiment) + 15 min wall-clock cap
- ~6 experiments/hr, ~$0.85/experiment on H100 (validate after first 10)
- H100 default ($3.95/hr). B200 available via AUTORESEARCH_GPU=B200 ($6.25/hr).

## Running
- Automated: `prose run research/autoresearch.prose`
- Manual: edit train.py → git commit → `modal run research/launch.py` → evaluate → keep/reset

## Preparation
```bash
python -m research.prepare   # builds mini subsets + worktree + results.tsv header
```

## Rules
- One change per experiment
- Simpler is better — removing code for equal results is celebrated
- results.tsv is the source of truth, not git history
- Phase 1: hyperparams only. Phase 2: MODEL_OVERRIDES (after Phase 1 converges).

## Phase 1 Search Order
1. LR sweep (1e-5, 2e-5, 3e-5, 5e-5)
2. Warmup ratio (0.03, 0.05, 0.10)
3. Dropout (0.05, 0.1, 0.15, 0.2)
4. Weight decay (0.001, 0.01, 0.1)
5. Mixture ratios (gold/silver_en/silver_ml)
6. Batch size (16, 32, 64) — token budget ensures comparability

## Stopping Criteria
- No improvement for 5 consecutive experiments, or 100 total experiments

## Metric Authority
- W&B summary + smoke_abort flag → authoritative for KEEP/REVERT decisions
- AUTORESEARCH_METRICS JSON from run.log → authoritative for archival in results.tsv
