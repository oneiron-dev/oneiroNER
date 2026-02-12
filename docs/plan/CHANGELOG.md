---
type: changelog
project: oneiron-ner
---

# Plan Change Log

## Round 5 — Validated Reviews + LLM Benchmark

Source: 5 independent reviews cross-validated against ONEIRON-RESEARCH-002/003, plus LLM conversationalization benchmark across 6 models × 4 languages.

### Must Fix (8 items)

| # | Fix | Resolution |
|---|-----|-----------|
| 1 | NFKC fallback broken | Search width-folded variants in original text. Length-guard normalized-text search. Never reuse offsets from different-length string. |
| 2 | Zero-shot holdout dependency cyclic | Move holdout selection to Pre-Task 4 (type frequency scan + equivalence classes). Task 5 reads holdout file, doesn't create it. |
| 3 | Task 3 "eval-only" is a lie | Renamed to "Eval + View B". Split mapping into train (high-precision) and eval (broader) artifacts. |
| 4 | Dedup merges query_types (explodes negatives) | Recompute query_types from scratch after merge. Don't union negative lists. |
| 5 | confidence = max is wrong | Changed to confidence = min for merged examples. |
| 6 | B2NERD stats wrong | Curated: 51,907 (341 types). Raw: 1,419,161. Use curated first, cap at 52K. |
| 7 | View A/B mixing ratio + split ordering | Split by text first, then generate View B within each split. VIEW_B_RATIO = 1.0 cap. |
| 8 | Training format conversion missing | Added schema_to_chat() for OpenAI chat format. Output: train_chat.jsonl, val_chat.jsonl. |

### Should Fix (8 items)

| # | Fix | Resolution |
|---|-----|-----------|
| 9 | Zero-shot eval set tiny | Reserve 50-100 texts per held-out type for zero_shot_eval. Target ≥500 total. |
| 10 | Benchmark protocol underspecified | Added hardware, prompt lengths, warmup, quantization matrix, LoRA anchor config. |
| 11 | Conversation rendering spec missing | Defined canonical "A: text\nB: text" format for training input. |
| 12 | Prompt template needs duplicate handling | Added "one entry per occurrence, order by first appearance" to template. |
| 13 | Mention cardinality not preserved in Task 6 | Updated prompt to specify per-entity counts. Validation checks cardinality. |
| 14 | open-ner caps should be per-language | Replaced total caps with per-language: en 30K, ja 20K, zh 20K, ko 15K, other 15K. |
| 15 | Zero-shot holdout cross-namespace | Equivalence classes map same semantic type across label namespaces (EN/ZH/JA/KO). |
| 16 | Split stratification after merge | Define primary_source = provenance[0] for stratification. |

### Task 6 LLM Benchmark Results

| Rank | Model | Role | Clean JSON | Critical Issues |
|------|-------|------|------------|-----------------|
| 1 | Gemini | Primary (50%) | 4/4 | None |
| 2 | Codex (gpt-5.3) | Fallback | 4/4 | None (paid) |
| 3 | Kimi K2.5 | Primary (50%) | 3/4 | Inconsistent fences |
| 4 | GLM-5 | Tertiary | 2/3 | Timeout (rate limit) |
| 5 | Sonnet 4.5 | DO NOT USE | 0/4 | Context confusion |
| 6 | Trinity | DO NOT USE | 2/4 | Entity hallucination |

Fallback chain updated: gemini → opencode → codex. Added fence-stripping + hallucination gate.

### Informational

- Training framework decided: MLX-LM (local M4 Max) + MS-SWIFT (cloud Modal). Ref: ONEIRON-RESEARCH-003.
- LoRA anchor: rank 32, alpha 64, dropout 0.05, 16 layers, lr 5e-5.
- Export: MLX → GGUF → CoreML/Luminal (device); MS-SWIFT → PEFT → vLLM (server).
- Tier 0 strategy: Resolve by eval. If 0.6B zero-shot F1 < 50%, train head-only LoRA.

## Round 4

| # | Fix | Resolution |
|---|-----|-----------|
| Tiers | Model tier expansion | 4-tier ladder: 0.6B/1.7B/4B/8B. All train on same data. Eval benchmarks all sizes + quantization. |
| 1 | Zero-shot leakage via negative sampling | Holdout types excluded from NegativeSampler pool. Splitter strips entities instead of moving examples. |
| 2 | span_computer.py bugs | Occurrence-based matching (not sequential). Handles out-of-order entities, repeated substrings. |
| 3 | Rename human-gold | Task 5 eval → synthetic-gold. human-gold reserved for actual human audit. |
| 4 | type vs original_type contradiction | Dual-view training: View A (raw types) + View B (canonical types). Both in train.jsonl. |
| 5 | Task 6 Codex split contradiction | 50/50 OpenCode/Gemini primary. Codex fallback only (<5%). |
| 6 | Verify B2NERD facts | Pre-Task 4 blocking verification step. Check counts, types, ZH pos field, fiNERweb size, chinese_ner_sft offsets. |
| 7 | Task 6 preserve gold entities | LLM does creative rewrite, NOT entity extraction. Types from gold annotations. |
| 8 | Open-ner dedup kills dual-view | Merge-on-duplicate (union entities), not drop-on-duplicate. provenance field tracks sources. |
| 9 | BIO detokenization | detokenize() in bio_to_spans.py. Punctuation spacing, CJK no-space join. Prefer raw text when available. |
| 10 | Dataset mixing caps | Per-dataset caps: fiNERweb 50K/lang, open-ner-std 100K, open-ner-core 50K. v2: per-epoch sampling. |
| 11 | CJK Unicode normalization | NFKC normalization fallback in span_computer.py. Handles full-width/half-width variants. |
| 12 | Training prompt template | Defined in schema.md. Per-type Q/A format with explicit empty-list response for negatives. |

## Round 3

| # | Issue | Resolution |
|---|-------|-----------|
| 1 | Add OpenCode CLI | Third labeling backend (Kimi K2.5, free tier). Fallback chain: opencode → gemini → codex. |
| 2 | LLM span offsets unreliable | span_computer.py computes spans programmatically via str.find(). LLM prompts request surface+type only. |
| 3 | Conversation span reference | start/end relative to turns[turn_index].text, not concatenated text. |
| 4 | fiNERweb language detection | Use HF dataset config/subset name as language ID, not langdetect. |
| 5 | Cost accounting | OpenCode + Gemini = $0 (primary). Codex = paid (fallback only). |
| 6 | CLI tools in Dependencies | Listed all 3 tools with versions, models, and cost tier. |
| 7 | B2NERD pos end semantics | Verify inclusive vs exclusive end on 10 samples before processing. |

## Round 2

| # | Issue | Resolution |
|---|-------|-----------|
| 1 | B2NERD span ambiguity | Use pos field (EN) when available; first-match + ambiguous case logging (ZH). |
| 2 | fiNERweb scale dominance | Per-language cap of 50K examples. |
| 3 | Zero-shot holdout leakage | Cross-dataset enforcement — held-out types excluded from ALL sources. |
| 4 | No train/val split | Stratified 95/5 by language × source via splitter.py. |
| 5 | Cross-dataset duplicates | Exact-match + MinHash near-dedup via dedup.py. |
| 6 | Phase A scale gap | Explicit: 7K pilot vs 50K in research doc. Script designed to scale. |
| 7 | Static negative sampling | Noted as v1; dynamic per-epoch as v2 TODO. |
| 8 | open-ner CJK tokenization | Detect tokenization style; join without spaces for CJK. |
| 9 | CLI subprocess overhead | Batch 5-10 passages per CLI call. |

## Round 1 (UniversalNER)

| Aspect | Original | Updated |
|--------|----------|---------|
| Type mapping | Collapse to 10 Oneiron types | Preserve original; mapping is eval-only |
| Schema | (text) → entities | (text, query_types) → entities |
| Negative types | Not in plan | 2-5 per example, frequency-weighted |
| Eval | Standard NER metrics | + zero-shot split on held-out types |
| Training scope | Per-tier type restriction | All models train on full vocabulary |
