---
type: changelog
project: oneiron-ner
---

# Plan Change Log

## Round 4 (Current)

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
