# Dataset Inventory

> Generated: 2026-02-12 | Version: 1.0

## Summary

- **12** active datasets (13 total, 1 skipped)
- **11,678,157** total examples across active datasets
- **7** ready for processing, **5** need extraction/labeling

## Datasets

| Dataset | Format | Size | Languages | Types | Span Format | Status |
|---------|--------|------|-----------|-------|-------------|--------|
| b2nerd_curated | json_zipped | 388,075 | EN, ZH | 492 | pos field [start, end] | needs_extraction |
| b2nerd_test | json_zipped | 407,036 | EN, ZH | 492 | pos field [start, end] | needs_extraction |
| b2nerd_raw | json_zipped | 4,262,938 | EN, ZH | 492 | pos field [start, end] | needs_extraction |
| open_ner_standardized | parquet_bio | 830,734 | 52 langs | 43 | BIO tags via ClassLabel | ready |
| open_ner_core_types | parquet_bio | 830,734 | 52 langs | 3 | BIO tags via ClassLabel | ready |
| finerweb | parquet_spans | 1,995,477 | 93 langs | 3978 | char_spans field with start/end/label objects | ready |
| klue_ner | parquet_bio | 26,008 | KO | 6 | Numeric BIO tags (0-12) in ner | ready |
| multiconer_v2 | conll_bio | 2,709,465 | 12 langs | 33 | CoNLL BIO format | ready |
| chinese_ner_sft | jsonl_spans | 182,293 | ZH | 69 | entity_label + start_idx/end_idx/entity_text fields | ready |
| stockmark_ner_ja | json_spans | 5,343 | JA | 8 | span field [start, end] | ready |
| chatharuhi_roleplaying | jsonl_dialogue | 11,558 | ZH, EN | — | none | needs_labeling |
| japanese_roleplay_dialogues | jsonl_dialogue | 28,496 | JA | — | none | needs_labeling |
| nuner | model_checkpoint | N/A | N/A | N/A | N/A | **skip** |

## Dataset Details

### b2nerd_curated

- **HF ID**: `Umean/B2NERD`
- **Local path**: `data/raw/B2NERD`
- **Format**: json_zipped
- **License**: MIT
- **Confidence**: synthetic-gold
- **Size**: 388,075 (EN: 196,043 + EN_other: 63,800 + ZH: 128,232)
- **Entity types**: 492 (AI algorithm, Abbreviation, Activity, Agreement, Anatomy, Animal, Award, Band, ...)
- **Span info**: pos field [start, end] — character offsets
- **Notes**: README states ~52K pruned samples from 54 datasets. Actual B2NERD/ train+dev files contain 388K entries (full source datasets before pruning). NER_en_other contains multiconer22 multilingual samples (11 languages).

### b2nerd_test

- **HF ID**: `Umean/B2NERD`
- **Local path**: `data/raw/B2NERD`
- **Format**: json_zipped
- **License**: MIT
- **Confidence**: synthetic-gold
- **Size**: 407,036 (EN: 121,156 + EN_other: 55,000 + ZH: 230,880)
- **Entity types**: 492 (AI algorithm, Abbreviation, Activity, Agreement, Anatomy, Animal, Award, Band, ...)
- **Span info**: pos field [start, end] — character offsets
- **Notes**: Test splits from B2NERD/ curated subset. Reserved for evaluation — do not use in training.

### b2nerd_raw

- **HF ID**: `Umean/B2NERD`
- **Local path**: `data/raw/B2NERD`
- **Format**: json_zipped
- **License**: varies_by_source
- **Confidence**: synthetic
- **Size**: 4,262,938
- **Entity types**: 492 (AI algorithm, Abbreviation, Activity, Agreement, Anatomy, Animal, Award, Band, ...)
- **Span info**: pos field [start, end] — character offsets
- **Notes**: Combines B2NERD_all/ (~2.1M standardized) and B2NERD_raw/ (~2.2M raw) from zip. README states ~1.4M for standardized subset. Reserve for v2 type coverage gaps.

### open_ner_standardized

- **HF ID**: `bltlab/open-ner-standardized`
- **Local path**: `data/raw/open-ner-standardized`
- **Format**: parquet_bio
- **License**: mixed
- **Confidence**: gold
- **Size**: 830,734 total across 52 languages (train: 624,532 + dev: 75,416 + test: 130,786)
- **Entity types**: 43 (LOC, ORG, PER, GPE, EVENT, DATE, MISC, PRODUCT, ...)
- **Span info**: BIO tags via ClassLabel — numeric codes mapped to B-TYPE/I-TYPE/O
- **Notes**: Multi-dataset collection with 66 configs across 52 languages. 43 entity types total. BIO format with HF ClassLabel feature. Total includes train (624,532) + dev (75,416) + test (130,786) across all configs.

### open_ner_core_types

- **HF ID**: `bltlab/open-ner-core-types`
- **Local path**: `data/raw/open-ner-core-types`
- **Format**: parquet_bio
- **License**: mixed
- **Confidence**: gold
- **Size**: 830,734 total across 52 languages (train: 624,532 + dev: 75,416 + test: 130,786)
- **Entity types**: 3 (LOC, ORG, PER)
- **Span info**: BIO tags via ClassLabel — numeric codes mapped to B-TYPE/I-TYPE/O
- **Notes**: Same sentences as open-ner-standardized but mapped to 3 core types only (PER, LOC, ORG). 66 configs, 52 languages. Total includes train (624,532) + dev (75,416) + test (130,786) across all configs.

### finerweb

- **HF ID**: `whoisjones/fiNERweb`
- **Local path**: `data/raw/fiNERweb`
- **Format**: parquet_spans
- **License**: CC-BY-4.0
- **Confidence**: synthetic-gold
- **Size**: 1,995,477 total across 93 languages
- **Entity types**: 3978 (abbreviation, abstract concept, academic conference, academic degree, academic discipline, animal, award, book, ...)
- **Span info**: char_spans field with start/end/label objects — native character offsets
- **Notes**: 93 languages, 3978 entity types. One parquet file per language. char_spans has list of {start, end, label} dicts. Very fine-grained types (e.g. 'location / theater').

### klue_ner

- **HF ID**: `klue/klue`
- **Local path**: `data/raw/klue`
- **Format**: parquet_bio
- **License**: CC-BY-SA-4.0
- **Confidence**: gold
- **Size**: 26,008 (train: 21,008, validation: 5,000)
- **Entity types**: 6 (PS, LC, OG, DT, TI, QT)
- **Span info**: Numeric BIO tags (0-12) in ner_tags column. Entity types in sentence markup as <text:TYPE>. PS=person, LC=location, OG=organization, DT=date, TI=time, QT=quantity.
- **Notes**: Korean NER benchmark. Train and validation splits only (no test). Character-level tokenization with numeric BIO tags.

### multiconer_v2

- **HF ID**: `MultiCoNER/multiconer_v2`
- **Local path**: `data/raw/multiconer_v2`
- **Format**: conll_bio
- **License**: CC-BY-4.0
- **Confidence**: gold
- **Size**: 2,709,465 total across 12 languages
- **Entity types**: 33 (AerospaceManufacturer, AnatomicalStructure, ArtWork, Artist, Athlete, CarManufacturer, Cleric, Clothing, ...)
- **Span info**: CoNLL BIO format — token per line with B-TYPE/I-TYPE/O tags. Blank lines separate sentences.
- **Notes**: 2.7M sentences across 13 language directories (12 languages + MULTI). CoNLL format with train/dev/test splits per language. Sentence count includes MULTI subset.

### chinese_ner_sft

- **HF ID**: `qgyd2021/chinese_ner_sft`
- **Local path**: `data/raw/chinese_ner_sft`
- **Format**: jsonl_spans
- **License**: Apache-2.0
- **Confidence**: gold
- **Size**: 182,293 total across 12 subsets
- **Entity types**: 69 (Abstract, BANK, COMMENTS_ADJ, COMMENTS_N, CONT, EDU, GPE, GPE.NAM, ...)
- **Span info**: entity_label + start_idx/end_idx/entity_text fields — character offsets. Some null values present.
- **Notes**: 12 Chinese NER subsets aggregated. 69 entity types including domain-specific (medical, financial). Has null values in some entity fields. entity_label field (not entity_type).

### stockmark_ner_ja

- **HF ID**: `stockmark/ner-wikipedia-dataset`
- **Local path**: `data/raw/stockmark-ner-ja`
- **Format**: json_spans
- **License**: CC-BY-SA-4.0
- **Confidence**: gold
- **Size**: 5,343
- **Entity types**: 8 (人名, 地名, 法人名, その他の組織名, イベント名, 政治的組織名, 施設名, 製品名)
- **Span info**: span field [start, end] — character offsets. Entities have name/span/type.
- **Notes**: Japanese Wikipedia NER. 8 types: 人名 (person), 地名 (location), 法人名 (corporation), その他の組織名 (other org), イベント名 (event), 政治的組織名 (political org), 施設名 (facility), 製品名 (product). Each entry has curid/text/entities.

### chatharuhi_roleplaying

- **HF ID**: `silk-road/ChatHaruhi-RolePlaying`
- **Local path**: `data/raw/ChatHaruhi-RolePlaying`
- **Format**: jsonl_dialogue
- **License**: Apache-2.0
- **Confidence**: unlabeled
- **Size**: 11,558
- **Entity types**: none
- **Span info**: none — needs LLM labeling
- **Notes**: 35 character JSONL files with text + embeddings (luotuo_openai, bge_zh_s15). Roleplay dialogues, no NER annotations. Needs LLM-based entity extraction.

### japanese_roleplay_dialogues

- **HF ID**: `OmniAICreator/Japanese-Roleplay-Dialogues`
- **Local path**: `data/raw/Japanese-Roleplay-Dialogues`
- **Format**: jsonl_dialogue
- **License**: CC-BY-NC-4.0
- **Confidence**: unlabeled
- **Size**: 28,496
- **Entity types**: none
- **Span info**: none — needs LLM labeling
- **Notes**: 2 JSONL files (Filtered: 4324, Original: 24172). Each entry has id/title/first_post/first_poster/posts array. Roleplay dialogues, no NER annotations.

### nuner

- **HF ID**: `numind/NuNER`
- **Local path**: `data/raw/NuNER`
- **Format**: model_checkpoint
- **License**: MIT
- **Confidence**: N/A
- **Size**: 0
- **Entity types**: none
- **Span info**: N/A
- **Notes**: Model checkpoint (pytorch_model.bin, config.json, tokenizer), NOT a dataset. Excluded from processing pipeline.

## Discrepancies from Plan

| Dataset | Plan Value | Actual Value | Notes |
|---------|-----------|--------------|-------|
| B2NERD curated | 51,907 | 388,075 (train+dev) | README ~52K is pruned count; actual files have full source entries |
| B2NERD test | — | 407,036 | Test splits separated for evaluation |
| B2NERD raw | 1,419,161 | 4,262,938 | Covers B2NERD_all/ + B2NERD_raw/ directories |
| fiNERweb types | 235K+ | 3,978 | Plan overestimated; actual unique label count is 3,978 |
| multiconer_v2 | ~170K | 2,709,465 | Plan severely underestimated; actual sentence count from # id lines |
| open-ner-standardized | ~830K | 830,734 | Was 755,318 (train only); corrected to include all splits |
| open-ner types | ~60 | 43 | Actual unique entity types from ClassLabel feature |
| stockmark types | 8 | 8 | All 8 types confirmed: 人名, 地名, 法人名, その他の組織名, イベント名, 政治的組織名, 施設名, 製品名 |
| japanese_roleplay | ~4.3K | 28,496 | Plan counted filtered only; actual includes Original (24K) |
