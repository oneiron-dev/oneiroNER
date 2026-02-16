# NER Dataset Research Report
## Additional Datasets for Korean, German, French, and Italian

**Research Date:** 2026-02-16
**Purpose:** Identify additional NER datasets on HuggingFace to supplement oneiroNER project coverage

---

## Executive Summary

### Current Coverage Analysis
- **Korean (86K):** Weakest diversity with only 2 real sources (KLUE NER, fiNERweb). Highest priority for expansion.
- **French (54K):** Lowest total count with only 2 sources. Second priority.
- **German (78K):** Good diversity with 4 sources.
- **Italian (87K):** Good diversity with 3 sources.

### Key Findings
- **Korean:** Found 4 promising additional datasets (kor_ner, polyglot_ner, NIKL, KBMC)
- **French:** Found 3 major datasets (frenchNER_3entities, WikiNER-fr, WikiNEuRal-fr)
- **German:** Found 3 additional datasets (GermEval 2014, german-ler, WikiNEuRal-de)
- **Italian:** Found 2 additional datasets (WikiNEuRal-it, MultiNERD-it)
- **Multilingual:** Several large multilingual datasets cover multiple target languages

---

## KOREAN DATASETS (HIGHEST PRIORITY)

### 1. nlp-kmu/kor_ner ⭐ RECOMMENDED
- **HuggingFace ID:** `nlp-kmu/kor_ner`
- **Size:** 1K-10K records (exact count TBD, likely ~5-8K based on typical range)
- **Entity Types:** 5 types using BIO tagging
  - PS (Person)
  - OG (Organization)
  - LC (Location)
  - DT (Date)
  - TI (Time)
- **Format:** Token classification with structured fields (text, tokens, pos_tags, ner_tags)
- **License:** MIT ✓ (permissive)
- **Content Type:** Passage-based (not conversational)
- **Quality:** Expert-generated annotations, maintained by Korea Maritime and Ocean University
- **Source:** https://github.com/kmounlp/NER
- **Notes:**
  - Dataset viewer disabled (requires loading script)
  - Also known as KMOUNLP-NER, openly available since 2016, officially published 2021
  - Compatible entity types with KLUE NER (both use PS, LC, OG, DT, TI)

### 2. polyglot_ner (Korean subset)
- **HuggingFace ID:** `polyglot_ner` (config: "ko")
- **Size:** Unknown (part of 40-language dataset)
- **Entity Types:** 3 basic types
  - PER (Person)
  - LOC (Location)
  - ORG (Organization)
- **Format:** Token classification, BIO tagging
- **License:** Check dataset page
- **Content Type:** Passage-based
- **Quality:** Silver-standard, auto-generated from Wikipedia and Freebase
- **Notes:**
  - Part of large multilingual dataset covering 40 languages
  - Training data automatically generated, not human-annotated
  - May have lower quality than manually annotated datasets
  - Simpler entity schema (3 types vs 5+ in other Korean datasets)

### 3. KETI-AIR/nikl (National Institute of Korean Language) ⭐ RECOMMENDED
- **HuggingFace ID:** `KETI-AIR/nikl` (config: "ne.v1.0")
- **Size:** Unknown (requires manual download)
- **Entity Types:** 9 types
  - PS (Person)
  - LC (Location)
  - OG (Organization)
  - DT (Date)
  - TI (Time)
  - QT (Quantity)
  - AF (Artifact)
  - CV (Civilization)
  - AM (Unknown)
- **Format:** Token classification
- **License:** Requires permission from National Institute of Korean Language
- **Content Type:** Passage-based
- **Quality:** High quality, official government corpus
- **Access:** Must download from National Institute of Korean Language (국립국어원) and use manual_dir parameter
- **Notes:**
  - Most comprehensive Korean entity schema (9 types)
  - Requires registration and approval from NIKL
  - Not directly downloadable from HuggingFace
  - Dataset viewer disabled
  - **LEGAL NOTE:** Verify licensing terms before use in commercial project

### 4. Korean Bio-Medical Corpus (KBMC)
- **HuggingFace ID:** Not found on HuggingFace (may be on GitHub or arXiv)
- **Size:** Unknown
- **Entity Types:** Medical domain entities (specific types TBD)
- **Format:** Token classification
- **License:** Check paper (arXiv:2403.16158)
- **Content Type:** Medical/biomedical passages
- **Quality:** Human-verified, constructed with ChatGPT assistance (2024)
- **Notes:**
  - First open-source medical NER dataset for Korean
  - Published March 2024
  - Shows 20% improvement in medical NER performance
  - Domain-specific (medical), may not be suitable for general NER
  - **Availability uncertain** - not found on HuggingFace during search

### 5. KLUE-NER (already in use - included for reference)
- **HuggingFace ID:** `klue/klue` (config: "ner")
- **Size:** 31K sentences
- **Entity Types:** 6 types
  - Person
  - Location
  - Organization
  - Date
  - Time
  - Quantity
- **License:** CC-BY-SA-4.0
- **Quality:** High quality, part of official Korean benchmark
- **Notes:** Already in use with 47K records in oneiroNER

---

## FRENCH DATASETS (SECOND PRIORITY)

### 1. CATIE-AQ/frenchNER_3entities ⭐ RECOMMENDED
- **HuggingFace ID:** `CATIE-AQ/frenchNER_3entities`
- **Size:** 420,264 total rows
  - Train: 346,071
  - Validation: 32,951
  - Test: 41,242
- **Entity Types:** 3 types (IOB format)
  - PER (Person)
  - ORG (Organization)
  - LOC (Location)
- **Format:** Token classification (tokens, ner_tags, dataset source)
- **License:** CC-BY-4.0 ✓ (permissive)
- **Content Type:** Passage-based
- **Quality:**
  - Consolidated from 5 open-source datasets
  - Duplicates and leaks removed
  - 8 test lines (0.019%) failed deduplication
- **Source Datasets:**
  1. Multiconer: 15,721 train / 827 val / 857 test
  2. Multinerd: 140,880 train / 17,610 val / 17,695 test
  3. Pii-masking-200k: 61,958 train
  4. Wikiann: 20,000 train / 10,000 val / 10,000 test
  5. Wikiner: 113,296 train / 5,994 val / 13,393 test
- **Notes:**
  - **Very large dataset** (420K rows, much larger than current 54K French coverage)
  - Well-curated with deduplication
  - Includes dataset source field for tracking provenance

### 2. CATIE-AQ/frenchNER_4entities
- **HuggingFace ID:** `CATIE-AQ/frenchNER_4entities`
- **Size:** 384,773 total rows
  - Train: 328,757
  - Validation: 24,131
  - Test: 31,885
- **Entity Types:** 4 types (IOB format)
  - PER, ORG, LOC (same as above)
  - MISC (Miscellaneous)
- **Format:** Token classification
- **License:** CC-BY-4.0 ✓
- **Content Type:** Passage-based
- **Quality:** Similar to 3entities version, with MISC category added
- **Notes:**
  - Slightly smaller than 3entities version due to MISC filtering
  - Use if MISC category is needed, otherwise prefer 3entities for larger size

### 3. Jean-Baptiste/wikiner_fr
- **HuggingFace ID:** `Jean-Baptiste/wikiner_fr`
- **Size:** ~134,092 rows (~170,634 sentences commonly cited)
- **Entity Types:** 5 types
  - PER, ORG, LOC, MISC (standard)
  - Plus one additional type (TBD)
- **Format:** Token classification
- **License:** Check dataset page
- **Content Type:** Passage-based (Wikipedia-derived)
- **Quality:** Silver-standard, auto-generated from Wikipedia
- **Notes:**
  - Used to train popular camembert-ner model (2M+ downloads)
  - Already included in CATIE-AQ/frenchNER_3entities as source
  - **Recommend using CATIE-AQ version** instead (already deduplicated and merged)

### 4. danrun/WikiNER-fr-gold
- **HuggingFace ID:** `danrun/WikiNER-fr-gold`
- **Size:** 26,818 sentences (~700K tokens, 20% sample of wikiner_fr)
- **Entity Types:** Same as wikiner_fr
- **Format:** Token classification
- **License:** Check dataset page (paper published 2024)
- **Content Type:** Passage-based
- **Quality:** Gold-standard, manually revised version of WikiNER-fr
- **Notes:**
  - Higher quality than silver-standard wikiner_fr
  - Smaller size (only 20% of original)
  - Published November 2024 (very recent)
  - Good for validation/testing due to high quality

### 5. Babelscape/wikineural (French subset)
- **HuggingFace ID:** `Babelscape/wikineural` (config: "fr")
- **Size:** 127K sentences, 3.24M tokens
  - PER: 76K entities
  - ORG: 25K entities
  - LOC: 101K entities
  - MISC: 29K entities
- **Entity Types:** 4 types (IOB2 format)
  - PER, ORG, LOC, MISC
- **Format:** Parquet, token classification (tokens, ner_tags, lang)
- **License:** CC-BY-NC-SA-4.0 ⚠️ (non-commercial only)
- **Content Type:** Passage-based (Wikipedia-derived)
- **Quality:** Machine-generated using BabelNet + BERT, shows 6-point F1 improvement over alternatives
- **Notes:**
  - **NON-COMMERCIAL license** - check if compatible with project
  - High quality for machine-generated data
  - Large coverage (127K sentences)

---

## GERMAN DATASETS (LOWER PRIORITY)

### 1. GermanEval/germeval_14 ⭐ RECOMMENDED
- **HuggingFace ID:** `GermanEval/germeval_14`
- **Size:** ~31,000 sentences, ~590,000 tokens
  - Train: 24,000 sentences
  - Validation: 2,200 sentences
  - Test: 5,100 sentences
- **Entity Types:** 12 classes with NoSta-D guidelines
  - Main: PER, LOC, ORG, OTH
  - Subtypes: -deriv (derivations), -part (compounds)
  - **Special:** Supports nested NER (e.g., [ORG FC Kickers [LOC Darmstadt]])
- **Format:** Token classification with nested annotations (tokens, ner_tags, nested_ner_tags)
- **License:** CC-BY-SA-4.0 ✓
- **Content Type:** Passage-based
- **Quality:** Professional annotation, German Wikipedia and news sources
- **Notes:**
  - Benchmark dataset for German NER
  - Unique nested entity support
  - Fine-grained entity classification with subtypes
  - Dataset viewer disabled (requires loading script)

### 2. elenanereiss/german-ler (Legal Entity Recognition)
- **HuggingFace ID:** `elenanereiss/german-ler`
- **Size:** ~67,000 sentences, ~54,000 entities
- **Entity Types:** 19 fine-grained legal entity classes
- **Format:** Token classification
- **License:** Check dataset page
- **Content Type:** Legal domain (German federal court decisions)
- **Quality:** Human-annotated
- **Notes:**
  - Domain-specific (legal)
  - Very fine-grained entity schema (19 classes)
  - May not be suitable for general NER
  - Good for legal domain specialization

### 3. Babelscape/wikineural (German subset)
- **HuggingFace ID:** `Babelscape/wikineural` (config: "de")
- **Size:** 124K sentences, 2.19M tokens
  - PER: 60K entities
  - ORG: 32K entities
  - LOC: 59K entities
  - MISC: 25K entities
- **Entity Types:** 4 types (IOB2 format)
  - PER, ORG, LOC, MISC
- **Format:** Parquet, token classification
- **License:** CC-BY-NC-SA-4.0 ⚠️ (non-commercial only)
- **Content Type:** Passage-based (Wikipedia)
- **Quality:** Machine-generated using BabelNet + BERT
- **Notes:**
  - **NON-COMMERCIAL license**
  - Large coverage (124K sentences)
  - Same quality notes as French subset

### 4. tner/multinerd (German subset)
- **HuggingFace ID:** `tner/multinerd` (config: "de")
- **Size:** 156,792 examples
- **Entity Types:** 18 fine-grained types
  - Standard: PER, LOC, ORG
  - Extended: ANIM, BIO, CEL, DIS, EVE, FOOD, INST, MEDIA, PLANT, MYTH, TIME, VEHI, MISC, SUPER, PHY
- **Format:** Parquet, IOB2 tagging
- **License:** Check dataset page (likely permissive, from ACL 2022)
- **Content Type:** Multi-genre (Wikipedia, WikiNews)
- **Quality:** Auto-generated with manually-annotated test set
- **Notes:**
  - **Very fine-grained** entity types (18 categories)
  - Includes disambiguation and image URLs
  - Multi-genre coverage
  - Large dataset (156K examples)

---

## ITALIAN DATASETS (LOWER PRIORITY)

### 1. Babelscape/wikineural (Italian subset)
- **HuggingFace ID:** `Babelscape/wikineural` (config: "it")
- **Size:** 111K sentences, 2.99M tokens
  - PER: 67K entities
  - ORG: 22K entities
  - LOC: 97K entities
  - MISC: 26K entities
- **Entity Types:** 4 types (IOB2 format)
  - PER, ORG, LOC, MISC
- **Format:** Parquet, token classification
- **License:** CC-BY-NC-SA-4.0 ⚠️ (non-commercial only)
- **Content Type:** Passage-based (Wikipedia)
- **Quality:** Machine-generated using BabelNet + BERT
- **Notes:**
  - **NON-COMMERCIAL license**
  - Large coverage (111K sentences)
  - Same architecture as French/German subsets

### 2. tner/multinerd (Italian subset) ⭐ RECOMMENDED
- **HuggingFace ID:** `tner/multinerd` (config: "it")
- **Size:** 181,927 examples (largest among target languages)
- **Entity Types:** 18 fine-grained types (same as German subset)
  - Standard: PER, LOC, ORG
  - Extended: ANIM, BIO, CEL, DIS, EVE, FOOD, INST, MEDIA, PLANT, MYTH, TIME, VEHI, MISC, SUPER, PHY
- **Format:** Parquet, IOB2 tagging
- **License:** ACL 2022, likely permissive
- **Content Type:** Multi-genre (Wikipedia, WikiNews)
- **Quality:** Auto-generated with manually-annotated test set
- **Notes:**
  - **Largest Italian dataset found** (181K examples)
  - Fine-grained entity types
  - Multi-genre coverage
  - Includes disambiguation and multimodal features

### 3. WikiNER Italian (via training instances)
- **HuggingFace ID:** Not clearly specified, may be part of `Babelscape/wikineural`
- **Size:** 102,352 training instances, 25,588 test instances
- **Entity Types:** Standard IOB2 (likely PER, LOC, ORG, MISC)
- **Format:** Token classification
- **License:** Check specific dataset
- **Content Type:** Passage-based (Wikipedia)
- **Quality:** Auto-generated
- **Notes:**
  - May overlap with WikiNEuRal Italian subset
  - Specific HuggingFace ID unclear from search results

---

## MULTILINGUAL DATASETS (COVER MULTIPLE TARGET LANGUAGES)

### 1. universalner/universal_ner
- **HuggingFace ID:** `universalner/universal_ner`
- **Languages Covered:** 13 languages including **German** only
  - ✓ German (de)
  - ✗ Korean, French, Italian NOT included
- **Size:** TBD per language (specific numbers not provided)
- **Entity Types:** 3 basic types (BIO tagging)
  - PER, LOC, ORG
- **Format:** Token classification (idx, text, tokens, ner_tags, annotator)
- **License:** CC-BY-SA-4.0 ✓
- **Content Type:** Passage-based
- **Quality:** Gold-standard, community-driven with native speakers, cross-lingually consistent
- **Notes:**
  - NAACL 2024 benchmark
  - Based on Universal Dependencies corpora
  - High quality annotations
  - Only German covered from target languages
  - Limited entity types (3 only)

### 2. Babelscape/wikineural
- **HuggingFace ID:** `Babelscape/wikineural`
- **Languages Covered:** 9 languages including German, French, Italian
  - ✓ German (de): 124K sentences
  - ✓ French (fr): 127K sentences
  - ✓ Italian (it): 111K sentences
  - ✗ Korean NOT included
  - Also: en, es, nl, pl, pt, ru
- **Size:** See individual language sections above
- **Entity Types:** 4 types (PER, ORG, LOC, MISC)
- **License:** CC-BY-NC-SA-4.0 ⚠️ (non-commercial only)
- **Notes:**
  - Covers 3 of 4 target languages (no Korean)
  - Large dataset with good coverage
  - **NON-COMMERCIAL license is major limitation**

### 3. tner/multinerd (MultiNERD)
- **HuggingFace ID:** `tner/multinerd`
- **Languages Covered:** 10 languages including German, French, Italian
  - ✓ German (de): 156,792 examples
  - ✓ French (fr): 176,185 examples
  - ✓ Italian (it): 181,927 examples
  - ✗ Korean NOT included
  - Also: en, es, nl, pl, pt, ru, zh
- **Size:** 1,479,336 total rows across all languages
- **Entity Types:** 18 fine-grained types
- **License:** ACL 2022 (likely permissive, verify)
- **Format:** Parquet, IOB2
- **Content Type:** Multi-genre (Wikipedia, WikiNews)
- **Quality:** Auto-generated with manual test set validation
- **Notes:**
  - **Most comprehensive multilingual dataset** for German/French/Italian
  - Fine-grained entity schema (18 types)
  - Large coverage for all three languages
  - Includes disambiguation and multimodal features
  - No Korean support

### 4. polyglot_ner
- **HuggingFace ID:** `polyglot_ner`
- **Languages Covered:** 40 languages including Korean
  - ✓ Korean (ko): size unknown
  - ? German, French, Italian (not explicitly confirmed but likely included)
- **Entity Types:** 3 basic types (PER, LOC, ORG)
- **Format:** BIO tagging
- **License:** Check dataset page
- **Quality:** Silver-standard, auto-generated from Wikipedia + Freebase
- **Notes:**
  - Broadest language coverage (40 languages)
  - Lower quality (auto-generated)
  - Simple entity schema
  - Good for Korean supplementation despite lower quality

### 5. DFKI-SLT/cross_ner (CrossNER)
- **HuggingFace ID:** `DFKI-SLT/cross_ner`
- **Languages:** English only
- **Domains:** 5 domains (Politics, Science, Music, Literature, AI)
- **Size:** 10K-100K examples across domains
- **Notes:**
  - **NOT MULTILINGUAL** - English only
  - Domain-specific entity categories
  - Not suitable for target languages

### 6. HuiHuang/NER-CoT
- **HuggingFace ID:** `HuiHuang/NER-CoT`
- **Languages:** English only
- **Size:** 45,800 rows
- **License:** CC-BY-NC-4.0 (non-commercial)
- **Notes:**
  - **NOT MULTILINGUAL** - English only
  - Chain-of-thought reasoning annotations (AAAI 2026)
  - Not suitable for target languages

---

## RECOMMENDATIONS BY PRIORITY

### KOREAN (Highest Priority)
1. **nlp-kmu/kor_ner** - Add immediately (MIT license, expert annotations, 5-8K records)
2. **KETI-AIR/nikl** - Pursue if licensing allows (9 entity types, official corpus)
3. **polyglot_ner (ko subset)** - Add for coverage despite lower quality
4. **KBMC** - Skip unless medical domain needed (specialized, availability unclear)

**Estimated Addition:** +5-15K records (excluding NIKL due to licensing)

### FRENCH (Second Priority)
1. **CATIE-AQ/frenchNER_3entities** - Add immediately (CC-BY-4.0, 420K records, well-curated)
   - This single dataset would increase French coverage by ~7.8x (from 54K to 474K)
2. **danrun/WikiNER-fr-gold** - Add for validation/test (gold standard, 26K sentences)
3. **Babelscape/wikineural (fr)** - Consider if license allows (127K sentences, NC license)

**Estimated Addition:** +420-550K records

### GERMAN (Lower Priority - Already Good Coverage)
1. **GermanEval/germeval_14** - Add for benchmark compatibility (31K sentences, nested NER)
2. **tner/multinerd (de)** - Add for fine-grained types (156K examples, 18 entity types)
3. **Babelscape/wikineural (de)** - Consider if license allows (124K sentences, NC license)

**Estimated Addition:** +150-200K records

### ITALIAN (Lower Priority - Already Good Coverage)
1. **tner/multinerd (it)** - Add immediately (181K examples, 18 entity types, largest found)
2. **Babelscape/wikineural (it)** - Consider if license allows (111K sentences, NC license)

**Estimated Addition:** +180-290K records

---

## LICENSE CONSIDERATIONS

### Permissive Licenses (Safe to Use)
- MIT: nlp-kmu/kor_ner
- CC-BY-4.0: CATIE-AQ/frenchNER_3entities, CATIE-AQ/frenchNER_4entities
- CC-BY-SA-4.0: GermanEval/germeval_14, universalner/universal_ner

### Restrictive Licenses (Check Project Compatibility)
- **CC-BY-NC-SA-4.0** (Non-Commercial): Babelscape/wikineural (all subsets)
  - Cannot use for commercial purposes
  - Check if oneiroNER project has commercial components
- **NIKL Corpus**: Requires permission from National Institute of Korean Language
  - May have usage restrictions
  - Verify terms before use

### Unknown/TBD Licenses
- polyglot_ner
- Jean-Baptiste/wikiner_fr
- elenanereiss/german-ler
- tner/multinerd (likely permissive from ACL 2022)

**Action:** Verify licenses on HuggingFace dataset pages before integration

---

## DATASETS NOT FOUND / NOT SUITABLE

### Not Found on HuggingFace
- KBMC (Korean Bio-Medical Corpus) - May be on arXiv/GitHub only
- Specific "KorQA" or "KB-NER" datasets
- LDC Korean NER datasets (LDC uses separate platform)

### Not Suitable for Target Languages
- DFKI-SLT/few-nerd - English only, not multilingual
- DFKI-SLT/cross_ner - English only, domain-specific
- HuiHuang/NER-CoT - English only, reasoning-focused
- ACE2004/ACE2005 - Primarily English, restricted LDC license

### Conversational NER
- No Korean conversational NER datasets found on HuggingFace
- Salesforce/dialogstudio exists but doesn't include Korean NER
- Most datasets are passage-based, not dialogue-based

---

## QUALITY ASSESSMENT SUMMARY

### Human-Annotated (Highest Quality)
- nlp-kmu/kor_ner (expert-generated)
- KETI-AIR/nikl (official government corpus)
- GermanEval/germeval_14 (professional annotation)
- elenanereiss/german-ler (human-annotated)
- danrun/WikiNER-fr-gold (manually revised)
- universalner/universal_ner (community-driven, native speakers)

### Auto-Generated + Manual Test Set (Medium-High Quality)
- tner/multinerd (all subsets)

### Machine-Generated (Medium Quality)
- Babelscape/wikineural (all subsets) - uses BabelNet + BERT, shows quality improvements
- CATIE-AQ/frenchNER_3entities - consolidated from mixed sources, deduplicated

### Silver-Standard (Lower Quality)
- polyglot_ner (all subsets) - auto-generated from Wikipedia + Freebase
- Jean-Baptiste/wikiner_fr - Wikipedia-derived

---

## TECHNICAL NOTES

### Dataset Viewer Disabled
Several datasets have disabled viewers on HuggingFace (require loading scripts):
- nlp-kmu/kor_ner
- KETI-AIR/nikl
- GermanEval/germeval_14

These can still be loaded programmatically via `datasets.load_dataset()`.

### Manual Download Required
- KETI-AIR/nikl requires downloading from NIKL website and using `data_dir` parameter

### Format Compatibility
All recommended datasets use:
- Token classification format
- BIO or IOB2 tagging scheme
- Compatible with Hugging Face `datasets` library
- Parquet or JSON formats

---

## SOURCES

### Korean Datasets
- [nlp-kmu/kor_ner on Hugging Face](https://huggingface.co/datasets/nlp-kmu/kor_ner)
- [polyglot_ner on Hugging Face](https://huggingface.co/datasets/polyglot_ner)
- [KETI-AIR/nikl on Hugging Face](https://huggingface.co/datasets/KETI-AIR/nikl)
- [Korean Bio-Medical Corpus (KBMC) paper](https://arxiv.org/abs/2403.16158)
- [KLUE Benchmark on Hugging Face](https://huggingface.co/datasets/klue/klue)
- [KLUE GitHub Repository](https://github.com/KLUE-benchmark/KLUE)
- [A Survey on Awesome Korean NLP Datasets](https://arxiv.org/pdf/2112.01624)
- [Open Korean Corpora GitHub](https://github.com/ko-nlp/Open-korean-corpora)
- [Korean NER PyTorch GitHub](https://github.com/monologg/korean-ner-pytorch)

### French Datasets
- [CATIE-AQ/frenchNER_3entities on Hugging Face](https://huggingface.co/datasets/CATIE-AQ/frenchNER_3entities)
- [CATIE-AQ/frenchNER_4entities on Hugging Face](https://huggingface.co/datasets/CATIE-AQ/frenchNER_4entities)
- [Jean-Baptiste/wikiner_fr on Hugging Face](https://huggingface.co/datasets/Jean-Baptiste/wikiner_fr)
- [danrun/WikiNER-fr-gold on Hugging Face](https://huggingface.co/datasets/danrun/WikiNER-fr-gold)
- [Jean-Baptiste/camembert-ner model](https://huggingface.co/Jean-Baptiste/camembert-ner)
- [cmarkea/distilcamembert-base-ner model](https://huggingface.co/cmarkea/distilcamembert-base-ner)

### German Datasets
- [GermanEval/germeval_14 on Hugging Face](https://huggingface.co/datasets/GermanEval/germeval_14)
- [elenanereiss/german-ler on Hugging Face](https://huggingface.co/datasets/elenanereiss/german-ler)
- [Legal NER with Hugging Face Transformers GitHub](https://github.com/elenanereiss/bert-legal-ner)
- [flair/ner-german-legal model](https://huggingface.co/flair/ner-german-legal)
- [HFforLegal organization](https://huggingface.co/HFforLegal)

### Italian Datasets
- [Italian NLP Resources Collection](https://huggingface.co/collections/gsarti/italian-nlp-resources)
- [DeepMount00/Italian_NER_XXL model](https://huggingface.co/DeepMount00/Italian_NER_XXL)
- [Evalita-LLM paper](https://arxiv.org/html/2502.02289v1)
- [Musixmatch/umberto-commoncrawl-cased-v1](https://huggingface.co/Musixmatch/umberto-commoncrawl-cased-v1)

### Multilingual Datasets
- [Babelscape/wikineural on Hugging Face](https://huggingface.co/datasets/Babelscape/wikineural)
- [tner/multinerd on Hugging Face](https://huggingface.co/datasets/tner/multinerd)
- [universalner/universal_ner on Hugging Face](https://huggingface.co/datasets/universalner/universal_ner)
- [WikiNEuRal GitHub Repository](https://github.com/Babelscape/wikineural)
- [MultiNERD GitHub Repository](https://github.com/Babelscape/multinerd)

### Other Resources
- [DFKI-SLT/cross_ner on Hugging Face](https://huggingface.co/datasets/DFKI-SLT/cross_ner)
- [HuiHuang/NER-CoT on Hugging Face](https://huggingface.co/datasets/HuiHuang/NER-CoT)
- [Entity Recognition Datasets GitHub Collection](https://github.com/juand-r/entity-recognition-datasets)
- [ACE 2005 Multilingual Training Corpus](https://catalog.ldc.upenn.edu/LDC2006T06)

---

## NEXT STEPS

1. **Immediate Actions (Korean)**
   - Download and test nlp-kmu/kor_ner integration
   - Test polyglot_ner Korean subset quality
   - Research NIKL licensing requirements

2. **Immediate Actions (French)**
   - Download CATIE-AQ/frenchNER_3entities (largest single addition)
   - Consider danrun/WikiNER-fr-gold for high-quality test set

3. **License Verification**
   - Check Babelscape/wikineural license compatibility with project
   - Verify tner/multinerd license terms
   - Contact NIKL about corpus usage rights

4. **Integration Testing**
   - Test entity type mapping compatibility with existing oneiroNER schema
   - Verify deduplication across new and existing datasets
   - Check for overlaps between datasets (e.g., wikiner_fr in frenchNER_3entities)

5. **Future Research**
   - Search for Korean conversational NER datasets outside HuggingFace
   - Monitor for new multilingual NER datasets (2026 conferences)
   - Investigate domain-specific datasets if needed (medical, legal, etc.)
