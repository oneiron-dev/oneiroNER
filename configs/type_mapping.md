# Type Mapping Documentation

> Ontology mapping from source dataset entity types to Oneiron canonical types.

## Overview

Two mapping files serve different consumers:

- **`type_mapping_train.json`** — High-precision only. Used by `convert_all.py` (Task 4) for View B generation. Wrong canonical labels poison training data, so only unambiguous mappings are included.
- **`type_mapping_eval.json`** — Broader coverage. Used by `eval_ner.py` (Task 5) for cross-dataset scoring. Includes everything in train plus fuzzy/domain-specific types mapped to OTHER or ACTIVITY.

**Invariant**: Train mapping is a strict subset of eval mapping. Every key→value pair in train also appears in eval.

**Guiding principle**: "When in doubt, omit a View B mapping. A missing View B copy is better than a wrong one."

Both files use a flat `{ "source_type": "CANONICAL" }` format — no nesting, no confidence scores. Each consumer loads the right file directly.

## Canonical Types

| Type | Tier | Definition | Train | Eval |
|------|------|-----------|-------|------|
| PERSON | Tier 0+ | People, named individuals, roles referring to specific persons | Yes | Yes |
| PLACE | Tier 0+ | Locations, GPEs, facilities, settlements, addresses | Yes | Yes |
| ORG | Tier 0+ | Organizations, companies, groups, government bodies | Yes | Yes |
| DATE | Tier 0+ | Dates, times, temporal expressions | Yes | Yes |
| EMOTION | Tier 1+ | Emotion expressions (source: nlpcc_2018_task4) | Yes | Yes |
| ACTIVITY | Tier 2+ | Activities, games, songs — too fuzzy for training | No | Yes |
| OTHER | — | Catch-all for domain-specific types (disease, chemical, etc.) | No | Yes |

## Per-Dataset Mapping

### open_ner_standardized (60 types)

| Source Type | Canonical | File |
|------------|-----------|------|
| PER | PERSON | train+eval |
| LOC | PLACE | train+eval |
| ORG | ORG | train+eval |
| GPE | PLACE | train+eval |
| GPE-LOC | PLACE | train+eval |
| GPE-ORG | ORG | train+eval |
| DATE | DATE | train+eval |
| TIME | DATE | train+eval |
| FACILITY | PLACE | train+eval |
| CORPORATION | ORG | train+eval |
| PER-DERIV | PERSON | eval only |
| LOC-DERIV | PLACE | eval only |
| ORG-DERIV | ORG | eval only |
| MISC-DERIV | OTHER | eval only |
| EVENT | OTHER | eval only |
| MISC | OTHER | eval only |
| CARDINAL | OTHER | eval only |
| MONEY | OTHER | eval only |
| PERCENT | OTHER | eval only |
| ORDINAL | OTHER | eval only |
| NORP | OTHER | eval only |
| QUANTITY | OTHER | eval only |
| LAW | OTHER | eval only |
| LANG | OTHER | eval only |
| DESIGNATION | OTHER | eval only |
| TITLE_AFFIX | OTHER | eval only |
| RELIGION | OTHER | eval only |
| DISEASE | OTHER | eval only |
| PRODUCT | OTHER | eval only |
| CREATIVE_WORK | OTHER | eval only |

Remaining types (if any beyond these 30) are not mapped — handled by runtime heuristics or omitted.

### open_ner_core_types (3 types)

All 3 types are subsets of open_ner_standardized mappings: PER → PERSON, LOC → PLACE, ORG → ORG.

### klue_ner (6 types)

| Source Type | Canonical | File |
|------------|-----------|------|
| PS | PERSON | train+eval |
| LC | PLACE | train+eval |
| OG | ORG | train+eval |
| DT | DATE | train+eval |
| TI | DATE | train+eval |
| QT | OTHER | eval only |

### multiconer_v2 (33 types)

| Source Type | Canonical | File |
|------------|-----------|------|
| Artist | PERSON | train+eval |
| Athlete | PERSON | train+eval |
| Politician | PERSON | train+eval |
| Scientist | PERSON | train+eval |
| Cleric | PERSON | train+eval |
| SportsManager | PERSON | train+eval |
| OtherPER | PERSON | train+eval |
| HumanSettlement | PLACE | train+eval |
| Station | PLACE | train+eval |
| Facility | PLACE | train+eval |
| OtherLOC | PLACE | train+eval |
| ORG | ORG | train+eval |
| PublicCorp | ORG | train+eval |
| PrivateCorp | ORG | train+eval |
| MusicalGRP | ORG | train+eval |
| SportsGRP | ORG | train+eval |
| AerospaceManufacturer | ORG | train+eval |
| CarManufacturer | ORG | train+eval |
| Disease | OTHER | eval only |
| Symptom | OTHER | eval only |
| AnatomicalStructure | OTHER | eval only |
| MedicalProcedure | OTHER | eval only |
| Medication/Vaccine | OTHER | eval only |
| ArtWork | OTHER | eval only |
| MusicalWork | OTHER | eval only |
| VisualWork | OTHER | eval only |
| WrittenWork | OTHER | eval only |
| Clothing | OTHER | eval only |
| Drink | OTHER | eval only |
| Food | OTHER | eval only |
| Software | OTHER | eval only |
| Vehicle | OTHER | eval only |
| OtherPROD | OTHER | eval only |

All 33 multiconer types are mapped.

### stockmark_ner_ja (8 types)

| Source Type | Canonical | File |
|------------|-----------|------|
| 人名 | PERSON | train+eval |
| 地名 | PLACE | train+eval |
| 法人名 | ORG | train+eval |
| その他の組織名 | ORG | train+eval |
| 政治的組織名 | ORG | train+eval |
| 施設名 | PLACE | train+eval |
| イベント名 | OTHER | eval only |
| 製品名 | OTHER | eval only |

All 8 stockmark types are mapped.

### chinese_ner_sft (72 types)

**Train mappings** (high-precision):

| Source Type | Canonical | Notes |
|------------|-----------|-------|
| PER, PER.NAM, PER.NOM | PERSON | OntoNotes-style fine-grained |
| LOC, LOC.NAM, LOC.NOM, Location | PLACE | |
| ORG, ORG.NAM, ORG.NOM, Organization | ORG | |
| GPE, GPE.NAM | PLACE | |
| emotion | EMOTION | nlpcc_2018_task4 subset |
| company | ORG | shared key with finerweb |
| Person | PERSON | shared key with b2nerd |
| organization | ORG | shared key with finerweb |

**Eval-only mappings**:

| Source Type | Canonical | Notes |
|------------|-----------|-------|
| BANK, government | ORG | |
| NAME, name, contact_name, singer | PERSON | NAME omitted from train (ambiguous across subsets) |
| Time | DATE | |
| address, destination, origin | PLACE | |
| game, song | ACTIVITY | Spec: "game, song are too fuzzy for training" |
| All remaining types | OTHER | Domain-specific: medical, e-commerce, resume, etc. |

See `dataset_inventory.json` for the full list of 72 entity types.

### b2nerd (492 types — static mapping covers common types only)

**Train mappings**:

| Source Type | Canonical |
|------------|-----------|
| Person | PERSON |
| Location | PLACE |
| City | PLACE |
| Country | PLACE |
| Organization | ORG |
| Company | ORG |
| Band | ORG |
| Date | DATE |
| Facility | PLACE |

**Eval-only mappings**:

| Source Type | Canonical |
|------------|-----------|
| Activity | ACTIVITY |
| Animal | OTHER |
| Award | OTHER |
| Chemical | OTHER |
| Disease | OTHER |
| Event | OTHER |

The remaining ~477 B2NERD types are handled by runtime heuristic rules (see below).

### finerweb (235K types — static mapping covers common types only)

**Train mappings**:

| Source Type | Canonical |
|------------|-----------|
| person | PERSON |
| location | PLACE |
| city | PLACE |
| country | PLACE |
| organization | ORG |
| company | ORG |
| date | DATE |
| building | PLACE |

**Eval-only mappings**:

| Source Type | Canonical |
|------------|-----------|
| animal | OTHER |
| book | OTHER |
| disease | OTHER |
| event | OTHER |
| food | OTHER |

The remaining ~235K fiNERweb types are handled by runtime heuristic rules (see below).

## Train vs Eval Rules

### When to include in train (View B)

- The source type **unambiguously** maps to exactly one canonical type
- Examples: `PER` → PERSON, `LOC` → PLACE, `Artist` → PERSON
- The mapping would produce correct training signal for the canonical type

### When to include in eval only

- The mapping is correct for scoring purposes but too noisy for training
- Domain-specific types that map to OTHER (e.g., `Disease` → OTHER)
- Fuzzy types like `game`, `song` → ACTIVITY
- Types ambiguous across subsets (e.g., `NAME` in chinese_ner_sft)

### When to omit entirely

- No canonical type exists (long-tail B2NERD/fiNERweb types)
- The mapping would be wrong or misleading
- Apply "when in doubt, omit" principle

## Runtime Heuristic Rules

For datasets with extremely large type sets (B2NERD: 492 types, fiNERweb: 235K types), static mapping files cover only common types. A shared set of keyword-matching heuristics handles the long tail at runtime.

**These rules are a shared spec.** Both Task 4 (`convert_all.py` for View B) and Task 5 (`eval_ner.py` for scoring) should implement them from a shared `lib/type_utils.py` module.

### Keyword matching rules

Applied in order, case-insensitive. First match wins.

| Pattern | Canonical | Examples |
|---------|-----------|----------|
| Contains "person", "人", "人物" | PERSON | "famous person", "人物描述" |
| Contains "location", "地", "place", "城" | PLACE | "geographic location", "地区", "城市" |
| Contains "org", "organization", "公司", "企业" | ORG | "nonprofit organization", "公司名" |
| Contains "date", "time", "日期" | DATE | "publication date", "日期信息" |

### Hierarchy parsing

Types containing `->` (B2NERD hierarchy format, e.g., `"organization -> corporation"`) should be parsed to extract the parent category, then the keyword rules applied to the parent.

### Fallback

If no heuristic matches and the type is not in the static mapping:
- **Train (View B)**: Skip — no View B copy generated
- **Eval (scoring)**: The type is unmapped — prediction is not collapsed

### Coverage note

Datasets with finite, small type sets are fully covered by the static eval mapping: MultiCoNER (33 types), KLUE (6), stockmark (8), open_ner (60), chinese_ner_sft (72). For these, heuristics are never needed.

B2NERD has 492 types, of which only ~15 are statically mapped. The remaining ~477 types **require runtime heuristics** for eval scoring. Task 5 (`eval_ner.py`) must implement the heuristic rules above for correct B2NERD evaluation. fiNERweb (235K types) similarly depends on heuristics for any types beyond the ~13 statically mapped.

## Adding New Mappings

When adding mappings for new datasets or newly discovered types:

1. **Check canonical type fit**: Does the source type unambiguously map to PERSON, PLACE, ORG, DATE, or EMOTION?
2. **Train vs eval**: If unambiguous, add to both files. If fuzzy, add to eval only.
3. **Maintain subset invariant**: Every train entry must also appear in eval with the same value.
4. **Run validation**: Execute the validation script to verify invariants.
5. **Update this doc**: Add the new dataset's mapping table.
