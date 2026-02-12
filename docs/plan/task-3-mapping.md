---
task: 3
title: Ontology Mapping (Eval + View B)
phase: 1
depends_on: []
agent_model: opus
outputs:
  - configs/type_mapping_train.json
  - configs/type_mapping_eval.json
  - configs/type_mapping.json  # if Option B (single file with confidence)
  - configs/type_mapping.md
---

# Task 3: Ontology Mapping (Eval + View B)

> Back to [README](README.md) | Prev: [Task 2](task-2-schema.md) | Next: [Pre-Task 4](task-4-pretask.md)

**Critical: Do NOT collapse types during View A conversion.** Type diversity IS the training signal for UniversalNER generalization. B2NERD's 400+ types, fiNERweb's 235K+ types, OpenNER's 60 types all stay as-is in View A training data. Canonical remapping happens only in [View B](task-2-schema.md#dual-view-training-view-a--view-b) copies and at eval time.

Mapping serves two purposes:
- **Eval scoring**: Broader/heuristic mappings acceptable for grouping metrics.
- **View B training**: High-precision mappings only — wrong canonical labels poison training data.

When in doubt, omit a View B mapping. A missing View B copy is better than a wrong one.

## Purpose

1. **Eval scoring**: Map predicted types → Oneiron canonical types for per-type F1.
2. **Cross-dataset metrics**: Aggregate PER/PERSON/person/人名/PS all as PERSON.
3. **Tier assignment**: Which Oneiron types go to which model tier.

## Oneiron Canonical Types by Tier

| Tier | Types |
|------|-------|
| Tier 0 (0.6B) | PERSON, PLACE, DATE, ORG (fixed list only if zero-shot F1 < 50%) |
| Tier 1 (1.7B) | PERSON, PLACE, DATE, ORG, EMOTION, GOAL, TEMPORAL_REF |
| Tier 1.5 (4B) | Full type vocabulary |
| Tier 2 (8B) | All + RELATIONSHIP_REF, ACTIVITY, SKILL_REF, RITUAL, LIFE_EVENT + plugin types |

## Mapping Rules (eval only)

| Oneiron Type | Source Types |
|-------------|-------------|
| PERSON | PER, person, PS, OtherPER, Artist, Athlete, Politician, Scientist, Cleric, SportsManager |
| PLACE | LOC, location, GPE, FAC, LC, Facility, HumanSettlement, Station, OtherLOC |
| ORG | ORG, organization, OG, PublicCorp, PrivateCorp, MusicalGRP, SportsGRP |
| DATE | DATE, TIME, DATETIME, DT, TI |
| EMOTION | emotion (from nlpcc_2018_task4) |
| ACTIVITY | game, song, Activity-like types |
| OTHER | Domain-specific types that don't map (disease, chemical, etc.) |

### B2NERD Special Handling

B2NERD has 15K+ types (unverified — see [Pre-Task 4](task-4-pretask.md)):
- Keyword matching: types containing "person"/"人" → PERSON, "location"/"地" → PLACE, etc.
- Types with `->` hierarchy (e.g., "organization -> corporation") → parse parent category.

## Mapping Artifacts

Two options (choose one during implementation):

**Option A — Two files:**
- `configs/type_mapping_train.json` — Conservative, high-precision only. Used by `convert_all.py` for View B generation.
- `configs/type_mapping_eval.json` — Can be broader/heuristic. Used by `eval_ner.py` for scoring.

**Option B — Single file with confidence:**
- `configs/type_mapping.json` — Each mapping entry includes `"confidence": "high"|"medium"|"low"`.
- View B generation only uses `confidence >= "high"` entries.
- Eval scoring uses all entries.

Either approach prevents low-confidence mappings from corrupting View B training data.
