---
task: 3
title: Ontology Mapping (Eval-Only)
phase: 1
depends_on: []
agent_model: opus
outputs:
  - configs/type_mapping.json
  - configs/type_mapping.md
---

# Task 3: Ontology Mapping (Eval-Only — NOT Training-Time Collapse)

> Back to [README](README.md) | Prev: [Task 2](task-2-schema.md) | Next: [Pre-Task 4](task-4-pretask.md)

**Critical: Do NOT collapse types during conversion.** The mapping file is used at evaluation time only, to score model output against Oneiron's canonical type vocabulary.

Type diversity IS the training signal for UniversalNER generalization. B2NERD's 400+ types, fiNERweb's 235K+ types, OpenNER's 60 types all stay as-is in training data. Canonical remapping happens only in [View B](task-2-schema.md#dual-view-training-view-a--view-b) copies.

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
