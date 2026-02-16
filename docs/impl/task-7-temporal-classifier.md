# Task 7: Temporal Granularity Classifier + Data Sweep

## Context

View B records currently map all DATE entities to flat `DATE`. We need to refine
them to `DATE -> Day`, `DATE -> Season`, etc. based on entity surface text, so the
model learns temporal granularity natively. This is programmatic (regex), no LLM.

## Architecture

### Current flow
```
source type "DT" → type_mapping_train.json → canonical "DATE"
```

### New flow
```
source type "DT" → type_mapping_train.json → canonical "DATE"
                  → temporal_classifier(surface) → "DATE -> Day"
```

No changes to `type_mapping_train.json`. Subtyping is dynamic at View B generation.

## Deliverables

### 1. `scripts/lib/temporal_classifier.py`

```python
def classify_granularity(surface: str) -> str:
    """Return one of: Exact, Hour, Day, Week, Month, Season, Year, Vague."""
```

Rules (first match wins, ordered specific→general):
- **Exact**: `\d{1,2}:\d{2}`, `at \d+(am|pm)`, CJK: 点/時 + minutes
- **Hour**: morning/afternoon/evening/night/dawn/dusk/noon/midnight, CJK: 朝/午後/早上/下午/저녁/아침
- **Day**: weekday names, yesterday/today/tomorrow, `\d{1,2}(st|nd|rd|th)`, specific dates like "January 5th", CJK: 昨日/今日/明日/어제/오늘
- **Week**: last/this/next week, weekend, CJK: 先週/上周/지난주
- **Month**: month names, last/this/next month, CJK: 先月/上个月/지난달
- **Season**: spring/summer/autumn/fall/winter, CJK: 春/夏/秋/冬
- **Year**: `\d{4}`, `\d+ years? ago`, last/this/next year, CJK: 年前/작년
- **Vague**: a while/long time/ages/recently/once/back then, CJK: 久しぶり/很久以前/오래전
- **Default**: Day (bias wider per TEMPORAL-NER-REQUIREMENTS.md)

### 2. Update `generate_view_b()` in `scripts/convert_all.py`

When canonical == "DATE":
1. Classify granularity from entity surface
2. Set entity type to `f"DATE -> {granularity}"`

Query types logic:
- Entity type is always specific: `DATE -> Day`
- With 30% probability, also add parent `DATE` to query_types
- When `DATE` is in query_types, ALL date entities match (chat format shows all)
- This teaches the model both hierarchical and specific extraction

Negative sampling:
- Pool: {PERSON, ORG, PLACE, EMOTION} ∪ {all DATE subtypes not in positives}
- Never use parent `DATE` as negative (it's a superset of all subtypes)
- Sample 2-5 negatives as before

### 3. Update `build_canonical_sampler()`

Extend canonical types to include all 8 DATE subtypes.
Remove plain `DATE` from the negative pool (it's only used as a positive when added).

### 4. `scripts/verify_temporal.py` — LLM verification

Sample ~500 DATE entities from View B output. Send to codex-spark:
```
Given this text snippet and the highlighted temporal expression,
classify the temporal granularity as one of:
Exact, Hour, Day, Week, Month, Season, Year, Vague

Text: "..."
Temporal expression: "..."
Granularity:
```

Compare programmatic vs LLM classification. Report agreement rate.

### 5. Data sweep

Re-run `python scripts/convert_all.py --skip-converters` to regenerate View B
with temporal subtypes. This only regenerates from dedup→split→View B→chat.

## Files touched

| File | Action |
|------|--------|
| `scripts/lib/temporal_classifier.py` | CREATE |
| `scripts/convert_all.py` | MODIFY (generate_view_b, build_canonical_sampler) |
| `scripts/verify_temporal.py` | CREATE |
| `tests/test_temporal_classifier.py` | CREATE |

## Verification

1. Unit tests for temporal_classifier (EN + CJK examples from subtypes doc)
2. Run convert_all.py --skip-converters → check View B has DATE -> {X} types
3. Spot-check: sample 20 records from output, verify granularity makes sense
4. codex-spark verification: ≥85% agreement target
