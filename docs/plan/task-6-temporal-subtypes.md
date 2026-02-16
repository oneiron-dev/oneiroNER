# Temporal Subtype Convention

> **Decision date:** 2026-02-16
> **Context:** Extends the NER schema to support temporal granularity classification natively in the model, rather than as a post-processing step.

## Decision

Train temporal granularity directly into the model using the existing UniversalNER open-vocabulary type system. No schema changes needed — `type` is already a free-form string.

### Why this works at 0.6B

The temporal requirements doc (§2, Option C) recommends a two-pass approach for 0.6B: extract DATE spans, then classify granularity with rules. However, Option B (granularity-specific types) fits UniversalNER natively:

- The model already handles 250K+ unique types from fiNERweb
- Adding 8 DATE subtypes is 7 net types — negligible capacity cost
- Types are model **input**, not an extra classification head
- The model already distinguishes similar types (e.g., "organization -> corporation" vs "organization -> government")
- Surface patterns for granularity are often more distinct than existing type distinctions

## Naming Convention

Use ` -> ` (space-arrow-space) separator, matching existing B2NERD hierarchical labels.

### Parent type

`DATE` — extracts all temporal expressions regardless of granularity.

### Subtypes (8 levels, matching Oneiron `TemporalGranularity` enum)

| Type | Granularity | σ | Examples (EN) | Examples (ZH) | Examples (JA) |
|------|-------------|---|---------------|---------------|---------------|
| `DATE -> Exact` | Exact | 1h | "at 3:15pm", "noon sharp" | 下午3点15分, 中午12点 | 3時15分に, 正午ちょうど |
| `DATE -> Hour` | Hour | 4h | "that afternoon", "this morning" | 那天下午, 今天早上 | その午後, 今朝 |
| `DATE -> Day` | Day | 1d | "last Tuesday", "yesterday" | 上周二, 昨天 | 先週の火曜日, 昨日 |
| `DATE -> Week` | Week | 7d | "last week", "this weekend" | 上周, 这个周末 | 先週, 今週末 |
| `DATE -> Month` | Month | 30d | "in March", "last month" | 三月份, 上个月 | 3月に, 先月 |
| `DATE -> Season` | Season | 90d | "last summer", "this spring" | 去年夏天, 今年春天 | 去年の夏, この春 |
| `DATE -> Year` | Year | 180d | "in 2023", "5 years ago" | 2023年, 五年前 | 2023年に, 5年前 |
| `DATE -> Vague` | Vague | 365d | "a while back", "long ago" | 很久以前, 前一阵子 | しばらく前, ずっと前 |

### Programmatic conventions

```python
# Parent extraction
"DATE -> Day".split(" -> ")[0]  # → "DATE"

# Subtype extraction
"DATE -> Day".split(" -> ")[1]  # → "Day"

# Check if temporal subtype
type_str.startswith("DATE -> ")

# All subtypes
TEMPORAL_SUBTYPES = [f"DATE -> {g}" for g in
    ["Exact", "Hour", "Day", "Week", "Month", "Season", "Year", "Vague"]]
```

## Training Strategy

### View A (raw labels)
- Keep original source `DATE` label — preserves source fidelity
- `original_type` always reflects the source dataset's label

### View B (canonical labels)
- Rule-based classifier assigns granularity from the entity surface text
- `type` = `DATE -> Day`, `DATE -> Season`, etc.
- `original_type` = source label (e.g., `DATE`, `DT`, `日期`)
- Records where granularity is ambiguous → keep plain `DATE`

### Query types
- Some records use parent `DATE` in `query_types` → model learns to extract all temporal expressions
- Some records use specific subtypes → model learns granularity-specific extraction
- At inference: query `DATE` for all temporal, or specific subtypes when granularity matters

### Granularity assignment

Rule-based classifier applied at View B generation time (not LLM-dependent):

```python
GRANULARITY_RULES = [
    (r'\b\d{1,2}:\d{2}\b|\bat \d{1,2}(am|pm)\b', 'Exact'),
    (r'\b(morning|afternoon|evening|night|dawn|dusk|noon|midnight)\b', 'Hour'),
    (r'\b(monday|tuesday|...|yesterday|today|tomorrow)\b', 'Day'),
    (r'\b(last|this|next)\s+week\b|\bweekend\b', 'Week'),
    (r'\b(january|february|...)\b|\b(last|this|next)\s+month\b', 'Month'),
    (r'\b(spring|summer|autumn|fall|winter)\b', 'Season'),
    (r'\b\d{4}\b|\b\d+\s+years?\s+ago\b', 'Year'),
    (r'\b(a while|long time|ages|recently|lately|once|back then)\b', 'Vague'),
]
# Default: Day (wider than Exact/Hour, narrower than Month — safe default)
```

CJK equivalents defined in TEMPORAL-NER-REQUIREMENTS.md §3.

## Error bias

Per TEMPORAL-NER-REQUIREMENTS.md §1: if the model must err, bias toward **wider** granularity. "Too wide" wastes compute but retrieves correctly. "Too narrow" breaks retrieval catastrophically.

## Future extensibility

The ` -> ` convention extends to other type hierarchies if needed:
- `PERSON -> Politician`, `PERSON -> Athlete`
- `ORG -> Company`, `ORG -> Government`
- `PLACE -> City`, `PLACE -> Country`

## References

- `docs/plan/TEMPORAL-NER-REQUIREMENTS.md` — full temporal requirements
- `configs/schema.json` — record schema (no changes needed)
- `configs/type_mapping_train.json` — will need granularity-aware mapping logic
