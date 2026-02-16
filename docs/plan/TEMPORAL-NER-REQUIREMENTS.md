# NER Temporal Extraction Requirements for Oneiron

> **Context:** This document specifies what the Oneiron NER model needs to extract for temporal scoring to work. Written for the claude session running the NER training pipeline on the VPS.
>
> **Key files to reference:**
> - `/Users/olety/Desktop/code/oneiron/.worktrees/rrf-pipeline/TEMPORAL-SCORING-SPEC.md` (v3.1) — the scoring system that consumes NER output
> - `/Users/olety/Desktop/code/oneiron/.worktrees/rrf-pipeline/DESIGN-NOTES-task6.md` §3 — NER SFT design notes
> - `/Users/olety/Desktop/code/oneironer/docs/plan/task-3-mapping.md` — current entity type tiers
> - `/Users/olety/Desktop/code/eiri-docs/docs/oneiron/architecture/ONEIRON-ARCH-015-extraction-pipeline-v1.md` — extraction pipeline

---

## TL;DR: What the NER model needs to do for temporal

Standard NER DATE/TIME entities are **not sufficient**. The Oneiron temporal scoring pipeline needs three things from the NER layer:

1. **Temporal span detection** — identify temporal expressions in text (this is standard NER — you already have DATE)
2. **Temporal granularity classification** — classify each temporal span into one of 8 granularity levels (this is NEW, not standard NER)
3. **Anchor mode intent** — detect whether the query is about "when it happened" vs "when I told you" (this should NOT be in the NER model — use heuristics post-extraction)

What the NER model should **NOT** do:
- **Timestamp resolution** — converting "last Tuesday" to a Unix timestamp. This requires knowing the current date and doing calendar arithmetic. It's a deterministic post-processing step, not a neural task. Use a rule-based temporal resolver (duckling, dateparser, sutime, or custom).
- **Anchor mode classification** — detecting Occurred vs Learned intent. This requires full-query understanding, not span-level classification. Use regex heuristics or a tiny separate classifier.

---

## 1. The Oneiron Temporal Scoring Pipeline (What Consumes NER Output)

### What the scoring system receives as input

The Oneiron retrieval pipeline has 4 temporal API tiers. The NER model's job is to produce the inputs for **Tier 3** (the natural-language tier):

```
Tier 1: search_temporal(anchor_start, anchor_end, limit)
        → σ inferred from range width. No NER needed — just timestamps.

Tier 2: search_temporal_with_sigma(anchor_start, anchor_end, σ_secs, anchor_mode, limit)
        → Caller provides explicit σ. For programmatic callers.

Tier 3: search_temporal_with_granularity(anchor_start, anchor_end, granularity, anchor_mode, limit)
        → NER model outputs this. Granularity maps to σ internally.

Tier 4: search_temporal_bitemporal(occ_start, occ_end, lrn_start, lrn_end, σ, limit)
        → Constrained bitemporal. For "what did I know in March about 2016?"
```

### The full extraction→scoring flow

```
User says: "Remember that trip we planned last summer?"
                    ↓
            NER model extracts:
              - "that trip" → EVENT span
              - "last summer" → TEMPORAL_REF span, granularity=Season
                    ↓
            Temporal resolver (rule-based, NOT NER):
              - "last summer" + current_date → [2025-06-01, 2025-08-31]
              - granularity: Season (from NER)
                    ↓
            Anchor mode heuristic (rule-based, NOT NER):
              - "remember that trip we planned" → Occurred (past event)
                    ↓
            Pipeline call:
              search_temporal_with_granularity(
                  anchor_start = 1748736000,  // 2025-06-01
                  anchor_end   = 1756684800,  // 2025-08-31
                  granularity  = Season,       // from NER → σ=90 days
                  anchor_mode  = Occurred,     // from heuristic
                  limit        = 20,
              )
```

### What happens if granularity is wrong

σ controls the decay width. Wrong granularity = wrong retrieval radius:

| Expression | Correct granularity | σ | Search radius | If classified as Day instead |
|---|---|---|---|---|
| "5 years ago" | Year | 180d | 540d | σ=1d, radius=7d → misses everything |
| "last summer" | Season | 90d | 270d | σ=1d, radius=7d → misses everything |
| "last Tuesday" | Day | 1d | 7d | Correct |
| "at 3pm" | Exact | 1h | 7d (floor) | σ=1d, radius=7d → wastefully wide but OK |

**Granularity errors in the "too narrow" direction are catastrophic.** Classifying "5 years ago" as Day means σ=1d and radius=7d — the query only searches within ±3.5 days, completely missing the target. Errors in the "too wide" direction (classifying "last Tuesday" as Month) waste compute but don't miss results.

**Implication for training:** If the model must err, bias toward wider granularity. A "too wide" classification wastes compute but retrieves correctly. A "too narrow" classification breaks retrieval.

---

## 2. What the NER Model Should Extract

### Entity types: Current state + what to add

Your current canonical types (from `type_mapping_train.json`):

| Type | Status | Temporal relevance |
|---|---|---|
| PERSON | ✅ Training | Names → PPR seeds, no temporal output needed |
| PLACE | ✅ Training | Locations → PPR seeds, no temporal output needed |
| ORG | ✅ Training | Organizations → PPR seeds, no temporal output needed |
| DATE | ✅ Training | **Temporal expressions — needs granularity sub-label** |
| EMOTION | ✅ Training | Emotional states → claims, no temporal output needed |

**What needs to change for temporal:**

The DATE type currently captures temporal expressions as flat spans. That's the right entity type — but it needs an **additional sub-classification**: granularity.

### Option A: Granularity as a sub-label on DATE (recommended)

Keep DATE as the entity type. Add granularity as a structured attribute on DATE spans:

```json
{
  "text": "Remember that trip last summer?",
  "query_types": ["PERSON", "DATE", "EMOTION"],
  "entities": [
    {
      "surface": "last summer",
      "type": "DATE",
      "original_type": "DATE",
      "start": 20,
      "end": 31,
      "granularity": "Season"
    }
  ]
}
```

This fits the UniversalNER per-type Q/A format. The model extracts DATE spans as usual, and granularity is either:
- Predicted as an attribute (requires schema extension)
- Classified by a post-hoc step on the extracted DATE span (simpler)

### Option B: Split DATE into granularity-specific types

Replace DATE with 8 granularity types that the model extracts directly:

```json
{
  "query_types": ["PERSON", "DATE_EXACT", "DATE_HOUR", "DATE_DAY", "DATE_WEEK",
                  "DATE_MONTH", "DATE_SEASON", "DATE_YEAR", "DATE_VAGUE"],
  "entities": [
    {"surface": "last summer", "type": "DATE_SEASON", ...}
  ]
}
```

**Pro:** Pure span classification — fits UniversalNER exactly, no schema extension.
**Con:** 8 types instead of 1. Negative sampling gets noisy. Model may confuse DATE_MONTH vs DATE_SEASON.

### Option C: Two-pass extraction (recommended for 0.6B)

1. **Pass 1 (NER model):** Extract DATE spans as usual (you already do this).
2. **Pass 2 (rule-based or tiny classifier):** For each extracted DATE span, classify granularity.

**Pro:** NER model stays simple. Granularity classification is a simpler problem (8-class on short spans) that can be rule-based for most cases.
**Con:** Two passes. But pass 2 is near-instant (regex/rules on extracted spans).

### Recommendation

**For the 0.6B model: Option C (two-pass).** The 0.6B model should focus on accurate span extraction — it's already stretching to do open-vocab NER at 0.6B params. Adding granularity classification as a joint task risks degrading span F1. A simple rule-based granularity classifier on extracted DATE spans will handle 90%+ of cases:

```python
# Rule-based granularity classifier for DATE spans
GRANULARITY_RULES = [
    # Exact: specific time of day
    (r'\b\d{1,2}:\d{2}\b|\bat \d{1,2}(am|pm)\b', 'Exact'),
    # Hour: parts of day
    (r'\b(morning|afternoon|evening|night|dawn|dusk|noon|midnight)\b', 'Hour'),
    # Day: specific days
    (r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|yesterday|today|tomorrow)\b', 'Day'),
    (r'\b\d{1,2}(st|nd|rd|th)?\s+(of\s+)?(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)', 'Day'),
    # Week: week references
    (r'\b(last|this|next|past)\s+week\b|\bweekend\b', 'Week'),
    # Month: month references
    (r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b', 'Month'),
    (r'\b(last|this|next|past)\s+month\b', 'Month'),
    # Season: season references
    (r'\b(spring|summer|autumn|fall|winter)\b', 'Season'),
    (r'\b(last|this|next)\s+(spring|summer|autumn|fall|winter)\b', 'Season'),
    # Year: year references
    (r'\b\d{4}\b|\b\d+\s+years?\s+ago\b', 'Year'),
    (r'\b(last|this|next|past)\s+year\b', 'Year'),
    # Vague: everything else temporal
    (r'\b(a while|long time|ages|recently|lately|before|earlier|once|back then)\b', 'Vague'),
]
# Default: Day (safest — wider than Exact/Hour, narrower than Month)
```

**For the 1.7B+ models: Option A or B.** Larger models can handle joint extraction + classification. Option B (granularity-specific types) fits UniversalNER cleanest but adds 7 types. Option A (attribute) is cleaner but needs schema extension.

**For the 8B server model:** Option A. The server model has capacity and does full claim extraction anyway — granularity is a trivial additional output.

---

## 3. TemporalGranularity: The 8 Levels

The Oneiron crate defines these (from `TEMPORAL-SCORING-SPEC.md` v3.1 §5):

| Granularity | Example expressions | σ (seconds) | σ (human) | Search radius (3σ) |
|---|---|---|---|---|
| **Exact** | "at 3:15pm", "at noon sharp" | 3,600 | 1 hour | 3h |
| **Hour** | "that afternoon", "this morning", "around 8pm" | 14,400 | 4 hours | 12h |
| **Day** | "last Tuesday", "on January 5th", "yesterday" | 86,400 | 1 day | 3d |
| **Week** | "last week", "the week before", "this weekend" | 604,800 | 7 days | 21d |
| **Month** | "in March", "last month", "early February" | 2,592,000 | 30 days | 90d |
| **Season** | "last summer", "this spring", "over the holidays" | 7,776,000 | 90 days | 270d |
| **Year** | "in 2023", "5 years ago", "back in college" | 15,552,000 | 180 days | 540d |
| **Vague** | "a while back", "a long time ago", "once", "before" | 31,536,000 | 365 days | ~3y |

### CJK equivalents (critical for your multilingual model)

| Granularity | Japanese | Chinese | Korean |
|---|---|---|---|
| Exact | 3時15分に, 正午ちょうど | 下午3点15分, 中午12点 | 오후 3시 15분에, 정오에 |
| Hour | その午後, 今朝, 夕方 | 那天下午, 今天早上, 傍晚 | 그날 오후, 오늘 아침, 저녁에 |
| Day | 先週の火曜日, 昨日 | 上周二, 昨天, 1月5日 | 지난 화요일, 어제 |
| Week | 先週, 今週末 | 上周, 这个周末 | 지난 주, 이번 주말 |
| Month | 3月に, 先月 | 三月份, 上个月 | 3월에, 지난 달 |
| Season | 去年の夏, この春 | 去年夏天, 今年春天 | 지난 여름, 이번 봄 |
| Year | 2023年に, 5年前 | 2023年, 五年前 | 2023년에, 5년 전 |
| Vague | しばらく前, ずっと前 | 很久以前, 前一阵子 | 한참 전에, 오래전에 |

### Training data for granularity classification

If you go with Option B (granularity-specific DATE types) or want to train a classifier:

**Source 1: Rule-based labeling of existing DATE spans.** Take all DATE spans from your converted datasets, apply the regex rules above to assign granularity, use these as training labels. This gives you thousands of labeled examples for free.

**Source 2: TempEval / TimeBank.** These datasets have temporal expressions with TIMEX3 annotations that include `type` (DATE, TIME, DURATION, SET) and `value` (ISO format). You can derive granularity:
- TIMEX3 `value="2023"` → Year
- TIMEX3 `value="2023-03"` → Month
- TIMEX3 `value="2023-03-15"` → Day
- TIMEX3 `value="2023-03-15T15:30"` → Exact
- TIMEX3 `value="P1W"` (duration) → Week
- TIMEX3 `type="SET"` (recurring) → depends on period

**Source 3: Synthetic generation.** For each granularity level, generate examples in all target languages:
```
Generate 200 natural conversation snippets per granularity level × 4 languages (EN/JA/ZH/KO).
Each snippet should contain one temporal expression at the target granularity.
Format: {"text": "...", "temporal_span": [start, end], "granularity": "..."}
```

---

## 4. What About TemporalAnchorMode?

The Oneiron scoring system has 4 anchor modes:

| Mode | Query intent | Example | Regex signal |
|---|---|---|---|
| **Occurred** | When events happened | "What happened last March?" | "happened", "occurred", "was", "went" |
| **Learned** | When info was told/recorded | "What did I tell you last March?" | "told you", "mentioned", "said", "discussed" |
| **Both** | Bitemporal constraint | "What did I know in March about 2016?" | Two temporal refs + knowledge verb |
| **Auto** | Ambiguous (default) | "Remember last summer?" | No clear signal |

### The NER model should NOT classify anchor mode

This is a query-level intent classification, not a span-level entity classification. Reasons:
1. It requires understanding the full query, not just temporal spans
2. It's a 4-class problem on the whole query, not per-entity
3. Rule-based heuristics handle 80%+ of cases
4. Getting it wrong is OK — Auto mode (default) is recall-maximizing and handles all cases acceptably
5. The 0.6B model has no capacity to spare for this

### Post-NER heuristic for anchor mode

```python
import re

def classify_anchor_mode(query: str) -> str:
    """Classify temporal anchor mode from query text. Returns 'Occurred'|'Learned'|'Both'|'Auto'."""
    q = query.lower()

    # Both: two temporal references + knowledge/memory verb
    temporal_count = len(re.findall(
        r'\b(in \d{4}|last \w+|this \w+|\d+ \w+ ago|january|february|march|april|may|june|'
        r'july|august|september|october|november|december)\b', q
    ))
    if temporal_count >= 2 and re.search(r'\b(knew|know|learned|found out)\b', q):
        return 'Both'

    # Learned: recording/telling verbs
    if re.search(r'\b(told you|mentioned|said|discussed|talked about|wrote|noted|'
                 r'brought up|shared|informed)\b', q):
        return 'Learned'

    # Occurred: event/happening verbs
    if re.search(r'\b(happened|occurred|took place|went|did|was|were|had|made|'
                 r'started|finished|began|ended|visited|met|saw)\b', q):
        return 'Occurred'

    return 'Auto'
```

This heuristic is imperfect but the cost of defaulting to Auto is low (recall-maximizing, just slightly noisier).

---

## 5. Entity Types That Need Temporal Metadata at Indexing Time

When the NER extracts entities for *indexing* (write path, not query path), these entity types need temporal timestamps assigned:

| Entity type | occurred_start | occurred_end | learned_at | Who assigns |
|---|---|---|---|---|
| MESSAGE | message timestamp | same (point) | ingestion time | System (automatic) |
| TURN | turn start time | turn end time | ingestion time | System (automatic) |
| SESSION | session start | session end | ingestion time | System (automatic) |
| EVENT | **event start** | **event end** | ingestion time | **NER + temporal resolver** |
| CLAIM | validFrom | validTo (or ∞) | ingestion time | Dreamer (server 8B) |
| SUMMARY | summarized period start | summarized period end | generation time | Dreamer |
| PERSON | first mention time | ∞ (ongoing) | first mention time | System |
| PLACE | first mention time | ∞ (ongoing) | first mention time | System |
| SKILL | skill detected time | ∞ (ongoing) | detection time | Dreamer |

**Key insight:** Only EVENT and CLAIM have user-specified temporal boundaries that come from NER extraction. Everything else has system-assigned timestamps. The NER model needs to extract temporal expressions so that the temporal resolver can compute `occurred_start`/`occurred_end` for EVENTs and `validFrom`/`validTo` for CLAIMs.

### What the NER model extracts vs what the resolver does

```
NER model extracts:
  "I'm meeting Sarah for coffee next Tuesday at 3pm"
    → PERSON: "Sarah"
    → EVENT: "meeting ... for coffee"
    → DATE: "next Tuesday at 3pm", granularity=Exact
    → ACTIVITY: "coffee" (Tier 2 only)

Temporal resolver (rule-based) converts:
  "next Tuesday at 3pm" + current_date(2026-02-16)
    → occurred_start = 2026-02-17T15:00:00Z (1739804400)
    → occurred_end   = 2026-02-17T15:00:00Z (same — point event)

Dreamer (8B server, async) creates:
  EVENT entity:
    occurred = [1739804400, 1739804400]
    learned_at = now()
    blob = {subject: "Sarah", type: "coffee meeting", ...}

  CLAIM entity (if pattern detected):
    predicate: "relationship.activity"
    value: "has coffee meetings with Sarah"
    validFrom = 1739804400
    validTo = null (ongoing)
```

---

## 6. Additional Entity Types for Temporal Awareness

Beyond the current 5 training types (PERSON, PLACE, ORG, DATE, EMOTION), the following types have temporal relevance and should be added to the type vocabulary:

### Tier 1 (1.7B on-device) — add these

| Type | Why | Temporal role | Training data |
|---|---|---|---|
| **TEMPORAL_REF** | Relative temporal expressions that aren't dates | Query anchor | "a while back", "recently", "before that" — these map to Vague granularity |
| **GOAL** | User goals with deadlines | validTo on CLAIMs | "learn Japanese by June" → CLAIM with validTo |

**TEMPORAL_REF vs DATE:** DATE captures explicit dates/times ("March 2024", "3:15pm", "last Tuesday"). TEMPORAL_REF captures relative/vague references ("before", "recently", "a while back", "around that time", "shortly after", "the other day"). Both feed into the temporal resolver but need different handling. TEMPORAL_REF always maps to Vague or Season granularity. If this distinction is too fine-grained for 0.6B/1.7B, merge them under DATE and let the granularity classifier sort it out.

### Tier 2 (8B server) — add these

| Type | Why | Temporal role | Training data |
|---|---|---|---|
| **RELATIONSHIP_REF** | Interpersonal relationships | Long-duration entities (spanner index) | "my sister", "our friendship" |
| **SKILL_REF** | Skills/capabilities | Persistent entities | "my Japanese", "coding ability" |
| **LIFE_EVENT** | Major life events | Strong temporal anchors | "when I graduated", "after the divorce" |
| **RITUAL** | Recurring activities | Periodic temporal patterns | "our weekly meeting", "morning yoga" |
| **ACTIVITY** | Activities (already in eval) | Event timestamps | "hiking", "cooking", "studying" |

### Do NOT add these as NER types

| Type | Why not |
|---|---|
| MESSAGE, TURN, SESSION | System-created, not extracted from text |
| SUMMARY | Dreamer-generated, not extracted from text |
| NOTIFICATION | System-created |
| ASSET, ASSET_TEXT | Attached media, not extracted from text |

---

## 7. Training Data Strategy for Temporal

### Phase 1: Use existing DATE annotations + rule-based granularity (now)

You already have DATE as a training type. Your converted datasets have DATE spans. Apply the regex-based granularity rules to auto-label them. This gives you:
- Correct DATE span extraction (from existing training)
- Rule-based granularity (no model changes needed)
- Works with the 0.6B model as-is

### Phase 2: Add TEMPORAL_REF type + synthetic data (next)

1. Generate synthetic conversation examples with temporal references in all 4 languages
2. Label with TEMPORAL_REF type for vague/relative references
3. Include in View A training data
4. Add TEMPORAL_REF → DATE in View B mapping (for canonical type training)

### Phase 3: Joint granularity prediction (future, 1.7B+)

For the 1.7B model, consider Option B (granularity-specific types): DATE_EXACT through DATE_VAGUE. This requires:
1. Re-label all DATE spans with granularity (rule-based, then human audit)
2. Train with 8 DATE subtypes in the query_types
3. Eval: measure both span F1 and granularity accuracy

### Phase 4: Conversational temporal extraction (future, synthetic)

Generate training data specifically for conversational temporal expressions:
```
A: When did you last see your sister?
B: Oh, it was sometime last spring. We had lunch near her office.
A: That's been a while! Was it before or after your birthday?
B: Right after, actually. Maybe late April?

Entities:
  - "last spring" → DATE, granularity=Season
  - "her office" → PLACE
  - "your birthday" → EVENT
  - "late April" → DATE, granularity=Month (overrides "last spring" — more specific)
```

This is where the conversationalized B2NERD data (Task 6) is valuable — it provides realistic dialogue with entity annotations. Add temporal granularity labels to the DATE spans in this data.

---

## 8. Practical Summary: What to Do Now

### For the 0.6B model (Tier 0)

1. **Keep DATE as-is.** Don't change the entity type.
2. **Add a rule-based granularity classifier** as a post-processing step on extracted DATE spans. This is a few hundred lines of Python regex, not a model change.
3. **Add a rule-based anchor mode classifier** as a post-processing step on the full query. ~50 lines of Python regex.
4. **Add a temporal resolver** (duckling, dateparser, or custom) to convert DATE spans to Unix timestamps. This is the biggest piece — but it's deterministic, not learned.
5. **Don't try to make the 0.6B model learn granularity.** It's already stretching to do open-vocab NER. Adding a sub-classification task will degrade span F1.

### For the 1.7B model (Tier 1)

1. **Consider adding TEMPORAL_REF** as a separate type for vague references
2. **Consider Option B** (granularity-specific types) if eval shows the rule-based classifier has poor accuracy on CJK temporal expressions
3. **Add GOAL** as a type — goals have deadlines (temporal boundaries for CLAIMs)

### For the 8B model (Tier 2, server)

1. **Add all Tier 2 types** (RELATIONSHIP_REF, SKILL_REF, LIFE_EVENT, RITUAL, ACTIVITY)
2. **Joint granularity prediction** via Option A (attribute on DATE spans)
3. **Full claim extraction** — the 8B model is part of the Dreamer pipeline, which does structured claim extraction including temporal validity (validFrom/validTo)

### The non-NER temporal pipeline (build alongside)

These components sit between NER output and the Oneiron scoring API:

```
NER output:  [{surface: "last summer", type: "DATE", span: [20, 31]}]
     ↓
Granularity classifier (rule-based):  granularity = "Season"
     ↓
Temporal resolver (rule-based):
     current_date = 2026-02-16
     "last summer" → [2025-06-01, 2025-08-31]
     ↓
Anchor mode classifier (rule-based):
     query = "Remember that trip last summer?"
     → mode = "Occurred" (past event verb "remember")
     ↓
Oneiron API call:
     search_temporal_with_granularity(
         anchor_start = 1748736000,
         anchor_end = 1756684800,
         granularity = Season,
         anchor_mode = Occurred,
         limit = 20,
     )
```

The temporal resolver is the most complex non-NER piece. For CJK languages, consider:
- **English:** dateparser, duckling, or sutime
- **Japanese:** dateparser (supports JA), or custom rules for JA-specific patterns (先週, 去年の夏, etc.)
- **Chinese:** dateparser (supports ZH), or custom rules for ZH patterns (上个月, 去年夏天, etc.)
- **Korean:** dateparser (limited KO support), custom rules likely needed (지난 주, 작년 여름, etc.)

---

## 9. Evaluation Criteria for Temporal NER

Add these to your eval harness (`eval_ner.py`):

### Span-level metrics (existing)
- DATE span F1 (strict + relaxed) — already planned

### Granularity metrics (new, if Option B or classifier eval)
- Granularity accuracy on gold DATE spans
- Granularity confusion matrix (is Month confused with Season? Day with Week?)
- Per-language granularity accuracy (CJK temporal expressions may have different accuracy)

### End-to-end temporal retrieval metrics (future, requires Oneiron integration)
- Given a temporal query + vault of entities, does the correct entity appear in top-K?
- This is a retrieval eval, not an NER eval — but it tests the full pipeline (NER → resolver → scoring)

### Critical failure modes to test
1. **"5 years ago" classified as Day** → σ=1d, radius=7d, completely misses target. CATASTROPHIC.
2. **"yesterday" classified as Year** → σ=180d, radius=540d, retrieves too much but doesn't miss. ACCEPTABLE.
3. **"last Tuesday" not extracted as DATE at all** → no temporal signal, falls back to vector/BM25. DEGRADED but not broken.
4. **CJK temporal expression missed** → "先週の火曜日" not extracted. Same as #3. Must test CJK recall.

---

## Appendix: TemporalGranularity Enum (Oneiron crate)

For reference, the exact Rust enum the crate implements:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TemporalGranularity {
    Exact,   // σ = 3,600s     (1h)
    Hour,    // σ = 14,400s    (4h)
    Day,     // σ = 86,400s    (1d)
    Week,    // σ = 604,800s   (1w)
    Month,   // σ = 2,592,000s (30d)
    Season,  // σ = 7,776,000s (90d)
    Year,    // σ = 15,552,000s (180d)
    Vague,   // σ = 31,536,000s (365d)
}
```

## Appendix: TemporalAnchorMode Enum (Oneiron crate)

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TemporalAnchorMode {
    Occurred,  // "what happened last March?"
    Learned,   // "what did I tell you last March?"
    Both,      // "what did I know in March 2025 about events from 2016?"
    #[default]
    Auto,      // ambiguous — noisy-OR, never penalizes divergence
}
```
