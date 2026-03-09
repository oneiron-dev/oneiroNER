#!/usr/bin/env python3
"""Shared constants and helpers for Task 9.5 silver data quality scripts."""

import json
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROMPT_VERSION = "task9_5_v1"

CANONICAL_TYPES = [
    "PERSON", "PLACE", "ORG",
    "EVENT", "EVENT/Life", "EVENT/General",
    "EMOTION", "GOAL", "ACTIVITY",
    "DATE", "DATE/Day", "DATE/Week", "DATE/Month", "DATE/Season",
    "DATE/Year", "DATE/Decade", "DATE/Relative", "DATE/Range",
    "RELATIONSHIP_REF", "RELATIONSHIP_REF/Family", "RELATIONSHIP_REF/Romantic",
    "RELATIONSHIP_REF/Friend", "RELATIONSHIP_REF/Professional",
    "RELATIONSHIP_REF/Acquaintance",
]

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SILVER_GLOB = "data/processed/task9_silver_*.jsonl"

_INHERENTLY_SPECIFIC = frozenset({
    "mom", "dad", "mother", "father", "mama", "papa", "mum", "mummy",
    "mommy", "daddy", "grandma", "grandpa", "grandmother", "grandfather",
    "granny", "nana", "husband", "wife", "boyfriend", "girlfriend",
    "fiancé", "fiancée", "ex", "ex-boyfriend", "ex-girlfriend",
})

_ALWAYS_LLM = frozenset({
    "friend", "friends", "doctor", "teacher", "boss", "neighbor",
    "neighbour", "partner", "colleague", "coworker", "acquaintance",
})


def entity_key(ent, source, source_id):
    return (source, source_id, ent.get("turn_index"), ent["start"], ent["end"], ent["surface"])


def make_provenance(source, method, model=None, confidence=1.0):
    return {
        "subtype_source": source,
        "subtype_method": method,
        "subtype_confidence": confidence,
        "subtype_model": model,
        "subtype_prompt_version": PROMPT_VERSION,
        "subtype_timestamp": datetime.now(timezone.utc).isoformat(),
    }


def load_silver_records(glob_pattern=None):
    if glob_pattern is None:
        glob_pattern = BASE_DIR / SILVER_GLOB
    else:
        glob_pattern = Path(glob_pattern)

    parent = glob_pattern.parent
    pattern = glob_pattern.name

    for filepath in sorted(parent.glob(pattern)):
        with open(filepath) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                yield filepath, line_num, record


def get_entity_text(record, entity):
    if record.get("format") == "conversation":
        return record["turns"][entity["turn_index"]]["text"]
    return record["text"]


def validate_offset(record, entity):
    text = get_entity_text(record, entity)
    return text[entity["start"]:entity["end"]] == entity["surface"]


def sample_review(changed, n=100, seed=42):
    rng = random.Random(seed)

    def _get_stratum(item):
        keys = []
        keys.append(f"method:{item.get('subtype_method', 'unknown')}")
        conf = item.get("subtype_confidence", 0.5)
        if conf >= 0.9:
            keys.append("conf:high")
        elif conf < 0.7:
            keys.append("conf:low")
        else:
            keys.append("conf:mid")
        keys.append(f"generic:{'yes' if item.get('generic_flag', False) else 'no'}")
        keys.append(f"source:{item.get('source', 'unknown')}")
        return tuple(keys)

    surface_counts = Counter(item.get("surface", "").lower() for item in changed)
    top_surfaces = {s for s, _ in surface_counts.most_common(20)}

    strata = {}
    for i, item in enumerate(changed):
        key = _get_stratum(item)
        strata.setdefault(key, []).append(i)
        if item.get("surface", "").lower() in top_surfaces:
            freq_key = ("freq:top20",)
            strata.setdefault(freq_key, []).append(i)

    target_per_stratum = max(1, 20)
    sampled_indices = set()

    for key in sorted(strata.keys(), key=lambda k: len(strata[k])):
        pool = [i for i in strata[key] if i not in sampled_indices]
        take = min(target_per_stratum, len(pool), n - len(sampled_indices))
        if take <= 0:
            break
        sampled_indices.update(rng.sample(pool, take))

    if len(sampled_indices) < n:
        remaining = [i for i in range(len(changed)) if i not in sampled_indices]
        take = min(n - len(sampled_indices), len(remaining))
        sampled_indices.update(rng.sample(remaining, take))

    return [changed[i] for i in sorted(sampled_indices)]


def write_output_jsonl(path, records, dry_run=False):
    if dry_run:
        print(f"[dry-run] Would write {len(records)} records to {path}")
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def print_summary(stats):
    if not stats:
        return
    max_key = max(len(str(k)) for k in stats)
    for key, value in stats.items():
        print(f"  {str(key).ljust(max_key)}  {value}")


def specificity_gate(surface, context):
    low_surface = surface.lower()

    if low_surface in _INHERENTLY_SPECIFIC:
        return True

    if low_surface in _ALWAYS_LLM:
        return False

    escaped = re.escape(surface)
    if re.search(rf"\b(?:my|our|his|her|their|your)\s+{escaped}", context, re.IGNORECASE):
        return True
    if re.search(rf"\w+'s\s+{escaped}", context, re.IGNORECASE):
        return True
    if re.search(rf"\b(?:that|this|the)\s+{escaped}", context, re.IGNORECASE):
        return True

    return False
