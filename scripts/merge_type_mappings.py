#!/usr/bin/env python3
"""Merge Gemini type mapping into train type mapping.

Adds ~499 Gemini-classified entries to the existing 69-entry train mapping.
Fixes: TEMPORAL_REF → DATE (removed from ontology).
"""

import json
from pathlib import Path

CONFIGS = Path(__file__).parent.parent / "configs"
TRAIN_PATH = CONFIGS / "type_mapping_train.json"
GEMINI_PATH = CONFIGS / "type_mapping_gemini.json"


def main():
    with open(TRAIN_PATH) as f:
        train = json.load(f)
    with open(GEMINI_PATH) as f:
        gemini = json.load(f)

    print(f"Train entries: {len(train)}")
    print(f"Gemini entries: {len(gemini)}")

    overlaps = set(train) & set(gemini)
    print(f"Overlapping keys: {len(overlaps)}")
    for k in sorted(overlaps):
        t, g = train[k], gemini[k]
        conflict = " CONFLICT" if t != g else ""
        print(f"  {k}: train={t}, gemini={g}{conflict}")

    merged = dict(train)
    new_keys = 0
    for k, v in gemini.items():
        if k not in merged:
            merged[k] = v
            new_keys += 1

    # Fix: TEMPORAL_REF was removed from ontology → remap to DATE
    fixed = 0
    for k, v in merged.items():
        if v == "TEMPORAL_REF":
            merged[k] = "DATE"
            print(f"  Fixed: {k}: TEMPORAL_REF → DATE")
            fixed += 1

    print(f"New keys added: {new_keys}")
    print(f"TEMPORAL_REF fixes: {fixed}")
    print(f"Merged total: {len(merged)}")

    canonical_types = sorted(set(merged.values()))
    print(f"Canonical types ({len(canonical_types)}): {canonical_types}")

    sorted_merged = dict(sorted(merged.items()))
    with open(TRAIN_PATH, "w") as f:
        json.dump(sorted_merged, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(sorted_merged)} entries to {TRAIN_PATH}")


if __name__ == "__main__":
    main()
