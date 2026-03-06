#!/usr/bin/env python3
"""Verify offset accuracy and entity quality in synthetic NER JSONL files."""
import json
import sys
from collections import Counter
from pathlib import Path


def verify_file(path: str) -> dict:
    path = Path(path)
    if not path.exists():
        return {"error": f"File not found: {path}"}

    records = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                return {"error": f"JSON parse error line {i}: {e}"}

    total_ents = 0
    bad_offsets = []
    type_counts = Counter()
    ents_per_convo = []

    for rec_idx, rec in enumerate(records):
        turns = rec.get("turns", [])
        entities = rec.get("entities", [])
        ents_per_convo.append(len(entities))
        total_ents += len(entities)

        for ent_idx, ent in enumerate(entities):
            ti = ent.get("turn_index", 0)
            start = ent.get("start", 0)
            end = ent.get("end", 0)
            surface = ent.get("surface", "")
            etype = ent.get("type", "UNKNOWN")
            type_counts[etype.split("/")[0]] += 1

            if ti >= len(turns):
                bad_offsets.append({
                    "record": rec_idx + 1,
                    "entity": ent_idx,
                    "issue": f"turn_index {ti} >= {len(turns)} turns",
                    "surface": surface,
                })
                continue

            text = turns[ti].get("text", "")
            actual = text[start:end]
            if actual != surface:
                bad_offsets.append({
                    "record": rec_idx + 1,
                    "entity": ent_idx,
                    "surface": surface,
                    "actual": actual,
                    "start": start,
                    "end": end,
                    "turn_index": ti,
                })

    avg_ents = sum(ents_per_convo) / len(ents_per_convo) if ents_per_convo else 0
    base_types = {"PERSON", "PLACE", "ORG", "DATE", "EVENT",
                  "RELATIONSHIP_REF", "EMOTION", "GOAL", "ACTIVITY"}
    covered = base_types & set(type_counts.keys())

    result = {
        "file": str(path),
        "records": len(records),
        "total_entities": total_ents,
        "avg_ents_per_convo": round(avg_ents, 1),
        "bad_offsets": len(bad_offsets),
        "type_coverage": f"{len(covered)}/{len(base_types)}",
        "type_counts": dict(type_counts.most_common()),
        "missing_types": sorted(base_types - covered),
    }
    if bad_offsets:
        result["bad_offset_details"] = bad_offsets[:10]

    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_synthetic.py <file.jsonl> [file2.jsonl ...]")
        sys.exit(1)

    all_pass = True
    for fpath in sys.argv[1:]:
        result = verify_file(fpath)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if result.get("bad_offsets", 0) > 0 or "error" in result:
            all_pass = False
        print()

    sys.exit(0 if all_pass else 1)
