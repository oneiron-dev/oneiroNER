#!/usr/bin/env python3
"""Fix entity offsets in synthetic NER JSONL files by re-finding surfaces in text."""
import json
import sys
from pathlib import Path


def fix_file(path: str) -> dict:
    path = Path(path)
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    fixed = 0
    unfixable = 0
    total = 0

    for rec in records:
        turns = rec.get("turns", [])
        new_entities = []
        for ent in rec.get("entities", []):
            total += 1
            ti = ent["turn_index"]
            surface = ent["surface"]
            text = turns[ti]["text"]

            if text[ent["start"]:ent["end"]] == surface:
                new_entities.append(ent)
                continue

            idx = text.find(surface)
            if idx >= 0:
                ent["start"] = idx
                ent["end"] = idx + len(surface)
                new_entities.append(ent)
                fixed += 1
            else:
                unfixable += 1

        rec["entities"] = new_entities

    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return {"file": str(path), "total": total, "fixed": fixed, "unfixable": unfixable}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python span_fixer.py <file.jsonl> [file2.jsonl ...]")
        sys.exit(1)

    for fpath in sys.argv[1:]:
        result = fix_file(fpath)
        print(json.dumps(result))
