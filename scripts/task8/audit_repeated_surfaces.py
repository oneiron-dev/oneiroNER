#!/usr/bin/env python3
"""Audit Task 8 output for ambiguous offsets from repeated surface strings.

text.find(surface) returns the FIRST occurrence. If the same surface appears
multiple times in a turn, the offset may point to the wrong one. This script
detects and reports such cases.

Usage:
    python scripts/task8/audit_repeated_surfaces.py                    # audit all
    python scripts/task8/audit_repeated_surfaces.py jp_roleplay        # one source
    python scripts/task8/audit_repeated_surfaces.py --file batch_071   # one file
"""

import argparse
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
BATCH_DIR = BASE_DIR / "data" / "task8_batches"
OUTPUT_DIR = BASE_DIR / "data" / "task8_output"


def audit_file(batch_path: Path, output_path: Path) -> dict:
    with open(batch_path) as f:
        batch = json.load(f)
    with open(output_path) as f:
        output = json.load(f)

    batch_by_id = {r["id"]: r for r in batch}

    total_entities = 0
    ambiguous = 0
    wrong_occurrence = 0
    examples = []

    for rec in output:
        conv = batch_by_id.get(rec["id"])
        if not conv:
            continue
        turns = conv["turns"]

        for ent in rec.get("entities", []):
            total_entities += 1
            ti = ent.get("turn_index", -1)
            start = ent.get("start", -1)
            surface = ent.get("surface", "")

            if ti < 0 or ti >= len(turns) or not surface:
                continue

            text = turns[ti]["text"]
            first_pos = text.find(surface)
            if first_pos == -1:
                continue

            count = text.count(surface)
            if count <= 1:
                continue

            ambiguous += 1

            if start != first_pos:
                all_positions = []
                pos = 0
                while True:
                    pos = text.find(surface, pos)
                    if pos == -1:
                        break
                    all_positions.append(pos)
                    pos += 1

                if start not in all_positions:
                    wrong_occurrence += 1
                    if len(examples) < 10:
                        examples.append(
                            f"{rec['id']} turn {ti}: '{surface}' appears {count}x, "
                            f"offset {start} not at any occurrence (positions: {all_positions})"
                        )
                elif len(examples) < 10:
                    examples.append(
                        f"{rec['id']} turn {ti}: '{surface}' appears {count}x, "
                        f"offset {start} (not first at {first_pos}) — valid but ambiguous"
                    )

    return {
        "total_entities": total_entities,
        "ambiguous": ambiguous,
        "wrong_occurrence": wrong_occurrence,
        "examples": examples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", nargs="?")
    parser.add_argument("--file", help="Specific batch stem")
    args = parser.parse_args()

    sources = []
    if args.source:
        sources = [args.source]
    elif OUTPUT_DIR.exists():
        sources = [d.name for d in OUTPUT_DIR.iterdir() if d.is_dir()]

    if not sources:
        print("No output found.")
        return

    grand = {"total": 0, "ambiguous": 0, "wrong": 0, "files": 0}

    for source in sorted(sources):
        batch_dir = BATCH_DIR / source
        out_dir = OUTPUT_DIR / source

        if not out_dir.exists():
            continue

        for of in sorted(out_dir.glob("*.json")):
            if args.file and of.stem != args.file:
                continue

            bf = batch_dir / of.name
            if not bf.exists():
                stem = of.stem.split("_gpt_")[0].split("_v")[0]
                bf = batch_dir / f"{stem}.json"
            if not bf.exists():
                continue

            result = audit_file(bf, of)
            grand["files"] += 1
            grand["total"] += result["total_entities"]
            grand["ambiguous"] += result["ambiguous"]
            grand["wrong"] += result["wrong_occurrence"]

            if result["ambiguous"] > 0:
                print(f"\n{source}/{of.name}:")
                print(f"  entities: {result['total_entities']}, "
                      f"ambiguous: {result['ambiguous']}, "
                      f"wrong: {result['wrong_occurrence']}")
                for ex in result["examples"]:
                    print(f"    {ex}")

    total = grand["total"]
    amb = grand["ambiguous"]
    wrong = grand["wrong"]
    amb_pct = (amb / total * 100) if total else 0
    wrong_pct = (wrong / total * 100) if total else 0

    print(f"\n{'='*50}")
    print(f"Files: {grand['files']}")
    print(f"Total entities: {total}")
    print(f"Ambiguous (repeated surface in turn): {amb} ({amb_pct:.1f}%)")
    print(f"Wrong occurrence (offset not at any position): {wrong} ({wrong_pct:.1f}%)")
    if amb > 0 and wrong == 0:
        print("All ambiguous offsets point to valid occurrences (but may not be the intended one).")


if __name__ == "__main__":
    main()
