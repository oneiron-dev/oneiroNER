#!/usr/bin/env python3
"""Verify Task 8 annotation output for offset accuracy.

Usage:
    python scripts/task8/verify_output.py                    # verify all
    python scripts/task8/verify_output.py jp_roleplay        # verify one source
    python scripts/task8/verify_output.py --file batch_000   # verify one file
"""

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
BATCH_DIR = BASE_DIR / "data" / "task8_batches"
OUTPUT_DIR = BASE_DIR / "data" / "task8_output"


def verify_file(batch_path: Path, output_path: Path) -> dict:
    with open(batch_path) as f:
        batch = json.load(f)
    with open(output_path) as f:
        output = json.load(f)

    batch_by_id = {r["id"]: r for r in batch}
    output_by_id = {r["id"]: r for r in output}

    stats = {
        "batch_convos": len(batch),
        "output_convos": len(output),
        "missing_convos": [],
        "total_entities": 0,
        "good_offsets": 0,
        "bad_offsets": 0,
        "bad_examples": [],
        "type_counts": {},
    }

    for conv_id in batch_by_id:
        if conv_id not in output_by_id:
            stats["missing_convos"].append(conv_id)

    for rec in output:
        conv = batch_by_id.get(rec["id"])
        if not conv:
            continue
        turns = conv["turns"]

        for ent in rec.get("entities", []):
            stats["total_entities"] += 1
            t = ent.get("type", "UNKNOWN")
            stats["type_counts"][t] = stats["type_counts"].get(t, 0) + 1

            ti = ent.get("turn_index", -1)
            start = ent.get("start", -1)
            end = ent.get("end", -1)
            surface = ent.get("surface", "")

            if ti < 0 or ti >= len(turns):
                stats["bad_offsets"] += 1
                if len(stats["bad_examples"]) < 5:
                    stats["bad_examples"].append(
                        f"{rec['id']}: turn_index {ti} out of range (max {len(turns)-1})"
                    )
                continue

            text = turns[ti]["text"]
            actual = text[start:end]

            if actual == surface:
                stats["good_offsets"] += 1
            else:
                stats["bad_offsets"] += 1
                if len(stats["bad_examples"]) < 5:
                    stats["bad_examples"].append(
                        f"{rec['id']}: '{surface}' != '{actual}' at [{start}:{end}] in turn {ti}"
                    )

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", nargs="?", help="Source name (jp_roleplay or chatharuhi)")
    parser.add_argument("--file", help="Specific batch stem (e.g. batch_000)")
    args = parser.parse_args()

    sources = []
    if args.source:
        sources = [args.source]
    else:
        for d in OUTPUT_DIR.iterdir():
            if d.is_dir():
                sources.append(d.name)

    if not sources:
        print("No output found yet.")
        return

    grand = {"total_entities": 0, "good": 0, "bad": 0, "files": 0, "missing_files": 0}

    for source in sorted(sources):
        batch_dir = BATCH_DIR / source
        out_dir = OUTPUT_DIR / source

        if not out_dir.exists():
            print(f"{source}: no output directory")
            continue

        batch_files = sorted(batch_dir.glob("*.json"))

        for bf in batch_files:
            if args.file and bf.stem != args.file:
                continue

            of = out_dir / bf.name
            if not of.exists():
                grand["missing_files"] += 1
                continue

            stats = verify_file(bf, of)
            grand["files"] += 1
            grand["total_entities"] += stats["total_entities"]
            grand["good"] += stats["good_offsets"]
            grand["bad"] += stats["bad_offsets"]

            if stats["bad_offsets"] > 0 or stats["missing_convos"]:
                print(f"\n{source}/{bf.name}:")
                print(f"  convos: {stats['output_convos']}/{stats['batch_convos']}, "
                      f"missing: {len(stats['missing_convos'])}")
                print(f"  entities: {stats['total_entities']}, "
                      f"good: {stats['good_offsets']}, bad: {stats['bad_offsets']}")
                for ex in stats["bad_examples"]:
                    print(f"    {ex}")

    total = grand["good"] + grand["bad"]
    if total > 0:
        pct = grand["good"] / total * 100
    else:
        pct = 0

    print(f"\n{'='*50}")
    print(f"Files verified: {grand['files']}, missing: {grand['missing_files']}")
    print(f"Total entities: {grand['total_entities']}")
    print(f"Offset accuracy: {grand['good']}/{total} ({pct:.1f}%)")
    if grand["bad"] > 0:
        print(f"BAD OFFSETS: {grand['bad']}")


if __name__ == "__main__":
    main()
