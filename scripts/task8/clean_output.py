#!/usr/bin/env python3
"""Post-processing filter for Task 8 annotation output.

Removes noise entities discovered from spot-checking GPT-5.4 output
on Japanese roleplay conversations. Blocklist-based.

Usage:
    python scripts/task8/clean_output.py                              # clean all
    python scripts/task8/clean_output.py jp_roleplay                  # clean one source
    python scripts/task8/clean_output.py --file batch_071 jp_roleplay # clean one file
    python scripts/task8/clean_output.py --dry-run                    # report only
"""

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
BATCH_DIR = BASE_DIR / "data" / "task8_batches"
OUTPUT_DIR = BASE_DIR / "data" / "task8_output"

NOISE_ACTIVITY = {"頭", "手", "足", "目", "耳", "地獄耳", "冬服", "制服", "寝顔"}
NOISE_EMOTION = {
    "ありがと", "ありがとう", "わーい", "おかえり", "おかえりなさい",
    "ただいま", "おはよう", "おやすみ", "ハゲる", "髪ぐしゃぐしゃ",
    "カッコイイ", "かっこいい",
}
BARE_GENDER = {"女", "男", "女の子", "男の子", "若い子", "今時女子", "話し相手"}
NOISE_EVENT = {"募集", "仕事終わった", "寝落ち"}
NOISE_DATE = {("DATE/Season", "冬服"), ("DATE/Season", "制服")}


def is_noise(ent: dict) -> bool:
    t = ent.get("type", "")
    surface = ent.get("surface", "")

    if t.startswith("ACTIVITY") and surface in NOISE_ACTIVITY:
        return True
    if t.startswith("EMOTION") and surface in NOISE_EMOTION:
        return True
    if t.startswith("RELATIONSHIP_REF") and surface in BARE_GENDER:
        return True
    if t.startswith("EVENT") and surface in NOISE_EVENT:
        return True
    if (t, surface) in NOISE_DATE:
        return True

    return False


def clean_file(output_path: Path) -> dict:
    with open(output_path) as f:
        records = json.load(f)

    total_before = 0
    total_after = 0
    dropped_examples = []

    for rec in records:
        entities = rec.get("entities", [])
        total_before += len(entities)

        kept = []
        for ent in entities:
            if is_noise(ent):
                if len(dropped_examples) < 20:
                    dropped_examples.append(
                        f"{rec['id']}: {ent.get('type')} '{ent.get('surface')}'"
                    )
            else:
                kept.append(ent)

        rec["entities"] = kept
        total_after += len(kept)

    dropped_count = total_before - total_after

    return {
        "total_before": total_before,
        "total_after": total_after,
        "dropped_count": dropped_count,
        "dropped_examples": dropped_examples,
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", nargs="?", help="Source name (jp_roleplay or chatharuhi)")
    parser.add_argument("--file", help="Specific batch stem (e.g. batch_071)")
    parser.add_argument("--dry-run", action="store_true", help="Report only, don't modify files")
    args = parser.parse_args()

    sources = []
    if args.source:
        sources = [args.source]
    else:
        if OUTPUT_DIR.exists():
            for d in OUTPUT_DIR.iterdir():
                if d.is_dir():
                    sources.append(d.name)

    if not sources:
        print("No output found yet.")
        return

    grand = {"files": 0, "before": 0, "after": 0, "dropped": 0}

    for source in sorted(sources):
        out_dir = OUTPUT_DIR / source

        if not out_dir.exists():
            print(f"{source}: no output directory")
            continue

        output_files = sorted(out_dir.glob("*.json"))

        for of in output_files:
            if args.file and of.stem != args.file:
                continue

            result = clean_file(of)
            grand["files"] += 1
            grand["before"] += result["total_before"]
            grand["after"] += result["total_after"]
            grand["dropped"] += result["dropped_count"]

            if result["dropped_count"] > 0:
                print(f"\n{source}/{of.name}:")
                print(f"  entities: {result['total_before']} -> {result['total_after']} "
                      f"(dropped {result['dropped_count']})")
                for ex in result["dropped_examples"]:
                    print(f"    {ex}")

                if not args.dry_run:
                    with open(of, "w") as f:
                        json.dump(result["records"], f, ensure_ascii=False, indent=2)
                        f.write("\n")

    mode = "DRY RUN" if args.dry_run else "CLEANED"
    print(f"\n{'='*50}")
    print(f"[{mode}] Files processed: {grand['files']}")
    print(f"Total entities: {grand['before']} -> {grand['after']}")
    print(f"Dropped: {grand['dropped']}")


if __name__ == "__main__":
    main()
