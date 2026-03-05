#!/usr/bin/env python3
"""Task 9 Step 3: post-process annotated silver data into NerRecord/ConversationRecord."""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.negative_sampler import NegativeSampler
from lib.schema import ConversationRecord, NerRecord
from lib.span_fixer import verify_and_fix_spans

ANNOTATED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "silver_annotated"
OUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"

ALL_SOURCES = [
    "reddit_confessions", "opencharacter", "therapy_conversations",
    "prosocial_dialog", "personachat", "pippa",
    "synthetic_persona_chat", "roleplay_hieu", "mentalchat",
]

SILVER_TYPE_COUNTS = {
    "PERSON": 1000, "PLACE": 1000, "ORG": 1000,
    "DATE/Day": 500, "DATE/Week": 200, "DATE/Month": 500, "DATE/Season": 200,
    "DATE/Year": 500, "DATE/Decade": 100, "DATE/Relative": 500, "DATE/Range": 200,
    "EVENT": 500,
    "RELATIONSHIP_REF/Family": 500, "RELATIONSHIP_REF/Romantic": 300,
    "RELATIONSHIP_REF/Friend": 300, "RELATIONSHIP_REF/Professional": 300,
    "RELATIONSHIP_REF/Acquaintance": 200,
    "EMOTION": 500, "GOAL": 300, "ACTIVITY": 500,
}


def dedup_entities(entities: list[dict]) -> list[dict]:
    seen = set()
    deduped = []
    for ent in entities:
        if ent.get("turn_index") is not None:
            key = (ent["surface"], ent["start"], ent["end"], ent["type"], ent["turn_index"])
        else:
            key = (ent["surface"], ent["start"], ent["end"], ent["type"])
        if key not in seen:
            seen.add(key)
            deduped.append(ent)
    return deduped


def build_entity_dicts(entities: list[dict], is_conversation: bool) -> list[dict]:
    result = []
    for ent in entities:
        d = {
            "surface": ent["surface"],
            "type": ent["type"],
            "original_type": ent["type"],
            "start": ent["start"],
            "end": ent["end"],
        }
        if is_conversation:
            d["turn_index"] = ent["turn_index"]
        result.append(d)
    return result


def process_source(source: str, sampler: NegativeSampler, rng: random.Random) -> dict:
    in_path = ANNOTATED_DIR / f"{source}.jsonl"
    if not in_path.exists():
        print(f"  Skipping {source}: no annotated file")
        return {"source": source, "records": 0, "entities": 0, "types": Counter(), "errors": 0}

    out_path = OUT_DIR / f"task9_silver_{source}.jsonl"
    stats = {"source": source, "records": 0, "entities": 0, "types": Counter(), "errors": 0}

    with open(in_path) as fin, open(out_path, "w") as fout:
        for line_no, line in enumerate(fin):
            rec = json.loads(line)
            fmt = rec.get("format", "passage")
            is_conv = fmt == "conversation"
            raw_entities = rec.get("entities", [])

            if is_conv:
                fixed, fix_stats = verify_and_fix_spans(raw_entities, "", turns=rec.get("turns", []))
            else:
                fixed, fix_stats = verify_and_fix_spans(raw_entities, rec.get("text", ""))

            fixed = dedup_entities(fixed)
            entity_dicts = build_entity_dicts(fixed, is_conv)
            pos_types = {e["type"] for e in entity_dicts}

            if not pos_types:
                neg_types = sampler.sample(set(), rng=rng)
            else:
                neg_types = sampler.sample(pos_types, rng=rng)
            query_types = sorted(pos_types) + neg_types

            sid = rec.get("source_id", str(line_no))

            try:
                if is_conv:
                    record = ConversationRecord(
                        source=source,
                        source_id=sid,
                        language="en",
                        split="train",
                        confidence="silver",
                        provenance=[source, "task9_silver"],
                        turns=rec["turns"],
                        query_types=query_types,
                        entities=entity_dicts,
                    )
                    record.validate()
                else:
                    record = NerRecord(
                        source=source,
                        source_id=sid,
                        language="en",
                        split="train",
                        confidence="silver",
                        provenance=[source, "task9_silver"],
                        text=rec["text"],
                        query_types=query_types,
                        entities=entity_dicts,
                    )
                    record.validate()

                fout.write(record.to_jsonl() + "\n")
                stats["records"] += 1
                stats["entities"] += len(entity_dicts)
                for e in entity_dicts:
                    stats["types"][e["type"]] += 1

            except AssertionError as e:
                stats["errors"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description="Convert annotated silver data to NER records")
    parser.add_argument("--source", default="all", help="Source name or 'all'")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sources = ALL_SOURCES if args.source == "all" else [args.source]
    for s in sources:
        if s not in ALL_SOURCES:
            print(f"Unknown source: {s}", file=sys.stderr)
            sys.exit(1)

    sampler = NegativeSampler(type_counts=SILVER_TYPE_COUNTS)
    rng = random.Random(42)

    total = {"records": 0, "entities": 0, "types": Counter(), "errors": 0}
    for s in sources:
        print(f"Processing {s}...")
        stats = process_source(s, sampler, rng)
        print(f"  records={stats['records']:,} entities={stats['entities']:,} errors={stats['errors']}")
        total["records"] += stats["records"]
        total["entities"] += stats["entities"]
        total["types"] += stats["types"]
        total["errors"] += stats["errors"]

    print(f"\nTotal: {total['records']:,} records, {total['entities']:,} entities, {total['errors']} errors")
    print("\nPer-type distribution:")
    for t, c in sorted(total["types"].items(), key=lambda x: -x[1]):
        print(f"  {t}: {c:,}")

    rel_types = {k: v for k, v in total["types"].items() if k.startswith("RELATIONSHIP_REF")}
    if rel_types:
        print("\nRELATIONSHIP_REF breakdown:")
        for t, c in sorted(rel_types.items(), key=lambda x: -x[1]):
            print(f"  {t}: {c:,}")

    print(f"\nOutput: {OUT_DIR}/task9_silver_*.jsonl")


if __name__ == "__main__":
    main()
