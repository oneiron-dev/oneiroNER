#!/usr/bin/env python3
"""Merge approved backfills into training-ready v2 silver files.

Output goes to data/processed/task9_5/ to prevent double-counting with v1.
"""

import argparse
import copy
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (
    BASE_DIR,
    CANONICAL_TYPES,
    entity_key,
    print_summary,
    validate_offset,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

BACKFILL_ONLY_FIELDS = {"entity_key", "generic_flag", "specific_ref_confidence"}


def update_query_types(query_types, old_type, new_type):
    types_set = set(query_types)
    types_set.add(new_type)
    types_set.add(old_type)
    canonical_order = {t: i for i, t in enumerate(CANONICAL_TYPES)}
    return sorted(types_set, key=lambda t: canonical_order.get(t, 999))


def load_backfill_index(path):
    index = {}
    if not path.exists():
        logger.warning("Backfill file not found: %s", path)
        return index
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for ent in record.get("entities", []):
                ekey = ent.get("entity_key")
                if ekey:
                    index[tuple(ekey)] = ent
    return index


def extract_source_name(filepath):
    name = filepath.stem
    m = re.match(r"task9_silver_(.*)", name)
    return m.group(1) if m else name


def main():
    parser = argparse.ArgumentParser(description="Merge approved backfills into v2 silver files")
    parser.add_argument("--original-glob", default="data/processed/task9_silver_*.jsonl")
    parser.add_argument("--event-backfill", default="data/processed/task9_5_event_backfill.jsonl")
    parser.add_argument("--rel-backfill", default="data/processed/task9_5_rel_backfill.jsonl")
    parser.add_argument("--output-dir", default="data/processed/task9_5/")
    parser.add_argument("--min-confidence", type=float, default=0.7)
    parser.add_argument("--exclude-generic", action="store_true", help="Skip entities with generic_flag=true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    event_path = BASE_DIR / args.event_backfill
    rel_path = BASE_DIR / args.rel_backfill
    output_dir = BASE_DIR / args.output_dir
    glob_path = BASE_DIR / args.original_glob

    logger.info("Loading EVENT backfill from %s", event_path)
    event_index = load_backfill_index(event_path)
    logger.info("Loaded %d EVENT backfill entries", len(event_index))

    logger.info("Loading REL backfill from %s", rel_path)
    rel_index = load_backfill_index(rel_path)
    logger.info("Loaded %d REL backfill entries", len(rel_index))

    backfill_index = {**event_index, **rel_index}
    stats = Counter()

    parent = glob_path.parent
    pattern = glob_path.name
    input_files = sorted(parent.glob(pattern))

    if not input_files:
        logger.error("No input files matched: %s", glob_path)
        return

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    for filepath in input_files:
        stats["files_processed"] += 1
        source_name = extract_source_name(filepath)
        out_path = output_dir / f"task9_silver_v2_{source_name}.jsonl"

        records_in = []
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if line:
                    records_in.append(json.loads(line))

        records_out = []
        for record in records_in:
            stats["records_processed"] += 1
            record = copy.deepcopy(record)
            source = record.get("source", "")
            source_id = record.get("source_id", "")

            for ent in record.get("entities", []):
                ekey = entity_key(ent, source, source_id)
                backfill = backfill_index.get(ekey)
                if not backfill:
                    continue

                conf = backfill.get("subtype_confidence", 0.0)
                if conf < args.min_confidence:
                    stats["skipped_low_confidence"] += 1
                    continue

                if args.exclude_generic and backfill.get("generic_flag", False):
                    stats["skipped_generic"] += 1
                    continue

                old_type = ent["type"]
                new_type = backfill.get("type", old_type)
                if new_type == old_type:
                    continue

                ent["type"] = new_type

                for prov_field in ("subtype_source", "subtype_method", "subtype_confidence",
                                   "subtype_model", "subtype_prompt_version", "subtype_timestamp"):
                    if prov_field in backfill:
                        ent[prov_field] = backfill[prov_field]

                query_types = record.get("query_types", [])
                record["query_types"] = update_query_types(query_types, old_type, new_type)
                stats["query_types_updated"] += 1

                if old_type.startswith("EVENT"):
                    stats["entities_updated_event"] += 1
                elif old_type.startswith("RELATIONSHIP_REF"):
                    stats["entities_updated_rel"] += 1
                stats["entities_updated_total"] += 1

                for field in BACKFILL_ONLY_FIELDS:
                    ent.pop(field, None)

            for ent in record.get("entities", []):
                assert validate_offset(record, ent), (
                    f"Offset mismatch: source={source} id={source_id} "
                    f"surface={ent.get('surface')!r}"
                )
                stats["offset_validations_passed"] += 1

            records_out.append(record)

        assert len(records_out) == len(records_in), (
            f"Record count mismatch for {filepath.name}: "
            f"{len(records_in)} in, {len(records_out)} out"
        )

        if args.dry_run:
            logger.info("[dry-run] Would write %d records to %s", len(records_out), out_path)
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                for rec in records_out:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            logger.info("Wrote %d records to %s", len(records_out), out_path)

    print("\n=== APPLY BACKFILLS SUMMARY ===")
    print_summary(stats)


if __name__ == "__main__":
    main()
