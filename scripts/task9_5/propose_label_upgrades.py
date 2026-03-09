#!/usr/bin/env python3
"""Compare silver labels against backfill proposals.

Generates upgrade candidates report for manual review.
No gold data exists — this compares original silver vs proposed backfills.
"""

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (
    BASE_DIR,
    CANONICAL_TYPES,
    entity_key,
    get_entity_text,
    load_silver_records,
    print_summary,
    write_output_jsonl,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


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


def extract_context_snippet(record, entity, window=100):
    text = get_entity_text(record, entity)
    start = max(0, entity["start"] - window // 2)
    end = min(len(text), entity["end"] + window // 2)
    return text[start:end]


def generate_summary_md(diffs, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    type_changes = Counter()
    confidence_buckets = Counter()
    method_counts = Counter()
    generic_count = 0
    surface_counts = Counter()

    for d in diffs:
        type_changes[f"{d['old_type']} -> {d['new_type']}"] += 1
        conf = d["confidence"]
        if conf < 0.5:
            confidence_buckets["<0.5"] += 1
        elif conf < 0.7:
            confidence_buckets["0.5-0.7"] += 1
        elif conf < 0.9:
            confidence_buckets["0.7-0.9"] += 1
        else:
            confidence_buckets[">=0.9"] += 1
        method_counts[d["method"]] += 1
        if d.get("generic_flag"):
            generic_count += 1
        surface_counts[d["surface"].lower()] += 1

    top_surfaces = surface_counts.most_common(20)
    low_conf = [d for d in diffs if d["confidence"] < 0.7][:5]

    total = len(diffs)
    lines = [
        f"# Task 9.5 Label Upgrade Summary",
        f"",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC",
        f"",
        f"Total upgrade candidates: {total}",
        f"",
        f"## Type Change Counts",
        f"",
        f"| Change | Count | % |",
        f"|--------|------:|--:|",
    ]
    for change, count in type_changes.most_common():
        pct = count / total * 100 if total else 0
        lines.append(f"| {change} | {count} | {pct:.1f}% |")

    lines += [
        f"",
        f"## Confidence Distribution",
        f"",
        f"| Bucket | Count | % |",
        f"|--------|------:|--:|",
    ]
    for bucket in ["<0.5", "0.5-0.7", "0.7-0.9", ">=0.9"]:
        count = confidence_buckets.get(bucket, 0)
        pct = count / total * 100 if total else 0
        lines.append(f"| {bucket} | {count} | {pct:.1f}% |")

    lines += [
        f"",
        f"## Method Breakdown",
        f"",
        f"| Method | Count | % |",
        f"|--------|------:|--:|",
    ]
    for method, count in method_counts.most_common():
        pct = count / total * 100 if total else 0
        lines.append(f"| {method} | {count} | {pct:.1f}% |")

    lines += [
        f"",
        f"## Generic-Flagged",
        f"",
        f"Total generic-flagged: {generic_count} ({generic_count / total * 100:.1f}%)" if total else "Total generic-flagged: 0",
        f"",
        f"## Top 20 Most Changed Surfaces",
        f"",
        f"| Surface | Count |",
        f"|---------|------:|",
    ]
    for surface, count in top_surfaces:
        lines.append(f"| {surface} | {count} |")

    lines += [
        f"",
        f"## Sample Low-Confidence Changes",
        f"",
    ]
    if low_conf:
        for d in low_conf:
            lines.append(f"- **{d['surface']}**: {d['old_type']} -> {d['new_type']} "
                         f"(conf={d['confidence']:.2f}, method={d['method']})")
            lines.append(f"  Context: `{d.get('context_snippet', 'N/A')}`")
            lines.append(f"")
    else:
        lines.append("No low-confidence changes found.")

    lines += [
        f"",
        f"## Recommended Next Steps",
        f"",
        f"1. Review low-confidence changes (conf < 0.7) before applying",
        f"2. Spot-check generic-flagged RELATIONSHIP_REF entities",
        f"3. Run `apply_backfills.py` with `--min-confidence 0.7` to apply approved changes",
        f"4. Consider LLM re-classification for remaining flat EVENT/RELATIONSHIP_REF spans",
        f"",
    ]

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Compare silver labels against backfill proposals")
    parser.add_argument("--silver-event", default="data/processed/task9_5_event_backfill.jsonl")
    parser.add_argument("--silver-rel", default="data/processed/task9_5_rel_backfill.jsonl")
    parser.add_argument("--original-glob", default="data/processed/task9_silver_*.jsonl")
    parser.add_argument("--output-jsonl", default="reports/task9_5_label_upgrade_candidates.jsonl")
    parser.add_argument("--output-md", default="reports/task9_5_label_upgrade_summary.md")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    event_path = BASE_DIR / args.silver_event
    rel_path = BASE_DIR / args.silver_rel
    out_jsonl = BASE_DIR / args.output_jsonl
    out_md = BASE_DIR / args.output_md

    logger.info("Loading EVENT backfill from %s", event_path)
    event_index = load_backfill_index(event_path)
    logger.info("Loaded %d EVENT backfill entries", len(event_index))

    logger.info("Loading REL backfill from %s", rel_path)
    rel_index = load_backfill_index(rel_path)
    logger.info("Loaded %d REL backfill entries", len(rel_index))

    backfill_index = {**event_index, **rel_index}
    if not backfill_index:
        logger.warning("No backfill entries found, nothing to compare")
        return

    glob_path = BASE_DIR / args.original_glob
    stats = Counter()
    diffs = []

    logger.info("Scanning original silver records...")
    for filepath, line_num, record in load_silver_records(glob_path):
        stats["records_scanned"] += 1
        source = record.get("source", "")
        source_id = record.get("source_id", "")

        for ent in record.get("entities", []):
            ekey = entity_key(ent, source, source_id)
            backfill = backfill_index.get(ekey)
            if not backfill:
                continue

            new_type = backfill.get("type", ent["type"])
            old_type = ent["type"]
            if new_type == old_type:
                stats["no_change"] += 1
                continue

            context = extract_context_snippet(record, ent)
            diff = {
                "entity_key": list(ekey),
                "old_type": old_type,
                "new_type": new_type,
                "confidence": backfill.get("subtype_confidence", 0.0),
                "method": backfill.get("subtype_method", "unknown"),
                "generic_flag": backfill.get("generic_flag", False),
                "context_snippet": context,
                "source": source,
                "source_id": source_id,
                "surface": ent["surface"],
            }
            diffs.append(diff)
            stats["upgrades_found"] += 1
            stats[f"change:{old_type}->{new_type}"] += 1

    logger.info("Found %d upgrade candidates across %d records", len(diffs), stats["records_scanned"])

    if args.dry_run:
        print("\n=== DRY RUN SUMMARY ===")
        print_summary(stats)
        return

    write_output_jsonl(out_jsonl, diffs)
    logger.info("Wrote %d diff records to %s", len(diffs), out_jsonl)

    generate_summary_md(diffs, out_md)
    logger.info("Wrote summary to %s", out_md)

    print("\n=== LABEL UPGRADE SUMMARY ===")
    print_summary(stats)


if __name__ == "__main__":
    main()
