"""Audit train/val split for window leakage."""

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

SUFFIX_RE = re.compile(r'__(w|chunk)\d+$')
CANONICAL_RE = re.compile(r'_canonical$')


def strip_base_id(source_id: str) -> str:
    base = CANONICAL_RE.sub('', source_id)
    base = SUFFIX_RE.sub('', base)
    return base


def load_ids(path: Path) -> dict[str, list[str]]:
    base_to_records: dict[str, list[str]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sid = rec.get("source_id", rec.get("id", ""))
            base = strip_base_id(sid)
            base_to_records[base].append(sid)
    return dict(base_to_records)


def audit(data_dir: Path) -> dict:
    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"

    if not train_path.exists():
        logger.error("train.jsonl not found in %s", data_dir)
        sys.exit(2)
    if not val_path.exists():
        logger.error("val.jsonl not found in %s", data_dir)
        sys.exit(2)

    train_map = load_ids(train_path)
    val_map = load_ids(val_path)

    train_bases = set(train_map.keys())
    val_bases = set(val_map.keys())
    leaked = train_bases & val_bases

    n_train_affected = sum(len(train_map[b]) for b in leaked)
    n_val_affected = sum(len(val_map[b]) for b in leaked)

    report = {
        "train_unique_base_ids": len(train_bases),
        "val_unique_base_ids": len(val_bases),
        "leaked_base_ids": len(leaked),
        "train_records_affected": n_train_affected,
        "val_records_affected": n_val_affected,
        "leaked_examples": sorted(leaked)[:10],
    }
    return report


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Audit train/val split for window leakage")
    parser.add_argument("--data-dir", type=str, default="data/processed/",
                        help="Directory containing train.jsonl and val.jsonl")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    report = audit(data_dir)

    print(f"Train unique base IDs: {report['train_unique_base_ids']}")
    print(f"Val   unique base IDs: {report['val_unique_base_ids']}")
    print(f"Leaked base IDs:       {report['leaked_base_ids']}")

    if report["leaked_base_ids"] > 0:
        print(f"  Train records affected: {report['train_records_affected']}")
        print(f"  Val   records affected: {report['val_records_affected']}")
        print("  First leaked IDs:")
        for lid in report["leaked_examples"]:
            print(f"    {lid}")
        sys.exit(1)
    else:
        print("No leakage detected.")
        sys.exit(0)


if __name__ == "__main__":
    main()
