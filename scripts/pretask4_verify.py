#!/usr/bin/env python3
"""Pre-Task 4: Dataset verification and type frequency scan."""

import json
from collections import defaultdict
from pathlib import Path

B2NERD_DIR = "/tmp/b2nerd_extract"
RAW_DIR = Path("/home/ubuntu/projects/oneiron-ner/data/raw")
OUT_DIR = Path("/home/ubuntu/projects/oneiron-ner/data/processed")


def scan_b2nerd_curated():
    """Count B2NERD curated examples and types, verify pos field."""
    base = Path(B2NERD_DIR) / "B2NERD"
    results = {
        "en_train_dev": 0, "zh_train_dev": 0,
        "en_test": 0, "zh_test": 0,
        "en_other_train_dev": 0, "en_other_test": 0,
        "types_train_dev": set(), "types_test": set(),
        "pos_checks": {},
    }
    type_counts = defaultdict(int)

    for partition in ["NER_en", "NER_zh", "NER_en_other"]:
        part_dir = base / partition
        if not part_dir.exists():
            continue
        for dataset_dir in sorted(part_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            for split_name in ["train.json", "dev.json", "test.json"]:
                split_file = dataset_dir / split_name
                if not split_file.exists():
                    continue
                with open(split_file) as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    continue
                count = len(data)
                is_test = split_name == "test.json"

                if partition == "NER_en":
                    if is_test:
                        results["en_test"] += count
                    else:
                        results["en_train_dev"] += count
                elif partition == "NER_zh":
                    if is_test:
                        results["zh_test"] += count
                    else:
                        results["zh_train_dev"] += count
                elif partition == "NER_en_other":
                    if is_test:
                        results["en_other_test"] += count
                    else:
                        results["en_other_train_dev"] += count

                for entry in data:
                    if not isinstance(entry, dict):
                        continue
                    for ent in entry.get("entities", []):
                        etype = ent.get("type", "")
                        if is_test:
                            results["types_test"].add(etype)
                        else:
                            results["types_train_dev"].add(etype)
                            type_counts[etype] += 1

    results["types_train_dev"] = sorted(results["types_train_dev"])
    results["types_test"] = sorted(results["types_test"])
    return results, dict(type_counts)


def verify_pos_field():
    """Verify pos field presence across B2NERD partitions (exhaustive, per-dataset)."""
    checks = {}
    base = Path(B2NERD_DIR)

    for partition, lang, expected_pos in [
        ("B2NERD/NER_en", "en_curated", True),
        ("B2NERD/NER_zh", "zh_curated", False),
        ("B2NERD_raw/NER_en", "en_raw", True),
        ("B2NERD_raw/NER_zh", "zh_raw", False),
        ("B2NERD_all/NER_en", "en_all", True),
        ("B2NERD_all/NER_zh", "zh_all", False),
    ]:
        part_dir = base / partition
        if not part_dir.exists():
            checks[lang] = {"exists": False}
            continue
        datasets_with_pos = []
        datasets_without_pos = []
        total_entities_with_pos = 0
        total_entities_without_pos = 0
        for dataset_dir in sorted(part_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            train_path = dataset_dir / "train.json"
            if not train_path.exists():
                continue
            with open(train_path) as f:
                data = json.load(f)
            ds_has_pos = False
            ds_has_no_pos = False
            for entry in data:
                if not isinstance(entry, dict):
                    continue
                for ent in entry.get("entities", []):
                    if "pos" in ent:
                        ds_has_pos = True
                        total_entities_with_pos += 1
                    else:
                        ds_has_no_pos = True
                        total_entities_without_pos += 1
            if ds_has_pos:
                datasets_with_pos.append(dataset_dir.name)
            if ds_has_no_pos and not ds_has_pos:
                datasets_without_pos.append(dataset_dir.name)
        total_datasets = len(datasets_with_pos) + len(datasets_without_pos)
        checks[lang] = {
            "datasets_with_pos": len(datasets_with_pos),
            "datasets_without_pos": len(datasets_without_pos),
            "total_datasets": total_datasets,
            "entities_with_pos": total_entities_with_pos,
            "entities_without_pos": total_entities_without_pos,
            "expected_pos": expected_pos,
            "datasets_without_pos_names": datasets_without_pos if expected_pos else None,
            "matches_expectation": (len(datasets_with_pos) > 0) == expected_pos if total_datasets > 0 else None,
        }
    return checks


def verify_pos_semantics():
    """Determine if pos field uses exclusive or inclusive end (exhaustive)."""
    base = Path(B2NERD_DIR) / "B2NERD" / "NER_en"
    exclusive_match = 0
    inclusive_match = 0
    neither = 0
    examples = []
    per_dataset = {}

    for dataset_dir in sorted(base.iterdir()):
        if not dataset_dir.is_dir():
            continue
        ds_excl = 0
        ds_incl = 0
        ds_neither = 0
        for split_name in ["train.json", "dev.json"]:
            split_path = dataset_dir / split_name
            if not split_path.exists():
                continue
            with open(split_path) as f:
                data = json.load(f)
            if not isinstance(data, list):
                continue
            for entry in data:
                if not isinstance(entry, dict):
                    continue
                for ent in entry.get("entities", []):
                    if "pos" not in ent:
                        continue
                    sent = entry["sentence"]
                    name = ent["name"]
                    start, end = ent["pos"]
                    excl = sent[start:end]
                    incl = sent[start:end + 1]
                    if excl == name:
                        ds_excl += 1
                    elif incl == name:
                        ds_incl += 1
                    else:
                        ds_neither += 1
                    if len(examples) < 10:
                        examples.append({
                            "name": name,
                            "pos": [start, end],
                            "slice_exclusive": excl[:50],
                            "slice_inclusive": incl[:50],
                            "match": "exclusive" if excl == name else ("inclusive" if incl == name else "neither"),
                        })
        if ds_excl + ds_incl + ds_neither > 0:
            per_dataset[dataset_dir.name] = {
                "exclusive": ds_excl, "inclusive": ds_incl, "neither": ds_neither
            }
        exclusive_match += ds_excl
        inclusive_match += ds_incl
        neither += ds_neither

    return {
        "exclusive_matches": exclusive_match,
        "inclusive_matches": inclusive_match,
        "neither": neither,
        "datasets_checked": len(per_dataset),
        "convention": "exclusive" if exclusive_match > inclusive_match else "inclusive",
        "per_dataset": per_dataset,
        "examples": examples[:5],
    }


def verify_chinese_ner_sft_offsets():
    """Detect offset convention (exclusive vs inclusive) per chinese_ner_sft subset."""
    data_dir = RAW_DIR / "chinese_ner_sft" / "data"
    results = {}

    for jsonl_file in sorted(data_dir.glob("*.jsonl")):
        subset = jsonl_file.stem
        exclusive_ok = 0
        inclusive_ok = 0
        neither = 0
        null_count = 0
        checked = 0

        with open(jsonl_file) as f:
            for line in f:
                if checked >= 50:
                    break
                entry = json.loads(line)
                text = entry.get("text", "")
                for ent in entry.get("entities", []):
                    if checked >= 50:
                        break
                    start = ent.get("start_idx")
                    end = ent.get("end_idx")
                    surface = ent.get("entity_text", "")
                    if start is None or end is None:
                        null_count += 1
                        continue
                    excl = text[start:end]
                    incl = text[start:end + 1]
                    if excl == surface:
                        exclusive_ok += 1
                    elif incl == surface:
                        inclusive_ok += 1
                    else:
                        neither += 1
                    checked += 1

        total = exclusive_ok + inclusive_ok + neither
        if total == 0:
            convention = "all_null"
        elif exclusive_ok / total > 0.9:
            convention = "exclusive"
        elif inclusive_ok / total > 0.9:
            convention = "inclusive"
        else:
            convention = "mixed"

        results[subset] = {
            "exclusive": exclusive_ok,
            "inclusive": inclusive_ok,
            "neither": neither,
            "null_offsets": null_count,
            "checked": total,
            "convention": convention,
        }

    return results


def scan_b2nerd_types():
    """Scan B2NERD curated train+dev for type frequencies."""
    base = Path(B2NERD_DIR) / "B2NERD"
    type_counts = defaultdict(int)
    for partition in ["NER_en", "NER_zh", "NER_en_other"]:
        part_dir = base / partition
        if not part_dir.exists():
            continue
        for dataset_dir in sorted(part_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            for split_name in ["train.json", "dev.json"]:
                split_file = dataset_dir / split_name
                if not split_file.exists():
                    continue
                with open(split_file) as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    continue
                for entry in data:
                    if not isinstance(entry, dict):
                        continue
                    for ent in entry.get("entities", []):
                        type_counts[ent.get("type", "")] += 1
    return dict(type_counts)


def scan_open_ner(base_path):
    """Scan open-ner-standardized or open-ner-core-types parquet files."""
    from datasets import load_dataset

    type_counts = defaultdict(int)
    base = Path(base_path)

    for config_dir in sorted(base.iterdir()):
        if not config_dir.is_dir() or config_dir.name == "README.md":
            continue
        for lang_dir in sorted(config_dir.iterdir()):
            if not lang_dir.is_dir():
                continue
            parquet_files = list(lang_dir.glob("*.parquet"))
            if not parquet_files:
                continue
            # Only use train split for frequency scan
            train_files = [f for f in parquet_files if "train" in f.name]
            if not train_files:
                train_files = parquet_files
            for pf in train_files:
                try:
                    ds = load_dataset("parquet", data_files=str(pf), split="train")
                except Exception:
                    continue
                label_names = ds.features["ner_tags"].feature.names
                for row in ds:
                    tags = row["ner_tags"]
                    for tag_id in tags:
                        tag_name = label_names[tag_id]
                        if tag_name.startswith("B-"):
                            etype = tag_name[2:]
                            type_counts[etype] += 1

    return dict(type_counts)


def scan_multiconer():
    """Scan MultiCoNER v2 CoNLL files for type frequencies."""
    base = RAW_DIR / "multiconer_v2"
    type_counts = defaultdict(int)

    for lang_dir in sorted(base.iterdir()):
        if not lang_dir.is_dir():
            continue
        for conll_file in lang_dir.glob("*_train.conll"):
            with open(conll_file) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) >= 4:
                        tag = parts[-1]
                        if tag.startswith("B-"):
                            type_counts[tag[2:]] += 1

    return dict(type_counts)


def scan_klue():
    """Scan KLUE NER parquet for type frequencies."""
    from datasets import load_dataset

    type_counts = defaultdict(int)
    train_file = RAW_DIR / "klue" / "ner" / "train-00000-of-00001.parquet"
    ds = load_dataset("parquet", data_files=str(train_file), split="train")
    label_names = ds.features["ner_tags"].feature.names

    for row in ds:
        for tag_id in row["ner_tags"]:
            tag_name = label_names[tag_id]
            if tag_name.startswith("B-"):
                type_counts[tag_name[2:]] += 1

    return dict(type_counts)


def scan_stockmark():
    """Scan stockmark NER JSON for type frequencies."""
    type_counts = defaultdict(int)
    with open(RAW_DIR / "stockmark-ner-ja" / "ner.json") as f:
        data = json.load(f)
    for entry in data:
        for ent in entry.get("entities", []):
            type_counts[ent["type"]] += 1
    return dict(type_counts)


def scan_chinese_ner_sft():
    """Scan chinese_ner_sft JSONL files for type frequencies."""
    type_counts = defaultdict(int)
    data_dir = RAW_DIR / "chinese_ner_sft" / "data"
    for jsonl_file in sorted(data_dir.glob("*.jsonl")):
        with open(jsonl_file) as f:
            for line in f:
                entry = json.loads(line)
                for ent in entry.get("entities", []):
                    label = ent.get("entity_label", "")
                    if label:
                        type_counts[label] += 1
    return dict(type_counts)


def scan_finerweb():
    """Scan fiNERweb parquet files for type frequencies."""
    import pyarrow.parquet as pq

    type_counts = defaultdict(int)
    data_dir = RAW_DIR / "fiNERweb" / "data"
    for pf in sorted(data_dir.glob("*.parquet")):
        table = pq.read_table(pf, columns=["char_spans"])
        spans_col = table.column("char_spans")
        for row_spans in spans_col:
            for span in row_spans.as_py():
                label = span.get("label", "")
                if label:
                    type_counts[label] += 1
    return dict(type_counts)


def main():
    print("=" * 60)
    print("Pre-Task 4: Dataset Verification")
    print("=" * 60)

    # Step 1: B2NERD count verification
    print("\n--- B2NERD Count Verification ---")
    b2nerd_results, b2nerd_type_counts = scan_b2nerd_curated()
    print(f"EN train+dev: {b2nerd_results['en_train_dev']}")
    print(f"ZH train+dev: {b2nerd_results['zh_train_dev']}")
    print(f"EN_other train+dev: {b2nerd_results['en_other_train_dev']}")
    total_train_dev = (b2nerd_results['en_train_dev'] +
                       b2nerd_results['zh_train_dev'] +
                       b2nerd_results['en_other_train_dev'])
    print(f"Total train+dev: {total_train_dev}")
    print(f"EN test: {b2nerd_results['en_test']}")
    print(f"ZH test: {b2nerd_results['zh_test']}")
    print(f"EN_other test: {b2nerd_results['en_other_test']}")
    total_test = (b2nerd_results['en_test'] +
                  b2nerd_results['zh_test'] +
                  b2nerd_results['en_other_test'])
    print(f"Total test: {total_test}")
    print(f"Unique types (train+dev): {len(b2nerd_results['types_train_dev'])}")
    print(f"Unique types (test): {len(b2nerd_results['types_test'])}")

    # Step 2: pos field verification
    print("\n--- B2NERD pos Field Verification (exhaustive, per-dataset) ---")
    pos_checks = verify_pos_field()
    for lang, info in pos_checks.items():
        if not info.get("exists", True):
            print(f"  {lang}: directory not found")
            continue
        status = "PASS" if info.get("matches_expectation") else "FAIL"
        print(f"  {lang}: {info['datasets_with_pos']}/{info['total_datasets']} datasets have pos "
              f"({info['entities_with_pos']:,} entities with, {info['entities_without_pos']:,} without) "
              f"[expected={info['expected_pos']}] -> {status}")
        if info.get("datasets_without_pos_names"):
            print(f"    Datasets WITHOUT pos: {info['datasets_without_pos_names']}")

    # Step 3: pos end semantics
    print("\n--- B2NERD pos End Semantics (exhaustive) ---")
    pos_semantics = verify_pos_semantics()
    print(f"  Datasets checked: {pos_semantics['datasets_checked']}")
    print(f"  Exclusive matches: {pos_semantics['exclusive_matches']:,}")
    print(f"  Inclusive matches: {pos_semantics['inclusive_matches']:,}")
    print(f"  Neither: {pos_semantics['neither']:,}")
    print(f"  Convention: {pos_semantics['convention']}")
    for ds_name, counts in pos_semantics["per_dataset"].items():
        print(f"    {ds_name}: excl={counts['exclusive']:,} incl={counts['inclusive']} neither={counts['neither']}")
    for ex in pos_semantics["examples"][:3]:
        print(f"    ex: name='{ex['name']}' pos={ex['pos']} -> {ex['match']}")

    # Step 4: chinese_ner_sft offset convention
    print("\n--- chinese_ner_sft Offset Convention ---")
    cns_offsets = verify_chinese_ner_sft_offsets()
    for subset, info in sorted(cns_offsets.items()):
        print(f"  {subset}: {info['convention']} "
              f"(excl={info['exclusive']} incl={info['inclusive']} "
              f"neither={info['neither']} null={info['null_offsets']})")

    # Step 5: Type frequency scan across all datasets
    print("\n--- Type Frequency Scan ---")
    all_type_freqs = {}

    print("  Scanning B2NERD curated...")
    all_type_freqs["b2nerd_curated"] = b2nerd_type_counts

    print("  Scanning open-ner-standardized...")
    all_type_freqs["open_ner_standardized"] = scan_open_ner(
        RAW_DIR / "open-ner-standardized"
    )
    print(f"    Found {len(all_type_freqs['open_ner_standardized'])} types")

    print("  Scanning open-ner-core-types...")
    all_type_freqs["open_ner_core_types"] = scan_open_ner(
        RAW_DIR / "open-ner-core-types"
    )
    print(f"    Found {len(all_type_freqs['open_ner_core_types'])} types")

    print("  Scanning multiconer_v2...")
    all_type_freqs["multiconer_v2"] = scan_multiconer()
    print(f"    Found {len(all_type_freqs['multiconer_v2'])} types")

    print("  Scanning klue_ner...")
    all_type_freqs["klue_ner"] = scan_klue()
    print(f"    Found {len(all_type_freqs['klue_ner'])} types")

    print("  Scanning stockmark_ner_ja...")
    all_type_freqs["stockmark_ner_ja"] = scan_stockmark()
    print(f"    Found {len(all_type_freqs['stockmark_ner_ja'])} types")

    print("  Scanning chinese_ner_sft...")
    all_type_freqs["chinese_ner_sft"] = scan_chinese_ner_sft()
    print(f"    Found {len(all_type_freqs['chinese_ner_sft'])} types")

    print("  Scanning fiNERweb...")
    all_type_freqs["finerweb"] = scan_finerweb()
    print(f"    Found {len(all_type_freqs['finerweb'])} types")

    # Save type frequencies
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "type_frequency_scan.json", "w") as f:
        json.dump(all_type_freqs, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved type frequencies to {OUT_DIR / 'type_frequency_scan.json'}")

    # Save verification report
    report = {
        "b2nerd_counts": {
            "en_train_dev": b2nerd_results["en_train_dev"],
            "zh_train_dev": b2nerd_results["zh_train_dev"],
            "en_other_train_dev": b2nerd_results["en_other_train_dev"],
            "total_train_dev": total_train_dev,
            "en_test": b2nerd_results["en_test"],
            "zh_test": b2nerd_results["zh_test"],
            "en_other_test": b2nerd_results["en_other_test"],
            "total_test": total_test,
            "unique_types_train_dev": len(b2nerd_results["types_train_dev"]),
            "unique_types_test": len(b2nerd_results["types_test"]),
        },
        "pos_field_checks": pos_checks,
        "pos_semantics": pos_semantics,
        "chinese_ner_sft_offsets": cns_offsets,
    }
    with open(OUT_DIR / "pretask4_verification_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  Saved verification report to {OUT_DIR / 'pretask4_verification_report.json'}")

    print("\n--- Done ---")


if __name__ == "__main__":
    main()
