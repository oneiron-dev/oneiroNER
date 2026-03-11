#!/usr/bin/env python3
"""Orchestrator: run all converters → dedup → split → View B → chat format."""

import json
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from lib.schema import NerRecord, record_from_jsonl
from lib.negative_sampler import NegativeSampler
from lib.dedup import dedup_files
from lib.splitter import stratified_split, apply_zero_shot_stripping, load_holdout_types

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CONFIGS_DIR = PROJECT_ROOT / "configs"

TYPE_MAPPING_PATH = CONFIGS_DIR / "type_mapping_train.json"
HOLDOUT_PATH = CONFIGS_DIR / "zero_shot_holdout_types.json"

VIEW_B_RATIO = 1.0
SEED = 42

SILVER_SOURCES = {"task8_", "task9_silver_"}

CONVERTER_OUTPUTS = [
    "stockmark_ner_ja.jsonl",
    "chinese_ner_sft.jsonl",
    "klue_ner.jsonl",
    "b2nerd_en.jsonl",
    "b2nerd_zh.jsonl",
    "b2nerd_en_other.jsonl",
    "kor_ner.jsonl",
    "french_ner.jsonl",
    "germeval_14.jsonl",
]

FINERWEB_GLOB = "finerweb_*.jsonl"
MULTICONER_GLOB = "multiconer_v2_*.jsonl"
OPEN_NER_FILES = [
    "open_ner_standardized.jsonl",
    "open_ner_core_types.jsonl",
]
SILVER_SYNTHETIC_FILES = [
    "silver_synthetic_ml.jsonl",
]


def collect_input_files() -> list[Path]:
    files = []
    for name in CONVERTER_OUTPUTS + OPEN_NER_FILES + SILVER_SYNTHETIC_FILES:
        p = PROCESSED_DIR / name
        if p.exists():
            files.append(p)
        else:
            logger.warning("Expected file not found: %s", p)

    skipped_empty = []
    for p in sorted(PROCESSED_DIR.glob(FINERWEB_GLOB)):
        if p.stat().st_size == 0:
            skipped_empty.append(p.name)
            continue
        files.append(p)
    for p in sorted(PROCESSED_DIR.glob(MULTICONER_GLOB)):
        if p.stat().st_size == 0:
            skipped_empty.append(p.name)
            continue
        files.append(p)

    for p in sorted(PROCESSED_DIR.glob("task8_*.jsonl")):
        if p.stat().st_size == 0:
            skipped_empty.append(p.name)
            continue
        files.append(p)
    for p in sorted(PROCESSED_DIR.glob("task9_silver_*.jsonl")):
        if p.stat().st_size == 0:
            skipped_empty.append(p.name)
            continue
        files.append(p)

    if skipped_empty:
        logger.info("Skipped %d empty shard files: %s", len(skipped_empty), skipped_empty)
    logger.info("Collected %d input files", len(files))
    return files


def load_type_mapping() -> dict[str, str]:
    with open(TYPE_MAPPING_PATH) as f:
        return json.load(f)


def generate_view_b(
    records: list[NerRecord],
    type_mapping: dict[str, str],
    sampler_canonical: NegativeSampler,
    rng: random.Random,
    ratio: float = VIEW_B_RATIO,
) -> list[NerRecord]:
    view_b = []
    for rec in records:
        if any(rec.source.startswith(p) for p in SILVER_SOURCES):
            continue
        if not isinstance(rec, NerRecord):
            continue
        mapped_entities = []
        for ent in rec.entities:
            original_type = ent.get("original_type", ent["type"])
            if ent["type"] in type_mapping:
                canonical = type_mapping[ent["type"]]
                mapped_entities.append({
                    "surface": ent["surface"],
                    "type": canonical,
                    "original_type": original_type,
                    "start": ent["start"],
                    "end": ent["end"],
                })

        if not mapped_entities:
            continue

        canonical_positive = set(e["type"] for e in mapped_entities)
        negatives = sampler_canonical.sample(canonical_positive, rng=rng)
        query_types = sorted(canonical_positive | set(negatives))

        b_rec = NerRecord(
            source=rec.source + "_canonical",
            source_id=rec.source_id + "_canonical",
            language=rec.language,
            split=rec.split,
            confidence=rec.confidence,
            provenance=rec.provenance,
            text=rec.text,
            query_types=query_types,
            entities=mapped_entities,
        )
        try:
            b_rec.validate()
            view_b.append(b_rec)
        except AssertionError as e:
            logger.debug("View B validation failed: %s", e)

    if ratio < 1.0 and len(view_b) > int(len(records) * ratio):
        rng.shuffle(view_b)
        view_b = view_b[:int(len(records) * ratio)]

    logger.info("Generated %d View B records (ratio=%.1f)", len(view_b), ratio)
    return view_b


def build_canonical_sampler(type_mapping: dict[str, str]) -> NegativeSampler:
    canonical_types = set(type_mapping.values())
    canonical_counts = {t: 1000 for t in canonical_types}
    return NegativeSampler(type_counts=canonical_counts)


def schema_to_chat(example: dict) -> list[dict]:
    text = example.get("text", "")
    if not text and "turns" in example:
        text = "\n".join(f'{t["speaker"]}: {t["text"]}' for t in example["turns"])

    system_msg = (
        "Extract entities of the requested type from the given text.\n"
        "Return a JSON list of entity surfaces. If none exist, return []."
    )

    chat_examples = []
    for query_type in example["query_types"]:
        matching = [e["surface"] for e in example["entities"] if e["type"] == query_type]
        chat_examples.append({
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f'Text: "{text}"\nExtract all entities of type: {query_type}'},
                {"role": "assistant", "content": json.dumps(matching if matching else [], ensure_ascii=False)},
            ]
        })
    return chat_examples


def write_records(records: list, path: Path):
    with open(path, "w") as f:
        for rec in records:
            f.write(rec.to_jsonl() + "\n")
    logger.info("Wrote %d records to %s", len(records), path)


def write_chat(records: list, path: Path):
    count = 0
    with open(path, "w") as f:
        for rec in records:
            d = json.loads(rec.to_jsonl())
            for chat_ex in schema_to_chat(d):
                f.write(json.dumps(chat_ex, ensure_ascii=False) + "\n")
                count += 1
    logger.info("Wrote %d chat examples to %s", count, path)


def run_converters():
    logger.info("Running individual converters...")
    import importlib

    converter_modules = [
        "convert_stockmark",
        "convert_finerweb",
        "convert_chinese_ner",
        "convert_b2nerd",
        "convert_open_ner",
        "convert_multiconer",
        "convert_klue",
        "convert_silver_synthetic",
    ]

    all_stats = {}
    for mod_name in converter_modules:
        logger.info("Running %s...", mod_name)
        try:
            mod = importlib.import_module(mod_name)
            stats = mod.convert(str(PROCESSED_DIR))
            all_stats[mod_name] = stats
            logger.info("%s: %s", mod_name, stats)
        except Exception as e:
            logger.error("Failed to run %s: %s", mod_name, e)
            raise

    return all_stats


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-converters", action="store_true", help="Skip running individual converters")
    args = parser.parse_args()

    rng = random.Random(SEED)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Run all converters (unless pre-computed)
    if args.skip_converters:
        logger.info("Skipping individual converters (using existing outputs)")
        converter_stats = {"skipped": True, "existing_outputs": {}}
        for p in sorted(PROCESSED_DIR.glob("*.jsonl")):
            if p.name in ("train.jsonl", "val.jsonl", "zero_shot_eval.jsonl",
                          "train_chat.jsonl", "val_chat.jsonl",
                          "deduped_all.jsonl", "b2nerd_ambiguous.jsonl"):
                continue
            with open(p) as f:
                count = sum(1 for _ in f)
            converter_stats["existing_outputs"][p.name] = count
        logger.info("Found %d existing converter outputs", len(converter_stats["existing_outputs"]))
    else:
        converter_stats = run_converters()

    # Step 2: Collect all per-dataset JSONL files
    input_files = collect_input_files()

    # Step 3: Dedup
    sampler = NegativeSampler()
    dedup_output = PROCESSED_DIR / "deduped_all.jsonl"
    dedup_stats = dedup_files(input_files, dedup_output, sampler, seed=SEED)

    with open(PROCESSED_DIR / "dedup_stats.json", "w") as f:
        json.dump(dedup_stats, f, indent=2)

    # Step 4: Load deduped records
    logger.info("Loading deduped records...")
    records = []
    with open(dedup_output) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(record_from_jsonl(line))
    logger.info("Loaded %d deduped records", len(records))

    # Step 5: Split
    train_records, val_records = stratified_split(records, val_ratio=0.05, seed=SEED)

    for rec in train_records:
        rec.split = "train"
    for rec in val_records:
        rec.split = "val"

    # Step 6: Zero-shot stripping (train split only)
    holdout_types = load_holdout_types()
    train_records, zero_shot_records = apply_zero_shot_stripping(
        train_records, sampler, holdout_types, seed=SEED
    )

    # Step 7: View B generation (within each split)
    type_mapping = load_type_mapping()
    sampler_canonical = build_canonical_sampler(type_mapping)

    def _is_silver(rec) -> bool:
        return rec.confidence == "silver" and getattr(rec, "format", None) == "conversation"

    train_non_silver = [r for r in train_records if not _is_silver(r)]
    val_non_silver = [r for r in val_records if not _is_silver(r)]

    train_view_b = generate_view_b(train_non_silver, type_mapping, sampler_canonical, rng, VIEW_B_RATIO)
    val_view_b = generate_view_b(val_non_silver, type_mapping, sampler_canonical, rng, VIEW_B_RATIO)

    # Step 8: Combine View A + View B
    train_all = train_records + train_view_b
    val_all = val_records + val_view_b

    rng.shuffle(train_all)
    rng.shuffle(val_all)

    # Step 9: Write outputs
    write_records(train_all, PROCESSED_DIR / "train.jsonl")
    write_records(val_all, PROCESSED_DIR / "val.jsonl")
    write_records(zero_shot_records, PROCESSED_DIR / "zero_shot_eval.jsonl")

    # Step 10: Chat format
    write_chat(train_all, PROCESSED_DIR / "train_chat.jsonl")
    write_chat(val_all, PROCESSED_DIR / "val_chat.jsonl")

    # Step 11: Training mix stats
    training_mix = {
        "view_b_ratio": VIEW_B_RATIO,
        "seed": SEED,
        "converter_stats": converter_stats,
        "dedup_stats": dedup_stats,
        "split_stats": {
            "train_view_a": len(train_records),
            "train_view_b": len(train_view_b),
            "train_total": len(train_all),
            "val_view_a": len(val_records),
            "val_view_b": len(val_view_b),
            "val_total": len(val_all),
            "zero_shot_eval": len(zero_shot_records),
        },
    }
    with open(CONFIGS_DIR / "training_mix.json", "w") as f:
        json.dump(training_mix, f, indent=2)
    logger.info("Training mix saved to %s", CONFIGS_DIR / "training_mix.json")

    # Cleanup temp dedup file
    dedup_output.unlink(missing_ok=True)

    logger.info("=" * 60)
    logger.info("CONVERSION PIPELINE COMPLETE")
    logger.info("Train: %d records (%d View A + %d View B)", len(train_all), len(train_records), len(train_view_b))
    logger.info("Val: %d records (%d View A + %d View B)", len(val_all), len(val_records), len(val_view_b))
    logger.info("Zero-shot eval: %d records", len(zero_shot_records))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
