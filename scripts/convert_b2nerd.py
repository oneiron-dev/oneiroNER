"""Convert B2NERD curated data to NER JSONL (en, zh, en_other partitions)."""

import json
import logging
import random
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from lib.schema import NerRecord, Entity
from lib.span_validator import validate_span, validate_and_fix
from lib.span_computer import compute_spans_batch
from lib.negative_sampler import NegativeSampler

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ZIP_PATH = Path(__file__).parent.parent / "data" / "raw" / "B2NERD" / "B2NERD_data.zip"
EXTRACT_DIR = Path("/tmp/b2nerd_extract")
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "processed"
SOURCE = "b2nerd"
CONFIDENCE = "synthetic-gold"
TOTAL_CAP = 52_000
SPLITS = ("train", "dev")

neg_sampler = NegativeSampler()
rng = random.Random(42)


def extract_zip():
    if (EXTRACT_DIR / "B2NERD").exists():
        logger.info("Already extracted to %s", EXTRACT_DIR)
        return
    logger.info("Extracting %s to %s", ZIP_PATH, EXTRACT_DIR)
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(EXTRACT_DIR)


def load_json(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def iter_datasets(partition_dir: Path):
    if not partition_dir.exists():
        return
    for ds_dir in sorted(partition_dir.iterdir()):
        if not ds_dir.is_dir():
            continue
        for split in SPLITS:
            split_file = ds_dir / f"{split}.json"
            if split_file.exists():
                yield ds_dir.name, split, load_json(split_file)


def make_record(text, entities, partition, dataset_name, split, idx, language):
    pos_types = {e["type"] for e in entities}
    neg_types = neg_sampler.sample(pos_types, rng=rng)
    query_types = sorted(pos_types) + neg_types

    rec = NerRecord(
        source=SOURCE,
        source_id=f"b2nerd_{partition}_{dataset_name}_{split}_{idx}",
        language=language,
        split="train",
        confidence=CONFIDENCE,
        provenance=[f"b2nerd_{partition}"],
        text=text,
        query_types=query_types,
        entities=entities,
    )
    rec.validate()
    return rec


def resolve_pilener_entity(text: str, name: str) -> tuple[int, int] | None:
    pattern = r'\b' + re.escape(name) + r'\b'
    m = re.search(pattern, text)
    if m:
        return m.start(), m.end()
    return None


def process_en(base_dir: Path, ambiguous_log: list) -> list[NerRecord]:
    records = []
    stats = {"total": 0, "written": 0, "skipped": 0, "entities": 0, "entities_skipped": 0}

    for dataset_name, split, entries in iter_datasets(base_dir / "NER_en"):
        is_pilener = dataset_name == "PileNER"
        logger.info("  NER_en/%s/%s: %d entries%s", dataset_name, split, len(entries),
                     " (word-boundary matching)" if is_pilener else "")
        for idx, entry in enumerate(entries):
            stats["total"] += 1
            text = entry["sentence"]
            if not text:
                stats["skipped"] += 1
                continue

            valid_entities = []

            if is_pilener:
                for ent in entry.get("entities", []):
                    span = resolve_pilener_entity(text, ent["name"])
                    if span:
                        valid_entities.append(Entity(
                            surface=ent["name"], type=ent["type"], original_type=ent["type"],
                            start=span[0], end=span[1],
                        ).to_dict())
                        stats["entities"] += 1
                    else:
                        stats["entities_skipped"] += 1
            else:
                for ent in entry.get("entities", []):
                    name = ent["name"]
                    pos = ent.get("pos")
                    if pos is None:
                        stats["entities_skipped"] += 1
                        continue
                    start, end = pos[0], pos[1]
                    if validate_span(text, name, start, end):
                        valid_entities.append(Entity(
                            surface=name, type=ent["type"], original_type=ent["type"],
                            start=start, end=end,
                        ).to_dict())
                        stats["entities"] += 1
                    else:
                        try:
                            s, e, _ = validate_and_fix(text, name, start, end)
                            valid_entities.append(Entity(
                                surface=name, type=ent["type"], original_type=ent["type"],
                                start=s, end=e,
                            ).to_dict())
                            stats["entities"] += 1
                        except ValueError:
                            stats["entities_skipped"] += 1

            if not valid_entities:
                stats["skipped"] += 1
                continue

            try:
                records.append(make_record(text, valid_entities, "en", dataset_name, split, idx, "en"))
                stats["written"] += 1
            except (AssertionError, Exception) as exc:
                stats["skipped"] += 1
                logger.debug("Record failed validation: %s", exc)

    logger.info("NER_en: %s", stats)
    return records


def process_zh(base_dir: Path, ambiguous_log: list) -> list[NerRecord]:
    records = []
    stats = {"total": 0, "written": 0, "skipped": 0, "entities": 0, "entities_skipped": 0}

    for dataset_name, split, entries in iter_datasets(base_dir / "NER_zh"):
        logger.info("  NER_zh/%s/%s: %d entries", dataset_name, split, len(entries))
        for idx, entry in enumerate(entries):
            stats["total"] += 1
            text = entry["sentence"]
            if not text:
                stats["skipped"] += 1
                continue

            raw_ents = [{"name": e["name"], "type": e["type"]} for e in entry.get("entities", [])]
            resolved = compute_spans_batch(text, raw_ents, use_word_boundary=False, ambiguous_log=ambiguous_log)

            valid_entities = []
            for e in resolved:
                if validate_span(text, e["name"], e["start"], e["end"]):
                    valid_entities.append(Entity(
                        surface=e["name"], type=e["type"], original_type=e["type"],
                        start=e["start"], end=e["end"],
                    ).to_dict())
                    stats["entities"] += 1
                else:
                    stats["entities_skipped"] += 1

            if not valid_entities:
                stats["skipped"] += 1
                continue

            try:
                records.append(make_record(text, valid_entities, "zh", dataset_name, split, idx, "zh"))
                stats["written"] += 1
            except (AssertionError, Exception) as exc:
                stats["skipped"] += 1
                logger.debug("Record failed validation: %s", exc)

    logger.info("NER_zh: %s", stats)
    return records


def parse_language(dirname: str) -> str:
    m = re.match(r"multiconer22_([a-z]{2})_sample_5000", dirname)
    if m:
        return m.group(1)
    return "en"


def process_en_other(base_dir: Path, ambiguous_log: list) -> list[NerRecord]:
    records = []
    stats = {"total": 0, "written": 0, "skipped": 0, "entities": 0, "entities_skipped": 0}

    for dataset_name, split, entries in iter_datasets(base_dir / "NER_en_other"):
        lang = parse_language(dataset_name)
        logger.info("  NER_en_other/%s/%s: %d entries (lang=%s)", dataset_name, split, len(entries), lang)
        for idx, entry in enumerate(entries):
            stats["total"] += 1
            text = entry["sentence"]
            if not text:
                stats["skipped"] += 1
                continue

            raw_ents = [{"name": e["name"], "type": e["type"]} for e in entry.get("entities", [])]
            resolved = compute_spans_batch(text, raw_ents, use_word_boundary=False, ambiguous_log=ambiguous_log)

            valid_entities = []
            for e in resolved:
                if validate_span(text, e["name"], e["start"], e["end"]):
                    valid_entities.append(Entity(
                        surface=e["name"], type=e["type"], original_type=e["type"],
                        start=e["start"], end=e["end"],
                    ).to_dict())
                    stats["entities"] += 1
                else:
                    stats["entities_skipped"] += 1

            if not valid_entities:
                stats["skipped"] += 1
                continue

            try:
                records.append(make_record(text, valid_entities, "en_other", dataset_name, split, idx, lang))
                stats["written"] += 1
            except (AssertionError, Exception) as exc:
                stats["skipped"] += 1
                logger.debug("Record failed validation: %s", exc)

    logger.info("NER_en_other: %s", stats)
    return records


def convert(output_dir: str | None = None) -> dict:
    out = Path(output_dir) if output_dir else OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    extract_zip()
    base_dir = EXTRACT_DIR / "B2NERD"
    ambiguous_log: list[dict] = []

    en_records = process_en(base_dir, ambiguous_log)
    zh_records = process_zh(base_dir, ambiguous_log)
    other_records = process_en_other(base_dir, ambiguous_log)

    all_records = en_records + zh_records + other_records
    total = len(all_records)
    logger.info("Total records before cap: %d", total)

    if total > TOTAL_CAP:
        cap_rng = random.Random(42)
        all_records = cap_rng.sample(all_records, TOTAL_CAP)
        logger.info("Capped to %d records", TOTAL_CAP)
        en_records = [r for r in all_records if "b2nerd_en_" in r.source_id and "b2nerd_en_other_" not in r.source_id]
        zh_records = [r for r in all_records if "b2nerd_zh_" in r.source_id]
        other_records = [r for r in all_records if "b2nerd_en_other_" in r.source_id]

    for fname, recs in [
        ("b2nerd_en.jsonl", en_records),
        ("b2nerd_zh.jsonl", zh_records),
        ("b2nerd_en_other.jsonl", other_records),
    ]:
        with open(out / fname, "w") as f:
            for r in recs:
                f.write(r.to_jsonl() + "\n")
        logger.info("Wrote %d records to %s", len(recs), fname)

    if ambiguous_log:
        with open(out / "b2nerd_ambiguous.jsonl", "w") as f:
            for entry in ambiguous_log:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info("Wrote %d ambiguous entries", len(ambiguous_log))

    stats = {
        "en": len(en_records),
        "zh": len(zh_records),
        "en_other": len(other_records),
        "total": len(en_records) + len(zh_records) + len(other_records),
        "ambiguous": len(ambiguous_log),
        "capped": total > TOTAL_CAP,
    }
    logger.info("Final stats: %s", stats)
    return stats


if __name__ == "__main__":
    convert()
