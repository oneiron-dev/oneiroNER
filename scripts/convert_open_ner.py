"""Convert Open NER (standardized + core-types) parquet BIO to NER JSONL."""

import logging
import random
import sys
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent))
from lib.bio_to_spans import bio_tags_to_spans, detokenize, is_cjk_language, tokens_to_char_spans
from lib.negative_sampler import NegativeSampler
from lib.schema import Entity, NerRecord
from lib.span_validator import validate_span

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR_STD = Path(__file__).parent.parent / "data" / "raw" / "open-ner-standardized"
DATA_DIR_CORE = Path(__file__).parent.parent / "data" / "raw" / "open-ner-core-types"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "processed"

CONFIDENCE = "silver"

LANG_MAP = {
    "eng": "en", "jpn": "ja", "jap": "ja", "cmn": "zh", "kor": "ko",
    "deu": "de", "fra": "fr", "spa": "es", "ita": "it", "por": "pt",
    "nld": "nl", "rus": "ru", "ara": "ar", "hin": "hi", "ben": "bn",
    "tur": "tr", "pol": "pl", "swe": "sv", "nor": "no", "dan": "da",
    "fin": "fi", "cat": "ca", "ukr": "uk", "fas": "fa", "tha": "th",
    "vie": "vi", "ind": "id", "msa": "ms", "heb": "he", "ell": "el",
    "ces": "cs", "ron": "ro", "hun": "hu", "bul": "bg", "hrv": "hr",
    "slk": "sk", "slv": "sl", "lit": "lt", "lav": "lv", "ekk": "et",
}

STD_CAPS = {"en": 30000, "ja": 20000, "zh": 20000, "ko": 15000}
STD_DEFAULT_CAP = 15000
CORE_CAPS = {"en": 15000, "ja": 10000, "zh": 10000, "ko": 7500}
CORE_DEFAULT_CAP = 7500

neg_sampler = NegativeSampler()
rng = random.Random(42)


def map_lang(lang3: str) -> str:
    return LANG_MAP.get(lang3, lang3)


def process_parquet(path: Path, dataset: str, lang3: str, source_prefix: str, source_name: str,
                    global_idx_start: int = 0) -> tuple[list[NerRecord], int]:
    lang2 = map_lang(lang3)
    records = []
    pq_stem = path.stem

    try:
        ds = load_dataset("parquet", data_files=str(path), split="train")
    except Exception as e:
        logger.warning("Failed to load %s: %s", path, e)
        return records, global_idx_start

    label_names = ds.features["ner_tags"].feature.names
    global_idx = global_idx_start

    for row in ds:
        tokens = row["tokens"]
        tags = row["ner_tags"]

        if not tokens:
            global_idx += 1
            continue

        token_spans = bio_tags_to_spans(tokens, tags, tag_map=label_names)

        use_cjk = is_cjk_language(lang3) or is_cjk_language(lang2)
        det_lang = lang3 if use_cjk else lang2
        text, token_offsets = detokenize(tokens, det_lang)

        entities = []
        for span in tokens_to_char_spans(token_spans, token_offsets, text):
            if validate_span(text, span["surface"], span["start"], span["end"]):
                entities.append(Entity(
                    surface=span["surface"],
                    type=span["type"],
                    original_type=span["type"],
                    start=span["start"],
                    end=span["end"],
                ).to_dict())

        if not entities:
            global_idx += 1
            continue

        pos_types = {e["type"] for e in entities}
        neg_types = neg_sampler.sample(pos_types, rng=rng)
        query_types = sorted(pos_types) + neg_types

        rec = NerRecord(
            source=source_name,
            source_id=f"{source_prefix}_{dataset}_{lang3}_{pq_stem}_{global_idx}",
            language=lang2,
            split="train",
            confidence=CONFIDENCE,
            provenance=[source_name],
            text=text,
            query_types=query_types,
            entities=entities,
        )
        rec.validate()
        records.append(rec)
        global_idx += 1

    return records, global_idx


def process_variant(data_dir: Path, source_name: str, source_prefix: str,
                    caps: dict, default_cap: int) -> tuple[dict, list[NerRecord]]:
    stats = {"total_parquets": 0, "total_rows": 0, "written": 0, "entities": 0, "langs": {}}
    lang_records: dict[str, list[NerRecord]] = {}

    if not data_dir.exists():
        logger.warning("Data directory not found: %s", data_dir)
        return stats, []

    for dataset_dir in sorted(data_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name

        for lang_dir in sorted(dataset_dir.iterdir()):
            if not lang_dir.is_dir():
                continue
            lang3 = lang_dir.name
            lang2 = map_lang(lang3)

            parquets = sorted(lang_dir.glob("*.parquet"))
            if not parquets:
                continue

            global_idx = 0
            for pq in parquets:
                stats["total_parquets"] += 1
                recs, global_idx = process_parquet(pq, dataset, lang3, source_prefix, source_name,
                                                   global_idx_start=global_idx)
                if recs:
                    lang_records.setdefault(lang2, []).extend(recs)
                    stats["total_rows"] += len(recs)

            logger.info("  %s/%s (%s): processed %d parquets", dataset, lang3, lang2, len(parquets))

    all_records = []
    for lang2, recs in sorted(lang_records.items()):
        cap = caps.get(lang2, default_cap)
        if len(recs) > cap:
            recs = rng.sample(recs, cap)
        all_records.extend(recs)
        stats["langs"][lang2] = len(recs)
        stats["written"] += len(recs)
        stats["entities"] += sum(len(r.entities) for r in recs)

    return stats, all_records


def convert(output_dir: str | None = None) -> dict:
    out = Path(output_dir) if output_dir else OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    combined_stats = {}

    logger.info("=== Processing open-ner-standardized ===")
    std_stats, std_records = process_variant(
        DATA_DIR_STD, "open_ner_standardized", "open_ner_std",
        STD_CAPS, STD_DEFAULT_CAP,
    )
    outfile = out / "open_ner_standardized.jsonl"
    with open(outfile, "w") as fout:
        for rec in std_records:
            fout.write(rec.to_jsonl() + "\n")
    logger.info("open_ner_standardized: %d records, %d entities, %d languages",
                std_stats["written"], std_stats["entities"], len(std_stats["langs"]))
    combined_stats["standardized"] = std_stats

    logger.info("=== Processing open-ner-core-types ===")
    core_stats, core_records = process_variant(
        DATA_DIR_CORE, "open_ner_core_types", "open_ner_core",
        CORE_CAPS, CORE_DEFAULT_CAP,
    )
    outfile = out / "open_ner_core_types.jsonl"
    with open(outfile, "w") as fout:
        for rec in core_records:
            fout.write(rec.to_jsonl() + "\n")
    logger.info("open_ner_core_types: %d records, %d entities, %d languages",
                core_stats["written"], core_stats["entities"], len(core_stats["langs"]))
    combined_stats["core_types"] = core_stats

    logger.info("open_ner total: %s", combined_stats)
    return combined_stats


if __name__ == "__main__":
    convert()
