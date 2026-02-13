"""Convert KLUE NER (Korean char-level BIO parquet) to NER JSONL."""

import logging
import random
import sys
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent))
from lib.bio_to_spans import bio_tags_to_spans
from lib.negative_sampler import NegativeSampler
from lib.schema import Entity, NerRecord
from lib.span_validator import validate_span

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data" / "raw" / "klue" / "ner"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "processed"

SOURCE = "klue_ner"
LANGUAGE = "ko"
CONFIDENCE = "gold"

neg_sampler = NegativeSampler()
rng = random.Random(42)


def process_split(path: Path, split_name: str, stats: dict, records: list):
    ds = load_dataset("parquet", data_files=str(path), split="train")
    label_names = ds.features["ner_tags"].feature.names

    for idx, row in enumerate(ds):
        stats["total"] += 1
        tokens = row["tokens"]
        tags = row["ner_tags"]

        token_spans = bio_tags_to_spans(tokens, tags, tag_map=label_names)

        text = "".join(tokens)
        token_offsets = []
        pos = 0
        for t in tokens:
            token_offsets.append((pos, pos + len(t)))
            pos += len(t)

        entities = []
        for span in token_spans:
            char_start = token_offsets[span["token_start"]][0]
            char_end = token_offsets[span["token_end"] - 1][1]
            surface = text[char_start:char_end]

            if validate_span(text, surface, char_start, char_end):
                entities.append(Entity(
                    surface=surface,
                    type=span["type"],
                    original_type=span["type"],
                    start=char_start,
                    end=char_end,
                ).to_dict())
                stats["entities"] += 1
            else:
                stats["entities_skipped"] += 1
                logger.debug("Span mismatch: '%s' vs text[%d:%d]='%s'",
                             surface, char_start, char_end, text[char_start:char_end])

        if not entities:
            stats["skipped"] += 1
            continue

        pos_types = {e["type"] for e in entities}
        neg_types = neg_sampler.sample(pos_types, rng=rng)
        query_types = sorted(pos_types) + neg_types

        rec = NerRecord(
            source=SOURCE,
            source_id=f"klue_ner_{split_name}_{idx}",
            language=LANGUAGE,
            split="train",
            confidence=CONFIDENCE,
            provenance=[SOURCE],
            text=text,
            query_types=query_types,
            entities=entities,
        )
        rec.validate()
        records.append(rec)


def convert(output_dir: str | None = None) -> dict:
    out = Path(output_dir) if output_dir else OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    outfile = out / f"{SOURCE}.jsonl"

    stats = {"total": 0, "written": 0, "skipped": 0, "entities": 0, "entities_skipped": 0}
    records = []

    for parquet in sorted(DATA_DIR.glob("*.parquet")):
        split_name = parquet.name.split("-")[0]  # train or validation
        logger.info("Processing KLUE %s: %s", split_name, parquet.name)
        process_split(parquet, split_name, stats, records)

    with open(outfile, "w") as fout:
        for rec in records:
            fout.write(rec.to_jsonl() + "\n")
            stats["written"] += 1

    logger.info("klue_ner: %s", stats)
    return stats


if __name__ == "__main__":
    convert()
