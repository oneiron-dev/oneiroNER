"""Convert GermanEval 2014 (German NER, TSV BIO) to NER JSONL."""

import csv
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from lib.bio_to_spans import bio_tags_to_spans, detokenize, tokens_to_char_spans
from lib.negative_sampler import NegativeSampler
from lib.schema import Entity, NerRecord
from lib.span_validator import validate_span

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data" / "raw" / "germeval_14"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "processed"

SOURCE = "germeval_14"
LANGUAGE = "de"
CONFIDENCE = "gold"

TYPE_MAP = {
    "PER": "PERSON", "PERderiv": "PERSON", "PERpart": "PERSON",
    "LOC": "PLACE", "LOCderiv": "PLACE", "LOCpart": "PLACE",
    "ORG": "ORG", "ORGderiv": "ORG", "ORGpart": "ORG",
}

neg_sampler = NegativeSampler()
rng = random.Random(42)


def parse_germeval_tsv(path: Path) -> list[dict]:
    sentences = []
    tokens = []
    tags = []

    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            if not row or (len(row) == 1 and not row[0].strip()):
                if tokens:
                    sentences.append({"tokens": tokens, "tags": tags})
                    tokens = []
                    tags = []
            elif row[0].startswith("#"):
                continue
            elif len(row) >= 3:
                tokens.append(row[1])
                tags.append(row[2])

    if tokens:
        sentences.append({"tokens": tokens, "tags": tags})

    return sentences


def remap_tags(tags: list[str]) -> list[str]:
    remapped = []
    for tag in tags:
        if tag == "O":
            remapped.append("O")
        elif tag.startswith("B-") or tag.startswith("I-"):
            prefix = tag[:2]
            etype = tag[2:]
            if etype in TYPE_MAP:
                remapped.append(f"{prefix}{etype}")
            else:
                remapped.append("O")
        else:
            remapped.append("O")
    return remapped


def convert(output_dir: str | None = None) -> dict:
    out = Path(output_dir) if output_dir else OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    outfile = out / f"{SOURCE}.jsonl"

    stats = {"total": 0, "written": 0, "skipped": 0, "entities": 0, "entities_skipped": 0}
    records = []

    for split_file, split_name in [("train.tsv", "train"), ("dev.tsv", "dev"), ("test.tsv", "test")]:
        path = DATA_DIR / split_file
        if not path.exists():
            logger.warning("Missing: %s", path)
            continue

        sentences = parse_germeval_tsv(path)
        logger.info("Processing germeval_14 %s: %d sentences", split_name, len(sentences))

        for idx, sent in enumerate(sentences):
            stats["total"] += 1
            cleaned_tags = remap_tags(sent["tags"])
            token_spans = bio_tags_to_spans(sent["tokens"], cleaned_tags)
            text, token_offsets = detokenize(sent["tokens"], LANGUAGE)

            entities = []
            for span in tokens_to_char_spans(token_spans, token_offsets, text):
                raw_type = span["type"]
                mapped_type = TYPE_MAP.get(raw_type, raw_type)
                if validate_span(text, span["surface"], span["start"], span["end"]):
                    entities.append(Entity(
                        surface=span["surface"],
                        type=mapped_type,
                        original_type=raw_type,
                        start=span["start"],
                        end=span["end"],
                    ).to_dict())
                    stats["entities"] += 1
                else:
                    stats["entities_skipped"] += 1
                    logger.debug("Span mismatch: '%s' vs text[%d:%d]='%s'",
                                 span["surface"], span["start"], span["end"],
                                 text[span["start"]:span["end"]])

            if not entities:
                stats["skipped"] += 1
                continue

            pos_types = {e["type"] for e in entities}
            neg_types = neg_sampler.sample(pos_types, rng=rng)
            query_types = sorted(pos_types) + neg_types

            rec = NerRecord(
                source=SOURCE,
                source_id=f"germeval_14_{split_name}_{idx}",
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

    with open(outfile, "w") as fout:
        for rec in records:
            fout.write(rec.to_jsonl() + "\n")
            stats["written"] += 1

    logger.info("germeval_14: %s", stats)
    return stats


if __name__ == "__main__":
    convert()
