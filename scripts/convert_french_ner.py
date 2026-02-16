"""Convert CATIE-AQ/frenchNER_3entities (French IO-tagged NER) to NER JSONL."""

import logging
import random
import sys
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent))
from lib.bio_to_spans import detokenize
from lib.negative_sampler import NegativeSampler
from lib.schema import Entity, NerRecord
from lib.span_validator import validate_span

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "processed"

SOURCE = "french_ner"
LANGUAGE = "fr"
CONFIDENCE = "gold"

TAG_MAP = {0: "O", 1: "PER", 2: "ORG", 3: "LOC"}
TYPE_MAP = {"PER": "PERSON", "ORG": "ORG", "LOC": "PLACE"}

neg_sampler = NegativeSampler()
rng = random.Random(42)


def io_tags_to_spans(tokens: list[str], tags: list[int]) -> list[dict]:
    spans = []
    current_type = None
    current_start = None

    for i, (token, tag) in enumerate(zip(tokens, tags)):
        tag_str = TAG_MAP.get(tag, "O")
        if tag_str == "O":
            if current_type is not None:
                spans.append({"type": current_type, "token_start": current_start, "token_end": i})
                current_type = None
        elif tag_str != current_type:
            if current_type is not None:
                spans.append({"type": current_type, "token_start": current_start, "token_end": i})
            current_type = tag_str
            current_start = i
        # else: same type continues

    if current_type is not None:
        spans.append({"type": current_type, "token_start": current_start, "token_end": len(tokens)})

    return spans


def convert(output_dir: str | None = None) -> dict:
    out = Path(output_dir) if output_dir else OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    outfile = out / f"{SOURCE}.jsonl"

    stats = {"total": 0, "written": 0, "skipped": 0, "entities": 0, "entities_skipped": 0}
    records = []

    for split_name in ["train", "validation", "test"]:
        ds = load_dataset("CATIE-AQ/frenchNER_3entities", split=split_name)
        logger.info("Processing frenchNER %s: %d rows", split_name, len(ds))

        for idx, row in enumerate(ds):
            stats["total"] += 1
            tokens = row["tokens"]
            tags = row["ner_tags"]

            token_spans = io_tags_to_spans(tokens, tags)
            text, token_offsets = detokenize(tokens, LANGUAGE)

            entities = []
            for span in token_spans:
                char_start = token_offsets[span["token_start"]][0]
                char_end = token_offsets[span["token_end"] - 1][1]
                surface = text[char_start:char_end]
                mapped_type = TYPE_MAP.get(span["type"], span["type"])

                if validate_span(text, surface, char_start, char_end):
                    entities.append(Entity(
                        surface=surface,
                        type=mapped_type,
                        original_type=span["type"],
                        start=char_start,
                        end=char_end,
                    ).to_dict())
                    stats["entities"] += 1
                else:
                    stats["entities_skipped"] += 1

            if not entities:
                stats["skipped"] += 1
                continue

            pos_types = {e["type"] for e in entities}
            neg_types = neg_sampler.sample(pos_types, rng=rng)
            query_types = sorted(pos_types) + neg_types

            rec = NerRecord(
                source=SOURCE,
                source_id=f"french_ner_{split_name}_{idx}",
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

    logger.info("french_ner: %s", stats)
    return stats


if __name__ == "__main__":
    convert()
