"""Convert nlp-kmu/kor_ner (Korean NER, custom annotated format) to NER JSONL."""

import logging
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from lib.negative_sampler import NegativeSampler
from lib.schema import Entity, NerRecord
from lib.span_validator import validate_span

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data" / "raw" / "kor_ner"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "processed"

SOURCE = "kor_ner"
LANGUAGE = "ko"
CONFIDENCE = "gold"

TYPE_MAP = {"PS": "PERSON", "OG": "ORG", "LC": "PLACE", "DT": "DATE", "TI": "DATE"}

_ENTITY_RE = re.compile(r"<([^:>]+):([A-Z]+)>")

neg_sampler = NegativeSampler()
rng = random.Random(42)


def parse_annotated_line(annot_text: str) -> tuple[str, list[dict]]:
    entities = []
    clean_parts = []
    clean_pos = 0
    last_end = 0

    for m in _ENTITY_RE.finditer(annot_text):
        before = annot_text[last_end:m.start()]
        clean_parts.append(before)
        clean_pos += len(before)

        surface = m.group(1)
        etype = m.group(2)
        mapped = TYPE_MAP.get(etype)
        if mapped:
            entities.append({
                "surface": surface,
                "type": mapped,
                "original_type": etype,
                "start": clean_pos,
                "end": clean_pos + len(surface),
            })
        clean_parts.append(surface)
        clean_pos += len(surface)
        last_end = m.end()

    clean_parts.append(annot_text[last_end:])
    clean_text = "".join(clean_parts)
    return clean_text, entities


def parse_file(path: Path) -> list[dict]:
    sentences = []
    text = None
    annot_text = None

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if annot_text is not None:
                    sentences.append({"text": text, "annot_text": annot_text})
                text = None
                annot_text = None
            elif line.startswith("; "):
                text = line[2:]
            elif line.startswith("$"):
                annot_text = line[1:]

    if annot_text is not None:
        sentences.append({"text": text, "annot_text": annot_text})

    return sentences


def convert(output_dir: str | None = None) -> dict:
    out = Path(output_dir) if output_dir else OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    outfile = out / f"{SOURCE}.jsonl"

    stats = {"total": 0, "written": 0, "skipped": 0, "entities": 0, "entities_skipped": 0}
    records = []

    for split_file in ["ner.train", "ner.dev", "ner.test"]:
        path = DATA_DIR / split_file
        if not path.exists():
            logger.warning("Missing: %s", path)
            continue

        split_name = split_file.split(".")[1]
        sentences = parse_file(path)
        logger.info("Processing kor_ner %s: %d sentences", split_name, len(sentences))

        for idx, sent in enumerate(sentences):
            stats["total"] += 1
            text = sent["text"]
            if not text:
                stats["skipped"] += 1
                continue

            clean_text, raw_entities = parse_annotated_line(sent["annot_text"])

            entities = []
            for ent in raw_entities:
                if validate_span(text, ent["surface"], ent["start"], ent["end"]):
                    entities.append(Entity(
                        surface=ent["surface"],
                        type=ent["type"],
                        original_type=ent["original_type"],
                        start=ent["start"],
                        end=ent["end"],
                    ).to_dict())
                    stats["entities"] += 1
                elif validate_span(clean_text, ent["surface"], ent["start"], ent["end"]):
                    entities.append(Entity(
                        surface=ent["surface"],
                        type=ent["type"],
                        original_type=ent["original_type"],
                        start=ent["start"],
                        end=ent["end"],
                    ).to_dict())
                    stats["entities"] += 1
                else:
                    stats["entities_skipped"] += 1
                    logger.debug("Span mismatch: '%s' at [%d:%d]", ent["surface"], ent["start"], ent["end"])

            if not entities:
                stats["skipped"] += 1
                continue

            pos_types = {e["type"] for e in entities}
            neg_types = neg_sampler.sample(pos_types, rng=rng)
            query_types = sorted(pos_types) + neg_types

            use_text = text if text == clean_text else clean_text
            rec = NerRecord(
                source=SOURCE,
                source_id=f"kor_ner_{split_name}_{idx}",
                language=LANGUAGE,
                split="train",
                confidence=CONFIDENCE,
                provenance=[SOURCE],
                text=use_text,
                query_types=query_types,
                entities=entities,
            )
            rec.validate()
            records.append(rec)

    with open(outfile, "w") as fout:
        for rec in records:
            fout.write(rec.to_jsonl() + "\n")
            stats["written"] += 1

    logger.info("kor_ner: %s", stats)
    return stats


if __name__ == "__main__":
    convert()
