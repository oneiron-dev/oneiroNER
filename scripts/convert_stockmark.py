"""Convert stockmark-ner-ja to NER JSONL."""

import json
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from lib.schema import NerRecord, Entity
from lib.span_validator import validate_span
from lib.negative_sampler import NegativeSampler

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent.parent / "data" / "raw" / "stockmark-ner-ja" / "ner.json"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "processed"

SOURCE = "stockmark_ner_ja"
LANGUAGE = "ja"
CONFIDENCE = "gold"

neg_sampler = NegativeSampler()
rng = random.Random(42)


def convert(output_dir: str | None = None) -> dict:
    out = Path(output_dir) if output_dir else OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    outfile = out / f"{SOURCE}.jsonl"

    with open(DATA_PATH) as f:
        data = json.load(f)

    stats = {"total": 0, "written": 0, "skipped": 0, "entities": 0, "entities_skipped": 0}

    with open(outfile, "w") as fout:
        for entry in data:
            stats["total"] += 1
            text = entry["text"]
            entities = []

            for ent in entry.get("entities", []):
                start, end = ent["span"][0], ent["span"][1]
                surface = ent["name"]
                if validate_span(text, surface, start, end):
                    entities.append(Entity(
                        surface=surface,
                        type=ent["type"],
                        original_type=ent["type"],
                        start=start,
                        end=end,
                    ).to_dict())
                    stats["entities"] += 1
                else:
                    stats["entities_skipped"] += 1
                    logger.debug("Span mismatch in %s: '%s' vs text[%d:%d]='%s'",
                                 entry["curid"], surface, start, end, text[start:end])

            if not entities:
                stats["skipped"] += 1
                continue

            pos_types = {e["type"] for e in entities}
            neg_types = neg_sampler.sample(pos_types, rng=rng)
            query_types = sorted(pos_types) + neg_types

            rec = NerRecord(
                source=SOURCE,
                source_id=f"stockmark_{entry['curid']}",
                language=LANGUAGE,
                split="train",
                confidence=CONFIDENCE,
                provenance=[SOURCE],
                text=text,
                query_types=query_types,
                entities=entities,
            )
            rec.validate()
            fout.write(rec.to_jsonl() + "\n")
            stats["written"] += 1

    logger.info("stockmark: %s", stats)
    return stats


if __name__ == "__main__":
    convert()
