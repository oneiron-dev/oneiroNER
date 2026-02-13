"""Convert chinese_ner_sft subsets to NER JSONL."""

import json
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from lib.schema import NerRecord, Entity
from lib.span_validator import validate_span
from lib.span_computer import compute_span
from lib.negative_sampler import NegativeSampler

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data" / "raw" / "chinese_ner_sft" / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "processed"

SOURCE = "chinese_ner_sft"
LANGUAGE = "zh"
CONFIDENCE = "gold"

KNOWN_EXCLUSIVE = {"bank", "ccfbdci", "ccks2019_task1", "dlner", "e_commerce", "finance_sina", "weibo", "youku"}
KNOWN_INCLUSIVE = {"cluener", "cmeee"}

neg_sampler = NegativeSampler()
rng = random.Random(42)


def detect_convention(lines: list[dict], subset: str) -> str:
    if subset in KNOWN_EXCLUSIVE:
        return "exclusive"
    if subset in KNOWN_INCLUSIVE:
        return "inclusive"

    exc_count = inc_count = 0
    tested = 0
    for line in lines:
        if tested >= 20:
            break
        for ent in line.get("entities", []):
            if tested >= 20:
                break
            start = ent.get("start_idx")
            end = ent.get("end_idx")
            surface = ent.get("entity_text", "")
            if start is None or end is None or not surface:
                continue
            text = line["text"]
            if text[start:end] == surface:
                exc_count += 1
            elif text[start:end + 1] == surface:
                inc_count += 1
            tested += 1

    if exc_count > 0 and inc_count == 0:
        return "exclusive"
    if inc_count > 0 and exc_count == 0:
        return "inclusive"
    if exc_count > 0 and inc_count > 0:
        return "mixed"
    return "null"


def resolve_entity(text: str, ent: dict, convention: str) -> tuple[int, int, str] | None:
    start = ent.get("start_idx")
    end = ent.get("end_idx")
    surface = ent.get("entity_text", "")

    if start is None or end is None:
        result = compute_span(text, surface)
        if result:
            return result[0], result[1], surface
        return None

    if convention == "exclusive":
        if validate_span(text, surface, start, end):
            return start, end, surface
    elif convention == "inclusive":
        if validate_span(text, surface, start, end + 1):
            return start, end + 1, surface
    elif convention == "mixed":
        if validate_span(text, surface, start, end):
            return start, end, surface
        if validate_span(text, surface, start, end + 1):
            return start, end + 1, surface
    elif convention == "null":
        result = compute_span(text, surface)
        if result:
            return result[0], result[1], surface

    result = compute_span(text, surface)
    if result:
        return result[0], result[1], surface
    return None


def convert(output_dir: str | None = None) -> dict:
    out = Path(output_dir) if output_dir else OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    outfile = out / f"{SOURCE}.jsonl"

    stats = {"total": 0, "written": 0, "skipped": 0, "entities": 0, "entities_skipped": 0}

    with open(outfile, "w") as fout:
        for jsonl_file in sorted(DATA_DIR.glob("*.jsonl")):
            subset = jsonl_file.stem
            lines = []
            with open(jsonl_file) as f:
                for raw in f:
                    raw = raw.strip()
                    if raw:
                        lines.append(json.loads(raw))

            convention = detect_convention(lines, subset)
            logger.info("chinese_ner_sft/%s: %d lines, convention=%s", subset, len(lines), convention)

            for idx, line in enumerate(lines):
                stats["total"] += 1
                text = line["text"]
                entities = []

                for ent in line.get("entities", []):
                    resolved = resolve_entity(text, ent, convention)
                    if resolved:
                        s, e, surf = resolved
                        entities.append(Entity(
                            surface=surf,
                            type=ent["entity_label"],
                            original_type=ent["entity_label"],
                            start=s,
                            end=e,
                        ).to_dict())
                        stats["entities"] += 1
                    else:
                        stats["entities_skipped"] += 1
                        logger.debug("Failed entity in %s/%d: %s", subset, idx, ent.get("entity_text"))

                if not entities:
                    stats["skipped"] += 1
                    continue

                pos_types = {e["type"] for e in entities}
                neg_types = neg_sampler.sample(pos_types, rng=rng)
                query_types = sorted(pos_types) + neg_types

                rec = NerRecord(
                    source=SOURCE,
                    source_id=f"chinese_ner_sft_{subset}_{idx}",
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

    logger.info("chinese_ner_sft: %s", stats)
    return stats


if __name__ == "__main__":
    convert()
