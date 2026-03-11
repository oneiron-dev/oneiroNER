#!/usr/bin/env python3
"""Convert raw silver_synthetic multilingual conversations to pipeline schema."""

import json
import logging
import random
from pathlib import Path

from lib.schema import ConversationRecord
from lib.negative_sampler import NegativeSampler

logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parent.parent / "data" / "raw" / "silver_synthetic"
SEED = 42


def _validate_relaxed(rec: ConversationRecord):
    """ConversationRecord.validate() but without the 2-4 turn limit."""
    assert rec.schema_version == "1.0"
    assert rec.split in {"train", "val", "eval", "zero_shot_eval"}
    assert rec.confidence in {"human-gold", "synthetic-gold", "gold", "silver"}
    assert len(rec.provenance) >= 1
    assert rec.format == "conversation"
    assert len(rec.turns) >= 2, f"Turn count {len(rec.turns)} < 2"
    for i, turn in enumerate(rec.turns):
        assert turn.get("speaker"), f"Turn {i} missing speaker"
        assert turn.get("text"), f"Turn {i} missing text"
    assert len(rec.query_types) >= 1
    assert len(rec.query_types) == len(set(rec.query_types))

    positive_types = set()
    for ent in rec.entities:
        assert ent["end"] > ent["start"]
        ti = ent["turn_index"]
        assert 0 <= ti < len(rec.turns)
        span = rec.turns[ti]["text"][ent["start"]:ent["end"]]
        assert span == ent["surface"], f"Span mismatch: '{span}' != '{ent['surface']}'"
        positive_types.add(ent["type"])

    assert positive_types <= set(rec.query_types)
    neg_count = len(rec.query_types) - len(positive_types)
    assert 2 <= neg_count <= 5


def convert(output_dir: str) -> dict:
    output_path = Path(output_dir) / "silver_synthetic_ml.jsonl"
    sampler = NegativeSampler()
    rng = random.Random(SEED)

    total = 0
    errors = 0
    by_lang = {}

    with open(output_path, "w") as out_f:
        for jsonl_file in sorted(RAW_DIR.glob("*_batch1.jsonl")):
            for line in open(jsonl_file):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    errors += 1
                    continue

                entities = []
                positive_types = set()
                for ent in raw.get("entities", []):
                    entities.append({
                        "surface": ent["surface"],
                        "type": ent["type"],
                        "original_type": ent["type"],
                        "start": ent["start"],
                        "end": ent["end"],
                        "turn_index": ent["turn_index"],
                    })
                    positive_types.add(ent["type"])

                negatives = sampler.sample(positive_types, rng=rng)
                query_types = sorted(positive_types | set(negatives))

                lang = raw.get("language", "unknown")
                rec = ConversationRecord(
                    source=raw.get("source", f"synthetic_{lang}"),
                    source_id=raw.get("source_id", f"synth_{lang}_{total}"),
                    language=lang,
                    split="train",
                    confidence="silver",
                    provenance=["multilingual_synthetic"],
                    format="conversation",
                    turns=raw.get("turns", []),
                    query_types=query_types,
                    entities=entities,
                )

                try:
                    _validate_relaxed(rec)
                except (AssertionError, Exception) as e:
                    logger.debug("Validation failed for %s: %s", rec.source_id, e)
                    errors += 1
                    continue

                out_f.write(rec.to_jsonl() + "\n")
                total += 1
                by_lang[lang] = by_lang.get(lang, 0) + 1

    logger.info("Converted %d silver synthetic records (%d errors) to %s", total, errors, output_path)
    logger.info("Languages: %d, distribution: %s", len(by_lang), dict(sorted(by_lang.items())))

    return {"total": total, "errors": errors, "languages": len(by_lang), "output": str(output_path)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    convert(str(Path(__file__).parent.parent / "data" / "processed"))
