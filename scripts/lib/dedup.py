"""Two-pass streaming deduplication with merge-on-duplicate."""

import hashlib
import json
import logging
import random
import unicodedata
from collections import defaultdict
from pathlib import Path

from .schema import NerRecord, min_confidence
from .negative_sampler import NegativeSampler

logger = logging.getLogger(__name__)


def text_hash(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def merge_records(records: list[NerRecord], sampler: NegativeSampler, rng: random.Random) -> NerRecord:
    base = records[0]

    all_entities = []
    seen_spans = set()
    all_provenance = []
    seen_prov = set()
    confidences = []

    for rec in records:
        confidences.append(rec.confidence)
        for prov in rec.provenance:
            if prov not in seen_prov:
                all_provenance.append(prov)
                seen_prov.add(prov)
        for ent in rec.entities:
            key = (ent["start"], ent["end"], ent["type"])
            if key not in seen_spans:
                seen_spans.add(key)
                all_entities.append(ent)

    positive_types = set(ent["type"] for ent in all_entities)
    negatives = sampler.sample(positive_types, rng=rng)
    query_types = sorted(positive_types | set(negatives))

    merged = NerRecord(
        source=base.source,
        source_id=base.source_id,
        language=base.language,
        split=base.split,
        confidence=min_confidence(*confidences),
        provenance=all_provenance,
        text=base.text,
        query_types=query_types,
        entities=all_entities,
    )
    return merged


def dedup_files(
    input_files: list[str | Path],
    output_file: str | Path,
    sampler: NegativeSampler,
    seed: int = 42,
) -> dict:
    rng = random.Random(seed)
    output_file = Path(output_file)
    stats = {"total_input": 0, "unique_texts": 0, "merged_count": 0, "output_count": 0}

    logger.info("Pass 1: Building text hash index from %d files", len(input_files))
    hash_to_records: dict[str, list[NerRecord]] = defaultdict(list)

    for fpath in input_files:
        fpath = Path(fpath)
        if not fpath.exists():
            logger.warning("File not found: %s", fpath)
            continue
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                stats["total_input"] += 1
                try:
                    rec = NerRecord.from_jsonl(line)
                    h = text_hash(rec.text)
                    hash_to_records[h].append(rec)
                except Exception as e:
                    logger.warning("Failed to parse record: %s", e)

    stats["unique_texts"] = len(hash_to_records)
    logger.info("Pass 1 complete: %d records, %d unique texts", stats["total_input"], stats["unique_texts"])

    logger.info("Pass 2: Merging duplicates and writing output")
    with open(output_file, "w") as out:
        for h, records in hash_to_records.items():
            if len(records) == 1:
                rec = records[0]
                positive_types = set(ent["type"] for ent in rec.entities)
                negatives = sampler.sample(positive_types, rng=rng)
                rec.query_types = sorted(positive_types | set(negatives))
                try:
                    rec.validate()
                    out.write(rec.to_jsonl() + "\n")
                    stats["output_count"] += 1
                except AssertionError as e:
                    logger.debug("Validation failed after recompute: %s", e)
            else:
                stats["merged_count"] += 1
                try:
                    merged = merge_records(records, sampler, rng)
                    merged.validate()
                    out.write(merged.to_jsonl() + "\n")
                    stats["output_count"] += 1
                except (AssertionError, Exception) as e:
                    logger.debug("Merge/validation failed: %s", e)

    logger.info(
        "Dedup complete: %d input → %d unique → %d merged → %d output",
        stats["total_input"], stats["unique_texts"], stats["merged_count"], stats["output_count"],
    )
    return stats
