"""Two-pass streaming deduplication with merge-on-duplicate.

Pass 1: Stream all input files, compute text hash per line, build
        hash → [(file_path, byte_offset)] index.  Only hashes and
        offsets are stored — no parsed records in memory.
Pass 2: Single-occurrence hashes → write the raw line through directly.
        Multi-occurrence hashes → seek-read each record, merge, write.
"""

import hashlib
import json
import logging
import random
import unicodedata
from pathlib import Path

from .schema import NerRecord, ConversationRecord, record_from_jsonl, min_confidence
from .negative_sampler import NegativeSampler
from .span_computer import compute_span

logger = logging.getLogger(__name__)


def text_hash(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _extract_text_fast(line: str) -> str | None:
    try:
        d = json.loads(line)
        text = d.get("text")
        if text is None and "turns" in d:
            text = "\n".join(t["text"] for t in d["turns"])
        return text
    except (json.JSONDecodeError, KeyError):
        return None


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
        needs_realign = rec is not base and rec.text != base.text
        for ent in rec.entities:
            if needs_realign:
                if base.text[ent["start"]:ent["end"]] != ent["surface"]:
                    span = compute_span(base.text, ent["surface"])
                    if span is None:
                        logger.debug("Cannot realign entity '%s' to base text, skipping", ent["surface"])
                        continue
                    ent = {**ent, "start": span[0], "end": span[1]}
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


def _recompute_negatives(line: str, sampler: NegativeSampler, rng: random.Random) -> str | None:
    try:
        d = json.loads(line)
        positive_types = set(ent["type"] for ent in d["entities"])
        negatives = sampler.sample(positive_types, rng=rng)
        d["query_types"] = sorted(positive_types | set(negatives))
        try:
            rec = record_from_jsonl(json.dumps(d, ensure_ascii=False))
            rec.validate()
            return rec.to_jsonl()
        except Exception:
            return json.dumps(d, ensure_ascii=False)
    except Exception as e:
        logger.debug("Recompute/validation failed: %s", e)
        return None


def dedup_files(
    input_files: list[str | Path],
    output_file: str | Path,
    sampler: NegativeSampler,
    seed: int = 42,
) -> dict:
    rng = random.Random(seed)
    output_file = Path(output_file)
    stats = {"total_input": 0, "unique_texts": 0, "merged_count": 0, "output_count": 0,
             "failed_singles": 0, "failed_merges": 0}

    # Pass 1: Build hash → [(file_path, byte_offset)] index
    logger.info("Pass 1: Building text hash index from %d files", len(input_files))
    hash_index: dict[str, list[tuple[str, int]]] = {}

    for fpath in input_files:
        fpath = Path(fpath)
        if not fpath.exists():
            logger.warning("File not found: %s", fpath)
            continue
        fpath_str = str(fpath)
        with open(fpath) as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                stats["total_input"] += 1
                text = _extract_text_fast(line)
                if text is None:
                    continue
                h = text_hash(text)
                if h not in hash_index:
                    hash_index[h] = [(fpath_str, offset)]
                else:
                    hash_index[h].append((fpath_str, offset))

    stats["unique_texts"] = len(hash_index)
    logger.info("Pass 1 complete: %d records, %d unique texts", stats["total_input"], stats["unique_texts"])

    # Pass 2: Write singles through, merge duplicates
    logger.info("Pass 2: Merging duplicates and writing output")

    # Cache open file handles for seek-reads
    import io
    file_handles: dict[str, io.TextIOWrapper] = {}

    def read_line_at(fpath: str, offset: int) -> str:
        if fpath not in file_handles:
            file_handles[fpath] = open(fpath)
        fh = file_handles[fpath]
        fh.seek(offset)
        return fh.readline().strip()

    try:
        with open(output_file, "w") as out:
            for h, locations in hash_index.items():
                if len(locations) == 1:
                    fpath, offset = locations[0]
                    line = read_line_at(fpath, offset)
                    result = _recompute_negatives(line, sampler, rng)
                    if result:
                        out.write(result + "\n")
                        stats["output_count"] += 1
                    else:
                        stats["failed_singles"] += 1
                        logger.warning("Failed to recompute single record from %s@%d", fpath, offset)
                else:
                    stats["merged_count"] += 1
                    parsed = []
                    for fpath, offset in locations:
                        line = read_line_at(fpath, offset)
                        try:
                            parsed.append(record_from_jsonl(line))
                        except Exception as e:
                            logger.debug("Failed to parse duplicate record: %s", e)
                    if not parsed:
                        continue
                    # ConversationRecords: keep first (merge only works for NerRecord)
                    if isinstance(parsed[0], ConversationRecord):
                        result = _recompute_negatives(parsed[0].to_jsonl(), sampler, rng)
                        if result:
                            out.write(result + "\n")
                            stats["output_count"] += 1
                        else:
                            stats["failed_merges"] += 1
                        continue
                    ner_records = [r for r in parsed if isinstance(r, NerRecord)]
                    if not ner_records:
                        continue
                    try:
                        merged = merge_records(ner_records, sampler, rng)
                        merged.validate()
                        out.write(merged.to_jsonl() + "\n")
                        stats["output_count"] += 1
                    except (AssertionError, Exception) as e:
                        logger.warning("Merge failed (%d records): %s — falling back to first valid record", len(ner_records), e)
                        stats["failed_merges"] += 1
                        for fallback in ner_records:
                            fb_line = _recompute_negatives(fallback.to_jsonl(), sampler, rng)
                            if fb_line:
                                out.write(fb_line + "\n")
                                stats["output_count"] += 1
                                break
    finally:
        for fh in file_handles.values():
            fh.close()

    if stats["failed_singles"] or stats["failed_merges"]:
        logger.warning(
            "Dedup failures: %d singles, %d merges (fallback attempted)",
            stats["failed_singles"], stats["failed_merges"],
        )
    logger.info(
        "Dedup complete: %d input → %d unique → %d merged → %d output",
        stats["total_input"], stats["unique_texts"], stats["merged_count"], stats["output_count"],
    )
    return stats
