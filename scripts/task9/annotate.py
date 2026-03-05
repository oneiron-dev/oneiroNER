#!/usr/bin/env python3
"""Task 9 Step 2: bulk LLM annotation of filtered silver records."""

import argparse
import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.llm_annotator import annotate_conversation, annotate_passage
from lib.passage_chunker import chunk_passage
from lib.span_fixer import verify_and_fix_spans

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

FILTERED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "silver_filtered"
OUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "silver_annotated"
CHECKPOINT_DIR = OUT_DIR / ".checkpoints"

ALL_SOURCES = [
    "mentalchat", "therapy_conversations", "roleplay_hieu",
    "synthetic_persona_chat", "personachat", "prosocial_dialog",
    "pippa", "opencharacter", "reddit_confessions",
]

FLAG_MIN_ENTITIES = 0
FLAG_MAX_ENTITIES = 10

MAX_RETRIES = 3
BACKOFF_BASE = 30

write_lock = threading.Lock()
checkpoint_lock = threading.Lock()


def load_checkpoint(source: str) -> set[str]:
    cp_path = CHECKPOINT_DIR / f"{source}.json"
    if cp_path.exists():
        with open(cp_path) as f:
            return set(json.load(f))
    return set()


def save_checkpoint(source: str, processed_ids: set[str]):
    with checkpoint_lock:
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        cp_path = CHECKPOINT_DIR / f"{source}.json"
        with open(cp_path, "w") as f:
            json.dump(sorted(processed_ids), f)


def annotate_passage_chunked(text: str, language: str, provider: str) -> list[dict]:
    chunks = chunk_passage(text)
    if len(chunks) == 1:
        return annotate_passage(text, language, provider)

    all_entities = []
    for chunk_text, offset in chunks:
        entities = annotate_passage(chunk_text, language, provider)
        for ent in entities:
            ent["start"] += offset
            ent["end"] += offset
        all_entities.extend(entities)

    seen = set()
    deduped = []
    for ent in all_entities:
        key = (ent["surface"], ent["start"], ent["end"], ent["type"])
        if key not in seen:
            seen.add(key)
            deduped.append(ent)
    return deduped


def is_flagged(entities: list[dict]) -> bool:
    n = len(entities)
    return n <= FLAG_MIN_ENTITIES or n > FLAG_MAX_ENTITIES


def process_record(rec: dict, provider: str) -> dict:
    fmt = rec.get("format", "passage")
    language = rec.get("language", "en")

    if fmt == "passage":
        raw_entities = annotate_passage_chunked(rec["text"], language, provider)
        fixed, stats = verify_and_fix_spans(raw_entities, rec["text"])
    else:
        raw_entities = annotate_conversation(rec["turns"], language, provider)
        fixed, stats = verify_and_fix_spans(raw_entities, "", turns=rec["turns"])

    rec["entities"] = fixed
    rec["annotation_stats"] = stats
    rec["flagged"] = is_flagged(fixed)
    rec["has_entities"] = len(fixed) > 0
    return rec


def process_record_with_retry(rec: dict, provider: str) -> dict | None:
    for attempt in range(MAX_RETRIES + 1):
        try:
            return process_record(rec, provider)
        except Exception as e:
            err = str(e).lower()
            if "rate" in err or "limit" in err or "429" in err or "quota" in err:
                if attempt < MAX_RETRIES:
                    wait = BACKOFF_BASE * (2 ** attempt)
                    logger.warning("Rate limited (attempt %d/%d), waiting %ds...", attempt + 1, MAX_RETRIES + 1, wait)
                    time.sleep(wait)
                    continue
                logger.error("Rate limited after %d retries, skipping record", MAX_RETRIES + 1)
                return None
            if attempt < MAX_RETRIES:
                logger.debug("Annotation error (attempt %d): %s", attempt + 1, e)
                time.sleep(2)
                continue
            logger.warning("Annotation failed after retries: %s", e)
            return None


def process_source(
    source: str,
    limit: int | None,
    resume: bool,
    concurrency: int,
    provider: str,
):
    in_path = FILTERED_DIR / f"{source}.jsonl"
    if not in_path.exists():
        logger.warning("Skipping %s: no filtered file at %s", source, in_path)
        return

    out_path = OUT_DIR / f"{source}.jsonl"
    processed_ids = load_checkpoint(source) if resume else set()

    records = []
    with open(in_path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            rec = json.loads(line)
            sid = rec.get("source_id", str(i))
            if sid in processed_ids:
                continue
            rec["_sid"] = sid
            records.append(rec)

    if not records:
        logger.info("%s: nothing to process (all checkpointed)", source)
        return

    logger.info("%s: %d records to annotate (concurrency=%d, provider=%s)", source, len(records), concurrency, provider)

    mode = "a" if resume else "w"
    fh = open(out_path, mode)
    written = 0
    failed = 0
    start_time = time.time()

    def worker(rec: dict) -> dict | None:
        return process_record_with_retry(rec, provider)

    try:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(worker, rec): rec for rec in records}
            for future in as_completed(futures):
                rec = futures[future]
                sid = rec["_sid"]
                try:
                    result = future.result()
                except Exception as e:
                    logger.warning("Worker error for %s: %s", sid, e)
                    failed += 1
                    continue

                if result is None:
                    failed += 1
                    continue

                with write_lock:
                    result.pop("_sid", None)
                    fh.write(json.dumps(result, ensure_ascii=False) + "\n")
                    fh.flush()
                    processed_ids.add(sid)
                    written += 1

                if written % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = written / elapsed if elapsed > 0 else 0
                    logger.info(
                        "%s: %d/%d written, %d failed | %.1f rec/min",
                        source, written, len(records), failed, rate * 60,
                    )
                    save_checkpoint(source, processed_ids)
    finally:
        fh.close()
        save_checkpoint(source, processed_ids)

    elapsed = time.time() - start_time
    logger.info(
        "%s: DONE — %d written, %d failed (%.0fs, %.1f rec/min)",
        source, written, failed, elapsed, written / elapsed * 60 if elapsed > 0 else 0,
    )


def main():
    parser = argparse.ArgumentParser(description="Annotate filtered silver records via LLM")
    parser.add_argument("--source", default="all", help="Source name or 'all'")
    parser.add_argument("--limit", type=int, default=None, help="Limit records per source")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--concurrency", type=int, default=5, help="Worker threads")
    parser.add_argument("--provider", default="spark", choices=["spark", "codex", "gemini"], help="LLM provider")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sources = ALL_SOURCES if args.source == "all" else [args.source]
    for s in sources:
        if s not in ALL_SOURCES:
            logger.error("Unknown source: %s", s)
            sys.exit(1)

    for s in sources:
        process_source(s, args.limit, args.resume, args.concurrency, args.provider)

    logger.info("All sources complete. Output: %s", OUT_DIR)


if __name__ == "__main__":
    main()
