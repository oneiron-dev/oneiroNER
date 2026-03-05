#!/usr/bin/env python3
"""Task 9 Step 2: bulk LLM annotation of filtered silver records."""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.llm_annotator import annotate_conversation, annotate_passage
from lib.passage_chunker import chunk_passage
from lib.span_fixer import verify_and_fix_spans

logger = logging.getLogger(__name__)

FILTERED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "silver_filtered"
OUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "silver_annotated"
CHECKPOINT_DIR = OUT_DIR / ".checkpoints"

ALL_SOURCES = [
    "reddit_confessions", "opencharacter", "therapy_conversations",
    "prosocial_dialog", "personachat", "pippa",
    "synthetic_persona_chat", "roleplay_hieu", "mentalchat",
]

FLAG_MIN_ENTITIES = 0
FLAG_MAX_ENTITIES = 10

MAX_RETRIES = 3
BACKOFF_BASE = 30  # seconds; doubles each retry (30, 60, 120)


def load_checkpoint(source: str) -> set[str]:
    cp_path = CHECKPOINT_DIR / f"{source}.json"
    if cp_path.exists():
        with open(cp_path) as f:
            return set(json.load(f))
    return set()


def save_checkpoint(source: str, processed_ids: set[str]):
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
            raise


def process_source(
    source: str,
    limit: int | None,
    resume: bool,
    concurrency: int,
    quality_pass: bool,
    provider: str,
):
    in_path = FILTERED_DIR / f"{source}.jsonl"
    if not in_path.exists():
        print(f"  Skipping {source}: no filtered file", file=sys.stderr)
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
            records.append(rec)

    if not records:
        print(f"  {source}: nothing to process (all checkpointed)")
        return

    mode = "a" if resume else "w"
    written = 0
    flagged_count = 0

    with open(out_path, mode) as fout:
        for rec in records:
            result = process_record_with_retry(rec, provider)
            if result is None:
                save_checkpoint(source, processed_ids)
                print(f"  {source}: stopped at {written}/{len(records)} due to rate limit. Use --resume to continue.")
                return
            processed_ids.add(rec.get("source_id", ""))

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            written += 1
            if result.get("flagged"):
                flagged_count += 1

            if written % 100 == 0:
                save_checkpoint(source, processed_ids)
                print(f"    ...{written}/{len(records)} (flagged: {flagged_count})")

    save_checkpoint(source, processed_ids)
    print(f"  {source}: {written} annotated, {flagged_count} flagged")

    if quality_pass and flagged_count > 0:
        run_quality_pass(source, out_path, provider="codex")


def run_quality_pass(source: str, annotated_path: Path, provider: str = "codex"):
    print(f"  {source}: running quality pass on flagged records...")
    records = []
    with open(annotated_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("flagged"):
                records.append(rec)

    if not records:
        return

    reannotated = 0
    all_records = []
    with open(annotated_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("flagged"):
                result = process_record_with_retry(rec, provider)
                if result is None:
                    all_records.append(rec)
                    continue
                result["flagged"] = False
                result["quality_pass"] = True
                reannotated += 1
                all_records.append(result)
            else:
                all_records.append(rec)

    with open(annotated_path, "w") as fout:
        for rec in all_records:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"  {source}: quality pass reannotated {reannotated} records")


def main():
    parser = argparse.ArgumentParser(description="Annotate filtered silver records via LLM")
    parser.add_argument("--source", default="all", help="Source name or 'all'")
    parser.add_argument("--limit", type=int, default=None, help="Limit records per source")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--concurrency", type=int, default=5, help="Concurrency limit (reserved for async)")
    parser.add_argument("--quality-pass", action="store_true", help="Run codex quality pass on flagged records")
    parser.add_argument("--provider", default="spark", choices=["spark", "codex"], help="LLM provider")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sources = ALL_SOURCES if args.source == "all" else [args.source]
    for s in sources:
        if s not in ALL_SOURCES:
            print(f"Unknown source: {s}", file=sys.stderr)
            sys.exit(1)

    for s in sources:
        print(f"Annotating {s}...")
        process_source(s, args.limit, args.resume, args.concurrency, args.quality_pass, args.provider)

    print("\nDone. Output:", OUT_DIR)


if __name__ == "__main__":
    main()
