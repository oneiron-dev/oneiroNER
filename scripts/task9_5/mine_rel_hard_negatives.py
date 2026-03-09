#!/usr/bin/env python3
"""Mine relationship hard negatives from silver data.

Finds generic/hypothetical relationship mentions that should NOT be tagged.
English-only for v1.
"""

import argparse
import logging
import re
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (
    BASE_DIR,
    entity_key,
    load_silver_records,
    print_summary,
    write_output_jsonl,
)
from _llm_classify import DEFAULT_MODEL, classify_is_specific_ref

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

REL_NOUNS = r"(?:friend|partner|boyfriend|girlfriend|husband|wife|spouse|boss|manager|coworker|colleague|teacher|doctor|therapist|counselor|mentor|neighbor|neighbour|acquaintance)"

GENERIC_PATTERNS = [
    (re.compile(r"\b(a|any|some|every|no)\s+" + REL_NOUNS + r"\b", re.IGNORECASE), "generic_indefinite"),
    (re.compile(r"\b(relationship|friendship|marriage|partnership)\s+(is|are|can|should|would|might)\b", re.IGNORECASE), "abstract_concept"),
    (re.compile(r"\b(any|a good|your average|the average|the typical)\s+" + REL_NOUNS + r"\b", re.IGNORECASE), "non_specific_role"),
    (re.compile(r"\b(if I had|imagine having|would want|wish I had|need a|looking for a|want a)\s+(a\s+)?" + REL_NOUNS + r"\b", re.IGNORECASE), "hypothetical"),
    (re.compile(r"\b(you should|you need|one should|everyone needs|people need)\s+(a\s+)?" + REL_NOUNS + r"\b", re.IGNORECASE), "general_advice"),
]


def spans_overlap(a_start, a_end, b_start, b_end):
    return a_start < b_end and b_start < a_end


def extract_context(text, start, end, window=200):
    ctx_start = max(0, start - window // 2)
    ctx_end = min(len(text), end + window // 2)
    return text[ctx_start:ctx_end]


def mine_record(record, source_file):
    source = record.get("source", "")
    source_id = record.get("source_id", "")
    is_conv = record.get("format") == "conversation" or record.get("turns")
    candidates = []

    existing_spans = set()
    for ent in record.get("entities", []):
        existing_spans.add((ent.get("turn_index"), ent["start"], ent["end"]))

    def check_text(text, turn_index=None):
        for pattern, reason in GENERIC_PATTERNS:
            for m in pattern.finditer(text):
                start, end = m.start(), m.end()
                surface = m.group(0)

                overlaps = False
                for (ti, es, ee) in existing_spans:
                    if ti == turn_index and spans_overlap(start, end, es, ee):
                        overlaps = True
                        break
                if overlaps:
                    continue

                ekey = list(entity_key(
                    {"turn_index": turn_index, "start": start, "end": end, "surface": surface},
                    source, source_id
                ))
                context = extract_context(text, start, end)

                candidates.append({
                    "source": source,
                    "source_id": source_id,
                    "source_file": source_file,
                    "entity_key": ekey,
                    "text_or_turn_text": context,
                    "surface": surface,
                    "start": start,
                    "end": end,
                    "turn_index": turn_index,
                    "negative_reason": reason,
                    "confidence": 0.95,
                    "triage_method": "regex",
                    "language": "en",
                })

    if is_conv:
        for ti, turn in enumerate(record.get("turns", [])):
            check_text(turn["text"], turn_index=ti)
    else:
        check_text(record.get("text", ""))

    return candidates


def main():
    parser = argparse.ArgumentParser(description="Mine relationship hard negatives from silver data")
    parser.add_argument("--input-glob", default="data/processed/task9_silver_*.jsonl")
    parser.add_argument("--output", default="data/processed/task9_5_rel_hard_negatives.jsonl")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--llm-triage", action="store_true", help="Confirm candidates via DeepSeek")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    glob_path = BASE_DIR / args.input_glob
    out_path = BASE_DIR / args.output

    stats = Counter()
    all_candidates = []

    logger.info("Scanning records (English only)...")
    for filepath, line_num, record in load_silver_records(glob_path):
        if args.limit and stats["records_scanned"] >= args.limit:
            break

        if record.get("language", "en") != "en":
            stats["records_skipped_non_en"] += 1
            continue

        stats["records_scanned"] += 1
        source_file = filepath.name
        candidates = mine_record(record, source_file)

        for c in candidates:
            stats[f"pattern:{c['negative_reason']}"] += 1

        all_candidates.extend(candidates)

    stats["total_candidates"] = len(all_candidates)
    logger.info("Found %d candidates from %d EN records", len(all_candidates), stats["records_scanned"])

    if args.dry_run:
        print("\n=== DRY RUN SUMMARY ===")
        print_summary(stats)
        return

    if args.llm_triage and all_candidates:
        logger.info("LLM triage: confirming %d candidates...", len(all_candidates))
        confirmed = []
        results_lock = threading.Lock()
        pbar = tqdm(total=len(all_candidates), desc="LLM triage", unit="cand")

        def worker(candidate):
            is_specific, confidence = classify_is_specific_ref(
                candidate["surface"], candidate["text_or_turn_text"], args.model
            )
            return candidate, is_specific, confidence

        try:
            with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                futures = {executor.submit(worker, c): c for c in all_candidates}
                for future in as_completed(futures):
                    try:
                        candidate, is_specific, confidence = future.result()
                    except Exception as e:
                        logger.warning("LLM triage error: %s", e)
                        stats["llm_errors"] += 1
                        pbar.update(1)
                        continue

                    if not is_specific:
                        candidate["triage_method"] = "llm"
                        candidate["confidence"] = confidence
                        with results_lock:
                            confirmed.append(candidate)
                        stats["llm_confirmed"] += 1
                    else:
                        stats["llm_rejected"] += 1

                    pbar.update(1)
        finally:
            pbar.close()

        all_candidates = confirmed

    stats["final_negatives"] = len(all_candidates)
    write_output_jsonl(out_path, all_candidates)
    logger.info("Wrote %d hard negatives to %s", len(all_candidates), out_path)

    print("\n=== HARD NEGATIVE MINING SUMMARY ===")
    print_summary(stats)


if __name__ == "__main__":
    main()
