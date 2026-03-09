#!/usr/bin/env python3
"""Backfill bare RELATIONSHIP_REF entities with subtypes.

Stage A: rule-based lexicon + specificity gate
Stage B: LLM classification for bare/unresolved spans
"""

import argparse
import copy
import json
import logging
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (
    BASE_DIR,
    entity_key,
    load_silver_records,
    make_provenance,
    print_summary,
    sample_review,
    specificity_gate,
    validate_offset,
    write_output_jsonl,
)
from _lexicons import classify_rel_by_lexicon
from _llm_classify import DEFAULT_MODEL, classify_rel_subtype

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

MAX_RETRIES = 3
BACKOFF_BASE = 30

checkpoint_lock = threading.Lock()


def extract_context(record, entity, window=200):
    if record.get("format") == "conversation" or record.get("turns"):
        turn_text = record["turns"][entity["turn_index"]]["text"]
        start = max(0, entity["start"] - window // 2)
        end = min(len(turn_text), entity["end"] + window // 2)
        return turn_text[start:end]
    text = record["text"]
    start = max(0, entity["start"] - window // 2)
    end = min(len(text), entity["end"] + window // 2)
    return text[start:end]


def load_checkpoint(cp_path):
    if cp_path.exists():
        with open(cp_path) as f:
            return {tuple(k) for k in json.load(f)}
    return set()


def save_checkpoint(cp_path, processed_keys):
    with checkpoint_lock:
        cp_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cp_path, "w") as f:
            json.dump([list(k) for k in sorted(processed_keys, key=str)], f)


def classify_with_retry(surface, context, model):
    for attempt in range(MAX_RETRIES + 1):
        try:
            return classify_rel_subtype(surface, context, model)
        except Exception as e:
            err = str(e).lower()
            if ("rate" in err or "limit" in err or "429" in err or "quota" in err) and attempt < MAX_RETRIES:
                wait = BACKOFF_BASE * (2 ** attempt)
                logger.warning("Rate limited (attempt %d/%d), waiting %ds...", attempt + 1, MAX_RETRIES + 1, wait)
                time.sleep(wait)
                continue
            if attempt < MAX_RETRIES:
                logger.debug("LLM error (attempt %d): %s", attempt + 1, e)
                time.sleep(2)
                continue
            logger.warning("LLM classification failed after retries: %s", e)
            return None, 0.0, False


def main():
    parser = argparse.ArgumentParser(description="Backfill bare RELATIONSHIP_REF entities with subtypes")
    parser.add_argument("--input-glob", default="data/processed/task9_silver_*.jsonl")
    parser.add_argument("--output", default="data/processed/task9_5_rel_backfill.jsonl")
    parser.add_argument("--review-file", default="data/processed/task9_5_rel_review.jsonl")
    parser.add_argument("--dry-run", action="store_true", help="Count only, no LLM calls, no writes")
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None, help="Record limit for testing")
    parser.add_argument("--min-context", type=int, default=200, help="Context window chars around entity")
    args = parser.parse_args()

    glob_path = BASE_DIR / args.input_glob
    out_path = BASE_DIR / args.output
    review_path = BASE_DIR / args.review_file
    cp_path = BASE_DIR / "data" / "processed" / ".checkpoints" / "task9_5_rel.json"

    # --- Pass 1: scan and apply rule-based classification ---
    logger.info("Pass 1: scanning records...")
    all_records = []
    record_modifications = {}  # record_idx -> {ent_idx -> {subtype, method, model, confidence, generic_flag}}
    llm_queue = []  # (record_idx, ent_idx, surface, context, ekey)
    stats = Counter()

    for filepath, line_num, record in load_silver_records(glob_path):
        if args.limit and len(all_records) >= args.limit:
            break
        rec_idx = len(all_records)
        all_records.append(record)
        stats["total_records"] += 1

        has_bare = False
        for ent_idx, ent in enumerate(record.get("entities", [])):
            if ent.get("type") != "RELATIONSHIP_REF":
                continue
            has_bare = True
            stats["bare_rel_entities"] += 1

            surface = ent["surface"]
            ekey = entity_key(ent, record.get("source", ""), record.get("source_id", ""))
            context = extract_context(record, ent, window=args.min_context)

            candidate = classify_rel_by_lexicon(surface)
            if candidate:
                if specificity_gate(surface, context):
                    record_modifications.setdefault(rec_idx, {})[ent_idx] = {
                        "subtype": candidate,
                        "method": "rule",
                        "model": None,
                        "confidence": 1.0,
                        "generic_flag": False,
                    }
                    stats["stage_a_promoted"] += 1
                    stats[f"stage_a_subtype:{candidate}"] += 1
                else:
                    llm_queue.append((rec_idx, ent_idx, surface, context, ekey))
                    stats["stage_a_gate_failed"] += 1
            else:
                llm_queue.append((rec_idx, ent_idx, surface, context, ekey))

        if has_bare:
            stats["records_with_bare_rel"] += 1

    logger.info("Pass 1 complete: %d records, %d bare REL entities, %d auto-promoted, %d gate-failed, %d need LLM",
                stats["total_records"], stats["bare_rel_entities"], stats["stage_a_promoted"],
                stats["stage_a_gate_failed"], len(llm_queue))

    if args.dry_run:
        stats["stage_b_pending"] = len(llm_queue)
        print("\n=== DRY RUN SUMMARY ===")
        print_summary(stats)
        return

    # --- Pass 2: LLM classification ---
    if llm_queue:
        processed_keys = load_checkpoint(cp_path)
        pending = [(r, e, s, c, k) for r, e, s, c, k in llm_queue if k not in processed_keys]
        logger.info("Pass 2: %d entities need LLM (%d already checkpointed)", len(pending), len(llm_queue) - len(pending))

        if pending:
            pbar = tqdm(total=len(pending), desc="LLM classify", unit="ent")
            results_lock = threading.Lock()

            def worker(item):
                rec_idx, ent_idx, surface, context, ekey = item
                subtype, confidence, generic_flag = classify_with_retry(surface, context, args.model)
                return rec_idx, ent_idx, subtype, confidence, generic_flag, ekey

            try:
                with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                    futures = {executor.submit(worker, item): item for item in pending}
                    for future in as_completed(futures):
                        try:
                            rec_idx, ent_idx, subtype, confidence, generic_flag, ekey = future.result()
                        except Exception as e:
                            logger.warning("Worker error: %s", e)
                            stats["llm_failures"] += 1
                            pbar.update(1)
                            continue

                        if subtype is None:
                            stats["llm_failures"] += 1
                        else:
                            with results_lock:
                                record_modifications.setdefault(rec_idx, {})[ent_idx] = {
                                    "subtype": subtype,
                                    "method": "llm",
                                    "model": args.model,
                                    "confidence": confidence,
                                    "generic_flag": generic_flag,
                                }
                            stats["stage_b_resolved"] += 1
                            if generic_flag:
                                stats["generic_detected"] += 1
                            else:
                                stats[f"stage_b_subtype:{subtype}"] += 1

                        with results_lock:
                            processed_keys.add(ekey)
                        if stats["stage_b_resolved"] % 50 == 0:
                            save_checkpoint(cp_path, processed_keys)

                        pbar.update(1)
            finally:
                pbar.close()
                save_checkpoint(cp_path, processed_keys)

    # --- Pass 3: emit modified records ---
    logger.info("Pass 3: writing output...")
    output_records = []
    changed_entities = []

    for rec_idx, mods in sorted(record_modifications.items()):
        record = copy.deepcopy(all_records[rec_idx])
        entities = record.get("entities", [])

        for ent_idx, mod in mods.items():
            ent = entities[ent_idx]
            ent["original_type"] = ent.get("original_type", ent["type"])

            if mod["generic_flag"]:
                pass  # keep type as RELATIONSHIP_REF
            else:
                ent["type"] = mod["subtype"]

            ent["generic_flag"] = mod["generic_flag"]
            ent["specific_ref_confidence"] = mod["confidence"]
            ekey = entity_key(ent, record.get("source", ""), record.get("source_id", ""))
            ent["entity_key"] = list(ekey)
            prov = make_provenance("task9_5_rel_backfill", mod["method"], mod["model"], mod["confidence"])
            ent.update(prov)

            changed_entities.append({
                "surface": ent["surface"],
                "type": ent["type"],
                "source": record.get("source", ""),
                "source_id": record.get("source_id", ""),
                "subtype_method": mod["method"],
                "subtype_confidence": mod["confidence"],
                "generic_flag": mod["generic_flag"],
            })

        for ent in entities:
            if not validate_offset(record, ent):
                logger.error("Offset mismatch: source=%s id=%s surface=%r",
                             record.get("source"), record.get("source_id"), ent.get("surface"))
                break
        else:
            output_records.append(record)

    write_output_jsonl(out_path, output_records)
    logger.info("Wrote %d records to %s", len(output_records), out_path)

    review_items = sample_review(changed_entities)
    write_output_jsonl(review_path, review_items)
    logger.info("Wrote %d review items to %s", len(review_items), review_path)

    total_bare = stats["bare_rel_entities"]
    stats["remaining_bare"] = total_bare - stats["stage_a_promoted"] - stats.get("stage_b_resolved", 0)
    stats["output_records"] = len(output_records)

    pct = lambda n, d: f"{n / d * 100:.1f}%" if d else "0%"
    print("\n=== RELATIONSHIP_REF BACKFILL SUMMARY ===")
    print(f"  Total records scanned          {stats['total_records']}")
    print(f"  Records with bare REL          {stats['records_with_bare_rel']}")
    print(f"  Total bare REL entities        {total_bare}")
    print(f"  Stage A auto-promoted (rule)   {stats['stage_a_promoted']} ({pct(stats['stage_a_promoted'], total_bare)})")
    print(f"  Stage A -> LLM (gate failed)   {stats['stage_a_gate_failed']}")
    print(f"  Stage B resolved (LLM)         {stats.get('stage_b_resolved', 0)} ({pct(stats.get('stage_b_resolved', 0), total_bare)})")
    print(f"  Generic detected               {stats.get('generic_detected', 0)} ({pct(stats.get('generic_detected', 0), total_bare)})")
    print(f"  Remaining bare                 {stats['remaining_bare']} ({pct(stats['remaining_bare'], total_bare)})")
    print(f"  LLM failures                   {stats.get('llm_failures', 0)}")
    print(f"  Output records written         {stats['output_records']}")
    print("\n  Stage A subtype distribution:")
    for key in sorted(k for k in stats if k.startswith("stage_a_subtype:")):
        print(f"    {key.replace('stage_a_subtype:', '').ljust(30)} {stats[key]}")
    print("\n  Stage B subtype distribution:")
    for key in sorted(k for k in stats if k.startswith("stage_b_subtype:")):
        print(f"    {key.replace('stage_b_subtype:', '').ljust(30)} {stats[key]}")


if __name__ == "__main__":
    main()
