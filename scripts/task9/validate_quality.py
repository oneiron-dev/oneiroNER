#!/usr/bin/env python3
"""Quality validation for Task 8 + Task 9 silver-labeled data.

Samples records, re-annotates via Gemini, computes span F1 + Cohen's kappa,
and writes a quality report + manual review sample.
"""

import argparse
import json
import logging
import random
import re
import subprocess
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.schema import record_from_jsonl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_REPORT = PROCESSED_DIR / "quality_report.json"
OUTPUT_SAMPLE = PROCESSED_DIR / "validation_sample_500.jsonl"

NER_PROMPT = """\
You are an expert NER annotator. Think step-by-step internally before outputting labels.

{text_block}

Entity types:
- PERSON: named individuals ("Sarah", "Dr. Chen")
- PLACE: locations ("Tokyo", "the park")
- ORG: organizations ("Google")
- DATE: temporal expressions. Subtypes: Day, Week, Month, Season, Year, Decade, Relative, Range
- EVENT: named events ("Christmas", "the wedding")
- RELATIONSHIP_REF: terms referring to a specific person via relationship role. Subtypes: Family, Romantic, Friend, Professional, Acquaintance
- EMOTION: emotional states ("happy", "anxious")
- GOAL: intentions/desires ("I want to travel")
- ACTIVITY: activities being done ("studying", "cooking")

Offsets: 0-indexed Unicode chars, end-exclusive, Python slicing (text[start:end] == surface).

Return ONLY a JSON array:
[{{"surface": "my mom", "type": "RELATIONSHIP_REF/Family", "start": 7, "end": 13}}]
{turn_instruction}\
If none found, return []."""

F1_TARGET = 0.85
KAPPA_TARGET = 0.70

sem_gemini = threading.Semaphore(5)


def call_gemini(prompt: str) -> str | None:
    with sem_gemini:
        try:
            r = subprocess.run(
                ["gemini", "-p", "", "--yolo", "-o", "text"],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if r.returncode != 0:
                logger.debug("gemini failed (rc=%d): %s", r.returncode, r.stderr[:200])
                return None
            return r.stdout.strip()
        except subprocess.TimeoutExpired:
            logger.debug("gemini timed out")
            return None
        except Exception as e:
            logger.debug("gemini error: %s", e)
            return None


def clean_llm_output(raw: str) -> list[dict] | None:
    cleaned = re.sub(r"^```(?:json)?\n?", "", raw.strip())
    cleaned = re.sub(r"\n?```$", "", cleaned)
    match = re.search(r"\[[\s\S]*\]", cleaned)
    if match:
        try:
            arr = json.loads(match.group())
            if isinstance(arr, list):
                return arr
        except json.JSONDecodeError:
            pass
    return None


def build_prompt(record: dict) -> str:
    if record.get("format") == "conversation":
        lines = []
        for i, turn in enumerate(record["turns"]):
            lines.append(f"[Turn {i}] {turn['speaker']}: {turn['text']}")
        text_block = "\n".join(lines)
        turn_instruction = 'For conversation, add "turn_index": N (0-indexed). Offsets are relative to turn text.\n'
    else:
        text_block = record["text"]
        turn_instruction = ""
    return NER_PROMPT.format(text_block=text_block, turn_instruction=turn_instruction)


def reannotate(record: dict) -> list[dict] | None:
    prompt = build_prompt(record)
    raw = call_gemini(prompt)
    if raw is None:
        return None
    entities = clean_llm_output(raw)
    if entities is None:
        return None
    return [e for e in entities if isinstance(e, dict) and "surface" in e and "type" in e]


def base_type(t: str) -> str:
    return t.split("/")[0].split(" -> ")[0]


def _span_set(entities: list[dict]) -> set[tuple]:
    spans = set()
    for e in entities:
        ti = e.get("turn_index", -1)
        spans.add((e.get("surface", ""), e.get("start", 0), e.get("end", 0), base_type(e.get("type", "")), ti))
    return spans


def _span_overlap(a: tuple, b: tuple) -> bool:
    if a[4] != b[4]:
        return False
    return a[1] < b[2] and b[1] < a[2] and base_type(a[3]) == base_type(b[3])


def compute_span_f1(gold_entities: list[dict], pred_entities: list[dict]) -> dict:
    gold_spans = _span_set(gold_entities)
    pred_spans = _span_set(pred_entities)

    exact_tp = len(gold_spans & pred_spans)
    exact_p = exact_tp / len(pred_spans) if pred_spans else 0.0
    exact_r = exact_tp / len(gold_spans) if gold_spans else 0.0
    exact_f1 = 2 * exact_p * exact_r / (exact_p + exact_r) if (exact_p + exact_r) > 0 else 0.0

    overlap_tp = 0
    for g in gold_spans:
        for p in pred_spans:
            if _span_overlap(g, p):
                overlap_tp += 1
                break
    overlap_r = overlap_tp / len(gold_spans) if gold_spans else 0.0
    overlap_p_count = 0
    for p in pred_spans:
        for g in gold_spans:
            if _span_overlap(p, g):
                overlap_p_count += 1
                break
    overlap_p = overlap_p_count / len(pred_spans) if pred_spans else 0.0
    overlap_f1 = 2 * overlap_p * overlap_r / (overlap_p + overlap_r) if (overlap_p + overlap_r) > 0 else 0.0

    return {
        "exact_precision": round(exact_p, 4),
        "exact_recall": round(exact_r, 4),
        "exact_f1": round(exact_f1, 4),
        "overlap_precision": round(overlap_p, 4),
        "overlap_recall": round(overlap_r, 4),
        "overlap_f1": round(overlap_f1, 4),
        "gold_count": len(gold_spans),
        "pred_count": len(pred_spans),
    }


def compute_cohens_kappa(gold_records: list[dict], pred_records: list[list[dict]]) -> dict:
    all_types = {"PERSON", "PLACE", "ORG", "DATE", "EVENT",
                 "RELATIONSHIP_REF", "EMOTION", "GOAL", "ACTIVITY"}

    kappa_per_type = {}
    for t in sorted(all_types):
        gold_binary = []
        pred_binary = []
        for rec, pred_ents in zip(gold_records, pred_records):
            gold_types = {base_type(e.get("type", "")) for e in rec.get("entities", [])}
            pred_types = {base_type(e.get("type", "")) for e in pred_ents}
            gold_binary.append(1 if t in gold_types else 0)
            pred_binary.append(1 if t in pred_types else 0)

        n = len(gold_binary)
        if n == 0:
            kappa_per_type[t] = 0.0
            continue

        agree = sum(1 for g, p in zip(gold_binary, pred_binary) if g == p)
        p_o = agree / n

        g1 = sum(gold_binary)
        p1 = sum(pred_binary)
        p_e = (g1 * p1 + (n - g1) * (n - p1)) / (n * n)

        if p_e == 1.0:
            kappa_per_type[t] = 1.0
        else:
            kappa_per_type[t] = round((p_o - p_e) / (1 - p_e), 4)

    return kappa_per_type


def load_all_records(sample_size: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    records_by_key: dict[str, list[dict]] = defaultdict(list)

    patterns = ["task8_*.jsonl", "task9_silver_*.jsonl"]
    for pattern in patterns:
        for path in sorted(PROCESSED_DIR.glob(pattern)):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    source = rec.get("source", "unknown")
                    ent_types = {base_type(e.get("type", "")) for e in rec.get("entities", [])}
                    for t in ent_types:
                        records_by_key[f"{source}:{t}"].append(rec)
                    if not ent_types:
                        records_by_key[f"{source}:NONE"].append(rec)

    if not records_by_key:
        logger.warning("No records found in %s matching task8_*/task9_silver_*", PROCESSED_DIR)
        return []

    per_key = max(1, sample_size // len(records_by_key))
    sampled_ids = set()
    sampled = []

    for key in sorted(records_by_key.keys()):
        pool = records_by_key[key]
        rng.shuffle(pool)
        for rec in pool[:per_key]:
            sid = rec.get("source_id", id(rec))
            if sid not in sampled_ids:
                sampled_ids.add(sid)
                sampled.append(rec)

    rng.shuffle(sampled)
    return sampled[:sample_size]


def main():
    parser = argparse.ArgumentParser(description="Quality validation for silver-labeled data")
    parser.add_argument("--sample-size", type=int, default=20000, help="Number of records to validate")
    parser.add_argument("--review-size", type=int, default=500, help="Manual review sample size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--concurrency", type=int, default=10, help="Gemini concurrency")
    parser.add_argument("--dry-run", action="store_true", help="Load and sample without LLM calls")
    args = parser.parse_args()

    global sem_gemini
    sem_gemini = threading.Semaphore(args.concurrency)

    logger.info("Loading records (sample_size=%d)...", args.sample_size)
    records = load_all_records(args.sample_size, args.seed)
    if not records:
        logger.error("No records found. Ensure task8/task9 output exists in %s", PROCESSED_DIR)
        sys.exit(1)
    logger.info("Loaded %d records for validation", len(records))

    rng = random.Random(args.seed + 1)
    review_sample = records[:args.review_size]

    if args.dry_run:
        logger.info("Dry run: %d records loaded, %d for review", len(records), len(review_sample))
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_SAMPLE, "w") as f:
            for rec in review_sample:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info("Wrote review sample to %s", OUTPUT_SAMPLE)
        return

    logger.info("Re-annotating %d records via Gemini...", len(records))
    pred_entities: list[list[dict] | None] = [None] * len(records)
    completed = 0
    failed = 0

    def worker(idx: int) -> None:
        nonlocal completed, failed
        result = reannotate(records[idx])
        pred_entities[idx] = result
        completed += 1
        if result is None:
            failed += 1
        if completed % 500 == 0:
            logger.info("Progress: %d/%d (failed: %d)", completed, len(records), failed)

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(worker, i): i for i in range(len(records))}
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                logger.error("Worker error: %s", e)

    valid_pairs = [(records[i], pred_entities[i]) for i in range(len(records)) if pred_entities[i] is not None]
    logger.info("Valid re-annotations: %d/%d", len(valid_pairs), len(records))

    if not valid_pairs:
        logger.error("No valid re-annotations. Cannot compute metrics.")
        sys.exit(1)

    valid_records = [p[0] for p in valid_pairs]
    valid_preds = [p[1] for p in valid_pairs]

    overall_metrics = compute_span_f1(
        [e for r in valid_records for e in r.get("entities", [])],
        [e for p in valid_preds for e in p],
    )

    per_type_f1: dict[str, dict] = {}
    all_types = {"PERSON", "PLACE", "ORG", "DATE", "EVENT",
                 "RELATIONSHIP_REF", "EMOTION", "GOAL", "ACTIVITY"}
    for t in sorted(all_types):
        gold_t = [e for r in valid_records for e in r.get("entities", []) if base_type(e.get("type", "")) == t]
        pred_t = [e for p in valid_preds for e in p if base_type(e.get("type", "")) == t]
        if gold_t or pred_t:
            per_type_f1[t] = compute_span_f1(gold_t, pred_t)

    kappa = compute_cohens_kappa(valid_records, valid_preds)

    per_source: dict[str, dict] = {}
    source_groups: dict[str, list[int]] = defaultdict(list)
    for i, rec in enumerate(valid_records):
        source_groups[rec.get("source", "unknown")].append(i)
    for source, indices in sorted(source_groups.items()):
        src_gold = [e for i in indices for e in valid_records[i].get("entities", [])]
        src_pred = [e for i in indices for e in valid_preds[i]]
        per_source[source] = {
            "count": len(indices),
            "metrics": compute_span_f1(src_gold, src_pred),
        }

    pass_fail = {}
    for t, metrics in per_type_f1.items():
        pass_fail[t] = {
            "exact_f1": "PASS" if metrics["exact_f1"] >= F1_TARGET else "FAIL",
            "kappa": "PASS" if kappa.get(t, 0) >= KAPPA_TARGET else "FAIL",
        }

    report = {
        "total_sampled": len(records),
        "total_reannotated": len(valid_pairs),
        "failed_reannotation": failed,
        "overall": overall_metrics,
        "per_type_f1": per_type_f1,
        "cohens_kappa": kappa,
        "per_source": per_source,
        "pass_fail": pass_fail,
        "thresholds": {"f1": F1_TARGET, "kappa": KAPPA_TARGET},
    }

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_REPORT, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Report written to %s", OUTPUT_REPORT)

    with open(OUTPUT_SAMPLE, "w") as f:
        for rec in review_sample:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Review sample (%d records) written to %s", len(review_sample), OUTPUT_SAMPLE)

    print("\n" + "=" * 60)
    print("QUALITY VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Records: {len(valid_pairs)}/{len(records)} successfully re-annotated")
    print(f"\nOverall exact F1: {overall_metrics['exact_f1']:.4f}")
    print(f"Overall overlap F1: {overall_metrics['overlap_f1']:.4f}")
    print(f"\nPer-type results (target: exact_f1 >= {F1_TARGET}, kappa >= {KAPPA_TARGET}):")
    for t in sorted(per_type_f1.keys()):
        m = per_type_f1[t]
        k = kappa.get(t, 0)
        pf = pass_fail[t]
        status = "PASS" if pf["exact_f1"] == "PASS" and pf["kappa"] == "PASS" else "FAIL"
        print(f"  {t:20s}  F1={m['exact_f1']:.3f} ({pf['exact_f1']})  kappa={k:.3f} ({pf['kappa']})  [{status}]")

    all_pass = all(
        pf["exact_f1"] == "PASS" and pf["kappa"] == "PASS"
        for pf in pass_fail.values()
    )
    print(f"\n{'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
