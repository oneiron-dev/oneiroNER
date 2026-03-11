"""Build mini-train / mini-eval subsets for autoresearch + setup worktree."""

import json
import logging
import os
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

MINI_TRAIN_SIZE = 5000
MINI_EVAL_SIZE = 1000
SEED = 42

WORKTREE_PATH = "../oneiron-ner-autoresearch"
WORKTREE_BRANCH = "autoresearch-ner"


def setup_worktree():
    wt = Path(WORKTREE_PATH)
    if wt.exists():
        logger.info("Worktree already exists at %s", wt)
        return str(wt.resolve())

    result = subprocess.run(
        ["git", "worktree", "add", "-b", WORKTREE_BRANCH, str(wt), "HEAD"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        if "already exists" in result.stderr:
            subprocess.run(
                ["git", "worktree", "add", str(wt), WORKTREE_BRANCH],
                capture_output=True, text=True, check=True,
            )
        else:
            logger.error("Failed to create worktree: %s", result.stderr)
            raise RuntimeError(result.stderr)

    logger.info("Created worktree at %s on branch %s", wt, WORKTREE_BRANCH)
    return str(wt.resolve())


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: list[dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Saved %d records to %s", len(records), path)


def stratified_sample(records: list[dict], n: int, seed: int = SEED) -> list[dict]:
    rng = random.Random(seed)

    by_bucket: dict[str, list[dict]] = {"gold": [], "silver_en": [], "silver_ml": []}
    for rec in records:
        bucket = rec.get("bucket", "gold")
        if bucket not in by_bucket:
            bucket = "gold"
        by_bucket[bucket].append(rec)

    bucket_dist = {k: len(v) / len(records) for k, v in by_bucket.items() if v}

    sampled = []
    for bucket, ratio in bucket_dist.items():
        pool = by_bucket[bucket]
        k = max(1, int(n * ratio))
        k = min(k, len(pool))
        sampled.extend(rng.sample(pool, k))

    rng.shuffle(sampled)
    return sampled[:n]


def verify_distribution(full: list[dict], mini: list[dict], tolerance: float = 0.05):
    def type_dist(records):
        counts = Counter()
        for rec in records:
            for ent in rec.get("entities", []):
                counts[ent.get("type", "UNKNOWN")] += 1
        total = sum(counts.values()) or 1
        return {t: c / total for t, c in counts.items()}

    full_dist = type_dist(full)
    mini_dist = type_dist(mini)

    violations = []
    for t, full_pct in full_dist.items():
        mini_pct = mini_dist.get(t, 0.0)
        if abs(full_pct - mini_pct) > tolerance:
            violations.append(f"{t}: full={full_pct:.3f} mini={mini_pct:.3f}")

    if violations:
        logger.warning("Distribution violations (>%.0f%%): %s",
                       tolerance * 100, "; ".join(violations))
    else:
        logger.info("Mini subset distribution within ±%.0f%% of full", tolerance * 100)

    return len(violations) == 0


def build_mini_subsets(
    train_path: str = "data/processed/train.jsonl",
    val_path: str = "data/processed/val.jsonl",
    output_dir: str = "data/autoresearch",
    train_size: int = MINI_TRAIN_SIZE,
    eval_size: int = MINI_EVAL_SIZE,
):
    logger.info("Building mini subsets: train=%d, eval=%d", train_size, eval_size)

    train_records = load_jsonl(train_path)
    val_records = load_jsonl(val_path)

    mini_train = stratified_sample(train_records, train_size)
    mini_eval = stratified_sample(val_records, eval_size)

    train_out = os.path.join(output_dir, "mini_train.jsonl")
    eval_out = os.path.join(output_dir, "mini_eval.jsonl")
    save_jsonl(mini_train, train_out)
    save_jsonl(mini_eval, eval_out)

    verify_distribution(train_records, mini_train)
    verify_distribution(val_records, mini_eval)

    return train_out, eval_out


def init_results_tsv(path: str = "results.tsv"):
    from research.score import TSV_HEADER
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(TSV_HEADER + "\n")
        logger.info("Created %s with header", path)


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    logger.info("=== Autoresearch preparation ===")

    wt_path = setup_worktree()
    logger.info("Worktree ready at %s", wt_path)

    train_out, eval_out = build_mini_subsets()
    logger.info("Mini train: %s", train_out)
    logger.info("Mini eval:  %s", eval_out)

    init_results_tsv()
    logger.info("=== Preparation complete ===")


if __name__ == "__main__":
    main()
