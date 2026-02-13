"""Stratified 95/5 train/val split with zero-shot holdout handling."""

import json
import logging
import random
from collections import defaultdict
from pathlib import Path

from .schema import NerRecord
from .negative_sampler import NegativeSampler

logger = logging.getLogger(__name__)

HOLDOUT_PATH = Path(__file__).parent.parent.parent / "configs" / "zero_shot_holdout_types.json"


def load_holdout_types(path: str | Path | None = None) -> set[str]:
    if path is None:
        path = HOLDOUT_PATH
    path = Path(path)
    if not path.exists():
        return set()
    with open(path) as f:
        data = json.load(f)
    holdout = set()
    for canonical, equivalences in data.items():
        holdout.add(canonical)
        for eq in equivalences:
            holdout.add(eq)
    return holdout


def stratified_split(
    records: list[NerRecord],
    val_ratio: float = 0.05,
    seed: int = 42,
) -> tuple[list[NerRecord], list[NerRecord]]:
    rng = random.Random(seed)

    strata: dict[str, list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        primary_source = rec.provenance[0] if rec.provenance else rec.source
        key = f"{rec.language}|{primary_source}"
        strata[key].append(i)

    train_indices = []
    val_indices = []

    for key, indices in strata.items():
        rng.shuffle(indices)
        n_val = max(1, int(len(indices) * val_ratio))
        if len(indices) <= 2:
            train_indices.extend(indices)
            continue
        val_indices.extend(indices[:n_val])
        train_indices.extend(indices[n_val:])

    train = [records[i] for i in train_indices]
    val = [records[i] for i in val_indices]

    logger.info("Split: %d train, %d val (%.1f%%)", len(train), len(val),
                100 * len(val) / (len(train) + len(val)) if (len(train) + len(val)) > 0 else 0)
    return train, val


def apply_zero_shot_stripping(
    records: list[NerRecord],
    sampler: NegativeSampler,
    holdout_types: set[str] | None = None,
    seed: int = 42,
) -> tuple[list[NerRecord], list[NerRecord]]:
    if holdout_types is None:
        holdout_types = load_holdout_types()

    rng = random.Random(seed)
    train_out = []
    zero_shot_out = []

    for rec in records:
        kept_entities = [e for e in rec.entities if e["type"] not in holdout_types]
        stripped_entities = [e for e in rec.entities if e["type"] in holdout_types]

        if not stripped_entities:
            train_out.append(rec)
            continue

        if kept_entities:
            positive_types = set(e["type"] for e in kept_entities)
            negatives = sampler.sample(positive_types, rng=rng)
            rec.entities = kept_entities
            rec.query_types = sorted(positive_types | set(negatives))
            train_out.append(rec)
        else:
            holdout_positive = set(e["type"] for e in rec.entities)
            negatives = sampler.sample(holdout_positive, rng=rng)
            rec.split = "zero_shot_eval"
            rec.query_types = sorted(holdout_positive | set(negatives))
            zero_shot_out.append(rec)

    logger.info("Zero-shot stripping: %d train (kept/stripped), %d zero_shot_eval (all-holdout)",
                len(train_out), len(zero_shot_out))
    return train_out, zero_shot_out
