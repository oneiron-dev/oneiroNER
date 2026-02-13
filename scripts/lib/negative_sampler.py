"""Frequency-weighted negative type sampler with holdout exclusion."""

import bisect
import itertools
import json
import math
import random
from pathlib import Path

HOLDOUT_PATH = Path(__file__).parent.parent.parent / "configs" / "zero_shot_holdout_types.json"
TYPE_FREQ_PATH = Path(__file__).parent.parent.parent / "data" / "processed" / "type_frequency_scan.json"


class NegativeSampler:
    def __init__(
        self,
        type_counts: dict[str, int] | None = None,
        holdout_types_file: str | Path | None = None,
    ):
        if holdout_types_file is None:
            holdout_types_file = HOLDOUT_PATH
        holdout_types_file = Path(holdout_types_file)

        self.holdout_strings: set[str] = set()
        if holdout_types_file.exists():
            with open(holdout_types_file) as f:
                holdout_data = json.load(f)
            for canonical, equivalences in holdout_data.items():
                self.holdout_strings.add(canonical)
                for eq in equivalences:
                    self.holdout_strings.add(eq)

        if type_counts is None:
            type_counts = self._load_default_counts()

        self.eligible_types: list[str] = []
        weights: list[float] = []
        for t, count in type_counts.items():
            if t not in self.holdout_strings and t:
                self.eligible_types.append(t)
                weights.append(math.sqrt(count))

        total = sum(weights) if weights else 1.0
        self.weights = [w / total for w in weights]
        self._cumweights = list(itertools.accumulate(self.weights))

    def _load_default_counts(self) -> dict[str, int]:
        merged: dict[str, int] = {}
        if TYPE_FREQ_PATH.exists():
            with open(TYPE_FREQ_PATH) as f:
                all_freqs = json.load(f)
            for dataset_counts in all_freqs.values():
                for t, c in dataset_counts.items():
                    merged[t] = merged.get(t, 0) + c
        return merged

    def sample(self, positive_types: set[str], n: int | None = None, rng: random.Random | None = None) -> list[str]:
        if rng is None:
            rng = random.Random()
        if n is None:
            n = rng.randint(2, 5)

        if not self._cumweights:
            return []

        chosen: set[str] = set()
        result: list[str] = []
        attempts = 0
        max_attempts = n * 20
        cum_total = self._cumweights[-1]

        while len(result) < n and attempts < max_attempts:
            r = rng.random() * cum_total
            idx = bisect.bisect_right(self._cumweights, r)
            if idx >= len(self.eligible_types):
                idx = len(self.eligible_types) - 1
            pick = self.eligible_types[idx]
            if pick not in positive_types and pick not in chosen:
                chosen.add(pick)
                result.append(pick)
            attempts += 1

        return result

    def is_holdout(self, type_name: str) -> bool:
        return type_name in self.holdout_strings
