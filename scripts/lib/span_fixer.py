"""Verify and fix entity spans with fallback chain.

Fallback order:
1. Exact match: text[start:end] == surface
2. Nearest occurrence: find surface near model-provided start
3. Fuzzy match: difflib.get_close_matches on surrounding window
4. Drop + log
"""

import difflib
import logging
import re

logger = logging.getLogger(__name__)


def _find_nearest(text: str, surface: str, hint_start: int) -> tuple[int, int] | None:
    pattern = re.escape(surface)
    matches = [(m.start(), m.end()) for m in re.finditer(pattern, text)]
    if not matches:
        return None
    return min(matches, key=lambda m: abs(m[0] - hint_start))


def _fuzzy_match(text: str, surface: str, hint_start: int, window: int = 200) -> tuple[int, int] | None:
    lo = max(0, hint_start - window)
    hi = min(len(text), hint_start + len(surface) + window)
    region = text[lo:hi]

    n = len(surface)
    candidates = [region[i:i + n] for i in range(len(region) - n + 1)]
    if not candidates:
        return None

    close = difflib.get_close_matches(surface, candidates, n=1, cutoff=0.8)
    if not close:
        return None

    idx = region.index(close[0])
    return lo + idx, lo + idx + len(close[0])


def verify_and_fix_spans(
    entities: list[dict],
    text: str,
    turns: list[dict] | None = None,
) -> tuple[list[dict], dict]:
    stats = {"exact": 0, "nearest": 0, "fuzzy": 0, "dropped": 0}
    fixed: list[dict] = []

    for ent in entities:
        surface = ent["surface"]
        start = ent["start"]
        end = ent["end"]

        if turns is not None:
            ti = ent.get("turn_index", 0)
            if ti < 0 or ti >= len(turns):
                stats["dropped"] += 1
                logger.debug("Dropped (bad turn_index %d): %s", ti, surface)
                continue
            ctx = turns[ti]["text"]
        else:
            ctx = text

        if start >= 0 and end <= len(ctx) and ctx[start:end] == surface:
            stats["exact"] += 1
            fixed.append(dict(ent))
            continue

        result = _find_nearest(ctx, surface, start)
        if result:
            stats["nearest"] += 1
            e = dict(ent)
            e["start"], e["end"] = result
            fixed.append(e)
            continue

        result = _fuzzy_match(ctx, surface, start)
        if result:
            stats["fuzzy"] += 1
            e = dict(ent)
            e["start"], e["end"] = result
            e["surface"] = ctx[result[0]:result[1]]
            fixed.append(e)
            continue

        stats["dropped"] += 1
        logger.debug("Dropped (unfixable): '%s' at [%d:%d]", surface, start, end)

    return fixed, stats
