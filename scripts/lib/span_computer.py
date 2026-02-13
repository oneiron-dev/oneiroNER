"""Compute character spans via string matching for entities without position offsets."""

import re
import logging
import unicodedata
from collections import defaultdict

logger = logging.getLogger(__name__)


def compute_span(text: str, surface: str, use_word_boundary: bool = False) -> tuple[int, int] | None:
    if use_word_boundary:
        pattern = r'\b' + re.escape(surface) + r'\b'
    else:
        pattern = re.escape(surface)

    m = re.search(pattern, text)
    if m:
        return m.start(), m.end()

    norm_surface = unicodedata.normalize("NFKC", surface)
    if norm_surface != surface:
        if use_word_boundary:
            pattern = r'\b' + re.escape(norm_surface) + r'\b'
        else:
            pattern = re.escape(norm_surface)
        m = re.search(pattern, text)
        if m:
            return m.start(), m.end()

    norm_text = unicodedata.normalize("NFKC", text)
    if len(norm_text) == len(text):
        norm_s = unicodedata.normalize("NFKC", surface)
        if use_word_boundary:
            pattern = r'\b' + re.escape(norm_s) + r'\b'
        else:
            pattern = re.escape(norm_s)
        m = re.search(pattern, norm_text)
        if m:
            return m.start(), m.end()

    return None


def compute_spans_batch(
    text: str,
    entities: list[dict],
    use_word_boundary: bool = False,
    ambiguous_log: list | None = None,
) -> list[dict]:
    occurrence_map: dict[str, list[tuple[int, int]]] = {}

    for ent in entities:
        surface = ent.get("surface") or ent.get("name") or ent.get("entity_text", "")
        if not surface or surface in occurrence_map:
            continue

        if use_word_boundary:
            pattern = r'\b' + re.escape(surface) + r'\b'
        else:
            pattern = re.escape(surface)

        matches = [(m.start(), m.end()) for m in re.finditer(pattern, text)]

        if not matches:
            norm_surface = unicodedata.normalize("NFKC", surface)
            if norm_surface != surface:
                if use_word_boundary:
                    pattern = r'\b' + re.escape(norm_surface) + r'\b'
                else:
                    pattern = re.escape(norm_surface)
                matches = [(m.start(), m.end()) for m in re.finditer(pattern, text)]

            if not matches:
                norm_text = unicodedata.normalize("NFKC", text)
                if len(norm_text) == len(text):
                    norm_s = unicodedata.normalize("NFKC", surface)
                    if use_word_boundary:
                        pattern = r'\b' + re.escape(norm_s) + r'\b'
                    else:
                        pattern = re.escape(norm_s)
                    matches = [(m.start(), m.end()) for m in re.finditer(pattern, norm_text)]

        occurrence_map[surface] = matches

        if len(matches) > 1 and ambiguous_log is not None:
            ambiguous_log.append({
                "text_preview": text[:100],
                "surface": surface,
                "occurrences": len(matches),
            })

    used: dict[str, int] = defaultdict(int)
    result = []

    def first_occ(ent):
        surface = ent.get("surface") or ent.get("name") or ent.get("entity_text", "")
        occs = occurrence_map.get(surface, [])
        return occs[0][0] if occs else float('inf')

    for ent in sorted(entities, key=first_occ):
        surface = ent.get("surface") or ent.get("name") or ent.get("entity_text", "")
        occs = occurrence_map.get(surface, [])
        idx = used[surface]
        if idx < len(occs):
            ent_copy = dict(ent)
            ent_copy["start"] = occs[idx][0]
            ent_copy["end"] = occs[idx][1]
            used[surface] = idx + 1
            result.append(ent_copy)
        else:
            logger.debug("No match for surface '%s' in text", surface)

    return result
