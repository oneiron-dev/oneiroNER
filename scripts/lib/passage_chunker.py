"""Chunk long passages on sentence boundaries for LLM annotation."""

import re

_SENTENCE_RE = re.compile(r"(?<=\. )|(?<=。)|(?<=\n)")


def chunk_passage(
    text: str, max_chars: int = 2000, overlap: int = 200
) -> list[tuple[str, int]]:
    if len(text) <= max_chars:
        return [(text, 0)]

    boundaries = [0]
    for m in _SENTENCE_RE.finditer(text):
        boundaries.append(m.start())
    boundaries.append(len(text))

    chunks: list[tuple[str, int]] = []
    start = 0

    while start < len(text):
        end = start + max_chars
        if end >= len(text):
            chunks.append((text[start:], start))
            break

        best = start
        for b in boundaries:
            if b <= start:
                continue
            if b <= end:
                best = b
            else:
                break

        if best <= start:
            best = end

        chunks.append((text[start:best], start))

        next_start = best - overlap
        if next_start <= start:
            next_start = best
        start = next_start

    return chunks
