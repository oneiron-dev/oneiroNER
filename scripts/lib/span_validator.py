"""Span validation and off-by-one fixing."""

import unicodedata


def validate_span(text: str, surface: str, start: int, end: int) -> bool:
    return text[start:end] == surface


def validate_and_fix(text: str, surface: str, start: int, end: int) -> tuple[int, int, bool]:
    if text[start:end] == surface:
        return start, end, False

    # Try off-by-one fixes
    for s, e in [(start, end + 1), (start - 1, end), (start + 1, end), (start, end - 1)]:
        if s >= 0 and e <= len(text) and text[s:e] == surface:
            return s, e, True

    # Try NFKC normalization only if length is preserved
    norm_text = unicodedata.normalize("NFKC", text)
    if len(norm_text) == len(text):
        norm_surface = unicodedata.normalize("NFKC", surface)
        if norm_text[start:end] == norm_surface:
            return start, end, True
        for s, e in [(start, end + 1), (start - 1, end)]:
            if s >= 0 and e <= len(norm_text) and norm_text[s:e] == norm_surface:
                return s, e, True

    raise ValueError(
        f"Unfixable span: text[{start}:{end}]='{text[start:end]}' != '{surface}'"
    )
