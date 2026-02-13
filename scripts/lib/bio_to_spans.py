"""BIO tag sequence → character-level spans with detokenization."""

import re

CJK_LANGS = {"cmn", "jap", "jpn", "ja", "zh", "kor", "ko", "tha", "th", "yue", "wuu", "lzh", "nan"}

CJK_RANGE = (
    r'[\u2E80-\u9FFF'
    r'\uF900-\uFAFF'
    r'\U00020000-\U0002A6DF'
    r'\U0002A700-\U0002B73F'
    r'\U0002B740-\U0002B81F'
    r'\U0002B820-\U0002CEAF'
    r'\U0002CEB0-\U0002EBEF'
    r'\U00030000-\U0003134F'
    r'\u3040-\u309F'
    r'\u30A0-\u30FF'
    r'\u31F0-\u31FF'
    r'\uAC00-\uD7AF'
    r'\u1100-\u11FF'
    r'\u3130-\u318F'
    r'\uA960-\uA97F'
    r'\uD7B0-\uD7FF]'
)

_CJK_RE = re.compile(CJK_RANGE)


def is_cjk_char(ch: str) -> bool:
    return bool(_CJK_RE.match(ch))


def is_cjk_language(lang: str) -> bool:
    return lang.lower() in CJK_LANGS


def smart_cjk_join(tokens: list[str]) -> tuple[str, list[tuple[int, int]]]:
    if not tokens:
        return "", []

    parts = []
    offsets = []
    pos = 0

    for i, token in enumerate(tokens):
        if i > 0 and token:
            prev_token = tokens[i - 1]
            if prev_token:
                last_char = prev_token[-1]
                first_char = token[0]
                need_space = not is_cjk_char(last_char) and not is_cjk_char(first_char)
                if need_space and not _is_no_space_before(first_char) and not _is_no_space_after(last_char):
                    parts.append(" ")
                    pos += 1

        start = pos
        parts.append(token)
        pos += len(token)
        offsets.append((start, pos))

    return "".join(parts), offsets


def _is_no_space_before(ch: str) -> bool:
    return ch in '.,!?;:)]}%\'">'


def _is_no_space_after(ch: str) -> bool:
    return ch in '([{<\'"'


def detokenize(tokens: list[str], language: str) -> tuple[str, list[tuple[int, int]]]:
    if is_cjk_language(language):
        return smart_cjk_join(tokens)

    if not tokens:
        return "", []

    parts = []
    offsets = []
    pos = 0

    for i, token in enumerate(tokens):
        if i > 0 and token:
            if not _is_no_space_before(token[0]) and not _is_no_space_after(tokens[i - 1][-1] if tokens[i - 1] else ""):
                parts.append(" ")
                pos += 1
        start = pos
        parts.append(token)
        pos += len(token)
        offsets.append((start, pos))

    return "".join(parts), offsets


def bio_tags_to_spans(
    tokens: list[str],
    tags: list[int | str],
    tag_map: list[str] | dict[int, str] | None = None,
) -> list[dict]:
    spans = []
    current_type = None
    current_tokens = []
    current_start = None

    for i, (token, tag) in enumerate(zip(tokens, tags)):
        if tag_map is not None:
            if isinstance(tag_map, list):
                tag_str = tag_map[tag]
            else:
                tag_str = tag_map[tag]
        else:
            tag_str = str(tag)

        if tag_str == "O" or tag_str == "0":
            if current_type is not None:
                spans.append({
                    "type": current_type,
                    "token_start": current_start,
                    "token_end": i,
                    "tokens": current_tokens[:],
                })
                current_type = None
                current_tokens = []
            continue

        if tag_str.startswith("B-"):
            if current_type is not None:
                spans.append({
                    "type": current_type,
                    "token_start": current_start,
                    "token_end": i,
                    "tokens": current_tokens[:],
                })
            current_type = tag_str[2:]
            current_start = i
            current_tokens = [token]
        elif tag_str.startswith("I-"):
            etype = tag_str[2:]
            if current_type == etype:
                current_tokens.append(token)
            else:
                if current_type is not None:
                    spans.append({
                        "type": current_type,
                        "token_start": current_start,
                        "token_end": i,
                        "tokens": current_tokens[:],
                    })
                current_type = etype
                current_start = i
                current_tokens = [token]
        else:
            if current_type is not None:
                spans.append({
                    "type": current_type,
                    "token_start": current_start,
                    "token_end": i,
                    "tokens": current_tokens[:],
                })
                current_type = None
                current_tokens = []

    if current_type is not None:
        spans.append({
            "type": current_type,
            "token_start": current_start,
            "token_end": len(tokens),
            "tokens": current_tokens[:],
        })

    return spans


def tokens_to_char_spans(
    token_spans: list[dict],
    token_offsets: list[tuple[int, int]],
    text: str,
) -> list[dict]:
    result = []
    for span in token_spans:
        char_start = token_offsets[span["token_start"]][0]
        char_end = token_offsets[span["token_end"] - 1][1]
        surface = text[char_start:char_end]
        result.append({
            "surface": surface,
            "type": span["type"],
            "start": char_start,
            "end": char_end,
        })
    return result
