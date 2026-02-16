"""Rule-based temporal granularity classifier with multilingual vocab support."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

GRANULARITIES = ["Exact", "Hour", "Day", "Week", "Month", "Season", "Year", "Vague"]
_CATEGORY_ORDER = tuple(GRANULARITIES)
_VOCAB_PATH = Path(__file__).parent.parent.parent / "configs" / "temporal_vocab.json"
_LATIN_FLAGS = re.IGNORECASE


@dataclass(frozen=True)
class GranularityResult:
    granularity: str
    ambiguous: bool = False
    matched_categories: tuple[str, ...] = ()


def _matches_any(text: str, patterns: list[re.Pattern[str]]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def _compile_all(patterns: list[str], flags: int = 0) -> list[re.Pattern[str]]:
    return [re.compile(pattern, flags) for pattern in patterns]


_EN_WEEKDAYS = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)
_EN_MONTH_FULL_NAMES = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)
_EN_MONTH_ABBREVIATIONS = (
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "sept",
    "oct",
    "nov",
    "dec",
)
_EN_WEEKDAY_RE = "(?:" + "|".join(_EN_WEEKDAYS) + ")"
_EN_MONTH_FULL_RE = "(?:" + "|".join(_EN_MONTH_FULL_NAMES) + ")"
_EN_MONTH_ABBR_RE = "(?:" + "|".join(_EN_MONTH_ABBREVIATIONS) + ")"
_EN_MONTH_ANY_RE = "(?:" + "|".join(_EN_MONTH_FULL_NAMES + _EN_MONTH_ABBREVIATIONS) + ")"

_HARDCODED_LATIN_PATTERNS_RAW: dict[str, list[str]] = {
    "Exact": [
        r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:am|pm)?\b",
        r"\bat\s+\d{1,2}\s*(?:am|pm)\b",
        r"\b\d{1,2}\s*o'clock\b",
        r"\bnoon\s+sharp\b",
        r"\bmidnight\s+sharp\b",
    ],
    "Hour": [
        r"\b(?:morning|afternoon|evening|night|dawn|dusk)\b",
        r"\bnoon\b(?!\s*sharp\b)",
        r"\bmidnight\b(?!\s*sharp\b)",
    ],
    "Day": [
        rf"\b{_EN_WEEKDAY_RE}\b",
        r"\b(?:yesterday|today|tomorrow)\b",
        r"\bthe\s+day\s+(?:before|after)\b",
        rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?{_EN_MONTH_ANY_RE}\b",
        rf"\b{_EN_MONTH_ANY_RE}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,\s*\d{{4}})?\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{1,2}\s+de\s+\w+\s+de\s+\d{4}\b",
    ],
    "Week": [
        r"\b(?:last|this|next|past)\s+week\b",
        r"\bweekend\b",
        r"\bweeklong\b",
        r"\bla\s+scorsa\s+settimana\b",
    ],
    "Month": [
        rf"\b{_EN_MONTH_FULL_RE}\b(?!\s+\d{{1,2}}(?:st|nd|rd|th)?\b)",
        r"\b(?:last|this|next|past)\s+month\b",
        rf"\b{_EN_MONTH_ABBR_RE}\b(?!\s+\d{{1,2}}(?:st|nd|rd|th)?\b)",
    ],
    "Season": [
        r"\b(?:spring|summer|autumn|fall|winter)\b",
        r"\b(?:last|this|next)\s+(?:spring|summer|autumn|fall|winter)\b",
        r"\bover\s+the\s+holidays\b",
        r"\bholiday\s+season\b",
    ],
    "Year": [
        r"^\s*\d{4}\s*$",
        r"\bin\s+\d{4}\b",
        r"\b\d+\s+years?\s+ago\b",
        r"\b(?:last|this|next|past)\s+year\b",
        r"\bback\s+in\s+\d{4}\b",
        r"\bthe\s+\d{4}s\b",
    ],
    "Vague": [
        r"\ba\s+while\b",
        r"\blong\s+time\b",
        r"\bages\s+ago\b",
        r"\brecently\b",
        r"\blately\b",
        r"\bonce\b",
        r"\bback\s+then\b",
        r"\bin\s+the\s+past\b",
        r"\blong\s+ago\b",
        r"\bsome\s+time\b",
        r"\bthe\s+other\s+day\b",
        r"\bnot\s+long\s+ago\b",
        r"\bearlier\b",
        r"\bbefore\b",
        r"\bformerly\b",
        r"\bpreviously\b",
    ],
}

_HARDCODED_CJK_PATTERNS_RAW: dict[str, list[str]] = {
    "Exact": [
        r"\d{1,2}\s*時\s*\d{1,2}\s*分",
        r"正午ちょうど",
        r"\d{1,2}\s*点\s*\d{1,2}\s*分",
        r"下午\s*\d{1,2}\s*点\s*\d{1,2}\s*分",
        r"\d{1,2}\s*시\s*\d{1,2}\s*분",
    ],
    "Hour": [
        r"今朝|今夜|朝|午前|午後|夕方|夜",
        r"今天早上|那天下午|早上|上午|下午(?!\s*\d{1,2}\s*点\s*\d{1,2}\s*分)|傍晚|晚上",
        r"아침|오전|오후(?!\s*\d{1,2}\s*시\s*\d{1,2}\s*분)|저녁|밤",
    ],
    "Day": [
        r"月曜日|火曜日|水曜日|木曜日|金曜日|土曜日|日曜日|月曜|火曜|水曜|木曜|金曜|土曜|日曜",
        r"一昨日|明後日|昨日|今日|明日",
        r"\d{1,2}\s*月\s*\d{1,2}\s*日",
        r"星期[一二三四五六日天]|周[一二三四五六日天]",
        r"前天|后天|昨天|明天|今天(?!\s*(?:早上|上午|下午|晚上|傍晚))",
        r"\d{1,2}\s*月\s*\d{1,2}\s*[日号號]",
        r"\d{1,2}[日号號]",
        r"월요일|화요일|수요일|목요일|금요일|토요일|일요일",
        r"그저께|모레|어제|오늘|내일",
        r"\d{1,2}\s*월\s*\d{1,2}\s*일",
    ],
    "Week": [
        r"先週|今週|来週|週末",
        r"上個星期|這個星期|下個星期|上周|这周|下周|周末|這個週末",
        r"지난\s*주|이번\s*주|다음\s*주|주말",
    ],
    "Month": [
        r"\d{1,2}\s*月(?!\s*\d{1,2}\s*日)",
        r"先月|今月|来月",
        r"\d{1,2}\s*月份?(?!\s*\d{1,2}\s*[日号號])",
        r"上个月|这个月|下个月",
        r"\d{1,2}\s*월(?!\s*\d{1,2}\s*일)",
        r"지난\s*달|이번\s*달|다음\s*달",
    ],
    "Season": [
        r"去年の夏|この春|今年の冬|春|夏|秋|冬",
        r"去年夏天|今年春天|春天?|夏天?|秋天?|冬天?",
        r"봄|여름|가을|겨울",
    ],
    "Year": [
        r"\d{4}\s*年(?!\s*\d{1,2}\s*月)",
        r"\d+\s*年前",
        r"去年(?!\s*(?:の)?(?:春|夏|秋|冬|[0-9０-９]{1,2}\s*月))",
        r"今年(?!\s*(?:の)?(?:春|夏|秋|冬|[0-9０-９]{1,2}\s*月))",
        r"来年(?!\s*(?:の)?(?:春|夏|秋|冬|[0-9０-９]{1,2}\s*月))",
        r"一昨年(?!\s*(?:の)?(?:春|夏|秋|冬|[0-9０-９]{1,2}\s*月))",
        r"再来年(?!\s*(?:の)?(?:春|夏|秋|冬|[0-9０-９]{1,2}\s*月))",
        r"\d{4}\s*年(?!\s*\d{1,2}\s*月)",
        r"\d+\s*年前",
        r"[一二三四五六七八九十百千万两兩]+\s*年前",
        r"去年(?!\s*(?:の)?(?:春|夏|秋|冬|月|周|週|天|日|号|號))",
        r"今年(?!\s*(?:の)?(?:春|夏|秋|冬|月|周|週|天|日|号|號))",
        r"明年(?!\s*(?:の)?(?:春|夏|秋|冬|月|周|週|天|日|号|號))",
        r"前年(?!\s*(?:の)?(?:春|夏|秋|冬|月|周|週|天|日|号|號))",
        r"后年(?!\s*(?:の)?(?:春|夏|秋|冬|月|周|週|天|日|号|號))",
        r"\d{4}\s*년(?!\s*\d{1,2}\s*월)",
        r"\d+\s*년\s*전",
        r"작년(?!\s*(?:봄|여름|가을|겨울|\d{1,2}\s*월))",
        r"올해(?!\s*(?:봄|여름|가을|겨울|\d{1,2}\s*월))",
        r"내년(?!\s*(?:봄|여름|가을|겨울|\d{1,2}\s*월))",
        r"재작년(?!\s*(?:봄|여름|가을|겨울|\d{1,2}\s*월))",
    ],
    "Vague": [
        r"しばらく|ずっと前|最近|以前|昔|かつて|少し前",
        r"很久以前|最近|以前|前一阵子|不久前|过去|从前|曾经|刚才",
        r"한참\s*전|오래전|최근|예전|얼마\s*전|옛날",
    ],
}

_HARDCODED_LATIN_PATTERNS = {
    category: _compile_all(_HARDCODED_LATIN_PATTERNS_RAW.get(category, []), _LATIN_FLAGS)
    for category in _CATEGORY_ORDER
}
_HARDCODED_CJK_PATTERNS = {
    category: _compile_all(_HARDCODED_CJK_PATTERNS_RAW.get(category, []))
    for category in _CATEGORY_ORDER
}


def _escape_vocab_phrase(phrase: str, allow_numeric_placeholder: bool) -> str:
    token = "__NUM_PLACEHOLDER__"
    value = phrase
    if allow_numeric_placeholder:
        value = value.replace("{X}", token).replace("X", token)

    escaped = re.escape(value).replace(r"\ ", r"\s+")
    if allow_numeric_placeholder:
        escaped = escaped.replace(re.escape(token), r"\d+")
    return escaped


def _load_vocab_patterns() -> dict[str, list[re.Pattern[str]]]:
    patterns_by_category = {category: [] for category in _CATEGORY_ORDER}
    seen_patterns: dict[str, set[str]] = {category: set() for category in _CATEGORY_ORDER}

    try:
        raw_data = json.loads(_VOCAB_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "Could not load temporal vocab from %s (%s). Falling back to hardcoded EN/JA/ZH/KO patterns.",
            _VOCAB_PATH,
            exc,
        )
        return patterns_by_category

    vocab_map: dict[str, tuple[str, ...]] = {
        "Day": ("weekdays", "yesterday_today_tomorrow"),
        "Week": ("week_words",),
        "Month": ("months", "month_words"),
        "Season": ("seasons",),
        "Year": ("year_words",),
        "Hour": ("time_of_day",),
        "Vague": ("vague",),
    }

    for language_table in raw_data.values():
        if not isinstance(language_table, dict):
            continue
        for category, keys in vocab_map.items():
            for key in keys:
                terms = language_table.get(key, [])
                if not isinstance(terms, list):
                    continue
                for term in terms:
                    if not isinstance(term, str):
                        continue
                    term = term.strip()
                    if not term:
                        continue

                    escaped = _escape_vocab_phrase(
                        term,
                        allow_numeric_placeholder=(category == "Year" and key == "year_words"),
                    )
                    # Vocab terms are intended as lexical items/phrases, so match full surface spans.
                    pattern_src = rf"^\s*{escaped}\s*$"
                    if pattern_src in seen_patterns[category]:
                        continue
                    seen_patterns[category].add(pattern_src)
                    patterns_by_category[category].append(re.compile(pattern_src, _LATIN_FLAGS))

    return patterns_by_category


_VOCAB_PATTERNS = _load_vocab_patterns()


def _category_matches(category: str, surface_orig: str, surface_lc: str) -> bool:
    if _matches_any(surface_lc, _HARDCODED_LATIN_PATTERNS[category]):
        return True
    if _matches_any(surface_orig, _HARDCODED_CJK_PATTERNS[category]):
        return True

    vocab_patterns = _VOCAB_PATTERNS[category]
    if _matches_any(surface_orig, vocab_patterns):
        return True
    if surface_lc != surface_orig and _matches_any(surface_lc, vocab_patterns):
        return True
    return False


def classify_granularity(surface: str) -> GranularityResult:
    """Classify temporal granularity from entity surface text."""
    if not surface or not surface.strip():
        return GranularityResult("Day")

    surface_orig = surface
    surface_lc = surface.lower()

    matched_categories: list[str] = []
    for category in _CATEGORY_ORDER:
        if _category_matches(category, surface_orig, surface_lc):
            matched_categories.append(category)

    if not matched_categories:
        return GranularityResult("Day")

    return GranularityResult(
        granularity=matched_categories[0],
        ambiguous=len(matched_categories) > 1,
        matched_categories=tuple(matched_categories),
    )
