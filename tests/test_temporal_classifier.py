import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.lib.temporal_classifier import classify_granularity


def _assert_case(surface: str, expected_granularity: str, expected_ambiguous: bool = False) -> None:
    result = classify_granularity(surface)
    assert result.granularity == expected_granularity
    assert result.ambiguous is expected_ambiguous


@pytest.mark.parametrize(
    ("surface", "expected_granularity", "expected_ambiguous"),
    [
        ("3:15pm", "Exact", False),
        ("at 8am", "Exact", False),
        ("noon sharp", "Exact", False),
        ("下午3点15分", "Exact", False),
        ("3時15分に", "Exact", False),
        ("오후 3시 15분", "Exact", False),
    ],
)
def test_exact_cases(surface: str, expected_granularity: str, expected_ambiguous: bool) -> None:
    _assert_case(surface, expected_granularity, expected_ambiguous)


@pytest.mark.parametrize(
    ("surface", "expected_granularity", "expected_ambiguous"),
    [
        ("this morning", "Hour", False),
        ("that afternoon", "Hour", False),
        ("last night", "Hour", False),
        ("今朝", "Hour", False),
        ("今天早上", "Hour", False),
        ("저녁에", "Hour", False),
        ("밤", "Hour", False),
    ],
)
def test_hour_cases(surface: str, expected_granularity: str, expected_ambiguous: bool) -> None:
    _assert_case(surface, expected_granularity, expected_ambiguous)


@pytest.mark.parametrize(
    ("surface", "expected_granularity", "expected_ambiguous"),
    [
        ("last Tuesday", "Day", False),
        ("yesterday", "Day", False),
        ("tomorrow", "Day", False),
        ("January 5th", "Day", False),
        ("5 de septiembre de 2011", "Day", False),
        ("昨日", "Day", False),
        ("昨天", "Day", False),
        ("어제", "Day", False),
        ("1月5日", "Day", False),
        ("2024-03-15", "Day", False),
        ("lunes", "Day", False),
        ("gestern", "Day", False),
        ("hier", "Day", False),
        ("ieri", "Day", False),
        ("ontem", "Day", False),
        ("вчера", "Day", False),
        ("gisteren", "Day", False),
        ("wczoraj", "Day", False),
        ("вчора", "Day", False),
        ("कल", "Day", False),
        ("أمس", "Day", False),
        ("เมื่อวาน", "Day", False),
        ("hôm qua", "Day", False),
        ("kemarin", "Day", False),
        ("xyzzy temporal", "Day", False),
    ],
)
def test_day_cases(surface: str, expected_granularity: str, expected_ambiguous: bool) -> None:
    _assert_case(surface, expected_granularity, expected_ambiguous)


@pytest.mark.parametrize(
    ("surface", "expected_granularity", "expected_ambiguous"),
    [
        ("last week", "Week", False),
        ("this weekend", "Week", False),
        ("先週", "Week", False),
        ("上周", "Week", False),
        ("지난 주", "Week", False),
        ("la semana pasada", "Week", False),
        ("letzte Woche", "Week", False),
        ("la semaine dernière", "Week", False),
        ("la scorsa settimana", "Week", False),
    ],
)
def test_week_cases(surface: str, expected_granularity: str, expected_ambiguous: bool) -> None:
    _assert_case(surface, expected_granularity, expected_ambiguous)


@pytest.mark.parametrize(
    ("surface", "expected_granularity", "expected_ambiguous"),
    [
        ("in March", "Month", False),
        ("last month", "Month", False),
        ("December 1830", "Month", False),
        ("August, 2006", "Month", False),
        ("先月", "Month", False),
        ("上个月", "Month", False),
        ("지난 달", "Month", False),
        ("1995年10月", "Month", False),
        ("3月", "Month", False),
        ("enero", "Month", False),
        ("Januar", "Month", False),
        ("février", "Month", False),
        ("gennaio", "Month", False),
        ("januari", "Month", False),
    ],
)
def test_month_cases(surface: str, expected_granularity: str, expected_ambiguous: bool) -> None:
    _assert_case(surface, expected_granularity, expected_ambiguous)


@pytest.mark.parametrize(
    ("surface", "expected_granularity", "expected_ambiguous"),
    [
        ("last summer", "Season", False),
        ("this spring", "Season", False),
        ("over the holidays", "Season", False),
        ("去年の夏", "Season", False),
        ("去年夏天", "Season", False),
        ("지난 여름", "Season", False),
        ("verano", "Season", False),
        ("Sommer", "Season", False),
        ("printemps", "Season", False),
        ("inverno", "Season", False),
        ("зима", "Season", False),
    ],
)
def test_season_cases(surface: str, expected_granularity: str, expected_ambiguous: bool) -> None:
    _assert_case(surface, expected_granularity, expected_ambiguous)


@pytest.mark.parametrize(
    ("surface", "expected_granularity", "expected_ambiguous"),
    [
        ("in 2023", "Year", False),
        ("1859", "Year", False),
        ("5 years ago", "Year", False),
        ("back in 2019", "Year", False),
        ("2023年", "Year", False),
        ("五年前", "Year", False),
        ("去年", "Year", False),
        ("작년", "Year", False),
    ],
)
def test_year_cases(surface: str, expected_granularity: str, expected_ambiguous: bool) -> None:
    _assert_case(surface, expected_granularity, expected_ambiguous)


@pytest.mark.parametrize(
    ("surface", "expected_granularity", "expected_ambiguous"),
    [
        ("a while back", "Vague", False),
        ("long time ago", "Vague", False),
        ("recently", "Vague", False),
        ("しばらく前", "Vague", False),
        ("很久以前", "Vague", False),
        ("오래전에", "Vague", False),
        ("recientemente", "Vague", False),
        ("vor langer Zeit", "Vague", False),
        ("récemment", "Vague", False),
        ("недавно", "Vague", False),
    ],
)
def test_vague_cases(surface: str, expected_granularity: str, expected_ambiguous: bool) -> None:
    _assert_case(surface, expected_granularity, expected_ambiguous)


def test_ambiguous_manana() -> None:
    result = classify_granularity("mañana")
    assert result.ambiguous is True
    assert "Hour" in result.matched_categories
    assert "Day" in result.matched_categories
    assert result.granularity == "Hour"


def test_ambiguous_morgen() -> None:
    result = classify_granularity("morgen")
    assert result.ambiguous is True
    assert "Hour" in result.matched_categories
    assert "Day" in result.matched_categories
    assert result.granularity == "Hour"


def test_unambiguous_yesterday() -> None:
    result = classify_granularity("yesterday")
    assert result.ambiguous is False
    assert result.granularity == "Day"


def test_unambiguous_summer() -> None:
    result = classify_granularity("summer")
    assert result.ambiguous is False
    assert result.granularity == "Season"
