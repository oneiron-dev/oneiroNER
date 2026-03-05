"""Shared LLM annotation harness for NER labeling via OpenCode CLI."""

import json
import logging
import os
import re
import subprocess

logger = logging.getLogger(__name__)

OPENCODE_BIN = "/home/ubuntu/.opencode/bin/opencode"

_DEFAULT_MODEL = "openai/gpt-5.3-codex-spark"
_CODEX_MODEL = "openai/gpt-5.3-codex"

NER_PROMPT = """\
You are an expert NER annotator. Think step-by-step internally before outputting labels.

{text_block}

Entity types:
- PERSON: named individuals ("Sarah", "Dr. Chen", "田中先生")
- PLACE: locations ("Tokyo", "the park", "学校")
- ORG: organizations ("Google", "東京大学")
- DATE: temporal expressions. Subtypes: Day, Week, Month, Season, Year, Decade, Relative, Range
- EVENT: named events ("Christmas", "the wedding", "文化祭")
- RELATIONSHIP_REF: terms that refer to or address a specific person via their relationship role. Subtypes: Family, Romantic, Friend, Professional, Acquaintance
  - Include bare kinship: "Mom" is Family. Include possessive: "my sister" is Family.
  - Include indefinite specific: "a friend told me" -> Friend. Skip purely generic: "I need a friend".
  - NOT generic nouns like "job", "relationship" — must refer to a specific person.
- EMOTION: emotional states ("happy", "anxious", "悲しい")
- GOAL: intentions/desires ("I want to travel", "勉強したい")
- ACTIVITY: activities being done ("studying", "cooking", "散歩")

Offsets: 0-indexed Unicode chars, end-exclusive, Python slicing (text[start:end] == surface).

Return ONLY a JSON array:
[{{"surface": "my mom", "type": "RELATIONSHIP_REF/Family", "start": 7, "end": 13}}]
{turn_instruction}\
If none found, return []."""


def _get_model(provider: str) -> str:
    if provider == "codex":
        return os.environ.get("OPENCODE_MODEL_CODEX", _CODEX_MODEL)
    return os.environ.get("OPENCODE_MODEL", _DEFAULT_MODEL)


def _parse_opencode_jsonl(raw: str) -> str:
    parts = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            evt = json.loads(line)
            if evt.get("type") == "text" and "text" in evt:
                parts.append(evt["text"])
        except json.JSONDecodeError:
            continue
    return "".join(parts) if parts else raw


def _clean_llm_output(raw: str) -> list[dict] | None:
    cleaned = re.sub(r"^```(?:json)?\n?", "", raw.strip())
    cleaned = re.sub(r"\n?```$", "", cleaned)

    match = re.search(r"\[[\s\S]*\]", cleaned)
    if match:
        try:
            arr = json.loads(match.group())
            if isinstance(arr, list):
                return arr
        except json.JSONDecodeError:
            pass

    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        try:
            obj = json.loads("[" + match.group() + "]")
            if isinstance(obj, list):
                return obj
        except json.JSONDecodeError:
            pass

    return None


def _call_opencode(prompt: str, model: str, timeout: int = 120) -> str | None:
    try:
        r = subprocess.run(
            [OPENCODE_BIN, "run", "-m", model, prompt],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if r.returncode != 0:
            logger.warning("opencode failed (rc=%d): %s", r.returncode, r.stderr[:200])
            return None
        raw = r.stdout.strip()
        return _parse_opencode_jsonl(raw)
    except subprocess.TimeoutExpired:
        logger.warning("opencode timed out (%ds)", timeout)
        return None
    except Exception as e:
        logger.warning("opencode error: %s", e)
        return None


def _build_passage_prompt(text: str) -> str:
    text_block = text
    return NER_PROMPT.format(text_block=text_block, turn_instruction="")


def _build_conversation_prompt(turns: list[dict]) -> str:
    lines = []
    for i, turn in enumerate(turns):
        lines.append(f"[Turn {i}] {turn['speaker']}: {turn['text']}")
    text_block = "\n".join(lines)
    turn_instruction = 'For conversation, add "turn_index": N (0-indexed). Offsets are relative to turn text, NOT the full concatenated conversation.\n'
    return NER_PROMPT.format(text_block=text_block, turn_instruction=turn_instruction)


def _validate_entity(ent: dict, is_conversation: bool) -> bool:
    required = {"surface", "type", "start", "end"}
    if not required.issubset(ent.keys()):
        return False
    if not isinstance(ent["start"], int) or not isinstance(ent["end"], int):
        return False
    if ent["end"] <= ent["start"]:
        return False
    if is_conversation and "turn_index" not in ent:
        return False
    return True


def annotate_passage(
    text: str, language: str, provider: str = "spark", timeout: int = 120
) -> list[dict]:
    prompt = _build_passage_prompt(text)
    model = _get_model(provider)

    raw = _call_opencode(prompt, model, timeout=timeout)
    if raw is None:
        return []

    entities = _clean_llm_output(raw)
    if entities is None:
        logger.warning("Failed to parse LLM output for passage annotation")
        return []

    return [e for e in entities if _validate_entity(e, is_conversation=False)]


def annotate_conversation(
    turns: list[dict], language: str, provider: str = "spark", timeout: int = 120
) -> list[dict]:
    prompt = _build_conversation_prompt(turns)
    model = _get_model(provider)

    raw = _call_opencode(prompt, model, timeout=timeout)
    if raw is None:
        return []

    entities = _clean_llm_output(raw)
    if entities is None:
        logger.warning("Failed to parse LLM output for conversation annotation")
        return []

    return [e for e in entities if _validate_entity(e, is_conversation=True)]
