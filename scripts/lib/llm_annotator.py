"""Shared LLM annotation harness for NER labeling via OpenCode CLI + Gemini + OpenRouter."""

import itertools
import json
import logging
import os
import re
import subprocess
import threading
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

_ROTATE_MODELS = [
    "openai/gpt-5-codex",
    "openai/gpt-5.1-codex-mini",
]
_rotate_idx = itertools.count()
_rotate_lock = threading.Lock()

OPENCODE_BIN = "/home/ubuntu/.opencode/bin/opencode"
CLAUDE_BIN = "/home/ubuntu/.local/bin/claude"

_DEFAULT_MODEL = "openai/gpt-5.3-codex-spark"
_CODEX_MODEL = "openai/gpt-5.3-codex"

NER_PROMPT = """\
You are an expert NER annotator. Output ONLY the JSON array — no reasoning, no explanation, no markdown.

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

Return ONLY a JSON array (no other text):
[{{"surface": "my mom", "type": "RELATIONSHIP_REF/Family", "start": 7, "end": 13}}]
{turn_instruction}\
If none found, return []."""


_CLAUDE_MODEL = "anthropic/claude-sonnet-4-20250514"


def _get_model(provider: str) -> str:
    if provider == "codex":
        return os.environ.get("OPENCODE_MODEL_CODEX", _CODEX_MODEL)
    if provider == "claude":
        return os.environ.get("OPENCODE_MODEL_CLAUDE", _CLAUDE_MODEL)
    if provider == "codex5":
        return "openai/gpt-5-codex"
    if provider == "mini":
        return "openai/gpt-5.1-codex-mini"
    return os.environ.get("OPENCODE_MODEL", _DEFAULT_MODEL)


def _parse_opencode_jsonl(raw: str) -> tuple[str, bool]:
    """Parse OpenCode JSON event stream. Returns (text, content_filtered)."""
    parts = []
    content_filtered = False
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            evt = json.loads(line)
            if not isinstance(evt, dict):
                continue
            if evt.get("type") == "text":
                part = evt.get("part", {})
                if isinstance(part, dict) and "text" in part:
                    parts.append(part["text"])
                elif "text" in evt:
                    parts.append(evt["text"])
            elif evt.get("type") == "step_finish":
                part = evt.get("part", {})
                if isinstance(part, dict) and part.get("reason") == "content-filter":
                    content_filtered = True
        except json.JSONDecodeError:
            continue
    text = "".join(parts) if parts else ""
    return text, content_filtered


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


def _call_opencode(prompt: str, model: str, timeout: int = 120) -> tuple[str | None, bool]:
    """Returns (parsed_text, content_filtered)."""
    try:
        r = subprocess.run(
            [OPENCODE_BIN, "run", "-m", model, "--format", "json", "-"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if r.returncode != 0:
            logger.warning("opencode failed (rc=%d): %s", r.returncode, r.stderr[:200])
            return None, False
        raw = r.stdout.strip()
        parsed, content_filtered = _parse_opencode_jsonl(raw)
        if content_filtered:
            logger.debug("opencode content-filtered")
            return None, True
        if not parsed:
            logger.debug("opencode empty response")
            return None, False
        return parsed, False
    except subprocess.TimeoutExpired:
        logger.warning("opencode timed out (%ds)", timeout)
        return None, False
    except Exception as e:
        logger.warning("opencode error: %s", e)
        return None, False


def _call_claude_cli(prompt: str, timeout: int = 120) -> str | None:
    env = {k: v for k, v in os.environ.items() if not k.startswith("CLAUDE")}
    try:
        r = subprocess.run(
            [
                CLAUDE_BIN, "-p",
                "--model", "sonnet",
                "--output-format", "text",
                "--tools", "",
                "--disable-slash-commands",
                "--strict-mcp-config",
                "--mcp-config", '{"mcpServers":{}}',
                "--no-session-persistence",
            ],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        if r.returncode != 0:
            logger.warning("claude cli failed (rc=%d): %s", r.returncode, r.stderr[:200])
            return None
        out = r.stdout.strip()
        if not out:
            logger.debug("claude cli empty response")
            return None
        return out
    except subprocess.TimeoutExpired:
        logger.warning("claude cli timed out (%ds)", timeout)
        return None
    except Exception as e:
        logger.warning("claude cli error: %s", e)
        return None


def _load_dotenv():
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

_load_dotenv()


def _get_openrouter_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise AnnotationError("No OPENROUTER_API_KEY in env or .env")
    return key


_OPENROUTER_SYSTEM = (
    "You are an expert NER annotator. Extract all matching entities from the user's text.\n\n"
    "Entity types:\n"
    "- PERSON: named individuals\n"
    "- PLACE: locations\n"
    "- ORG: organizations\n"
    "- DATE: temporal expressions. Subtypes: Day, Week, Month, Season, Year, Decade, Relative, Range\n"
    "- EVENT: named events\n"
    "- RELATIONSHIP_REF: terms referring to a specific person via relationship role. Subtypes: Family, Romantic, Friend, Professional, Acquaintance\n"
    "  - Include bare kinship: 'Mom' is Family. Include possessive: 'my sister' is Family.\n"
    "  - Include indefinite specific: 'a friend told me' -> Friend. Skip purely generic: 'I need a friend'.\n"
    "- EMOTION: emotional states\n"
    "- GOAL: intentions/desires\n"
    "- ACTIVITY: activities being done\n\n"
    "Rules:\n"
    "- start/end are 0-indexed character offsets within the user's text. First char = 0. end exclusive.\n"
    "- text[start:end] must equal surface exactly."
)

_OPENROUTER_PROVIDERS = {
    "only": ["atlas-cloud", "novita", "siliconflow"],
    "quantizations": ["fp8"],
}

_CANONICAL_TYPES = [
    "PERSON", "PLACE", "ORG",
    "EVENT", "EVENT/Life", "EVENT/General",
    "EMOTION", "GOAL", "ACTIVITY",
    "DATE", "DATE/Day", "DATE/Week", "DATE/Month", "DATE/Season",
    "DATE/Year", "DATE/Decade", "DATE/Relative", "DATE/Range",
    "RELATIONSHIP_REF", "RELATIONSHIP_REF/Family", "RELATIONSHIP_REF/Romantic",
    "RELATIONSHIP_REF/Friend", "RELATIONSHIP_REF/Professional",
    "RELATIONSHIP_REF/Acquaintance",
]

_OPENROUTER_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "ner_entities",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "surface": {"type": "string"},
                            "type": {"type": "string", "enum": _CANONICAL_TYPES},
                            "start": {"type": "integer"},
                            "end": {"type": "integer"},
                        },
                        "required": ["surface", "type", "start", "end"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["entities"],
            "additionalProperties": False,
        },
    },
}

_OPENROUTER_CONV_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "ner_entities",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "surface": {"type": "string"},
                            "type": {"type": "string", "enum": _CANONICAL_TYPES},
                            "start": {"type": "integer"},
                            "end": {"type": "integer"},
                            "turn_index": {"type": "integer"},
                        },
                        "required": ["surface", "type", "start", "end", "turn_index"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["entities"],
            "additionalProperties": False,
        },
    },
}


def _call_openrouter(prompt: str, model: str = "deepseek/deepseek-v3.2", timeout: int = 120, is_conversation: bool = False) -> list[dict] | None:
    key = _get_openrouter_key()
    schema = _OPENROUTER_CONV_SCHEMA if is_conversation else _OPENROUTER_SCHEMA
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": _OPENROUTER_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 4096,
        "response_format": schema,
        "provider": _OPENROUTER_PROVIDERS,
    }).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        content = data["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        return parsed.get("entities", [])
    except Exception as e:
        logger.warning("openrouter error: %s", e)
        return None


def _call_gemini(prompt: str, timeout: int = 120) -> str | None:
    try:
        r = subprocess.run(
            ["gemini", "-p", "", "--yolo", "-o", "text"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if r.returncode != 0:
            logger.debug("gemini failed (rc=%d): %s", r.returncode, r.stderr[:200])
            return None
        return r.stdout.strip()
    except subprocess.TimeoutExpired:
        logger.debug("gemini timed out")
        return None
    except Exception as e:
        logger.debug("gemini error: %s", e)
        return None


def _build_passage_prompt(text: str) -> str:
    return NER_PROMPT.format(text_block=text, turn_instruction="")


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


class AnnotationError(RuntimeError):
    pass


def _annotate_single(prompt: str, is_conversation: bool, provider: str, timeout: int) -> list[dict]:
    if provider == "deepseek":
        entities = _call_openrouter(prompt, timeout=timeout, is_conversation=is_conversation)
        if entities is None:
            raise AnnotationError("deepseek call failed")
        return [e for e in entities if _validate_entity(e, is_conversation)]

    if provider in ("gemini", "claude_cli"):
        call_fn = _call_gemini if provider == "gemini" else _call_claude_cli
        raw = call_fn(prompt, timeout=timeout)
        if raw is None:
            raise AnnotationError(f"{provider} call failed")
        entities = _clean_llm_output(raw)
        if entities is None:
            raise AnnotationError(f"Failed to parse {provider} output: {raw[:200]}")
        return [e for e in entities if _validate_entity(e, is_conversation)]

    model = _get_model(provider)
    raw, content_filtered = _call_opencode(prompt, model, timeout=timeout)

    if content_filtered:
        logger.debug("Content-filtered by OpenCode, falling back to Gemini")
        raw = _call_gemini(prompt, timeout=timeout)
        if raw is None:
            raise AnnotationError("Gemini fallback failed after content filter")
        entities = _clean_llm_output(raw)
        if entities is None:
            raise AnnotationError(f"Failed to parse Gemini fallback output: {raw[:200]}")
        return [e for e in entities if _validate_entity(e, is_conversation)]

    if raw is None:
        raise AnnotationError(f"opencode call failed (model={model})")

    entities = _clean_llm_output(raw)
    if entities is None:
        raise AnnotationError(f"Failed to parse LLM output: {raw[:200]}")

    return [e for e in entities if _validate_entity(e, is_conversation)]


def _annotate(prompt: str, is_conversation: bool, provider: str, timeout: int) -> list[dict]:
    if provider != "rotate":
        return _annotate_single(prompt, is_conversation, provider, timeout)

    idx = next(_rotate_idx) % len(_ROTATE_MODELS)
    model = _ROTATE_MODELS[idx]
    raw, content_filtered = _call_opencode(prompt, model, timeout=timeout)
    if content_filtered:
        raw = _call_gemini(prompt, timeout=timeout)
        if raw is None:
            raise AnnotationError("Gemini fallback failed after content filter (rotate)")
        entities = _clean_llm_output(raw)
        if entities is None:
            raise AnnotationError(f"Failed to parse Gemini fallback (rotate): {raw[:200]}")
        return [e for e in entities if _validate_entity(e, is_conversation)]
    if raw is None:
        raise AnnotationError(f"opencode call failed (rotate, model={model})")
    entities = _clean_llm_output(raw)
    if entities is None:
        raise AnnotationError(f"Failed to parse LLM output (rotate, model={model}): {raw[:200]}")
    return [e for e in entities if _validate_entity(e, is_conversation)]


def annotate_passage(
    text: str, language: str, provider: str = "spark", timeout: int = 120
) -> list[dict]:
    if provider == "deepseek":
        return _annotate(text, is_conversation=False, provider=provider, timeout=timeout)
    prompt = _build_passage_prompt(text)
    return _annotate(prompt, is_conversation=False, provider=provider, timeout=timeout)


def annotate_conversation(
    turns: list[dict], language: str, provider: str = "spark", timeout: int = 120
) -> list[dict]:
    if provider == "deepseek":
        lines = []
        for i, turn in enumerate(turns):
            lines.append(f"[Turn {i}] {turn['speaker']}: {turn['text']}")
        text = "\n".join(lines)
        text += "\n\nFor conversation, add turn_index (0-indexed). Offsets relative to each turn's text."
        return _annotate(text, is_conversation=True, provider=provider, timeout=timeout)
    prompt = _build_conversation_prompt(turns)
    return _annotate(prompt, is_conversation=True, provider=provider, timeout=timeout)
