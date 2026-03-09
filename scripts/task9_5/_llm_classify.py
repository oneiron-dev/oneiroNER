#!/usr/bin/env python3
"""LLM-based subtype classification via OpenRouter/DeepSeek.

Thin wrapper for classification only (not extraction). Copies safety patterns from
scripts/task9/annotate.py: strict JSON schema, provider pinning, retries, checkpointing.
"""

import json
import logging
import os
import time

import requests

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "deepseek/deepseek-v3.2"
MAX_RETRIES = 3
BACKOFF_BASE = 30

OPENROUTER_PROVIDERS = {
    "only": ["atlas-cloud", "novita", "siliconflow"],
    "quantizations": ["fp8"],
}

EVENT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "event_classification",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "subtype": {"type": "string", "enum": ["EVENT/Life", "EVENT/General", "EVENT"]},
                "confidence": {"type": "number"},
            },
            "required": ["subtype", "confidence"],
            "additionalProperties": False,
        },
    },
}

REL_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "rel_classification",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "subtype": {
                    "type": "string",
                    "enum": [
                        "RELATIONSHIP_REF/Family", "RELATIONSHIP_REF/Romantic",
                        "RELATIONSHIP_REF/Friend", "RELATIONSHIP_REF/Professional",
                        "RELATIONSHIP_REF/Acquaintance", "RELATIONSHIP_REF",
                        "GENERIC_OR_NON_SPECIFIC",
                    ],
                },
                "confidence": {"type": "number"},
            },
            "required": ["subtype", "confidence"],
            "additionalProperties": False,
        },
    },
}

SPECIFICITY_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "specificity_check",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "is_specific": {"type": "boolean"},
                "confidence": {"type": "number"},
            },
            "required": ["is_specific", "confidence"],
            "additionalProperties": False,
        },
    },
}


def _get_api_key():
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Export it or add to .env file."
        )
    return key


def _call_openrouter(messages, schema, model=DEFAULT_MODEL):
    key = _get_api_key()
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "response_format": schema,
        "provider": OPENROUTER_PROVIDERS,
    }

    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.post(
                OPENROUTER_URL,
                headers=headers,
                json=body,
                timeout=60,
            )
            if resp.status_code == 429:
                if attempt < MAX_RETRIES:
                    wait = BACKOFF_BASE * (2 ** attempt)
                    logger.warning("Rate limited (attempt %d/%d), waiting %ds", attempt + 1, MAX_RETRIES + 1, wait)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()

            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return json.loads(content)

        except requests.exceptions.HTTPError as e:
            err = str(e).lower()
            if ("429" in err or "rate" in err) and attempt < MAX_RETRIES:
                wait = BACKOFF_BASE * (2 ** attempt)
                logger.warning("Rate limit error (attempt %d/%d), waiting %ds", attempt + 1, MAX_RETRIES + 1, wait)
                time.sleep(wait)
                continue
            if attempt < MAX_RETRIES:
                logger.warning("HTTP error (attempt %d/%d): %s", attempt + 1, MAX_RETRIES + 1, e)
                time.sleep(2)
                continue
            raise

        except Exception as e:
            if attempt < MAX_RETRIES:
                logger.warning("Request error (attempt %d/%d): %s", attempt + 1, MAX_RETRIES + 1, e)
                time.sleep(2)
                continue
            raise

    raise RuntimeError(f"OpenRouter call failed after {MAX_RETRIES + 1} attempts")


def classify_event_subtype(surface, context, model=DEFAULT_MODEL):
    messages = [
        {
            "role": "user",
            "content": (
                "You are classifying an already-detected EVENT span into a subtype.\n"
                "Do not find new entities. Do not change boundaries.\n"
                f'Span: "{surface}"\n'
                f'Context: "...{context}..."\n'
                "Choose exactly one: EVENT/Life, EVENT/General, EVENT\n"
                "EVENT/Life = personal milestones (weddings, funerals, births, divorces, graduations, surgeries, pregnancies)\n"
                "EVENT/General = holidays, festivals, concerts, parties, meetings, competitions, social gatherings\n"
                "If uncertain, return EVENT."
            ),
        }
    ]
    result = _call_openrouter(messages, EVENT_SCHEMA, model)
    logger.debug("classify_event: surface=%r -> %s (%.2f)", surface, result["subtype"], result["confidence"])
    return result["subtype"], result["confidence"]


def classify_rel_subtype(surface, context, model=DEFAULT_MODEL):
    messages = [
        {
            "role": "user",
            "content": (
                "You are classifying an already-detected RELATIONSHIP_REF span into a subtype.\n"
                "Do not find new entities. Do not change boundaries.\n"
                f'Span: "{surface}"\n'
                f'Context: "...{context}..."\n'
                "Choose exactly one: RELATIONSHIP_REF/Family, RELATIONSHIP_REF/Romantic, "
                "RELATIONSHIP_REF/Friend, RELATIONSHIP_REF/Professional, "
                "RELATIONSHIP_REF/Acquaintance, RELATIONSHIP_REF, GENERIC_OR_NON_SPECIFIC\n"
                "GENERIC_OR_NON_SPECIFIC = the span refers to a hypothetical, generic, or "
                "non-specific person/role, not a real individual.\n"
                "If uncertain, return RELATIONSHIP_REF."
            ),
        }
    ]
    result = _call_openrouter(messages, REL_SCHEMA, model)
    subtype = result["subtype"]
    confidence = result["confidence"]
    if subtype == "GENERIC_OR_NON_SPECIFIC":
        generic_flag = True
        subtype = "RELATIONSHIP_REF"
    else:
        generic_flag = False
    logger.debug("classify_rel: surface=%r -> %s (%.2f, generic=%s)", surface, subtype, confidence, generic_flag)
    return subtype, confidence, generic_flag


def classify_is_specific_ref(surface, context, model=DEFAULT_MODEL):
    messages = [
        {
            "role": "user",
            "content": (
                "Does the following span refer to a specific individual "
                "(not a hypothetical, generic, or abstract reference)?\n"
                f'Span: "{surface}"\n'
                f'Context: "...{context}..."\n'
                "Answer true if it refers to a specific person the speaker knows, "
                "false if generic/hypothetical/abstract."
            ),
        }
    ]
    result = _call_openrouter(messages, SPECIFICITY_SCHEMA, model)
    logger.debug("classify_specific: surface=%r -> %s (%.2f)", surface, result["is_specific"], result["confidence"])
    return result["is_specific"], result["confidence"]
