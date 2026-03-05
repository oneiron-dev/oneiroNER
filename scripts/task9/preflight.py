#!/usr/bin/env python3
"""Task 9 Step 0: verify all 9 silver source corpora exist and parse correctly."""

import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "silver_sources"

SOURCES = {
    "reddit_confessions": {
        "file": "reddit_confessions.jsonl",
        "required_fields": ["selftext"],
        "format": "passage",
    },
    "opencharacter": {
        "file": "opencharacter.jsonl",
        "required_fields": ["character_answer"],
        "format": "passage",
    },
    "therapy_conversations": {
        "file": "therapy_conversations.jsonl",
        "required_fields": ["input", "output"],
        "format": "passage",
    },
    "prosocial_dialog": {
        "file": "prosocial_dialog.jsonl",
        "required_fields": ["context", "response"],
        "format": "conversation",
    },
    "personachat": {
        "file": "personachat.jsonl",
        "required_fields": ["utterances"],
        "format": "conversation",
    },
    "pippa": {
        "file": "pippa_deduped.jsonl",
        "required_fields": ["conversation"],
        "format": "conversation",
    },
    "synthetic_persona_chat": {
        "file": "synthetic_persona_chat.jsonl",
        "required_fields": ["Best Generated Conversation"],
        "format": "conversation",
    },
    "roleplay_hieu": {
        "file": "roleplay_hieu.jsonl",
        "required_fields": ["text"],
        "format": "conversation",
    },
    "mentalchat": {
        "file": "mentalchat.jsonl",
        "required_fields": ["Context", "Response"],
        "format": "passage",
    },
}


def count_lines(path: Path) -> int:
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def check_source(name: str, spec: dict) -> dict:
    result = {"name": name, "ok": True, "errors": []}
    path = BASE / name / spec["file"]

    if not path.exists():
        result["ok"] = False
        result["errors"].append(f"File not found: {path}")
        return result

    result["path"] = str(path)

    # Sample first 10 records
    sample_records = []
    parse_errors = 0
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= 10:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                sample_records.append(rec)
            except json.JSONDecodeError as e:
                parse_errors += 1
                result["errors"].append(f"Line {i}: JSON parse error: {e}")

    if parse_errors > 0:
        result["ok"] = False
        return result

    if not sample_records:
        result["ok"] = False
        result["errors"].append("No records found")
        return result

    # Check required fields
    all_fields = set()
    for rec in sample_records:
        all_fields.update(rec.keys())
        for field in spec["required_fields"]:
            if field not in rec:
                result["ok"] = False
                result["errors"].append(f"Missing required field '{field}'")
                return result

    result["sample_fields"] = sorted(all_fields)
    result["format"] = spec["format"]

    # Validate nested structure for specific sources
    first = sample_records[0]
    if name == "personachat":
        utts = first.get("utterances", [])
        if not isinstance(utts, list) or len(utts) == 0:
            result["ok"] = False
            result["errors"].append("utterances is not a non-empty list")
            return result
        u0 = utts[0]
        if "history" not in u0 or "candidates" not in u0:
            result["ok"] = False
            result["errors"].append(f"utterances[0] missing history/candidates, got: {list(u0.keys())}")
            return result

    elif name == "pippa":
        conv = first.get("conversation", [])
        if not isinstance(conv, list) or len(conv) == 0:
            result["ok"] = False
            result["errors"].append("conversation is not a non-empty list")
            return result
        c0 = conv[0]
        if "message" not in c0:
            result["ok"] = False
            result["errors"].append(f"conversation[0] missing 'message', got: {list(c0.keys())}")
            return result

    # Count total records
    result["record_count"] = count_lines(path)
    return result


def main():
    print(f"Silver source preflight — checking {len(SOURCES)} sources")
    print(f"Base path: {BASE}\n")

    all_ok = True
    results = []

    for name, spec in SOURCES.items():
        r = check_source(name, spec)
        results.append(r)
        status = "OK" if r["ok"] else "FAIL"
        print(f"  [{status}] {name}")
        if r.get("record_count"):
            print(f"         records: {r['record_count']:,}")
        if r.get("sample_fields"):
            print(f"         fields:  {r['sample_fields']}")
        print(f"         format:  {spec['format']}")
        if r["errors"]:
            for err in r["errors"]:
                print(f"         ERROR: {err}")
        if not r["ok"]:
            all_ok = False
        print()

    if all_ok:
        print("All sources OK.")
    else:
        failed = [r["name"] for r in results if not r["ok"]]
        print(f"FAILED sources: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
