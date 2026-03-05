#!/usr/bin/env python3
"""Task 9 Step 1: parse silver sources, keyword-filter for RELATIONSHIP_REF density, normalize, window."""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.windower import window_turns

BASE = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "silver_sources"
OUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "silver_filtered"

# Relationship keywords for prefilter
KEYWORDS_EN = [
    "mom", "dad", "mother", "father", "sister", "brother", "aunt", "uncle",
    "grandma", "grandpa", "grandmother", "grandfather", "cousin",
    "husband", "wife", "spouse", "partner", "boyfriend", "girlfriend",
    "friend", "best friend", "boss", "manager", "coworker", "colleague",
    "mentor", "teacher", "professor", "doctor", "therapist",
    "neighbor", "roommate", "ex", r"fianc[eé]e?",
    "son", "daughter", "niece", "nephew",
    "stepmother", "stepfather", "in-law", "mother-in-law", "father-in-law",
]
KEYWORDS_ZH = [
    "妈妈", "爸爸", "姐姐", "哥哥", "弟弟", "妹妹", "奶奶", "爷爷",
    "老婆", "老公", "朋友", "同事", "老师", "老板",
]
KEYWORDS_JA = [
    "お母さん", "お父さん", "姉", "兄", "弟", "妹", "祖母", "祖父",
    "妻", "夫", "友達", "先生", "上司", "同僚",
]
KEYWORDS_KO = [
    "엄마", "아빠", "언니", "오빠", "동생", "할머니", "할아버지",
    "아내", "남편", "친구", "선생님", "상사",
]

_KW_PATTERN = re.compile(
    r"\b(?:" + "|".join(KEYWORDS_EN) + r")\b|"
    + "|".join(KEYWORDS_ZH + KEYWORDS_JA + KEYWORDS_KO),
    re.IGNORECASE,
)

_URL_RE = re.compile(r"https?://\S+")
_MARKDOWN_RE = re.compile(r"[*_~`#>\[\]()!]")
_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")

ALL_SOURCES = [
    "reddit_confessions", "opencharacter", "therapy_conversations",
    "prosocial_dialog", "personachat", "pippa",
    "synthetic_persona_chat", "roleplay_hieu", "mentalchat",
]


def clean_text(text: str) -> str:
    text = _URL_RE.sub("", text)
    text = _MARKDOWN_RE.sub("", text)
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def has_keyword(text: str) -> bool:
    return bool(_KW_PATTERN.search(text))


def concat_text_from_turns(turns: list[dict]) -> str:
    return " ".join(t["text"] for t in turns)


# --- Source parsers ---
# Each yields dicts: {"format": "passage"|"conversation", "text": str} or {"format": "conversation", "turns": [{"speaker":..,"text":..}]}


def parse_reddit_confessions(path: Path, limit: int | None):
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            rec = json.loads(line)
            text = rec.get("selftext", "")
            if not text or text in ("[removed]", "[deleted]"):
                continue
            text = clean_text(text)
            if len(text) < 50:
                continue
            yield {"format": "passage", "text": text, "source_id": rec.get("id", str(i))}


def parse_opencharacter(path: Path, limit: int | None):
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            rec = json.loads(line)
            text = clean_text(rec.get("character_answer", ""))
            if len(text) < 50:
                continue
            yield {"format": "passage", "text": text, "source_id": rec.get("question_id", str(i))}


def parse_therapy_conversations(path: Path, limit: int | None):
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            rec = json.loads(line)
            inp = clean_text(rec.get("input", ""))
            out = clean_text(rec.get("output", ""))
            combined = f"{inp}\n\n{out}".strip()
            if len(combined) < 50:
                continue
            yield {"format": "passage", "text": combined, "source_id": str(i)}


def parse_mentalchat(path: Path, limit: int | None):
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            rec = json.loads(line)
            ctx = clean_text(rec.get("Context", ""))
            resp = clean_text(rec.get("Response", ""))
            combined = f"{ctx}\n\n{resp}".strip()
            if len(combined) < 50:
                continue
            yield {"format": "passage", "text": combined, "source_id": str(i)}


def parse_prosocial_dialog(path: Path, limit: int | None):
    dialogues: dict[str, list] = defaultdict(list)
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            rec = json.loads(line)
            did = str(rec.get("dialogue_id", i))
            dialogues[did].append(rec)

    for did, rows in dialogues.items():
        rows.sort(key=lambda r: int(r.get("response_id", 0)))
        turns = []
        for r in rows:
            ctx = clean_text(r.get("context", ""))
            resp = clean_text(r.get("response", ""))
            if ctx:
                turns.append({"speaker": "user", "text": ctx})
            if resp:
                turns.append({"speaker": "assistant", "text": resp})
        if len(turns) < 2:
            continue
        yield {"format": "conversation", "turns": turns, "source_id": did}


def parse_personachat(path: Path, limit: int | None):
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            rec = json.loads(line)
            utts = rec.get("utterances", [])
            if not utts:
                continue
            last = utts[-1]
            history = last.get("history", [])
            candidates = last.get("candidates", [])
            gold_resp = candidates[-1] if candidates else ""
            turns = []
            for j, h in enumerate(history):
                speaker = "user" if j % 2 == 0 else "assistant"
                text = clean_text(h)
                if text:
                    turns.append({"speaker": speaker, "text": text})
            if gold_resp:
                gold_text = clean_text(gold_resp)
                if gold_text:
                    speaker = "user" if len(turns) % 2 == 0 else "assistant"
                    turns.append({"speaker": speaker, "text": gold_text})
            if len(turns) < 2:
                continue
            yield {"format": "conversation", "turns": turns, "source_id": str(i)}


def parse_pippa(path: Path, limit: int | None):
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            rec = json.loads(line)
            conv = rec.get("conversation", [])
            if not conv:
                continue
            turns = []
            for c in conv:
                is_human = str(c.get("is_human", "")).lower() == "true"
                speaker = "user" if is_human else "assistant"
                text = clean_text(c.get("message", ""))
                if text:
                    turns.append({"speaker": speaker, "text": text})
            if len(turns) < 2:
                continue
            yield {"format": "conversation", "turns": turns, "source_id": rec.get("bot_id", str(i))}


_SPC_TURN_RE = re.compile(r"^(User [12]):\s*", re.MULTILINE)


def parse_synthetic_persona_chat(path: Path, limit: int | None):
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            rec = json.loads(line)
            raw = rec.get("Best Generated Conversation", "")
            if not raw:
                continue
            parts = _SPC_TURN_RE.split(raw)
            turns = []
            j = 1
            while j < len(parts) - 1:
                role = parts[j].strip()
                text = clean_text(parts[j + 1].strip())
                if text:
                    speaker = "user" if role == "User 1" else "assistant"
                    turns.append({"speaker": speaker, "text": text})
                j += 2
            if len(turns) < 2:
                continue
            yield {"format": "conversation", "turns": turns, "source_id": str(i)}


_RP_TAG_RE = re.compile(r"<\|(system|user|assistant)\|>")


def parse_roleplay_hieu(path: Path, limit: int | None):
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            rec = json.loads(line)
            text = rec.get("text", "")
            if not text:
                continue
            parts = _RP_TAG_RE.split(text)
            turns = []
            j = 1
            while j < len(parts) - 1:
                role = parts[j]
                content = parts[j + 1].replace("</s>", "").strip()
                content = clean_text(content)
                if content and role in ("user", "assistant"):
                    turns.append({"speaker": role, "text": content})
                j += 2
            if len(turns) < 2:
                continue
            yield {"format": "conversation", "turns": turns, "source_id": rec.get("name", str(i))}


SOURCE_FILES = {
    "reddit_confessions": ("reddit_confessions.jsonl", parse_reddit_confessions),
    "opencharacter": ("opencharacter.jsonl", parse_opencharacter),
    "therapy_conversations": ("therapy_conversations.jsonl", parse_therapy_conversations),
    "prosocial_dialog": ("prosocial_dialog.jsonl", parse_prosocial_dialog),
    "personachat": ("personachat.jsonl", parse_personachat),
    "pippa": ("pippa_deduped.jsonl", parse_pippa),
    "synthetic_persona_chat": ("synthetic_persona_chat.jsonl", parse_synthetic_persona_chat),
    "roleplay_hieu": ("roleplay_hieu.jsonl", parse_roleplay_hieu),
    "mentalchat": ("mentalchat.jsonl", parse_mentalchat),
}


def process_source(name: str, limit: int | None) -> dict:
    filename, parser = SOURCE_FILES[name]
    path = BASE / name / filename
    out_path = OUT_DIR / f"{name}.jsonl"

    stats = {"source": name, "parsed": 0, "matched": 0, "written": 0}

    with open(out_path, "w") as fout:
        for item in parser(path, limit):
            stats["parsed"] += 1

            if item["format"] == "passage":
                if not has_keyword(item["text"]):
                    continue
                stats["matched"] += 1
                rec = {
                    "source": name,
                    "source_id": item["source_id"],
                    "format": "passage",
                    "text": item["text"],
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                stats["written"] += 1

            elif item["format"] == "conversation":
                full_text = concat_text_from_turns(item["turns"])
                if not has_keyword(full_text):
                    continue
                stats["matched"] += 1

                windows = window_turns(item["turns"], size=4, stride=2)
                for wi, window in enumerate(windows):
                    window_text = concat_text_from_turns(window)
                    if not has_keyword(window_text):
                        continue
                    sid = f"{item['source_id']}__w{wi}" if len(windows) > 1 else item["source_id"]
                    rec = {
                        "source": name,
                        "source_id": sid,
                        "format": "conversation",
                        "turns": window,
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    stats["written"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description="Prefilter silver sources for RELATIONSHIP_REF density")
    parser.add_argument("--source", default="all", help="Source name or 'all'")
    parser.add_argument("--limit", type=int, default=None, help="Limit records per source (for testing)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sources = ALL_SOURCES if args.source == "all" else [args.source]
    for s in sources:
        if s not in SOURCE_FILES:
            print(f"Unknown source: {s}", file=sys.stderr)
            sys.exit(1)

    total_stats = {"parsed": 0, "matched": 0, "written": 0}
    for s in sources:
        print(f"Processing {s}...", end=" ", flush=True)
        stats = process_source(s, args.limit)
        print(f"parsed={stats['parsed']:,} matched={stats['matched']:,} written={stats['written']:,}")
        for k in total_stats:
            total_stats[k] += stats[k]

    print(f"\nTotal: parsed={total_stats['parsed']:,} matched={total_stats['matched']:,} written={total_stats['written']:,}")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
