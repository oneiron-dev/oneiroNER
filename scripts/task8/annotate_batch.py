#!/usr/bin/env python3
"""Generate annotation prompts for Task 8 batch files.

Usage:
    python scripts/task8/annotate_batch.py data/task8_batches/jp_roleplay/batch_000.json
    python scripts/task8/annotate_batch.py --list jp_roleplay
    python scripts/task8/annotate_batch.py --list chatharuhi
"""

import argparse
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
BATCH_DIR = BASE_DIR / "data" / "task8_batches"
OUTPUT_DIR = BASE_DIR / "data" / "task8_output"

ENTITY_TYPES_WITH_EXAMPLES = {
    "PERSON": "character names, real people (及川, マッキー, Harry, 邓布利多)",
    "PLACE": "locations, buildings, rooms (武道館, 教室, Hogwarts, 东京, Baker Street)",
    "ORG": "organizations, teams, schools (CRYSTAL, 青城, Gryffindor, Ministry of Magic)",
    "DATE/Day": "specific days (月曜日, 今日, Tuesday)",
    "DATE/Week": "week references (今週, 先週, this week)",
    "DATE/Month": "months (三月, 先月, February)",
    "DATE/Season": "seasons (夏, 冬休み, summer)",
    "DATE/Year": "years (2024年, 去年, 1935)",
    "DATE/Decade": "decades (90年代, the 80s)",
    "DATE/Relative": "relative time (この間, さっき, 明日, the other day, tomorrow)",
    "DATE/Range": "time ranges (夏休み中, from March to May)",
    "EVENT": "events, occasions, holidays (誕生日, 文化祭, 試合, Christmas, Triwizard Tournament)",
    "RELATIONSHIP_REF/Family": "family terms (お母さん, 兄, 妹, mom, uncle, cousin)",
    "RELATIONSHIP_REF/Romantic": "romantic references (彼氏, 彼女, 恋人, boyfriend, crush)",
    "RELATIONSHIP_REF/Friend": "friend references (幼なじみ, 親友, best friend, childhood friend)",
    "RELATIONSHIP_REF/Professional": "professional roles (先輩, 後輩, 先生, coach, Captain, prefect)",
    "RELATIONSHIP_REF/Acquaintance": "acquaintance terms (知り合い, 隣人, neighbor)",
    "EMOTION": "expressed feelings (嬉しい, 悲しい, 不安, 寂しい, happy, worried, angry)",
    "GOAL": "stated intentions/desires (奢ってあげる, 話したい, 頑張る, want to win, planning to visit)",
    "ACTIVITY": "actions/activities being done (練習, シュークリーム食べる, Quidditch, studying, cooking)",
}

PROMPT_TEMPLATE = """Read the file {batch_path}.

It contains {n_convos} roleplay conversations in JSON format. Each conversation has an "id", "language" (ja/en/zh), "source", and "turns" array where each turn has "speaker" and "text".

Annotate ALL entities in every conversation. You MUST find entities across ALL types, not just PERSON. Aim for diverse type coverage.

Entity types with examples:
{type_list}

IMPORTANT: Do NOT over-index on PERSON. Actively look for ALL types in each turn. Roleplay conversations are rich in RELATIONSHIP_REF (family terms, honorifics), EMOTION (feelings expressed), ACTIVITY (actions described), GOAL (intentions), DATE (time references), PLACE (locations), and EVENT (occasions).


For each entity found, record:
- "surface": the exact text span as it appears in the turn
- "type": one of the types above (use the exact string including / for subtypes)
- "turn_index": 0-based index of the turn containing the entity
- "start": character offset within that turn's "text" where the entity starts
- "end": character offset where the entity ends (start + len(surface))

CRITICAL OFFSET RULES:
- Compute start using: turns[turn_index]["text"].find(surface)
- Compute end as: start + len(surface)
- Verify: turns[turn_index]["text"][start:end] == surface
- If find() returns -1, the surface string is wrong — fix it
- For Japanese text, len() counts characters (codepoints), not bytes — this is correct

Write your results to {output_path} as a JSON array:
[
  {{
    "id": "conversation_id",
    "entities": [
      {{"surface": "東京", "type": "PLACE", "turn_index": 3, "start": 5, "end": 7}},
      ...
    ]
  }},
  ...
]

Include ALL conversations in the output, even those with 0 entities (empty entities array).

After writing, verify a sample of offsets by reading back the output and checking turns[ti]["text"][start:end] == surface for at least 20 entities. Report the verification result.
"""


def generate_prompt(batch_path: str) -> str:
    import json
    batch_path = os.path.abspath(batch_path)
    with open(batch_path) as f:
        data = json.load(f)
    n_convos = len(data)

    rel = os.path.relpath(batch_path, BASE_DIR)
    stem = Path(batch_path).stem
    source_dir = Path(batch_path).parent.name
    output_path = os.path.abspath(str(OUTPUT_DIR / source_dir / f"{stem}.json"))

    type_list = "\n".join(f"- {t}: {ex}" for t, ex in ENTITY_TYPES_WITH_EXAMPLES.items())

    return PROMPT_TEMPLATE.format(
        batch_path=batch_path,
        n_convos=n_convos,
        type_list=type_list,
        output_path=output_path,
    )


def list_batches(source: str) -> list[str]:
    source_dir = BATCH_DIR / source
    if not source_dir.exists():
        print(f"Source dir not found: {source_dir}")
        return []
    files = sorted(source_dir.glob("*.json"))
    return [str(f) for f in files]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_file", nargs="?", help="Path to batch file")
    parser.add_argument("--list", dest="list_source", help="List batch files for source")
    parser.add_argument("--show-prompt", action="store_true", help="Print the prompt")
    args = parser.parse_args()

    if args.list_source:
        files = list_batches(args.list_source)
        done_dir = OUTPUT_DIR / args.list_source
        for f in files:
            stem = Path(f).stem
            out = done_dir / f"{stem}.json"
            status = "DONE" if out.exists() else "TODO"
            print(f"[{status}] {f}")
        total = len(files)
        done = sum(1 for f in files if (done_dir / f"{Path(f).stem}.json").exists())
        print(f"\n{done}/{total} complete")
        return

    if args.batch_file:
        prompt = generate_prompt(args.batch_file)
        if args.show_prompt:
            print(prompt)
        else:
            print(prompt)


if __name__ == "__main__":
    main()
