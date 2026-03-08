#!/usr/bin/env python3
"""Prepare batch files for Task 8 coding-agent annotation.

Reads JP-RP and ChatHaruhi datasets, converts to clean turn format,
and splits into batches sized by total characters (not count) to fit
within agent context windows.

Usage:
    python scripts/task8/prepare_batches.py [--max-bytes 300000] [--skip-oversized 500000]
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASE_DIR = Path(__file__).parent.parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
BATCH_DIR = BASE_DIR / "data" / "task8_batches"

JP_RP_PATH = RAW_DIR / "Japanese-Roleplay-Dialogues" / "Japanese-Roleplay-Dialogues-Filtered.jsonl"
CHATHARUHI_DIR = RAW_DIR / "ChatHaruhi-RolePlaying"

EN_CHARACTERS = {
    "Harry", "Hermione", "Sheldon", "Luna", "Malfoy",
    "McGonagall", "Penny", "Raj", "Ron", "Snape", "Dumbledore",
}


def parse_jp_roleplay() -> list[dict]:
    records = []
    with open(JP_RP_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            poster_map = {}
            first_poster = raw["first_poster"]
            poster_map[first_poster] = "A"

            turns = []
            for post in raw["posts"]:
                poster = post["poster"]
                if poster not in poster_map:
                    label = chr(ord("A") + len(poster_map))
                    if len(poster_map) >= 26:
                        label = f"Speaker{len(poster_map) + 1}"
                    poster_map[poster] = label
                turns.append({
                    "speaker": poster_map[poster],
                    "text": post["post_content"],
                })

            if len(turns) < 2:
                continue

            rec = {
                "id": f"jp_rp_{raw['id']}",
                "title": raw["title"],
                "turns": turns,
                "language": "ja",
                "source": "jp_roleplay",
            }
            rec["_size"] = len(json.dumps(rec, ensure_ascii=False).encode("utf-8"))
            records.append(rec)
    return records


def parse_chatharuhi() -> list[dict]:
    all_records = []
    for path in sorted(CHATHARUHI_DIR.glob("*.jsonl")):
        char_name = path.stem
        if char_name == "README":
            continue
        language = "en" if char_name in EN_CHARACTERS else "zh"

        with open(path) as f:
            lines = f.readlines()

        for idx, raw_line in enumerate(lines):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                d = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if d.get("luotuo_openai") in ("system_prompt", "config"):
                continue
            if d.get("bge_zh_s15") in ("system_prompt", "config"):
                continue

            text = d.get("text", "")
            if not text:
                continue

            turns = []
            for turn_line in text.split("\n"):
                turn_line = turn_line.strip()
                if not turn_line:
                    continue
                sep_idx = -1
                for i, ch in enumerate(turn_line):
                    if ch in (":", "："):
                        sep_idx = i
                        break
                if sep_idx == -1:
                    if turns:
                        turns[-1]["text"] += " " + turn_line
                    continue

                speaker = turn_line[:sep_idx].strip()
                utterance = turn_line[sep_idx + 1:].strip().strip("「」")
                if not utterance:
                    continue
                turns.append({"speaker": speaker, "text": utterance})

            if len(turns) < 2:
                continue

            rec = {
                "id": f"ch_{char_name}_{idx}",
                "turns": turns,
                "language": language,
                "source": "chatharuhi",
            }
            rec["_size"] = len(json.dumps(rec, ensure_ascii=False).encode("utf-8"))
            all_records.append(rec)

    return all_records


def pack_batches(records: list[dict], max_bytes: int, skip_threshold: int) -> tuple[list[list[dict]], list[dict], list[dict]]:
    normal = [r for r in records if r["_size"] <= skip_threshold]
    oversized = [r for r in records if r["_size"] > skip_threshold]

    normal.sort(key=lambda r: r["_size"])

    batches = []
    current_batch = []
    current_size = 0

    for rec in normal:
        s = rec["_size"]
        if current_size + s > max_bytes and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_size = 0
        current_batch.append(rec)
        current_size += s

    if current_batch:
        batches.append(current_batch)

    solo = []
    for rec in oversized:
        if rec["_size"] <= max_bytes * 3:
            solo.append(rec)

    return batches, solo, [r for r in oversized if r["_size"] > max_bytes * 3]


def write_batch(records: list[dict], path: Path):
    clean = []
    for r in records:
        out = {k: v for k, v in r.items() if not k.startswith("_")}
        clean.append(out)
    with open(path, "w") as f:
        json.dump(clean, f, ensure_ascii=False, separators=(",", ":"))


def main():
    parser = argparse.ArgumentParser(description="Prepare Task 8 annotation batches")
    parser.add_argument("--max-bytes", type=int, default=300_000,
                        help="Target max JSON bytes per batch (default 300KB)")
    parser.add_argument("--skip-oversized", type=int, default=600_000,
                        help="Skip convos larger than this (default 600KB)")
    args = parser.parse_args()

    if BATCH_DIR.exists():
        shutil.rmtree(BATCH_DIR)
    BATCH_DIR.mkdir(parents=True, exist_ok=True)

    def run_source(name, records, subdir):
        batches, solo, skipped = pack_batches(records, args.max_bytes, args.skip_oversized)
        out_dir = BATCH_DIR / subdir
        out_dir.mkdir(exist_ok=True)
        for i, batch in enumerate(batches):
            write_batch(batch, out_dir / f"batch_{i:03d}.json")
        for i, rec in enumerate(solo):
            write_batch([rec], out_dir / f"solo_{i:03d}.json")

        total = sum(len(b) for b in batches) + len(solo)
        print(f"  {len(batches)} batches + {len(solo)} solo = {total} convos")
        print(f"  {len(skipped)} skipped (>{args.skip_oversized} bytes)")
        if batches:
            sizes = [len(b) for b in batches]
            print(f"  convos/batch: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)/len(sizes):.0f}")
            file_sizes = []
            for i in range(len(batches)):
                file_sizes.append((out_dir / f"batch_{i:03d}.json").stat().st_size)
            print(f"  file KB: min={min(file_sizes)//1024}, max={max(file_sizes)//1024}, avg={sum(file_sizes)//len(file_sizes)//1024}")
        return total

    print("Parsing JP-Roleplay...")
    jp_records = parse_jp_roleplay()
    print(f"  {len(jp_records)} conversations parsed")
    jp_total = run_source("jp_roleplay", jp_records, "jp_roleplay")

    print("\nParsing ChatHaruhi...")
    ch_records = parse_chatharuhi()
    print(f"  {len(ch_records)} conversations parsed")
    ch_total = run_source("chatharuhi", ch_records, "chatharuhi")

    n_files = sum(1 for _ in BATCH_DIR.rglob("*.json"))
    print(f"\nTotal: {n_files} batch files, {jp_total + ch_total} conversations")
    print(f"Output: {BATCH_DIR}")


if __name__ == "__main__":
    main()
