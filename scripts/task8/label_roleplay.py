#!/usr/bin/env python3
"""Task 8: Silver-label roleplay datasets (JP-RP + ChatHaruhi) via LLM NER annotation.

LEGACY: This was the initial annotation pipeline using provider="spark".
The current re-annotation pipeline uses scripts/task8/annotate_batch.py with
GPT-5.4 (primary) and DeepSeek V3.2 (for Sonnet JP-RP re-annotation).
See scripts/task8/README.md for the current pipeline.
"""

import argparse
import json
import logging
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.llm_annotator import annotate_conversation
from lib.negative_sampler import NegativeSampler
from lib.schema import ConversationRecord
from lib.span_fixer import verify_and_fix_spans
from lib.windower import window_turns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_DIR = DATA_DIR / "processed"

JP_RP_PATH = RAW_DIR / "Japanese-Roleplay-Dialogues" / "Japanese-Roleplay-Dialogues-Filtered.jsonl"
CHATHARUHI_DIR = RAW_DIR / "ChatHaruhi-RolePlaying"

EN_CHARACTERS = {
    "Harry", "Hermione", "Sheldon", "Luna", "Malfoy",
    "McGonagall", "Penny", "Raj", "Ron", "Snape", "Dumbledore",
}

SILVER_TYPE_COUNTS = {
    "PERSON": 1000, "PLACE": 1000, "ORG": 1000,
    "DATE/Day": 500, "DATE/Week": 200, "DATE/Month": 500, "DATE/Season": 200,
    "DATE/Year": 500, "DATE/Decade": 100, "DATE/Relative": 500, "DATE/Range": 200,
    "EVENT": 500,
    "RELATIONSHIP_REF/Family": 500, "RELATIONSHIP_REF/Romantic": 300,
    "RELATIONSHIP_REF/Friend": 300, "RELATIONSHIP_REF/Professional": 300,
    "RELATIONSHIP_REF/Acquaintance": 200,
    "EMOTION": 500, "GOAL": 300, "ACTIVITY": 500,
}

write_lock = threading.Lock()
checkpoint_lock = threading.Lock()
stop_event = threading.Event()
stats_lock = threading.Lock()
stats = {"total": 0, "success": 0, "fail": 0, "by_source": {}}


# --- JP-RP parsing ---

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

            records.append({
                "id": str(raw["id"]),
                "title": raw["title"],
                "turns": turns,
                "language": "ja",
                "source": "jp_roleplay",
            })
    return records


# --- ChatHaruhi parsing ---

def parse_chatharuhi_file(path: Path, language: str) -> list[dict]:
    records = []
    with open(path) as f:
        lines = f.readlines()

    char_name = path.stem
    data_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("luotuo_openai") in ("system_prompt", "config"):
            continue
        if d.get("bge_zh_s15") in ("system_prompt", "config"):
            continue
        data_lines.append(d)

    for idx, d in enumerate(data_lines):
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
            utterance = turn_line[sep_idx + 1:].strip()
            if speaker == "旁白":
                speaker = "Narrator"
            utterance = utterance.strip("「」")
            if not utterance:
                continue
            turns.append({"speaker": speaker, "text": utterance})

        if len(turns) < 2:
            continue

        records.append({
            "id": f"{char_name}_{idx}",
            "turns": turns,
            "language": language,
            "source": "chatharuhi",
        })

    return records


def parse_chatharuhi() -> list[dict]:
    all_records = []
    for path in sorted(CHATHARUHI_DIR.glob("*.jsonl")):
        char_name = path.stem
        if char_name == "README":
            continue
        language = "en" if char_name in EN_CHARACTERS else "zh"
        records = parse_chatharuhi_file(path, language)
        all_records.extend(records)
        logger.info("ChatHaruhi %s: %d conversations (%s)", char_name, len(records), language)
    return all_records


# --- Pipeline ---

def load_checkpoint(path: Path) -> set[str]:
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        return set(data.get("processed_ids", []))
    return set()


def save_checkpoint(path: Path, processed_ids: set[str]):
    with checkpoint_lock:
        data = {"processed_ids": sorted(processed_ids), "stats": dict(stats)}
        with open(path, "w") as f:
            json.dump(data, f)


def process_window(
    window: list[dict],
    source_id: str,
    language: str,
    source_name: str,
    neg_sampler: NegativeSampler,
    rng: random.Random,
) -> ConversationRecord | None:
    try:
        raw_entities = annotate_conversation(window, language, provider="spark")
    except Exception as e:
        logger.debug("[%s] annotation failed: %s", source_id, e)
        return None

    if not raw_entities:
        logger.debug("[%s] no entities returned", source_id)
        return None

    fixed, fix_stats = verify_and_fix_spans(raw_entities, text="", turns=window)
    if not fixed:
        logger.debug("[%s] all entities dropped after span fix", source_id)
        return None

    positive_types = {e["type"] for e in fixed}
    negatives = neg_sampler.sample(positive_types, rng=rng)

    provenance = [source_name, "task8"]
    rec = ConversationRecord(
        source=source_name,
        source_id=source_id,
        language=language,
        split="train",
        confidence="silver",
        provenance=provenance,
        turns=window,
        query_types=sorted(positive_types | set(negatives)),
        entities=[e for e in fixed],
    )

    try:
        rec.validate()
    except AssertionError as e:
        logger.debug("[%s] validation failed: %s", source_id, e)
        return None

    return rec


def run_pipeline(
    conversations: list[dict],
    source_name: str,
    output_path: Path,
    checkpoint_path: Path,
    neg_sampler: NegativeSampler,
    seed: int,
    concurrency: int,
    limit: int | None,
    resume: bool,
):
    rng = random.Random(seed)

    processed_ids: set[str] = set()
    if resume:
        processed_ids = load_checkpoint(checkpoint_path)
        logger.info("Resumed with %d already-processed IDs", len(processed_ids))

    work_items = []
    for conv in conversations:
        windows = window_turns(conv["turns"], size=4, stride=2)
        for wi, window in enumerate(windows):
            sid = f"{conv['source']}_{conv['id']}_w{wi}"
            if sid in processed_ids:
                continue
            work_items.append((window, sid, conv["language"]))

    if limit is not None:
        work_items = work_items[:limit]

    total = len(work_items)
    logger.info("Source %s: %d windows to process", source_name, total)

    if total == 0:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_mode = "a" if resume else "w"
    fh = open(output_path, file_mode)

    completed = 0
    start_time = time.time()
    window_seeds = {sid: rng.randint(0, 2**32 - 1) for _, sid, _ in work_items}

    def worker(item: tuple[list[dict], str, str]) -> bool:
        nonlocal completed
        if stop_event.is_set():
            return False

        window, sid, language = item
        per_rng = random.Random(window_seeds[sid])

        rec = process_window(window, sid, language, source_name, neg_sampler, per_rng)

        if rec is not None:
            with write_lock:
                fh.write(rec.to_jsonl() + "\n")
                fh.flush()
                processed_ids.add(sid)
            with stats_lock:
                stats["success"] += 1
                stats["by_source"][source_name] = stats["by_source"].get(source_name, 0) + 1
        else:
            with stats_lock:
                stats["fail"] += 1

        completed += 1
        if completed % 50 == 0:
            elapsed = time.time() - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            with stats_lock:
                s, f = stats["success"], stats["fail"]
            logger.info(
                "[%s] %d/%d (%.0f%%) | ok=%d fail=%d | %.1f/min",
                source_name, completed, total,
                100 * completed / total, s, f, rate * 60,
            )
            save_checkpoint(checkpoint_path, processed_ids)

        return rec is not None

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(worker, item): item for item in work_items}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error("Worker exception: %s", e)

    fh.close()
    save_checkpoint(checkpoint_path, processed_ids)

    with stats_lock:
        s, f = stats["success"], stats["fail"]
    logger.info("[%s] Done: %d success, %d fail", source_name, s, f)


def main():
    parser = argparse.ArgumentParser(description="Task 8: Silver-label roleplay datasets")
    parser.add_argument("--source", choices=["jp_roleplay", "chatharuhi", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None, help="Max windows to process (for testing)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    neg_sampler = NegativeSampler(type_counts=SILVER_TYPE_COUNTS)

    if args.source in ("jp_roleplay", "all"):
        logger.info("Parsing JP-Roleplay...")
        jp_convs = parse_jp_roleplay()
        logger.info("JP-Roleplay: %d conversations", len(jp_convs))
        run_pipeline(
            conversations=jp_convs,
            source_name="jp_roleplay",
            output_path=OUTPUT_DIR / "task8_jp_roleplay.jsonl",
            checkpoint_path=OUTPUT_DIR / ".task8_jp_roleplay_checkpoint.json",
            neg_sampler=neg_sampler,
            seed=args.seed,
            concurrency=args.concurrency,
            limit=args.limit,
            resume=args.resume,
        )

    if args.source in ("chatharuhi", "all"):
        stats["success"] = 0
        stats["fail"] = 0
        logger.info("Parsing ChatHaruhi...")
        ch_convs = parse_chatharuhi()
        logger.info("ChatHaruhi: %d conversations", len(ch_convs))
        run_pipeline(
            conversations=ch_convs,
            source_name="chatharuhi",
            output_path=OUTPUT_DIR / "task8_chatharuhi.jsonl",
            checkpoint_path=OUTPUT_DIR / ".task8_chatharuhi_checkpoint.json",
            neg_sampler=neg_sampler,
            seed=args.seed + 1,
            concurrency=args.concurrency,
            limit=args.limit,
            resume=args.resume,
        )

    logger.info("Task 8 complete. Output in %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
