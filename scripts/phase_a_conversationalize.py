"""Phase A: Conversationalize B2NERD passages into 2-4 turn dialogues.

LLM tool benchmark (2026-02-14, same prompt, 4-entity record):
┌──────────┬────────┬───────┬──────────────────────────────────┬────────┐
│ Tool     │ Time   │ Turns │ Quality                          │ Cost   │
├──────────┼────────┼───────┼──────────────────────────────────┼────────┤
│ Gemini   │ 19.5s  │ 3-4   │ Natural, creative, best spread   │ Free   │
│ Sonnet   │ 11.0s  │ 3-4   │ Natural, creative, preamble+fence│ Paid   │
│ K2.5     │  7.8s  │ 3     │ Natural, good spread, clean JSON │ Free   │
│ Haiku    │  7.3s  │ 2     │ Concise, valid, fences           │ Paid   │
│ Spark    │  4.3s  │ 2     │ Formulaic copy-paste, dup output │ Paid   │
│ Codex    │ 10.4s  │ 2     │ Like spark but slower             │ Paid   │
└──────────┴────────┴───────┴──────────────────────────────────┴────────┘

Strategy: Gemini/Sonnet/K2.5 primary (round-robin), Haiku/Spark fast
fallback, Codex normal last resort.
"""

import argparse
import json
import logging
import os
import re
import subprocess
import tempfile
import threading
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import random

from lib.schema import ConversationRecord
from lib.negative_sampler import NegativeSampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
INPUT_EN = DATA_DIR / "processed" / "b2nerd_en.jsonl"
INPUT_ZH = DATA_DIR / "processed" / "b2nerd_zh.jsonl"
OUTPUT_DIR = DATA_DIR / "labeled" / "phase_a"
CHECKPOINT_FILE = OUTPUT_DIR / ".checkpoint.json"

OPENCODE_BIN = "/home/ubuntu/.opencode/bin/opencode"

PROMPT_TEMPLATE = """\
You are a data augmentation assistant. Given a passage and its gold entity annotations, rewrite the passage as a natural 2-4 turn dialogue between speakers A and B.

RULES:
1. You MUST include each entity surface string at least once, exactly as written.
2. Do NOT paraphrase, abbreviate, or modify entity names.
3. Do NOT invent new entities.
4. Return ONLY valid JSON, no markdown fences.
5. The dialogue MUST be in the same language as the input passage.

PASSAGE:
{text}

ENTITIES TO PRESERVE (surface strings with required minimum counts):
{entity_surfaces_with_counts}

Return JSON:
{{"turns": [{{"speaker": "A", "text": "..."}}, ...], "entity_locations": [{{"surface": "...", "turn_index": 0}}, ...]}}\
"""

sem_gemini = threading.Semaphore(5)
sem_opencode = threading.Semaphore(3)
sem_sonnet = threading.Semaphore(70)
sem_haiku = threading.Semaphore(1)
sem_spark = threading.Semaphore(1)
sem_codex = threading.Semaphore(1)

write_lock = threading.Lock()
checkpoint_lock = threading.Lock()
stop_event = threading.Event()

COOLDOWN_FAILS = 3
COOLDOWN_SECONDS = 2 * 3600
SONNET_STOP_LIMIT = 10


class ToolCircuitBreaker:
    def __init__(self):
        self._lock = threading.Lock()
        self._consecutive_fails: dict[str, int] = {}
        self._disabled_until: dict[str, float] = {}
        self._sonnet_consecutive = 0

    def is_available(self, tool_name: str) -> bool:
        with self._lock:
            until = self._disabled_until.get(tool_name, 0)
            if until and time.time() < until:
                return False
            if until and time.time() >= until:
                self._disabled_until.pop(tool_name, None)
                self._consecutive_fails[tool_name] = 0
            return True

    def record_success(self, tool_name: str):
        with self._lock:
            self._consecutive_fails[tool_name] = 0
            if tool_name == "sonnet":
                self._sonnet_consecutive = 0

    def record_failure(self, tool_name: str):
        with self._lock:
            if tool_name in self._disabled_until:
                return
            self._consecutive_fails[tool_name] = self._consecutive_fails.get(tool_name, 0) + 1
            count = self._consecutive_fails[tool_name]
            if count >= COOLDOWN_FAILS:
                self._disabled_until[tool_name] = time.time() + COOLDOWN_SECONDS
                logger.warning("%s disabled for %dh after %d consecutive failures",
                               tool_name, COOLDOWN_SECONDS // 3600, count)
            if tool_name == "sonnet":
                self._sonnet_consecutive += 1
                if self._sonnet_consecutive >= SONNET_STOP_LIMIT:
                    logger.error("Sonnet hit %d consecutive failures — stopping run", SONNET_STOP_LIMIT)
                    stop_event.set()


circuit_breaker = ToolCircuitBreaker()

stats_lock = threading.Lock()
stats = {
    "total": 0,
    "success": 0,
    "fail": 0,
    "retries": 0,
    "by_model": {
        "gemini": 0, "sonnet": 0, "k2.5": 0,
        "haiku": 0, "spark": 0, "codex": 0,
    },
    "by_language": {"en": 0, "zh": 0},
    "times": [],
}


def build_prompt(text: str, entities: list[dict]) -> str:
    surface_counts = Counter(e["surface"] for e in entities)
    parts = [f'"{s}" (x{c})' for s, c in surface_counts.items()]
    return PROMPT_TEMPLATE.format(
        text=text,
        entity_surfaces_with_counts=", ".join(parts),
    )


def call_gemini(prompt: str) -> str | None:
    with sem_gemini:
        try:
            r = subprocess.run(
                ["gemini", "-p", "", "--yolo", "-o", "text"],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if r.returncode != 0:
                logger.warning("gemini failed (rc=%d): %s", r.returncode, r.stderr[:200])
                return None
            return r.stdout.strip()
        except subprocess.TimeoutExpired:
            logger.warning("gemini timed out")
            return None
        except Exception as e:
            logger.warning("gemini error: %s", e)
            return None


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


def call_k25(prompt: str) -> str | None:
    with sem_opencode:
        try:
            r = subprocess.run(
                [OPENCODE_BIN, "run", "-m", "opencode/kimi-k2.5-free", prompt],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if r.returncode != 0:
                logger.warning("k2.5 failed (rc=%d): %s", r.returncode, r.stderr[:200])
                return None
            raw = r.stdout.strip()
            return _parse_opencode_jsonl(raw)
        except subprocess.TimeoutExpired:
            logger.warning("k2.5 timed out")
            return None
        except Exception as e:
            logger.warning("k2.5 error: %s", e)
            return None


def _call_claude(prompt: str, model: str, sem: threading.Semaphore, label: str) -> str | None:
    with sem:
        try:
            env = dict(os.environ)
            env.pop("CLAUDECODE", None)
            r = subprocess.run(
                ["claude", "-p", prompt, "--model", model, "--output-format", "text"],
                capture_output=True,
                text=True,
                timeout=120,
                env=env,
            )
            if r.returncode != 0:
                logger.warning("%s failed (rc=%d): %s", label, r.returncode, r.stderr[:200])
                return None
            return r.stdout.strip()
        except subprocess.TimeoutExpired:
            logger.warning("%s timed out", label)
            return None
        except Exception as e:
            logger.warning("%s error: %s", label, e)
            return None


def call_sonnet(prompt: str) -> str | None:
    return _call_claude(prompt, "claude-sonnet-4-5-20250929", sem_sonnet, "sonnet")


def call_haiku(prompt: str) -> str | None:
    return _call_claude(prompt, "claude-haiku-4-5-20251001", sem_haiku, "haiku")


def _call_codex_model(prompt: str, model: str, sem: threading.Semaphore, label: str) -> str | None:
    tmppath = None
    with sem:
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, dir="/tmp"
            ) as tf:
                tmppath = tf.name

            cmd = [
                "codex", "exec",
                "--skip-git-repo-check",
                "-s", "read-only",
                "-o", tmppath,
            ]
            if model:
                cmd.extend(["--model", model])
            cmd.append("-")

            r = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if r.returncode != 0:
                logger.warning("%s failed (rc=%d): %s", label, r.returncode, r.stderr[:200])
                return None

            if os.path.exists(tmppath):
                with open(tmppath) as f:
                    result = f.read().strip()
                return result if result else None
            return None
        except subprocess.TimeoutExpired:
            logger.warning("%s timed out", label)
            return None
        except Exception as e:
            logger.warning("%s error: %s", label, e)
            return None
        finally:
            if tmppath and os.path.exists(tmppath):
                try:
                    os.unlink(tmppath)
                except OSError:
                    pass


def call_spark(prompt: str) -> str | None:
    return _call_codex_model(prompt, "gpt-5.3-codex-spark", sem_spark, "spark")


def call_codex(prompt: str) -> str | None:
    return _call_codex_model(prompt, "", sem_codex, "codex")


def clean_llm_output(raw: str) -> dict | None:
    cleaned = re.sub(r"^```(?:json)?\n?", "", raw.strip())
    cleaned = re.sub(r"\n?```$", "", cleaned)
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def validate_llm_response(parsed: dict, gold_surfaces: Counter) -> bool:
    if "turns" not in parsed:
        return False

    turns = parsed["turns"]
    if not (2 <= len(turns) <= 4):
        return False

    for turn in turns:
        if not isinstance(turn, dict):
            return False
        if not turn.get("speaker") or not turn.get("text"):
            return False

    if "entity_locations" in parsed:
        llm_surfaces = {loc.get("surface") for loc in parsed["entity_locations"] if isinstance(loc, dict)}
        hallucinated = llm_surfaces - set(gold_surfaces.keys())
        if hallucinated:
            logger.debug("Hallucinated entities: %s", hallucinated)
            return False

    full_text = " ".join(t["text"] for t in turns)
    for surface, required_count in gold_surfaces.items():
        if full_text.count(surface) < required_count:
            logger.debug("Surface '%s' count insufficient", surface)
            return False

    return True


def _find_all_matches(turns: list[dict], surface: str) -> list[tuple[int, int, int]]:
    matches = []
    pattern = re.escape(surface)
    for ti, turn in enumerate(turns):
        for m in re.finditer(pattern, turn["text"]):
            matches.append((ti, m.start(), m.end()))
    if matches:
        return matches
    norm_surface = unicodedata.normalize("NFKC", surface)
    for ti, turn in enumerate(turns):
        norm_text = unicodedata.normalize("NFKC", turn["text"])
        if len(norm_text) == len(turn["text"]):
            for m in re.finditer(re.escape(norm_surface), norm_text):
                matches.append((ti, m.start(), m.end()))
    return matches


def compute_conversation_spans(
    turns: list[dict],
    gold_entities: list[dict],
    llm_locations: list[dict] | None,
) -> list[dict] | None:
    location_hints: dict[str, list[int]] = {}
    if llm_locations:
        for loc in llm_locations:
            if isinstance(loc, dict) and "surface" in loc and "turn_index" in loc:
                ti = loc["turn_index"]
                if isinstance(ti, int) and 0 <= ti < len(turns):
                    location_hints.setdefault(loc["surface"], []).append(ti)

    occurrence_map: dict[str, list[tuple[int, int, int]]] = {}
    for surface in {e["surface"] for e in gold_entities}:
        occurrence_map[surface] = _find_all_matches(turns, surface)

    used: dict[str, int] = {}
    result = []

    for ent in gold_entities:
        surface = ent["surface"]
        matches = occurrence_map.get(surface, [])
        idx = used.get(surface, 0)

        if idx >= len(matches):
            logger.debug("No more matches for '%s' (%d/%d used)", surface, idx, len(matches))
            return None

        best = matches[idx]
        hints = location_hints.get(surface, [])
        if idx < len(hints):
            hinted_ti = hints[idx]
            for ti, s, e in matches[idx:]:
                if ti == hinted_ti:
                    best = (ti, s, e)
                    break

        ti, s, e = best
        result.append({
            "surface": surface,
            "type": ent["type"],
            "original_type": ent["original_type"],
            "start": s,
            "end": e,
            "turn_index": ti,
        })
        used[surface] = idx + 1

    return result


# Primary (quality): round-robin across gemini, sonnet, k2.5
# Fast fallback: haiku, spark
# Last resort: codex normal
PRIMARY_TOOLS = [
    ("sonnet", call_sonnet),
]

FALLBACK_TOOLS = [
    ("gemini", call_gemini),
    ("codex", call_codex),
]


def process_record(
    record: dict,
    neg_sampler: NegativeSampler,
    rng: random.Random,
    idx: int,
) -> tuple[str | None, str | None]:
    source_id = record["source_id"]
    entities = record["entities"]

    if not entities:
        return None, "no_entities"

    prompt = build_prompt(record["text"], entities)
    gold_surfaces = Counter(e["surface"] for e in entities)

    primary_idx = idx % len(PRIMARY_TOOLS)
    tool_chain = [
        PRIMARY_TOOLS[primary_idx],
        PRIMARY_TOOLS[(primary_idx + 1) % len(PRIMARY_TOOLS)],
        PRIMARY_TOOLS[(primary_idx + 2) % len(PRIMARY_TOOLS)],
        *FALLBACK_TOOLS,
    ]

    for tool_name, tool_fn in tool_chain:
        if not circuit_breaker.is_available(tool_name):
            continue

        raw = tool_fn(prompt)
        if raw is None:
            circuit_breaker.record_failure(tool_name)
            with stats_lock:
                stats["retries"] += 1
            continue

        parsed = clean_llm_output(raw)
        if parsed is None:
            logger.debug("[%s] %s: JSON parse failed", source_id, tool_name)
            circuit_breaker.record_failure(tool_name)
            with stats_lock:
                stats["retries"] += 1
            continue

        if not validate_llm_response(parsed, gold_surfaces):
            logger.debug("[%s] %s: validation failed", source_id, tool_name)
            with stats_lock:
                stats["retries"] += 1
            continue

        conv_entities = compute_conversation_spans(
            parsed["turns"], entities, parsed.get("entity_locations"),
        )
        if conv_entities is None:
            logger.debug("[%s] %s: span computation failed", source_id, tool_name)
            with stats_lock:
                stats["retries"] += 1
            continue

        positive_types = {e["type"] for e in conv_entities}
        negatives = neg_sampler.sample(positive_types, rng=rng)

        conv_record = ConversationRecord(
            source="b2nerd_conv",
            source_id=source_id + "_conv",
            language=record["language"],
            split=record["split"],
            confidence="silver",
            provenance=list(record["provenance"]),
            turns=parsed["turns"],
            query_types=list(positive_types) + negatives,
            entities=conv_entities,
        )

        try:
            conv_record.validate()
        except AssertionError as e:
            logger.debug("[%s] %s: validation error: %s", source_id, tool_name, e)
            with stats_lock:
                stats["retries"] += 1
            continue

        circuit_breaker.record_success(tool_name)
        with stats_lock:
            stats["by_model"][tool_name] += 1

        return conv_record.to_jsonl(), None

    return None, "all_tools_failed"


def load_records(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_checkpoint() -> tuple[set[str], dict]:
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            data = json.load(f)
        return set(data.get("processed_ids", [])), data.get("stats", {})
    return set(), {}


def save_checkpoint(processed_ids: set[str]):
    with checkpoint_lock:
        data = {"processed_ids": sorted(processed_ids), "stats": dict(stats)}
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Conversationalize B2NERD records")
    parser.add_argument("--pilot", action="store_true", help="Pilot: 500 EN + 200 ZH")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--en-count", type=int, default=None, help="Override EN count")
    parser.add_argument("--zh-count", type=int, default=None, help="Override ZH count")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    en_count = args.en_count if args.en_count is not None else (500 if args.pilot else 5000)
    zh_count = args.zh_count if args.zh_count is not None else (200 if args.pilot else 2000)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    processed_ids: set[str] = set()
    if args.resume:
        processed_ids, _ = load_checkpoint()
        logger.info("Resumed with %d already-processed IDs", len(processed_ids))

    logger.info("Loading EN records from %s", INPUT_EN)
    all_en = load_records(INPUT_EN)
    logger.info("Loading ZH records from %s", INPUT_ZH)
    all_zh = load_records(INPUT_ZH)

    all_en = [r for r in all_en if r.get("entities")]
    all_zh = [r for r in all_zh if r.get("entities")]
    logger.info("EN with entities: %d, ZH with entities: %d", len(all_en), len(all_zh))

    rng.shuffle(all_en)
    rng.shuffle(all_zh)
    sampled_en = all_en[:en_count]
    sampled_zh = all_zh[:zh_count]

    all_records: list[tuple[str, dict]] = []
    for r in sampled_en:
        all_records.append(("en", r))
    for r in sampled_zh:
        all_records.append(("zh", r))

    if args.resume:
        all_records = [(lang, r) for lang, r in all_records if r["source_id"] not in processed_ids]
        logger.info("After resume filter: %d records to process", len(all_records))

    stats["total"] = len(all_records) + len(processed_ids)

    record_seeds = {r["source_id"]: rng.randint(0, 2**32 - 1) for _, r in all_records}

    neg_sampler = NegativeSampler()

    out_en = OUTPUT_DIR / "b2nerd_conv_en.jsonl"
    out_zh = OUTPUT_DIR / "b2nerd_conv_zh.jsonl"
    failures_file = OUTPUT_DIR / "failures.jsonl"

    file_mode = "a" if args.resume else "w"
    fh_en = open(out_en, file_mode)
    fh_zh = open(out_zh, file_mode)
    fh_fail = open(failures_file, file_mode)

    completed = 0
    total = len(all_records)
    start_time = time.time()

    def worker(item: tuple[str, dict]) -> bool:
        nonlocal completed
        if stop_event.is_set():
            return False
        lang, record = item
        idx = hash(record["source_id"]) % len(PRIMARY_TOOLS)
        per_thread_rng = random.Random(record_seeds[record["source_id"]])

        t0 = time.time()
        result, error = process_record(record, neg_sampler, per_thread_rng, idx)
        elapsed = time.time() - t0

        with stats_lock:
            stats["times"].append(elapsed)

        if result is not None:
            with write_lock:
                fh = fh_en if lang == "en" else fh_zh
                fh.write(result + "\n")
                fh.flush()
                processed_ids.add(record["source_id"])
            with stats_lock:
                stats["success"] += 1
                stats["by_language"][lang] = stats["by_language"].get(lang, 0) + 1
        else:
            with write_lock:
                fh_fail.write(json.dumps({
                    "source_id": record["source_id"],
                    "language": lang,
                    "error": error,
                }, ensure_ascii=False) + "\n")
                fh_fail.flush()
            with stats_lock:
                stats["fail"] += 1

        completed += 1
        if completed % 10 == 0:
            with stats_lock:
                s, f = stats["success"], stats["fail"]
            elapsed_total = time.time() - start_time
            rate = completed / elapsed_total if elapsed_total > 0 else 0
            logger.info(
                "Progress: %d/%d (%.0f%%) | success=%d fail=%d | %.1f rec/min",
                completed, total, 100 * completed / total if total else 0, s, f, rate * 60,
            )

        if completed % 50 == 0:
            save_checkpoint(processed_ids)

        return result is not None

    logger.info("Starting conversationalization: %d EN + %d ZH = %d total",
                len(sampled_en), len(sampled_zh), total)

    with ThreadPoolExecutor(max_workers=80) as executor:
        futures = {executor.submit(worker, item): item for item in all_records}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error("Worker exception: %s", e)

    fh_en.close()
    fh_zh.close()
    fh_fail.close()

    save_checkpoint(processed_ids)

    avg_time = sum(stats["times"]) / len(stats["times"]) if stats["times"] else 0
    final_stats = {
        "total": stats["total"],
        "success": stats["success"],
        "fail": stats["fail"],
        "retries": stats["retries"],
        "by_model": stats["by_model"],
        "by_language": stats["by_language"],
        "avg_time_per_record": round(avg_time, 2),
        "total_time": round(time.time() - start_time, 2),
        "success_rate": round(stats["success"] / stats["total"] * 100, 1) if stats["total"] else 0,
    }

    stats_file = OUTPUT_DIR / "stats.json"
    with open(stats_file, "w") as f:
        json.dump(final_stats, f, indent=2)

    logger.info("Done! Success: %d/%d (%.1f%%)",
                final_stats["success"], final_stats["total"], final_stats["success_rate"])
    logger.info("Stats written to %s", stats_file)


if __name__ == "__main__":
    main()
