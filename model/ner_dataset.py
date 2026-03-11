"""JSONL -> tokenized + BIO labels for NER training."""

import json
import logging
import random
from collections import Counter, defaultdict

import torch
from torch.utils.data import Dataset

from model.config import (
    IGNORE_INDEX,
    LABEL2ID,
    MAX_SEQ_LEN,
    SOURCE_CAP,
    SYNC_TYPES,
    normalize_type,
)

logger = logging.getLogger(__name__)

_SYNC_SET = set(SYNC_TYPES)


class NerDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer,
        type_mapping: dict[str, str] | None = None,
        max_seq_len: int = MAX_SEQ_LEN,
        is_train: bool = True,
    ):
        self.tokenizer = tokenizer
        self.type_mapping = type_mapping
        self.max_seq_len = max_seq_len
        self.is_train = is_train

        raw_records = _load_jsonl(data_path)
        self.records = []
        entity_count = 0
        truncated_entities = 0
        bucket_counts: Counter = Counter()
        type_counts: Counter = Counter()

        for rec in raw_records:
            processed = self._process_record(rec)
            if processed is None:
                continue
            self.records.append(processed)
            entity_count += processed["_entity_count"]
            truncated_entities += processed["_truncated"]
            bucket_counts[processed["bucket"]] += 1
            for t in processed["_type_list"]:
                type_counts[t] += 1

        total = len(self.records)
        trunc_pct = (truncated_entities / max(entity_count, 1)) * 100
        logger.info(
            f"Loaded {total} records, {entity_count} entities "
            f"({trunc_pct:.1f}% truncated) from {data_path}"
        )
        logger.info(f"Buckets: {dict(bucket_counts)}")
        logger.info(f"Type dist: {dict(type_counts.most_common(25))}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        return {
            k: v
            for k, v in rec.items()
            if not k.startswith("_")
        }

    def _process_record(self, rec: dict) -> dict | None:
        if rec.get("format") == "conversation":
            text, entities = _flatten_conversation(rec)
        else:
            text = rec.get("text", "")
            entities = rec.get("entities", [])

        if not text:
            return None

        filtered = []
        for ent in entities:
            raw_type = ent.get("type", "")
            norm = normalize_type(raw_type)
            if norm is None and self.type_mapping:
                mapped = self.type_mapping.get(raw_type) or self.type_mapping.get(
                    ent.get("original_type", "")
                )
                if mapped:
                    norm = normalize_type(mapped) if mapped not in _SYNC_SET else mapped
            if norm is None:
                continue
            filtered.append({
                "surface": ent["surface"],
                "type": norm,
                "start": ent["start"],
                "end": ent["end"],
            })

        encoding = self.tokenizer(
            text,
            max_length=self.max_seq_len,
            truncation=True,
            return_offsets_mapping=True,
            padding=False,
            return_tensors=None,
        )

        offset_mapping = encoding["offset_mapping"]
        seq_len = len(encoding["input_ids"])
        labels = [IGNORE_INDEX] * seq_len

        for i, (ts, te) in enumerate(offset_mapping):
            if ts == 0 and te == 0:
                continue
            labels[i] = LABEL2ID["O"]

        max_char = offset_mapping[-1][1] if seq_len > 0 else 0
        for i in range(seq_len - 1, -1, -1):
            if offset_mapping[i][1] > 0:
                max_char = offset_mapping[i][1]
                break

        truncated = sum(1 for e in filtered if e["start"] >= max_char)

        filtered.sort(key=lambda e: (e["start"], -(e["end"] - e["start"])))
        claimed = [False] * seq_len
        type_list = []

        for ent in filtered:
            es, ee = ent["start"], ent["end"]
            if es >= max_char:
                continue

            b_label = LABEL2ID.get(f"B-{ent['type']}")
            i_label = LABEL2ID.get(f"I-{ent['type']}")
            if b_label is None or i_label is None:
                continue

            first = True
            conflict = False
            for i, (ts, te) in enumerate(offset_mapping):
                if ts == te == 0:
                    continue
                if te <= es:
                    continue
                if ts >= ee:
                    break
                if claimed[i]:
                    conflict = True
                    break

            if conflict:
                logger.debug(f"Overlap: dropping '{ent['surface']}' ({ent['type']})")
                continue

            first = True
            for i, (ts, te) in enumerate(offset_mapping):
                if ts == te == 0:
                    continue
                if te <= es:
                    continue
                if ts >= ee:
                    break
                labels[i] = b_label if first else i_label
                claimed[i] = True
                first = False

            type_list.append(ent["type"])

        bucket = _assign_bucket(rec)

        result = {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": labels,
            "bucket": bucket,
            "source": rec.get("source", ""),
            "_entity_count": len(filtered),
            "_truncated": truncated,
            "_type_list": type_list,
        }
        if not self.is_train:
            result["text"] = text
            result["entities_gold"] = filtered
            result["offset_mapping"] = offset_mapping
        return result


def _load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning(f"Bad JSON at line {line_num} in {path}")
    return records


def _flatten_conversation(rec: dict) -> tuple[str, list[dict]]:
    turns = rec.get("turns", [])
    parts = []
    offsets = []
    pos = 0
    for turn in turns:
        prefix = f"{turn['speaker']}: "
        parts.append(prefix + turn["text"])
        offsets.append(pos + len(prefix))
        pos += len(prefix) + len(turn["text"]) + 1  # +1 for \n

    text = "\n".join(f"{t['speaker']}: {t['text']}" for t in turns)

    entities = []
    for ent in rec.get("entities", []):
        ti = ent.get("turn_index", 0)
        if ti >= len(turns):
            continue
        turn_start = sum(
            len(f"{turns[j]['speaker']}: {turns[j]['text']}") + 1 for j in range(ti)
        )
        prefix_len = len(f"{turns[ti]['speaker']}: ")
        abs_start = turn_start + prefix_len + ent["start"]
        abs_end = turn_start + prefix_len + ent["end"]
        entities.append({
            "surface": ent["surface"],
            "type": ent["type"],
            "start": abs_start,
            "end": abs_end,
        })

    return text, entities


def _assign_bucket(rec: dict) -> str:
    confidence = rec.get("confidence", "")
    if confidence in ("human-gold", "synthetic-gold", "gold"):
        return "gold"
    source = rec.get("source", "")
    lang = rec.get("language", "en")
    if source.startswith(("task9_", "task8_")):
        return "silver_en" if lang == "en" else "silver_ml"
    if source.startswith(("multilingual_", "ml_synthetic_", "silver_ml_")):
        return "silver_ml"
    return "gold"


def apply_source_caps(
    records: list[dict], cap: float = SOURCE_CAP, seed: int = 42
) -> list[dict]:
    rng = random.Random(seed)
    total = len(records)
    max_per_source = int(cap * total)
    by_source: defaultdict[str, list] = defaultdict(list)
    for i, rec in enumerate(records):
        by_source[rec.get("source", "")].append(i)

    keep = set()
    for source, indices in by_source.items():
        if len(indices) > max_per_source:
            logger.info(
                f"Capping source '{source}': {len(indices)} -> {max_per_source}"
            )
            indices = rng.sample(indices, max_per_source)
        keep.update(indices)

    return [records[i] for i in sorted(keep)]


def ner_collate_fn(batch: list[dict]) -> dict:
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids = []
    attention_mask = []
    labels = []
    for b in batch:
        pad_len = max_len - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [0] * pad_len)
        attention_mask.append(b["attention_mask"] + [0] * pad_len)
        labels.append(b["labels"] + [IGNORE_INDEX] * pad_len)
    result = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
    if "offset_mapping" in batch[0]:
        result["offset_mapping"] = [b["offset_mapping"] for b in batch]
    return result
