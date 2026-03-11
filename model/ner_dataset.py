"""JSONL -> tokenized + BIO labels for NER training."""

import json
import logging
import os
import random
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
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
_BUCKET_MAP = {"gold": 0, "silver_en": 1, "silver_ml": 2}
_BUCKET_NAMES = {v: k for k, v in _BUCKET_MAP.items()}


def _build_or_load_index(data_path: str) -> dict:
    path = Path(data_path)
    stat = path.stat()
    fingerprint = f"{stat.st_size}:{stat.st_mtime_ns}"

    index_dir = path.parent / f"{path.stem}_index"
    meta_path = index_dir / "meta.json"

    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("fingerprint") == fingerprint:
            logger.info("Loaded sidecar index from %s (%d rows)", index_dir, meta["num_rows"])
            return {
                "offsets": np.load(index_dir / "offsets.npy", mmap_mode="r"),
                "bucket_ids": np.load(index_dir / "bucket_ids.npy", mmap_mode="r"),
                "bucket_indices": {
                    name: np.load(index_dir / f"{name}_idx.npy", mmap_mode="r")
                    for name in _BUCKET_MAP
                    if (index_dir / f"{name}_idx.npy").exists()
                },
                "num_rows": meta["num_rows"],
                "bucket_counts": meta["bucket_counts"],
                "source_counts": meta["source_counts"],
            }

    logger.info("Building sidecar index for %s ...", data_path)
    offsets = []
    bucket_ids = []
    sources = []

    with open(path, "rb") as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Bad JSON at offset %d in %s", offset, data_path)
                continue

            if rec.get("format") == "conversation":
                if not rec.get("turns"):
                    continue
            else:
                if not rec.get("text", ""):
                    continue

            bucket = _assign_bucket(rec)
            bucket_id = _BUCKET_MAP.get(bucket, 0)
            offsets.append(offset)
            bucket_ids.append(bucket_id)
            sources.append(rec.get("source", ""))

    offsets_arr = np.array(offsets, dtype=np.uint64)
    bucket_ids_arr = np.array(bucket_ids, dtype=np.uint8)

    bucket_indices = {}
    bucket_counts = {}
    for name, bid in _BUCKET_MAP.items():
        idx = np.where(bucket_ids_arr == bid)[0].astype(np.int32)
        bucket_indices[name] = idx
        bucket_counts[name] = int(len(idx))

    source_counts = dict(Counter(sources))
    num_rows = len(offsets)

    tmp_dir = tempfile.mkdtemp(dir=path.parent)
    try:
        np.save(os.path.join(tmp_dir, "offsets.npy"), offsets_arr)
        np.save(os.path.join(tmp_dir, "bucket_ids.npy"), bucket_ids_arr)
        for name, idx_arr in bucket_indices.items():
            np.save(os.path.join(tmp_dir, f"{name}_idx.npy"), idx_arr)
        with open(os.path.join(tmp_dir, "meta.json"), "w") as f:
            json.dump({
                "fingerprint": fingerprint,
                "num_rows": num_rows,
                "bucket_counts": bucket_counts,
                "source_counts": source_counts,
            }, f)
        if index_dir.exists():
            shutil.rmtree(index_dir)
        os.rename(tmp_dir, index_dir)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    logger.info("Built sidecar index: %d rows, buckets=%s", num_rows, bucket_counts)

    return {
        "offsets": offsets_arr,
        "bucket_ids": bucket_ids_arr,
        "bucket_indices": bucket_indices,
        "num_rows": num_rows,
        "bucket_counts": bucket_counts,
        "source_counts": source_counts,
    }


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
        self._init_lazy(data_path)

    def _init_lazy(self, data_path):
        self._lazy = True
        self.records = None
        self._data_path = data_path
        self._fh = None

        index = _build_or_load_index(data_path)
        self._offsets = index["offsets"]
        self._num_rows = index["num_rows"]
        self.bucket_indices = index["bucket_indices"]

        split = "train" if self.is_train else "val"
        logger.info("Lazy %s dataset: %d rows, buckets=%s",
                     split, self._num_rows, index["bucket_counts"])

    def __len__(self):
        if self._lazy:
            return self._num_rows
        return len(self.records)

    def __getitem__(self, idx):
        self._ensure_open()
        self._fh.seek(int(self._offsets[idx]))
        line = self._fh.readline()
        raw = json.loads(line)
        processed = self._process_record(raw)
        if processed is None:
            raise RuntimeError(
                f"Record at index {idx} returned None — sidecar index may be stale. "
                f"Delete the _index/ directory and retry."
            )
        result = {
            "input_ids": processed["input_ids"],
            "attention_mask": processed["attention_mask"],
            "labels": processed["labels"],
        }
        if self.is_train:
            result["meta"] = {
                "entity_count": processed["_entity_count"],
                "truncated": processed["_truncated"],
                "type_list": processed["_type_list"],
                "bucket": processed["bucket"],
                "source": processed["source"],
            }
        else:
            result["text"] = processed["text"]
            result["entities_gold"] = processed["entities_gold"]
            result["offset_mapping"] = processed["offset_mapping"]
        return result

    def _ensure_open(self):
        if self._fh is None or self._fh.closed:
            self._fh = open(self._data_path, "rb")

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_fh"] = None
        return state

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
    if confidence == "silver":
        if rec.get("format") != "conversation":
            return "gold"
        lang = rec.get("language", "en")
        return "silver_en" if lang == "en" else "silver_ml"
    source = rec.get("source", "")
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
    if "meta" in batch[0]:
        result["meta"] = [b["meta"] for b in batch]
    return result
