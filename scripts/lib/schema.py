"""NerRecord dataclass + validation + serialization matching configs/schema.json."""

import json
from dataclasses import dataclass

SCHEMA_VERSION = "1.0"

CONFIDENCE_ORDER = {"human-gold": 0, "synthetic-gold": 1, "gold": 2, "silver": 3}
VALID_SPLITS = {"train", "val", "eval", "zero_shot_eval"}
VALID_CONFIDENCES = set(CONFIDENCE_ORDER.keys())


@dataclass
class Entity:
    surface: str
    type: str
    original_type: str
    start: int
    end: int

    def to_dict(self) -> dict:
        return {
            "surface": self.surface,
            "type": self.type,
            "original_type": self.original_type,
            "start": self.start,
            "end": self.end,
        }


@dataclass
class NerRecord:
    source: str
    source_id: str
    language: str
    split: str
    confidence: str
    provenance: list[str]
    text: str
    query_types: list[str]
    entities: list[dict]
    schema_version: str = SCHEMA_VERSION

    def validate(self):
        assert self.schema_version == SCHEMA_VERSION, f"Bad schema_version: {self.schema_version}"
        assert self.split in VALID_SPLITS, f"Bad split: {self.split}"
        assert self.confidence in VALID_CONFIDENCES, f"Bad confidence: {self.confidence}"
        assert len(self.provenance) >= 1, "Empty provenance"
        assert len(self.text) > 0, "Empty text"
        assert len(self.query_types) >= 1, "Empty query_types"
        assert len(self.query_types) == len(set(self.query_types)), "Duplicate query_types"

        positive_types = set()
        for ent in self.entities:
            assert ent["end"] > ent["start"], f"end <= start: {ent}"
            span = self.text[ent["start"]:ent["end"]]
            assert span == ent["surface"], (
                f"Span mismatch: text[{ent['start']}:{ent['end']}]='{span}' != '{ent['surface']}'"
            )
            positive_types.add(ent["type"])

        assert positive_types <= set(self.query_types), (
            f"Entity types not in query_types: {positive_types - set(self.query_types)}"
        )

        neg_count = len(self.query_types) - len(positive_types)
        assert 2 <= neg_count <= 5, f"Negative count {neg_count} not in [2,5]"

    def to_jsonl(self) -> str:
        d = {
            "schema_version": self.schema_version,
            "source": self.source,
            "source_id": self.source_id,
            "language": self.language,
            "split": self.split,
            "confidence": self.confidence,
            "provenance": self.provenance,
            "text": self.text,
            "query_types": self.query_types,
            "entities": self.entities,
        }
        return json.dumps(d, ensure_ascii=False)

    @classmethod
    def from_jsonl(cls, line: str) -> "NerRecord":
        d = json.loads(line)
        return cls(
            schema_version=d["schema_version"],
            source=d["source"],
            source_id=d["source_id"],
            language=d["language"],
            split=d["split"],
            confidence=d["confidence"],
            provenance=d["provenance"],
            text=d["text"],
            query_types=d["query_types"],
            entities=d["entities"],
        )


def min_confidence(*confidences: str) -> str:
    return min(confidences, key=lambda c: CONFIDENCE_ORDER[c])
