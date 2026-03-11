"""NER label scheme, type mappings, and training hyperparameters."""

import json
from pathlib import Path

MODEL_NAME = "Alibaba-NLP/gte-multilingual-mlm-base"
MAX_SEQ_LEN = 512
IGNORE_INDEX = -100

SYNC_TYPES = [
    "PERSON", "PLACE", "ORG", "DATE", "EVENT", "RELATIONSHIP_REF",
    "DATE/Day", "DATE/Week", "DATE/Month", "DATE/Season",
    "DATE/Year", "DATE/Decade", "DATE/Relative", "DATE/Range",
    "EVENT/Life", "EVENT/General",
    "RELATIONSHIP_REF/Family", "RELATIONSHIP_REF/Romantic",
    "RELATIONSHIP_REF/Friend", "RELATIONSHIP_REF/Professional",
    "RELATIONSHIP_REF/Acquaintance",
]

LABEL_LIST = [
    "O",
    "B-PERSON", "I-PERSON",
    "B-PLACE", "I-PLACE",
    "B-ORG", "I-ORG",
    "B-DATE", "I-DATE",
    "B-DATE/Day", "I-DATE/Day",
    "B-DATE/Week", "I-DATE/Week",
    "B-DATE/Month", "I-DATE/Month",
    "B-DATE/Season", "I-DATE/Season",
    "B-DATE/Year", "I-DATE/Year",
    "B-DATE/Decade", "I-DATE/Decade",
    "B-DATE/Relative", "I-DATE/Relative",
    "B-DATE/Range", "I-DATE/Range",
    "B-EVENT", "I-EVENT",
    "B-EVENT/Life", "I-EVENT/Life",
    "B-EVENT/General", "I-EVENT/General",
    "B-RELATIONSHIP_REF", "I-RELATIONSHIP_REF",
    "B-RELATIONSHIP_REF/Family", "I-RELATIONSHIP_REF/Family",
    "B-RELATIONSHIP_REF/Romantic", "I-RELATIONSHIP_REF/Romantic",
    "B-RELATIONSHIP_REF/Friend", "I-RELATIONSHIP_REF/Friend",
    "B-RELATIONSHIP_REF/Professional", "I-RELATIONSHIP_REF/Professional",
    "B-RELATIONSHIP_REF/Acquaintance", "I-RELATIONSHIP_REF/Acquaintance",
]

NUM_LABELS = len(LABEL_LIST)
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}

BASE_TYPES = {"PERSON", "PLACE", "ORG", "DATE", "EVENT", "RELATIONSHIP_REF"}

_SYNC_SET = set(SYNC_TYPES)


def collapse_to_base(type_str: str) -> str:
    return type_str.split("/")[0]


def normalize_type(raw_type: str) -> str | None:
    normalized = raw_type.replace(" -> ", "/").replace("->", "/").replace(": ", "/").replace(":", "/")
    normalized = normalized.strip()
    return normalized if normalized in _SYNC_SET else None


def load_type_mapping(path: str = "configs/type_mapping_train.json") -> dict[str, str]:
    raw = json.loads(Path(path).read_text())
    result = {}
    for k, v in raw.items():
        norm = normalize_type(v)
        if norm is not None:
            result[k] = norm
    return result


# Mixture ratios
GOLD_RATIO = 0.75
SILVER_EN_RATIO = 0.20
SILVER_ML_RATIO = 0.05

# Sub-bucket weights
TIER_WEIGHTS = {"T1": 0.9, "T2": 0.7, "T3": 0.5}

SOURCE_TIERS = {
    "mentalchat": "T1",
    "therapy_conversations": "T1",
    "personachat": "T1",
    "prosocial_dialog": "T1",
    "reddit_confessions": "T2",
    "pippa": "T3",
    "opencharacter": "T3",
    "roleplay_hieu": "T3",
    "synthetic_persona_chat": "T3",
}

# Hyperparameters
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.05
BATCH_SIZE = 32
NUM_EPOCHS = 3
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

SOURCE_CAP = 0.25
