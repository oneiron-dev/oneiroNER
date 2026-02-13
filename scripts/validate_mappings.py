#!/usr/bin/env python3
"""Validate type_mapping_train.json and type_mapping_eval.json."""

import json
import sys
from collections import Counter
from pathlib import Path

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def load_json_check_dupes(path):
    """Load JSON and detect duplicate keys (json.load silently takes last)."""
    dupes = []
    with open(path) as f:
        raw = f.read()
    seen = Counter()

    def pair_hook(pairs):
        d = {}
        for k, v in pairs:
            seen[k] += 1
            if seen[k] > 1:
                dupes.append(k)
            d[k] = v
        return d

    result = json.loads(raw, object_pairs_hook=pair_hook)
    return result, dupes

TRAIN_CANONICAL = {"PERSON", "PLACE", "ORG", "DATE", "EMOTION"}
EVAL_CANONICAL = TRAIN_CANONICAL | {"ACTIVITY", "OTHER"}

REQUIRED_IN_TRAIN = {
    "open_ner_core_types": ["PER", "LOC", "ORG"],
    "klue_ner": ["PS", "LC", "OG", "DT", "TI"],
    "stockmark_ner_ja": ["人名", "地名", "法人名", "その他の組織名", "政治的組織名", "施設名"],
    "multiconer_v2_person": [
        "Artist", "Athlete", "Politician", "Scientist",
        "Cleric", "SportsManager", "OtherPER",
    ],
    "multiconer_v2_place": ["HumanSettlement", "Station", "Facility", "OtherLOC"],
    "multiconer_v2_org": ["ORG", "PublicCorp", "PrivateCorp", "MusicalGRP", "SportsGRP"],
    "open_ner_standard": [
        "PER", "LOC", "ORG", "GPE", "GPE-LOC", "GPE-ORG",
        "DATE", "TIME", "FACILITY", "CORPORATION",
    ],
    "chinese_ner_sft_train": [
        "LOC.NAM", "LOC.NOM", "Location", "ORG.NAM", "ORG.NOM",
        "GPE.NAM", "PER.NAM", "PER.NOM", "emotion",
    ],
    "b2nerd_train": [
        "Person", "Location", "City", "Country", "Organization",
        "Company", "Band", "Date", "Facility",
    ],
    "finerweb_train": [
        "person", "location", "city", "country",
        "organization", "company", "date", "building",
    ],
}

REQUIRED_IN_EVAL_ONLY = {
    "open_ner_remainder": [
        "EVENT", "MISC", "CARDINAL", "MONEY", "PERCENT", "ORDINAL",
        "NORP", "QUANTITY", "LAW", "LANG", "DESIGNATION", "TITLE_AFFIX",
        "RELIGION", "DISEASE", "PRODUCT", "CREATIVE_WORK",
        "PER-DERIV", "LOC-DERIV", "ORG-DERIV", "MISC-DERIV",
        "ADAGE", "ART", "ART-DERIV", "ART-PART", "CONTACT", "DATETIME",
        "DERIV", "EVENT-DERIV", "EVENT-PART", "FESTIVAL", "GAME", "GROUP",
        "LANG-DERIV", "LANG-PART", "LITERATURE", "LOC-PART", "MEASURE",
        "MISC-PART", "MOVEMENT", "NON_HUMAN", "NUM", "ORG-PART",
        "PER-PART", "PERCENTAGE", "PERIOD", "PET_NAME", "PHONE",
        "POSITION", "PROJECT", "RELIGION-DERIV",
    ],
    "multiconer_v2_other": [
        "Disease", "Symptom", "AnatomicalStructure", "MedicalProcedure",
        "Medication/Vaccine", "ArtWork", "MusicalWork", "VisualWork",
        "WrittenWork", "Clothing", "Drink", "Food", "Software",
        "Vehicle", "OtherPROD",
    ],
    "klue_eval": ["QT"],
    "stockmark_eval": ["イベント名", "製品名"],
    "chinese_ner_sft_eval": ["BANK", "NAME", "疾病和诊断", "解剖部位"],
    "b2nerd_eval": ["Animal", "Disease", "Chemical", "Activity", "Award", "Event"],
    "finerweb_eval": ["animal", "disease", "food", "book", "event"],
}


def main():
    errors = []

    train_path = CONFIGS / "type_mapping_train.json"
    eval_path = CONFIGS / "type_mapping_eval.json"

    train, train_dupes = load_json_check_dupes(train_path)
    eval_map, eval_dupes = load_json_check_dupes(eval_path)

    for k in train_dupes:
        errors.append(f"Train: duplicate key '{k}'")
    for k in eval_dupes:
        errors.append(f"Eval: duplicate key '{k}'")

    print(f"Train entries: {len(train)}")
    print(f"Eval entries:  {len(eval_map)}")

    # Check train values in allowed set
    for k, v in train.items():
        if v not in TRAIN_CANONICAL:
            errors.append(f"Train: '{k}' → '{v}' not in {TRAIN_CANONICAL}")

    # Check eval values in allowed set
    for k, v in eval_map.items():
        if v not in EVAL_CANONICAL:
            errors.append(f"Eval: '{k}' → '{v}' not in {EVAL_CANONICAL}")

    # Train must NOT contain OTHER or ACTIVITY
    for k, v in train.items():
        if v in ("OTHER", "ACTIVITY"):
            errors.append(f"Train contains forbidden canonical type: '{k}' → '{v}'")

    # Train is proper subset of eval (train ⊂ eval)
    for k, v in train.items():
        if k not in eval_map:
            errors.append(f"Train key '{k}' missing from eval")
        elif eval_map[k] != v:
            errors.append(
                f"Train/eval mismatch for '{k}': train='{v}', eval='{eval_map[k]}'"
            )
    if len(train) >= len(eval_map):
        errors.append(
            f"Train ({len(train)}) must be smaller than eval ({len(eval_map)})"
        )

    # Whitespace check
    for label, mapping in [("train", train), ("eval", eval_map)]:
        for k, v in mapping.items():
            if k != k.strip():
                errors.append(f"{label}: key '{k}' has whitespace")
            if v != v.strip():
                errors.append(f"{label}: value '{v}' for key '{k}' has whitespace")

    # Required coverage: train
    for group, types in REQUIRED_IN_TRAIN.items():
        for t in types:
            if t not in train:
                errors.append(f"REQUIRED_IN_TRAIN[{group}]: '{t}' missing from train")

    # Required coverage: eval-only (must be in eval but NOT in train)
    for group, types in REQUIRED_IN_EVAL_ONLY.items():
        for t in types:
            if t not in eval_map:
                errors.append(
                    f"REQUIRED_IN_EVAL_ONLY[{group}]: '{t}' missing from eval"
                )
            if t in train:
                errors.append(
                    f"REQUIRED_IN_EVAL_ONLY[{group}]: '{t}' should NOT be in train"
                )

    # All 33 multiconer types accounted for in eval
    multiconer_all = [
        "AerospaceManufacturer", "AnatomicalStructure", "ArtWork", "Artist",
        "Athlete", "CarManufacturer", "Cleric", "Clothing", "Disease", "Drink",
        "Facility", "Food", "HumanSettlement", "MedicalProcedure",
        "Medication/Vaccine", "MusicalGRP", "MusicalWork", "ORG", "OtherLOC",
        "OtherPER", "OtherPROD", "Politician", "PrivateCorp", "PublicCorp",
        "Scientist", "Software", "SportsGRP", "SportsManager", "Station",
        "Symptom", "Vehicle", "VisualWork", "WrittenWork",
    ]
    for t in multiconer_all:
        if t not in eval_map:
            errors.append(f"MultiCoNER type '{t}' missing from eval")

    # All 8 stockmark types accounted for in eval
    stockmark_all = ["人名", "地名", "法人名", "その他の組織名", "イベント名", "政治的組織名", "施設名", "製品名"]
    for t in stockmark_all:
        if t not in eval_map:
            errors.append(f"Stockmark type '{t}' missing from eval")

    # All 6 KLUE types accounted for in eval
    klue_all = ["PS", "LC", "OG", "DT", "TI", "QT"]
    for t in klue_all:
        if t not in eval_map:
            errors.append(f"KLUE type '{t}' missing from eval")

    # Cross-reference with dataset_inventory.json (warn on drift)
    inv_path = CONFIGS / "dataset_inventory.json"
    if inv_path.exists():
        with open(inv_path) as f:
            inventory = json.load(f)
        FULLY_COVERED = {
            "klue_ner": 6,
            "multiconer_v2": 33,
            "stockmark_ner_ja": 8,
            "open_ner_standardized": 60,
            "chinese_ner_sft": 72,
        }
        for ds_name, expected_count in FULLY_COVERED.items():
            ds = inventory.get("datasets", {}).get(ds_name, {})
            actual_count = ds.get("entity_types", {}).get("count", 0)
            if actual_count != expected_count:
                errors.append(
                    f"Inventory drift: {ds_name} has {actual_count} types "
                    f"(expected {expected_count}). Update coverage lists."
                )

    # Distribution
    print("\n--- Train distribution ---")
    train_dist = {}
    for v in train.values():
        train_dist[v] = train_dist.get(v, 0) + 1
    for ct in sorted(train_dist):
        print(f"  {ct}: {train_dist[ct]}")

    print("\n--- Eval distribution ---")
    eval_dist = {}
    for v in eval_map.values():
        eval_dist[v] = eval_dist.get(v, 0) + 1
    for ct in sorted(eval_dist):
        print(f"  {ct}: {eval_dist[ct]}")

    if errors:
        print(f"\n{'='*60}")
        print(f"FAILED — {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print(f"\n{'='*60}")
        print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
