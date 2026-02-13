"""Convert MultiCoNER v2 (CoNLL BIO, 13 language dirs) to NER JSONL."""

import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from lib.bio_to_spans import bio_tags_to_spans, detokenize, tokens_to_char_spans, is_cjk_language
from lib.negative_sampler import NegativeSampler
from lib.schema import Entity, NerRecord
from lib.span_validator import validate_span

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data" / "raw" / "multiconer_v2"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "processed"

SOURCE = "multiconer_v2"
CONFIDENCE = "gold"

DIR_TO_LANG = {
    "BN-Bangla": "bn", "DE-German": "de", "EN-English": "en",
    "ES-Spanish": "es", "FA-Farsi": "fa", "FR-French": "fr",
    "HI-Hindi": "hi", "IT-Italian": "it", "MULTI-Multilingual": "multi",
    "PT-Portuguese": "pt", "SV-Swedish": "sv", "UK-Ukrainian": "uk",
    "ZH-Chinese": "zh",
}

neg_sampler = NegativeSampler()
rng = random.Random(42)


def parse_conll(path: Path) -> list[dict]:
    sentences = []
    tokens = []
    tags = []
    sent_id = None

    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("# id"):
                sent_id = line.split()[2] if len(line.split()) >= 3 else None
            elif line == "":
                if tokens:
                    sentences.append({"tokens": tokens, "tags": tags, "id": sent_id})
                    tokens = []
                    tags = []
                    sent_id = None
            else:
                parts = line.split()
                if len(parts) >= 4:
                    tokens.append(parts[0])
                    tags.append(parts[-1])

    if tokens:
        sentences.append({"tokens": tokens, "tags": tags, "id": sent_id})

    return sentences


def convert_lang(lang_dir: Path, lang: str, stats: dict) -> list[NerRecord]:
    records = []

    for split in ["train", "dev"]:
        conll_file = lang_dir / f"{lang}_{split}.conll"
        if not conll_file.exists():
            pattern = f"*_{split}.conll"
            matches = list(lang_dir.glob(pattern))
            if not matches:
                logger.warning("No %s file found in %s", split, lang_dir)
                continue
            conll_file = matches[0]

        sentences = parse_conll(conll_file)
        logger.info("  %s %s: %d sentences", lang, split, len(sentences))

        for idx, sent in enumerate(sentences):
            stats["total"] += 1
            token_spans = bio_tags_to_spans(sent["tokens"], sent["tags"])

            det_lang = lang if lang != "multi" else "en"
            text, token_offsets = detokenize(sent["tokens"], det_lang)

            entities = []
            for span in tokens_to_char_spans(token_spans, token_offsets, text):
                if validate_span(text, span["surface"], span["start"], span["end"]):
                    entities.append(Entity(
                        surface=span["surface"],
                        type=span["type"],
                        original_type=span["type"],
                        start=span["start"],
                        end=span["end"],
                    ).to_dict())
                    stats["entities"] += 1
                else:
                    stats["entities_skipped"] += 1
                    logger.debug("Span mismatch in %s: '%s' vs text[%d:%d]='%s'",
                                 lang, span["surface"], span["start"], span["end"],
                                 text[span["start"]:span["end"]])

            if not entities:
                stats["skipped"] += 1
                continue

            pos_types = {e["type"] for e in entities}
            neg_types = neg_sampler.sample(pos_types, rng=rng)
            query_types = sorted(pos_types) + neg_types

            sid = sent.get("id") or f"{idx}"
            rec = NerRecord(
                source=SOURCE,
                source_id=f"multiconer_v2_{lang}_{split}_{sid}",
                language=lang,
                split="train",
                confidence=CONFIDENCE,
                provenance=[SOURCE],
                text=text,
                query_types=query_types,
                entities=entities,
            )
            rec.validate()
            records.append(rec)

    return records


def convert(output_dir: str | None = None) -> dict:
    out = Path(output_dir) if output_dir else OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    stats = {"total": 0, "written": 0, "skipped": 0, "entities": 0, "entities_skipped": 0}

    for dir_name, lang in sorted(DIR_TO_LANG.items()):
        lang_dir = DATA_DIR / dir_name
        if not lang_dir.is_dir():
            logger.warning("Missing directory: %s", lang_dir)
            continue

        logger.info("Processing %s (%s)", dir_name, lang)
        records = convert_lang(lang_dir, lang, stats)

        outfile = out / f"multiconer_v2_{lang}.jsonl"
        with open(outfile, "w") as fout:
            for rec in records:
                fout.write(rec.to_jsonl() + "\n")
                stats["written"] += 1

        logger.info("  %s: %d records written", lang, len(records))

    logger.info("multiconer_v2 total: %s", stats)
    return stats


if __name__ == "__main__":
    convert()
