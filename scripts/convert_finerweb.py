"""Convert fiNERweb multilingual parquet files to NER JSONL."""

import logging
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from lib.schema import NerRecord, Entity
from lib.span_validator import validate_span
from lib.negative_sampler import NegativeSampler

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data" / "raw" / "fiNERweb" / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "processed"

SOURCE = "finerweb"
CONFIDENCE = "synthetic-gold"
CAP_LANGS = {"en", "ja", "zh", "ko"}
CAP_SIZE = 50_000

ISO3_TO_2 = {
    "eng": "en", "jpn": "ja", "cmn": "zh", "kor": "ko", "deu": "de",
    "fra": "fr", "spa": "es", "ita": "it", "por": "pt", "nld": "nl",
    "rus": "ru", "pol": "pl", "tur": "tr", "ara": "ar", "hin": "hi",
    "ben": "bn", "tha": "th", "vie": "vi", "ind": "id", "msa": "ms",
    "swe": "sv", "dan": "da", "nor": "no", "fin": "fi", "ces": "cs",
    "slk": "sk", "hun": "hu", "ron": "ro", "bul": "bg", "hrv": "hr",
    "srp": "sr", "slv": "sl", "ukr": "uk", "ell": "el", "heb": "he",
    "kat": "ka", "hye": "hy", "urd": "ur", "fas": "fa", "tam": "ta",
    "tel": "te", "kan": "kn", "mal": "ml", "mar": "mr", "guj": "gu",
    "pan": "pa", "nep": "ne", "sin": "si", "mya": "my", "khm": "km",
    "lao": "lo", "zho": "zh", "yue": "yue", "lit": "lt", "lav": "lv",
    "ekk": "et", "cat": "ca", "eus": "eu", "glg": "gl", "afr": "af",
    "swa": "sw", "amh": "am", "som": "so", "hau": "ha", "yor": "yo",
    "ibo": "ig", "ceb": "ceb", "tgl": "tl", "jav": "jv", "sun": "su",
    "zsm": "ms", "arb": "ar",
}

neg_sampler = NegativeSampler()
rng = random.Random(42)


def convert(output_dir: str | None = None) -> dict:
    out = Path(output_dir) if output_dir else OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    total_stats = {"total": 0, "written": 0, "skipped": 0, "entities": 0,
                   "entities_skipped": 0, "files": 0}

    for pq_file in sorted(DATA_DIR.glob("*.parquet")):
        lang3 = pq_file.stem.split("-")[0]
        lang = ISO3_TO_2.get(lang3, lang3)

        table = pq.read_table(pq_file)
        rows = table.to_pydict()
        n_rows = len(rows["text"])

        if lang in CAP_LANGS and n_rows > CAP_SIZE:
            indices = sorted(rng.sample(range(n_rows), CAP_SIZE))
        else:
            indices = list(range(n_rows))

        outfile = out / f"finerweb_{lang}.jsonl"
        file_written = 0

        with open(outfile, "w") as fout:
            for row_idx in indices:
                total_stats["total"] += 1
                text = rows["text"][row_idx]
                char_spans = rows["char_spans"][row_idx]

                if not char_spans:
                    total_stats["skipped"] += 1
                    continue

                entities = []
                for span in char_spans:
                    start = span.get("start")
                    end = span.get("end")
                    label = span.get("label")
                    if start is None or end is None or not label:
                        total_stats["entities_skipped"] += 1
                        continue
                    if end <= start:
                        total_stats["entities_skipped"] += 1
                        continue
                    surface = text[start:end]
                    if not surface or not surface.strip():
                        total_stats["entities_skipped"] += 1
                        continue
                    if validate_span(text, surface, start, end):
                        entities.append(Entity(
                            surface=surface,
                            type=label,
                            original_type=label,
                            start=start,
                            end=end,
                        ).to_dict())
                        total_stats["entities"] += 1
                    else:
                        total_stats["entities_skipped"] += 1

                if not entities:
                    total_stats["skipped"] += 1
                    continue

                pos_types = {e["type"] for e in entities}
                neg_types = neg_sampler.sample(pos_types, rng=rng)
                query_types = sorted(pos_types) + neg_types

                rec = NerRecord(
                    source=SOURCE,
                    source_id=f"finerweb_{lang}_{row_idx}",
                    language=lang,
                    split="train",
                    confidence=CONFIDENCE,
                    provenance=[SOURCE],
                    text=text,
                    query_types=query_types,
                    entities=entities,
                )
                rec.validate()
                fout.write(rec.to_jsonl() + "\n")
                total_stats["written"] += 1
                file_written += 1

        total_stats["files"] += 1
        logger.info("finerweb %s (%s): %d rows -> %d records", lang3, lang, n_rows, file_written)

    logger.info("finerweb total: %s", total_stats)
    return total_stats


if __name__ == "__main__":
    convert()
