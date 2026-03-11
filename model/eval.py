"""Exact char-span F1 evaluation for NER."""

import logging
from collections import defaultdict

import numpy as np

from model.config import ID2LABEL, SYNC_TYPES, BASE_TYPES, IGNORE_INDEX, collapse_to_base

logger = logging.getLogger(__name__)

REL_REF_HARD_NEGATIVES = {
    "i", "me", "my", "mine", "myself", "you", "your", "yours",
    "he", "him", "his", "she", "her", "hers", "it", "its",
    "we", "us", "our", "they", "them", "their",
    "someone", "anyone", "everyone", "nobody", "somebody",
    "people", "person", "man", "woman", "guy", "girl",
    "friend", "family", "partner", "colleague", "neighbor",
    "the other", "that person", "this guy", "some guy",
    "folks", "crew", "team", "group", "couple",
}


def decode_bio_to_char_spans(
    tag_ids: list[int],
    offset_mapping: list[tuple[int, int]],
) -> list[dict]:
    spans = []
    cur_type = None
    cur_start = -1
    cur_end = -1

    for i, tid in enumerate(tag_ids):
        if tid == IGNORE_INDEX or i >= len(offset_mapping):
            if cur_type is not None:
                spans.append({"type": cur_type, "start": cur_start, "end": cur_end})
                cur_type = None
            continue

        cs, ce = offset_mapping[i]
        if cs == 0 and ce == 0:
            if cur_type is not None:
                spans.append({"type": cur_type, "start": cur_start, "end": cur_end})
                cur_type = None
            continue

        label = ID2LABEL.get(tid, "O")

        if label == "O":
            if cur_type is not None:
                spans.append({"type": cur_type, "start": cur_start, "end": cur_end})
                cur_type = None
        elif label.startswith("B-"):
            if cur_type is not None:
                spans.append({"type": cur_type, "start": cur_start, "end": cur_end})
            cur_type = label[2:]
            cur_start = cs
            cur_end = ce
        elif label.startswith("I-"):
            etype = label[2:]
            if cur_type == etype:
                cur_end = ce
            else:
                if cur_type is not None:
                    spans.append({"type": cur_type, "start": cur_start, "end": cur_end})
                cur_type = etype
                cur_start = cs
                cur_end = ce
        else:
            if cur_type is not None:
                spans.append({"type": cur_type, "start": cur_start, "end": cur_end})
                cur_type = None

    if cur_type is not None:
        spans.append({"type": cur_type, "start": cur_start, "end": cur_end})

    return spans


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def _f_beta(p: float, r: float, beta: float) -> float:
    if p + r == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


def compute_span_metrics(pred_spans: list[dict], gold_spans: list[dict]) -> dict:
    pred_set = {(s["type"], s["start"], s["end"]) for s in pred_spans}
    gold_set = {(s["type"], s["start"], s["end"]) for s in gold_spans}

    tp_set = pred_set & gold_set
    fp_set = pred_set - gold_set
    fn_set = gold_set - pred_set

    counts: dict[str, dict] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for t, _, _ in tp_set:
        counts[t]["tp"] += 1
    for t, _, _ in fp_set:
        counts[t]["fp"] += 1
    for t, _, _ in fn_set:
        counts[t]["fn"] += 1

    return dict(counts)


def compute_all_metrics(
    predictions: list[list[dict]],
    golds: list[list[dict]],
) -> dict:
    subtype_counts: dict[str, dict] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for pred, gold in zip(predictions, golds):
        per_type = compute_span_metrics(pred, gold)
        for t, c in per_type.items():
            subtype_counts[t]["tp"] += c["tp"]
            subtype_counts[t]["fp"] += c["fp"]
            subtype_counts[t]["fn"] += c["fn"]

    result = {}

    for t in SYNC_TYPES:
        c = subtype_counts.get(t, {"tp": 0, "fp": 0, "fn": 0})
        m = _prf(c["tp"], c["fp"], c["fn"])
        result[f"subtype/{t}/precision"] = m["precision"]
        result[f"subtype/{t}/recall"] = m["recall"]
        result[f"subtype/{t}/f1"] = m["f1"]

    base_counts: dict[str, dict] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for t, c in subtype_counts.items():
        base = collapse_to_base(t)
        base_counts[base]["tp"] += c["tp"]
        base_counts[base]["fp"] += c["fp"]
        base_counts[base]["fn"] += c["fn"]

    base_f1s = []
    for bt in sorted(BASE_TYPES):
        c = base_counts.get(bt, {"tp": 0, "fp": 0, "fn": 0})
        m = _prf(c["tp"], c["fp"], c["fn"])
        result[f"base/{bt}/precision"] = m["precision"]
        result[f"base/{bt}/recall"] = m["recall"]
        result[f"base/{bt}/f1"] = m["f1"]
        base_f1s.append(m["f1"])

    result["macro_f1"] = sum(base_f1s) / len(base_f1s) if base_f1s else 0.0

    total_tp = sum(c["tp"] for c in subtype_counts.values())
    total_fp = sum(c["fp"] for c in subtype_counts.values())
    total_fn = sum(c["fn"] for c in subtype_counts.values())
    micro = _prf(total_tp, total_fp, total_fn)
    result["micro_f1"] = micro["f1"]
    result["micro_precision"] = micro["precision"]
    result["micro_recall"] = micro["recall"]

    rel_c = base_counts.get("RELATIONSHIP_REF", {"tp": 0, "fp": 0, "fn": 0})
    rel_m = _prf(rel_c["tp"], rel_c["fp"], rel_c["fn"])
    result["rel_ref_precision"] = rel_m["precision"]
    result["rel_ref_f0.5"] = _f_beta(rel_m["precision"], rel_m["recall"], 0.5)

    return result


def compute_metrics_for_trainer(eval_pred) -> dict:
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    preds = np.argmax(logits, axis=-1)

    base_counts: dict[str, dict] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    total_tp = total_fp = total_fn = 0

    for pred_seq, label_seq in zip(preds, labels):
        for p, g in zip(pred_seq, label_seq):
            if g == IGNORE_INDEX:
                continue
            p_label = ID2LABEL.get(int(p), "O")
            g_label = ID2LABEL.get(int(g), "O")

            p_base = collapse_to_base(p_label.split("-", 1)[1]) if "-" in p_label else "O"
            g_base = collapse_to_base(g_label.split("-", 1)[1]) if "-" in g_label else "O"

            if p_base == g_base and p_base != "O":
                base_counts[p_base]["tp"] += 1
                total_tp += 1
            else:
                if p_base != "O":
                    base_counts[p_base]["fp"] += 1
                    total_fp += 1
                if g_base != "O":
                    base_counts[g_base]["fn"] += 1
                    total_fn += 1

    result = {}
    base_f1s = []
    for bt in sorted(BASE_TYPES):
        c = base_counts.get(bt, {"tp": 0, "fp": 0, "fn": 0})
        m = _prf(c["tp"], c["fp"], c["fn"])
        result[f"eval_{bt}_f1"] = m["f1"]
        base_f1s.append(m["f1"])

    result["eval_macro_f1"] = sum(base_f1s) / len(base_f1s) if base_f1s else 0.0

    micro = _prf(total_tp, total_fp, total_fn)
    result["eval_micro_f1"] = micro["f1"]

    rel_c = base_counts.get("RELATIONSHIP_REF", {"tp": 0, "fp": 0, "fn": 0})
    rel_m = _prf(rel_c["tp"], rel_c["fp"], rel_c["fn"])
    result["eval_rel_ref_precision"] = rel_m["precision"]

    return result


def check_rel_ref_false_positives(pred_spans: list[dict], text: str) -> dict:
    total = 0
    fp_count = 0
    examples = []

    for span in pred_spans:
        if collapse_to_base(span["type"]) != "RELATIONSHIP_REF":
            continue
        total += 1
        surface = text[span["start"]:span["end"]].lower().strip()
        if surface in REL_REF_HARD_NEGATIVES:
            fp_count += 1
            examples.append({"surface": surface, "start": span["start"], "end": span["end"]})

    return {
        "total_rel_ref": total,
        "false_positive_count": fp_count,
        "false_positive_rate": fp_count / total if total else 0.0,
        "examples": examples,
    }


def truncation_report(dataset) -> dict:
    n_records = 0
    n_truncated_records = 0
    n_total_entities = 0
    n_truncated_entities = 0
    by_source: dict[str, dict] = defaultdict(lambda: {"total": 0, "truncated": 0})
    by_type: dict[str, dict] = defaultdict(lambda: {"total": 0, "truncated": 0})

    for rec in dataset:
        n_records += 1
        text = rec.get("text", "")
        offsets = rec.get("offset_mapping", [])
        source = rec.get("source", "unknown")
        entities = rec.get("entities", [])

        max_char = 0
        for _, e in offsets:
            if e > max_char:
                max_char = e

        truncated = len(text) > max_char > 0
        if truncated:
            n_truncated_records += 1
            by_source[source]["truncated"] += 1
        by_source[source]["total"] += 1

        for ent in entities:
            n_total_entities += 1
            etype = ent.get("type", "UNKNOWN")
            by_type[etype]["total"] += 1
            if ent.get("end", 0) > max_char and max_char > 0:
                n_truncated_entities += 1
                by_type[etype]["truncated"] += 1

    for src, counts in by_source.items():
        pct = counts["truncated"] / counts["total"] if counts["total"] else 0.0
        if pct > 0.10:
            logger.warning("Source '%s' has %.1f%% truncated records", src, 100 * pct)

    return {
        "pct_records_truncated": n_truncated_records / n_records if n_records else 0.0,
        "pct_entities_truncated": n_truncated_entities / n_total_entities if n_total_entities else 0.0,
        "by_source": dict(by_source),
        "by_type": dict(by_type),
    }


def run_full_eval(model, eval_dataloader, id2label, device) -> dict:
    """Run full char-span eval. Returns metrics dict.

    Reused by: AutoresearchCallback, manual eval scripts, score.py.
    Single source of truth for NER evaluation.
    """
    import torch
    model.eval()
    all_preds, all_golds = [], []
    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
        logits = outputs["logits"]
        pred_ids = logits.argmax(dim=-1).cpu().tolist()
        label_ids = batch["labels"].tolist()
        offset_mappings = batch.get("offset_mapping", [None] * len(pred_ids))

        for pred_seq, label_seq, offsets in zip(pred_ids, label_ids, offset_mappings):
            if offsets is None:
                continue
            pred_spans = decode_bio_to_char_spans(pred_seq, offsets)
            gold_spans = decode_bio_to_char_spans(
                [l if l != IGNORE_INDEX else 0 for l in label_seq], offsets
            )
            all_preds.append(pred_spans)
            all_golds.append(gold_spans)

    return compute_all_metrics(all_preds, all_golds)


def eval_multilingual(_model, _tokenizer, _holdout_path: str) -> dict:
    raise NotImplementedError("Multilingual holdout not yet selected")


def benchmark_latency(_model, _tokenizer, _text: str, _n_runs: int = 100) -> dict:
    raise NotImplementedError("Latency benchmark not yet implemented")
