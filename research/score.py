"""Composite NER score, hard gates, smoke checks, and TSV formatting."""

import math

GATES = {
    "macro_f1_floor": 0.40,
    "rel_ref_precision_floor": 0.30,
    "model_size_mb_max": 1500,
}

WEIGHTS = {
    "rel_ref_f05": 0.40,
    "macro_f1": 0.25,
    "rel_ref_hard_neg_precision": 0.20,
    "multilingual_f1": 0.10,
    "latency_bonus": 0.05,
}

TSV_HEADER = "\t".join([
    "commit", "status", "composite", "macro_f1", "micro_f1",
    "rel_ref_f05", "rel_ref_precision", "rel_hard_neg_prec",
    "multilingual_f1", "latency_bonus", "tokens_seen", "steps",
    "peak_vram_mb", "description",
])


def check_gates(metrics: dict) -> tuple[bool, list[str]]:
    failures = []
    macro_f1 = metrics.get("macro_f1", 0.0)
    if macro_f1 < GATES["macro_f1_floor"]:
        failures.append(f"macro_f1 {macro_f1:.3f} < {GATES['macro_f1_floor']}")

    rel_prec = metrics.get("rel_ref_precision", 0.0)
    if rel_prec < GATES["rel_ref_precision_floor"]:
        failures.append(f"rel_ref_precision {rel_prec:.3f} < {GATES['rel_ref_precision_floor']}")

    model_mb = metrics.get("model_size_mb", 0.0)
    if model_mb > GATES["model_size_mb_max"]:
        failures.append(f"model_size_mb {model_mb:.0f} > {GATES['model_size_mb_max']}")

    return len(failures) == 0, failures


def compute_composite(metrics: dict) -> float:
    score = 0.0
    score += WEIGHTS["rel_ref_f05"] * metrics.get("rel_ref_f0.5", 0.0)
    score += WEIGHTS["macro_f1"] * metrics.get("macro_f1", 0.0)
    score += WEIGHTS["rel_ref_hard_neg_precision"] * metrics.get("rel_ref_hard_neg_precision", 0.0)
    score += WEIGHTS["multilingual_f1"] * metrics.get("multilingual_f1", 0.0)
    score += WEIGHTS["latency_bonus"] * metrics.get("latency_bonus", 0.0)
    return score


def compare(current: dict, best: dict) -> str:
    passed, _ = check_gates(current)
    if not passed:
        return "discard"
    cur_score = compute_composite(current)
    best_score = compute_composite(best) if best else -1.0
    return "keep" if cur_score > best_score else "discard"


def check_smoke(metrics: dict) -> tuple[bool, str]:
    loss = metrics.get("eval_loss", metrics.get("loss", None))
    if loss is not None:
        if math.isnan(loss) or math.isinf(loss):
            return False, f"NaN/inf loss: {loss}"
        if loss > 10.0:
            return False, f"Loss {loss:.2f} > 10.0"

    macro_f1 = metrics.get("eval_macro_f1", None)
    if macro_f1 is not None and macro_f1 == 0.0:
        return False, "eval_macro_f1 is exactly 0.0 — model not learning"

    vram = metrics.get("peak_vram_mb", 0)
    if vram > 78_000:
        return False, f"peak_vram_mb {vram:.0f} > 78GB — likely OOM"

    return True, ""


def format_tsv_row(commit: str, metrics: dict, status: str,
                   description: str) -> str:
    composite = compute_composite(metrics)
    fields = [
        commit[:8],
        status,
        f"{composite:.4f}",
        f"{metrics.get('macro_f1', 0.0):.4f}",
        f"{metrics.get('micro_f1', 0.0):.4f}",
        f"{metrics.get('rel_ref_f0.5', 0.0):.4f}",
        f"{metrics.get('rel_ref_precision', 0.0):.4f}",
        f"{metrics.get('rel_ref_hard_neg_precision', 0.0):.4f}",
        f"{metrics.get('multilingual_f1', 0.0):.4f}",
        f"{metrics.get('latency_bonus', 0.0):.4f}",
        str(metrics.get("tokens_seen", 0)),
        str(metrics.get("num_steps", 0)),
        f"{metrics.get('peak_vram_mb', 0.0):.0f}",
        description.replace("\t", " ").replace("\n", " "),
    ]
    return "\t".join(fields)
