#!/usr/bin/env python3
"""Multilingual NER batch2 generation orchestrator.

Generates scenario-specific prompts for depth-first coverage on product languages.
This is a prompt/manifest generator that delegates actual generation to provider CLIs.
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from random import Random

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))
from multilingual_prompt_templates import LANG_CONFIG, build_prompt

DEFAULT_LANGS = ["de", "fr", "es", "it", "pt", "pl", "nl", "uk"]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "silver_synthetic_batch2"

SCENARIOS = {
    "relationship_heavy": {
        "weight": 1.0,
        "description": "Conversations with 3+ RELATIONSHIP_REF entities with diverse subtypes",
        "prompt_addendum": """
IMPORTANT SCENARIO REQUIREMENTS:
- Each conversation MUST contain at least 3 RELATIONSHIP_REF entities
- Use diverse subtypes: Family, Romantic, Friend, Professional, Acquaintance
- Include at least 2 different RELATIONSHIP_REF subtypes per conversation
- Include possessive forms ("my sister", "his boss") and named references ("Sarah's mom")
""",
    },
    "hard_negatives": {
        "weight": 1.0,
        "description": "Includes generic relationship nouns that should NOT be tagged",
        "prompt_addendum": """
IMPORTANT SCENARIO REQUIREMENTS:
- Include generic/hypothetical relationship mentions that should NOT be annotated
- Examples of spans to NOT annotate: "a friend told me" (generic), "any good doctor" (non-specific),
  "everyone needs a therapist" (general advice), "if I had a partner" (hypothetical)
- DO annotate specific references: "my friend Sarah", "his therapist", "our neighbor"
- The distinction: specific = refers to a real individual the speaker knows; generic = hypothetical/abstract
- Aim for at least 2 generic relationship mentions per conversation that are NOT in the entities list
""",
    },
    "asr_noisy": {
        "weight": 1.0,
        "description": "Simulated ASR errors (dropped articles, phonetic spelling)",
        "prompt_addendum": """
IMPORTANT SCENARIO REQUIREMENTS:
- Simulate realistic ASR transcription errors in the conversation text
- Include: dropped articles, phonetic misspellings, run-on words, missing punctuation
- Entity annotations must use the ACTUAL text (with errors), not corrected forms
- Examples: "gonna see my sis tmrw" (sis=RELATIONSHIP_REF/Family), "went 2 the concert" (concert=EVENT/General)
- 20-30% of turns should contain ASR-style noise
""",
    },
    "code_switching": {
        "weight": 1.0,
        "description": "Mixed target-language + English turns",
        "prompt_addendum": """
IMPORTANT SCENARIO REQUIREMENTS:
- Mix the target language with English within conversations
- Some turns entirely in English, some in target language, some mixed within a turn
- Entity annotations must be correct regardless of language of the span
- Cultural references should be natural for bilingual speakers
- At least 30% of turns should contain code-switching or be fully in the other language
""",
    },
    "cultural_kinship": {
        "weight": 1.0,
        "description": "Culture-specific kinship terms",
        "prompt_addendum": """
IMPORTANT SCENARIO REQUIREMENTS:
- Use culture-specific kinship and relationship terms native to the target language
- Include terms that don't have direct English equivalents where possible
- Examples: German "Schwiegermutter", Polish "teściowa", French "belle-mère"
- Include extended family terms, in-law terms, and culturally specific social roles
- All kinship terms should be annotated as RELATIONSHIP_REF with appropriate subtype
""",
    },
    "date_expressions": {
        "weight": 1.0,
        "description": "Dense date/temporal expressions across DATE subtypes",
        "prompt_addendum": """
IMPORTANT SCENARIO REQUIREMENTS:
- Each conversation MUST contain at least 4 DATE entities
- Use diverse DATE subtypes: Day, Week, Month, Year, Relative, Range
- Include culture-specific date formats and calendar references
- Include relative dates ("last Tuesday", "next month", "a few years ago")
- Include date ranges ("from March to June", "the whole summer")
""",
    },
}

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "conversations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "turns": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "speaker": {"type": "string", "enum": ["user", "assistant"]},
                                "text": {"type": "string"},
                            },
                            "required": ["speaker", "text"],
                        },
                    },
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "surface": {"type": "string"},
                                "type": {"type": "string"},
                                "start": {"type": "integer"},
                                "end": {"type": "integer"},
                                "turn_index": {"type": "integer"},
                            },
                            "required": ["surface", "type", "start", "end", "turn_index"],
                        },
                    },
                    "language": {"type": "string"},
                    "scenario": {"type": "string"},
                    "provider": {"type": "string"},
                    "prompt_version": {"type": "string", "const": "task9_5_v1"},
                    "holdout": {"type": "boolean"},
                },
                "required": ["id", "turns", "entities", "language", "scenario", "provider", "prompt_version", "holdout"],
            },
        },
    },
    "required": ["conversations"],
}


def distribute_counts(total: int, scenarios: list[str], weights: dict[str, float]) -> dict[str, int]:
    raw = {s: weights.get(s, 1.0) for s in scenarios}
    total_weight = sum(raw.values())
    fractional = {s: total * (w / total_weight) for s, w in raw.items()}

    allocated = {s: int(f) for s, f in fractional.items()}
    remainders = sorted(scenarios, key=lambda s: fractional[s] - allocated[s], reverse=True)
    deficit = total - sum(allocated.values())
    for i in range(deficit):
        allocated[remainders[i]] += 1

    return allocated


def holdout_indices(count: int, holdout_pct: int, seed: int, lang: str, scenario: str) -> set[int]:
    h = int(hashlib.sha256(f"{lang}_{scenario}".encode()).hexdigest(), 16)
    rng = Random(seed + (h % (2**31)))
    n_holdout = max(0, round(count * holdout_pct / 100))
    return set(rng.sample(range(count), min(n_holdout, count)))


def build_scenario_prompt(lang: str, scenario: str, count: int, provider: str) -> str:
    base = build_prompt(lang, count=count, mode="batch", batch_num=2, provider=provider)
    addendum = SCENARIOS[scenario]["prompt_addendum"]
    return f"{base}\n{addendum}"


def generate(args):
    output_dir = Path(args.output_dir)
    prompts_dir = output_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    langs = [l.strip() for l in args.langs.split(",")]
    for lang in langs:
        if lang not in LANG_CONFIG:
            print(f"ERROR: unknown language '{lang}' — not in LANG_CONFIG", file=sys.stderr)
            sys.exit(1)

    if args.scenarios == "all":
        scenarios = list(SCENARIOS.keys())
    else:
        scenarios = [s.strip() for s in args.scenarios.split(",")]
        for s in scenarios:
            if s not in SCENARIOS:
                print(f"ERROR: unknown scenario '{s}'", file=sys.stderr)
                sys.exit(1)

    weights = {}
    if args.scenario_weights:
        weights = json.loads(args.scenario_weights)

    distribution = distribute_counts(args.count, scenarios, weights)

    providers = [args.provider]
    if args.shadow_provider:
        providers.append(args.shadow_provider)

    prompt_files = []
    for lang in langs:
        for provider in providers:
            for scenario in scenarios:
                count = distribution[scenario]
                if count == 0:
                    continue

                holdout_set = holdout_indices(count, args.holdout_pct, args.seed, lang, scenario)
                holdout_count = len(holdout_set)

                prompt_text = build_scenario_prompt(lang, scenario, count, provider)
                header = (
                    f"# Lang: {lang}, Scenario: {scenario}, Provider: {provider}, "
                    f"Count: {count}, Holdout: {holdout_count}\n\n"
                )

                prompt_filename = f"{lang}_{provider}_{scenario}.txt"
                prompt_path = prompts_dir / prompt_filename
                if not args.dry_run:
                    prompt_path.write_text(header + prompt_text, encoding="utf-8")

                schema_filename = f"{lang}_{provider}_{scenario}_schema.json"
                schema_path = prompts_dir / schema_filename
                if not args.dry_run:
                    schema_path.write_text(json.dumps(OUTPUT_SCHEMA, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

                output_filename = f"{lang}_{provider}_batch2_{scenario}.jsonl"

                prompt_files.append({
                    "lang": lang,
                    "scenario": scenario,
                    "provider": provider,
                    "count": count,
                    "prompt_file": f"prompts/{prompt_filename}",
                    "schema_file": f"prompts/{schema_filename}",
                    "output_file": output_filename,
                    "holdout_count": holdout_count,
                    "holdout_indices": sorted(holdout_set),
                })

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_conversations": args.count * len(langs) * len(providers),
        "langs": langs,
        "count_per_lang": args.count,
        "provider": args.provider,
        "shadow_provider": args.shadow_provider,
        "holdout_pct": args.holdout_pct,
        "seed": args.seed,
        "scenario_distribution": distribution,
        "prompt_files": prompt_files,
    }

    manifest_path = output_dir / "batch2_manifest.json"
    if not args.dry_run:
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print_summary(manifest, langs, providers, scenarios, distribution, args)

    if not args.dry_run:
        print(f"\nManifest: {manifest_path}")
        print(f"Prompts:  {prompts_dir}/")

    print(f"\nAfter generation, run the verification pipeline:")
    print(f"  python scripts/task8/verify_output.py --output-dir {args.output_dir}")
    print(f"  python scripts/task8/audit_repeated_surfaces.py --output-dir {args.output_dir}")
    print(f"  python scripts/task8/clean_output.py --output-dir {args.output_dir}")


def print_summary(manifest, langs, providers, scenarios, distribution, args):
    mode = "DRY RUN" if args.dry_run else "GENERATED"
    print(f"\n{'='*60}")
    print(f"  Batch2 Multilingual NER — {mode}")
    print(f"{'='*60}")
    print(f"\nProviders: {', '.join(providers)}")
    print(f"Languages: {', '.join(langs)} ({len(langs)} total)")
    print(f"Count per lang per provider: {args.count}")
    print(f"Holdout: {args.holdout_pct}%")

    print(f"\nScenario distribution (per lang per provider):")
    for s in scenarios:
        print(f"  {s:25s} {distribution[s]:4d} convos")

    print(f"\nPer-language breakdown:")
    for lang in langs:
        lang_entries = [e for e in manifest["prompt_files"] if e["lang"] == lang]
        total_convos = sum(e["count"] for e in lang_entries)
        total_holdout = sum(e["holdout_count"] for e in lang_entries)
        providers_str = ", ".join(sorted({e["provider"] for e in lang_entries}))
        print(f"  {lang:4s} ({LANG_CONFIG[lang]['name']:20s}): {total_convos:4d} convos, {total_holdout:3d} holdout  [{providers_str}]")

    total_prompts = len(manifest["prompt_files"])
    total_convos = manifest["total_conversations"]
    print(f"\nTotal prompt files: {total_prompts}")
    print(f"Total expected conversations: {total_convos}")


def main():
    parser = argparse.ArgumentParser(description="Multilingual NER batch2 prompt/manifest generator")
    parser.add_argument("--langs", default=",".join(DEFAULT_LANGS),
                        help="Comma-separated language codes (default: %(default)s)")
    parser.add_argument("--count", type=int, default=100,
                        help="Total conversations PER LANGUAGE PER PROVIDER (default: %(default)s)")
    parser.add_argument("--provider", default="gpt54",
                        help="Primary provider: gpt54 or claude (default: %(default)s)")
    parser.add_argument("--shadow-provider", default=None,
                        help="Optional secondary provider")
    parser.add_argument("--scenarios", default="all",
                        help="Comma-separated scenarios or 'all' (default: %(default)s)")
    parser.add_argument("--scenario-weights", default=None,
                        help='JSON weights: {"relationship_heavy": 0.3, ...}')
    parser.add_argument("--dry-run", action="store_true",
                        help="Print manifest only, don't write files")
    parser.add_argument("--holdout-pct", type=int, default=10,
                        help="Percent of conversations to mark as holdout (default: %(default)s)")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                        help="Output directory (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for holdout assignment (default: %(default)s)")
    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
