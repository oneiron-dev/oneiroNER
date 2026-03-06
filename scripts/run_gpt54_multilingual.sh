#!/bin/bash
# GPT-5.4 multilingual NER generation via opencode
# Usage: ./run_gpt54_multilingual.sh [tuning|batch] [lang1 lang2 ...]
# Example: ./run_gpt54_multilingual.sh tuning da no fi
# Example: ./run_gpt54_multilingual.sh batch da no fi

set -euo pipefail

MODE="${1:-tuning}"
shift || true
LANGS="${@:-da no fi hr hu sk et lt lv ca af tl ms sw el he fa bg sr bn ta te ur ml pa}"

COUNT=10
if [ "$MODE" = "batch" ]; then
    COUNT=50
fi

OUTDIR="/home/ubuntu/projects/oneiron-ner/data/raw/silver_synthetic"

# Disable opencode skills to avoid interference
mv ~/.config/opencode/skills ~/.config/opencode/skills.bak 2>/dev/null || true
trap 'mv ~/.config/opencode/skills.bak ~/.config/opencode/skills 2>/dev/null || true' EXIT

for LANG in $LANGS; do
    if [ "$MODE" = "tuning" ]; then
        OUTFILE="${OUTDIR}/${LANG}_gpt54_tuning.jsonl"
        ID_PREFIX="${LANG}_gpt54_t"
    else
        OUTFILE="${OUTDIR}/${LANG}_gpt54_batch1.jsonl"
        ID_PREFIX="${LANG}_gpt54_b1"
    fi

    if [ -f "$OUTFILE" ] && [ "$(wc -l < "$OUTFILE")" -eq "$COUNT" ]; then
        echo "SKIP $LANG: $OUTFILE already has $COUNT lines"
        continue
    fi

    echo "=== Generating $LANG ($MODE, $COUNT convos) ==="

    # Use the prompt template generator with gpt54 provider
    PROMPT=$(python3 /home/ubuntu/projects/oneiron-ner/scripts/multilingual_prompt_templates.py "$LANG" "$COUNT" "$MODE" "gpt54")

    opencode run -m "openai/gpt-5.4" --format json "$PROMPT" 2>&1 | tail -5

    # Verify
    if [ -f "$OUTFILE" ]; then
        LINES=$(wc -l < "$OUTFILE")
        BAD=$(python3 -c "
import json
bad=0
with open('$OUTFILE') as f:
    for line in f:
        rec=json.loads(line)
        for e in rec['entities']:
            text=rec['turns'][e['turn_index']]['text']
            if text[e['start']:e['end']]!=e['surface']:bad+=1
print(bad)
")
        echo "$LANG: $LINES records, $BAD bad offsets"
        if [ "$BAD" -gt 0 ]; then
            echo "  Running span_fixer..."
            python3 /home/ubuntu/projects/oneiron-ner/scripts/span_fixer.py "$OUTFILE"
        fi
    else
        echo "$LANG: OUTPUT FILE NOT CREATED"
    fi

    echo ""
done

echo "=== Done ==="
