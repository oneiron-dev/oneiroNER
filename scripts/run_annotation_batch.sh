#!/bin/bash
# Sequential annotation batch for Task 9 silver data
# Run one source at a time to avoid API rate limits
# Usage: bash scripts/run_annotation_batch.sh [--resume]

set -e
cd "$(dirname "$0")/.."

RESUME_FLAG=""
if [ "$1" = "--resume" ]; then
    RESUME_FLAG="--resume"
fi

# Order: smallest first, highest RELATIONSHIP_REF density prioritized
# mentalchat: 2,446 records (~2.5h)
# therapy_conversations: 5,721 records (~6h)
# synthetic_persona_chat: 6,852 records (~7h)
# roleplay_hieu: 5,734 records (~6h)
# personachat: 23,699 records - sample 10K
# prosocial_dialog: 30,874 records - sample 15K
# pippa: 66,456 records - sample 10K
# opencharacter: 100,593 records - sample 5K
# reddit_confessions: 210,099 records - sample 5K

echo "=== Task 9 Sequential Annotation Batch ==="
echo "Started at: $(date)"

for source in mentalchat therapy_conversations synthetic_persona_chat roleplay_hieu; do
    echo ""
    echo "--- Processing $source (full) at $(date) ---"
    python3 scripts/task9/annotate.py --source "$source" --concurrency 10 --provider spark $RESUME_FLAG
done

for source_limit in "personachat 10000" "prosocial_dialog 15000" "pippa 10000" "opencharacter 5000" "reddit_confessions 5000"; do
    source=$(echo "$source_limit" | cut -d' ' -f1)
    limit=$(echo "$source_limit" | cut -d' ' -f2)
    echo ""
    echo "--- Processing $source (limit $limit) at $(date) ---"
    python3 scripts/task9/annotate.py --source "$source" --limit "$limit" --concurrency 10 --provider spark $RESUME_FLAG
done

echo ""
echo "=== Batch complete at $(date) ==="
