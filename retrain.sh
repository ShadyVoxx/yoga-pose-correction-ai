#!/usr/bin/env bash
# retrain.sh — Full pipeline: download new Drive data → merge → train
# Usage: bash retrain.sh
set -euo pipefail
cd "$(dirname "$0")"

echo "═══════════════════════════════════════════════════════"
echo " YogaAlign Retrain Pipeline"
echo "═══════════════════════════════════════════════════════"

# ── Step 1: Download new landmarks from Google Drive ─────────────────────────
echo ""
echo "▶ Step 1/3 — Downloading new data from Google Drive..."
python3 download_drive_data.py --out ./new_sessions

# ── Step 2: Merge into training_data.json ────────────────────────────────────
echo ""
echo "▶ Step 2/3 — Merging new sessions into training_data.json..."
BEFORE=$(python3 -c "import json; d=json.load(open('training_data.json')); print(len(d))")
python3 merge_new_participants.py ./new_sessions training_data.json
AFTER=$(python3 -c "import json; d=json.load(open('training_data.json')); print(len(d))")
echo "   Entries: $BEFORE → $AFTER (+$((AFTER - BEFORE)))"

# ── Step 3: Retrain model ────────────────────────────────────────────────────
echo ""
echo "▶ Step 3/3 — Training TF.js model..."
node train_model.js 2>&1 | tee train_output.txt

echo ""
echo "✅ Done! New model saved to tfjs_model/"
echo "   Restart the server to pick up the updated weights."
