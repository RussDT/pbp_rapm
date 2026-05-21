#!/bin/bash
# Sync CSV files from nba_pipeline/results to the viewer

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$SCRIPT_DIR/../nba_pipeline/results"
DEST_DIR="$SCRIPT_DIR/public/data"

mkdir -p "$DEST_DIR"
cp "$SRC_DIR"/*.csv "$DEST_DIR/" 2>/dev/null

# Generate manifest
ls "$DEST_DIR"/*.csv 2>/dev/null | xargs -n1 basename | sort | jq -R . | jq -s . > "$DEST_DIR/manifest.json"

echo "Synced $(ls "$DEST_DIR"/*.csv 2>/dev/null | wc -l | tr -d ' ') CSV files"
