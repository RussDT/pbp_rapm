#!/bin/bash
# Build today's VPM board from R2 and publish it if every gate passes.
set -euo pipefail
PIPE=/Users/russ/Documents/Codex/basketball/nba/research/rapm/nba_pipeline
PY=$PIPE/../.venv/bin/python
TAG=$(TZ=America/Los_Angeles date +%Y%m%d)
LOG=$PIPE/logs/vpm_daily; mkdir -p "$LOG"
exec >>"$LOG/$TAG.log" 2>&1
notify_failure() {
  local code=$?
  if [ "$code" -ne 0 ]; then
    echo "== FAILED ($code) $(date)"
    /usr/bin/osascript -e "display notification \"Log: logs/vpm_daily/$TAG.log\" with title \"VPM nightly failed\"" || true
  fi
}
trap notify_failure EXIT
echo "== start $(date)"
cd "$PIPE"
"$PY" scripts/vpm_daily/run_vpm_daily.py
"$PY" scripts/vpm_daily/publish_vpm_preview.py "results/estimated_skills/current_refresh_${TAG}_sq_v1/supabase_preview.json" --publish
"$PY" scripts/vpm_daily/prune_stages.py --keep 3
echo "== done $(date)"
