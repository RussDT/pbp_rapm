#!/bin/bash
# Daily refresh of the WNBA Role / zTS surface consumed by databallr's stats table.
#
# Flow:
#   1. sync Gabriel's public WNBA play-by-play checkout
#   2. rebuild the playtype-difficulty artifact and publish the role board
#   3. upsert the board into Supabase so the live site picks it up (~25 min:
#      900s role cache + 600s response cache) with no databallr deploy
#   4. stage the fallback CSV into databallr's public folders
#
# Staging only copies files. Committing and deploying databallr stays a
# deliberate operator step, so this script never runs git in that repo.
#
# The job is idempotent: out of season the source does not move and the publish
# is a byte-identical no-op.

set -uo pipefail

REPO_ROOT="/Users/russ/Documents/Codex/basketball/nba/research/rapm"
DATABALLR_ROOT="/Users/russ/Documents/Codex/basketball/nba/products/databallr"
PYTHON="${REPO_ROOT}/.venv/bin/python"
LOG_DIR="${REPO_ROOT}/wnba_test/logs"
LOG_FILE="${LOG_DIR}/playtype_role_refresh.log"

mkdir -p "${LOG_DIR}"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >> "${LOG_FILE}"
}

cd "${REPO_ROOT}" || { log "FATAL: cannot cd to ${REPO_ROOT}"; exit 1; }

if [ ! -x "${PYTHON}" ]; then
  log "FATAL: interpreter missing at ${PYTHON}"
  exit 1
fi

log "=== refresh start ==="

# Fail soft: a transient GitHub outage should still let us republish from the
# checkout we already have rather than skipping the day entirely.
if "${PYTHON}" wnba_test/sync_gabriel_wnba.py >> "${LOG_FILE}" 2>&1; then
  log "gabriel sync ok ($(git -C nba_pipeline/external/gabriel_wnba rev-parse --short HEAD 2>/dev/null))"
else
  log "WARN: gabriel sync failed; republishing from the existing checkout"
fi

PUBLISH_ARGS=(--rebuild --stage-databallr "${DATABALLR_ROOT}")
if [ "${WNBA_ROLE_UPLOAD:-1}" = "1" ]; then
  PUBLISH_ARGS+=(--upload)
fi

if "${PYTHON}" wnba_test/publish_wnba_playtype_role.py "${PUBLISH_ARGS[@]}" >> "${LOG_FILE}" 2>&1; then
  log "publish ok"
else
  log "FATAL: publish failed; the previous board is still live"
  log "=== refresh end (failed) ==="
  exit 1
fi

log "=== refresh end (ok) ==="
