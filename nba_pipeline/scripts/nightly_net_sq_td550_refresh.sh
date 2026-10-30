#!/usr/bin/env bash
# Nightly refresh of the Net SQ td550 rolling chain, gated per stage.
#
# Nothing in this repo was scheduled: no launchd plist, no caller for
# build_pbp_shot_quality.py, none for run_net_sq_decomp_rolling.py, none for
# publish_td550_rolling_archive.py. That single gap is why the three surfaces
# are each stale at a different date -- the shot-quality table at 2026-07-22,
# the net-lane targets at 2026-09-06, the published archive at end year 2026 --
# while raw_data/NBA27.parquet landed 2026-10-26. This script closes the gap.
#
# Run it on a schedule. Each tick costs a handful of stat() calls and a JSON
# read, and does nothing unless an upstream has actually moved. Rebuilding
# 6.3M shots and five RAPM windows unconditionally would be hours of pointless
# compute every night.
#
# Stages, each with its own gate, in dependency order:
#
#   A  shot quality  <- raw_data/NBA*.parquet
#        Signal: the newest raw season file is newer than the shot-quality
#        parquet for this season year. Mtime-derived, so it is self-calibrating
#        and needs no state; a stage that fails leaves its output old and the
#        next tick retries rather than waiting out an interval.
#        It builds BESIDE the existing artifact, never over it:
#        --output-name defaults to pbp_shot_quality_all, so the
#        pbp_shot_quality_1997_2026 name in the committed configs was passed
#        explicitly and a rerun under a new name cannot clobber it. Nothing
#        here repoints a consumer off the 2026-named file.
#
#   B  per-shot EV  <- the sibling VPM harness (chronological-sq)
#        Signal: clean_ev.parquet is older than the stage A artifact.
#        ALSO gated on config coverage, which is why this stage can report
#        BLOCKED rather than RUN: the committed clean_sq config stops at
#        score_seasons 2026 and lists 2027 in protected_seasons. Scoring a
#        protected confirmation season is an owner decision, not something a
#        nightly may take, so when the config does not cover SEASON_YEAR this
#        stage and everything downstream of it skip and the script says so.
#        Point VPM_SQ_CONFIG at a config that covers the season to unblock it.
#
#   C  net lane  <- stage B
#        Signal: the three target parquets for this season year are missing
#        from the net-lane processed dir, or older than clean_ev.parquet.
#
#   D  rolling solve  <- stage C
#        Signal: any of the five td550 window files for this release end year
#        is missing from master_results, or older than the stage C targets.
#        Uses --allow-missing-playoffs, so a season that has played no
#        playoffs yet solves its regular-season rows and records
#        end_year_playoffs_included=false in its manifest.
#
#   E  Supabase publish  --  OFF BY DEFAULT, and deliberately so.
#        nba_no_prior_vpm_td550_rolling is live and feeds the site, and Russ
#        has not authorised an unattended write to it. Every tick computes and
#        validates the archive without --publish; the write happens only when
#        PUBLISH=1 is set explicitly. Its success stamp is the sha256 of the
#        staged sources.json and is recorded only after the publish returns 0,
#        so a failed publish retries on the next tick.
#
# Env overrides:
#   SEASON_YEAR=2027      season end year (N means season N-1 to N)
#   FORCE=1               run every stage regardless of its gate
#   PUBLISH=1             opt in to the Supabase write (stage E)
#   ONLY=A,C              run only these stages (gates still apply)
#   NIGHTLY_STUB=1        log each stage's command instead of running it,
#                         for verifying the gate and stamp logic
#   NIGHTLY_STUB_FAIL=D   stages the stub should fail, to check that a failure
#                         records no stamp and blocks what depends on it
#   NIGHTLY_PIPELINE_DIR  run the gates against another nba_pipeline tree
#   VPM_REPO              sibling VPM checkout
#   VPM_SQ_CONFIG         clean_sq harness config

set -uo pipefail

SEASON_YEAR="${SEASON_YEAR:-2027}"
YY="$(printf '%02d' $((SEASON_YEAR % 100)))"

# NIGHTLY_PIPELINE_DIR exists so the gates can be exercised against the real
# data tree from a worktree checkout, which has no results/ or master_results/.
PIPELINE_DIR="${NIGHTLY_PIPELINE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO_ROOT="$(cd "$PIPELINE_DIR/.." && pwd)"
PY="$REPO_ROOT/.venv/bin/python"

RAW_DIR="$PIPELINE_DIR/raw_data"
SQ_DIR="$PIPELINE_DIR/results/pbp_shot_quality"
SQ_NAME="pbp_shot_quality_1997_${SEASON_YEAR}"
SQ_PARQUET="$SQ_DIR/$SQ_NAME.parquet"

VPM_REPO="${VPM_REPO:-$(cd "$REPO_ROOT/../vpm" 2>/dev/null && pwd)}"
VPM_SQ_CONFIG="${VPM_SQ_CONFIG:-$VPM_REPO/vpm/configs/research_harness_v39_clean_sq.json}"
EV_PARQUET="$VPM_REPO/vpm/results/harness/vpm_clean_sq_preseason_v2/sq/clean_ev.parquet"

NET_LANE_DIR="$PIPELINE_DIR/results/vpm_target_swap_netcont/processed"
MASTER_DIR="$PIPELINE_DIR/master_results/net_sq_decomp_rolling"
PUBLICATION_DIR="$PIPELINE_DIR/results/td550_rolling_publication"

STATE_DIR="$PIPELINE_DIR/logs/nightly_net_sq"
PUBLISH_STATE="$STATE_DIR/last-published-sources-$SEASON_YEAR"
LOG="$STATE_DIR/$(date +%F).log"

mkdir -p "$STATE_DIR"
exec > >(tee -a "$LOG") 2>&1

log() { echo "[$(date '+%F %T %Z')] $*"; }

# Newer than, by mtime, treating an absent reference as "yes".
newer_than() {
    local candidate="$1" reference="$2"
    [[ -e "$candidate" ]] || return 1
    [[ -e "$reference" ]] || return 0
    [[ "$candidate" -nt "$reference" ]]
}

wants() {
    [[ -z "${ONLY:-}" ]] && return 0
    [[ ",${ONLY}," == *",$1,"* ]]
}

log "=== nightly Net SQ td550 check (season year $SEASON_YEAR, token $YY)"
[[ -x "$PY" ]] || { log "FATAL: no venv python at $PY"; exit 1; }
[[ -d "$VPM_REPO" ]] || log "WARNING: no sibling VPM checkout at $VPM_REPO; stage B will block"

NEED_SQ=0
NEED_EV=0
NEED_NET=0
NEED_SOLVE=0
EV_BLOCKED=0
BLOCKED=()

# --- gate A: shot quality, against the raw season files ---------------------
NEWEST_RAW="$(ls -t "$RAW_DIR"/NBA*.parquet 2>/dev/null | head -n 1)"
if [[ -z "$NEWEST_RAW" ]]; then
    log "shot quality: no raw season files under $RAW_DIR; nothing to build from"
    BLOCKED+=("A: no raw PBP")
elif [[ ! -e "$SQ_PARQUET" ]]; then
    log "shot quality: $SQ_NAME.parquet does not exist yet"
    NEED_SQ=1
elif newer_than "$NEWEST_RAW" "$SQ_PARQUET"; then
    log "shot quality: $(basename "$NEWEST_RAW") is newer than $SQ_NAME.parquet"
    NEED_SQ=1
else
    log "shot quality: up to date with $(basename "$NEWEST_RAW")"
fi

# --- gate B: per-shot EV, against the shot-quality artifact -----------------
# The coverage half of this gate is a refusal, not a staleness check: a nightly
# must not score a season the config marks protected.
EV_COVERAGE="$("$PY" - "$VPM_SQ_CONFIG" "$SEASON_YEAR" <<'PYGATE'
import json, sys
from pathlib import Path

path, year = Path(sys.argv[1]), int(sys.argv[2])
if not path.exists():
    print(f"BLOCKED no clean_sq config at {path}")
    raise SystemExit(0)
try:
    spec = json.loads(path.read_text()).get("clean_sq") or {}
except Exception as exc:
    print(f"BLOCKED unreadable clean_sq config: {type(exc).__name__}: {exc}")
    raise SystemExit(0)

scored = [int(v) for v in spec.get("score_seasons") or []]
protected = [int(v) for v in spec.get("protected_seasons") or []]
if year in protected:
    print(f"BLOCKED {year} is in clean_sq.protected_seasons {protected}; "
          "scoring a reserved confirmation season is an owner decision")
elif year not in scored:
    print(f"BLOCKED {year} is not in clean_sq.score_seasons "
          f"(ends {max(scored) if scored else 'n/a'})")
else:
    print(f"COVERED {year} is in clean_sq.score_seasons and not protected")
PYGATE
)"
log "per-shot EV config: $EV_COVERAGE"
if [[ "$EV_COVERAGE" == BLOCKED* ]]; then
    BLOCKED+=("B: ${EV_COVERAGE#BLOCKED }")
    EV_BLOCKED=1
elif (( NEED_SQ )) || newer_than "$SQ_PARQUET" "$EV_PARQUET"; then
    log "per-shot EV: clean_ev.parquet is older than the shot-quality artifact"
    NEED_EV=1
else
    log "per-shot EV: up to date with $SQ_NAME.parquet"
fi

# --- gate C: net lane, against the EV parquet -------------------------------
NET_TARGETS=(
    "$NET_LANE_DIR/FIRST_CHANCE$YY.parquet"
    "$NET_LANE_DIR/SECOND_CHANCE_CLEAN$YY.parquet"
    "$NET_LANE_DIR/RAPM$YY.parquet"
)
MISSING_NET=()
for target in "${NET_TARGETS[@]}"; do
    [[ -e "$target" ]] || MISSING_NET+=("$(basename "$target")")
done
if ((${#MISSING_NET[@]})); then
    log "net lane: ${#MISSING_NET[@]} of 3 ${SEASON_YEAR} targets missing (${MISSING_NET[*]})"
    NEED_NET=1
elif (( NEED_EV )) || newer_than "$EV_PARQUET" "${NET_TARGETS[0]}"; then
    log "net lane: clean_ev.parquet is newer than the $SEASON_YEAR targets"
    NEED_NET=1
else
    log "net lane: $SEASON_YEAR targets up to date"
fi

# --- gate D: rolling solve, against the net-lane targets --------------------
WINDOW_FILES=()
for seasons in 1 2 3 4 5; do
    start=$((SEASON_YEAR - seasons + 1))
    if (( seasons == 1 )); then
        compact="$YY"
    else
        compact="$(printf '%02d' $((start % 100)))_$YY"
    fi
    WINDOW_FILES+=("$MASTER_DIR/weighted_factors_vpm_net_sq_resid_${compact}_all_rb_se_td550.csv")
done
MISSING_WINDOWS=()
for window in "${WINDOW_FILES[@]}"; do
    [[ -e "$window" ]] || MISSING_WINDOWS+=("$(basename "$window")")
done
if ((${#MISSING_WINDOWS[@]})); then
    log "rolling solve: ${#MISSING_WINDOWS[@]} of 5 $SEASON_YEAR windows missing"
    NEED_SOLVE=1
elif (( NEED_NET )) || newer_than "${NET_TARGETS[0]}" "${WINDOW_FILES[0]}"; then
    log "rolling solve: the $SEASON_YEAR targets are newer than the windows"
    NEED_SOLVE=1
else
    log "rolling solve: $SEASON_YEAR windows up to date"
fi

if [[ -n "${FORCE:-}" ]]; then
    log "FORCE set; running every stage whose inputs are not blocked"
    NEED_SQ=1
    [[ "$EV_COVERAGE" == BLOCKED* ]] || NEED_EV=1
    NEED_NET=1
    NEED_SOLVE=1
fi

if ((${#BLOCKED[@]})); then
    for reason in "${BLOCKED[@]}"; do log "BLOCKED $reason"; done
fi

# --- work -------------------------------------------------------------------
FAILED=()
step() {
    local label="$1"; shift
    log "--> $label"
    if [[ -n "${NIGHTLY_STUB:-}" ]]; then
        log "    STUB: $*"
        if [[ ",${NIGHTLY_STUB_FAIL:-}," == *",${label%%:*},"* ]]; then
            log "    FAILED (stubbed): $label"
            FAILED+=("$label")
        else
            log "    ok (stubbed): $label"
        fi
        return
    fi
    if "$@"; then
        log "    ok: $label"
    else
        log "    FAILED: $label"
        FAILED+=("$label")
    fi
}
failed_with() { [[ " ${FAILED[*]-} " == *" $1 "* ]]; }

cd "$PIPELINE_DIR" || exit 1

if (( NEED_SQ )) && wants A; then
    step "A: shot quality" "$PY" "$PIPELINE_DIR/scripts/build_pbp_shot_quality.py" \
        --years "1997-$SEASON_YEAR" --season-types all \
        --output-dir "$SQ_DIR" --output-name "$SQ_NAME"
fi

# Each stage depends on the one above it, so a failure upstream stops the
# chain here rather than solving against a stale or half-written input.
if (( NEED_EV )) && wants B; then
    if failed_with "A: shot quality"; then
        log "skipping B: shot quality failed"
    else
        step "B: per-shot EV" env -C "$VPM_REPO" "$PY" -m vpm.harness.cli \
            chronological-sq --config "$VPM_SQ_CONFIG" --stage prepare
    fi
fi

if (( NEED_NET )) && wants C; then
    if (( EV_BLOCKED )); then
        log "skipping C: no $SEASON_YEAR per-shot EV, because stage B is blocked"
    elif failed_with "B: per-shot EV"; then
        log "skipping C: per-shot EV failed"
    elif [[ ! -e "$EV_PARQUET" ]]; then
        log "skipping C: no EV parquet at $EV_PARQUET"
    else
        step "C: net lane" "$PY" "$PIPELINE_DIR/scripts/build_vpm_target_swap.py" \
            --net-lane --years "$SEASON_YEAR" --ev-parquet "$EV_PARQUET" \
            --link-unchanged-families --overwrite
    fi
fi

if (( NEED_SOLVE )) && wants D; then
    if (( EV_BLOCKED )); then
        log "skipping D: no $SEASON_YEAR net-lane targets, because stage B is blocked"
    elif failed_with "C: net lane"; then
        log "skipping D: net lane failed"
    else
        step "D: rolling solve" "$PY" "$PIPELINE_DIR/scripts/run_net_sq_decomp_rolling.py" \
            --current-release --release-end-year "$SEASON_YEAR" \
            --profile vpm --timedecay --half-life 550 --allow-missing-playoffs
    fi
fi

# --- stage E: validate always, publish only on an explicit opt-in -----------
# Re-validating 140 unchanged window CSVs every night is the pointless compute
# this script exists to avoid, so stage E sits out a tick where nothing moved.
# An explicit PUBLISH or FORCE still runs it.
if wants E && (( NEED_SQ + NEED_EV + NEED_NET + NEED_SOLVE == 0 )) \
        && [[ -z "${PUBLISH:-}" && -z "${FORCE:-}" ]]; then
    log "skipping E: nothing upstream moved, so the archive is unchanged"
elif wants E; then
    # Any earlier failure, not just a failed D: a failed C skips D, which leaves
    # the windows unresolved, and an archive validated over a half-built season
    # must not reach the live table.
    if ((${#FAILED[@]})); then
        log "skipping E: an earlier stage failed (${FAILED[*]})"
    else
        step "E: validate archive" "$PY" "$PIPELINE_DIR/scripts/publish_td550_rolling_archive.py"
        if failed_with "E: validate archive"; then
            log "not publishing: the archive did not validate"
        elif [[ -z "${PUBLISH:-}" ]]; then
            log "not publishing: PUBLISH is unset (the default)."
            log "  The archive is validated and staged under $PUBLICATION_DIR."
            log "  nba_no_prior_vpm_td550_rolling is live, so the write is opt-in."
        else
            SOURCES_SHA="$(shasum -a 256 "$PUBLICATION_DIR/sources.json" 2>/dev/null | cut -d' ' -f1)"
            LAST_SHA="$(cat "$PUBLISH_STATE" 2>/dev/null || echo "")"
            if [[ -n "$SOURCES_SHA" && "$SOURCES_SHA" == "$LAST_SHA" ]]; then
                log "not publishing: this archive was already published ($SOURCES_SHA)"
            else
                step "E: publish" "$PY" \
                    "$PIPELINE_DIR/scripts/publish_td550_rolling_archive.py" --publish
                # Stamped only on success, so a failed publish retries next tick.
                if ! failed_with "E: publish" && [[ -n "$SOURCES_SHA" ]]; then
                    echo "$SOURCES_SHA" > "$PUBLISH_STATE"
                    log "recorded published archive $SOURCES_SHA"
                fi
            fi
        fi
    fi
fi

if ((${#FAILED[@]})); then
    log "=== finished WITH FAILURES: ${FAILED[*]}"
    exit 1
fi
if ((${#BLOCKED[@]})); then
    log "=== done, with blocked stages: ${BLOCKED[*]}"
    exit 0
fi
log "=== done"
