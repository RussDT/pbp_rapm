#!/bin/bash
# Daily NBA RAPM Update Script
# Runs at 6 AM via launchd
#
# Season handling
#   The season-year convention is that year N means season (N-1)-N, so
#   SEASON_YEAR=2027 is the 2026-27 season. Every two-digit token in result
#   folder/file names ("27", "22_27", ...) is derived from SEASON_YEAR; there
#   are no hardcoded season literals in the pipeline path.
#
#   Select the season with `--year 2027` or NBA_RAPM_SEASON_YEAR=2027.
#
# Path overrides (all optional; see nba_pipeline/CLAUDE.md)
#   NBA_RAPM_SEASON_YEAR            season end year (default 2027)
#   NBA_RAPM_PIPELINE_DIR           nba_pipeline dir (default: this script's dir)
#   NBA_RAPM_CSVS_DIR               csvs repo checkout (default <repo>/../csvs)
#   NBA_RAPM_CSVS_BRANCH            expected csvs branch (default test-branch)
#   NBA_RAPM_RAPMS_DIR              rapms repo checkout (default <repo>/../rapms)
#   NBA_RAPM_CURRENT_METRICS_DIR    nba_rapm dir of the downstream NBA pipeline
#   NBA_RAPM_CURRENT_METRICS_SCRIPT default collect_source_<SEASON_YEAR>.py
#   NBA_RAPM_PUSH                   0 to skip the csvs/rapms commits and pushes

set -e

# ---------------------------------------------------------------------
# Season selection
# ---------------------------------------------------------------------
SEASON_YEAR="${NBA_RAPM_SEASON_YEAR:-2027}"

usage() {
    cat <<USAGE
Usage: $(basename "$0") [--year YYYY]

  --year YYYY   Season end year to build (default $SEASON_YEAR; 2027 = 2026-27)
USAGE
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --year)
            SEASON_YEAR="$2"
            shift 2
            ;;
        --year=*)
            SEASON_YEAR="${1#*=}"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$SEASON_YEAR" in
    [0-9][0-9][0-9][0-9]) ;;
    *)
        echo "ERROR: --year expects a four-digit season end year, got '$SEASON_YEAR'" >&2
        exit 2
        ;;
esac

export NBA_RAPM_SEASON_YEAR="$SEASON_YEAR"

# Two-digit season-end form, e.g. 2027 -> 27
yy() {
    printf '%02d' $(( $1 % 100 ))
}

# Window token for a window starting N seasons before SEASON_YEAR:
#   win 0 -> "27", win 5 -> "22_27"
win() {
    local back="$1"
    if [ "$back" -le 0 ]; then
        yy "$SEASON_YEAR"
    else
        echo "$(yy $(( SEASON_YEAR - back )))_$(yy "$SEASON_YEAR")"
    fi
}

SEASON_YY="$(yy "$SEASON_YEAR")"
W1="$(win 0)"    # 1-year  window token
W2="$(win 1)"    # 2-year
W3="$(win 2)"    # 3-year
W4="$(win 3)"    # 4-year
W5="$(win 4)"    # 5-year
W6="$(win 5)"    # 6-year
W13="$(win 12)"  # 13-year

# Two-digit window start years, for scripts that take start/end positionally
S1="$(yy "$SEASON_YEAR")"
S2="$(yy $(( SEASON_YEAR - 1 )))"
S3="$(yy $(( SEASON_YEAR - 2 )))"
S4="$(yy $(( SEASON_YEAR - 3 )))"
S5="$(yy $(( SEASON_YEAR - 4 )))"
S6="$(yy $(( SEASON_YEAR - 5 )))"
S13="$(yy $(( SEASON_YEAR - 12 )))"

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
# Resolve this script's real location so the pipeline root does not depend on
# an absolute machine path or on being invoked through the scripts/ symlink.
resolve_script_dir() {
    local src="${BASH_SOURCE[0]}"
    while [ -L "$src" ]; do
        local dir
        dir="$(cd -P "$(dirname "$src")" && pwd)"
        src="$(readlink "$src")"
        case "$src" in
            /*) ;;
            *) src="$dir/$src" ;;
        esac
    done
    cd -P "$(dirname "$src")" && pwd
}

SCRIPT_DIR="$(resolve_script_dir)"
# The canonical location is nba_pipeline/daily_rapm_update.sh; tolerate a copy
# living in nba_pipeline/scripts/.
if [ "$(basename "$SCRIPT_DIR")" = "scripts" ]; then
    DEFAULT_PIPELINE_DIR="$(dirname "$SCRIPT_DIR")"
else
    DEFAULT_PIPELINE_DIR="$SCRIPT_DIR"
fi

PIPELINE_DIR="${NBA_RAPM_PIPELINE_DIR:-$DEFAULT_PIPELINE_DIR}"
PROJECT_DIR="$(dirname "$PIPELINE_DIR")"
LOG_FILE="$PIPELINE_DIR/logs/daily_update_$(date +%Y%m%d).log"

CSVS_DIR="${NBA_RAPM_CSVS_DIR:-$PROJECT_DIR/../csvs}"
CSVS_BRANCH="${NBA_RAPM_CSVS_BRANCH:-test-branch}"
RAPMS_DIR="${NBA_RAPM_RAPMS_DIR:-$PROJECT_DIR/../rapms}"
CURRENT_METRICS_DIR="${NBA_RAPM_CURRENT_METRICS_DIR:-}"
CURRENT_METRICS_SCRIPT="${NBA_RAPM_CURRENT_METRICS_SCRIPT:-collect_source_${SEASON_YEAR}.py}"
PUSH_DOWNSTREAM="${NBA_RAPM_PUSH:-1}"

# Downstream producers read these.
export NBA_RAPM_CSVS_DIR="$CSVS_DIR"

# Initialize pyenv (needed for launchd which doesn't source .zshrc)
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

PYTHON_PATH="python -u"  # -u disables output buffering

# Create logs directory if needed
mkdir -p "$PIPELINE_DIR/logs"

# Log function - prints to terminal AND appends to log file
log() {
    echo "$@" | tee -a "$LOG_FILE"
}

# Run command with output to both terminal and log
run() {
    "$@" 2>&1 | tee -a "$LOG_FILE"
}

log "========================================"
log "NBA RAPM Daily Update - $(date)"
log "Season: $SEASON_YEAR (season end year; two-digit $SEASON_YY)"
log "Pipeline dir: $PIPELINE_DIR"
log "========================================"

SCRIPTS_DIR="$PIPELINE_DIR/scripts"
RAW_DATA_DIR="$PIPELINE_DIR/raw_data"

copy_publish_artifact() {
    local source_path="$1"
    local publish_name="$2"
    local fallback_path="${3:-}"

    if [ -f "$source_path" ]; then
        cp "$source_path" "$MASTER_DIR/$publish_name"
        log "  Copied $(basename "$source_path") to master_results/$publish_name"
    elif [ -n "$fallback_path" ] && [ -f "$fallback_path" ]; then
        cp "$fallback_path" "$MASTER_DIR/$publish_name"
        log "  Copied fallback $(basename "$fallback_path") to master_results/$publish_name"
    else
        log "  WARNING: neither $(basename "$source_path") nor ${fallback_path:+$(basename "$fallback_path")} found"
    fi
}

# 1. Fetch latest season data
log ""
log "[$(date +%H:%M:%S)] Fetching latest regular-season PBP data..."
run $PYTHON_PATH "$SCRIPTS_DIR/01_fetch_pbp_data.py" "$SEASON_YY"

log ""
log "[$(date +%H:%M:%S)] Fetching latest playoff PBP data..."
run $PYTHON_PATH "$SCRIPTS_DIR/01_fetch_pbp_data.py" "$SEASON_YY" PS

# 2. Enrich with ShotQuality data (adds initial_ev, transition, etc.)
# The enrichment script processes both NBA<yy>.parquet and NBA<yy>_PS.parquet.
log ""
log "[$(date +%H:%M:%S)] Enriching PBP with ShotQuality data..."
run $PYTHON_PATH "$SCRIPTS_DIR/01b_enrich_pbp_shotquality.py" "$SEASON_YY"

# 3. Process RAPM for current season
log ""
log "[$(date +%H:%M:%S)] Processing regular-season RAPM data..."
run $PYTHON_PATH "$SCRIPTS_DIR/02_process_rapm.py" "$RAW_DATA_DIR/NBA${SEASON_YY}.parquet"

if [ -f "$RAW_DATA_DIR/NBA${SEASON_YY}_PS.parquet" ]; then
    log ""
    log "[$(date +%H:%M:%S)] Processing playoff RAPM data..."
    run $PYTHON_PATH "$SCRIPTS_DIR/02_process_rapm.py" "$RAW_DATA_DIR/NBA${SEASON_YY}_PS.parquet"
else
    log "  WARNING: $RAW_DATA_DIR/NBA${SEASON_YY}_PS.parquet not found; playoff processing skipped"
fi

# 4. Run RAPM analyses (most common year ranges)
log ""
log "[$(date +%H:%M:%S)] Running RAPM analysis (1 year)..."
run $PYTHON_PATH "$SCRIPTS_DIR/03_run_rapm_analysis.py" "$S1" "$SEASON_YY" ALL --season-effects

log ""
log "[$(date +%H:%M:%S)] Running RAPM analysis (2 years)..."
run $PYTHON_PATH "$SCRIPTS_DIR/03_run_rapm_analysis.py" "$S2" "$SEASON_YY" ALL --season-effects

log ""
log "[$(date +%H:%M:%S)] Running RAPM analysis (3 years)..."
run $PYTHON_PATH "$SCRIPTS_DIR/03_run_rapm_analysis.py" "$S3" "$SEASON_YY" ALL --season-effects

log ""
log "[$(date +%H:%M:%S)] Running RAPM analysis (4 years)..."
run $PYTHON_PATH "$SCRIPTS_DIR/03_run_rapm_analysis.py" "$S4" "$SEASON_YY" ALL --season-effects

log ""
log "[$(date +%H:%M:%S)] Running RAPM analysis (5 years)..."
run $PYTHON_PATH "$SCRIPTS_DIR/03_run_rapm_analysis.py" "$S5" "$SEASON_YY" ALL --season-effects

log ""
log "[$(date +%H:%M:%S)] Running RAPM analysis (6 years)..."
run $PYTHON_PATH "$SCRIPTS_DIR/03_run_rapm_analysis.py" "$S6" "$SEASON_YY" ALL --season-effects

# 4. Additional weighted-factors runs
log ""
log "[$(date +%H:%M:%S)] Running RAPM analysis (6 years, time decay 700)..."
run $PYTHON_PATH "$SCRIPTS_DIR/03_run_rapm_analysis.py" "$S6" "$SEASON_YY" ALL --season-effects --timedecay --half-life 700

log ""
log "[$(date +%H:%M:%S)] Running RAPM analysis (6 years, time decay 700 + rubberband)..."
run $PYTHON_PATH "$SCRIPTS_DIR/03_run_rapm_analysis.py" "$S6" "$SEASON_YY" ALL --season-effects --timedecay --half-life 700 --rubberband

log ""
log "[$(date +%H:%M:%S)] Uploading timedecay RAPM to Supabase..."
run $PYTHON_PATH "$SCRIPTS_DIR/upload_timedecay_rapm.py"

log ""
log "[$(date +%H:%M:%S)] Running RAPM analysis (13 years, rubberband)..."
run $PYTHON_PATH "$SCRIPTS_DIR/03_run_rapm_analysis.py" "$S13" "$SEASON_YY" ALL --rubberband

log ""
log "[$(date +%H:%M:%S)] Rebuilding gross DECOMP compatibility bundles (Net DECOMP migration pending)..."
DECOMP_WINDOWS="$SEASON_YEAR"
for back in 1 2 3 4; do
    DECOMP_WINDOWS="$DECOMP_WINDOWS,$(( SEASON_YEAR - back ))-$SEASON_YEAR"
done
run $PYTHON_PATH "$SCRIPTS_DIR/run_decomp_rolling.py" \
    --windows "$DECOMP_WINDOWS" \
    --force-components ALL \
    --workers 4 \
    --rapm-workers 4 \
    --cores-per-rapm 2

log ""
log "[$(date +%H:%M:%S)] Rebuilding legacy Alt3 EFG-value team audit bundle..."
run $PYTHON_PATH "$SCRIPTS_DIR/build_team_alt3_efg_value_weighted_factors.py" "$S3" "$SEASON_YY" ALL --alpha 25

log ""
log "[$(date +%H:%M:%S)] Uploading six-factor RAPM to Supabase..."
run $PYTHON_PATH "$SCRIPTS_DIR/upload_six_factor.py"

log ""
log "[$(date +%H:%M:%S)] Uploading WOWY team-player presence to Supabase..."
run $PYTHON_PATH "$SCRIPTS_DIR/upload_wowy_team_player_presence.py" \
    --start-year "$(( SEASON_YEAR - 4 ))" \
    --end-year "$SEASON_YEAR" \
    --season-types ALL \
    --upload

# 5. Standalone metric runs (time decay 700)
log ""
log "[$(date +%H:%M:%S)] Running standalone metrics (6-year TD 700)..."
run $PYTHON_PATH "$SCRIPTS_DIR/rapm.py" RIM_FREQ "$S6" "$SEASON_YY" ALL --timedecay --half-life 700
run $PYTHON_PATH "$SCRIPTS_DIR/rapm.py" RIM_FG_PCT "$S6" "$SEASON_YY" ALL --timedecay --half-life 700
run $PYTHON_PATH "$SCRIPTS_DIR/rapm.py" MIDRANGE_FREQ "$S6" "$SEASON_YY" ALL --timedecay --half-life 700

log ""
log "[$(date +%H:%M:%S)] Running standalone metrics (3-year TD 700, ShotQuality-dependent)..."
run $PYTHON_PATH "$SCRIPTS_DIR/rapm.py" TRANSITION_FREQ "$S3" "$SEASON_YY" ALL --timedecay --half-life 700
run $PYTHON_PATH "$SCRIPTS_DIR/rapm.py" TRANSITION_RIM "$S3" "$SEASON_YY" ALL --timedecay --half-life 700
run $PYTHON_PATH "$SCRIPTS_DIR/rapm.py" INITIAL_EV "$S3" "$SEASON_YY" ALL --timedecay --half-life 700

# 6. Copy standalone TD results to master_results
log ""
log "[$(date +%H:%M:%S)] Copying standalone TD results to master_results..."
MASTER_DIR="$PIPELINE_DIR/master_results"
TD_DIR="$PIPELINE_DIR/results/td"
mkdir -p "$MASTER_DIR"

for f in \
    "rimfreq_${W6}_all_td700_results.csv" \
    "rimfgpct_${W6}_all_td700_results.csv" \
    "midrangefreq_${W6}_all_td700_results.csv"; do
    if [ -f "$TD_DIR/$f" ]; then
        cp "$TD_DIR/$f" "$MASTER_DIR/$f"
        log "  Copied $f to master_results/"
    else
        log "  WARNING: $f not found in $TD_DIR"
    fi
done

copy_publish_artifact \
    "$TD_DIR/transition_freq_${W3}_all_td700_results.csv" \
    "transitionfreq_${W3}_all_td700_results.csv" \
    "$TD_DIR/transitionfreq_${W3}_all_td700_results.csv"

copy_publish_artifact \
    "$TD_DIR/transition_rim_${W3}_all_td700_results.csv" \
    "transitionrim_${W3}_all_td700_results.csv" \
    "$TD_DIR/transitionrim_${W3}_all_td700_results.csv"

copy_publish_artifact \
    "$TD_DIR/initial_ev_${W3}_all_td700_results.csv" \
    "initialev_${W3}_all_td700_results.csv" \
    "$TD_DIR/initialev_${W3}_all_td700_results.csv"

SPECIAL_RAPM_FILE="special_rapm_${W3}_all_results.csv"
if [ -f "$PIPELINE_DIR/results/$SPECIAL_RAPM_FILE" ]; then
    cp "$PIPELINE_DIR/results/$SPECIAL_RAPM_FILE" "$MASTER_DIR/"
    log "  Copied $SPECIAL_RAPM_FILE to master_results/"
else
    log "  WARNING: $SPECIAL_RAPM_FILE not found in $PIPELINE_DIR/results"
fi

# 7. TS Decomposition components (3-year TD 700, ShotQuality-dependent)
log ""
log "[$(date +%H:%M:%S)] Running TS decomposition components ($W3 TD 700)..."
run $PYTHON_PATH "$SCRIPTS_DIR/rapm.py" TS "$S3" "$SEASON_YY" ALL --timedecay --half-life 700
run $PYTHON_PATH "$SCRIPTS_DIR/rapm.py" SQ_POSS "$S3" "$SEASON_YY" ALL --timedecay --half-life 700
run $PYTHON_PATH "$SCRIPTS_DIR/rapm.py" FT_PREMIUM "$S3" "$SEASON_YY" ALL --timedecay --half-life 700
run $PYTHON_PATH "$SCRIPTS_DIR/rapm.py" CONTEST "$S3" "$SEASON_YY" ALL --timedecay --half-life 700

# 8. Run WLS regression decomposition and upload
log ""
log "[$(date +%H:%M:%S)] Running TS decomposition regression (td700)..."
run $PYTHON_PATH "$SCRIPTS_DIR/ts_decomp_regression.py" "$S3" "$SEASON_YY" ALL --timedecay
run $PYTHON_PATH "$SCRIPTS_DIR/upload_v3_sq_rapm.py"

# 8b. Copy TS decomposition factors to master_results
TS_DECOMP_DIR="$PIPELINE_DIR/results/ts_decomp_${W3}_all_td700"
TS_DECOMP_FILE="ts_decomp_factors_${W3}_all_td700.csv"
if [ -f "$TS_DECOMP_DIR/$TS_DECOMP_FILE" ]; then
    cp "$TS_DECOMP_DIR/$TS_DECOMP_FILE" "$MASTER_DIR/"
    log "  Copied $TS_DECOMP_FILE to master_results/"
else
    log "  WARNING: ts_decomp_factors not found"
fi

# 9. Update downstream CSV exports and current metric boards
#    These producers resolve paths from their own location, but still expect to
#    be run from the pbp_rapm root.
log ""
log "[$(date +%H:%M:%S)] Updating downstream CSV exports..."
cd "$PROJECT_DIR"
run $PYTHON_PATH "$PROJECT_DIR/update_2026_josh_rapm.py" --year "$SEASON_YEAR"
run $PYTHON_PATH "$PROJECT_DIR/update_2026_purerapm.py" --year "$SEASON_YEAR"
run $PYTHON_PATH "$PROJECT_DIR/update_2026_purerapm_peaks.py"
run $PYTHON_PATH "$PROJECT_DIR/update_2026_scaledoutput.py" --year "$SEASON_YEAR"

# 10. Sync master_results to rapms repo and push
log ""
log "[$(date +%H:%M:%S)] Syncing master_results to rapms repo..."
RAPMS_MASTER="$RAPMS_DIR/master_results"

if [ -d "$RAPMS_MASTER" ]; then
    for f in \
        "weighted_factors_${W1}_all_se.csv" \
        "weighted_factors_${W2}_all_se.csv" \
        "weighted_factors_${W3}_all_se.csv" \
        "weighted_factors_${W4}_all_se.csv" \
        "weighted_factors_${W5}_all_se.csv" \
        "weighted_factors_${W6}_all_se.csv" \
        "weighted_factors_${W6}_all_se_td700.csv" \
        "weighted_factors_${W13}_all_rb.csv" \
        "weighted_factors_alt3_${W1}_all_se.csv" \
        "weighted_factors_alt3_${W2}_all_se.csv" \
        "weighted_factors_alt3_${W3}_all_se.csv" \
        "weighted_factors_alt3_${W4}_all_se.csv" \
        "weighted_factors_alt3_${W5}_all_se.csv" \
        "weighted_factors_alt3_${W6}_all_se.csv" \
        "weighted_factors_alt3_${W6}_all_se_td700.csv" \
        "weighted_factors_alt3_${W6}_all_rb_se_td700.csv" \
        "weighted_factors_alt3_${W13}_all_rb.csv" \
        "rimfreq_${W6}_all_td700_results.csv" \
        "rimfgpct_${W6}_all_td700_results.csv" \
        "midrangefreq_${W6}_all_td700_results.csv" \
        "transitionfreq_${W3}_all_td700_results.csv" \
        "transitionrim_${W3}_all_td700_results.csv" \
        "initialev_${W3}_all_td700_results.csv" \
        "special_rapm_${W3}_all_results.csv" \
        "ts_decomp_factors_${W3}_all_td700.csv"; do
        if [ -f "$MASTER_DIR/$f" ]; then
            cp "$MASTER_DIR/$f" "$RAPMS_MASTER/$f"
            log "  Synced $f"
        else
            log "  WARNING: $f not found in master_results"
        fi
    done

    for w in "$W1" "$W2" "$W3" "$W4" "$W5"; do
        for ext in csv parquet; do
            f="weighted_factors_decomp_${w}_all_rb_se_a2000_4000.${ext}"
            if [ -f "$MASTER_DIR/$f" ]; then
                cp "$MASTER_DIR/$f" "$RAPMS_MASTER/$f"
                log "  Synced $f"
            else
                log "  WARNING: $f not found in master_results"
            fi
        done
    done

    for ext in csv parquet; do
        f="team_weighted_factors_alt3_efg_value_${W3}_all_a25.${ext}"
        if [ -f "$MASTER_DIR/$f" ]; then
            cp "$MASTER_DIR/$f" "$RAPMS_MASTER/$f"
            log "  Synced $f"
        else
            log "  WARNING: $f not found in master_results"
        fi
    done
else
    log "  WARNING: rapms master_results not found at $RAPMS_MASTER; sync skipped"
    log "           set NBA_RAPM_RAPMS_DIR to the rapms checkout to enable it"
fi

log "[$(date +%H:%M:%S)] Skipping legacy Alt3-to-scposs mapper pending its 97-column DECOMP schema audit."

log ""
log "[$(date +%H:%M:%S)] Updating current metrics and peak leaderboards..."
if [ -n "$CURRENT_METRICS_DIR" ] && [ -f "$CURRENT_METRICS_DIR/$CURRENT_METRICS_SCRIPT" ]; then
    cd "$CURRENT_METRICS_DIR"
    run env NBA_PIPELINE_CSV_DIR="$CSVS_DIR" $PYTHON_PATH "$CURRENT_METRICS_DIR/$CURRENT_METRICS_SCRIPT" --no-sync
    cd "$PROJECT_DIR"
elif [ -z "$CURRENT_METRICS_DIR" ]; then
    log "  WARNING: NBA_RAPM_CURRENT_METRICS_DIR unset; current metrics step skipped"
else
    log "  WARNING: $CURRENT_METRICS_DIR/$CURRENT_METRICS_SCRIPT not found; current metrics step skipped"
    log "           set NBA_RAPM_CURRENT_METRICS_SCRIPT if the collector is named for another season"
fi

if [ "$PUSH_DOWNSTREAM" != "1" ]; then
    log "[$(date +%H:%M:%S)] NBA_RAPM_PUSH=$PUSH_DOWNSTREAM; skipping csvs/rapms commits and pushes"
elif [ ! -d "$CSVS_DIR/.git" ]; then
    log "  WARNING: csvs repo not found at $CSVS_DIR; push skipped"
    log "           set NBA_RAPM_CSVS_DIR to the csvs checkout to enable it"
else
    log "[$(date +%H:%M:%S)] Pushing csvs repo..."
    cd "$CSVS_DIR"
    CURRENT_CSVS_BRANCH="$(git branch --show-current)"
    if [ "$CURRENT_CSVS_BRANCH" != "$CSVS_BRANCH" ]; then
        log "  ERROR: csvs repo is on $CURRENT_CSVS_BRANCH, expected $CSVS_BRANCH"
        exit 1
    fi
    git add PureRAPM.csv PureRAPMPeaks.csv SCALEDOUTPUT_SMALLER.csv current_comp.csv structuredtest.csv DARKO.csv lebron.csv players_by_player.csv autocomplete_map.csv josh_rapm scposs
    git diff --cached --quiet || git commit -m "Daily RAPM csvs update $(date +%Y-%m-%d)"
    git push origin "$CSVS_BRANCH"

    if [ -d "$RAPMS_DIR/.git" ]; then
        log "[$(date +%H:%M:%S)] Pushing rapms repo..."
        cd "$RAPMS_DIR"
        git add -A
        git diff --cached --quiet || git commit -m "Daily RAPM update $(date +%Y-%m-%d)"
        git push origin main
    else
        log "  WARNING: rapms repo not found at $RAPMS_DIR; push skipped"
    fi
fi

cd "$PIPELINE_DIR"

log ""
log "========================================"
log "Daily update completed at $(date)"
log "========================================"

# Cleanup old logs (keep last 30 days)
find "$PIPELINE_DIR/logs" -name "daily_update_*.log" -mtime +30 -delete 2>/dev/null || true
