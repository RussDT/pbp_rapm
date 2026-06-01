#!/bin/bash
# Daily NBA RAPM Update Script
# Runs at 6 AM via launchd

set -e

# Configuration
PIPELINE_DIR="/Users/russellthomas/Docs/pbp_rapm/nba_pipeline"
LOG_FILE="$PIPELINE_DIR/logs/daily_update_$(date +%Y%m%d).log"

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

# 1. Fetch latest 2026 data
log ""
log "[$(date +%H:%M:%S)] Fetching latest regular-season PBP data..."
run $PYTHON_PATH "$SCRIPTS_DIR/01_fetch_pbp_data.py" 26

log ""
log "[$(date +%H:%M:%S)] Fetching latest playoff PBP data..."
run $PYTHON_PATH "$SCRIPTS_DIR/01_fetch_pbp_data.py" 26 PS

# 2. Enrich with ShotQuality data (adds initial_ev, transition, etc.)
# The enrichment script processes both NBA26.parquet and NBA26_PS.parquet for year 26.
log ""
log "[$(date +%H:%M:%S)] Enriching PBP with ShotQuality data..."
run $PYTHON_PATH "$SCRIPTS_DIR/01b_enrich_pbp_shotquality.py" 26

# 3. Process RAPM for current season
log ""
log "[$(date +%H:%M:%S)] Processing regular-season RAPM data..."
run $PYTHON_PATH "$SCRIPTS_DIR/02_process_rapm.py" "$RAW_DATA_DIR/NBA26.parquet"

if [ -f "$RAW_DATA_DIR/NBA26_PS.parquet" ]; then
    log ""
    log "[$(date +%H:%M:%S)] Processing playoff RAPM data..."
    run $PYTHON_PATH "$SCRIPTS_DIR/02_process_rapm.py" "$RAW_DATA_DIR/NBA26_PS.parquet"
else
    log "  WARNING: $RAW_DATA_DIR/NBA26_PS.parquet not found; playoff processing skipped"
fi

# 4. Run RAPM analyses (most common year ranges)
log ""
log "[$(date +%H:%M:%S)] Running RAPM analysis (1 year)..."
run $PYTHON_PATH "$SCRIPTS_DIR/03_run_rapm_analysis.py" 26 26 ALL

log ""
log "[$(date +%H:%M:%S)] Running RAPM analysis (2 years)..."
run $PYTHON_PATH "$SCRIPTS_DIR/03_run_rapm_analysis.py" 25 26 ALL

log ""
log "[$(date +%H:%M:%S)] Running RAPM analysis (3 years)..."
run $PYTHON_PATH "$SCRIPTS_DIR/03_run_rapm_analysis.py" 24 26 ALL

log ""
log "[$(date +%H:%M:%S)] Running RAPM analysis (4 years)..."
run $PYTHON_PATH "$SCRIPTS_DIR/03_run_rapm_analysis.py" 23 26 ALL

log ""
log "[$(date +%H:%M:%S)] Running RAPM analysis (5 years)..."
run $PYTHON_PATH "$SCRIPTS_DIR/03_run_rapm_analysis.py" 22 26 ALL

log ""
log "[$(date +%H:%M:%S)] Running RAPM analysis (6 years)..."
run $PYTHON_PATH "$SCRIPTS_DIR/03_run_rapm_analysis.py" 21 26 ALL

# 4. Additional weighted-factors runs
log ""
log "[$(date +%H:%M:%S)] Running RAPM analysis (6 years, time decay 700)..."
run $PYTHON_PATH "$SCRIPTS_DIR/03_run_rapm_analysis.py" 21 26 ALL --timedecay --half-life 700

log ""
log "[$(date +%H:%M:%S)] Running RAPM analysis (6 years, time decay 700 + rubberband)..."
run $PYTHON_PATH "$SCRIPTS_DIR/03_run_rapm_analysis.py" 21 26 ALL --timedecay --half-life 700 --rubberband

log ""
log "[$(date +%H:%M:%S)] Uploading timedecay RAPM to Supabase..."
run $PYTHON_PATH "$SCRIPTS_DIR/upload_timedecay_rapm.py"

log ""
log "[$(date +%H:%M:%S)] Running RAPM analysis (13 years, rubberband)..."
run $PYTHON_PATH "$SCRIPTS_DIR/03_run_rapm_analysis.py" 14 26 ALL --rubberband

log ""
log "[$(date +%H:%M:%S)] Rebuilding active Alt3 EFG-value player bundles for 2026 rolling windows..."
run $PYTHON_PATH "$SCRIPTS_DIR/run_alt3_efg_value_rolling.py" \
    --intersect-years 2026 \
    --force-base \
    --force-components ALL \
    --workers 4 \
    --rapm-workers 4 \
    --cores-per-rapm 2

log ""
log "[$(date +%H:%M:%S)] Rebuilding active Alt3 EFG-value team bundle..."
run $PYTHON_PATH "$SCRIPTS_DIR/build_team_alt3_efg_value_weighted_factors.py" 24 26 ALL --alpha 25

log ""
log "[$(date +%H:%M:%S)] Uploading six-factor RAPM to Supabase..."
run $PYTHON_PATH "$SCRIPTS_DIR/upload_six_factor.py"

# 5. Standalone metric runs (time decay 700)
log ""
log "[$(date +%H:%M:%S)] Running standalone metrics (6-year TD 700)..."
run $PYTHON_PATH "$SCRIPTS_DIR/rapm.py" RIM_FREQ 21 26 ALL --timedecay --half-life 700
run $PYTHON_PATH "$SCRIPTS_DIR/rapm.py" RIM_FG_PCT 21 26 ALL --timedecay --half-life 700
run $PYTHON_PATH "$SCRIPTS_DIR/rapm.py" MIDRANGE_FREQ 21 26 ALL --timedecay --half-life 700

log ""
log "[$(date +%H:%M:%S)] Running standalone metrics (3-year TD 700, ShotQuality-dependent)..."
run $PYTHON_PATH "$SCRIPTS_DIR/rapm.py" TRANSITION_FREQ 24 26 ALL --timedecay --half-life 700
run $PYTHON_PATH "$SCRIPTS_DIR/rapm.py" TRANSITION_RIM 24 26 ALL --timedecay --half-life 700
run $PYTHON_PATH "$SCRIPTS_DIR/rapm.py" INITIAL_EV 24 26 ALL --timedecay --half-life 700

# 6. Copy standalone TD results to master_results
log ""
log "[$(date +%H:%M:%S)] Copying standalone TD results to master_results..."
MASTER_DIR="$PIPELINE_DIR/master_results"
TD_DIR="$PIPELINE_DIR/results/td"
mkdir -p "$MASTER_DIR"

for f in \
    "rimfreq_21_26_all_td700_results.csv" \
    "rimfgpct_21_26_all_td700_results.csv" \
    "midrangefreq_21_26_all_td700_results.csv"; do
    if [ -f "$TD_DIR/$f" ]; then
        cp "$TD_DIR/$f" "$MASTER_DIR/$f"
        log "  Copied $f to master_results/"
    else
        log "  WARNING: $f not found in $TD_DIR"
    fi
done

copy_publish_artifact \
    "$TD_DIR/transition_freq_24_26_all_td700_results.csv" \
    "transitionfreq_24_26_all_td700_results.csv" \
    "$TD_DIR/transitionfreq_24_26_all_td700_results.csv"

copy_publish_artifact \
    "$TD_DIR/transition_rim_24_26_all_td700_results.csv" \
    "transitionrim_24_26_all_td700_results.csv" \
    "$TD_DIR/transitionrim_24_26_all_td700_results.csv"

copy_publish_artifact \
    "$TD_DIR/initial_ev_24_26_all_td700_results.csv" \
    "initialev_24_26_all_td700_results.csv" \
    "$TD_DIR/initialev_24_26_all_td700_results.csv"

SPECIAL_RAPM_FILE="special_rapm_24_26_all_results.csv"
if [ -f "$PIPELINE_DIR/results/$SPECIAL_RAPM_FILE" ]; then
    cp "$PIPELINE_DIR/results/$SPECIAL_RAPM_FILE" "$MASTER_DIR/"
    log "  Copied $SPECIAL_RAPM_FILE to master_results/"
else
    log "  WARNING: $SPECIAL_RAPM_FILE not found in $PIPELINE_DIR/results"
fi

# 7. TS Decomposition components (3-year TD 700, ShotQuality-dependent)
log ""
log "[$(date +%H:%M:%S)] Running TS decomposition components (24-26 TD 700)..."
run $PYTHON_PATH "$SCRIPTS_DIR/rapm.py" TS 24 26 ALL --timedecay --half-life 700
run $PYTHON_PATH "$SCRIPTS_DIR/rapm.py" SQ_POSS 24 26 ALL --timedecay --half-life 700
run $PYTHON_PATH "$SCRIPTS_DIR/rapm.py" FT_PREMIUM 24 26 ALL --timedecay --half-life 700
run $PYTHON_PATH "$SCRIPTS_DIR/rapm.py" CONTEST 24 26 ALL --timedecay --half-life 700

# 8. Run WLS regression decomposition and upload
log ""
log "[$(date +%H:%M:%S)] Running TS decomposition regression (td700)..."
run $PYTHON_PATH "$SCRIPTS_DIR/ts_decomp_regression.py" 24 26 ALL --timedecay
run $PYTHON_PATH "$SCRIPTS_DIR/upload_v3_sq_rapm.py"

# 8b. Copy TS decomposition factors to master_results
TS_DECOMP_DIR="$PIPELINE_DIR/results/ts_decomp_24_26_all_td700"
if [ -f "$TS_DECOMP_DIR/ts_decomp_factors_24_26_all_td700.csv" ]; then
    cp "$TS_DECOMP_DIR/ts_decomp_factors_24_26_all_td700.csv" "$MASTER_DIR/"
    log "  Copied ts_decomp_factors_24_26_all_td700.csv to master_results/"
else
    log "  WARNING: ts_decomp_factors not found"
fi

# 9. Update downstream CSV exports (josh_rapm, PureRAPM, scposs, SCALEDOUTPUT)
#    These scripts use relative paths — must run from pbp_rapm root
log ""
log "[$(date +%H:%M:%S)] Updating downstream CSV exports..."
PROJECT_DIR="$(dirname "$PIPELINE_DIR")"
cd "$PROJECT_DIR"
run $PYTHON_PATH "$PROJECT_DIR/update_2026_josh_rapm.py"
run $PYTHON_PATH "$PROJECT_DIR/update_2026_purerapm.py"
run $PYTHON_PATH "$PROJECT_DIR/update_2026_scaledoutput.py"
run $PYTHON_PATH "$PROJECT_DIR/update_2026_scposs.py"

# 10. Sync master_results to rapms repo and push
log ""
log "[$(date +%H:%M:%S)] Syncing master_results to rapms repo..."
RAPMS_DIR="/Users/russellthomas/Docs/rapms"
RAPMS_MASTER="$RAPMS_DIR/master_results"

for f in \
    "weighted_factors_26_all.csv" \
    "weighted_factors_25_26_all.csv" \
    "weighted_factors_24_26_all.csv" \
    "weighted_factors_23_26_all.csv" \
    "weighted_factors_22_26_all.csv" \
    "weighted_factors_21_26_all.csv" \
    "weighted_factors_21_26_all_td700.csv" \
    "weighted_factors_14_26_all_rb.csv" \
    "weighted_factors_alt3_26_all.csv" \
    "weighted_factors_alt3_25_26_all.csv" \
    "weighted_factors_alt3_24_26_all.csv" \
    "weighted_factors_alt3_23_26_all.csv" \
    "weighted_factors_alt3_22_26_all.csv" \
    "weighted_factors_alt3_21_26_all.csv" \
    "weighted_factors_alt3_21_26_all_td700.csv" \
    "weighted_factors_alt3_21_26_all_rb_td700.csv" \
    "weighted_factors_alt3_14_26_all_rb.csv" \
    "rimfreq_21_26_all_td700_results.csv" \
    "rimfgpct_21_26_all_td700_results.csv" \
    "midrangefreq_21_26_all_td700_results.csv" \
    "transitionfreq_24_26_all_td700_results.csv" \
    "transitionrim_24_26_all_td700_results.csv" \
    "initialev_24_26_all_td700_results.csv" \
    "special_rapm_24_26_all_results.csv" \
    "ts_decomp_factors_24_26_all_td700.csv"; do
    if [ -f "$MASTER_DIR/$f" ]; then
        cp "$MASTER_DIR/$f" "$RAPMS_MASTER/$f"
        log "  Synced $f"
    else
        log "  WARNING: $f not found in master_results"
    fi
done

for f in \
    "weighted_factors_alt3_efg_value_26_all_rb_se_a2000_4000.csv" \
    "weighted_factors_alt3_efg_value_26_all_rb_se_a2000_4000.parquet" \
    "weighted_factors_alt3_efg_value_25_26_all_rb_se_a2000_4000.csv" \
    "weighted_factors_alt3_efg_value_25_26_all_rb_se_a2000_4000.parquet" \
    "weighted_factors_alt3_efg_value_24_26_all_rb_se_a2000_4000.csv" \
    "weighted_factors_alt3_efg_value_24_26_all_rb_se_a2000_4000.parquet" \
    "weighted_factors_alt3_efg_value_23_26_all_rb_se_a2000_4000.csv" \
    "weighted_factors_alt3_efg_value_23_26_all_rb_se_a2000_4000.parquet" \
    "weighted_factors_alt3_efg_value_22_26_all_rb_se_a2000_4000.csv" \
    "weighted_factors_alt3_efg_value_22_26_all_rb_se_a2000_4000.parquet" \
    "team_weighted_factors_alt3_efg_value_24_26_all_a25.csv" \
    "team_weighted_factors_alt3_efg_value_24_26_all_a25.parquet"; do
    if [ -f "$MASTER_DIR/$f" ]; then
        cp "$MASTER_DIR/$f" "$RAPMS_MASTER/$f"
        log "  Synced $f"
    else
        log "  WARNING: $f not found in master_results"
    fi
done

log "[$(date +%H:%M:%S)] Pushing rapms repo..."
cd "$RAPMS_DIR"
git add -A
git diff --cached --quiet || git commit -m "Daily RAPM update $(date +%Y-%m-%d)"
git push origin main
cd "$PIPELINE_DIR"

log ""
log "========================================"
log "Daily update completed at $(date)"
log "========================================"

# Cleanup old logs (keep last 30 days)
find "$PIPELINE_DIR/logs" -name "daily_update_*.log" -mtime +30 -delete 2>/dev/null || true
