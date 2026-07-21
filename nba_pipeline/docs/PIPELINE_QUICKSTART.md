# NBA RAPM Pipeline - Quick Start Guide

## Directory Structure

```
nba_pipeline/
├── raw_data/          # Input: NBA PBP parquet files (NBA23.parquet, NBA23_PS.parquet, etc.)
├── processed/         # Output: Processed RAPM parquet files
├── results/           # Output: RAPM regression results (CSV)
├── scripts/
│   ├── 01_fetch_pbp_data.py # Fetches raw PBP data from NBA API
│   ├── 02_process_rapm.py   # Processes raw PBP into RAPM format
│   └── rapm.py              # Runs ridge regression
└── docs/
```

---

## Step 0: Fetching / Re-fetching Modern Seasons (2014+)

`01_fetch_pbp_data.py` pulls PlayByPlayV3 + GameRotation and writes
`NBA{YY}.parquet` with reconstructed 5-man lineups. Two operational facts:

- **stats.nba.com requires a browser TLS fingerprint.** `01_fetch_pbp_data.py`
  imports `_nba_http_curlcffi.py`, which routes nba_api through `curl_cffi`
  (Chrome impersonation). Without it, plain `requests` is silently stalled by
  Akamai (looks like an IP ban). No-op fallback if curl_cffi is missing.
- **`gamerotation` throttles per-IP and cumulatively.** A single IP decays to
  ~0% success under sustained load, so bulk historical refetch needs the
  rotating-IP proxy (`--proxy` / fetch default). Failures are mostly server-side
  cold-cache 500s that recover over time, so multiple passes converge.

For a bulk refill use the resumable driver, which wraps the fetcher in
update-mode chunks (checkpoints each chunk, skips already-fetched games,
survives restarts, detects stalls):

```bash
cd nba_pipeline/scripts
# Full modern refill, proxy on (rotating IPs), 3 workers:
python fetch_modern_seasons.py --start 14 --end 25 --season-types RS PS --proxy --workers 3
```

Env knobs on `01_fetch_pbp_data.py`: `NBA_INTER_GAME_DELAY` (seconds between
games, for gentle serial pacing), `NBA_ROT_ATTEMPTS` (gamerotation retries; lower
= fail-fast + rely on resumable re-passes), `NBA_IMPERSONATE` (curl_cffi target,
default `chrome120`). Single-season update (daily path):
`python 01_fetch_pbp_data.py 26` and `python 01_fetch_pbp_data.py 26 PS`.

---

## Step 1: Process Raw Data

### Historical ESPN Seasons (2003-2014 Primary Path)
```bash
cd nba_pipeline/scripts

# Build a historical raw parquet from SportsDataverse / hoopR ESPN season files
python 01_fetch_historical_pbp_espn.py 2003

# Optional playoff run
python 01_fetch_historical_pbp_espn.py 2003 PS

# Then process it through the standard RAPM pipeline
python 02_process_rapm.py ../raw_data/NBA03.parquet
```

Notes:
- `01_fetch_historical_pbp_espn.py` reconstructs lineups from ESPN starter flags plus substitution rows.
- The loader maps exact-name matches into repo `nba_id` values from `autocomplete_map.csv`.
- Unresolved historical players are pushed into a synthetic high-ID namespace so they do not collide with modern NBA IDs.
- The current recommended source plan for pre-2014 seasons is documented in [Historical Pre-2014 Ingest Plan](./HISTORICAL_PRE2014_INGEST_PLAN.md).

### Single File
```bash
cd nba_pipeline/scripts
python 02_process_rapm.py ../raw_data/NBA26.parquet
```

### All Files (Batch)
Edit `process_historical_rapm.py` to set seasons, then:
```bash
python process_historical_rapm.py
```

**Output files created in `processed/` (parquet format for space efficiency):**
- `RAPM26.parquet`, `LA_RAPM26.parquet` - Point differential (with/without luck adjustment)
- `TS26.parquet` - True shooting
- `TOV26.parquet` - Turnovers
- `REB26.parquet` - Offensive rebounds
- `RIM_FREQ26.parquet` - Rim attempt frequency
- `RIM_FG_PCT26.parquet` - Rim FG%
- `ASSIST_POINTS26.parquet` - Assisted FG points per possession
- `RIM_ASSIST26.parquet` - Assisted rim makes per possession
- `THREE_FREQ26.parquet` - 3-point attempt frequency
- `THREE_FG_PCT26.parquet` - 3-point percentage
- `PLAYTYPE_TS_MIX26.parquet` - Descriptor-based expected scoring mix per FGA
- `PLAYTYPE_PROXY_PTS26.parquet` - Synergy-proxy category value per FGA using season-relative actual points
- `MIDRANGE_FREQ26.parquet` - Midrange attempt frequency
- `MIDRANGE_FG_PCT26.parquet` - Midrange FG%

Add `_PS` suffix for playoffs (e.g., `NBA26_PS.parquet` → `RAPM26_PS.parquet`)

---

## Step 2: Run RAPM Analysis

### Command Format
```bash
cd nba_pipeline/scripts
python rapm.py <TYPE> <START_YEAR> <END_YEAR> <SEASON_TYPE> [--pure]
```

### Parameters
| Param | Values | Description |
|-------|--------|-------------|
| TYPE | `RAPM`, `LA_RAPM`, `TS`, `TOV`, `REB`, `RIM_FREQ`, `RIM_FG_PCT`, `ASSIST_POINTS`, `RIM_ASSIST`, `THREE_FREQ`, `THREE_FG_PCT`, `PLAYTYPE_TS_MIX`, `PLAYTYPE_PROXY_PTS`, `MIDRANGE_FREQ`, `MIDRANGE_FG_PCT` | Which metric |
| START_YEAR | 23, 24, 25, 26... | Two-digit start year |
| END_YEAR | 23, 24, 25, 26... | Two-digit end year (inclusive) |
| SEASON_TYPE | `RS`, `PS`, `ALL` | Regular season, playoffs, or both |
| --pure | flag | Use raw points (no luck adjustment) |
| --cv-alpha | flag | Cross-validate offense/defense alphas separately |
| --season-effects | flag | Add season-phase fixed effects (earliest observed season-phase omitted as reference, preferring RS over PS within a season) |
| --age-dummies | flag | Add signed age-bin fixed effects (18-38, 38+ -> 38); `03_run_rapm_analysis.py` applies it to every weighted-factor metric solve |
| --age-curve | flag | RAPM only: shrink toward a learned age-mean prior |
| --strength-splits | flag | RAPM only: add DARKO strong/not-strong opponent interaction columns for each player |
| --strength-legacy-scale | flag | Use with `--strength-splits`: center/scale `overall_off` and `overall_def` to the legacy strong/not-strong target standard deviations |
| --strength-sensitivity | flag | RAPM only: estimate continuous season-standardized DARKO opponent-strength slopes for each player |
| --strength-sensitivity-alpha-mult | number | Ridge penalty multiplier for strength-sensitivity slopes relative to base player coefficients (default `4.0`) |
| --darko-history | path | Override the DARKO history CSV used by `--strength-splits` |

### Examples
```bash
# Current season only
python rapm.py RAPM 26 26 RS

# Multi-year with playoffs
python rapm.py RAPM 23 26 ALL

# Cross-century historical window; `97 0` expands to 1997-2000 and writes `97_00`
python rapm.py RAPM 97 0 ALL --pure --rubberband --season-effects

# Luck-adjusted RAPM
python rapm.py LA_RAPM 23 26 ALL

# Rim metrics
python rapm.py RIM_FREQ 23 26 ALL
python rapm.py RIM_FG_PCT 23 26 ALL
python rapm.py ASSIST_POINTS 23 26 ALL
python rapm.py RIM_ASSIST 23 26 ALL
python rapm.py THREE_FREQ 23 26 ALL
python rapm.py THREE_FG_PCT 23 26 ALL
python rapm.py MIDRANGE_FREQ 23 26 ALL
python rapm.py MIDRANGE_FG_PCT 23 26 ALL
python rapm.py PLAYTYPE_TS_MIX 24 26 ALL
python rapm.py PLAYTYPE_PROXY_PTS 24 26 ALL

# True shooting
python rapm.py TS 24 26 RS

# Pure RAPM (no FT/3PT luck adjustment)
python rapm.py RAPM 23 26 ALL --pure

# Cross-validate offense/defense alphas separately
python rapm.py RAPM 26 26 RS --pure --cv-alpha

# Cross-validate and publish the finished result CSV to rapms
python rapm.py SPECIAL_RAPM 24 26 ALL --cv-alpha --publish-to-rapms

# Multi-year style metric with season fixed effects
python rapm.py MIDRANGE_FREQ 14 26 ALL --season-effects

# Multi-year metric solve with signed age fixed effects
python rapm.py RAPM 14 26 ALL --pure --rubberband --season-effects --age-dummies
python rapm.py TS 14 26 ALL --rubberband --season-effects --age-dummies

# Build a combined rim/midrange/three frequency + FG% RAPM export
python build_master_frequency_rapm.py 23 26 ALL

# Age-curve RAPM prior mode
python rapm.py RAPM 14 26 ALL --rubberband --age-curve --pure

# DARKO strong/not-strong opponent split RAPM
python rapm.py RAPM 14 26 ALL --strength-splits

# DARKO split plus legacy target standard-deviation scaling
python rapm.py RAPM 14 26 ALL --strength-splits --strength-legacy-scale

# Continuous DARKO opponent-strength sensitivity
python rapm.py RAPM 14 26 ALL --strength-sensitivity
```

**Output**: Results saved to `results/` (e.g., `rapm_23_26_all_results.csv`)

`rapm.py` now backfills missing `player_name` values from the shared `autocomplete_map.csv`, the sibling `2026_NBA_PIPELINE` identity outputs, and raw PBP event columns before writing results. This keeps active fringe players from leaking into published CSVs as `ID_<player_id>`.

With `--age-curve`, the RAPM result filename gets `_age` (for example `rapm_14_26_all_pure_rb_age_results.csv`).

With `--age-dummies`, the RAPM result filename gets `_agefe` and `rapm.py` also writes a companion `*_age_effects.csv` file with separate offense, defense, and net age curves plus age-slot counts.

With `--strength-splits`, the RAPM result filename gets `_strength` and each player receives binary DARKO interaction columns: `off_vs_strong`, `off_vs_not_strong`, `def_vs_strong`, `def_vs_not_strong`, matching `facing_*` coefficients, `faced_*` counts, exposure percentages, `overall_off`, `overall_def`, and `overall_net_rapm`. By default it reads `/Users/russellthomas/Docs/2026_NBA_PIPELINE/databallr/darko/dpm_history.csv`; override that with `--darko-history`.

With `--strength-legacy-scale`, the filename gets `_strength_scaled`. The export keeps the unscaled strength columns and adds `scaled_overall_off`, `scaled_overall_def`, `scaled_off_vs_*`, `scaled_def_vs_*`, and `scaled_overall_net_rapm`; `net_rapm` is sorted from the scaled net. The target standard deviations are `1.48` for offense and `1.36` for defense, matching the legacy strong/not-strong script.

With `--strength-sensitivity`, the filename gets `_strengthsens`. The export estimates separate offensive and defensive continuous slopes against season-standardized DARKO lineup strength, then reports `net_strength_slope`, `net_vs_weak_1sd`, `net_vs_avg_strength`, `net_vs_strong_1sd`, `strong_minus_weak`, and `weak_minus_strong_drop`. Negative `strong_minus_weak` means the player projects worse as opponent quality rises. The default slope penalty is 4x the base player penalty; increase `--strength-sensitivity-alpha-mult` for a more conservative diagnostic.

With `--season-effects`, result filenames get `_se` and `rapm.py` also writes a companion `*_season_effects.csv` file that shows the estimated `(season, RS/PS)` coefficients relative to the omitted reference season-phase. In `ALL` runs, regular season and playoffs are estimated separately rather than sharing one season dummy.

With `--cv-alpha`, result filenames get `_cv` and `rapm.py` also writes a companion `*_alpha_cv.csv` file with per-fold losses and the selected offense/defense alpha path. The default grid is `300,500,1000,2000,3000,5000,7000,10000`, and you can override it with `--cv-alpha-grid`.

With `--publish-to-rapms`, `rapm.py` copies the main result CSV into local `master_results/`, syncs it to `/Users/russellthomas/Docs/rapms/master_results/`, and commits/pushes the downstream `rapms` repo if the published file changed. Use `--publish-name` if the downstream filename should differ from the solver output filename.

For the standalone ShotQuality exports in the daily pipeline, keep the solver filenames as the canonical source of truth:
- `transition_freq_24_26_all_td700_results.csv`
- `transition_rim_24_26_all_td700_results.csv`
- `initial_ev_24_26_all_td700_results.csv`

The publish step intentionally renames those onto the legacy downstream `rapms` filenames:
- `transitionfreq_24_26_all_td700_results.csv`
- `transitionrim_24_26_all_td700_results.csv`
- `initialev_24_26_all_td700_results.csv`

### Standard Weighted-Factors Run
```bash
python 03_run_rapm_analysis.py 14 26 ALL --rubberband
python 03_run_rapm_analysis.py 14 26 ALL --season-effects
python 03_run_rapm_analysis.py 14 26 ALL --rubberband --age-curve
python 03_run_rapm_analysis.py 14 26 ALL --rubberband --age-dummies
python 03_run_rapm_analysis.py 1 4 ALL --rapm-workers 8 --cores-per-rapm 4
python 03_run_rapm_analysis.py 97 0 ALL --rubberband --season-effects --publish-to-rapms
python 03_run_rapm_analysis.py 97 99 ALL --rubberband --season-effects --off-alpha 2000 --def-alpha 4000 --publish-to-rapms
python 03_run_rapm_analysis.py 21 26 ALL --publish-to-rapms
```

The season-effects run writes separate `_se` result files/folders. The age-curve run writes a separate folder and weighted-factors file with `_age` in the name. The age-dummies run writes separate `_agefe` metric results, per-metric `*_age_effects.csv` artifacts, and `weighted_factors_*_agefe.csv` outputs.
By default, `03_run_rapm_analysis.py` runs 4 RAPM subprocesses at a time and lets each `rapm.py` subprocess auto-select its load-worker count. Use `--rapm-workers` to change the number of concurrent metric solves and `--cores-per-rapm` to pass an explicit `rapm.py --cores` value into every subprocess.
Use `--off-alpha` and `--def-alpha` to pass fixed offense/defense ridge penalties into every metric solve. Non-default alpha weighted-factor outputs include an `a{off}_{def}` suffix, such as `weighted_factors_97_99_all_rb_se_a2000_4000.csv`.
Both `rapm.py` and `03_run_rapm_analysis.py` accept two-digit windows that cross 1999-2000. For example, `97 0` expands to 1997, 1998, 1999, and 2000, and result filenames use `97_00`.
Use `--publish-to-rapms` when a completed weighted-factors run should copy its standard and alt3 CSVs into `/Users/russellthomas/Docs/rapms/master_results/`, commit only those files, and push the downstream `rapms` repo if they changed.
The master weighted-factors runner also solves and exports seven auxiliary shot-profile RAPMs in both standard and alt3 CSVs: `RIM_FREQ`, `RIM_FG_PCT`, `THREE_FREQ`, `THREE_FG_PCT`, `MIDRANGE_FREQ`, `MIDRANGE_FG_PCT`, and `ASSIST_POINTS`. These columns are pass-through diagnostics (`o*` raw offense, `d*` sign-flipped so positive means good defense); they are not part of the six-factor or alt3 additive reconstruction.

Both `weighted_factors_alt3_efg_value_*` and `weighted_factors_alt3_*` are legacy/audit families. The active public output is `weighted_factors_decomp_*`, built by `scripts/run_decomp_rolling.py`; see [Public Eight-Component DECOMP](./PUBLIC_DECOMP.md).

The daily job rebuilds the active DECOMP rolling files (`26`, `25_26`, `24_26`, `23_26`, `22_26` with `all_rb_se_a2000_4000`) and syncs both CSV and parquet artifacts. The team-level Alt3 bundle is retained as an audit artifact.

Databallr WOWY RAPM mode uses `player_alt3_efg_factors` for the 1Y-5Y DECOMP values and `wowy_team_player_presence` for team membership. Rebuild the PBP-derived presence table from raw lineup columns and upload it to Supabase with:

```bash
python nba_pipeline/scripts/upload_wowy_team_player_presence.py \
  --start-year 2022 \
  --end-year 2026 \
  --season-types ALL \
  --upload
```

For non-rolling RAPM DECOMP era windows, use the explicit-window runner:

```bash
python nba_pipeline/scripts/run_alt3_efg_value_rolling.py \
  --windows 2000-2009,2010-2019,2020-2026,1997-2006,2017-2026 \
  --force-base \
  --force-components ALL \
  --publish-to-rapms
```

The current peak-metrics flow reads `2000s RAPM`, `2010s RAPM`, and `2020s RAPM` from the native DECOMP files in `/Users/russellthomas/Docs/rapms/master_results`: `weighted_factors_alt3_efg_value_00_09_all_rb_se_a2000_4000.csv`, `weighted_factors_alt3_efg_value_10_19_all_rb_se_a2000_4000.csv`, and `weighted_factors_alt3_efg_value_20_26_all_rb_se_a2000_4000.csv`.

To repair already-published 4-year rolling alt3 files in the downstream `rapms` repo without rerunning RAPM solves:

```bash
python nba_pipeline/scripts/center_published_alt3_weighted_factors.py \
  --target-dir /Users/russellthomas/Docs/rapms/master_results
```

### Build a 3D Galaxy from Weighted Factors
```bash
cd nba_pipeline/scripts

# Canonical six-factor galaxy
python build_weighted_factors_galaxy.py \
  --input ../master_results/weighted_factors_14_26_all_rb.csv \
  --feature-set six_factor \
  --min-poss 2000

# Eight-factor galaxy with FT split
python build_weighted_factors_galaxy.py \
  --input ../master_results/weighted_factors_14_26_all_rb.csv \
  --feature-set eight_factor \
  --min-poss 2000

# Same pipeline against any other weighted file
python build_weighted_factors_galaxy.py \
  --input ../master_results/weighted_factors_21_26_all_rb_td700.csv \
  --feature-set eight_factor
```

This writes a `galaxy.json`-style payload plus an embeddings CSV under `nba_pipeline/results/galaxy/`.

- `six_factor` uses `oTS, oTOV, oREB, dTS, dTOV, dREB`
- `eight_factor` uses `oEFG, oFT, oTOV, oREB, dEFG, dFT, dTOV, dREB`
- `oEFG = oTS - oFT`
- `dEFG = dTS - dFT`
- by default, the similarity/clustering space also adds `0.5 * standardized net_rapm`

Cluster ids default to `kmeans --n-clusters 10` on the scaled raw factor space, while visible `x/y/z` positions come from a separate 3D UMAP layout. Similarity edges also default to the scaled raw factor space rather than the rendered 3D coordinates. Use `--cluster-space latent --cluster-method hdbscan` only if you want experimental latent-space clustering. See [Weighted Factors Galaxy](./WEIGHTED_FACTORS_GALAXY.md) for the full workflow.

### Build Season Intercepts
```bash
cd nba_pipeline/scripts
python build_season_intercepts.py
```

This writes season-average artifacts under `nba_pipeline/results/season_intercepts/`.

- `raw_season_column_means.csv`: all numeric parquet columns
- `solver_season_intercepts.csv`: signed solver-space baselines, including alias metrics like `BADPASS_TOV`, `SCORING_TOV`, `ALT_TS`, and `ALT_TOV`
- `processed_file_audit.csv`: season-resolution audit, including parquet-vs-filename mismatches
- `season_intercepts.json`: nested lookup for downstream code

The harness uses the processed filename as the canonical season key because older processed parquets do not all encode `Season` consistently inside the file.

---

## Refreshing Data

### Daily Update (Current Season)
```bash
# 1. Fetch latest games (updates NBA26.parquet and NBA26_PS.parquet automatically)
python 01_fetch_pbp_data.py 26
python 01_fetch_pbp_data.py 26 PS

# 2. Enrich ShotQuality data for both regular season and playoffs
python 01b_enrich_pbp_shotquality.py 26

# 3. Reprocess regular season and playoffs
python 02_process_rapm.py ../raw_data/NBA26.parquet
python 02_process_rapm.py ../raw_data/NBA26_PS.parquet

# 4. Re-run analysis; ALL includes both NBA26 and NBA26_PS processed files
python rapm.py RAPM 26 26 ALL
python rapm.py RIM_FREQ 26 26 ALL
python rapm.py RIM_FG_PCT 26 26 ALL
python rapm.py ASSIST_POINTS 26 26 ALL
python rapm.py RIM_ASSIST 26 26 ALL
python rapm.py THREE_FREQ 26 26 ALL
python rapm.py THREE_FG_PCT 26 26 ALL
python rapm.py MIDRANGE_FREQ 26 26 ALL
python rapm.py MIDRANGE_FG_PCT 26 26 ALL
python rapm.py PLAYTYPE_TS_MIX 26 26 ALL
python rapm.py PLAYTYPE_PROXY_PTS 26 26 ALL

# 5. Rebuild active eight-component DECOMP CSV/parquet artifacts
python run_decomp_rolling.py --windows 2026,2025-2026,2024-2026,2023-2026,2022-2026 --force-components ALL
# Legacy team audit bundle
python build_team_alt3_efg_value_weighted_factors.py 24 26 ALL --alpha 25
```

Raw fetch guardrails:
- `01_fetch_pbp_data.py` now normalizes `game_id` to a stable 10-character string before write.
- The fetch step removes exact duplicate raw PBP rows before saving, including duplicate rows that can appear when an update run appends already-fetched events.
- Shared raw readers also defensively dedupe exact rows on read so older bad parquets do not silently inflate possession or scoring totals.
- Shared raw readers also repair missing lineup slots with in-game `ffill/bfill` before dropping rows, which recovers sparse raw events where lineup IDs are present elsewhere in the same game.

### Full Refresh (All Seasons)
```bash
# 1. Update process_historical_rapm.py with desired seasons
# 2. Run batch processing
python process_historical_rapm.py

# 3. Re-run all analyses
python rapm.py RAPM 23 26 ALL
python rapm.py LA_RAPM 23 26 ALL
python rapm.py TS 23 26 ALL
python rapm.py TOV 23 26 ALL
python rapm.py REB 23 26 ALL
python rapm.py RIM_FREQ 23 26 ALL
python rapm.py RIM_FG_PCT 23 26 ALL
python rapm.py ASSIST_POINTS 23 26 ALL
python rapm.py RIM_ASSIST 23 26 ALL
python rapm.py THREE_FREQ 23 26 ALL
python rapm.py THREE_FG_PCT 23 26 ALL
python rapm.py MIDRANGE_FREQ 23 26 ALL
python rapm.py MIDRANGE_FG_PCT 23 26 ALL
python build_master_frequency_rapm.py 23 26 ALL
python rapm.py PLAYTYPE_TS_MIX 24 26 ALL
python rapm.py PLAYTYPE_PROXY_PTS 24 26 ALL
```

### RAPMnoFT Audit Parquet

Build a standard RAPM possession parquet that keeps the usual possession
parser and lineup columns, but scores free throws by actual made/missed points
instead of shooter expected FT%.

```bash
PYTHONPATH=nba_pipeline/scripts \
python nba_pipeline/scripts/build_rapm_noft.py --year 26 --season-type PS
```

Default output:
- `nba_pipeline/processed/RAPMnoFT26_PS.parquet`

---

## WOWY Lineups (2026)

Build a single 2026 lineup table from raw 2026 play-by-play. The parser counts team events first, then derives ORTG/DRTG, TS%, FGORB%, and rim/mid/3 frequencies and accuracies from those raw counts.

```bash
cd nba_pipeline/scripts
python lineup_stats.py build --year 26
```

Default output:
- `nba_pipeline/processed/LINEUP_STATS26.parquet`

Query exact or subset lineups without `pbpstats`:

```bash
python lineup_stats.py query --year 26 --team DEN --exact "203932,203999,1627750,1629661,1631128"
python lineup_stats.py query --year 26 --include "203999" --exclude "201566" --min-off-poss 100 --sort off_poss
python lineup_stats.py player 203999 --year 26
```

The materialized table stores one row per unique sorted 5-man lineup and team with raw counts plus derived rates:
- `team_abbr`, `lineup_key`, `P1`-`P5` as NBA IDs
- raw offensive counts:
  `off_poss`, `off_points`, `off_fga`, `off_fgm`, `off_fg_miss`, `off_fta`, `off_ftm`, `off_orb`, `off_tov`, `off_rim_att`, `off_rim_made`, `off_mid_att`, `off_mid_made`, `off_three_att`, `off_three_made`
- raw defensive allowed counts:
  `def_poss`, `def_opp_points`, `def_opp_fga`, `def_opp_fgm`, `def_opp_fg_miss`, `def_opp_fta`, `def_opp_ftm`, `def_opp_orb`, `def_tov_forced`, `def_opp_rim_att`, `def_opp_rim_made`, `def_opp_mid_att`, `def_opp_mid_made`, `def_opp_three_att`, `def_opp_three_made`
- derived rates:
  `ortg`, `drtg`, `net`, `off_ts_pct`, `def_ts_pct_allowed`, `off_fgorb_pct`, `def_fgorb_pct_allowed`, `off_rim/mid/three_freq`, `def_*_allowed`, `off_rim/mid/three_acc`, `def_*_allowed`

Use `--show-names` if you want the query output to append `lineup_name`.

Implementation notes:
- the parser uses raw `NBA26.parquet` only; it does not use `RAPM26.parquet` or other processed metric parquets for on/off ratings
- possessions use a dedicated pbpstats-style end-of-possession parser with lineup propagation through foul/FT substitution edge cases
- free throws now fall back to the shooting side for offense assignment when the normal possession-side inference is blank
- foreign-side technical/admin FT scoring is preserved as zero-possession scoring rows so points go to the shooting team without adding possessions
- `TSA = FGA + 0.44 * FTA`
- `TS% = Points / (2 * TSA)`
- `ORTG = 100 * Points / OffPoss`
- `DRTG = 100 * OppPoints / DefPoss`
- `FGORB% = ORB / FG misses`

## Career Teammates (1997-2026)

Build each player's top teammates by shared career minutes, with regular season
and playoffs included by default:

```bash
python nba_pipeline/scripts/build_career_teammate_summary.py \
  --start-year 1997 \
  --end-year 2026 \
  --season-types all \
  --top-n 15
```

Default outputs:
- `nba_pipeline/results/career_teammates/career_teammate_pairs_97_26_all.csv`
- `nba_pipeline/results/career_teammates/career_teammate_top15_97_26_all.csv`

The builder gets minutes from raw PBP event-clock deltas, then joins RAPM
processed possessions for ORTG, DRTG, and net rating while the pair is together.
Top-N ranking uses shared possessions, and the output includes both possession
share and minute share. It also subtracts the shared pair possessions from each
player's total on-court possessions to expose each player's net rating without
that teammate. See [Career Teammate Summary](./CAREER_TEAMMATE_SUMMARY.md) for
column definitions.

---

## Quick Reference

| Metric | Measures | Positive Off = | Negative Def = |
|--------|----------|----------------|----------------|
| RAPM | Points | Scores more | Allows less |
| LA_RAPM | Points (luck-adj) | Scores more | Allows less |
| TS | Scoring efficiency | More efficient | Allows less |
| TOV | Turnover rate | Fewer TOs | Forces more TOs |
| REB | OREB rate | More OREBs | Fewer OREBs allowed |
| RIM_FREQ | Rim attack rate | Attacks rim more | Deters rim attacks |
| RIM_FG_PCT | Rim efficiency | Finishes better | Protects rim better |
| ASSIST_POINTS | Assisted scoring | Creates more assisted points | Allows less assisted scoring |
| RIM_ASSIST | Assisted rim creation | Creates more assisted rim makes | Allows fewer assisted rim makes |
| THREE_FREQ | 3PA rate | Takes more 3s | Allows fewer 3s |
| THREE_FG_PCT | 3PT efficiency | Shoots better from 3 | Allows worse 3PT shooting |
| PLAYTYPE_TS_MIX | Descriptor scoring mix | Shifts toward better descriptor bundles | Suppresses better descriptor bundles |
| PLAYTYPE_PROXY_PTS | Proxy shot-diet value | Shifts toward better Synergy-like categories | Suppresses those categories |
| MIDRANGE_FREQ | Midrange rate | Takes more midrange | Forces more midrange |
| MIDRANGE_FG_PCT | Midrange efficiency | Makes more midrange | Allows worse midrange shooting |
