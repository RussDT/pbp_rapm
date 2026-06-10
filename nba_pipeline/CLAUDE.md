# NBA RAPM Pipeline

## Agent Ops Workflow

- Treat `nba_pipeline/docs/` as this repo's FreshDocs surface. Start with `nba_pipeline/docs/README.md`.
- Use `nba_pipeline/docs/Codex_Scratchpad.md` as the repo-versioned short-term memory file.
- At session start: read `/Users/russellthomas/Docs/pbp_rapm/AGENTS.md`, `/Users/russellthomas/Docs/pbp_rapm/CLAUDE.md`, this file, and the scratchpad, then run `git status --short`.
- When methodology, pipeline behavior, outputs, env vars, or operator workflow changes, update the matching docs in `nba_pipeline/docs/` in the same pass.
- At session end: append one short scratchpad entry, promote durable lessons into the root instruction files, and keep only the latest 5 scratchpad entries.
- Git backup policy: commit source, docs, tests, and small curated reference files; leave generated CSV/parquet/log/model artifacts out of git unless explicitly force-added with a documented reason. See `docs/GIT_BACKUP_STRATEGY.md`.
- Repo cleanup/onboarding policy: start with `docs/CONTRIBUTOR_ONBOARDING.md`, `docs/ARTIFACT_MANIFEST.md`, and `docs/REPO_CLEANUP_PLAN.md`; use `scripts/audit_repo_inventory.py` for dry-run inventory before moving, deleting, or reclassifying artifacts/scripts.

## Overview
This pipeline processes NBA play-by-play data to calculate Regularized Adjusted Plus-Minus (RAPM) and related metrics.

## Directory Structure
```
nba_pipeline/
├── raw_data/          # NBA PBP parquet files (NBA23.parquet, NBA23_PS.parquet, etc.)
├── processed/         # Processed metric parquet files (RAPM26.parquet, TS26.parquet, etc.)
├── results/           # RAPM regression results (CSV)
├── master_results/    # Consolidated weighted_factors CSVs
├── scripts/
│   ├── 01_fetch_pbp_data.py      # Fetches raw PBP data from NBA API
│   ├── 01b_enrich_pbp_shotquality.py  # Enriches raw PBP with ShotQuality data
│   ├── build_pbp_shot_quality.py # Builds repo-owned per-FGA shot quality from raw PBP
│   ├── 02_process_rapm.py        # Processes raw PBP into metric parquets
│   ├── 03_run_rapm_analysis.py   # Runs all metrics + regression (creates weighted_factors)
│   ├── rapm.py                   # Ridge regression for individual metrics
│   └── factor_analysis.py        # Analyzes weighted_factors relationships
├── logs/              # Daily update logs
└── docs/              # Documentation
```

## Data Flow
1. **Fetch**: `01_fetch_pbp_data.py` downloads PBP data → `raw_data/NBAXX.parquet`
2. **Enrich**: `01b_enrich_pbp_shotquality.py` adds ShotQuality columns → `raw_data/NBAXX.parquet`
3. **Process**: `02_process_rapm.py` creates metric files → `processed/*.parquet`
4. **Analyze**: `rapm.py` runs ridge regression → `results/*_results.csv`
5. **Combine**: `03_run_rapm_analysis.py` runs all metrics + regression → `weighted_factors*.csv`
6. **PBP Shot Quality**: `build_pbp_shot_quality.py` parses raw FGA descriptions → `results/pbp_shot_quality/*.parquet`

## File Naming Conventions
- **Raw data**: `NBA{YY}.parquet` (regular season), `NBA{YY}_PS.parquet` (playoffs)
- **Processed**: `{METRIC}{YY}.parquet` or `{METRIC}{YY}_PS.parquet`
- **Results**: `{metric}_{start}_{end}_{type}_results.csv` (e.g., `rapm_23_26_all_results.csv`)
- **Weighted factors**: `weighted_factors_{start}_{end}_{type}.csv`

For season-summary harnesses, treat the processed filename as the canonical season key. Legacy processed parquets do not all store the same embedded `Season` convention.
- When syncing published artifacts into the downstream `rapms` repo, check both `nba_pipeline/master_results` and `nba_pipeline/results`; some metric result CSVs, such as `rim_assist_*`, can live only in `results` until the publish step copies them into `master_results`.
- `rapm.py --age-dummies` uses `dpm_history.csv` only as a player-season age lookup (`nba_id`, `season`, `age`); the age fixed effects are still estimated inside the possession regression. `03_run_rapm_analysis.py --age-dummies` applies age fixed effects to the full daily weighted-factor metric suite and writes `_agefe` outputs.
- `FIRST_CHANCE` carries ShotQuality-era `Is_FC_Transition_Possession` and `Has_FC_FGA` labels for the 2023-24 through 2025-26 transition/half-court split. The `FC_TRANSITION_SCORING` / `FC_HALFCOURT_SCORING` aliases use same-sample PPP placeholders, while `FC_MODE_MIX + FC_TRANSITION_VALUE + FC_HALFCOURT_VALUE = FIRST_CHANCE.Net_Diff` at row level.
- Supabase-backed player-stat processors prefer `SUPABASE_SERVICE_ROLE_KEY` over `SUPABASE_KEY`; keep the service role key available for local backend processing so FT% / 3P% lookups do not silently fall back to defaults.
- Preferred long historical core runs use `run_historical_core_weighted_factors.py --fixed-season-effects --age-poly-coefficients ...`: fixed RS raw season baselines and fixed polynomial age offsets are subtracted from the row target before centering and solving, instead of adding estimated nuisance columns.
- Databallr WOWY / lineup net-rating work should use `scripts/build_clean_lineup_net_rating.py` as the duplicate raw action-score possession path instead of `RAPM*.parquet`; it treats period boundaries as hard splits, emits foreign-side technical/admin FT points as zero-possession scoring rows for the shooting team, and drops empty admin-only period-end rows.

## Metrics
| Metric | Column | Description | Years |
|--------|--------|-------------|-------|
| RAPM | Net_Diff, Off_Diff, Def_Diff | Point differential per possession | All |
| LA_RAPM | LA_OffNet, LA_DefNet | Luck-adjusted (FT/3PT regression) | All |
| TS | Net_Diff | True shooting efficiency | All |
| TOV | Is_Turnover | Turnover rate | All |
| REB | Offensive_Rebound | Offensive rebound rate | All |
| SHOOTER_OREB | Offensive_Rebound, Self_Offensive_Rebound | Missed-FGA team/self OREB with shooter separated from four non-shooters | All |
| BLOCK_RECOVERY | Block_Recovered_By_Defense | Defensive recovery rate after blocked FGAs | All |
| RIM_FREQ | Is_Rim_Attempt | Rim attempt frequency (layups/dunks/tips) | All |
| RIM_FG_PCT | Is_Rim_Make | Rim FG% | All |
| ASSIST_POINTS | Assist_Points | Assisted FG points per possession | All |
| RIM_ASSIST | Is_Rim_Assist | Assisted rim makes per possession | All |
| THREE_FREQ | Is_Three_Attempt | 3PA frequency (3PA / FGA) | All |
| THREE_FG_PCT | Is_Three_Make | 3PT FG% (3PM / 3PA) | All |
| MIDRANGE_FREQ | Is_Midrange_Attempt | Midrange frequency (not 3PT, not rim) | All |
| MIDRANGE_FG_PCT | Is_Midrange_Make | Midrange FG% (midrange FGM / midrange FGA) | All |
| PLAYTYPE_TS_MIX | Playtype_Exp_PTS | Descriptor-based expected scoring mix per FGA | 24-26 |
| PLAYTYPE_PROXY_PTS | Playtype_Proxy_PTS | Synergy-proxy category value per FGA using season-relative actual points | 24-26 |
| TRANSITION_FREQ | Is_Transition | Transition shot frequency | 24-26 |
| TRANSITION_RIM | Is_Transition_Rim | Transition rim attempt frequency | 24-26 |
| INITIAL_EV | Initial_EV | Shot quality expected value | 24-26 |
| RUSSELL_SHOTQUALITY | Net_Diff, Off_Diff, Def_Diff | Possession SQ value using repo-owned `shot_quality_season_ev` for all FGAs, expected FT for FTs, and zero for other events | All with PBP SQ artifact |
| CONTEXT_SHOTQUALITY | Net_Diff, Off_Diff, Def_Diff | Possession SQ value using repo-owned context-only `shot_context_ev` for all FGAs, expected FT for FTs, and zero for other events | All with PBP SQ artifact |
| BADPASS_TOV | Is_BadPass_TOV | Bad pass turnover rate (from TOV parquet) | All |
| SCORING_TOV | Is_Turnover - Is_BadPass_TOV | Scoring/ball-handling turnover rate (derived) | All |
| SECOND_CHANCE | Off_Diff, Def_Diff | Second-chance points per possession | All |
| SPECIAL_RAPM | Net_Diff, Off_Diff, Def_Diff | Possession RAPM (rim=actual, non-rim=initial_ev, FT=expected) | 24-26 |
| FIRST_CHANCE | Net_Diff, Is_Turnover | Clean first-chance scoring before first offensive miss | All |
| FC_TRANSITION_SCORING | Is_FC_Transition_Possession, Net_Diff | Same-denominator transition first-chance scoring with transition PPP placeholders | 24-26 |
| FC_HALFCOURT_SCORING | Is_FC_Transition_Possession, Net_Diff | Same-denominator non-transition/half-court first-chance scoring with half-court PPP placeholders | 24-26 |
| FC_MODE_MIX | Is_FC_Transition_Possession, Net_Diff | Additive first-chance mode-mix baseline | 24-26 |
| FC_TRANSITION_VALUE | Is_FC_Transition_Possession, Net_Diff | Transition first-chance scoring above/below same-sample transition PPP | 24-26 |
| FC_HALFCOURT_VALUE | Is_FC_Transition_Possession, Net_Diff | Non-transition/half-court first-chance scoring above/below same-sample half-court PPP | 24-26 |
| SECOND_CHANCE_CLEAN | Net_Diff, Off_Diff, Def_Diff | RAPM-aligned second-chance scoring after first offensive miss | All |

## Common Commands

### Daily Update (Current Season)
```bash
cd nba_pipeline/scripts
python 01_fetch_pbp_data.py 26
python 01_fetch_pbp_data.py 26 PS
python 01b_enrich_pbp_shotquality.py 26
python 02_process_rapm.py ../raw_data/NBA26.parquet
python 02_process_rapm.py ../raw_data/NBA26_PS.parquet
```

### Build Repo-Owned PBP Shot Quality
```bash
python nba_pipeline/scripts/build_pbp_shot_quality.py \
  --years 1997-2026 \
  --season-types all \
  --output-dir nba_pipeline/results/pbp_shot_quality \
  --output-name pbp_shot_quality_1997_2026
```

### Run Individual Metric Analysis
```bash
python rapm.py RAPM 26 26 RS             # Single year regular season
python rapm.py RAPM 23 26 ALL            # Multi-year all games
python rapm.py RAPM 23 26 ALL --pure     # Pure RAPM (no luck adjustment)
python rapm.py RAPM 26 26 RS --pure --cv-alpha  # Cross-validate offense/defense alphas
python rapm.py SPECIAL_RAPM 24 26 ALL --cv-alpha --publish-to-rapms  # Tune and publish result CSV
python rapm.py MIDRANGE_FREQ 23 26 ALL   # Midrange frequency
python rapm.py MIDRANGE_FG_PCT 23 26 ALL # Midrange FG%
python build_master_frequency_rapm.py 23 26 ALL  # Combined rim/mid/three frequency + FG% RAPM export
python rapm.py TRANSITION_FREQ 24 26 ALL # Transition frequency (years 24-26 only)
python rapm.py INITIAL_EV 24 26 ALL      # Shot quality EV (years 24-26 only)
python reprocess_metric.py RUSSELL_SHOTQUALITY 22 26 --season-types all --workers 3
python rapm.py RUSSELL_SHOTQUALITY 22 26 ALL
python reprocess_metric.py CONTEXT_SHOTQUALITY 22 26 --season-types all --workers 3
python rapm.py CONTEXT_SHOTQUALITY 22 26 ALL
python rapm.py SPECIAL_RAPM 24 26 ALL    # Special RAPM (years 24-26 only)
python rapm.py BADPASS_TOV 21 26 ALL     # Bad pass turnover rate
python rapm.py SCORING_TOV 21 26 ALL     # Scoring turnover rate
python rapm.py BLOCK_RECOVERY 21 26 ALL  # Defensive recovery rate after blocked FGAs
python rapm.py SECOND_CHANCE 21 26 ALL   # Second-chance points
python rapm.py FIRST_CHANCE 21 26 ALL    # Clean first-chance scoring
python rapm.py FC_HALFCOURT_SCORING 24 26 ALL --timedecay --half-life 700  # Same-denominator half-court first-chance scoring
python rapm.py FC_TRANSITION_SCORING 24 26 ALL --timedecay --half-life 700 # Same-denominator transition first-chance scoring
python rapm.py ALT_TS 21 26 ALL          # ALT_TS alias off FIRST_CHANCE
python rapm.py ALT_TOV 21 26 ALL         # ALT_TOV alias off FIRST_CHANCE
python rapm.py SECOND_CHANCE_CLEAN 21 26 ALL  # RAPM-aligned second-chance scoring
```

### Reprocess Metric Parquets
```bash
python reprocess_metric.py TOV 14 25              # Reprocess TOV for all historical seasons
python reprocess_metric.py SECOND_CHANCE 14 25     # Reprocess any metric
python reprocess_metric.py FIRST_CHANCE 14 25      # Clean first-chance parquets
python reprocess_metric.py SECOND_CHANCE_CLEAN 14 25  # Clean second-chance parquets
python reprocess_metric.py ASSIST_POINTS 14 26     # Historical assisted points parquets
python reprocess_metric.py THREE_FREQ 14 26        # Historical 3PA frequency parquets
python reprocess_metric.py MIDRANGE_FG_PCT 14 26   # Historical midrange FG% parquets
python reprocess_metric.py PLAYTYPE_TS_MIX 24 26   # Descriptor playtype-mix parquets
python reprocess_metric.py PLAYTYPE_PROXY_PTS 24 26  # Synergy-proxy playtype-value parquets
python reprocess_metric.py FIRST_CHANCE 1997 2013 --season-types rs --workers 6  # Gabriel historical RS range
python reprocess_metric.py FIRST_CHANCE 1997 2013 --season-types ps --workers 6  # Gabriel historical PS range
python reprocess_metric.py LA_RAPM 1997 2013 --season-types rs --workers 6       # Historical LA_RAPM backfill
```

### Build Gabriel Historical Parquets
```bash
python nba_pipeline/scripts/build_gabriel_old_pbp.py \
  --years 1997-2013 \
  --season-types all \
  --metrics RAPM,TOV,REB,TS \
  --report nba_pipeline/validation/gabriel_old_pbp/build_1997_2013_report.json
```
This converts Gabriel `merged_playbyplay/old_data` team files into repo raw
season files plus RS/PS processed parquets. The converted raw rows intentionally
use partial-lineup mode: unknown player slots are `0`, processors keep rows with
at least four known offensive and defensive players, and `rapm.py` excludes
player id `0` from coefficients.

The builder also supports non-ShotQuality factor metrics such as `LA_RAPM`,
`ASSIST_POINTS`, `RIM_ASSIST`, `RIM_FREQ`, `RIM_FG_PCT`, `MIDRANGE_FREQ`,
`MIDRANGE_FG_PCT`, `THREE_FREQ`, `THREE_FG_PCT`, `FT_PREMIUM`, `BLOCK_RECOVERY`, `FIRST_CHANCE`,
`FIRST_CHANCE_CLEAN`, `SECOND_CHANCE`, and `SECOND_CHANCE_CLEAN`. For targeted historical RS
backfills, prefer `reprocess_metric.py METRIC 1997 2013 --season-types rs|ps`
because it parallelizes across seasons. `reprocess_metric.py` also accepts
`--raw-dir` and `--processed-dir` for validation-first historical rebuilds. If
the Gabriel source checkout lacks PS files, repair existing PS raw files with
`repair_existing_historical_lineup_sides.py` before reprocessing. Plain `RAPM`
uses the no-luck surface; `LA_RAPM` is the luck-adjusted surface.

The alt3 metrics `ALT_TS`, `ALT_EFG`, `ALT_FT`, `ALT_TOV`,
`ALT_BADPASS_TOV`, and `ALT_SCORING_TOV` are solver aliases off
`FIRST_CHANCE`, not separate physical `ALT_*` parquets. The regular-season
processed parquet averages builder derives those audit columns from
`FIRST_CHANCE`.

For Gabriel historical starter repairs, the supplemental NBA stats archive at
`/Users/russellthomas/Docs/2026_NBA_PIPELINE/archive (1)` can validate games,
teams, players, and substitutions. Use `PlayByPlay.parquet` substitution
descriptions plus `PlayerStatistics.csv` active/DNP rows, `Players.csv`, and
`TeamHistories.csv`; do not treat old `PlayerStatistics*.csv startingPosition`
values as starter flags without auditing counts, because 2009 rows can have more
than five nonblank values per team.

Historical RAPM uses the shared standard possession flow. Keep EOP/offense
assignment in `process_rapm_blocks/common.py`: infer neutral-description team
turnovers from game-local side labels plus NBA abbreviation/nickname aliases,
run FT correction, fill `poss_offense` before O/D mapping, and let blank
terminal period/neutral rows inherit the opposite side of the previous completed
possession in the same game/period.

Gabriel historical raw rebuilds should apply game-local lineup-side guardrails
before processing: infer player side from non-overlap lineup evidence, strip
impossible same-player home/away collisions, dedupe repeated positive player ids
within one side, and drop only rows that remain below the partial-lineup minimum
after dedupe.

See `docs/HISTORICAL_1997_2013_PARQUETS.md` for the current 1997-2013 file
locations, rebuild commands, FT fallback notes, validation, and caveats.

### Run Full Analysis (All Metrics + Regression)
```bash
python 03_run_rapm_analysis.py 23 26 ALL
python 03_run_rapm_analysis.py 1 4 ALL --rapm-workers 8 --cores-per-rapm 4  # concurrency/load-core benchmark
python 03_run_rapm_analysis.py 97 0 ALL --rubberband --season-effects --publish-to-rapms  # cross-century 1997-2000 window
python 03_run_rapm_analysis.py 21 26 ALL --publish-to-rapms  # copy standard + alt3 weighted factors to rapms and push if changed
```
This creates `weighted_factors` for the standard six-factor build and the legacy `weighted_factors_alt3_*` alternate clean 3-factor build. For current public Alt3 work, use `weighted_factors_alt3_efg_value_*`: it uses the six atomic first-chance EFG-value shot pieces, `ALT_FT`, point-valued `ALT_TOV_VALUE`, and direct `SECOND_CHANCE_CLEAN`; the display closes to total RAPM through `ALT_EFG_BASELINE`, not by making second chance a residual bucket.
Two-digit season windows may cross 1999-2000; `97 0` expands to 1997-2000 and filenames use `97_00`.

## Key Implementation Details

### ShotQuality Enrichment
`01b_enrich_pbp_shotquality.py` merges data from ShotQuality CSVs into raw PBP parquets.

**Source:** `~/Docs/2026_shotquality/nba_games/{year}/*.csv`

**Columns added to raw_data/NBA*.parquet:**
| Column | Description |
|--------|-------------|
| `is_transition` | Shot taken in transition (bool) |
| `is_transition_dunk_or_layup` | Transition dunk or layup (bool) |
| `initial_ev` | Shot expected value at release |
| `dynamic_ev` | Shot expected value with game context |
| `sq_shooter_id` | Shooter NBA player ID (from SQ) |
| `sq_defender_id` | Closest defender NBA player ID (from SQ) |

**Join key:** `game_id` + `event_num` (matches `GAME_EVENT_ID` in SQ CSVs)

**Coverage:** ~35-40% of PBP rows have SQ data (only shot attempts have SQ data)

### Repo-Owned PBP Shot Quality
`build_pbp_shot_quality.py` is independent of the external ShotQuality
`initial_ev` feed. It parses raw PBP field-goal descriptions for distance, shot
zone, shot family, action type, and non-outcome modifiers, then writes per-shot
`shot_context_ev`, career and player-season shooter talent, defender-on-court
residual allowed, `shot_quality_season_ev`, `shot_quality_with_defense_ev`, and
prior-only `shot_quality_prior_ev`. The current full-history artifact is
`results/pbp_shot_quality/pbp_shot_quality_1997_2026.parquet`; summary outputs
include `*_shooter_season_talent.csv` and `*_defender_season_impact.csv`. See
`docs/PBP_SHOT_QUALITY.md`.

### Processed Parquet Columns
All processed files must include these columns for `rapm.py` to work:
- `game_id`, `O1-O5`, `D1-D5` (offensive/defensive players)
- `period`, `away_score`, `home_score`, `score_margin`
- `game_date`, `Season`
- Metric-specific column (e.g., `Is_Midrange_Attempt`)

### rapm.py Detection Logic
The script auto-detects file type by checking for specific columns:
- `Off_Diff` + `Def_Diff` → RAPM file (also SPECIAL_RAPM)
- `Is_Turnover` → TOV file (also contains `Is_BadPass_TOV`)
- prefix=`BADPASS_TOV` → loads TOV file, uses `Is_BadPass_TOV`
- prefix=`SCORING_TOV` → loads TOV file, derives `Is_Turnover - Is_BadPass_TOV`
- prefix=`ALT_TS` → loads FIRST_CHANCE file, computes same-sample non-turnover first-chance baseline, and imputes turnover rows to that baseline
- prefix=`ALT_TOV` → loads FIRST_CHANCE file, uses the same baseline on non-turnover rows and `0` on turnover rows
- `Offensive_Rebound` → REB file
- `Is_Rim_Attempt` → RIM_FREQ file
- `Is_Rim_Make` → RIM_FG_PCT file
- `Assist_Points` → ASSIST_POINTS file
- `Is_Rim_Assist` → RIM_ASSIST file
- `Is_Three_Attempt` → THREE_FREQ file
- `Is_Three_Make` → THREE_FG_PCT file
- `Is_Midrange_Attempt` → MIDRANGE_FREQ file
- `Is_Midrange_Make` → MIDRANGE_FG_PCT file
- `Playtype_Exp_PTS` → PLAYTYPE_TS_MIX file
- `Playtype_Proxy_PTS` → PLAYTYPE_PROXY_PTS file
- `Is_Transition` → TRANSITION_FREQ file
- `Is_Transition_Rim` → TRANSITION_RIM file
- `Initial_EV` → INITIAL_EV file
- `Net_Diff` only → TS file

### Rim Attempt Detection
Uses `RIM_ACTION_TYPES` set (action codes 5,6,7,9,41,43,50,52,75,87,97,98,99,100,107,108) plus description patterns (`layup`, `dunk`, ` tip `).

### Midrange Detection
`Is_Midrange_Attempt = NOT is_3pt AND NOT is_rim`
- 3PT detected via "3pt" in description
- Rim detected via action types + description patterns

## Weighted Factors Output
The `weighted_factors*.csv` files contain regression-weighted factor contributions:
```
player_id, player_name, Latest_Year,
oTS, oTOV, oTOV_bp, oTOV_sc, oREB,       # offensive factors
dTS, dTOV, dTOV_bp, dTOV_sc, dREB,       # defensive factors
off, def, net_rapm, RESID,                # overall RAPM
o_sc, o_fc, d_sc, d_fc,                  # second-chance / first-chance split
o_pval, d_pval,                           # possession value (oTOV+oREB, dTOV+dREB)
possessions, off_poss, def_poss
```

See `docs/tov-decomposition.md` for full details on the TOV decomposition, second-chance integration, and column definitions.

## Rubberband Effect (Added Feb 2025)

The Engelmann rubberband effect: teams ahead coast (score less), teams behind push harder (score more). `rapm.py` now supports this as an optional adjustment via `--rubberband` / `-rb`.

### How It Works
- Uses **pre-possession score margin** (previous row's `score_margin` to avoid endogeneity — the current row's margin includes the current possession's scoring)
- Margin is converted to **offensive-team-relative** `score_diff` during `transform_to_home_away_format()`
- 6 score-margin bins × 4 periods = **24 binary indicator features** appended to the design matrix
- Solved as a **3rd alternating block** in ridge regression: offense → defense → rubberband on residual
- Close game (-5 to +5) is the reference category (omitted)

### Score Margin Bins
| Bin | Range | Label |
|-----|-------|-------|
| Far behind | sd <= -15 | `rb_neg3` |
| Behind | -15 < sd <= -10 | `rb_neg2` |
| Slightly behind | -10 < sd <= -5 | `rb_neg1` |
| **Close game (ref)** | **-5 < sd <= 5** | **dropped** |
| Slightly ahead | 5 < sd <= 10 | `rb_pos1` |
| Ahead | 10 < sd <= 15 | `rb_pos2` |
| Far ahead | sd > 15 | `rb_pos3` |

### Usage
```bash
python rapm.py RAPM 26 26 ALL --rubberband --pure        # Single year
python rapm.py RAPM 23 26 ALL -rb --pure                 # Multi-year (recommended)
python rapm.py TOV 23 26 ALL -rb                         # Works with all metrics
python rapm.py RAPM 23 26 ALL -rb --alpha-rb 50          # Override regularization
```

### Outputs (alongside standard results)
- `*_rb_results.csv` — player ratings with rubberband adjustment
- `*_rb_rubberband_coefficients.csv` — 24 bin × period coefficients
- `*_rb_rubberband_effects.csv` — predicted effect at representative margins per quarter
- `Rubberband_Results/rubberband_report.html` — visual report explaining the effect

### Key Parameters
- `--rubberband` / `-rb`: Enable the adjustment
- `--alpha-rb`: Rubberband regularization (default: 10). Low because only 24 well-identified features vs ~1000 player features at alpha=3000

### Key Findings (23-26 ALL, 920K possessions, Pure RAPM)
- **Team ahead by 20 in Q4**: -5.8 pts/100 (scores less due to +3.1 more turnovers per 100 — sloppiness/garbage time)
- **Team behind by 20 in Q4**: +5.7 pts/100 (scores more due to +5.0 better true shooting — facing relaxed defense)
- Effects near zero in Q1, build through Q2-Q3, maximal in Q4
- Mean player shift: 0.24 pts/100 (conservative). Stars barely move due to ridge L2 penalty mechanics; role players on dominant teams benefit most
- Works across all metrics: RAPM, TS, TOV, REB, etc.

### Sign Convention Warning
The TOV rubberband coefficients are in `Off_Diff = -Is_Turnover` units. So a negative coefficient in the effects CSV means MORE turnovers, not fewer. All metrics use the convention: positive Off_Diff = good for offense.

### Critical Implementation Detail
`score_margin` in processed parquets is **post-possession** (includes current scoring). The rubberband uses the **previous row's margin** within each game to get the pre-possession game state. This avoids endogeneity (mechanical correlation between scoring and margin). See `transform_to_home_away_format()` in `rapm.py`.

## Daily Automation (launchd)

**Plist:** `~/Library/LaunchAgents/com.russellthomas.nba-rapm-update.plist`
**Script:** `nba_pipeline/daily_rapm_update.sh` (canonical; `scripts/daily_rapm_update.sh` is a symlink)
**Schedule:** Daily at 6:00 AM
**Logs:** `nba_pipeline/logs/launchd_stdout.log`, `launchd_stderr.log`, `daily_update_YYYYMMDD.log`

### Pipeline Steps
1. Fetch latest 2026 PBP data
2. Process RAPM parquets
3. Run RAPM analysis for 1-6 year windows (26-26 through 21-26)
4. Time-decay 700 runs (21-26 plain, 21-26 td700+rubberband) + upload to Supabase
5. 13-year rubberband run (14-26)
6. Active Alt3 EFG-value player rolling bundles for windows intersecting 2026, with both CSV and parquet outputs, plus the team-level 24-26 ALL CSV/parquet bundle
7. Standalone metrics with td700 (RIM_FREQ, RIM_FG_PCT, MIDRANGE_FREQ, MIDRANGE_FG_PCT, TRANSITION_FREQ, TRANSITION_RIM, INITIAL_EV)
8. Copy standalone TD results to `master_results/`, mapping canonical `transition_freq_24_26_all_td700_results.csv`, `transition_rim_24_26_all_td700_results.csv`, and `initial_ev_24_26_all_td700_results.csv` onto the published downstream filenames `transitionfreq_24_26_all_td700_results.csv`, `transitionrim_24_26_all_td700_results.csv`, and `initialev_24_26_all_td700_results.csv`, plus curated non-TD `special_rapm_24_26_all_results.csv`
9. TS decomposition components (TS, SQ_POSS, FT_PREMIUM, CONTEST) + WLS regression + Supabase upload
10. Downstream CSV exports (runs from `pbp_rapm/` root):
   - `update_2026_josh_rapm.py` → `josh_rapm/` + `../csvs/josh_rapm/`
   - `update_2026_purerapm.py` → `../csvs/PureRAPM.csv` (legacy)
   - `update_2026_purerapm_peaks.py` → `../csvs/PureRAPMPeaks.csv`
   - `update_2026_scaledoutput.py` → `../csvs/SCALEDOUTPUT_SMALLER.csv`
11. Sync curated standard and alt3 files from `master_results/` → `/Users/russellthomas/Docs/rapms/master_results/`, including the active Alt3 EFG-value `.csv` and `.parquet` artifacts
12. Rebuild the player-facing `csvs/scposs/{nba_id}.csv` files from the active RAPM DECOMP source family using `/Users/russellthomas/Docs/REPLIT_NBA_RAPM/scripts/sync_alt3_decomp_to_scposs.py`. Do not use `update_2026_scposs.py` as the current player-facing scposs writer; it is legacy six-factor output.
13. Current metrics and peak leaderboards:
   - run `/Users/russellthomas/Docs/2026_NBA_PIPELINE/nba_rapm/collect_source_2026.py --no-sync` with `NBA_PIPELINE_CSV_DIR=/Users/russellthomas/Docs/csvs`
   - outputs include `current_comp.csv`, `structuredtest.csv`, `DARKO.csv`, `lebron.csv`, and `players_by_player.csv`
   - current 1Y/2Y/3Y/4Y/5Y RAPM rows/columns are read from the freshly rebuilt RAPM DECOMP `csvs/scposs` files; `TimedecayRAPM` and `*_timedecay` columns remain on the original timedecay source
14. Commit & push the touched `csvs` outputs (`PureRAPM.csv`, `PureRAPMPeaks.csv`, `SCALEDOUTPUT_SMALLER.csv`, `current_comp.csv`, `structuredtest.csv`, `DARKO.csv`, `lebron.csv`, `players_by_player.csv`, `autocomplete_map.csv`, `josh_rapm/`, and `scposs/`) to `RussDT/csvs@test-branch`, which is the branch read by the app.
15. Commit & push the `rapms` repo

### rapms Repo
**Location:** `/Users/russellthomas/Docs/rapms/` (GitHub: RussDT/rapms)
**Contents:** `master_results/` with the published standard and alt3 CSVs:
- `weighted_factors_{26,25_26,24_26,23_26,22_26,21_26}_all.csv`
- `weighted_factors_21_26_all_td700.csv`
- `weighted_factors_14_26_all_rb.csv`
- `weighted_factors_alt3_{26,25_26,24_26,23_26,22_26,21_26}_all.csv`
- `weighted_factors_alt3_efg_value_{26,25_26,24_26,23_26,22_26}_all_rb_se_a2000_4000.{csv,parquet}`
- `weighted_factors_alt3_efg_value_{00_09,10_19,20_26,97_06,17_26}_all_rb_se_a2000_4000.{csv,parquet}` for explicit RAPM DECOMP era/support windows; current peak metrics use `00_09`, `10_19`, and `20_26` for the `2000s RAPM`, `2010s RAPM`, and `2020s RAPM` rows.
- `team_weighted_factors_alt3_efg_value_24_26_all_a25.{csv,parquet}`
- `weighted_factors_alt3_21_26_all_td700.csv`
- `weighted_factors_alt3_21_26_all_rb_td700.csv`
- `weighted_factors_alt3_14_26_all_rb.csv`
- `{rimfreq,rimfgpct,midrangefreq}_21_26_all_td700_results.csv`
- `{transitionfreq,transitionrim,initialev}_24_26_all_td700_results.csv`
- `special_rapm_24_26_all_results.csv`

## Career Teammate Outputs

`scripts/build_career_teammate_summary.py` builds career teammate shared-minute
and net-rating outputs by combining raw `NBA*.parquet` event-clock deltas for
minutes with processed `RAPM*.parquet` possessions for ORTG/DRTG/net. Generated
CSVs under `results/career_teammates/` are artifacts unless explicitly curated
for git.

## External Dependencies
- `autocomplete_map.csv` in project root: Maps player IDs to names
- Supabase: Player stats for luck adjustment calculations
- Game dates CSV: Downloaded from GitHub for game date merging

If `autocomplete_map.csv` lags active fringe players, `scripts/rapm.py` should backfill names from raw PBP event columns and the sibling `2026_NBA_PIPELINE` identity outputs instead of publishing `ID_<player_id>` placeholders.

## Friend PBP Ingest Validation

- BKN 2020 friend-PBP fixture: `raw_data/BKN_2020_rs.parquet`
- Validation harness: `scripts/validate_friend_pbp_ingest.py`
- Isolated artifacts: `validation/friend_ingest/`
- Current best status: shot-zone metrics are exact, TOV flags are exact on matched rows, RAPM has 14,541/14,542 official rows matched with one official-only goaltend-and-one event, remaining friend-only rows are mostly zero-value team rebounds around blocked-shot jump-ball/admin sequences, and the bigger solve-level gap is lineup parity (`740` of `14,541` matched RAPM rows differ in O/D player sets).


<claude-mem-context>
# Recent Activity

<!-- This section is auto-generated by claude-mem. Edit content outside the tags. -->

### Feb 20, 2026

| ID | Time | T | Title | Read |
|----|------|---|-------|------|
| #5870 | 2:04 AM | 🔴 | Fixed Working Directory for Update Scripts | ~244 |
| #5857 | 1:14 AM | 🟣 | Added Combined Time Decay and Rubberband Analysis to Daily Pipeline | ~264 |
| #5856 | 1:11 AM | 🟣 | Daily Pipeline Now Updates Downstream CSVs and Auto-Publishes to RAPMS Repo | ~425 |
| #5855 | 1:09 AM | 🔵 | Complete Daily RAPM Pipeline Automation Workflow | ~519 |
</claude-mem-context>
