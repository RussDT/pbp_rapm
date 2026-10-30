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

Modern raw PBP is canonically ordered by game, period clock descending, then
event number within one clock. Use `scripts/repair_modern_raw_lineups.py` for
2014+ lineup integrity audits and evidence-constrained repairs; see
`docs/MODERN_LINEUP_REPAIR.md`.

## File Naming Conventions
- **Raw data**: `NBA{YY}.parquet` (regular season), `NBA{YY}_PS.parquet` (playoffs)
- **Processed**: `{METRIC}{YY}.parquet` or `{METRIC}{YY}_PS.parquet`
- **Results**: `{metric}_{start}_{end}_{type}_results.csv` (e.g., `rapm_23_26_all_results.csv`)
- **Weighted factors**: `weighted_factors_{start}_{end}_{type}.csv`

For season-summary harnesses, treat the processed filename as the canonical season key. Legacy processed parquets do not all store the same embedded `Season` convention.
- When syncing published artifacts into the downstream `rapms` repo, check both `nba_pipeline/master_results` and `nba_pipeline/results`; some metric result CSVs, such as `rim_assist_*`, can live only in `results` until the publish step copies them into `master_results`.
- `rapm.py --age-dummies` uses `dpm_history.csv` only as a player-season age lookup (`nba_id`, `season`, `age`); the age fixed effects are still estimated inside the possession regression. `03_run_rapm_analysis.py --age-dummies` applies age fixed effects to the full daily weighted-factor metric suite and writes `_agefe` outputs.
- The 1997-2026 age-adjusted career O/D parent uses fixed RS scoring baselines, joint possession age bins, a slot-weighted cubic over ages 20-38 anchored at 27, and frozen `--age-curve-values` offsets. Player + age + freely fitted season effects do not identify the linear age tilt; the near-OLS annual curve is rejected. See `docs/THIRTY_YEAR_AGE_ADJUSTED_RAPM.md`.
- Brand the parent as **21st Century RAPM** with an explicit 1997-2026 subtitle. The negative-LAKER research challenger applies only to negative age-27 career LAKER players at `<=1000` RS minutes and uses `minutes/(minutes+500)` O/D stabilization at scale `1.0`; keep its `neglaker_m1000_k500_s1` output distinct from the zero-prior control. See `docs/THIRTY_YEAR_AGE_ADJUSTED_RAPM.md`.
- Its adaptive-precision challenger uses exact generalized ridge with O/D multipliers `4/3` only for one-season players age `<=22.5` whose nearest observation is over four years from age 27. The rule was selected on 1998-2021 first-to-second-season age-27 stability and improved in 2022-2025 stress. Keep `young1s_o4_d3` distinct, research-only, and preserve fixed-alpha controls. See `docs/THIRTY_YEAR_AGE_ADJUSTED_RAPM.md`.
- The selected forward factor family is `weighted_factors_decomp_net_*`: five exact `DECOMP_NET_*` shot children plus `ALT_FT`, `ALT_TOV_NET_VALUE`, and `SECOND_CHANCE_NET`. The owner-expanded artifact contract is all 140 available rolling 1Y-5Y windows from 1997 through 2026 for actual Net, common-alpha SQ-v2 plus residual Net, and component-alpha TD550 SQ-v2 research VPM. Build them separately with `run_rapm_overnight.py --full-window-grid` and require CSV/parquet pairs. Consumer cutover remains pending and must be reversible. Gross `weighted_factors_decomp_*` remains compatibility/rollback during migration; Alt3 files are legacy/audit. See `docs/NET_CONTINUATION_DECOMP.md`.
- `scripts/tune_forward_component_alphas.py` is the production-side compatibility module used by the private `alt3_3factor_research` stabilized-RAPM and VPM scripts. It must keep RS/PS family loading, training-split baselines, and target construction aligned with `rapm.py`.
- `scripts/build_vpm_target_swap.py` is the RAPM-side adapter at the cross-repo seam. RAPM owns canonical PBP/DECOMP/solver behavior; the sibling VPM repo owns frozen EV sources and experiment decisions and invokes this adapter through `vpm/run_harness.py prepare-targets`. Do not duplicate target-writing logic in the research repo.
- The long-term VPM goal is to beat standalone EPM on predeclared, date-correct future-game forecasts. Keep future actual-possession prediction as the intermediate gate, use common availability/minutes assumptions to isolate player-metric quality, and test lineup synergy or full-system forecasting as separate layers.
- The sibling EPM12 direct-parent lane is the mandatory simplicity control for VPM: exact date-safe side-wise EPM12 `p_*` unions map to matured 365-day future pure O/D RAPM, then enter one parent solve. Selected development settings are mapper alpha `0.1`, O/D alphas `4000/8000`, and prior scale `1.5`. On frozen 2023-2026 stress the parent beats RAPM by `+254.11e-6 [171.85,336.95]`; decomposed selected VPM adds only `+41.55e-6 [-82.46,165.29]`, while a 25% post-solve pull clearly loses to the parent. Keep decomposition for now, but make the parent—not plain RAPM—the minimum baseline for incremental component claims. Preserve untouched 2027. Canonical sibling receipt: `vpm/docs/EPM12_PARENT_BASELINE.md`.
- The corrected EPM12 counting-rate lane derives `3FGM/100`, rim `FGM/100`, mid `FGM/100`, and `FTM/100` from attempt rates times predictive percentages before mapping; never treat percentage and attempts as merely separate additive scoring-volume terms. The old lane remains a labeled misspecified historical incumbent because the post-hoc correction lost on spent stress. Do not change the board; carry corrected v13/v14 into protected 2027. Canonical sibling receipt: `vpm/docs/EPM12_COUNTING_RATE_CORRECTION.md`.
- The accepted VPM player board comes from the sibling harness `estimate-board` command after an as-of prior build. Use the standalone v2 x2 TD550/component-alpha specification without an EPM blend or rejected absence adjustment. Determine current players from latest-season PBP participation and report box-feature coverage separately.
- Databallr's `/impactmonster/vpm` comparison surface stages three separate validated artifacts from one frozen as-of specification: the selected prior-centred board, an `ev_noprior` board with identical target/history/TD550/component alphas, and the raw x2 prior anchors before lineup evidence. Keep the 582-player current-PBP universe identical, retain all ten atomic fields, flip raw defense once to positive-good, and leave unmodeled component-sides as explicit zeroes. The solved boards use the full configured 2016-2026 history; the raw-prior tab is not a time-decayed lineup solve.
- Historical 1997-2015 no-prior VPM work uses `build_russell_v2_historical_bridge.py`: v2-distilled midrange, explicit portable-v3 three fallback, and native v2 unchanged from 2016 onward. Safe-crossfit remains audit-only. Use fixed season effects or an audited equivalent in a pooled 1997-2026 solve because the raw 2015/2016 EV-probability seam is nonzero. Canonical doc: `docs/RUSSELL_V2_HISTORICAL_BRIDGE.md`.
- `vpm-explorer/` is the public Sites reader for the VPM board (`https://vpm-lab-2026.russdt.chatgpt.site`). Its current frozen CSV/manifest are the EPM12 x2 predictive research board; label it as development-leading rather than untouched confirmation. It verifies all ten component identities plus six-factor/offense/defense/net closure and labels five-player sums as additive studies rather than lineup predictions. Keep broader RAPM/research drilldowns in a separate non-additive lane, preserve the bounded-scroll board, and make the player view a connected identity → offense/defense/total VPM → `3 x 2` factor → component hierarchy. The `3 x 2` map stays intact on mobile. Its TS tiles condense the eight raw TS inputs into four exact display rollups: Rim = rim frequency + rim FG; Mid = mid EV + mid residual; 3P = three frequency + three EV + three residual; FT = FT value. Keep the original ten-component engine and tests intact. On desktop, keep the connected header and both factor rows visible as a compact single-screen read by controlling header, intro, side-bar, tile, and inter-row height while preserving the explanatory copy. Remove secondary prose before values on mobile and use explicit turnover-value wording. The default player-card theme is a warm-white/black Geist editorial treatment with chartreuse reserved for aggregate VPM totals; keep the restrained blue-offense/green-defense dark treatment as the switchable alternate. Refresh the board only after a deliberate new harness build.
- `FIRST_CHANCE` carries ShotQuality-era `Is_FC_Transition_Possession` and `Has_FC_FGA` labels for the 2023-24 through 2025-26 transition/half-court split. The `FC_TRANSITION_SCORING` / `FC_HALFCOURT_SCORING` aliases use same-sample PPP placeholders, while `FC_MODE_MIX + FC_TRANSITION_VALUE + FC_HALFCOURT_VALUE = FIRST_CHANCE.Net_Diff` at row level.
- Supabase-backed player-stat processors prefer `SUPABASE_SERVICE_ROLE_KEY` over `SUPABASE_KEY`; keep the service role key available for local backend processing so FT% / 3P% lookups do not silently fall back to defaults.
- Publish curated reader-facing result CSVs through `scripts/publish_supabase_results.py` and `config/supabase_publications.json`; dry-run must pass before the explicit `--publish`. The generic `analytics_*` hub is content-addressed, retains old versions, promotes only complete uploads, and gives sites read-only RLS access through `analytics_current_rows`. Register new public results or document a dedicated-table replacement, and never publish raw licensed/provider rows, PII, auth data, credentials, or large intermediate artifacts there.
- Preferred long historical core runs use `run_historical_core_weighted_factors.py --fixed-season-effects --age-poly-coefficients ...`: fixed RS raw season baselines and fixed polynomial age offsets are subtracted from the row target before centering and solving, instead of adding estimated nuisance columns.
- Databallr WOWY / lineup net-rating work should use `scripts/build_clean_lineup_net_rating.py` as the duplicate raw action-score possession path instead of `RAPM*.parquet`; it treats period boundaries as hard splits, emits foreign-side technical/admin FT points as zero-possession scoring rows for the shooting team, and drops empty admin-only period-end rows.
- The Five-Man Fit Lab historical refit starts at 2014 and uses clean lineup possessions, season-end EPM/E-Skills, on-ball time, and potential assists. Normalize the five EPM `p_usg` predictions to 100% for displayed allocation/concentration; raw sums are role-demand diagnostics, not lineup usage. Keep the fitted construction model, actual scoreboard, and repo-owned `FIRST_CHANCE + SECOND_CHANCE_CLEAN` mechanism ledger separate. The current defensive challenger failed 2024-2026 confirmation, so public defense remains the five-player D-EPM sum. Never substitute DPM.
- The Five-Man Fit hypothetical builder exports the latest complete-season player pool with the frozen offensive model parameters. Hypothetical lineups get no exact-five identity term; show raw role demand, normalized usage allocation, domain and feature contributions, O-EPM-based offense, and D-EPM-sum defense.

## Metrics
| Metric | Column | Description | Years |
|--------|--------|-------------|-------|
| RAPM | Net_Diff, Off_Diff, Def_Diff | Point differential per possession | All |
| LA_RAPM | LA_OffNet, LA_DefNet | Luck-adjusted (FT/3PT regression) | All |
| TS | Net_Diff | True shooting efficiency | All |
| TS_SELF_SHOT | TS_Points | Full TS attempt value with the shooter as focal player | All |
| TS_TEAMMATE_SHOT | TS_Points | Full TS attempt value with each nonshooter as focal player | All |
| TOV | Is_Turnover | Turnover rate | All |
| REB | Offensive_Rebound | Offensive rebound rate | All |
| SHOOTER_OREB | Offensive_Rebound, Self_Offensive_Rebound | Missed-FGA team/self OREB with shooter separated from four non-shooters | All |

For forward SQ + second-chance evaluation, use first-FGA shot points plus the
matching terminal `SECOND_CHANCE_CLEAN` value, fit OREB and continuation
pricing only on prior seasons, and compare shooter × zone against the identical
shooter-hidden shot-context/exact-lineup continuation lane. The rolling
2024-25 and 2025-26 tests confirm contextual continuation in both seasons but
not a stable pooled shooter × zone gain; the 2025-26 pooled increment crosses
zero, and Embiid's forward long-two role-vacancy effect shrinks from `-1.15`
OREB percentage points to `-0.13`.
| BLOCK_RECOVERY | Block_Recovered_By_Defense | Defensive recovery rate after blocked FGAs | All |
| RIM_FREQ | Is_Rim_Attempt | Rim attempt frequency (layups/dunks/tips) | All |
| RIM_FG_PCT | Is_Rim_Make | Rim FG% | All |
| ASSIST_POINTS | Assist_Points | Assisted FG points per possession | All |
| ASSIST_POINTS_RECEIVER | Receiver_Assist_Points | Assisted FG points received by each focal offensive player; five rows per possession | All |
| ASSIST_POINTS_PASSER | Passer_Assist_Points | Assisted FG points created by each focal passer; five rows per possession | All |
| SCORING_PARENT | Scoring_Points | Actual FG plus shooter-expected FT points per standard possession | All |
| SCORING_ONBALL | OnBall_Scoring_Points | Focal player scoring value when taking the shot/FT; five rows per possession | All |
| SCORING_OFFBALL | OffBall_Scoring_Points | One-quarter non-shooter share of teammate scoring; five rows per possession | All |
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
python reprocess_metric.py ASSIST_POINTS_RECEIVER 26 26 --season-types rs  # Role-attributed receiver opportunities
python reprocess_metric.py ASSIST_POINTS_PASSER 26 26 --season-types rs    # Role-attributed passer opportunities
python reprocess_metric.py SCORING_ROLES 22 26 --season-types rs          # Parent + paired on/off-ball scoring roles
python reprocess_metric.py TS_SHOT_ROLES 22 26 --season-types rs         # Paired self-shot/teammate-shot TS roles
python reprocess_metric.py THREE_FREQ 14 26        # Historical 3PA frequency parquets
python reprocess_metric.py MIDRANGE_FG_PCT 14 26   # Historical midrange FG% parquets
python reprocess_metric.py PLAYTYPE_TS_MIX 24 26   # Descriptor playtype-mix parquets
python reprocess_metric.py PLAYTYPE_PROXY_PTS 24 26  # Synergy-proxy playtype-value parquets
python reprocess_metric.py INITIAL_EV_BETA 24 26    # Initial-EV beta-adjusted parquets (requires SQ enrichment)
python reprocess_metric.py EV_RAPM 24 26            # EV-based possession parquets (requires SQ enrichment)
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
This creates `weighted_factors` for the standard six-factor build and the legacy Alt3 families. Gross compatibility DECOMP is built separately with `run_decomp_rolling.py`; the selected forward product is Net DECOMP and its runner now supports the current 1Y-5Y release. Keep actual Net DECOMP, uniform-alpha SQ+residual Net DECOMP, and component-alpha TD550 Net SQ VPM distinct in `config/rapm_table_registry.json`. See `docs/NET_CONTINUATION_DECOMP.md`, `docs/RAPM_TABLE_REGISTRY.md`, and `docs/PUBLIC_DECOMP.md`. `weighted_factors_alt3_efg_value_*` remains the immediate predecessor/audit family: it uses six atomic first-chance EFG-value shot pieces, `ALT_FT`, point-valued `ALT_TOV_VALUE`, and direct `SECOND_CHANCE_CLEAN`, closing through `ALT_EFG_BASELINE`.
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
| `sq_primary_defender_id` | ShotQuality closest-defender NBA player ID |
| `sq_provider` | Defender/enrichment provider provenance |
| `sq_match_status` | Exact, provider-ambiguous, raw-key-ambiguous, or unmatched |
| `sq_match_candidate_count` | Distinct SQ candidates for the transitional event key |
| `sq_source_row_count` | Raw SQ rows observed for the transitional event key |

**Transitional join key:** `game_id` + `event_num` (matches `GAME_EVENT_ID` in
SQ CSVs). This pair is not guaranteed unique. Exact duplicate SQ payloads may
collapse, but conflicting provider rows and duplicate raw keys remain explicit
ambiguities with null payload. Never resolve them with `keep="first"`.

**Coverage:** ~35-40% of PBP rows have SQ data (only shot attempts have SQ data)

NBA matchup defender assignments and ShotQuality closest-defender assignments
are separate providers. Do not coalesce either into a generic `defender_id`.
ShotQuality has no native second-defender ID; keep second-defender geometry but
label any reconstructed identity as inferred. Canonical contract:
`docs/NBA_EVENT_ENRICHMENT_CONTRACT.md`.
The strict 2023-24-to-2024-25 three-point comparison preserves every
pipe-separated NBA assignment and finds the assignment prior neutral versus
context+EPM but clearly worse than rich SQ release geometry; see
`docs/SQ_DEFENDER_PROVIDER_COMPARISON.md`. Neither defender benchmark includes
OREB continuation in its immediate points-per-shot target.
The same comparison now joins exact raw event lineups and five as-of D-EPM
values. Both summed and richer opposing-lineup D-EPM aggregates are
neutral-to-worse, including when added to SQ geometry+identity; retain this as
a failed challenger rather than assuming all-in personnel impact predicts a
conditional three-point outcome.

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
- `Receiver_Assist_Points` / `Passer_Assist_Points` → five focal-player opportunity rows used by `run_assist_points_hierarchy.py` to fit a role contrast and produce additive children of parent offensive `ASSIST_POINTS`
- `OnBall_Scoring_Points` / `OffBall_Scoring_Points` → five focal-player scoring-role rows used by `run_scoring_role_hierarchy.py`; the final components add to `SCORING_PARENT` offensive RAPM
- `TS_Points` in `TS_SELF_SHOT` / `TS_TEAMMATE_SHOT` → conditional shot-role rows used by `run_ts_shot_role_hierarchy.py`; parent oTS is split exactly into self-shot, teammate-shot, and shot-mix components
- Prefer `run_joint_ts_shooter_interaction.py` for current oTS role work: one row per attempt, player-season-centered shooter interactions, grouped game CV with the one-SE role-alpha rule, and exact direct self-shot + teammate-shot closure. Keep the separate-role Shapley runner as an audit prototype.
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

## Clock-Aware Rubberband Effect

`rapm.py --rubberband` / `-rb` defaults to the validated smooth rubberband model:

- Uses offensive-team-relative score margin and exact clock at the **start** of the possession.
- Fits separate continuous leading and trailing margin-by-regulation-progress surfaces, anchored to zero at a tie, plus separate overtime level/progress terms.
- Uses 80 piecewise-linear basis features and default `alpha_rb=100`.
- Solves offense, defense, rubberband, player priors, time weights, and unpenalized season/age nuisance terms in one exact augmented sparse LSQR system.
- Keeps the established `_rb` output token and adds `*_rubberband_surface.csv` alongside coefficient and representative-effect files.

The old 24 margin-bin × period model remains available with `--rubberband-model legacy`; it defaults to `alpha_rb=10` and uses `_rb_legacy` filenames so it cannot overwrite smooth outputs.

Rolling five-season development (2018-2021 tests) selected smooth alpha 100. After historical score-state repair, frozen historical stress on 2022-2026 improves possession MSE versus no adjustment by `0.0022336 [0.0020922, 0.0023686]` and versus legacy by `0.0018924 [0.0017722, 0.0020101]`, winning all five seasons. Player net ratings remain stable versus legacy (`r=0.9975`, median absolute shift `0.059` points per 100).

The basis and `alpha_rb=100` are fixed, but the coefficients are not. Each solve fits one common surface over its loaded seasons. A frozen 1997-2001 surface still beats no adjustment in every later era, but rolling five-season surfaces beat the fixed early curve by `0.000503 [0.000474,0.000530]` MSE overall and `0.000971 [0.000909,0.001032]` in 2018-2026. Treat a 1997-2026 surface as pooled, not era-specific. See `docs/RUBBERBAND_ADJUSTMENT.md` and `docs/RUBBERBAND_ERA_STABILITY.md`.

### Home-Court Research Boundary

The global centered home-offense nuisance term is a rolling-window promotion candidate, not yet a universal default. After historical venue repair, adding it after smooth rubberband improves held-out MSE in 2002-2009 by `0.000537 [0.000461,0.000610]`, in 2010-2017 by `0.000339 [0.000272,0.000407]`, and in 2018-2026 by `0.000120 [0.000076,0.000167]`. Player net ratings remain stable (`r=0.99928`, median absolute shift `0.042`).

Historical score/venue state must use the repaired transform: it carries sparse scoreboard snapshots forward, seeds venue identity from exclusive score changes, and propagates identity through lineup continuity. It resolves roughly 99.3%-99.8% of 1997-2026 possessions; unresolved rows receive zero home context. The smooth-plus-home coefficient declines from about `+4.85` points/100 in 2002-2009 tests to `+2.95` in 2018-2026, so a long pooled coefficient is not era-specific. Canonical receipt: `docs/HOMECOURT_ADJUSTMENT.md`.

### Usage
```bash
python rapm.py RAPM 23 26 ALL --rubberband --pure          # smooth, alpha 100
python rapm.py TOV 23 26 ALL --rubberband                  # same game-state basis, TOV target units
python rapm.py RAPM 23 26 ALL --rubberband --alpha-rb 1000 # explicit sensitivity
python rapm.py RAPM 23 26 ALL --rubberband --rubberband-model legacy
```

### Sign Convention Warning
The TOV rubberband coefficients are in `Off_Diff = -Is_Turnover` units. So a negative coefficient in the effects CSV means MORE turnovers, not fewer. All metrics use the convention: positive Off_Diff = good for offense.

### Critical Implementation Detail
`score_margin` and `time_quarter` in processed parquets are **post-possession**. The rubberband uses the preceding possession end state within a game and period; new periods start at 12:00 in regulation or 5:00 in overtime. This avoids mechanical target leakage from the current possession. See `transform_to_home_away_format()` and `rubberband_model.py`.

## Daily Automation (launchd)

**Plist:** `~/Library/LaunchAgents/com.russellthomas.nba-rapm-update.plist`
**Script:** `nba_pipeline/daily_rapm_update.sh` (canonical; `scripts/daily_rapm_update.sh` is a symlink)
**Schedule:** Daily at 6:00 AM
**Logs:** `nba_pipeline/logs/launchd_stdout.log`, `launchd_stderr.log`, `daily_update_YYYYMMDD.log`

### Season Selection

The orchestrator is season parameterized. Season year N means season (N-1)-N,
so `2027` is the 2026-27 season. Every two-digit result token (`27`, `22_27`,
`15_27`, ...) and every window argument is derived from the single season
variable; there are no season literals left in the pipeline path.

```bash
./nba_pipeline/daily_rapm_update.sh --year 2027      # 2026-27 season
NBA_RAPM_SEASON_YEAR=2027 ./nba_pipeline/daily_rapm_update.sh
```

`--year` requires a four-digit year and overrides the env var. The default is
`2027`.

Window derivation from `SEASON_YEAR`: the 1Y-6Y runs use starts
`SEASON_YEAR-0` .. `SEASON_YEAR-5`, the rubberband run uses `SEASON_YEAR-12`
(13 seasons), the DECOMP rolling windows are `SEASON_YEAR` plus
`SEASON_YEAR-1..-4` paired with `SEASON_YEAR`, and the WOWY presence upload
spans `SEASON_YEAR-4 .. SEASON_YEAR`.

### Path And Behavior Environment Variables

The script no longer contains absolute machine paths. `PIPELINE_DIR` is
resolved from the script's own (symlink-followed) location, and everything
downstream is overridable:

| Variable | Default | Purpose |
| --- | --- | --- |
| `NBA_RAPM_SEASON_YEAR` | `2027` | Season end year; `--year` overrides it |
| `NBA_RAPM_PIPELINE_DIR` | the script's own directory | `nba_pipeline` root |
| `NBA_RAPM_CSVS_DIR` | `<repo>/../csvs` | `csvs` repo checkout; exported to the producers |
| `NBA_RAPM_CSVS_BRANCH` | `test-branch` | Branch the `csvs` push asserts |
| `NBA_RAPM_RAPMS_DIR` | `<repo>/../rapms` | `rapms` repo checkout; sync/push skipped with a warning when absent |
| `NBA_RAPM_CURRENT_METRICS_DIR` | unset | `nba_rapm` dir of the downstream NBA pipeline; step skipped with a warning when unset |
| `NBA_RAPM_CURRENT_METRICS_SCRIPT` | `collect_source_<SEASON_YEAR>.py` | Current-metrics collector filename |
| `NBA_RAPM_PUSH` | `1` | Set to `0` to skip the `csvs`/`rapms` commits and pushes |

The Python producers read the same variables plus:

| Variable | Default | Purpose |
| --- | --- | --- |
| `NBA_RAPM_RESULTS_DIR` | `<repo>/nba_pipeline/results` | RAPM results root |
| `NBA_RAPM_JOSH_DIR` | `<repo>/josh_rapm` | In-repo `josh_rapm` output dir |
| `NBA_RAPM_SEASON_LIMITS` | `<repo>/../REPLIT_NBA_RAPM/first_last_season.csv` | Peak builder season limits |

Missing `rapms`, `csvs`, or current-metrics locations now log a warning and
skip rather than aborting the run.

### Pipeline Steps
1. Fetch latest `SEASON_YEAR` PBP data (regular season and playoffs)
2. Process RAPM parquets
3. Run RAPM analysis for 1-6 year windows (`SEASON_YEAR` through `SEASON_YEAR-5`..`SEASON_YEAR`), all with `--season-effects` (August 2026 playoff-handling change; outputs carry `_se`)
4. Time-decay 700 runs (21-26 se+td700, 21-26 rb+se+td700) + upload to Supabase (`weighted_factors_21_26_all_rb_se_td700.csv`)
5. 13-year rubberband run (14-26)
6. Active eight-component DECOMP player rolling bundles for the 1Y-5Y current windows, with both CSV and parquet outputs; the team-level Alt3 bundle remains an audit artifact
7. Standalone metrics with td700 (RIM_FREQ, RIM_FG_PCT, MIDRANGE_FREQ, MIDRANGE_FG_PCT, TRANSITION_FREQ, TRANSITION_RIM, INITIAL_EV)
8. Copy standalone TD results to `master_results/`, mapping canonical `transition_freq_24_26_all_td700_results.csv`, `transition_rim_24_26_all_td700_results.csv`, and `initial_ev_24_26_all_td700_results.csv` onto the published downstream filenames `transitionfreq_24_26_all_td700_results.csv`, `transitionrim_24_26_all_td700_results.csv`, and `initialev_24_26_all_td700_results.csv`, plus curated non-TD `special_rapm_24_26_all_results.csv`
9. TS decomposition components (TS, SQ_POSS, FT_PREMIUM, CONTEST) + WLS regression + Supabase upload
10. Downstream CSV exports (runs from `pbp_rapm/` root). The `2026` in these
   filenames is historical: each producer is season parameterized and is
   invoked with `--year "$SEASON_YEAR"`, resolving its own paths from
   `Path(__file__).parent` and `NBA_RAPM_CSVS_DIR`.
   - `update_2026_josh_rapm.py --year N` → `josh_rapm/` + `<csvs>/josh_rapm/`
   - `update_2026_purerapm.py --year N` → `<csvs>/PureRAPM.csv` (legacy)
   - `update_2026_purerapm_peaks.py` → `<csvs>/PureRAPMPeaks.csv` (season agnostic; derives peaks from whatever PureRAPM.csv holds, so it takes no `--year`)
   - `update_2026_scaledoutput.py --year N` → `<csvs>/SCALEDOUTPUT_SMALLER.csv`
   - `update_2026_scposs.py --year N` → `<csvs>/scposs/{nba_id}.csv` (also parameterized; still not wired into the orchestrator, see step 12)
11. During migration, sync curated standard and gross compatibility `weighted_factors_decomp_*` CSV/parquet files from `master_results/` → downstream `rapms/master_results/`; the 1Y-5Y Net release audit passed on 2026-09-04, but replace that consumer with explicit `weighted_factors_decomp_net_*` only through a separately verified mapper and rollback cutover
12. The old Alt3-to-scposs mapper is intentionally skipped until its 97-column DECOMP schema mapping is audited; do not silently feed it the new files
13. Current metrics and peak leaderboards:
   - run `$NBA_RAPM_CURRENT_METRICS_DIR/$NBA_RAPM_CURRENT_METRICS_SCRIPT --no-sync` with `NBA_PIPELINE_CSV_DIR=$NBA_RAPM_CSVS_DIR`; the collector defaults to `collect_source_<SEASON_YEAR>.py`, so a new season needs either a collector named for it or an explicit `NBA_RAPM_CURRENT_METRICS_SCRIPT`
   - outputs include `current_comp.csv`, `structuredtest.csv`, `DARKO.csv`, `lebron.csv`, and `players_by_player.csv`
   - current 1Y/2Y/3Y/4Y/5Y RAPM rows remain on the last audited `csvs/scposs` surface until the DECOMP mapper is restored; `TimedecayRAPM` and `*_timedecay` remain on their original source
14. Commit & push the touched `csvs` outputs (`PureRAPM.csv`, `PureRAPMPeaks.csv`, `SCALEDOUTPUT_SMALLER.csv`, `current_comp.csv`, `structuredtest.csv`, `DARKO.csv`, `lebron.csv`, `players_by_player.csv`, `autocomplete_map.csv`, `josh_rapm/`, and `scposs/`) to `RussDT/csvs@test-branch`, which is the branch read by the app.
15. Commit & push the `rapms` repo

### rapms Repo
**Location:** `$NBA_RAPM_RAPMS_DIR` (default `<repo>/../rapms`; GitHub: RussDT/rapms)
**Contents:** `master_results/` with the published standard and alt3 CSVs:
- `weighted_factors_{26,25_26,24_26,23_26,22_26,21_26}_all_se.csv` (pre-August-2026 syncs used `_all.csv`; stale copies may remain downstream)
- `weighted_factors_21_26_all_se_td700.csv`
- `weighted_factors_14_26_all_rb.csv`
- `weighted_factors_alt3_{26,25_26,24_26,23_26,22_26,21_26}_all_se.csv`
- `weighted_factors_alt3_efg_value_{26,25_26,24_26,23_26,22_26}_all_rb_se_a2000_4000.{csv,parquet}`
- `weighted_factors_alt3_efg_value_{00_09,10_19,20_26,97_06,17_26}_all_rb_se_a2000_4000.{csv,parquet}` for explicit RAPM DECOMP era/support windows; current peak metrics use `00_09`, `10_19`, and `20_26` for the `2000s RAPM`, `2010s RAPM`, and `2020s RAPM` rows.
- `team_weighted_factors_alt3_efg_value_24_26_all_a25.{csv,parquet}`
- `weighted_factors_alt3_21_26_all_se_td700.csv`
- `weighted_factors_alt3_21_26_all_rb_se_td700.csv`
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

## WNBA Gabriel Ingest And Benchmark

- Sync the ignored public checkout at `external/gabriel_wnba/` with
  `../wnba_test/sync_gabriel_wnba.py`, then run
  `../wnba_test/audit_wnba_source_inputs.py` before processing.
- Modern Gabriel CSV headers evolve per game. Normalize by column name and
  reconstruct lineups from cached boxscore starters plus ordered consecutive
  substitution blocks. Use outs-before-ins unless one player appears in both
  directions inside the same team block, then replay that block sequentially.
  A corrected non-sub action may borrow the nearest same-clock actor-compatible
  snapshot without mutating the persistent lineup.
- Preserve the full normalized archive. Use
  `validate_wnba_normalized_lineups.py --write-clean` for the conservative
  whole-game-exclusion solve surface.
- Proprietary WNBA daily ratings remain ignored local inputs. Main validation
  uses latest `rating_date < game_date`; same-day and future rows are forbidden
  until provider timing semantics are confirmed.
- `../wnba_test/build_wnba_box_priors.py` is the preferred current WNBA research
  build: factor-specific public box predictions trained only on earlier seasons
  and inserted inside the exact single-season prior-centered ridge. The
  forward-selected specification is offense alpha `5000`, defense alpha
  `40000`, and prior scale `1.5`; defense is now on the audited grid edge and
  should be extended next without using 2026 for selection.
  `../wnba_test/build_wnba_metric_history.py` produces the canonical
  2022-2026 reader, with 2022 labeled as a no-prior warm-up and 2023-2026 using
  only earlier seasons for box-prior training. Preserve exact FT, TOV,
  atomic-total, and fitted display identities.
  `build_prior_informed_wnba.py` remains the historical-RAPM-state prototype,
  and proprietary ratings remain external benchmarks only.
- Use `../wnba_test/tune_wnba_component_alphas.py` for VPM-style WNBA
  factor-alpha research. It must score the summed component coefficient maps
  against future actual possessions and keep 2026 out of selection. The July
  29 factor-specific candidate lost to uniform `5000/40000` on 2026, so do not
  promote it; `factor_trust.csv` is a marginal diagnostic only.
- Build WNBA attempt-level expected points with
  `../wnba_test/build_wnba_pbp_shot_quality.py`. Model features must be
  available before the result; raw action labels, outcomes, actual points,
  full descriptions, and made-only assist tags are forbidden. The current
  impact scope is MID and THREE only, with RIM left on actual outcomes. The
  modern-only season-forward model improves 2026 eligible-shot MSE by `1.068%`
  over a shot-value baseline. Classic PBP contributes 239,829 additional
  historical MID/THREE attempts, but neither the 2018-plus nor full-history
  window beat modern-only on 2026, so historical input stays opt-in. The model
  remains research-only until it is attached additively to the MID/THREE clean
  first-chance targets with RIM unchanged and passes a forward component-sum
  gate. See
  `docs/WNBA_PBP_SHOT_QUALITY.md`.
- Run the current 2026 rating through
  `../wnba_test/update_wnba_metrics.py --as-of "$(date +%F)"`. This stages the
  WNBA CDN pull, requires a clean current-season lineup report, force-refreshes
  PBP Stats totals, rebuilds all six CSV surfaces, and uses only the frozen
  pre-2026 `5000/40000/1.5` solve. Stable outputs are under
  `../wnba_test/Results/current/`; build the self-contained results/box-prior
  review with `../wnba_test/build_wnba_results_review.py`. Current box-prior
  eligibility requires positive offensive and defensive possessions; PBP Stats
  `Minutes` is audit-only because valid active-player rows can report
  nonpositive minutes. Preserve those exposure conflicts in the manifest. See
  `docs/WNBA_DAILY_UPDATE.md`.
- Audit the ignored WNBA RTSS/Second Spectrum export with
  `../wnba_test/audit_rtss_wnba_features.py`. Common facts come from 2D
  markings; 3D is an ID-keyed enrichment because seven games have populated
  2D and empty 3D possessions and some nonempty event tables differ. Prefer
  `chance_players` for rich five-on-five reconstruction, use game plus shot
  `pbpId` for numeric WNBA identity evidence, and finish unresolved identities
  through roster/team/stint constraints rather than fuzzy names. Provider
  `qSQ`/`qSP` remain benchmarks, post-release/outcome fields are not pre-shot
  features, and rich ShotQuality impact remains MID/THREE only. See
  `docs/WNBA_RTSS_FEATURE_AUDIT.md`.
- Refresh that ignored mirror from `s3://rtss-data/wnba/`: verify the active
  AWS identity, list the prefix, inspect `api/portal/schedules.json`, dry-run
  `aws s3 sync`, and then sync to `../downloads/rtss-data/wnba/`. The former
  local IAM user `rtss-data` no longer exists; manual reads currently work
  through the bucket policy, while unattended automation should use a new
  least-privilege profile. Never store signed URLs or credentials in docs or
  generated manifests.
- Build WNBA RTSS component-prior candidates with
  `../wnba_test/build_wnba_rtss_prior_features.py` and evaluate them with
  `../wnba_test/evaluate_wnba_rtss_priors.py`. The first paired
  2023->2024-selection / 2025-confirmation screen produced nine two-metric wins
  among 14 development-eligible component sides, strongest for defensive THREE
  frequency, defensive FT, and offensive second chance. It is underpowered
  because selection fits only the 2023 RTSS season and does not alter the
  frozen live box prior. Keep the six RIM/MID/THREE frequency/finishing priors
  separate, train the box backbone on every earlier 2022+ season, and use only
  compact multi-year stabilized RTSS corrections. The 2025 holdout is spent,
  and raw `qSP-qSQ` needs a frozen season-relative transform plus provider
  timing audit before fresh future confirmation. See
  `docs/WNBA_RTSS_PRIOR_RESEARCH.md`.
- Use `../wnba_test/train_wnba_rtss_pbp_student.py` for the WNBA qSQ-to-PBP
  distillation lane. Join provider games through the audited inventory and
  `shots.pbpId`, exclude every conflicting event, treat `2*qSQ` as the
  context-only expected-points teacher, and forbid outcomes/provider fields
  from student inputs. The standalone student is not a PBP replacement; the
  2024-selected context blend improves 2025 development MSE by
  `+0.001247 [+0.000151,+0.002282]`, while the shooter-inclusive comparison is
  inconclusive. Keep RTSS 2026 sealed and require an exact EV/residual
  possession attachment plus summed future-factor gate before VPM use. See
  `docs/WNBA_RTSS_PBP_SHOT_QUALITY.md`.
- Use `../wnba_test/cross_validate_wnba_rtss_priors.py` for symmetric
  leave-one-2023/2024/2025-season-out development with a full non-held-out
  2022-2025 box backbone and RTSS fit only to cross-fitted box residuals.
  `shots.assistOpp` labels potential-assisted versus self-created attempts on
  makes and misses; `shots.assisted` is made-only outcome information.
  `passes.assistOpp` supplies potential assists created. Current development
  improves both pooled RMSE and correlation for 15/24 component-sides; the
  offensive shot candidates are RIM frequency, MID frequency, MID finishing,
  and THREE frequency, while RIM/THREE finishing stay box-only. All three RTSS
  seasons are now development data.
- Build the research-only WNBA box-estimated rZTOV board with
  `../wnba_test/build_wnba_rztov.py`. RTSS 2023-2025 supplies the potentially
  assisted-share and potential-assist labels; scored seasons use only the
  public PBP Stats-style box/PBP panel. Keep estimated self-created plus
  assisted TSA equal to box TSA exactly, preserve the negative-good rate and
  separate scoring/bad-pass centering, and do not promote the current
  `14.88% / 6.71% / 18.59%` break-even fit into live WNBA VPM without fresh
  observed-label and future-impact confirmation. See `docs/WNBA_RZTOV.md`.
- Use `../wnba_test/cross_validate_wnba_rtss_total_priors.py` for the separate
  direct-total prior diagnostic. It builds annual clean-total ORAPM/DRAPM
  targets, varies offense and defense target penalties in orthogonal sweeps,
  and interprets RTSS with held-out-season permutation families plus sign
  stability. `off_alpha=250` is the lowest defensible exploratory offense
  target (RTSS-only `R2=0.292`, correlation `0.541`); even
  `def_alpha=1000` gives RTSS-only `R2=0.007`, so low-alpha direct defense is
  not a usable prior target. Box remains stronger and monolithic RTSS additions
  do not help it. Keep the lane research-only and preserve factor-specific
  priors for the additive model.
- Use `../wnba_test/build_wnba_rtss_playmaking_roles.py` for the separate
  FGA-only shooter/passer/presence role model. Exact `chance_players` rows and
  player-centered shooter/potential-assist-passer interactions must close
  `scoring + passer shot + presence` to the offensive shot parent. Freeze one
  pooled-CV role alpha across all annual targets, and price
  `passes.turnover` as a separate cost child of net playmaking. Current
  development supports on-ball time plus potential assists for passer shot
  value (`R2 0.597`), finds rim assists redundant, and finds no creation-stat
  prior for presence. Do not call the FGA-only output full oTS until the FT
  component is added. See `docs/WNBA_RTSS_PLAYMAKING_RAPM.md`.
- Bootstrap WNBA box/component priors from NBA history only with aligned-target
  hierarchical partial pooling: league-season standardization, NBA slope
  priors, WNBA deviation penalties, and component-specific transfer strength
  selected on WNBA folds. Compare WNBA-only, NBA-only, naive pooled, and
  hierarchical candidates. There is no local men's RTSS markings surface with
  the identical `assistOpp`/`qSQ` schema, so rich creation-role and geometry
  residual coefficients remain WNBA-only.
- Canonical workflow and initial 2026 holdout:
  `docs/WNBA_GABRIEL_INGEST_AND_BENCHMARK.md`.


## NBA parent-prior target contract

- Build sibling VPM annual ORAPM/DRAPM prior targets as independent single-season RS pure+rubberband solves with fixed offense/defense alphas `2000/4000`; raw `def` flips once to positive-good display DRAPM.
- The sibling `build-parent-rapm-priors` lane freezes completed 2018-2023 development and uses cross-fitted predictive-skill opponent strength for matchup burden. Complete EPM, BAM, and result-derived suppression remain benchmark/diagnostic fields.
- Frozen 2024-2026 stress is positive for the full bank, but the matchup increment is inconclusive and the result is not admitted without target-covariance-aware uncertainty plus a summed future-possession VPM gate. Do not emit 2027 from incomplete 2026 matchups.

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
