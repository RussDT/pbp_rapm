# Dunk and Dunk Assist Handoff

This document is a handoff for the dunk-assist work in `/Users/russellthomas/Docs/pbp_rapm`.
It covers the event-level assisted-dunk extracts, passer-dunker combo tables, league trend files,
regular-season-to-playoff translation tables, and the related `DUNK`, `DUNK_ASSIST`, and
`RIM_ASSIST` RAPM surfaces.

## Definitions

`DUNK` is a made-field-goal event-count RAPM surface. It is built from the shared `is_dunk`
flag in `nba_pipeline/scripts/process_rapm_blocks/common.py`. The processed parquet target is
`Is_Dunk`.

`DUNK_ASSIST` is a made assisted-dunk event-count RAPM surface. It uses the same `is_dunk`
flag and additionally requires an assist tag in the made-dunk description. The processed parquet
target is `Is_Dunk_Assist`.

`RIM_ASSIST` is the broader assisted-rim-make RAPM surface. Use it when the question is about
assisted rim makes generally, not only dunk assists.

The passer-dunker combo CSVs are separate from the RAPM parquets. They are event-level extracts
that parse made dunk descriptions, read the assist tag, resolve the passer against the offensive
lineup when possible, and keep both passer and dunker IDs/names.

Important raw-PBP note: fetched raw PBP often does not populate `player2_id` / `player2_name` for
assists, so assist-derived work should parse the `(... N AST)` text in shot descriptions.

## Main Event and Combo Files

Regular-season assisted-dunk events:

`/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/validation/dunk_assist_events_rs_1997_2026.csv`

Postseason assisted-dunk events:

`/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/validation/dunk_assist_events_ps_1997_2026.csv`

Regular-season passer-dunker combos by season:

`/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/validation/dunk_assist_combos_by_season_rs_1997_2026.csv`

Postseason passer-dunker combos by season:

`/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/validation/dunk_assist_combos_by_season_ps_1997_2026.csv`

Extraction summaries:

`/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/validation/dunk_assist_extraction_summary_rs_1997_2026.csv`

`/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/validation/dunk_assist_extraction_summary_ps_1997_2026.csv`

Current row counts:

| File | Rows | Notes |
|---|---:|---|
| `dunk_assist_events_rs_1997_2026.csv` | 209,021 | RS assisted-dunk events, 1996-97 through 2025-26 |
| `dunk_assist_events_ps_1997_2026.csv` | 12,770 | PS assisted-dunk events, 1996-97 through 2025-26 |
| `dunk_assist_combos_by_season_rs_1997_2026.csv` | 61,354 | RS passer-dunker-season combo rows |
| `dunk_assist_combos_by_season_ps_1997_2026.csv` | 6,659 | PS passer-dunker-season combo rows |

The event files include:

- `season_end_year`
- `season`
- `team`
- `game_id`
- `event_num`
- `period`
- `passer_id`
- `passer_name`
- `passer_name_raw`
- `passer_resolve_status`
- `dunker_id`
- `dunker_name`
- `event_action_type`
- `event_text`

The combo files include:

- `season_end_year`
- `season`
- `team`
- `passer_id`
- `passer_name`
- `passer_resolve_status`
- `dunker_id`
- `dunker_name`
- `dunk_assists`

## League Trend and Derived Tables

Regular-season league average by season:

`/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/validation/dunk_assist_league_average_by_season_rs_1997_2026.csv`

This file contains league-wide assisted dunks, games, team offensive possessions, assisted dunks per
game, per team-game, and per 100 team offensive possessions. Recent values:

| Season | Dunk assists | DunkAst/100 team poss |
|---|---:|---:|
| 2023-24 | 8,843 | 3.68 |
| 2024-25 | 9,026 | 3.74 |
| 2025-26 | 9,115 | 3.73 |

3PA/FGA correlation file:

`/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/validation/dunk_assist_3pr_correlation_by_season_rs_1997_2026.csv`

This joins league `3PA/FGA` from `THREE_FREQYY.parquet` to league `DunkAst/100`. In the current
extract, 1996-97 through 2025-26 has Pearson `r = 0.885` between league 3PA/FGA and league
assisted dunks per 100 team possessions.

Top 50 regular-season DunkAst/100 players with postseason translation:

`/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/validation/top50_rs_dunk_ast_per100_playoff_translation_1997_2026.csv`

By-season detail for that table:

`/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/validation/top50_rs_dunk_ast_per100_playoff_translation_by_season_1997_2026.csv`

The top-50 table uses a minimum of `8,000` regular-season offensive possessions. For each top
regular-season passer, it reports:

- career RS offensive possessions
- career RS dunk assists
- RS DunkAst/100
- playoff offensive possessions
- playoff dunk assists
- playoff DunkAst/100
- same-season RS DunkAst/100 weighted by the player's postseason possession mix
- playoff drop versus that weighted RS expectation

Current top-50 aggregate:

| Group | DunkAst/100 |
|---|---:|
| Actual playoff rate | 1.56 |
| Same-season RS rate weighted to playoff mix | 1.95 |
| Difference | -0.40 |
| Percent change | -20.4% |

## Processed RAPM Parquet Surfaces

All season files exist for end-year suffixes `97`, `98`, `99`, and `00` through `26`, with
regular-season and postseason versions.

`DUNK` processed parquet pattern:

`/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/processed/DUNK{YY}.parquet`

`/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/processed/DUNK{YY}_PS.parquet`

`DUNK_ASSIST` processed parquet pattern:

`/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/processed/DUNK_ASSIST{YY}.parquet`

`/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/processed/DUNK_ASSIST{YY}_PS.parquet`

`RIM_ASSIST` processed parquet pattern:

`/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/processed/RIM_ASSIST{YY}.parquet`

`/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/processed/RIM_ASSIST{YY}_PS.parquet`

Each family currently has 60 parquets: 30 regular-season files and 30 postseason files.

## RAPM Result Files

Dunk RAPM:

`/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/results/dunk_97_26_all_results.csv`

Dunk Assist RAPM:

`/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/results/dunk_assist_97_26_all_results.csv`

Long Rim Assist RAPM:

`/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/results/rim_assist_97_26_all_rb_se_a2000_4000_results.csv`

Long Rim Assist season effects:

`/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/results/rim_assist_97_26_all_rb_se_a2000_4000_season_effects.csv`

Long Rim Assist rubberband coefficients:

`/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/results/rim_assist_97_26_all_rb_se_a2000_4000_rubberband_coefficients.csv`

Long Rim Assist rubberband effects:

`/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/results/rim_assist_97_26_all_rb_se_a2000_4000_rubberband_effects.csv`

Age-FE long Rim Assist RAPM:

`/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/results/rim_assist_97_26_all_rb_se_agefe_a2000_4000_results.csv`

`/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/results/rim_assist_97_26_all_rb_se_agefe_a2000_4000_season_effects.csv`

`/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/results/rim_assist_97_26_all_rb_se_agefe_a2000_4000_age_effects.csv`

Modern / time-decay Rim Assist runs:

`/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/results/rim_assist_22_26_rs_se_a2000_2000_results.csv`

`/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/results/rim_assist_22_26_rs_se_a2000_2000_season_effects.csv`

`/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/results/td/rim_assist_22_26_rs_se_a4000_4000_offdefprior_td700_results.csv`

`/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/results/td/rim_assist_22_26_rs_se_a4000_4000_offdefprior_td700_season_effects.csv`

## Source PBP Files

Regular season raw PBP:

`/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/raw_data/NBA{YY}.parquet`

Postseason raw PBP:

`/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/raw_data/NBA{YY}_PS.parquet`

The same end-year suffix convention applies: `NBA97.parquet` is `1996-97`,
`NBA00.parquet` is `1999-00`, and `NBA26.parquet` is `2025-26`.

## Processor and Solver Code

Dunk processor:

`/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/scripts/process_rapm_blocks/process_dunk.py`

Dunk assist processor:

`/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/scripts/process_rapm_blocks/process_dunk_assist.py`

Rim assist processor:

`/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/scripts/process_rapm_blocks/process_rim_assist.py`

Shared processing entrypoint:

`/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/scripts/02_process_rapm.py`

Targeted metric reprocessor:

`/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/scripts/reprocess_metric.py`

RAPM solver:

`/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/scripts/rapm.py`

Relevant audit/average builder:

`/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/scripts/build_processed_rs_parquet_averages.py`

## Rebuild Commands

Reprocess one regular-season metric window:

```bash
cd /Users/russellthomas/Docs/pbp_rapm/nba_pipeline/scripts
python reprocess_metric.py DUNK_ASSIST 1997 2026 --season-types rs
```

Reprocess playoffs:

```bash
cd /Users/russellthomas/Docs/pbp_rapm/nba_pipeline/scripts
python reprocess_metric.py DUNK_ASSIST 1997 2026 --season-types ps
```

Run a full DUNK_ASSIST RAPM solve:

```bash
cd /Users/russellthomas/Docs/pbp_rapm/nba_pipeline/scripts
python rapm.py DUNK_ASSIST 97 26 ALL
```

Run a full DUNK RAPM solve:

```bash
cd /Users/russellthomas/Docs/pbp_rapm/nba_pipeline/scripts
python rapm.py DUNK 97 26 ALL
```

Run the long Rim Assist rubberband / season-effect solve:

```bash
cd /Users/russellthomas/Docs/pbp_rapm/nba_pipeline/scripts
python rapm.py RIM_ASSIST 97 26 ALL --rubberband --season-effects --off-alpha 2000 --def-alpha 4000
```

## Social / Design Artifacts

Portrait matrix:

`/Users/russellthomas/.gstack/projects/RussDT-pbp_rapm/designs/dunk-assist-portraits-20260520`

DunkAst/100 social variants:

`/Users/russellthomas/.gstack/projects/RussDT-pbp_rapm/designs/dunkast-per100-social-20260521`

League trend and 3PA/FGA correlation variants:

`/Users/russellthomas/.gstack/projects/RussDT-pbp_rapm/designs/dunk-assist-league-trend-20260521`

## Practical Notes for Future Agents

- Use the validation CSVs for passer-dunker questions. The RAPM parquets do not carry the passer
  and dunker identities.
- Use `DUNK_ASSIST` when the target is specifically assisted dunks.
- Use `RIM_ASSIST` when the target is broader assisted rim finishing.
- Use `DUNK` when the target is dunk frequency / dunk impact regardless of whether the dunk was
  assisted.
- For per-100 player leaderboards, count offensive possessions from `RAPM{YY}.parquet` / `RAPM{YY}_PS.parquet`
  by player appearances in `O1` through `O5`.
- For postseason translation, weight each player's same-season regular-season `DunkAst/100` by
  that player's postseason offensive possessions in the same season.
- Remap final player names from a player directory when presenting historical outputs. Assist
  text often contains only last names, especially in older seasons.
