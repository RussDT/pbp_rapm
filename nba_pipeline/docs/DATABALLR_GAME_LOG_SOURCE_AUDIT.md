# Databallr merged game logs source audit

Read-only production API inspection, October 21, 2026. No ratings, filters, forecasts or protected 2027 outcomes changed or queried. Counts below describe live responses observed on this date, not a full league-wide data-quality certification.

## What is accessible

The [team-log endpoint](https://api.databallr.com/docs#tag/nba-pbp/GET/v1/nba/pbp/team-log) and sibling player-log endpoint serve column-oriented per-season game logs from Cloudflare R2. The important option is **`report=true`**, which adds the merged report beyond the default PBP core.

```text
GET https://api.databallr.com/v1/nba/pbp/team-log
    ?team_id=1610612743&season=2026&ps=false&report=true

GET https://api.databallr.com/v1/nba/pbp/player-log
    ?person_id=203999&season=2026&ps=false&report=true
```

`season` is the season-ending year (2026 means 2025–26); `ps=true` selects playoffs. The live handler reads `ps`, not a `playoffs` query flag. `report=true` is a literal boolean string. The response supplies `core.rows`, `core.columns`, and optionally `report.rows`/`report.columns`. Each column is an array; join core/report by `game_id`, preserving same-named fields with source prefixes. Production API-key documentation specifies a Bearer key with `stats:read`; these particular live reads succeeded without adding credentials in this inspection.

The public OpenAPI inventory gives response schemas but omits query parameter definitions for these two routes. Parameter behavior was checked against [the local route implementation](../../../../products/databallr/apps/api/src/nba-pbp-routes.ts) and live requests, rather than inferred from missing documentation.

## Verified responses

| Sample | Core rows | Core columns | Report columns | Report columns with at least one non-null value |
|---|---:|---:|---:|---:|
| Denver 2025–26 RS | 82 | 164 | 658 | 649 |
| Jokić 2025–26 RS | 65 | 96 | 558 | 535 |
| Jokić 2024–25 RS | 70 | 96 | 558 | 513 |
| LeBron 2013–14 RS | 77 | 96 | 558 | 281 |
| Jordan 1996–97 RS | 82 | 96 | Report null | — |

All sampled non-null core/report files had matching `game_id` arrays. Column counts include metadata, duplicate measures, derived rates and tracking vintages; they are not counts of independent candidate predictors.

Denver core logs returned successfully for **all 30 ending seasons 1997–2026**, totaling **2,393 team-game rows**, each with the same 164 column names. Date range: November 1, 1996 through April 12, 2026. This confirms one franchise's regular-season endpoint coverage; it does not establish every player's coverage or playoff completeness. Twelve Denver rows had `lineups_clean=false`; one had a non-null `known_issue`. A null issue label alone therefore is not a clean-lineup guarantee.

## What the logs contain

| Family | Examples observed in the payload |
|---|---|
| Identity and exposure | `game_id`, `date`, team/opponent IDs, `is_home`, `seconds_pbp`, `off_poss`, `def_poss` |
| PBP box counts | Points, FGA/FGM, threes, FT, assists, steals, blocks, offensive/defensive rebounds, turnovers |
| Shooting detail | Rim, short/long midrange, corner/above-break threes; makes/attempts; assisted makes by zone |
| Passing detail | Assists by scoring zone and two/three-point destination |
| Fouls and turnovers | Shooting/offensive/personal/loose-ball/technical fouls; bad-pass, lost-ball, traveling and out-of-bounds turnovers |
| Possession context | First/second-chance, transition, halfcourt, paint, penalty and points off turnovers |
| Tracking report | Potential/secondary assists, drives, passes, touches, possession time, distance/speed, contested rebounds and rebound chances |
| Shot/defense report | Catch-and-shoot, pull-ups, closest-defender buckets, defended attempts/makes by range, under-six-foot fields |
| Hustle report | Deflections, contested shots, charges drawn, screen assists, loose-ball recoveries and box-outs in player reports |
| Quality and source status | `lineups_clean`, `rows_without_10_players`, `provably_wrong_rows`, `known_issue`, `tracking_vintage`, `ft_tracking_status`, `ft_ast_status`, report join flags |

Team core fields generally have `own_*` and `opp_*` prefixes; player core fields describe individual production and on-court exposure. Team report contains additional detailed factors, heaves, penalty/second-chance splits and free-throw-trip fields, but some are wholly null in the modern Denver sample. Their presence is not evidence of usable measurements.

## Source semantics that matter for VPM

1. **Historical core and modern tracking have different coverage.** Jordan's 1997 core has daily shot-zone, assisted-make, steal/block and other counts; its report is null. LeBron's 2014 report has populated v1 tracking, while hustle and several defended-shot families are null. Modern coverage cannot be projected backward from a stable schema.
2. **Tracking vintages differ materially.** For Jokić's first 2026 row, potential assists are 16 in v1 and 17 in v2; contested DREB is 4 versus 5; under-six-foot defended FGA is 6 versus 5. Pick a documented source vintage, rather than silently filling across versions or treating them as extra independent evidence.
3. **2024–25 FT tracking is explicitly unreliable.** Jokić's 70 report rows all carry `ft_tracking_status=unreliable_nba_2024_25` and the same FT-assist status. `drive_fta_v2` is null throughout that sample, while v1 has values. Filling v2 from v1 would bypass the source warning.
4. **Quality flags require interpretation.** Reconcile the flagged games and canonical exclusions against our clean target manifest. Do not equate a returned row or non-null counter with admission to the selected model.
5. **Logs are postgame observations.** They can feed a chronological filter only after the game. Reconstruct next-game states using legal parameter vintages; neither source arrival nor current merged reports certify what was published historically.
6. **These are game aggregates.** They can supply historical predictive-stat measurements. The prior's possession-level scoring target still requires the separate ten-player lineup evidence; the team-log response does not expose individual possession lineup rows.

## Implication for the pregame prior plan

This is a promising earlier daily observation source for [the pregame plan](VPM_PREGAME_PRIOR_PLAN.md). It could replace some early annual-only inputs and extend reconstructed predictive stats before 2014. It does **not** by itself supply legal pre-2015 OU parameters or establish compatibility with our current stat definitions.

Next source milestone: map the relevant core/report counts and denominators to current estimator measurements; compare overlapping player-games with the selected sources; census coverage and flagged games; preserve tracking-vintage/missingness semantics; then reconstruct the pregame panel. Any source/history extension remains a separate controlled change from mapper input timing.

## Saved inspection artifacts

Generated files remain outside Git under `output/data/databallr_game_logs_20261021/`:

- [All column names and sampled non-null coverage](../../output/data/databallr_game_logs_20261021/column_inventory.csv): 1,476 entity/part/column entries. Player coverage columns refer to different named samples above, not a league census.
- [Jokić 2025–26 merged sample](../../output/data/databallr_game_logs_20261021/player_2026_sample.csv): 65 rows, with `core.`/`report.` prefixes.
- [Denver 2025–26 merged sample](../../output/data/databallr_game_logs_20261021/team_2026_sample.csv): 82 rows, with source prefixes.
- [Thirty-season Denver coverage](../../output/data/databallr_game_logs_20261021/denver_30_season_coverage.json).

Live examples: [Denver core and report](https://api.databallr.com/v1/nba/pbp/team-log?team_id=1610612743&season=2026&ps=false&report=true), [Jokić core and report](https://api.databallr.com/v1/nba/pbp/player-log?person_id=203999&season=2026&ps=false&report=true), [Jordan 1997](https://api.databallr.com/v1/nba/pbp/player-log?person_id=893&season=1997&ps=false&report=true), [OpenAPI inventory](https://api.databallr.com/docs/openapi.json).
