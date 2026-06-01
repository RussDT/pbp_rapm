# PBP Shot Quality

`scripts/build_pbp_shot_quality.py` builds a repo-owned shot-quality table from
raw play-by-play. It does not use the external ShotQuality `initial_ev` column.

## Command

```bash
python nba_pipeline/scripts/build_pbp_shot_quality.py \
  --years 1997-2026 \
  --season-types all \
  --output-dir nba_pipeline/results/pbp_shot_quality \
  --output-name pbp_shot_quality_1997_2026
```

Default input files are `nba_pipeline/raw_data/NBA??.parquet` and
`NBA??_PS.parquet`. Debug runs can use `--years`, `--season-types`, and
`--limit-files`.

## Outputs

The full-history build writes:

- `nba_pipeline/results/pbp_shot_quality/pbp_shot_quality_1997_2026.parquet`
- `nba_pipeline/results/pbp_shot_quality/pbp_shot_quality_1997_2026_model_summary.csv`
- `nba_pipeline/results/pbp_shot_quality/pbp_shot_quality_1997_2026_shooter_talent.csv`
- `nba_pipeline/results/pbp_shot_quality/pbp_shot_quality_1997_2026_shooter_season_talent.csv`
- `nba_pipeline/results/pbp_shot_quality/pbp_shot_quality_1997_2026_defender_season_impact.csv`
- `nba_pipeline/results/pbp_shot_quality/pbp_shot_quality_1997_2026_calibration.csv`

These are generated artifacts and should stay out of git unless explicitly
force-added with a documented reason.

## Model

Each field-goal attempt gets four related estimates:

- `shot_context_ev`: expected points from shot context only.
- `shot_quality_ev`: `shot_context_ev` plus a career-window shooter talent residual.
- `shot_quality_season_ev`: `shot_context_ev` plus a season-specific shooter talent residual.
- `shot_quality_with_defense_ev`: `shot_quality_season_ev` plus the five-defender on-court adjustment.

The context model parses the raw PBP shot description into:

- `shot_distance_ft`, parsed from strings such as `25'`; missing distances are
  imputed from shot type.
- `shot_zone`: `rim`, `short_mid`, `long_mid`, `deep_two`, or `three`.
- `shot_family`: `dunk`, `layup`, `tip`, `hook`, `floater`, `three`,
  `jumper`, or `other_two`.
- `modifier_bundle`: non-outcome text descriptors such as `driving`,
  `running`, `cutting`, `pullup`, `step_back`, `turnaround`, `fadeaway`,
  `bank`, `alley_oop`, `putback`, `reverse`, `finger_roll`, and `floating`.
- `event_action_type`, when present in the raw PBP.

The model deliberately does not use made-shot assist tags as a context feature,
because assist tags are only visible after a made basket and would leak the
outcome.

## Estimation

The context EV is an empirical-Bayes, leave-one-shot estimate:

1. Season/season-type/shot-value mean, shrunk to the pooled 2PT or 3PT mean.
2. Zone/distance/family mean, shrunk to the season shot-value estimate.
3. Granular context mean, adding action type and modifier bundle, shrunk to the
   zone estimate.

The shooter layer starts from:

```text
context_residual_points = actual_points - shot_context_ev
```

Then it computes player residual talent by `shooter_id` and `talent_bucket`,
where the bucket is one of:

- `three`
- `rim_finish`
- `short_touch`
- `midrange_jumper`

`shooter_talent_ev_added` is the career-window leave-one-shot empirical-Bayes
residual for the bucket. `shooter_season_talent_ev_added` is then estimated by
`shooter_id`, `season_end_year`, and `talent_bucket`, shrunk toward that
career-window bucket talent. This gives a player-season shooting talent estimate
without letting low-attempt player seasons swing all the way to their raw
make/miss result.

`shooter_talent_prior_ev_added` is a chronological prior using only earlier
shots for that shooter/bucket in the sorted raw-file sequence. The resulting
live-style estimate is `shot_quality_prior_ev`.

## Defense Adjustment

The shot table now carries `O1-O5` and `D1-D5` for the offensive and defensive
players on the court. After context and shooter-season talent are estimated, the
remaining residual is:

```text
season_shooter_residual_points =
actual_points - (shot_context_ev + shooter_season_talent_ev_added)
```

For every defender on court, the script estimates a season/bucket residual
allowed from those opponent residuals:

- key: `defender_id`, `season_end_year`, `talent_bucket`
- shrink: `defender_shrink`, default `750` defender-shot rows
- sign: positive means opponents scored above context-plus-shooter expectation;
  negative means the defender suppressed opponent shooting

The row-level `defender_lineup_ev_allowed_added` is the average of the available
five defender estimates. It is added to `shot_quality_season_ev` to create
`shot_quality_with_defense_ev`.

This is an on-court dEFG-style residual, not a fully deconfounded RAPM solve.
It is useful for adding defender context to shot EV and for quick
player-season defensive shot suppression tables. A later version can replace
this empirical-Bayes on-court estimate with a ridge solve if we want stronger
teammate/opponent separation.

## RUSSELL_SHOTQUALITY RAPM Surface

`process_rapm_blocks/process_russell_shotquality.py` converts the per-FGA table
into standard possession parquets for RAPM:

- FGA events use `shot_quality_season_ev`.
- Free throws use shooter expected FT value, matching the EV-style possession
  processors.
- Turnovers and other non-shot events use `0`.
- Processed parquets write `Def_Diff = -Off_Diff`.

`shot_quality_season_ev` is the default instead of
`shot_quality_with_defense_ev` because the latter already includes a defender
lineup residual. Using the defense-adjusted EV as the RAPM target would make the
defensive side partly circular. The resulting defensive coefficient is therefore
an estimate of suppressing opponent shot-quality value, not a re-solve of the
empirical-Bayes defender residual.

Example last-five-seasons rebuild and solve:

```bash
python nba_pipeline/scripts/reprocess_metric.py RUSSELL_SHOTQUALITY 22 26 \
  --season-types all \
  --workers 3
python nba_pipeline/scripts/rapm.py RUSSELL_SHOTQUALITY 22 26 ALL
```

The 2022-2026 RS+PS run wrote `10` processed files with `1,296,325`
possession rows and solved to
`nba_pipeline/results/russell_shotquality_22_26_all_results.csv`.

For a context-only version without shooter talent, use `CONTEXT_SHOTQUALITY`.
It uses the same possession construction but swaps in `shot_context_ev` for
every FGA:

```bash
python nba_pipeline/scripts/reprocess_metric.py CONTEXT_SHOTQUALITY 22 26 \
  --season-types all \
  --workers 3
python nba_pipeline/scripts/rapm.py CONTEXT_SHOTQUALITY 22 26 ALL
```

## Important Columns

- `actual_points`: 0, 2, or 3.
- `shot_value`: 2 or 3.
- `shot_context_ev`: shot-context expected points.
- `shooter_talent_ev_added`: retrospective leave-one-shot shooter residual.
- `shot_quality_ev`: context plus career-window retrospective shooter talent.
- `shooter_season_talent_ev_added`: season-specific shooter residual shrunk
  toward career-window bucket talent.
- `shot_quality_season_ev`: context plus season-specific shooter talent.
- `defender_lineup_ev_allowed_added`: five-defender on-court residual allowed.
- `shot_quality_with_defense_ev`: context plus shooter season talent plus defense.
- `shooter_talent_prior_ev_added`: chronological prior shooter residual.
- `shot_quality_prior_ev`: context plus prior-only shooter talent.
- `shot_context_make_prob`, `shot_quality_make_prob`,
  `shot_quality_prior_make_prob`: EV divided by `shot_value`.
- `raw_shooter_name`: raw PBP name before mapping.
- `shooter_name`: canonical name when `autocomplete_map.csv` has the NBA ID.

## Current Validation

The 1997-2026 RS+PS run produced `6,307,287` unique shot rows across `60` raw
files. Core EV/feature columns had zero nulls, and all EV-derived make
probabilities were inside `[0, 1]`.

`6,283,922` of `6,307,287` shots had all five defenders available. The remaining
rows are mostly historical partial-lineup rows; defender EV still fills to `0`
when no valid defender slots are present.

Full-run means:

| Metric | Value |
| --- | ---: |
| actual points per shot | 1.008491 |
| `shot_context_ev` mean | 1.006545 |
| `shot_quality_ev` mean | 1.014300 |
| `shot_quality_season_ev` mean | 1.015342 |
| `shot_quality_with_defense_ev` mean | 1.011494 |
| `shot_quality_prior_ev` mean | 1.011941 |

The slight upward shift after adding shooter talent is expected from clipping
near the low end and from empirical-Bayes residuals being applied player by
player. Use `shot_context_ev` when evaluating shot diet independent of shooter,
use `shot_quality_season_ev` when the current-season shooter talent should
matter, and use `shot_quality_with_defense_ev` when both shooter and defensive
on-court context should affect expected value.
