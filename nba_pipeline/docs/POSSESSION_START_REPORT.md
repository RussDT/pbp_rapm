# Possession Start Report

`scripts/build_possession_start_report.py` recreates a possession-start offensive-rating chart from local raw NBA play-by-play.

The report uses the standard RAPM possession parser, so the possession denominator matches the repo's processed `RAPM{YY}.parquet` surface. Each possession is scored with actual offensive points, then labeled by the terminal event that created the possession. Categories are not mutually exclusive: for example, an at-rim missed field goal is also a missed 2PT and a missed FG.

## Command

```bash
python nba_pipeline/scripts/build_possession_start_report.py --year 26 --season-type RS
```

Outputs default to `nba_pipeline/results/possession_start_report/`:

- `possession_start_league_26_rs.csv`
- `possession_start_teams_26_rs.csv`
- `possession_start_league_26_rs_league.png`
- `possession_start_league_26_rs_league.svg`

To render a team chart:

```bash
python nba_pipeline/scripts/build_possession_start_report.py --year 26 --season-type RS --team OKC
```

Use `--compare scope` when a team chart should compare categories to that team's own average offensive possession instead of league average.

## Category Notes

- `Live Ball Turnover` is a previous terminal turnover with a steal tag in the event text.
- `Deadball` is a previous terminal turnover without a steal tag.
- Miss categories are based on the field-goal miss that immediately precedes the terminal defensive rebound.
- `At Rim Miss` uses rim action-type codes when available, then layup/dunk/tip text, then `<=4` foot distance fallback.
- `FT Miss` includes both previous terminal missed free throws and defensive rebounds that terminate a possession after a missed free throw. Offensive rebounds after a missed free throw keep the same possession alive, so they should not create a new possession-start row.
- `Corner 3 Miss` and `Above The Break 3 Miss` are distance approximations because this raw PBP source does not contain shot coordinates or explicit corner labels. Missed threes at `<=24` feet are labeled corner-range; missed threes at `>=25` feet are labeled above-break-range; threes without a parsed distance remain only in `Missed 3PT`.
- `Timeout` is applied when a timeout appears before the first live action in the new possession.
