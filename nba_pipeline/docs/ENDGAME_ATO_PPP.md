# End-Game ATO PPP

## Purpose

`scripts/analyze_endgame_ato.py` measures actual points per possession after an
offensive timeout late in the fourth quarter. It reports the result by the
offense's pre-possession deficit (`0`, `1`, `2`, or `3`) and by seconds left at
the timeout (`1` through `7`, then `8-24`).

## Definition

The analysis covers 2013-14 through 2025-26 regular seasons and playoffs. An
eligible ATO possession must:

- occur in period 4 with more than 0 and at most 24 seconds left;
- begin with the offense tied or trailing by 1-3 points;
- contain a regular or short timeout by the team that owns the ensuing
  possession before its first competitive action; and
- resolve through the repo's standard possession parser, with Q4 and overtime
  separated in the grouping key.

Coach challenges, official timeouts, defensive timeouts, and timeouts after a
competitive action are not ATOs. Multiple qualifying timeout rows in one
possession are collapsed to one possession and timed from the last offensive
timeout. PPP is actual offensive points through the end of that standard
possession, including free throws and offensive-rebound continuations.

Clock bins use a ceiling convention: `(0,1]` is labeled `1`, `(1,2]` is `2`,
and so on through `(6,7]`; `(7,24]` is `8-24`.

## Historical score repair

Some 2014-17 raw files contain `0-0` placeholders on non-scoring rows after the
game has begun. The analyzer identifies impossible `0-0` rows after cumulative
action scoring becomes positive and carries the last real score forward within
the game. Genuine opening `0-0` states are unchanged. This prevents false tied
possessions in the historical seasons.

## Comparison and uncertainty

The comparison group contains otherwise eligible same-clock, same-deficit
possessions with no team timeout before the first competitive action. For
those possessions, clock is measured at the preceding possession's terminal
event. This is a descriptive benchmark, not a causal estimate of timeout value:
timeout use is selected by coaches, game context, personnel, and advance status.

PPP intervals and ATO-minus-baseline intervals resample games as clusters.
The 1-7 second cells are often sparse, so every estimate must be read with its
possession count and interval. The `8-24` bucket is substantially more stable.

## Outputs

- `results/endgame_ato/endgame_ato_possessions_2014_2026.parquet`: eligible
  possession detail and flags.
- `results/endgame_ato/endgame_ato_by_clock_deficit_2014_2026.csv`: the full
  32-cell table with PPP, outcome rates, comparison values, and intervals.
- `results/endgame_ato/endgame_ato_by_deficit_2014_2026.csv`: deficit-level
  rollup.
- `results/endgame_ato/endgame_ato_quality_2014_2026.json`: structural data
  quality checks.
- `notebooks/endgame_ato_analysis.ipynb`: reproducible validation and charts.

## Rebuild

```bash
PYTHONPATH=nba_pipeline/scripts .venv/bin/python \
  nba_pipeline/scripts/analyze_endgame_ato.py \
  --start-year 14 --end-year 26 --season-types all --workers 4 \
  --bootstrap-reps 2000
```

