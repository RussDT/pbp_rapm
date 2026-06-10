# Clean Lineup Net-Rating Possessions

`scripts/build_clean_lineup_net_rating.py` starts a separate possession parquet
path for Databallr WOWY / lineup net rating. It intentionally does not replace
`processed/RAPM*.parquet`.

## Why This Exists

The RAPM possession parquet is solver-oriented. It can exclude or transform
events that are reasonable for RAPM, including expected free-throw treatment and
the existing standard possession boundary rule:

```text
EndOfPeriod is terminal only when prev_seconds > 0
```

That rule is not strict enough for raw on-court net rating. In 2026 line-by-line
audits, `EndOfPeriod` rows at `00:00` often failed to close the live possession,
which let the parser merge an end-of-quarter heave or miss into the next period.

## Current Duplicate Rule

The clean net-rating path:

- reads `raw_data/NBA*.parquet`
- runs the shared lineup propagation and standard possession parser
- treats every `EndOfPeriod` row as a hard segment boundary
- drops empty admin-only period-end segments with no live offensive event or
  scoring
- keeps actual action-score points from `Home_Action_Score` /
  `Away_Action_Score`, avoiding replay score-correction artifacts in raw
  scoreboard deltas
- preserves foreign-side technical/admin FT scoring as zero-possession
  `admin_free_throw` rows for the team that actually shot the FT
- emits one row per counted possession with `O1-O5`, `D1-D5`, `off_points`,
  `def_points`, `off_poss`, `def_poss`, and `row_type`

This preserves final buzzer possessions as zero-point possessions while avoiding
extra phantom possessions when a turnover, make, or defensive rebound already
closed the period.

## Commands

Audit a game line by line:

```bash
python nba_pipeline/scripts/audit_net_rating_possessions.py \
  --year 26 \
  --season-type RS \
  --game-id 0022500001
```

Build or smoke-test the duplicate possession surface:

```bash
python nba_pipeline/scripts/build_clean_lineup_net_rating.py \
  --year 26 \
  --season-type RS \
  --game-id 0022500001 \
  --no-write
```

Full-season output defaults to:

```text
nba_pipeline/processed/LINEUP_NET_RATING_POSSESSIONS26.parquet
```

Generated parquets and validation CSVs are artifacts and should stay out of git
unless explicitly curated.

## Initial Evidence

`0022500001` produced four period-boundary anomalies under the standard split:

- `poss_group 47`: period-1 `EndOfPeriod` at `00:00` was not terminal.
- `poss_group 92`: period-2 heave/rebound/end-period segment was merged with an
  out-of-order turnover event.
- `poss_group 185`: period-4 final miss/rebound/end-period segment merged into
  overtime and ended on an OT turnover.
- `poss_group 222`: final 2OT miss/rebound/end-period segment had no terminal
  possession under the standard split.

`0022500002` repeated the same shape:

- `poss_group 56`: 1Q buzzer miss/rebound/end-period merged into a 2Q turnover.
- `poss_group 154`: 3Q heave/rebound/end-period merged into the first made shot
  of 4Q.

A 2026 RS scan found `3,507` anomaly groups across `1,220` games using the
standard split, including `2,717` groups that crossed period boundaries. The
duplicate builder smoke tests on `0022500001` and `0022500002` both reconciled to
the final scoreboard with `period_change_rows = 0` and no zero-lineup rows.
