# WNBA LUMA Historical RAPM Bridge

Last updated: August 24, 2026

## Status

`wnba_test/build_luma_historical_metrics.py` builds a retrospective,
LUMA-compatible WNBA RAPM decomposition for every season from 2003 through
2026. It is the historical companion to the modern event-sequenced WNBA
DECOMP, not a replacement for it.

The completed build contains:

- 5,623 regular-season and playoff games;
- 267,293 source stints;
- 292,446 retained offense-side observations after strict validation;
- 3,922 annual player rows and 973 unique LUMA player ids;
- annual no-prior `PURE_RAPM` plus true-prior `LUMA_ORAPM`, `LUMA_DRAPM`, and
  `LUMA_VPM`;
- offensive, positive-good defensive, and net values for shot value, turnover
  value, and offensive-rebound opportunity value;
- exact eight-child row and fitted-player closure.

Generated artifacts are gitignored under
`wnba_test/Results/luma_historical_our_priors/`. The canonical files are:

- `luma_historical_metrics_2003_2026.csv`;
- `annual/luma_historical_metrics_YEAR.csv`;
- `annual_component_targets.csv`;
- `comparison_to_event_decomp.csv`;
- `manifest.json`.

## Source Contract

The bridge reads LUMA's annual `data/stints/rs_YEAR.json` and
`po_YEAR.json` files. Each source stint supplies two five-player lineups,
duration, points, and a 16-field team tally. The supported tally information is
rim/mid/three attempts and points, free throws, turnovers, and offensive
rebounds.

The default build retains a team-side observation only when:

- both lineups contain five unique players with no cross-team overlap;
- duration is positive;
- every tally is finite and nonnegative;
- tally points equal the source score;
- pass, handle, and system turnovers sum to total turnovers;
- Oliver possessions are positive.

The full build excludes 242,140 candidate side rows before solving. Most are
zero-duration, zero-action source fragments; the manifest records mutually
exclusive counts by season and reason. `--allow-zero-seconds` is available as
a sensitivity, but the canonical build uses the conservative positive-duration
contract.

## Exact Compatible Algebra

For each season, define:

```text
poss = FGA + 0.44*FTA + TOV - OREB
b    = league points / league possessions
r,m,t = league points per rim, mid, and three attempt
```

The point numerators are:

```text
RIM_FREQ  = (r - m) * rim_FGA
RIM_FG    = rim_points - r * rim_FGA
MID_BASE  = m*(rim_FGA + three_FGA) + mid_points - b*FGA
THREE_FREQ = (t - m) * three_FGA
THREE_FG   = three_points - t * three_FGA
FT_VALUE   = FT_points - 0.44*b*FTA
TOV_VALUE  = -b*TOV
OREB_OPPORTUNITY_VALUE = b*OREB
```

Therefore, for every retained row:

```text
sum(eight components) = actual points - b*poss
```

The same design matrix, possession weights, side penalties, and additive prior
are used for all targets. Ridge is linear, so the eight fitted player
coefficients also sum to the separately fitted total. The completed build's
maximum row error is `5.33e-15`; maximum player-component error is `4.00e-14`.

Reader parents are:

```text
TS    = RIM_FREQ + RIM_FG + MID_BASE + THREE_FREQ + THREE_FG + FT_VALUE
TOV   = TOV_VALUE
REB   = OREB_OPPORTUNITY_VALUE
TOTAL = TS + TOV + REB
```

Defense is converted from raw points allowed to positive-good defense before
publication. All values are points per 100 possessions.

## Prior Contract

The build is two-pass:

1. Fit low-alpha annual component targets at offense alpha `750` and defense
   alpha `1000`.
2. For season Y, train component/side Ridge mappers only on target seasons
   before Y, predict from season-Y stabilized PBP Stats player features, and
   use those predictions as the center of the annual `5000/40000` solve at
   prior scale `1.5`.

The feature surface is the exact contract used by
`build_wnba_box_priors.py`, not LUMA Box+:

- offense uses smoothed zone frequency/accuracy, shot and FT volume, usage,
  assist points, rim assists, detailed turnover types, self OREB, and fouls
  drawn as appropriate for each component;
- defense uses smoothed steals and steal types, blocks and blocked-shot types,
  defensive rebounds, fouls, and shooting fouls as appropriate for each
  component;
- offensive rates receive 250 prior possessions, defensive rates 400, and
  zone shares/accuracy use the same attempt smoothing as the modern build;
- target support uses the square root of clipped target possessions;
- mapper alpha is selected separately for every component/side from
  `0.1,1,10,100` by expanding-season RMSE, with correlation as the tie break;
- mapper predictions are clipped to the historical target's 1st/99th
  percentiles.

Archived PBP Stats player totals under
`nba_pipeline/external/gabriel_wnba/data/` provide the complete feature schema
for 2009-2021. The canonical backend totals provide 2022 onward. Stable ids are
used where LUMA supplies them; otherwise an audited, season-specific normalized
name join is used. Feature coverage is 96.9% of LUMA possessions in 2009,
95.9% in 2010, at least 98% by 2015, and effectively complete from 2021 onward.

There is no feature file for 2003-2008, and 2009 has no earlier feature season
from which to train. Those seasons are explicit zero-prior warm-ups;
component-specific true priors begin in 2010. `PURE_RAPM` remains available at
alpha `3000` for every season.

This is a retrospective descriptive board because season-Y player features
use the completed/current season. It is not a preseason forecast. The
`5000/40000`, target alphas, and scale `1.5` come from the modern WNBA
architecture; they have not yet been reselected on a LUMA-specific summed
future-possession gate.

## Relationship to Modern Event DECOMP

Aggregate stints cannot identify event order. Consequently this series cannot
recover:

- first chance versus direct second chance;
- `SECOND_CHANCE_CLEAN` or net-continuation kappa;
- ShotQuality expected value and shotmaking residuals;
- event-level FT trip frequency/severity or direct turnover children.

`OREB_OPPORTUNITY_VALUE` is the exact Oliver-possession opportunity credit. It
must not be renamed `SECOND_CHANCE_CLEAN` or stacked with the modern event
component.

On players with at least 1,000 possessions in both families, the corrected
PBP-feature LUMA VPM correlates with modern event VPM at `0.656`, `0.964`,
`0.965`, `0.977`, and `0.968` for 2022 through 2026. In 2026 its net standard
deviation is `2.34`, versus `2.35` for the event board; defense is `0.95` versus
`0.82`. The deprecated generic-LUMA-rate prototype had net/defense standard
deviations of only `1.71/0.23` and correlation `0.806`.

This close contemporaneous agreement verifies feature/scale reproduction, not
future predictive admission, and it does not make the factor definitions
interchangeable. The descriptive comparison file separately labels rebound
opportunity versus observed second-chance value. The deprecated prototype is
retained under `wnba_test/Results/luma_historical/` only as an audit artifact.

## Reproduction

```bash
.venv/bin/python wnba_test/build_luma_historical_metrics.py \
  --luma-root /path/to/LUMA/WNBA \
  --start-year 2003 \
  --end-year 2026 \
  --season-type ALL \
  --output-dir wnba_test/Results/luma_historical_our_priors

.venv/bin/python -m pytest \
  nba_pipeline/tests/test_wnba_luma_historical_metrics.py -q
```

The manifest pins every source JSON and the optional modern comparison board
by SHA-256, records exclusions and baselines by season, and exposes every
hyperparameter and prior-training year.
