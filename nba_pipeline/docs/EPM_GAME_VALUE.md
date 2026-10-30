# EPM Game and Season Value Calibration

`scripts/build_epm_game_value.py` owns the empirical EPM-to-game and
EPM-to-season conversion used by the From Impact to Winning explainer. It
replaces hand-entered pace and win constants with date-stamped EPM plus exact
repo-owned player exposure.

This is a standardized value translation, not a matchup forecast and not an
official Dunks & Threes `eWINS` reproduction.

## Accounting contract

Keep the additive layer linear:

```text
game margin = EPM * actual player possessions per game / 100
season wins vs average = EPM * actual season possessions / 100 * wins per point
modeled WAR = (EPM - replacement EPM) * actual season possessions / 100 * wins per point
```

Only expected margin to win probability is nonlinear:

```text
neutral win probability = logistic(k * expected game margin)
```

The player possession count is the average of his offensive and defensive
possessions on court. This avoids double-counting the same team possession as
both one offense and one opposing defense. Do not substitute a fixed
possessions-per-minute constant.

## Current-player inputs

- EPM: final 2025-26 regular-season snapshot
  `cache/epm/epm_all_2026-04-15_go0.csv`, requested and provider-effective on
  April 15, 2026.
- Minutes: exact between-event clock intervals from `raw_data/NBA26.parquet`.
  Events are sorted clock-first, then by event number; the final event at one
  clock supplies the post-substitution lineup for the next interval.
- Possessions: exact lineup exposure from
  `processed/LINEUP_NET_RATING_POSSESSIONS26.parquet`.

The minute reconstruction totals `593,500` player-minutes, exactly ten times
the game-clock total including overtime.

## Reader universe

The public framework table uses every player who joins the final EPM snapshot
to at least one positive raw-PBP lineup interval: `582` players across all 30
teams. The remaining 20 provider snapshot rows have no positive lineup
exposure in `NBA26.parquet`; they stay outside the realized-value table rather
than receiving fabricated GP, minutes, possessions, game margin, or WAR.

The site keeps the smaller enriched player set only for worked article,
salary, contract, and playoff-placeholder examples. The framework leaderboard
uses the full 582-player derived surface, with unavailable contract/playoff
fields left blank.

## Historical calibration

The build reconstructs 14,437 regular-season games from 2013-14 through
2024-25. Season wins are regressed on accumulated point differential through
the natural `.500` origin:

```text
wins - GP / 2 = beta * season point differential
```

The pooled result is:

- `0.0297496` wins per accumulated point;
- `33.6139` accumulated points per win;
- season-cluster bootstrap 95% interval: `[0.0284425, 0.0311566]` wins per
  point.

For the game curve, each team's final regular-season point differential per
game is computed with the focal game removed. A 2013-14 through 2021-22 fit
maps the home-away strength difference into expected margin, then calibrates
the probability link. The 2022-23 through 2024-25 seasons are held out.

```text
expected home margin = 2.35186 + 0.88810 * LOO strength difference
P(win | expected margin M) = logistic(0.144482 * M)
```

The logistic slope's season-cluster bootstrap 95% interval is
`[0.134117, 0.154005]`. At neutral margin its derivative is `3.61` percentage
points per expected point.

Held-out scoring:

| Mapping | Brier | Log loss |
|---|---:|---:|
| Logistic | 0.212738 | 0.613242 |
| Probit | **0.212722** | **0.613224** |
| Global linear, clipped | 0.212973 | 0.628447 |

Logistic and probit are effectively tied. Logistic is retained because its
single slope is easier to communicate. The previous prototype `k=0.16` was
too steep.

## Should NET WIN% be linear?

Use a linear quantity for the additive headline if desired: **Game +/-** is
already exactly linear and has an honest unit (expected points of margin).

A linear NET WIN% is a reasonable local approximation around 50%. The fitted
slope is `0.0308148` probability per expected point, or about `3.08` percentage
points. But it is not a coherent global probability: it eventually leaves
`[0,1]`, requires clipping, and loses held-out log loss. The recommended reader
surface is therefore:

- show `Game +/-` as the additive, linear player contribution;
- show calibrated logistic `NET WIN%` as the bounded probability translation;
- never add player NET WIN% values across a lineup.

## Real equal-WAR examples

The table below uses exact 2025-26 exposure, the calibrated win conversion,
and an explicit `-2.0 EPM` replacement assumption. The replacement assumption
is not estimated by this build.

| Player | EPM | MPG | GP | Modeled WAR | Game +/- vs 0 EPM alternative | Neutral NET WIN% |
|---|---:|---:|---:|---:|---:|---:|
| Giannis Antetokounmpo | +6.93 | 28.88 | 36 | 5.66 | +4.10 | 64.4% |
| Austin Reaves | +3.31 | 34.23 | 51 | 5.70 | +2.34 | 58.4% |
| Alex Caruso | +2.23 | 18.23 | 56 | 2.66 | +0.84 | 53.0% |
| DeMar DeRozan | -0.23 | 31.23 | 77 | 2.63 | -0.15 | 49.5% |

The equal generic WAR is not a tie for a contender. With actual alternative
quality `b`, roster-relative margin is `(EPM - b) * possessions`. For an
equal-WAR pair defined against replacement `R = -2`:

- if `b > -2`, the higher-impact, concentrated player is more valuable;
- if `b = -2`, the pair ties by construction;
- if `b < -2`, the volume player's extra minutes can be more valuable.

That is the precise version of “regular-season volume accumulates; playoff
quality concentrates.”

## Replacement caveat

The EPM API and historical score calibration do not identify replacement
level. `-2 EPM` remains an explicit sensitivity parameter until replacement is
defined against a reproducible availability pool (transactions, two-way and
minimum-contract players, or another predeclared labor-market rule). Tables
must say “modeled WAR at a -2 baseline,” not imply that this is official EPM
WAR.

## Rebuild

```bash
PYTHONPATH=. .venv/bin/python \
  nba_pipeline/scripts/build_epm_game_value.py
```

Generated CSV/JSON outputs live under `results/epm_game_value/` and remain
derived artifacts. Publish only the curated player rows and calibration
constants needed by the reader, never the raw provider snapshot.
