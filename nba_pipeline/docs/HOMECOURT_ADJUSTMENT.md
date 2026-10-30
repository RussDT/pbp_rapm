# Home-Court RAPM Adjustment

## Decision

A centered global home-offense nuisance term adds small, consistent forward
value beyond the smooth rubberband model from 1997 onward. It is a promotion
candidate for rolling/current RAPM, not yet a production default.

The home coefficient is not constant. In trailing-five-season fits after the
rubberband adjustment, it declines from about `+4.85` points per 100 in the
2002-2009 test era to `+4.31` in 2010-2017 and `+2.95` in 2018-2026. A single
long-span coefficient is therefore a historical average, not an era-correct
home effect.

The term represents the home team's scoring advantage per possession after
lineup and game-state adjustment. It is context, not player quality, and it does
not identify whether the mechanism is offense, defense, officiating, crowd,
travel, familiarity, altitude, or another channel.

## Specification

For possessions with resolved venue orientation:

```text
home_context = is_home_offense - resolved training home-possession share
```

Unresolved possessions receive zero home context. Centering makes the fitted
coefficient the home-minus-away scoring contrast and prevents the unpenalized
context column from acting as an intercept against penalized player columns.

The coefficient is estimated in the same joint generalized-ridge system as
offense, defense, and the smooth alpha-100 rubberband surface. The home term is
global and unpenalized; there are no team-, arena-, rest-, travel-, or
altitude-specific interactions.

## Historical venue repair and audit

Older processed possessions often store cumulative score only on scoreboard
updates and use `0-0` placeholders on other rows. The previous transform treated
those placeholders as score resets, making historical `score_diff` and
`home_poss` unusable as context labels.

The transform now carries each observed score pair forward, seeds venue identity
only from exclusive home or away score changes, and propagates player identity
through lineup continuity. Audited 1997-2026 seasons now resolve approximately
`99.3%-99.8%` of possession orientations. Resolved home-offense shares are near
`49.7%-49.9%`, including the formerly invalid 2013-2017 seasons.

The repair changes context labels only. Offense/defense player assignment was
already preserved because lineup side and the `home_poss` sign moved together.

## Forward design

For every test season from 2002 through 2026, each fold trains on the preceding
five seasons and predicts the same next-season possessions under:

1. Plain pure RAPM.
2. Plain RAPM plus home.
3. Smooth rubberband RAPM.
4. Smooth rubberband RAPM plus home.

Inference is paired and clustered by game. All seasons are now spent historical
evidence rather than untouched confirmation.

## Results

Adding home after the smooth rubberband model improves held-out possession MSE
in every broad era:

| Test era | Mean fitted home effect | MSE improvement vs smooth |
| --- | ---: | ---: |
| 2002-2009 | +4.85 pts/100 | +0.000537 `[0.000461, 0.000610]` |
| 2010-2017 | +4.31 pts/100 | +0.000339 `[0.000272, 0.000407]` |
| 2018-2026 | +2.95 pts/100 | +0.000120 `[0.000076, 0.000167]` |
| 2002-2026 | — | +0.000319 `[0.000284, 0.000356]` |

The effect declines smoothly across trailing windows. The smooth-plus-home
coefficient is `+4.44` for the first 1997-2001 training window and `+2.33` for
the latest 2021-2025 window.

The MSE gain is small relative to the rubberband gain, but residual calibration
improves materially. Across 2010-2026 folds, mean absolute home/away residual
bias falls from `3.12` to `0.58` points per 100. Player estimates remain stable:
correlation `0.99928`, median absolute net shift `0.042` points per 100, 90th
percentile `0.136`, and maximum `0.315` across 1,096 players in the latest fold.

## Status and next gate

The evidence supports a centered global home term inside time-local rolling
RAPM windows. Production should neutralize unresolved rows, export the fitted
home coefficient, and keep the term out of player credit. A single 1997-2026
coefficient may be retained only as a pooled historical nuisance average.

This result does not test rest, travel, altitude, arena, team-specific home
effects, or a complete game forecast. Those require date-correct schedule data
and separate forward ablations.

The reproducible harness is
`nba_pipeline/scripts/research_homecourt_adjustment.py`. Generated tables and the
full report live under `nba_pipeline/results/homecourt_research/`.
