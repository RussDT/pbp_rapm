# Clock-Aware Rubberband Adjustment

## Decision

`rapm.py --rubberband` now uses a continuous score-margin-by-game-clock nuisance
surface with `alpha_rb=100`. It replaces the legacy 24 hard buckets, which remain
available as `--rubberband-model legacy` for reproduction.

This is a game-state adjustment inside RAPM, not a player metric. It estimates how
possession scoring changes when an offense is leading or trailing at a particular
point in the game, then keeps that systematic state effect out of the player
coefficients.

## State and basis

The model uses the offensive team's margin at the **start** of the possession.
Processed `score_margin` and `time_quarter` describe the end of the current row, so
the start state is the preceding row's end state within a game and period. A new
regulation period starts at 12:00 and an overtime period starts at 5:00.

The production basis has 80 continuous features:

- Separate leading and trailing surfaces; no forced symmetry.
- Piecewise-linear absolute-margin hinges at 0, 5, 10, 20, and 30 points, capped
  at 45.
- Regulation-progress hinges at game start, quarter boundaries, five minutes
  remaining, and two minutes remaining.
- Separate overtime level and within-overtime progress terms instead of folding
  overtime into the fourth quarter.
- Every feature is exactly zero at a tied score, which makes tie the identifiable
  reference state.

For side `s` (leading or trailing), the regulation surface is a tensor product:

```text
rubberband_s(margin, clock)
  = sum[j,k] beta[s,j,k] * margin_hinge[j] * time_hinge[k]
```

This is continuous in score margin and game time. It removes the arbitrary jumps
at the old ±5/±10/±15 boundaries while allowing late-game behavior to differ from
early-game behavior.

## Estimation

Player and rubberband coefficients are estimated in one generalized ridge system:

```text
minimize ||W^(1/2) (y - X_player b - X_rb g - X_nuisance d)||^2
       + alpha_off ||b_off - prior_off||^2
       + alpha_def ||b_def - prior_def||^2
       + alpha_rb  ||g||^2
```

Season and age fixed-effect columns remain unpenalized. Time-decay weights and
offense/defense priors enter the same objective. The implementation solves the
augmented sparse system with LSQR; it does not alternate approximately between
player and rubberband blocks.

The basis and penalty are fixed, but the fitted coefficient vector is not. One
surface is estimated over every season loaded into a particular solve. Thus a
single 1997-2026 run uses one pooled historical surface, while separate rolling
five- or six-season runs each learn their own local surface.

## Historical score and venue state

Pre-2018 processed possessions often store score only on scoreboard-update rows
and use `0-0` placeholders elsewhere. The transform now reconstructs cumulative
score by carrying observed score pairs forward and recovers venue side from
unambiguous score changes plus lineup continuity. This resolves approximately
`99.3%-99.8%` of possession orientations from 1997-2026; unresolved rows receive
neutral game-state context. This repair is required before interpreting early
rubberband or home-court effects.

## Model selection and forward evidence

The research harness used rolling five-season training windows and next-season
tests. Development seasons 2018-2021 selected among no adjustment, the legacy
bucket model, a possession-pressure model, and the smooth surface at penalties
100, 1,000, and 10,000. `smooth_a100` won development and was frozen before the
2022-2026 stress period was opened.

Across 6,572 stress-test games:

| Model | Possession MSE | State-calibration RMSE |
| --- | ---: | ---: |
| Smooth, alpha 100 | 1.3729070 | 0.02289 |
| Legacy buckets, alpha 10 | 1.3747992 | 0.03674 |
| No rubberband | 1.3751404 | 0.04165 |

The smooth model won in every stress season. Its game-clustered possession-MSE
improvement was:

- `+0.0022336 [0.0020922, 0.0023686]` versus no adjustment.
- `+0.0018924 [0.0017722, 0.0020101]` versus the legacy buckets.

Player ratings remain stable versus legacy: correlation `0.9975`, median absolute
net shift `0.059` points per 100, 90th percentile `0.235`, and maximum `1.031`
across 1,096 players.

## Is the curve constant since 1997?

No. A frozen 1997-2001 surface still improves every later era versus no
adjustment, proving that the underlying mechanism is persistent. But a trailing
five-season surface beats the frozen early curve in all three broad eras. Across
all 2002-2026 tests, rolling re-estimation improves possession MSE by
`+0.000503 [0.000474, 0.000530]` versus the fixed early curve; in 2018-2026 the
increment is `+0.000971 [0.000909, 0.001032]`.

The latest tested surface has `0.942` correlation with the early surface on a
dense state grid, with `1.71` points-per-100 RMS difference. The production rule
is therefore to keep the selected architecture and alpha constant while
re-estimating its coefficients inside the relevant rolling window. A single
1997-2026 solve remains a pooled historical nuisance adjustment, not an
era-correct curve. See [Rubberband Era Stability](./RUBBERBAND_ERA_STABILITY.md).

These are chronological historical forward tests, not untouched 2026-27 evidence.
They establish that the nuisance adjustment improves held-out possession fit and
state calibration without materially rewriting the player board. They do not
establish causal effort, "coasting," garbage-time player value, or future-game
forecast superiority by the full RAPM system.

## Usage and artifacts

```bash
# Production smooth model (alpha_rb defaults to 100)
python nba_pipeline/scripts/rapm.py RAPM 22 26 ALL \
  --pure --rubberband --season-effects --timedecay --half-life 700

# Reproduce the old model (alpha_rb defaults to 10)
python nba_pipeline/scripts/rapm.py RAPM 22 26 ALL \
  --pure --rubberband --rubberband-model legacy
```

Smooth runs retain the established `_rb` filename token and write:

- `*_rubberband_coefficients.csv`: fitted basis coefficients.
- `*_rubberband_effects.csv`: readable effects at representative margins and
  quarter midpoints.
- `*_rubberband_surface.csv`: a dense margin/clock surface for audit and plots.

Legacy runs use `_rb_legacy` so they cannot silently overwrite production smooth
outputs. The selection harness is
`nba_pipeline/scripts/research_rubberband_adjustment.py`; generated validation
tables live under `nba_pipeline/results/rubberband_research/` and remain research
artifacts.

For targets such as turnover rate, interpret coefficients in that target's solver
units. For example, TOV uses positive-good `Off_Diff = -Is_Turnover`, so a negative
rubberband effect means more turnovers.
