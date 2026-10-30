# WNBA Play-by-Play Shot Quality

`wnba_test/build_wnba_pbp_shot_quality.py` builds a repo-owned expected-points
model for WNBA field-goal attempts. It is the first WNBA equivalent of the
NBA play-by-play ShotQuality lane, but it is deliberately a research surface:
it has not been inserted into the live WNBA impact metric.

## What It Estimates

For every eligible midrange or three-point attempt, the model publishes four
nested predictions:

1. `shot_value_baseline_ev`: historical mean for two- or three-point attempts;
2. `shot_zone_ev`: baseline plus distance, zone, shot family, and court side;
3. `shot_context_ev`: zone EV plus shot-form and period-clock context;
4. `shot_quality_prior_ev`: context EV plus a strictly prior-season shooter
   residual.

The output is expected points on the attempt, not make probability. Dividing
by `shot_value` gives the implied make probability. Rim attempts are retained
in the source audit but excluded from tuning, validation, and impact-model
outputs.

## Source and Coverage

The current build reads the conservative, lineup-validated Gabriel normalized
files under `wnba_test/GabrielNormalizedClean/`.

| Season | Games | Source FGA | Modeled mid/three FGA |
| --- | ---: | ---: | ---: |
| 2022 | 239 | 32,511 | 21,662 |
| 2023 | 259 | 35,476 | 23,063 |
| 2024 | 261 | 35,629 | 22,923 |
| 2025 | 309 | 41,725 | 26,758 |
| 2026 | 207 | 28,304 | 17,977 |
| Total | 1,275 | 173,645 | 112,383 |

The 2022-2025 totals include regular season and playoffs; 2026 is regular
season through the current cutoff. More than 99% of attempts have positive
reported distance and all retained attempts have nonzero coordinate coverage.
Only `short_mid`, `long_mid`, `deep_two`, and `three` are eligible for model
use. Downstream scope is restricted to `ALT_EFG_MID_*` and
`ALT_EFG_THREE_*`; no shot-quality value may enter `ALT_EFG_RIM_*`.

## Outcome-Safe Features

The model can use only information observable when the shot is taken:

- two- versus three-point value;
- shot distance;
- short-mid, long-mid, deep-two, or three zone;
- dunk, layup, hook, floater, tip, or jumper family;
- period-clock bucket;
- description-derived shot-form modifiers such as driving, running, cutting,
  pull-up, step-back, turnaround, fadeaway, bank, putback, reverse, finger
  roll, floating, and tip.

The following fields are audit labels only and are forbidden from model
features:

- made/missed result;
- actual points;
- the raw `actiontype` value, because it contains Made Shot/Missed Shot;
- the full description string;
- made-only assist tags.

Every retained description modifier occurs on both makes and misses in all four
forward-scored seasons. The regression tests also verify that changing a
validation shot's outcome does not change its prediction.

## Model and Forward Protocol

The model is a hierarchical empirical-Bayes estimator:

```text
shot-value baseline
  -> smoothed zone/geometry EV
  -> smoothed granular context EV
  -> smoothed prior-season shooter residual
```

Smoothing was selected only on the 2023, 2024, and 2025 season-forward folds.
The selected constants are:

| Layer | Shrinkage |
| --- | ---: |
| Zone/geometry | 150 |
| Granular context | 75 |
| Shooter residual | 250 |

The 2026 fold was excluded from smoothing selection. Its model was trained on
2022-2025 only.

## Forward Results

The full context-plus-prior-shooter model improves mean squared error over the
shot-value baseline in every forward season:

| Validation season | MSE improvement |
| --- | ---: |
| 2023 | 0.875% |
| 2024 | 0.863% |
| 2025 | 0.882% |
| 2026 | 1.068% |

Detailed 2026 results:

| Model | RMSE | R² | MSE gain vs baseline |
| --- | ---: | ---: | ---: |
| Shot-value baseline | 1.255962 | 0.005799 | — |
| Zone/geometry | 1.252714 | 0.010935 | 0.517% |
| Full context | 1.250690 | 0.014127 | 0.838% |
| Context + prior shooter | 1.249239 | 0.016414 | 1.068% |

The prior-season shooter layer adds another `0.232%` MSE improvement over the
context model on 2026. Geometry, shot-form context, and shooter history each
contribute modestly.

## RTSS Teacher/Student Follow-Up

`train_wnba_rtss_pbp_student.py` distills provider `qSQ` into a model that uses
only the outcome-safe PBP feature contract. The exact game/event join validates
61,521 MID/THREE teacher shots from 2023-2025. The standalone student does not
beat this PBP model, but a student/context blend selected on 2024 improves 2025
all-PBP MSE by `0.001247`, with a paired game-clustered interval of
`[+0.000151,+0.002282]`. The analogous shooter-inclusive blend is positive but
inconclusive. This is development evidence only; RTSS 2026 remains sealed and
the live WNBA impact metric is unchanged. See
[WNBA RTSS-to-PBP ShotQuality Student](./WNBA_RTSS_PBP_SHOT_QUALITY.md).

## Historical Sample Audit

Gabriel classic PBP adds 345,314 field-goal attempts from 2010-2021, including
239,829 attempts classified into the same MID/THREE target scope. Player IDs
and action codes are complete; shot-family descriptions cover essentially
100% of attempts. Explicit distance coverage rises from `58%` in 2010 to `93%`
in 2021.

More rows did not automatically produce a better current model:

| Data window | Eligible attempts through 2026 | Pre-2026 forward folds | 2026 RMSE | 2026 gain vs baseline |
| --- | ---: | ---: | ---: | ---: |
| Modern 2022-2026 | 112,383 | 3 | 1.249239 | 1.068% |
| 2018-2026 expansion | 185,460 | 7 | 1.249319 | 1.055% |
| 2010-2026 expansion | 352,186 | 15 | 1.251520 | 0.768% |

On the common 2023-2025 development seasons, the 2018 expansion had slightly
lower pooled MSE than modern-only (`1.502949` versus `1.504076`). On 2026 it
was essentially tied but worse by `0.000079` RMSE. The full 2010 expansion was
clearly worse because old description/context distributions do not transfer
cleanly into the modern feed.

Therefore the default artifact remains modern-only. Historical input is an
opt-in comparison, not a way to inflate sample size:

```bash
# Relevant-era comparison
.venv/bin/python wnba_test/build_wnba_pbp_shot_quality.py \
  --include-classic \
  --classic-start-year 2018 \
  --output-dir wnba_test/Results/pbp_shot_quality_2018_comparison

# Full-history stress test
.venv/bin/python wnba_test/build_wnba_pbp_shot_quality.py \
  --include-classic \
  --classic-start-year 2010 \
  --output-dir wnba_test/Results/pbp_shot_quality_all_history_comparison
```

Do not promote a larger historical window unless it beats the modern-only
model on a future independent season. More dependent or shifted rows are not
a substitute for more independent validation seasons.

## Build

```bash
.venv/bin/python wnba_test/build_wnba_pbp_shot_quality.py
```

Generated artifacts under `wnba_test/Results/pbp_shot_quality/`:

- `forward_validation.csv`
- `smoothing_grid.csv`
- `wnba_pbp_shot_quality_forward.parquet`
- `wnba_pbp_shot_quality_2026.parquet`
- `shooter_season_summary.csv`
- `manifest.json`

The forward parquet contains 90,721 eligible attempts from 2023-2026. Generated
data artifacts remain gitignored; the script, tests, and methodology docs
belong in git.

## What It Does Not Yet Prove

This validates an attempt-level expected-points model, not an incremental player
impact metric. Before promotion, the EV must be attached to the clean
first-chance possession rows and split into:

```text
midrange/three shot-quality opportunity value
  + actual-minus-expected shotmaking residual
```

That split must close exactly to the existing MID and THREE shot targets while
leaving RIM untouched, then survive the same season-forward component-sum test
used for WNBA lineup-factor alpha selection. The live `5000/40000/1.5` WNBA
board remains unchanged until that forward gate is passed.
