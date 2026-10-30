# WNBA box-estimated rZTOV

## 2026-09-05 canonical TSA correction

The emitted self-created and assisted volumes now partition canonical PBP TSA100,
not `FGA + 0.44*FTA`. The role models retain their existing predictor definitions
and fitted share/potential-assist estimates; only the shot-volume basis changes.
Turnover allowances, involvement denominators, lane centers, and derived rZTOV
are rebuilt on the corrected volumes. Earlier numerical receipts below describe
the pre-correction build; use the regenerated manifest and CSVs for current values.
This does not promote the research series or update a public database.

WNBA uses `FG2A + FG3A + TwoPtShootingFoulsDrawn +
ThreePtShootingFoulsDrawn - 2pt And 1 Free Throw Trips -
3pt And 1 Free Throw Trips + NonShootingFoulsDrawn`, divided by OffPoss and
multiplied by 100. Required source columns must exist; no 0.44 fallback is allowed.
The observed-share price-training volumes use this basis as well. Current rebuilt break-evens are 14.9394% self-created, 6.6840% assisted, and 18.5934% potential assist. All 890 exported player-seasons close to the source TSA100 within `7.2e-15`.

## Status

Research only. `wnba_test/build_wnba_rztov.py` produces the first WNBA rZTOV
surface from public PBP Stats-style box/PBP fields. RTSS supplies 2023-2025
training labels, but no tracking field is required when scoring a new season.
The output does not change the live WNBA VPM board.

The public Supabase results hub exposes the current derived board as
`wnba_box_estimated_rztov_2026` (229 rows) and the consolidated descriptive
history as `wnba_box_estimated_rztov_2022_2026` (890 player-seasons). The
2023-2025 RTSS label and price-fit rows remain development inputs and are not
public result rows.

Negative rZTOV is good: the player committed fewer turnovers than the estimated
allowance bought by her scoring and passing workload.

## Estimating the missing workload

Ordinary WNBA box scores do not contain self-created attempts, potentially
assisted attempts received, or potential assists created. The existing RTSS
feature surface does:

- `rtss_shot_assist_opp_share` identifies the share of attempts carrying an
  assist opportunity;
- `rtss_potential_assist_rate100` identifies potential assists created;
- `shots.assistOpp`, rather than made-only official assists, is the tracking
  contract for the shot split.

Two compact ridge models map stabilized PBP Stats-style player-season fields
onto those labels. Alpha is selected by symmetric leave-one-season-out
development over 2023-2025. The fit uses 423 player-seasons with at least 250
RTSS offensive chances.

| Estimated label | LOSO RMSE | LOSO R2 | LOSO correlation |
| --- | ---: | ---: | ---: |
| Potentially assisted share | `0.0773` | `0.558` | `0.747` |
| Potential assists / 100 | `0.7168` | `0.950` | `0.975` |

The shot estimates obey a hard box-score identity:

```text
estimated self-created TSA/100
+ estimated assisted TSA/100
= canonical TSA/100

canonical TSA = FGA + non-and-one shooting foul trips + nonshooting foul trips
```

The model predicts the assisted share, then divides known box TSA by that
share. This is preferable to independently predicting two volumes that can
drift away from a player's actual shot workload. Potential assists remain a
separate prediction anchored mainly by assists, assist points, rim assists,
bad-pass turnovers, total turnovers, TSA, and usage.

These are estimated opportunity types, not official made-shot assists. The
assisted bucket means the shot carried a potential-assist opportunity.

## WNBA turnover prices

The builder reuses the annual WNBA clean turnover targets under
`wnba_test/Results/box_prior_true_prior/annual_targets/`. Because the total and
bad-pass targets use the same linear solve contract, scoring-turnover impact is
the exact coefficient difference:

```text
scoring target = total TOV target - bad-pass TOV target
```

One joint no-intercept alternating-least-squares fit prices the 2023-2025
observed RTSS roles against scoring, bad-pass, and total offensive turnover
impact. The fitted WNBA break-evens are:

| Workload | Turnovers allowed per opportunity |
| --- | ---: |
| Self-created TSA | `14.88%` |
| Assisted TSA | `6.71%` |
| Potential assist | `18.59%` |

The total-equation fit uses 294 qualified player-seasons. Its `R2` is `0.271`;
the scoring and bad-pass lane values are `0.112` and `0.075`. Those modest lane
fits are an important limitation: the volume estimators are useful, while the
exact WNBA turnover prices remain noisy with only three tracking seasons.

## Rate and lane accounting

For each season, scoring and bad-pass excess are centered separately so their
possession-weighted league means are zero. The two measured lanes then close:

```text
scoring excess = scoring turnovers - scoring allowance
bad-pass excess = bad-pass turnovers - bad-pass allowance
turnover excess = scoring excess + bad-pass excess

plays = self-created TSA + assisted TSA + potential assists + turnovers
rZTOV% = 100 * turnover excess / plays
turnovers saved / 100 = -turnover excess
```

The generated board also carries `ztov_pct` as a direct alias for consumers
that use the shorter zTOV label. Preserve the negative-good sign convention.

## Historical 2022-2026 contract

The compatible local PBP Stats-style box/PBP panel begins in 2022, so that is
the honest start of the consolidated history. One fixed role model trained on
the 2023-2025 RTSS labels and one fixed turnover-price fit estimated on the
same three seasons score every year. The result is deliberately labeled by
season:

| Seasons | `historical_score_status` | Interpretation |
| --- | --- | --- |
| 2022 | `pre_training_retrospective_score` | backcast before the role/price training window |
| 2023-2025 | `training_period_retrospective_score` | in-sample descriptive estimate |
| 2026 | `current_unlabeled_score` | current box score without observed RTSS confirmation labels |

This is a comparable descriptive workload/rZTOV history, not a date-safe
walk-forward series and not observed tracking history. The fixed fit avoids
silently changing definitions across the five seasons, but it must not be used
as evidence that the 2023-2025 model would have been available in 2022.

The generated history has 890 player-seasons and 461 qualified rows. Per-season
row counts are 166, 156, 157, 182, and 229 from 2022 through 2026. Player-season
keys are unique, self-created plus assisted TSA closes exactly to box TSA, and
both turnover lanes are separately centered within every season.

## August 24, 2026 research snapshot

The current box panel produces 229 player rows and 103 qualified rows at 1,000
offensive possessions plus 10 workload involvements per 100. Selected examples:

| Player | Self-created | Assisted | Potential assists | TOV allowance | rZTOV% |
| --- | ---: | ---: | ---: | ---: | ---: |
| Veronica Burton | `11.82` | `8.90` | `13.92` | `4.83` | `-5.34` |
| Paige Bueckers | `16.04` | `9.95` | `11.59` | `5.10` | `-4.28` |
| A'ja Wilson | `21.62` | `11.15` | `6.97` | `5.15` | `-2.95` |
| Breanna Stewart | `15.44` | `12.12` | `6.87` | `4.28` | `-2.76` |
| Caitlin Clark | `22.86` | `5.74` | `16.28` | `6.70` | `+0.86` |

All workload and allowance fields in the table are per 100 offensive
possessions. The current ranking is descriptive. The 2023-2025 RTSS seasons
were used for model and price development, and the current 2026 row set does
not provide observed RTSS labels for a clean confirmation of the box bridge.

## Reproduction and artifacts

```bash
.venv/bin/python wnba_test/build_wnba_rztov.py
.venv/bin/python -m pytest -q nba_pipeline/tests/test_wnba_rztov.py
```

Generated, gitignored artifacts live under
`wnba_test/Results/wnba_rztov/`:

- `wnba_rztov_2026.csv`;
- `wnba_box_estimated_rztov_2022_2026.csv`;
- `label_model_loso_folds.csv`;
- `label_model_loso_predictions.csv`;
- `break_even_training_rows.csv`;
- `manifest.json`.

The build hard-checks unique historical player-season keys and the
self-created-plus-assisted TSA identity. Tests also verify the turnover-lane
identity, the rate formula, fixed-fit season labels, and separate league
centering of scoring and bad-pass excess. Do not extend the history before 2022
without first producing the same box/PBP feature contract, and never concatenate
RTSS training labels into the reader surface.
