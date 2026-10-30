# WNBA RTSS qSQ/qSP Challengers

## Status

Research-only development result. Nothing here changes the live WNBA rating,
the frozen RTSS component-prior confirmation, or the repo-owned PBP shot-
quality target. Season 2026 remains sealed.

## Exact provider fields and the scale correction

The RTSS `shots` table carries two provider scores:

- `qSQ`: shot-context quality for a league-average shooter, per the current
  project semantic contract;
- `qSP`: the corresponding score after including the shooter.

The raw values empirically live on an expected-effective-field-goal scale,
not a direct make-probability scale. The decisive audit is three-point shots:
raw `qSQ` averages about `.49-.50` in every season while actual 3P% is about
`.33-.34`. The calibrated make-probability interpretation is therefore:

```text
provider expected points = 2 * q
provider make probability = 2 * q / shot value
```

For a two this leaves `q` unchanged; for a three it multiplies `q` by `2/3`.
This is an empirical scale identification, not a substitute for missing
provider documentation. The older RTSS prior surface multiplied three-point
`q` by `3`; its feature-standardization makes the frozen Ridge experiments
largely scale-invariant, but its raw expected-points labels should not be
reused as calibrated shot predictions.

## Data and feature contract

`wnba_test/train_wnba_rtss_q_models.py` reads all 761 local 2D markings games
from 2023-2025:

| Season | Games | FGA | Sanctioned MID/3 FGA |
| --- | ---: | ---: | ---: |
| 2023 | 173 | 25,460 | 15,496 |
| 2024 | 266 | 39,161 | 23,555 |
| 2025 | 322 | 47,335 | 27,835 |

The MID/3 definition exactly matches the rich-tracking audit: every three plus
non-three `jumper`, `post`, or `floater`. RIM remains outside the current
WNBA impact target even though it is valuable for provider calibration.

Allowed release-time 2D features are provider q, shot value/distance/location,
clock with a missingness flag, region and shot family, dribbles/catch-and-
shoot/creation state, shooter motion, and defender distance/contest state.
Same-row outcome, realized assist, block, foul, rim result, arc, pose, and
trajectory fields are forbidden. The qSP challenger adds only a 250-attempt-
shrunk shooter residual versus qSQ and exposure, computed from completed prior
games; the entire current game is excluded.

The fixed regularized histogram-gradient-boosting model is scored in two
chronological folds:

- fit 2023, score 2024;
- fit 2023-2024, score 2025.

All comparisons use identical shots and paired game-clustered bootstrap
intervals. Negative challenger-minus-provider loss is better.

## Results

### All field-goal attempts

| Fold | Benchmark | Provider Brier | Challenger Brier | Delta | 95% game-clustered CI |
| --- | --- | ---: | ---: | ---: | ---: |
| 2024 | qSQ | `.200193` | `.199294` | `-.000899` | `[-.001433, -.000381]` |
| 2025 | qSQ | `.201642` | `.199431` | `-.002210` | `[-.002609, -.001811]` |
| 2024 | qSP | `.198063` | `.197766` | `-.000297` | `[-.000871, +.000291]` |
| 2025 | qSP | `.199719` | `.197602` | `-.002117` | `[-.002504, -.001714]` |

The conservative rim-gated variants leave the provider prediction untouched
off-rim and apply the learned correction only at the rim. Their Brier deltas
are `-.001155` / `-.002148` against qSQ and `-.000940` / `-.001946` against
qSP in 2024 / 2025. This is the most interpretable stable candidate.

The gain is overwhelmingly rim calibration. On 2025 rim attempts:

- actual FG%: `.4756`;
- provider qSQ-implied FG%: `.5118`;
- context challenger: `.4781`;
- qSQ Brier: `.177915 -> .172990`.

On the full 2025 qSQ ablation, cumulative Brier gain versus the raw provider
is `-.000386` from anchor recalibration, `-.001432` after geometry, `-.001631`
after shot clock, `-.002004` after creation/action context, and `-.002210`
after contest/defender geometry. The ordering is descriptive, but physical
geometry is the largest incremental block. Adding provider qSP to the full
context model supplies most shooter gain; the repo-owned prior-game shooter
history adds only another `-.000119` Brier.

### Sanctioned MID/3 scope

The result is deliberately negative/cautious for the current impact lane:

| Fold | Benchmark | Provider Brier | Challenger Brier | Delta | 95% game-clustered CI |
| --- | --- | ---: | ---: | ---: | ---: |
| 2024 | qSQ | `.218078` | `.219256` | `+.001178` | `[+.000479, +.001872]` |
| 2025 | qSQ | `.219600` | `.219758` | `+.000158` | `[-.000224, +.000547]` |
| 2024 | qSP | `.216313` | `.217128` | `+.000815` | `[+.000060, +.001531]` |
| 2025 | qSP | `.218170` | `.217885` | `-.000285` | `[-.000712, +.000173]` |

Provider qSQ/qSP already capture MID/3 context well. The richer challenger
does not beat either provider benchmark consistently and earns no promotion.
qSP itself does improve materially over qSQ in both folds, which says the
provider shooter layer contains real signal; our extra empirical shooter
history adds little.

## Decision

There is a real, forward-repeatable improvement available for provider-wide
qSQ/qSP, but it is a rim calibration correction. There is no demonstrated
upgrade yet for the sanctioned WNBA MID/3 impact target. Keep the current live
metric unchanged, retain the rim-gated correction as a frozen research
candidate, seek provider confirmation of the q scale/timing contract, and
only evaluate 2026 after the existing one-shot RTSS confirmation has fired.

## Reproduction

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python wnba_test/train_wnba_rtss_q_models.py --workers 4 --rebuild
```

Ignored generated artifacts are written under
`wnba_test/Results/rtss_q_models/`:

- `coverage.csv`;
- `forward_metrics.csv`;
- `paired_game_clustered_comparisons.csv`;
- `forward_predictions.parquet`;
- `development_shots.parquet`;
- `manifest.json`.
