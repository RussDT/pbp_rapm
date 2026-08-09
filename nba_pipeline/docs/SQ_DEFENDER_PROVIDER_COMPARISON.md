# SQ Defender Providers: NBA Assignments vs Rich Geometry

`scripts/compare_sq_defender_providers.py` compares the NBA matchup-assignment
archive and opposing-lineup D-EPM with ShotQuality's recovered release-geometry
fields on 2024-25 three-point shots.

This is an isolated RAPM/ShotQuality research challenger. It does not modify
`RUSSELL_SHOTQUALITY`, legacy Alt3, or the active public
`weighted_factors_decomp_*` family.

## Provider distinction

The NBA archive's `DEF_ID` is a matchup assignment, not a measured
closest-defender distance. A shot can contain several pipe-separated assigned
defenders. The adapter therefore:

- preserves every assigned defender;
- collapses exact duplicates created by the per-defender CSV layout;
- excludes conflicting event keys;
- creates single- versus multi-assignment features;
- averages prior-game-only, strongly shrunk make-rate-allowed priors across
  every assigned defender;
- excludes same-season `D_LEBRON`, full-sample matchup FG%, and post-outcome
  fields.

The rich ShotQuality arm instead uses physical pre-release geometry: closest
and second-closest distances and angles, defender position relative to the
shooter/basket, height, paint occupancy, spacing, contest score, and related
obstruction features.

## Validation

- Train: 2023-24 threes.
- Test: 2024-25 threes.
- NBA and SQ providers join independently by game/event key.
- Defender outcome priors exclude the whole current game.
- The calibrated `initial_ev` arm fits a prior-season-only Platt map on
  2023-24 and applies it unchanged to 2024-25.
- Loss uncertainty resamples whole games.
- Target: immediate field-goal points per shot. No offensive-rebound or
  second-chance continuation value enters this comparison.

The NBA archive resolves 411,603 events across the two season folders after
57,972 exact duplicate rows are removed. It contains 54,810 multi-assignment
events. NBA assignment coverage is 84,253 of 97,267 test threes (86.62%).
Both NBA assignment and SQ geometry are present on 75,886 threes.

## 2024-25 three-point results

All-shot results:

| Model | Predicted PPS | Actual PPS | PPS bias | Points MSE | Log loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw provider `initial_ev` | 1.071 | 1.081 | -0.009 | 2.05410 | 0.64823 |
| Prior-season calibrated `initial_ev` | 1.056 | 1.081 | -0.024 | 2.05466 | 0.64837 |
| Context + EPM | 1.081 | 1.081 | +0.000 | 2.05780 | 0.64920 |
| Context + EPM + NBA assignment | 1.081 | 1.081 | +0.000 | 2.05787 | 0.64921 |
| Context + EPM + SQ geometry | 1.073 | 1.081 | -0.008 | 2.05298 | 0.64797 |
| SQ geometry + prior-game SQ identity | 1.077 | 1.081 | -0.004 | **2.05249** | **0.64783** |

Aggregate PPS calibration and shot-level predictive loss answer different
questions. The NBA-assignment arm is very close to the correct overall PPS,
but SQ geometry is substantially better at ranking the shot-by-shot make
probability.

Game-cluster points-MSE comparisons, where negative favors the challenger:

| Challenger | Reference | Delta | 95% interval | Read |
| --- | --- | ---: | ---: | --- |
| NBA assignment | Context + EPM | +0.00007 | [-0.00081, +0.00087] | neutral |
| SQ geometry | Context + EPM | **-0.00482** | [-0.00617, -0.00362] | clear gain |
| NBA assignment | SQ geometry | **+0.00489** | [+0.00349, +0.00624] | geometry wins |
| NBA assignment | SQ geometry + SQ identity | **+0.00537** | [+0.00400, +0.00679] | complete SQ arm wins |
| NBA assignment | calibrated `initial_ev` | **+0.00320** | [+0.00127, +0.00522] | calibrated `initial_ev` wins |
| SQ geometry | calibrated `initial_ev` | -0.00168 | [-0.00344, +0.00013] | geometry point win; interval touches zero |
| SQ geometry + SQ identity | calibrated `initial_ev` | **-0.00217** | [-0.00406, -0.00023] | complete SQ arm wins |

On the strict 75,886-shot provider-overlap sample, NBA assignment remains
worse than SQ geometry by `+0.00430` points MSE, with a game-cluster 95%
interval of `[+0.00265, +0.00580]`.

The prior-season Platt calibration is not a success: it lowers `initial_ev`
from 1.071 to 1.056 predicted PPS and worsens points MSE by `+0.00056`
`[+0.00027, +0.00087]` versus raw provider `initial_ev`. The fitted slope is
`0.9518` and intercept is `-0.0497`. This is evidence that a calibration learned
from 2023-24 did not transport cleanly into 2024-25, not permission to calibrate
on test-season outcomes.

## Opposing-lineup D-EPM result

The lineup extension joins each shot to the exact raw PBP event, identifies
offense as the home/away five-man side containing the shooter, and uses the
other side as defense. Invalid, duplicate-player, missing-shooter, or
both-sides shooter orientations are excluded. Each defender then receives the
newest Dunks & Threes `depm` snapshot requested on or before the game date.

Test-season coverage is high:

- 96,903 of 97,267 threes have a valid oriented opposing lineup (99.63%);
- 95,121 threes have D-EPM for all five defenders (97.80%);
- only 364 test threes lack a valid raw event lineup.

Two personnel specifications were tested:

1. minimal total: five-man D-EPM sum, coverage, and availability;
2. richer aggregate: sum, weakest defender, strongest defender, spread,
   coverage, and availability.

Neither improves forward shot-level loss:

| Challenger | Predicted PPS | Points MSE | Delta vs same-seed reference | Game-cluster 95% interval |
| --- | ---: | ---: | ---: | ---: |
| Context + EPM | 1.081 | 2.05780 | reference | — |
| + lineup D-EPM total | 1.082 | 2.05840 | +0.00060 | [-0.00009, +0.00125] |
| + richer lineup D-EPM | 1.079 | 2.05804 | +0.00025 | [-0.00043, +0.00096] |
| SQ geometry + SQ identity | 1.077 | **2.05249** | reference | — |
| + lineup D-EPM total | 1.079 | 2.05333 | **+0.00084** | [+0.00020, +0.00151] |
| + richer lineup D-EPM | 1.083 | 2.05309 | +0.00060 | [-0.00005, +0.00129] |

Positive deltas are worse. On the fully covered test sample, five-man D-EPM
sum has effectively zero correlation with three-point makes (`-0.00015`) and
with context-model residuals (`-0.00053`). This does not mean personnel is
irrelevant. It means all-in D-EPM, aggregated without role or location, does
not add transported information about a conditional three-point result beyond
the current context/talent model. Shot-specific geometry remains materially
more useful.

The next personnel challenger should be mechanism-specific rather than another
all-in impact aggregate: assigned-defender D-EPM relative to the other four,
lineup length/position, block/rim-help priors, and shot-zone suppression.

## Second-chance boundary

This defender comparison and `train_sq_walk_forward_challenger.py` predict
immediate make probability and immediate points per shot only.

Second-chance EV exists in the separate shot-clock/grenade challenger:

```text
expected possession value at release
= immediate shot EV
+ P(miss) × P(OREB | miss, shooter, zone) × expected points per OREB
```

That downstream attachment uses the cross-fitted player-zone opportunity prior.
It has not yet been recomputed with defender geometry or NBA matchup assignment
inside the OREB model. Canonical `RUSSELL_SHOTQUALITY` also remains immediate
shot value; active public DECOMP carries physical realized second-chance value
separately through `SECOND_CHANCE_CLEAN`.

## Run

The source checkout must contain `matchup_fga/defender/2024` and
`matchup_fga/defender/2025`.

```bash
PYTHONPATH=. .venv/bin/python \
  nba_pipeline/scripts/compare_sq_defender_providers.py \
  --test-season 2025 \
  --segment three \
  --bootstrap-iterations 1000
```

Generated artifacts remain gitignored under:

```text
nba_pipeline/results/sq_defender_provider_comparison/
```

- `metrics.csv`;
- `intervals.csv`;
- `predictions.parquet`;
- `manifest.json`.

Focused tests:

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q \
  nba_pipeline/tests/test_compare_sq_defender_providers.py
```
