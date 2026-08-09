# Season-Forward SQ: EPM and Closest-Defender Features

`scripts/train_sq_walk_forward_challenger.py` tests whether date-as-of EPM
shooting skill and ShotQuality closest-defender information improve repo-owned
shot prediction when the entire test season is unseen by the fitted model.

This is an isolated RAPM/ShotQuality research challenger. It does not modify
`RUSSELL_SHOTQUALITY`, legacy Alt3, or the active public
`weighted_factors_decomp_*` family.

## Information contract

For test season S:

- model coefficients train only on seasons before S;
- shooter and assigned-defender outcome priors use only completed earlier
  games, excluding every shot in the current game;
- EPM predictive rates join from the most recent cached requested date at or
  before the game date;
- all geometry is pre-release ShotQuality-provider information;
- current-shot make/miss, points, assists, rebounds, and blocks are excluded.

The run compares five arms on identical test rows:

1. ShotQuality `initial_ev`;
2. internal context plus prior-game shooter talent;
3. internal context plus EPM `p_fg3pct` / `p_fgpct_mid`;
4. EPM/context plus closest- and second-closest-defender geometry;
5. geometry plus prior-game primary-defender make rate allowed.

The primary output is immediate expected points per shot. Brier and log loss
remain the scoring rules used to distinguish calibrated probability forecasts.

## Provider boundary

The current rich-feature experiment uses ShotQuality fields from the surviving
`sq_shots_keyed.parquet`:

- `opp_personId` is the ShotQuality primary-defender assignment;
- closest- and second-closest-defender distances, angles, positions, height,
  paint occupancy, contest, and spacing are ShotQuality geometry;
- no inferred second-defender identity is used.

NBA matchup assignments are a separate provider. The committed
`matchup_fga/defender/` tree runs from season-end 2017 through 2025—2017 is
the 2016-17 season. Its `DEF_ID` may contain multiple pipe-separated assigned
defenders, so it must be normalized to provider/event/defender link rows rather
than forced into a single closest-defender column. This longer identity lane
should be tested separately, then compared on the provider-overlap sample.
That forward provider comparison is now implemented and documented in
[SQ Defender Providers: NBA Assignments vs Rich Geometry](./SQ_DEFENDER_PROVIDER_COMPARISON.md).

## Data

Source:

```text
/Users/russ/Documents/Codex/basketball/nba/research/vpm/
vpm/data/sq_shots_keyed.parquet
```

The source has 697,925 rows across 2023-24, 2024-25, and 2025-26. After
removing missing/ambiguous provider event keys, retaining midrange/threes, and
requiring an `initial_ev` benchmark, the model uses 436,235 shots across 3,902
games.

Overall feature coverage:

- as-of EPM shooter skill: 99.57%;
- ShotQuality primary-defender identity: 75.10%;
- closest-defender geometry: 75.45%.

Geometry coverage falls from about 90% in the 2024-25 test season to about 41%
on midrange/threes in 2025-26. Results are therefore reported both on all shots
with explicit missingness and on the geometry-available subset.

## Points-per-shot results

The complete model is context + EPM + geometry + prior-game defender identity.

| Test season | Segment | Shots | Actual PPS | `initial_ev` PPS | Context+EPM PPS | Complete PPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2024-25 | Mid | 49,009 | 0.890 | 0.861 | 0.888 | **0.889** |
| 2024-25 | Three | 97,275 | 1.081 | 1.071 | 1.079 | **1.081** |
| 2025-26 | Mid | 50,632 | 0.894 | 0.824 | 0.874 | **0.886** |
| 2025-26 | Three | 94,137 | 1.078 | 1.083 | 1.116 | **1.076** |

This is calibration in points per attempt, not FG% or eFG%.

## Forward loss results

Negative loss differences mean the challenger is better.

### Defender geometry versus context + EPM

| Test season | Segment | Points-MSE delta | Game-cluster 95% interval | Result |
| --- | --- | ---: | ---: | --- |
| 2024-25 | Mid | **-0.003521** | [-0.004640, -0.002327] | clear gain |
| 2024-25 | Three | **-0.005076** | [-0.006427, -0.003642] | clear gain |
| 2025-26 | Mid, all shots | -0.000555 | [-0.001701, +0.000542] | uncertain |
| 2025-26 | Mid, geometry present | **-0.002346** | [-0.004685, -0.000217] | gain on covered rows |
| 2025-26 | Three | **-0.003609** | [-0.005232, -0.002236] | clear gain |

The 2025-26 all-shot midrange result is diluted by missing provider geometry,
not evidence that observed geometry stopped working.

### Complete model versus `initial_ev`

| Test season | Segment | Points-MSE delta | Game-cluster 95% interval |
| --- | --- | ---: | ---: |
| 2024-25 | Mid | **-0.002992** | [-0.004790, -0.000985] |
| 2024-25 | Three | -0.001134 | [-0.002940, +0.000658] |
| 2025-26 | Mid | **-0.013826** | [-0.016736, -0.010857] |
| 2025-26 | Three | **-0.004990** | [-0.007170, -0.002943] |

Point estimates favor the complete model in all four cells. Three cells have
intervals entirely below zero; 2024-25 threes are directionally positive but
not decisive versus `initial_ev`.

### EPM contribution

Adding EPM shooting skill to the same internal context/prior model improves
points MSE clearly for 2024-25 mids, 2024-25 threes, and 2025-26 threes.
It is neutral for 2025-26 mids:

| Test season | Segment | EPM-minus-internal points-MSE delta |
| --- | --- | ---: |
| 2024-25 | Mid | **-0.001442** |
| 2024-25 | Three | **-0.004016** |
| 2025-26 | Mid | +0.000003 |
| 2025-26 | Three | **-0.002362** |

This supports EPM as a shooter-talent prior, especially for threes. It does not
support replacing shot context with EPM.

## Defender identity interpretation

The prior-game ShotQuality-assigned defender rate adds a clear incremental
gain for 2025-26 threes and a smaller 2024-25 midrange gain. It is neutral for
2024-25 threes and 2025-26 mids. Identity should therefore remain a strongly
shrunk prior over geometry, not a direct causal defender effect.

The next long-history test should use NBA matchup assignments to construct:

- single-assignment and multi-assignment indicators;
- prior-game defender zone make rate allowed;
- prior-game block rate from ordinary PBP;
- shooter-versus-defender exposure counts;
- NBA/SQ agreement and disagreement slices on overlapping seasons.

Do not import same-season `D_LEBRON`, full-sample matchup FG%, or any
post-outcome field as a pre-shot feature.

## Run

```bash
PYTHONPATH=. .venv/bin/python \
  nba_pipeline/scripts/train_sq_walk_forward_challenger.py \
  --test-seasons 2025 2026 \
  --bootstrap-iterations 500
```

Generated artifacts remain gitignored under:

```text
nba_pipeline/results/sq_walk_forward_challenger/
```

- `walk_forward_predictions.parquet`;
- `walk_forward_metrics.csv`;
- `walk_forward_intervals.csv`;
- `manifest.json`.

Focused tests:

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q \
  nba_pipeline/tests/test_sq_walk_forward_challenger.py
```
