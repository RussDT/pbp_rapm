# Block-Aware Rim ShotQuality Challenger

## Scope

`scripts/train_sq_rim_block_challenger.py` is an isolated season-forward rim
make-probability experiment. It does not change `RUSSELL_SHOTQUALITY`, public
DECOMP, the accepted VPM target, or `SECOND_CHANCE_CLEAN`.

The research question is whether an explicit pre-shot block-risk channel adds
forward information beyond ShotQuality `initial_ev`, shot context, shooter
skill, and defender geometry.

## Data and labels

The keyed ShotQuality table contributes the provider event key, `initial_ev`,
shot descriptors, shot location, primary-defender assignment, and pre-release
geometry. NBA raw PBP contributes the independently keyed block label and the
five defenders on court.

After resolving provider/raw conflicts and removing unmatched or ambiguous
rows, the current table contains:

- 248,802 rim attempts;
- 3,902 games across season-end 2024, 2025, and 2026;
- 12.38% blocked attempts;
- 99.87% resolved opposing-five lineups;
- 99.63% as-of EPM rim-shooter coverage;
- 98.10% complete opposing-five as-of EPM block-rate coverage.

One event labeled both made and blocked is excluded. The block label is parsed
from the raw NBA event description and is never a feature for that shot.

## Feature contract

The direct model combines:

- ShotQuality `initial_ev` as an anchor;
- shot location, distance, clock, spacing, and outcome-invariant descriptors;
- whole-current-game-excluded shooter rim make history;
- date-aware Dunks & Threes `p_fgpct_rim` and rim attempt rate;
- closest/second-closest defender geometry and paint occupancy;
- whole-current-game-excluded shooter block susceptibility;
- whole-current-game-excluded assigned-defender block association;
- date-aware assigned-defender `p_blk_100`;
- sum, maximum, and top-two date-aware `p_blk_100` among the five defenders.

Model coefficients train only on seasons before the test season. Dunks &
Threes fields use the latest requested snapshot on or before the game date;
the provider contract says a date-D prediction excludes games on D. Current
shot make, miss, block, rebound, points, assist, and post-release fields are
excluded.

The staged diagnostic separately estimates:

```text
P(make) = (1 - P(block)) * P(make | not blocked)
```

Both stages use histogram gradient boosting with the frozen feature contract.
The direct anchored model is currently the more stable candidate; the staged
identity is retained as a mechanism diagnostic.

## Forward results

The experimental unit is a shot. Loss is averaged over every held-out rim
attempt, and paired uncertainty resamples whole games.

### Anchored direct model versus ShotQuality `initial_ev`

| Test season | Attempts | Initial-EV points MSE | Challenger points MSE | Delta | Relative reduction | 95% game-cluster interval |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2024-25 | 80,965 | 0.824100 | **0.815103** | **-0.008997** | **1.09%** | `[-0.010896, -0.006991]` |
| 2025-26 | 81,917 | 0.869936 | **0.840521** | **-0.029415** | **3.38%** | `[-0.031805, -0.027237]` |

The anchored challenger is also much closer to actual mean points per shot in
both folds. It predicts `1.2211` versus `1.2195` actual in 2024-25 and `1.2373`
versus `1.2264` actual in 2025-26.

### Does block information add?

Yes inside the repo-owned unanchored ablation. Adding the block feature family
to the otherwise identical context + EPM + geometry model changes points MSE
by:

| Test season | Block-family points-MSE delta | 95% interval |
| --- | ---: | --- |
| 2024-25 | **-0.001668** | `[-0.002584, -0.000748]` |
| 2025-26 | **-0.009681** | `[-0.011092, -0.008403]` |

The separate block classifier has:

| Test season | Actual block rate | Predicted block rate | ROC AUC |
| --- | ---: | ---: | ---: |
| 2024-25 | 12.35% | 11.99% | **0.714** |
| 2025-26 | 12.22% | 12.30% | **0.697** |

On 2025-26 attempts with rich geometry, block AUC is `0.755`. AUC is a
ranking diagnostic, not accuracy or calibration.

### Staged model

The Initial-EV-anchored staged model also beats provider `initial_ev` in both
folds. Relative to the direct anchored model, however, it is neutral-to-worse
in 2024-25 (`+0.000545` points MSE; interval crosses zero) and better in
2025-26 (`-0.002946`; interval entirely below zero). Do not force the staged
identity into a production target based on these two already inspected folds.

## Important deferred continuation issue

This version predicts **immediate rim points only**:

```text
immediate rim EV = 2 * P(make)
```

A blocked rim attempt is not equivalent to an ordinary miss. Blocks can have
a different offensive-rebound and offense-retention rate, including blocked
balls recovered by the offense or knocked out of bounds. Therefore this
immediate EV must not be described as total possession value.

A later continuation model should separately estimate block-specific and
ordinary-miss retention/second-chance value. If that continuation is admitted
into a possession target, it must replace the corresponding second-chance
expectation or carry an exact actual-minus-expected residual. Never add it on
top of unchanged `SECOND_CHANCE_CLEAN`, which would double count value.

## Interpretation and admission status

The result establishes useful development evidence:

- block risk is predictable from pre-release context;
- block-rate features add forward make-probability information;
- an `initial_ev`-anchored rich model beats provider `initial_ev` on both
  inspected forward folds.

It does **not** establish a production rim target or a dRAPM improvement. Model
design changed after examining the 2024-25 and 2025-26 results, so both folds
are spent development evidence. Promotion requires:

1. a frozen 2026-27 or other untouched forward shot test;
2. block/ordinary-miss continuation accounting;
3. exact DECOMP/VPM row algebra if rim EV replaces actual rim points;
4. a summed future-possession test against the current rim-actual target;
5. a future defensive game-prediction test before making a dRAPM claim.

## Reproduction

```bash
PYTHONPATH=. .venv/bin/python \
  nba_pipeline/scripts/train_sq_rim_block_challenger.py \
  --test-seasons 2025 2026 \
  --bootstrap-iterations 500
```

Generated artifacts remain gitignored under:

```text
nba_pipeline/results/sq_rim_block_challenger/
```

Focused tests:

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q \
  nba_pipeline/tests/test_sq_rim_block_challenger.py
```
