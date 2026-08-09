# Forward SQ + Second-Chance Release Value

`scripts/compare_sq_second_chance_forward.py` tests whether the
shooter-specific rebound-opportunity layer improves prediction of value from
shot release through the miss/rebound continuation branch.

This is an isolated RAPM research challenger. It does not modify canonical
`RUSSELL_SHOTQUALITY`, legacy Alt3, or the active public
`weighted_factors_decomp_*` family. Physical realized second-chance value
continues to live in `SECOND_CHANCE_CLEAN` on the public DECOMP surface.

## Target and fair comparison

The benchmark evaluates the first FGA of each standard possession:

```text
actual release points
= points on the initial FGA
 + SECOND_CHANCE_CLEAN points after its miss
```

First-chance free throws are excluded. The target is therefore a clean
shot-plus-second-chance branch, not every possible point on the possession.

Expected value is:

```text
release EV
= immediate make probability × shot value
 + (1 - make probability)
   × P(OREB | miss, shot context, exact lineups)
   × train-only points per initial-shot OREB
```

The comparison has two continuation lanes:

1. `context_lineup`: shot descriptors, four exact non-shooters, and five exact
   defenders; shooter identity is hidden;
2. `actual_config`: the same model plus shooter identity × shot zone.

The second lane is the predictive version of the Embiid effect. The separate
role swap moves each teammate into the same-zone shot and Embiid into the
vacated non-shooter role for explanatory accounting.

It is not fair to claim that a continuation-adjusted model beats raw
`initial_ev` merely because `initial_ev` predicts immediate FGA points. The
primary competitor is therefore:

```text
provider initial_ev + the same context/lineup continuation
```

## First forward contract

- OREB fit: 2021-22 through 2023-24 RS+PS;
- test: 2024-25 RS+PS;
- no 2024-25 OREB outcome enters model coefficients;
- points per initial-shot OREB is estimated from 2022-24 only:
  - midrange: `1.22546`;
  - three: `1.17918`;
- loss uncertainty resamples whole games;
- test universe: `131,477` first-FGA releases over `1,310` games after exact
  SQ, possession, and complete-lineup joins.

The OREB fit contains `320,120` pre-test first-FGA misses. The role-swap
coefficient identity closes to `1.4e-16`. A second rolling test now fits
through 2024-25 and evaluates 2025-26 below.

## First forward result: 2024-25

Combined midrange and threes:

| Immediate model | Continuation | Predicted | Actual | Points MSE |
| --- | --- | ---: | ---: | ---: |
| Provider `initial_ev` | none | 1.001 | 1.211 | 1.75779 |
| Provider `initial_ev` | context/lineup | 1.183 | 1.211 | **1.71238** |
| Provider `initial_ev` | shooter-specific | 1.183 | 1.211 | 1.71231 |
| Context + EPM | context/lineup | 1.196 | 1.211 | 1.71679 |
| Context + EPM | shooter-specific | 1.196 | 1.211 | 1.71665 |
| Geometry + identity | context/lineup | 1.197 | 1.211 | 1.71267 |
| Geometry + identity | shooter-specific | 1.197 | 1.211 | 1.71253 |

Adding shooter × zone over the identical context/lineup continuation produces:

| Immediate model | MSE delta | Game-cluster 95% interval | Read |
| --- | ---: | ---: | --- |
| Provider `initial_ev` | -0.000072 | [-0.000224, +0.000066] | directional |
| Context + EPM | **-0.000142** | **[-0.000277, -0.000003]** | small gain |
| Geometry + identity | **-0.000140** | **[-0.000283, -0.000006]** | small gain |

Negative is better. The Embiid-style shooter/zone term adds a small but real
increment on the combined sample for the two internal SQ arms.

It does **not** yet make the full internal model beat the fair provider
benchmark:

- geometry + shooter-specific continuation versus provider `initial_ev` +
  context/lineup continuation: `+0.000144` MSE,
  95% interval `[-0.001274, +0.001409]`;
- context + EPM + shooter-specific continuation is clearly worse by
  `+0.004267`, interval `[+0.002792, +0.005688]`.

The best internal arm is therefore statistically tied with, not better than,
provider `initial_ev` once both receive a forward-safe continuation in this
first test season.

Adding any continuation to immediate-only `initial_ev` improves MSE by
`-0.04540`, but that large result is expected because the evaluation target
explicitly contains second-chance points. It is not evidence that the
shooter-specific term alone created that gain.

## Second forward result: 2025-26

The rolling replication uses:

- OREB fit: 2021-22 through 2024-25 RS+PS;
- test: 2025-26 RS+PS;
- `427,532` pre-test first-FGA misses;
- `130,229` held-out first-FGA releases over `1,280` games;
- train-only points per initial-shot OREB:
  - midrange: `1.22188`;
  - three: `1.17992`;
- role-swap coefficient identity gap: `1.2e-16`.

Combined midrange and threes:

| Immediate model | Continuation | Predicted | Actual | Points MSE |
| --- | --- | ---: | ---: | ---: |
| Provider `initial_ev` | none | 0.993 | 1.215 | 1.75035 |
| Provider `initial_ev` | context/lineup | 1.180 | 1.215 | 1.70018 |
| Provider `initial_ev` | shooter-specific | 1.180 | 1.215 | 1.70010 |
| Context + EPM | context/lineup | 1.215 | 1.215 | 1.69731 |
| Context + EPM | shooter-specific | 1.215 | 1.215 | 1.69723 |
| Geometry + identity | context/lineup | 1.195 | 1.215 | 1.69564 |
| Geometry + identity | shooter-specific | 1.195 | 1.215 | **1.69555** |

Adding shooter × zone over the identical context/lineup continuation produces:

| Immediate model | MSE delta | Game-cluster 95% interval | Read |
| --- | ---: | ---: | --- |
| Provider `initial_ev` | -0.000081 | [-0.000214, +0.000060] | directional |
| Context + EPM | -0.000080 | [-0.000211, +0.000049] | directional |
| Geometry + identity | -0.000088 | [-0.000231, +0.000045] | directional |

The pooled shooter-specific continuation increment is again favorable in
point estimate but does not clear zero in 2025-26. By segment, the internal
midrange arms do clear zero:

- context + EPM: `-0.000209`, interval
  `[-0.000390, -0.000021]`;
- geometry + identity: `-0.000191`, interval
  `[-0.000374, -0.000037]`.

The three-point shooter-specific increment is neutral. The combined evidence
therefore supports a modest shooter/zone signal whose forward strength is not
stable enough to describe as a general league-wide gain.

The complete internal immediate-shot models perform better in this second
season once every arm receives the fair continuation:

- geometry + shooter-specific continuation versus provider `initial_ev` +
  context/lineup continuation: `-0.004628` MSE,
  95% interval `[-0.006233, -0.003059]`;
- context + EPM + shooter-specific continuation versus the same provider
  reference: `-0.002947`, interval `[-0.004557, -0.001383]`.

Adding contextual continuation to immediate-only `initial_ev` improves MSE by
`-0.05017`, interval `[-0.05287, -0.04760]`. As in 2024-25, that large gain is
expected because the target explicitly includes second-chance points.

## Descriptive zone baselines

The player × zone layer is an adjustment to the shot-zone baseline, not the
whole continuation effect of selecting that shot. Pooling the 2024-25 and
2025-26 held-out predictions and using provider `initial_ev` plus the
shooter-hidden `context_lineup` continuation gives:

| Initial shot zone | Held-out attempts | P(OREB \| miss) | Expected second-chance points per attempt |
| --- | ---: | ---: | ---: |
| Rim | 4,330 | 36.1% | 0.241 |
| Short midrange | 33,611 | 30.6% | 0.208 |
| Long two | 52,563 | 23.7% | 0.171 |
| Three | 171,202 | 24.2% | 0.183 |

`long_mid` and `deep_two` are combined as `long two`. These are descriptive
held-out zone averages, not a causal same-state counterfactual: shooters,
lineups, shot contexts, and defensive contexts differ across zones.

The model makes OREB probability zone-sensitive. It does not separately
estimate points after the rebound for every exact zone; the train-only
conversion is pooled at the broader midrange-versus-three segment. Therefore
`0.171` is expected continuation per initial long-two attempt, not proof that a
rebound after a long two has a distinct exact-zone second-chance PPP.

## Embiid examples and replication

On 2024-25 first-FGA long twos:

- `96` eligible attempts;
- shooter-role OREB delta versus the equal-teammate role swap:
  `-1.15` percentage points;
- second-chance value delta: `-0.0084` points per attempt;
- total modeled role-vacancy cost: `-0.81` points.

The smaller one-season total reflects Embiid's limited 2024-25 sample. The
earlier 2022-26 retrospective accounting remains the higher-exposure
descriptive result. This forward run answers transport and predictive value;
it does not replace the multi-season player accounting.

On 2025-26 first-FGA long twos:

- `295` eligible attempts (`241` long mids and `54` deep twos);
- context/lineup OREB probability: `20.18%`;
- actual Embiid configuration: `19.15%`;
- equal-teammate role swap: `19.28%`;
- shooter-role OREB delta: `-0.13` percentage points;
- second-chance value delta: `-0.0010` points per attempt;
- total modeled role-vacancy cost: `-0.28` points.

The stronger Embiid role-vacancy effect does not replicate in 2025-26. That
does not invalidate the higher-exposure retrospective accounting, but it
changes the reader-facing conclusion: the player example is descriptive and
unstable across single forward seasons, not a confirmed player penalty.

## Two-season reading

Across both forward seasons:

1. contextual miss/OREB continuation materially improves prediction of the
   continuation-inclusive release target;
2. the shot-zone continuation baseline must be established before interpreting
   any player × zone adjustment;
3. shooter × zone is consistently favorable in point estimate but clears zero
   only in selected season/model or midrange cells;
4. the complete geometry lane is tied with the fair provider benchmark in
   2024-25 and clearly better in 2025-26;
5. the Embiid-specific long-two effect is much smaller in the second forward
   season.

The result is now a two-season rolling confirmation for contextual
continuation, but not a stable two-season confirmation of a large
shooter-specific or Embiid-specific effect.

## Run

```bash
PYTHONPATH=. .venv/bin/python \
  nba_pipeline/scripts/compare_sq_second_chance_forward.py \
  --test-season 2025 \
  --train-start-season 2022 \
  --season-type ALL \
  --alpha 300 \
  --bootstrap-iterations 1000 \
  --output-dir nba_pipeline/results/sq_second_chance_forward/2025
```

For the rolling 2025-26 replication, use `--test-season 2026` and output to
`nba_pipeline/results/sq_second_chance_forward/2026`.

Generated artifacts remain gitignored:

- `metrics.csv`;
- `intervals.csv`;
- `predictions.parquet`;
- `player_zone_accounting.csv`;
- `manifest.json`.

Focused tests:

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q \
  nba_pipeline/tests/test_compare_sq_second_chance_forward.py \
  nba_pipeline/tests/test_second_chance_opportunity_prior.py
```
