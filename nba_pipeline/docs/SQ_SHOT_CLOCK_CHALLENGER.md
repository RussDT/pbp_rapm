# Shot-Clock SQ and Grenade Burden

`scripts/train_sq_shot_clock_challenger.py` tests shot clock as a pre-release
feature over repo-owned Shot Quality and measures the burden carried by players
who take end-of-clock attempts.

This is a RAPM-owned experimental overlay. It does not modify
`RUSSELL_SHOTQUALITY`, `FIRST_CHANCE`, `SECOND_CHANCE_CLEAN`, legacy Alt3, or
the active public `weighted_factors_decomp_*` family.

## Why this exists

Two visually identical shots can have different meaning:

- with 18 seconds left, a difficult jumper may be a poor decision;
- with 2 seconds left, the same attempt may be the best available bailout.

Ordinary shot quality asks how likely the attempt is to score. Possession-aware
shot quality also needs the state that forced the attempt. The first useful
state variable is time remaining on the shot clock.

This distinction prevents three concepts from being collapsed:

1. **Clock-conditioned shot expectation:** how likely this shot was to score
   given the actual clock.
2. **Neutral-clock shot expectation:** the same fold model with the clock set
   to 14 seconds while shot context remains fixed.
3. **Grenade burden:** neutral-clock EV minus clock-conditioned EV for attempts
   at or below 4 seconds.

Grenade burden is workload context, not automatically player credit. The
current PBP does not identify who caused the late possession or literally
handed the shooter the grenade.

## Inputs

- repo-owned SQ:
  `results/pbp_shot_quality/pbp_shot_quality_1997_2026.parquet`;
- calibrated 2026 RS clock rows:
  `validation/shot_clock_calibration/shot_clock_26_rs_applied.parquet`;
- player-zone continuation prior:
  `results/second_chance_opportunity_prior/22_26_all_a300/player_zone_accounting.csv`.

SQ and clock rows join one-to-one on:

```text
normalized game_id + period + event_num
```

The build also verifies shooter identity. The current 2026 RS join is
`219,159 / 219,159` shots across `1,230` games with zero mismatches.

## Model

The anchor is `shot_quality_prior_make_prob`, converted to a logit. The
challenger adds only pre-release clock information:

- calibrated continuous clock and its nonlinear transforms;
- seconds below 7 and below 4;
- clock-by-shot-zone interactions;
- calibration interval width and confidence/review state;
- reset reason;
- final-24-game-clock state;
- 2PT/3PT indicator.

The model is a regularized logistic regression. Five-fold `GroupKFold` by
`game_id` produces the shot-level predictions. The baseline-only recalibration
uses the same folds and regularization, so incremental clock gain is not merely
an intercept correction.

The clock estimator itself uses no current-shot outcome, player identity, or
future event. This challenger adds no make/miss, assist, rebound, block, or raw
defender field.

`shot_quality_prior_make_prob` is only chronological in its shooter-residual
layer; the current repo context maps remain retrospective leave-one-shot
encodings over the loaded history. This makes the experiment a clean
incremental clock comparison on the existing SQ surface, not yet a fully
forward-safe production model.

## Neutral-clock counterfactual

Each validation fold model scores the held-out shot twice:

```text
conditioned probability = model(actual calibrated clock)
neutral probability     = model(clock set to 14)
```

Shot context, SQ anchor, zone, reset provenance, and uncertainty remain fixed.
The counterfactual therefore isolates modeled clock time, including its
zone-specific nonlinear terms. Fourteen seconds is an interpretable research
reference, not a claim that every possession should reach a shot at 14.

The shot-EV identity is exact:

```text
clock-conditioned shot EV
+ grenade burden EV
= neutral-clock shot EV
```

## Expected possession value

The build then adds the existing miss/rebound continuation prior:

```text
expected second-chance EV
= (1 - make probability)
 × P(OREB | miss, shooter, zone)
 × expected points per OREB
```

For both actual and neutral clock:

```text
shot possession EV = immediate shot EV + expected second-chance EV
```

The lower make probability on a grenade creates slightly more expected OREB
continuation. Therefore:

```text
possession grenade burden
= shot grenade burden
- second-chance recovery offset
```

Both the possession identity and this bridge close at machine precision. This
is the first working version of the desired pre-shot object:

```text
expected possession value at shot release
```

The continuation attachment is still approximate. It uses the cross-fitted
2022-26 player-zone OREB prior, not an exact all-shot lineup prediction, and
uses the current league-window points-per-OREB conversion.

## 2026 RS validation

| Model | OOF log loss | OOF Brier |
|---|---:|---:|
| Canonical chronological-prior SQ | 0.639902 | 0.225145 |
| Baseline-only recalibration | 0.639822 | 0.225136 |
| Clock-conditioned SQ | **0.638388** | **0.224478** |

Incremental clock gain versus baseline-only recalibration:

- log loss: `-0.001434`, game-cluster 95% interval
  `[-0.001668, -0.001204]`;
- Brier: `-0.000658`, game-cluster 95% interval
  `[-0.000762, -0.000551]`.

Negative challenger-minus-baseline loss is better. The chronologically latest
20% of games also improves:

- baseline-only log loss: `0.639375`;
- clock-conditioned log loss: `0.638197`;
- baseline-only Brier: `0.225008`;
- clock-conditioned Brier: `0.224443`.

### Calibration by clock

| Calibrated clock | Shots | Actual make% | Canonical SQ | Clock SQ |
|---|---:|---:|---:|---:|
| 0–4 seconds | 14,103 | 36.26% | 43.11% | 36.34% |
| 4–7 seconds | 20,217 | 42.70% | 44.40% | 42.61% |
| 7–14 seconds | 99,648 | 46.86% | 46.54% | 46.95% |
| 14–24 seconds | 85,191 | 50.22% | 49.14% | 50.12% |

The main win is exactly the grenade problem: canonical SQ substantially
overstates the expected result of 0–4-second attempts.

### Illustrative grenade loads

These are descriptive 2026 RS burdens, not causal player ratings:

| Player | Grenades | Grenade rate | Shot burden | Possession burden |
|---|---:|---:|---:|---:|
| Payton Pritchard | 157 | 14.39% | 33.16 | 28.26 |
| James Harden | 150 | 13.36% | 29.93 | 26.11 |
| Kevin Durant | 135 | 9.81% | 29.86 | 25.99 |
| Jalen Brunson | 123 | 8.34% | 26.79 | 22.91 |
| Nikola Jokić | 92 | 8.13% | 18.56 | 16.17 |
| Luka Dončić | 76 | 5.22% | 15.17 | 13.52 |
| Joel Embiid | 56 | 8.03% | 11.20 | 10.13 |
| Stephen Curry | 24 | 3.00% | 4.67 | 4.11 |

Values are total expected points over that player's grenade attempts. The
difference between shot and possession burden is expected second-chance
recovery.

## Outputs

Default generated directory:
`results/sq_shot_clock_challenger/2026_rs/`

- `sq_shot_clock_oof_predictions.parquet`;
- `player_grenade_burden.csv`;
- `metrics.json`;
- `manifest.json`;
- `sq_shot_clock_model.joblib`.

Run:

```bash
.venv/bin/python \
  nba_pipeline/scripts/train_sq_shot_clock_challenger.py \
  --season-end-year 2026 \
  --season-type RS \
  --folds 5 \
  --neutral-clock 14 \
  --grenade-threshold 4 \
  --output-dir nba_pipeline/results/sq_shot_clock_challenger/2026_rs
```

## RAPM and VPM ownership

The cross-repo contract in
`/Users/russ/Documents/Codex/basketball/shared/NBA_EVENT_ENRICHMENT_PLAN.md`
applies:

### RAPM-owned overlays

- shot-level clock-conditioned and neutral-clock OOF probabilities;
- expected possession value at shot release;
- shot- and possession-level grenade burden;
- shot-level OREB continuation attachments;
- eventual on-ball/off-ball defender and state/event-value models.

### Possible VPM priors after reliability gates

- stabilized player-season grenade attempt rate;
- stabilized player-season possession grenade burden per attempt;
- stabilized clock-adjusted shooter-talent residual.

Required gates are multi-season backfill, forward-season validation,
year-to-year stability, minimum-attempt reliability, and incremental VPM target
validation.

### Never direct VPM inputs

- raw shot or event rows;
- provider-specific defender joins;
- shot-level reset reasons or confidence fields;
- fold-level predictions;
- raw state/event attribution rows.

VPM consumes only stabilized player-season outputs after those tests. Until
then, every output from this build is an explanatory RAPM research overlay.

## Next work

1. Backfill calibrated clock over 2022-26 RS/PS and repeat forward-season
   validation.
2. Compare internal chronological shooter talent against date-as-of EPM
   `p_fg3pct` / `p_fgpct_mid` and a frozen blend, holding shot context and
   clock features fixed. Report points per shot and expected possession value,
   not only probability loss.
3. Add clock to the OREB opportunity model rather than attaching only its
   player-zone summary.
4. Learn full continuation value after the first OREB instead of using a
   league-window conversion.
4. Materialize the exact actual-points residual tree:

```text
actual possession points
= pre-shot possession EV
+ shot realization residual
+ rebound realization residual
+ second-chance conversion residual
```

5. Solve those children as isolated RAPM research targets before any public
   DECOMP proposal.
