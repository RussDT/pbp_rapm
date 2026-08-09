# ShotQuality-Era Challenger Models

## Outcome

The recovered 2024-25 ShotQuality feature store supports two same-row
shot-make challengers and two historical-safe transition classifiers.

- Midrange: the nested regularized-logit challenger beats `initial_ev` on
  grouped out-of-game Brier and log loss, with both game-cluster confidence
  intervals below zero. The original nonlinear anchored model remains in the
  report and is only a statistical tie.
- Three-pointers: both anchored challengers beat `initial_ev`; the nonlinear
  model has the best log loss, while regularized logit has the best Brier and
  calibration in grouped CV.
- Transition labels: ordinary play-by-play action, clock, and prior-event
  sequence predict the recovered SQ `transition` and
  `transition + dunk/lay_up` labels well on held-out games. These are models of
  the vendor descriptor semantics, not a new human-labeled ground truth.

No RAPM processor or public factor has been changed. These remain isolated
model-development artifacts until a forward-season validation is available.

## Data inventory

The rich input is the surviving sibling checkout:

```text
/Users/russ/Documents/Codex/basketball/nba/research/shotquality/nba_games/2025
```

It contains 1,138 RS/PS game CSVs, 203,807 rows, and 203,181 FGA rows from the
2024-25 season. The fair `initial_ev` comparison retains:

| Segment | Shots | Games |
| --- | ---: | ---: |
| Midrange | 20,163 | 1,137 |
| Three-pointer | 85,155 | 1,138 |

The transition build recovers 199,854 SQ-labeled FGA rows and joins 199,849 of
them to `NBA25.parquet` / `NBA25_PS.parquet` by normalized
`game_id + GAME_EVENT_ID` (`99.9975%`, 1,137 games).

This is one rich tracking season, not three. `NBA26.parquet` and
`NBA26_PS.parquet` retain SQ descriptors, transition labels, and EV fields,
but they do not retain the motion/defender feature store. The local NBA24 and
NBA25 RAPM raw files lost their SQ columns during lineup repair; the sibling
game CSVs are the recovered 2024-25 source.

The cross-repo inventory contains a broader recovered 2024-25 rich store
(1,274 games and 227,032 FGA); this benchmark's filtered training extract is
the 1,138-game/203,181-FGA subset above. True SQ primary-defender identity is
available on about 76% of the broader FGA surface. ShotQuality supplies
second-defender geometry but no native second-defender ID. The challenger uses
the geometry and no player identifiers; any repo biometric reconstruction of a
second identity is inferred and is not a provider label. See
[NBA Event-Enrichment Contract](./NBA_EVENT_ENRICHMENT_CONTRACT.md).

## Commands

Full shot-value benchmark:

```bash
.venv/bin/python nba_pipeline/scripts/train_sq_shot_value_challenger.py \
  --bootstrap-iterations 1000
```

Reuse already verified nonlinear OOF predictions while rebuilding the nested
regularized-logit benchmark:

```bash
.venv/bin/python nba_pipeline/scripts/train_sq_shot_value_challenger.py \
  --bootstrap-iterations 1000 \
  --reuse-hgb-oof
```

Refresh nonlinear permutation importance correctly on the untouched ordered
last-20%-of-games split:

```bash
.venv/bin/python nba_pipeline/scripts/train_sq_shot_value_challenger.py \
  --bootstrap-iterations 1000 \
  --reuse-hgb-oof \
  --refresh-hgb-importance
```

Transition classifiers:

```bash
.venv/bin/python nba_pipeline/scripts/train_sq_transition_classifier.py
```

Historical production scoring and calibration are documented in
[Historical Transition Backfill](./HISTORICAL_TRANSITION_BACKFILL.md).

Refresh only the untouched 2025-26 forward-season evaluation:

```bash
.venv/bin/python nba_pipeline/scripts/train_sq_transition_classifier.py \
  --reuse-models-forward-only
```

Focused tests:

```bash
.venv/bin/python -m pytest \
  nba_pipeline/tests/test_sq_challenger_models.py -q
```

## Validation contract

The primary validation is five-fold shuffled `GroupKFold`, grouped by
`game_id`. A shot from one game can never appear in training for another shot
from that game's outer test fold.

The regularized-logit challenger has a second three-fold game-group loop
inside each outer training fold. It selects `C` from
`[0.001, 0.003, 0.01, 0.03, 0.1]` using only inner-training games. The outer
test game is not used to select the regularization. The selected midrange
values were `[0.003, 0.01, 0.003, 0.01, 0.003]`; every three-point fold selected
`0.001`.

The secondary robustness check trains on the first 80 percent of ordered game
ids and evaluates the last 20 percent. Game id is only an approximate
chronology, so a future build should use canonical game dates.

The paired differences use 1,000 bootstrap draws clustered by game. Negative
challenger-minus-`initial_ev` Brier/log loss is better.

## Shot-value benchmark

### Grouped out-of-fold probability metrics

| Segment | Model | Log loss | Brier | ROC AUC | PR AUC | ECE-10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Midrange | `initial_ev` | 0.671249 | 0.239596 | 0.583595 | 0.549376 | 0.028892 |
| Midrange | nonlinear anchored | 0.671159 | 0.239506 | 0.586598 | 0.559383 | 0.037988 |
| Midrange | nested regularized logit | **0.664694** | **0.236455** | **0.600765** | **0.568528** | **0.005992** |
| Three | `initial_ev` | 0.650011 | 0.228990 | 0.537919 | 0.385602 | 0.011648 |
| Three | nonlinear anchored | **0.646825** | 0.227655 | **0.558993** | **0.406778** | 0.011417 |
| Three | nested regularized logit | 0.647002 | **0.227618** | 0.558481 | 0.405686 | **0.004658** |

### Paired game-cluster uncertainty

| Segment/model | Brier delta | 95% CI | Log-loss delta | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Midrange nonlinear anchored | -0.000091 | [-0.001271, +0.001137] | -0.000090 | [-0.002661, +0.002486] |
| Midrange nested logit | **-0.003141** | **[-0.003956, -0.002319]** | **-0.006555** | **[-0.008268, -0.004798]** |
| Three nonlinear anchored | **-0.001335** | **[-0.001685, -0.001005]** | **-0.003186** | **[-0.003943, -0.002426]** |
| Three nested logit | **-0.001372** | **[-0.001623, -0.001141]** | **-0.003009** | **[-0.003561, -0.002496]** |

The original midrange nonlinear result is preserved because it is the honest
first benchmark. Its interval crosses zero, so it is not evidence of a better
midrange model. The nested regularized-logit alternative is the successful
midrange result.

### Ordered last-20%-of-games check

| Segment | Model | Log loss | Brier | ROC AUC | PR AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| Midrange | `initial_ev` | 0.673683 | 0.240590 | 0.577602 | 0.544109 |
| Midrange | nested regularized logit | **0.666916** | **0.237372** | **0.599212** | **0.564831** |
| Three | `initial_ev` | 0.652125 | 0.230063 | 0.544423 | 0.395004 |
| Three | nonlinear anchored | **0.650239** | **0.229288** | **0.562416** | **0.415091** |
| Three | nested regularized logit | 0.650835 | 0.229476 | 0.560964 | 0.414554 |

## Exact shot-value features

Both nested logit models have 44 inputs.

Common physical/context inputs (16):

- `baseline_initial_logit`
- `contest_score`, `spacing`, `shot_x`, `shot_y`
- `feature_store.Shot Distance (ft.)`
- closest and second-closest defender distance
- closest-defender height and weight
- number of defenders closer to the basket
- cosine angle to the closest and second-closest defender
- closest-defender behind, inside, and blocking flags

Segment-specific prior inputs (five):

- midrange: player midrange, player jumpshot, men's midrange, closest-defender
  midrange, and closest-defender jumpshot priors;
- three: player three, player jumpshot, men's three, closest-defender three,
  and closest-defender jumpshot priors.

Descriptor indicators (18):

```text
alley_oop catch_and_shoot dunk floater hook in_paint jumpshot lay_up
off_cut off_drive pick_and_roll post_up pull_up reverse step_back tip_in
transition turnaround
```

Zone indicators (five):

```text
Above the Break 3, Left Corner 3, Right Corner 3, Mid-Range, UNKNOWN
```

The exact ordered lists are also stored in
`sq_shot_value_challenger_metrics.json` and in each serialized model bundle.

The nonlinear models use the broader explicit allowlist in
`train_sq_shot_value_challenger.py`: all 44 fields above plus release-clock
context, broader defender geometry/bounding-box fields, paint occupancy, and
shooter/first-/second-defender velocity and acceleration. The only EV input
is `initial_ev / shot_value`.

## Leakage controls and interpretation

Excluded from both challenger families:

- `dynamic_ev`, `live_ev`, all shot-taking outputs;
- current-shot make/miss and NBA `SHOT_MADE_FLAG`;
- current-shot points, assist, rebound, and block fields;
- score differential and post-shot score;
- player/team identifiers.

The player and closest-defender “Prior” columns are vendor-provided pre-shot
features, but their chronological construction cannot be audited from this
checkout. They make this a player-informed challenger, not player-agnostic
xFG. A forward-season test is required before production use.

NBA matchup defender assignments are a separate provider and are not silently
substituted for these ShotQuality fields. Any provider-comparison model must
retain NBA-only, SQ-only, agreement/high-confidence, and geometry variants.

The nonlinear feature-importance CSVs use a model fit only on the first 80
percent of ordered games and permutation scoring only on the untouched final
20 percent. Full-sample model gain is not used as evidence.

## Transition classifiers

### Results

| Target | Positives | Rate | OOF log loss | OOF Brier | OOF ROC AUC | OOF PR AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `is_transition` | 28,856 | 14.44% | 0.138476 | 0.038273 | 0.966235 | 0.894131 |
| `is_transition_dunk_or_layup` | 15,412 | 7.71% | 0.056825 | 0.015404 | 0.990166 | 0.931290 |

Ordered last-20%-of-games:

| Target | Log loss | Brier | ROC AUC | PR AUC |
| --- | ---: | ---: | ---: | ---: |
| `is_transition` | 0.159728 | 0.045299 | 0.964035 | 0.897651 |
| `is_transition_dunk_or_layup` | 0.071346 | 0.020077 | 0.988275 | 0.925196 |

Untouched 2025-26 forward descriptor-label season, with no refit:

| Target | Shots | Rate | Log loss | Brier | ROC AUC | PR AUC | ECE-10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `is_transition` | 226,986 | 15.61% | 0.146636 | 0.041792 | 0.969107 | 0.901124 | 0.014445 |
| `is_transition_dunk_or_layup` | 226,986 | 8.61% | 0.065877 | 0.018714 | 0.989828 | 0.933124 | 0.007874 |

The forward season strengthens the discrimination result and reveals a
calibration drift: the transition model predicts a 14.34% mean rate against a
15.61% observed rate, and the transition-rim model predicts 7.84% against
8.61%. Historical backfills should therefore retain the raw probability and
apply season-level calibration rather than assuming one fixed threshold or
base rate.

### Exact historical-safe features

The 97-column run contains:

- eight numeric fields: period, current period clock, current action-type
  code, elapsed seconds to each of the prior three events, seconds since the
  last possession-change candidate, and parsed shot distance;
- 15 current-shot action flags:
  `3pt`, `dunk`, `layup`, `hook`, `float`, `jump`, `pullup`, `step_back`,
  `fadeaway`, `cutting`, `driving`, `putback`, `turnaround`, `alley_oop`,
  `bank`;
- current `event_action_type` one-hot values observed in the training data
  (only code `0` survived in this recovered raw surface);
- for each of lags 1, 2, and 3:
  - one-hot prior event types `1..13` and `18`;
  - whether the prior event description was on the opponent side;
  - text flags for `steal`, `rebound`, `turnover`, `miss`, `pts`, `block`,
    `free throw`, and `jump ball`;
- one-hot last possession-change candidate types `1`, `4`, `5`, and `10`.

Event order is always:

```text
game_id, period ascending, period clock descending, event_num ascending
```

This preserves corrected NBA events whose later event number occurs at an
earlier game clock.

Current-shot outcome, score, player/team identity, all SQ features, all EV
fields, and the transition label itself are excluded. Prior-event outcomes
are allowed because they are known before the current shot and are central to
recognizing a change of possession.

## Generated artifacts

Shot-value artifacts:

```text
nba_pipeline/results/sq_challenger/
  sq_shot_value_challenger_metrics.json
  sq_midrange_oof_predictions.parquet
  sq_three_oof_predictions.parquet
  sq_midrange_anchored_model.joblib
  sq_three_anchored_model.joblib
  sq_midrange_regularized_logit_model.joblib
  sq_three_regularized_logit_model.joblib
  sq_midrange_feature_importance.csv
  sq_three_feature_importance.csv
```

Transition artifacts:

```text
nba_pipeline/results/sq_transition_classifier/
  sq_transition_classifier_metrics.json
  is_transition_oof_predictions.parquet
  is_transition_dunk_or_layup_oof_predictions.parquet
  is_transition_forward_2025_26_predictions.parquet
  is_transition_dunk_or_layup_forward_2025_26_predictions.parquet
  is_transition_historical_safe_model.joblib
  is_transition_dunk_or_layup_historical_safe_model.joblib
```

These are generated model artifacts and stay out of git under the repository
backup policy.

## Remaining gaps

1. Freeze the feature contract and score an untouched rich future season.
2. Audit how the vendor constructed every player/defender prior.
3. Recover canonical game dates for a true chronological split.
4. Obtain independently labeled historical transition samples by era. The
   completed 1997-2026 backfill uses game-cross-fitted calibration in labeled
   2025-26 and explicitly retrospective pooled calibration before 2025; it
   does not establish 30-year stationarity.

Primary-source design rationale is in
[Public Shot-Quality Methods](./SHOT_QUALITY_PUBLIC_METHODS.md).
