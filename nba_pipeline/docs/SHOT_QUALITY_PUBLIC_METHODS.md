# Public Shot-Quality Methods: Transferable Design Notes

## Scope

This note reviews primary public sources that are useful for the repo-owned
ShotQuality challenger. It is intentionally narrower than a literature review:
the question is which pre-shot features and validation claims can be defended
with the data that survived locally.

## What the public methods support

### Separate the physical shot problem into coherent families

The NBA's official xFG explainer says its current system is player agnostic,
uses pose-derived features from the one second before release, and fits
separate models for jump shots and finishes. Its disclosed feature families
include shot distance; shooter tilt, head angle, speed, velocity to the basket,
rotation, and balance/air-time measures; and defender joint distance, relative
contest height, contest angle, vision interference, hip distance, velocity,
and pressure. It explicitly excludes shot clock, player identity, and score
differential from that player-agnostic xFG model.

Source: [NBA, “Intro to Expected Field Goal Percentage”](https://www.nba.com/news/intro-to-expected-field-goal-percentage).

Transferable decision:

- keep the current midrange and three-point challengers separate;
- use release-context geometry and motion rather than outcome descriptions;
- publish a context-only model if the intended interpretation is “average
  shooter in this physical situation”;
- treat a model containing player shooting priors as a different,
  player-informed estimate, not as player-agnostic xFG.

### Location and defender geometry have nonlinear, region-specific effects

Franks, Miller, Bornn, and Goldsberry model shot make probability using shot
region, shooter skill, defender identity, and defender distance. In their
tracking sample, the relationship between defender distance and make
log-odds is approximately linear only until roughly five or six feet, after
which additional distance adds little. Their full spatial/defensive model
beats simpler no-defense and no-spatial models on out-of-sample
log-likelihood under 10-fold cross-validation.

Source: [Franks et al., “Characterizing the spatial structure of defensive skill in professional basketball”](https://arxiv.org/abs/1405.0231).

Transferable decision:

- do not force defender distance into one global linear coefficient;
- allow region and shot-family interactions;
- regularize player and defender effects rather than treating small samples
  as stable skill;
- benchmark by held-out probability loss, not by an in-sample fit statistic.

### Shot selection and shot conversion are different targets

Miller, Bornn, Adams, and Goldsberry represent shot selection with a
low-dimensional spatial decomposition. The discovered components correspond
to recognizable shot-location habits and can also support shooting-accuracy
models.

Source: [Miller et al., “Factorized Point Process Intensities: A Spatial Analysis of Professional Basketball”](https://proceedings.mlr.press/v32/miller14.html).

Transferable decision:

- do not call a transition-frequency classifier a shot-make model;
- retain transition, transition-rim, shot-location, and make-probability
  outputs as distinct targets;
- use the transition classifier to extend a shot-context label historically,
  then use that label as an input or reporting slice in a separate xFG model.

### Post-release trajectory is valuable, but it is not pre-shot quality

Daly-Grafstein and Bornn show that shot trajectory plus defender height,
distance, and contest angle can stabilize perimeter-defense estimates. Their
trajectory work is useful evidence that defender geometry matters, but
post-release trajectory features describe execution after the shooting
decision and therefore cannot enter a strictly pre-shot xFG target.

Source: [Daly-Grafstein and Bornn, “Using In-Game Shot Trajectories to Better Understand Defensive Impact in the NBA”](https://arxiv.org/abs/1905.00822).

Transferable decision:

- defender height, distance, and angle are valid pre-release candidates;
- entry angle, depth, left/right miss, live EV, and other post-release
  trajectory outputs must remain outside the challenger feature set;
- if post-release data is modeled later, label it shot execution or
  make-quality, not pre-shot shot quality.

## Validation and calibration contract

The experimental unit is a shot, but shots from one game share lineups,
opponents, arena conditions, and event sequence. Random shot-level folds would
put those shared conditions on both sides of the split. `GroupKFold` guarantees
that a group occurs in exactly one test fold, so `game_id` is the outer-fold
group in both challenger scripts.

Source: [scikit-learn `GroupKFold` documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html).

Game-group cross-validation still mixes early and late games. A second check
therefore holds out the last 20 percent of ordered game ids. This is not a
perfect date split, but it points in the same direction as time-series
validation: do not train on future observations and evaluate on the past.

Source: [scikit-learn `TimeSeriesSplit` documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).

For a probability model, discrimination alone is insufficient. The benchmark
reports log loss, Brier loss, ROC AUC, precision-recall AUC, and a quantile-bin
calibration error. Brier and log loss are proper probability scores, but each
combines calibration and discrimination, so calibration bins and mean
prediction versus observed frequency are reported separately.

Source: [scikit-learn probability calibration guide](https://scikit-learn.org/stable/modules/calibration.html).

Every “better than `initial_ev`” claim must satisfy all of these conditions:

1. the challenger and `initial_ev` are scored on identical non-null rows;
2. no game crosses the train/test boundary;
3. the primary comparison is paired loss on out-of-fold predictions;
4. uncertainty is clustered by game;
5. the ordered-game holdout does not show a major reversal;
6. calibration and discrimination are both reported.

## Mapping the recovered feature store

The recovered rich feature store under the sibling `shotquality` checkout
contains one season, 2024-25: 1,138 RS/PS game CSVs and 203,181 field-goal
rows. This is not three rich tracking seasons. The 2025-26 RAPM raw parquets
retain ShotQuality descriptors, transition labels, and `initial_ev`, but not
the full motion/defender feature store. No 2023-24 or second rich-feature
season survives locally.

The rich 2024-25 columns map well to the public methods:

- location: `shot_x`, `shot_y`, shot distance, NBA zone;
- shot family: pre-shot descriptor tags such as catch-and-shoot, pull-up,
  drive, step-back, floater, transition, and turnaround;
- pressure: `contest_score`, spacing, nearest/second-nearest defender
  distance, inside/behind/blocking flags;
- physical context: defender height/weight and bounding-box height;
- occupancy: defenders and teammates in the paint/restricted area;
- motion: shooter and first/second-defender velocity and acceleration,
  including motion toward the basket or shooter;
- player-informed priors: shooter and closest-defender three/midrange/jumper
  priors.

These are ShotQuality-provider fields. NBA matchup defender assignments are a
separate provider and must not be coalesced into one defender identity.
ShotQuality provides second-defender geometry but no native second-defender
ID; any repo biometric reconstruction is inferred and can duplicate the
primary defender. Provider comparisons therefore need separate NBA-only,
SQ-only, agreement/high-confidence, and geometry variants. The cross-repo
contract is documented in
[NBA Event-Enrichment Contract](./NBA_EVENT_ENRICHMENT_CONTRACT.md).

The feature-only challenger excludes all EV fields. The anchored challenger
adds exactly `initial_ev / shot_value` and lets the allowlisted pre-shot
features learn a residual correction. Both exclude `dynamic_ev`, `live_ev`,
shot-taking outputs, current-shot make/miss, points, assist, block, rebound,
score differential, and identifiers.

## Claims the current data cannot support

- It cannot establish multi-season tracking generalization; there is only one
  rich season.
- It cannot establish that a player-prior model is player agnostic.
- It cannot establish that 2024-25 tracking relationships are stationary over
  the last 30 years.
- It cannot assign causal defensive credit from nearest-defender or on-court
  associations.
- It cannot claim to beat `initial_ev` from training loss, full-sample feature
  importance, or an unpaired row sample.
- It cannot recreate missing licensed tracking features for 2023-24 or
  2025-26 from descriptors alone.

The defensible next validation step is to recover another full-feature season
or acquire a forward season, freeze the feature contract and hyperparameters,
and score that season without refitting.
