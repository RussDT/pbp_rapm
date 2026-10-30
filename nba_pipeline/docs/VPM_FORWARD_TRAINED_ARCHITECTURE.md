# Forward-trained predictive VPM architecture

October 15, 2026. Owner requested a concrete architecture after questioning
the variance of the 365-day RAPM prior-training target. This is the recommended
design, not an implemented or selected model. Existing v18, v39/v40 and v42
artifacts remain controls. It occupies the existing NBA VPM research scope.

## First-stage implementation receipt

The owner-authorized [v43 prior-mean learner](VPM_FORWARD_PRIOR_TRAINING.md) is
implemented and audited. It loses to the existing label-trained control on
future games and actual possessions and is rejected as a replacement. The
full architecture below, including heterogeneous precision and explicit
temporal residual filtering, remains incomplete. This is not a selected model.

## Decision

Build a hierarchical, time-aware SQ-parent VPM whose skill-to-impact means
and shrinkage are learned through its lineup solve against subsequent actual
game outcomes. Keep the simple offensive/defensive parent as the forecasting
backbone, with exact SQ EV/residual accounting. Learn component detail only
when the summed forecast benefits. Retain pure 365-day RAPM as a diagnostic
and baseline training method, rather than the new learner's response variable.

The key change is the training objective. Actual possessions are themselves
noisy. Direct fitting does not magically remove that noise. It preserves the
lineup information and lets the model learn the prior that makes its deployed
estimator forecast better, instead of imitating an already-shrunk player label.

## Model contract

1. **Dated evidence.** Exact-ID NBA, college/G League and draft evidence, with
   source availability, exposure, missingness, fitted-parameter cutoff and
   estimate date. Keep opportunity and conversion separate. Retain makes and
   attempts together. Source gaps have explicit coverage states.
2. **Neutral skill forecasts.** Estimate skill means, reliability, age-related
   movement and staleness from earlier data. Different skills may have different
   dynamics. Portable talent and opponent/rest/team context remain separate.
   A same-game statistic cannot enter that game's pregame skill packet.
3. **Impact mean and precision.** Map the packet to offensive and defensive
   prior means. Keep a compact regularized linear mapper first. Include entrant
   population, draft and lower-league evidence in the same predictive contract.
   Prior strength is a separately parameterized, positive quantity in the
   actual weighted solver's units. Coverage, source reliability and staleness
   can change it, with strong pooling and bounds. Career minutes alone must
   not permanently penalize a player's mean. O/D strengths remain separate.
4. **Lineup correction.** Fit actual observed lineups with priors inside the
   regression. Preserve the v39 EV plus mid/three residual construction,
   shooter talent within SQ, actual FT and pre-cut continuation prices. Start
   shotmaking residual prior means at zero. Use identifiable past-only league,
   venue and margin-state nuisance terms shared across comparisons.
5. **Temporal impact residual.** In the complete model, portable impact is
   `skill-predicted impact + persistent lineup residual`. Residual persistence
   and drift are learned separately from skill dynamics; elapsed time can
   increase forecast uncertainty without manufacturing new observations.
   Age changes the projected skill/impact mean; trades need not reset talent
   to zero. Estimate any trade-specific uncertainty or role effect rather than
   hard-coding a universal improvement/decline.
6. **Forecast.** Sum offensive and defensive player values over lineups, then
   aggregate to game margin. Keep availability, minutes, pace and game context
   as explicit forecast inputs. A preseason frozen board and a daily updated
   board are different emission protocols from the same model family.

## First implementation: train through the existing solve

Reuse the exact factorized ridge interface in the sibling VPM harness. With
historical checkpoint c, write the parent solve schematically as:

```
mu_c = mapper(dated_skills_c, dated_background_c; B)
beta_c = argmin_beta [
    (y_past - X_past beta)' W_c (y_past - X_past beta)
    + (beta - mu_c)' Lambda_c (beta - mu_c)
]
vpm_c = beta_c + existing_strong_minus_weak_SQ_residual_correction
```

The formula omits the existing centering/intercept/nuisance terms only for
readability. Preserve them, their units, and their prediction identities in
the implementation. The mean mapper and shrinkage parameters are fit using:

```
loss = mean_over_seasons(mean_over_future_games(
           (actual_margin - forecast_from_vpm_c)^2
       )) + regularization_on_mapper_and_shrinkage_parameters
```

Actual future lineups/minutes/pace are permitted only in the explicitly labeled
conditional player-quality test, identically for every system. They never
enter checkpoint features or the inner fit. A separate full pregame test uses
only projected availability/minutes/pace known before tipoff.

For the immediate seven-test request, the first challenger must use the same
frozen preseason emission dates and forecast construction as v40. Train its
parameters on completed earlier seasons using nested chronological folds.
Daily replay is a separate subsequent protocol; do not tune next-game forecasts
and describe that optimization as a frozen preseason result.

The current 4000/8000 parent, 12000/48000 residuals and TD550 are starting
controls, not architecture-level truths. First isolate the new mean-training
criterion at those settings; then fit a small pooled precision model. Explicit
temporal residual filtering follows in the same architecture after those
interfaces reproduce the baseline. This order identifies regressions and
avoids changing means, precision, decay and feature families in one opaque run.

The sparse normal equations can be differentiated implicitly; use the existing
factorization for linear solves, never an explicit matrix inverse. For fixed
penalties, the derivative of beta with respect to a prior direction dmu is
`(X'WX + Lambda)^(-1) Lambda dmu`. Validate gradients by finite differences
on a small synthetic lineup system before a historical replay.

## Uncertainty and component discipline

Separate uncertainty in the prior prediction, current lineup estimate, future
talent movement, and the game outcome. Skills and lineup data share games;
do not multiply their evidence as if independent or add overlapping empirical
error estimates twice. Initially describe inference as empirical Bayes and
calibrate errors on chronological predictions. Tuned penalties are not a
certificate of exact posterior coverage.

Carry lineup coefficient covariance, or joint game-resampled draws, when
forming O+D, team contrasts and component sums. Observation count alone does
not resolve players who always share the floor. Correlated skill features need
regularization; correlated components must not be treated as independent
confirmations of the same impact signal.

Retain exact actual = EV + residual target closure and exact player O+D/tree
closure. Narrow component priors are a separately gated extension of this
same parent architecture. Do not invent component shares to make a total
forecast look decomposed. Shooter talent already in SQ is not a license to
fit an additional unvalidated shotmaking residual prior.

## Training, evaluation and operator boundary

- Fit every skill filter, age curve, transfer, imputation/scaling rule, mapper,
  precision and forecast calibration using the training prefix of its fold.
  Forward training outcomes must be complete by the next model-emission date.
- Use completed seasons as outer time blocks. Within the earlier training
  prefix, optimize the forward objective with regularization. Evaluate the
  complete selected procedure on later blocks; in-sample forecast loss is not
  evidence of improvement. Preserve player/teammate dependence in uncertainty.
- Main selection target: conditional game-margin MSE on the frozen protocol.
  Keep summed actual-possession loss, calibration slope, all seven field
  tests and exposure/continuity/coverage slices visible as diagnostics. Do not
  create seven independently tuned models for seven favorable table entries.
- Skill/context modeling may use raw event evidence in RAPM, but VPM receives
  only the pinned talent exports and canonical possession targets through the
  existing repo boundary. Complete EPM and eWINS remain excluded from priors.
- Compare against repaired prior VPM, matched zero-prior VPM, actual-point
  RAPM and dated EPM. All evidence through 2025–26 remains development/spent.
  Freeze the complete procedure before any 2026–27 outcome is used.
- Reuse one sibling harness and immutable config/output IDs. This document
  changes no model, benchmark definition, public board or promotion status.

## Primary references and scope of inference

[EPM's official methodology](https://dunksandthrees.com/about/epm) describes
estimated skills informing a statistical prior inside a time-decayed RAPM.
[DARKO's official methodology](https://www.darko.app/about) describes
stat-specific decay, filtering and aging. These support the broad skills and
lineup backbone; neither page establishes that the proposed training objective
will beat our present model.

[Bengio, Gradient-based optimization of hyperparameters, 2000](https://pubmed.ncbi.nlm.nih.gov/10953243/)
describes differentiating a validation criterion through an inner learner.
Its use here with chronological basketball forecasts is our design proposal.
Our prior/no-prior comparison establishes useful prior signal, not the
optimality of any proposed replacement.

## October 15 EPM methodology clarification

Owner asked whether EPM first forecasts/selects skills, fits stable impact
weights retrospectively, and then calibrates the lineup/prior balance. The
[official EPM methodology](https://dunksandthrees.com/about/epm), checked October
15, distinguishes these tasks:

- Individual statistical forecasts are optimized to predict each player's next
  game in that statistic. That establishes predictive skill estimation, not
  documented feature selection for incremental future impact.
- SPM coefficients are calibrated with matching-period player statistics and
  multi-season RAPM. The stated base sample is 2001–02 through2018–19; a
  separate tracking SPM uses 2017–18 through2022–23. The page does not fully
  specify label aggregation, all feature-selection decisions or optimizer order.
- Apply those impact weights to the dated predicted skills to form the SPM
  prior, then use it inside exponentially decayed career RAPM. The exact
  prior-penalty/lineup-decay selection procedure is not published there.

The owner's proposed separation is consistent with those disclosed stages;
forward impact feature selection and subsequent forward tuning of the evidence
balance are plausible choices, not confirmed details of Snarr's implementation.
His [2020 comparison](https://dunksandthrees.com/blog/metric-comparison) separately
reports that adding direct RAPM improves over the statistical prior, but it is
an older version and does not document the current tuning recipe.

A long multi-year impact calibration is a meaningful missing control: establish
which dated skills add forward signal, estimate their impact conversion using
long-run lineup evidence with matching covariates, and tune prior precision
and decay on chronological future predictions. A future-game promotion
objective does not require every coefficient to be fitted directly to game
margin. The recommendation above to replace the response with direct game
risk was too categorical as an architectural prescription. V43 is one failed
fixed-precision mean-learning candidate; it neither tests this longer-target
SPM design nor decides which full architecture is optimal. No new model was
fit or selected during this clarification.
