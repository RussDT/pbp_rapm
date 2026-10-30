# Forward-trained VPM prior means: first implementation

Implemented October 15, 2026 after owner approval of the
[architecture recommendation](VPM_FORWARD_TRAINED_ARCHITECTURE.md).
Immutable sibling config `research_harness_v43_forward_prior_mean.json`,
experiment `vpm_forward_prior_mean_v1`; canonical command `forward-prior`.

**Result: the first forward-trained mean candidate is rejected as a
replacement for the existing label-trained SQ parent.** It improves game
prediction over zero-prior VPM and actual-point RAPM, but loses to the existing
prior VPM on both future games and common-native future actual possessions.
No model is promoted or published; 2026–27 remains untouched.

## What was implemented

The first stage changes how own-skill prior coefficients are learned. It holds
the v39 repaired SQ targets, feature list, own-feature preprocessing, TD550,
parent4000/8000, residual12000/48000, entrant model and blend, missing-player
fills, game construction and seven-test scorer fixed. The skill prior's
amplitude is learned directly; it receives no additional x1.5 multiplier.
The x1.5 label-trained control is preserved and reconstructed exactly.

At fixed lineup penalties, the estimated coefficients are affine in the prior
means. With `A = X'WX + Lambda`, the response to a prior direction `F` is
`A^-1 Lambda F`. Factorize A using the existing exact sparse ridge solver;
solve for all 38 O/D feature directions together without an explicit inverse.
Preserve the existing strong-minus-weak SQ residual correction and the
prediction-preserving, full-universe side-wise exposure centering.

For each historical preseason date, this produces an exact game forecast:

```
predicted margin = fixed_background_and_lineup_forecast + game_design @ theta
```

The fixed part includes the dated entrant prior and the SQ solve. Game design
uses actual future minutes/pace only in the shared conditional training loss;
those values never enter the checkpoint skills, priors or possession fit.
Missing-player fills stay fixed. Learn theta by minimizing equal-season actual
game-margin MSE plus `alpha * ||theta||²`. This outer problem is convex and
solved exactly by regularized normal equations. Finite differences validate
the derivative through the inner O/D solve and the future game objective.

Offense and defense use their existing separate feature recipes. A net-margin
objective alone does not certify the O/D allocation; common-native future
actual-possession error is an additional diagnostic. No new synergy, minutes,
availability, context forecast or shotmaking-residual prior is introduced.

## Training dates and selection

Added a 2018-10-01 warm-up control using the original dated own-prior mapper,
actual-only label package and 2018 entrant model. It has 714 fitted players,
uses only pre-cut rows and predicts 1,230 games in 2018–19. The remaining seven
checkpoints reuse the exact repaired v39 board universes and preprocessors.
Every own-skill observation and parameter-fit cutoff precedes its checkpoint.

For emission season Y, fit only on completed seasons before Y whose last game
precedes the emission cutoff. Thus 2020–21 uses 2018–19 and 2019–20 training
forecasts; 2025–26 uses 2018–19 through 2024–25. Training has fewer historical
windows than the label-trained mapper; this is a limitation, not a matched
test of statistical efficiency with every possible earlier source recovered.

The saved covariate imputation, coherent shooting construction, missing flags,
standardization and centering are reused from each date's control. These
preprocessors were fitted using matured historical label cohorts and capped
future-exposure weights, but **no RAPM response values or mapper slopes enter
the new learner's response or regularization**. The new model is therefore
not fully independent of the old label-based *preprocessing cohort*. Preserve
that qualification when describing this stage as a response-method change.

Within each completed training prefix, expanding-year inner folds select from
the registered mapper penalty grid `0.001, 0.01, 0.1, 1, 10`. Use the strongest
penalty within paired one standard error of the minimum equal-season loss.
With two or more validation seasons, use whole-season paired differences;
the first single-validation-season choice uses paired game errors. The 2019–20
extension board has only one training season, so it uses the registered
warm-up penalty1 without selecting on its own outcomes.

| Forecast season | Prior-training seasons | Selected mapper penalty |
|---|---|---:|
| 2019–20 (extension) | 2018–19 | 1 (fixed warm-up) |
| 2020–21 | 2018–19 through 2019–20 | 1 |
| 2021–22 | 2018–19 through 2020–21 | 10 |
| 2022–23 | 2018–19 through 2021–22 | 10 |
| 2023–24 | 2018–19 through 2022–23 | 10 |
| 2024–25 | 2018–19 through 2023–24 | 10 |
| 2025–26 | 2018–19 through 2024–25 | 10 |

This is a chronological reconstruction of a retrospectively chosen method.
All 2020–26 evaluation years remain spent development evidence. The architecture
and source-availability caveats of the repaired benchmark still apply.

## Seven-test result

Same 7,230 full games / 180 team-seasons, 2020–21 through 2025–26; low continuity
853 games / 20 team-seasons. Higher R² is better; lower RMSE is better. R² is
squared correlation, with direct forecast R² saved separately.

| Test | Forward-trained VPM | Existing prior VPM | Zero-prior VPM | EPM |
|---|---:|---:|---:|---:|
| Game-margin R² | .1744 | .1789 | .1600 | .1785 |
| Low-continuity game R² | .2892 | .2860 | .2562 | .2850 |
| Team win-percentage R² | .6178 | .6379 | .5907 | .6544 |
| Low-continuity team R² | .3811 | .3865 | .3905 | .4092 |
| Team net-rating RMSE | 3.3399 | 3.1861 | 3.4807 | 3.0839 |
| Veteran game R² | .1624 | .1676 | .1575 | .1684 |
| Veteran team win-percentage R² | .5888 | .6121 | .5768 | .6320 |

Game RMSE is **14.0476**, versus existing VPM **13.9928**, EPM **14.0085**,
zero-prior VPM **14.1726**, and actual-point RAPM **14.1752**. The RAPM reference
uses zero priors, actual FG+FT targets, TD550 and 2000/4000 penalties on the same
past rows and player universe; it has no SQ residual correction.

Positive paired MSE gain favors the new candidate:

| Target / reference | MSE gain [95% interval] |
|---|---:|
| Games / existing prior VPM | −1.5354 [−2.2348, −0.8312] |
| Games / zero-prior VPM | +3.5299 [+2.6563, +4.4423] |
| Games / actual-point RAPM | +3.6013 [+2.3425, +4.9120] |
| Games / EPM | −1.0970 [−2.7391, +0.5502] |
| Team net rating / existing prior VPM | −1.0034 [−1.8000, −0.3215] |
| Team net rating / EPM | −1.6442 [−2.9304, −0.3623] |

The narrow low-continuity game R² point lead does not establish an improvement;
direct low-continuity game error is worse than the existing model, with an
interval crossing zero. The new candidate loses game MSE in five of the six
headline seasons. The 2025–26 RMSE loss is largest (14.8637 versus14.6703).

The selected regularization heavily suppresses own-skill coefficients: their
norm is about0.12 in the latest model versus1.28 for the control in the same
standardized coordinates. That is an observed behavior, not proof that the
feature effects are zero or an explanation established by a separate test.

On **600,243 future RS possessions** in accepted games with all ten players
native in the shared fitted universe, new actual-point MSE is1.41440064 versus
control1.41411816. Paired gain is **−282.48e-6 [−350.11, −212.91]e-6**. Against
zero-prior VPM the gain is+21.47e-6 [−40.25,+83.53]e-6, unresolved. This diagnostic
does not invent an O/D split for unseen players and is not the full-game census.

## Verification

- 37 focused tests pass, including exact derivatives/finite differences,
  equal-season weighting, nested selection, future-outcome exclusion, coherent
  feature centering, primary-system aliases and existing target/benchmark tests.
- Eight checkpoints reconstruct every saved own prior and control rating;
  maximum control game-forecast difference is below6e-14.
- Independently fit all seven new models with augmented least squares rather
  than the implementation's normal equations; coefficient differences are
  below1.2e-15.
- Independently verify all seven headline scores and paired MSE directions,
  all incumbent game/team predictions, source/artifact hashes, full clock mass,
  score/win accounting, veteran overrides and the protected-season exclusion.
- Perturb current/future training packets' scores, forecast bases and designs;
  all earlier selected penalties, coefficients and emitted rating responses
  remain exactly unchanged. Reuse the upstream repaired SQ/own-skill chronology
  evidence without claiming this new test independently refits the shot models.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  ../vpm/vpm/run_harness.py \
  --config ../vpm/vpm/configs/research_harness_v43_forward_prior_mean.json \
  forward-prior --stage all

OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  ../vpm/vpm/scripts/validate_forward_prior.py \
  ../vpm/vpm/results/harness/vpm_forward_prior_mean_v1
```

Stages `prepare`, `fit`, `evaluate`, `audit` may also run separately. Immutable
source/code/config locks reject changed inputs. Outputs remain private/ignored:
affine forecast packets, cut-specific preprocessors and priors, nested-fold
selection receipts, seven dated boards, full predictions, paired intervals,
native-possession diagnostics, temporal audit and completion/validation receipts.

## Scope and decision

The implemented first stage is **forward learning of own-prior means**.
Heterogeneous prior precision and an explicit evolving impact-residual filter
are not implemented by this receipt. Existing lineup correction and TD550
remain in place. The full proposed architecture is not complete.

Preserve this failed candidate and the stronger current model. A one-year RAPM
target being noisy is not evidence that this direct game-loss alternative is
better. Further architecture work must explain the loss, recover any additional
strictly dated training history, and retain explicit chronological selection
and untouched confirmation. Do not patch this scorecard into a promotion.
