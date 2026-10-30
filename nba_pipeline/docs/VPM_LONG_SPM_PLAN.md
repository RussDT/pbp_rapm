# Long-sample SPM calibration and forward evidence tuning

Implemented and independently validated as sibling immutable v44; see
[implementation, results and commands](VPM_LONG_SPM_TRAINING.md).

Owner authorized October 15, 2026. This replaces the research project's next
action; preserve all original, repaired, zero-prior and failed v43 controls.
No 2026–27 outcomes, publication, model promotion or unrelated cleanup.

## Plan and frozen first specification

1. Recover exact actual-FG/FT canonical possession rows for2014–2015 through
   a RAPM-owned adapter, with source hashes and lineup/game/event/date parity.
   Combine them with existing2016–2026 actual atoms. This supports six complete
   calibration seasons at every2019–2025 preseason cutoff. Keep dated own-skill
   forecasts unchanged; recover same-window observed covariates separately.
2. Fit one pooled offensive/defensive coefficient per player over the preceding
   six completed seasons, with unpenalized season intercepts and uniform
   possession weights. Keep pure zero-prior RAPM2000/4000 as an audit output.
   Learn compact SPM coefficients using the full profiled lineup likelihood
   with those residual penalties. This accounts for shared O/D and teammate
   uncertainty and avoids treating zero-shrunk point estimates as exact labels.
   It is covariance-aware long-window calibration, not a claim to reproduce
   EPM's undisclosed fitting details or literal OLS on saved RAPM ratings.
3. Pair the lineup window with observed count/exposure features on exactly
   its game IDs and RS/PS universe. Fit imputation/scaling in that prefix.
   Require2000 pooled side possessions for a player's own-stat calibration
   direction; all players remain in the lineup likelihood. Include12 current
   offensive inputs and compare core defense(blocks,steals,DREB) with the
   existing7-field defensive recipe. Missing measured tracking remains missing.
   Observed makes/attempts and usage/PF units must match predictive emissions.
4. Apply coefficients to the current, dated neutral skill states. Retain
   coherent shooting, a missing indicator per feature, full-universe prior
   centering, and existing entrant priors with their compact-support blend.
   Mapper ridge is0.1; output scale is1, with no automatic1.5 inflation.
5. Gate the defensive feature extension on2020–2022 future game losses using
   fixed parent4000/8000, TD550. Use equal-season loss and a paired one-SE rule
   preferring the simpler core recipe. Show entrant-only as a diagnostic;
   do not hide failure to add value beyond it.
6. For the selected feature family, tune the independent O/D evidence
   penalties and decay on the same declared development seasons:
   offense2000/4000/8000; defense4000/8000/16000; half-life275/550/1100 days.
   Keep residual12000/48000, SQ, FT, continuation, fills, home and game
   construction unchanged. Select the minimum-loss candidate or the closest
   current-control setting within paired one SE. Closeness is summed absolute
   log2 distance from4000/8000/550, with deterministic candidate-ID ties.
7. Freeze those choices before scoring2023–2026. Refit the six-year
   calibration at each date with only completed evidence; do not reselect
   family or evidence hyperparameters on stress outcomes. Run all seven
   identical field tests, direct actual-point diagnostics and paired errors
   against current VPM, zero-prior VPM, RAPM and dated EPM/lagged comparators.
8. Verify dense/sparse calibration equivalence, all count/rate identities,
   source hashes, prefix invariance of coefficient fits and selection,
   independent game/team scores, and original-control preservation. Write
   immutable config/run receipts and a curated small evidence summary.

## Evaluation interpretation

The2020–2022 seasons are development. The2023–2026 seasons are frozen
historical stress, already spent in the wider research program. The legacy
2021–2026 pooled table mixes development and stress and must be labeled as
such. Actual future minutes/pace are shared oracle inputs in the conditional
quality test. Historical model/source publication vintages remain uncertified.

Raw long-period statistics and neutral predictive skill estimates have
different errors and context; transferring their coefficients is an explicit
empirical assumption evaluated forward. Source/model coverage and calibration
sample exposure are not a certified roster. Never interpret correlated
regularized slopes as independently identified causal action prices.

## Work split

- Root: immutable configs, feature aggregation, harness runner, forward tuning,
  field evaluation and final methodology/result receipt.
- Source audit agent: same-period source contract and actual2014/2015 adapter.
- Numerical core agent: sparse pooled RAPM/profiled SPM calibration and tests.
- Validation agent: independent dense-equivalence and leakage/scoring audits.

Implementation stays in the existing sibling VPM harness; RAPM owns source
row/target adapters. See [source audit](VPM_LONG_SPM_SOURCE_AUDIT.md) and
[prior-training architecture clarification](VPM_FORWARD_TRAINED_ARCHITECTURE.md).
