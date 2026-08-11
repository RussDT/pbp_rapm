# Preregistration: WNBA Cold-Start / Coverage Priors

Frozen August 10, 2026. Phase 3 of `WNBA_BETTING_PROGRAM.md`.

## Problem

20-23 percent of 2026 WNBA player-slots are unrated (scored league-average);
only about 10 percent of possessions have all ten players rated, versus about
99 percent coverage for DARKO. The gap concentrates in season openers,
post-draft debuts, call-ups, and the 2026-27 expansion rosters — exactly the
market's softest windows — and creates adverse selection for any
edge-thresholded bet selector.

## Builds

1. **NBA-to-WNBA hierarchical partial pooling** for box-to-component prior
   slopes, per the existing transfer contract: align atomic targets,
   standardize within league-season, fit NBA slopes, shrink WNBA slopes
   toward them with component-specific transfer strength chosen on WNBA
   folds. Candidates compared: WNBA-only, NBA-only, naive pooled,
   hierarchical. RTSS creation-role and geometry residuals stay WNBA-only
   (no men's rich-markings surface with the identical schema exists locally).
2. **NCAA-W-to-WNBA translation lane** adapting the unified-rapm
   crosswalk/translation machinery. Expected value per its own evidence:
   little accuracy gain for rated players, real gain in cold-start
   uncertainty for debuts. Scope: rookie-season priors and uncertainty
   widths, not veteran rating changes.

## Binding evaluation

- Development: 2022-2025 WNBA folds (rookies and low-coverage player-seasons
  as the test strata).
- Endpoints: (a) component-prior RMSE on first-half-season component RAPMs
  for debut players; (b) joint component-sum future-possession MSE with the
  cold-start priors filling previously unrated slots; (c) forecast-log
  calibration on low-coverage games.
- 2026 is not used for selection. The 2027 season (expansion debuts) is the
  natural confirmation.

## Interim fence

Until these ship and pass: hard no-bet filter on games where either team's
projected rotation has below-threshold rated-minutes coverage (threshold set
in development, logged with every forecast).

## Amendments

(none)
