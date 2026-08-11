# Preregistration: WNBA Player-Prop Distributional Model

Frozen August 10, 2026. Phase 3 of `WNBA_BETTING_PROGRAM.md`. Development may
begin any time; the confirmation split below is binding.

## Product definition

Role-conditional distributional projections of player PTS, REB, AST (and
derived combos) per game, conditioned on the minutes/availability simulation
from `wnba_test/project_wnba_minutes.py`. The edge hypothesis: WNBA prop
lines are largely per-game-average heuristics, and the widest per-bet edges
occur on lineup-change days when role reallocation moves the conditional
distribution, not just the mean.

## Data split (binding)

- Development: 2023-2025 (all spent seasons; free to iterate).
- Shadow confirmation: 2026 — frozen model emits pre-tip prop distributions
  into the same append-only forecast log as Phase 0, scored against realized
  box lines (and against captured prop odds when an odds source exists).
  No 2026-informed refitting during the shadow.
- Live: 2027, only if the shadow calibration passes.

## Model skeleton (development guidance, not binding)

- Counts: negative-binomial (or Conway-Maxwell if dispersion demands) per
  stat, mean = f(projected minutes, on-court role rates, opponent context),
  dispersion fit per stat on development seasons.
- Role rates from the component surfaces: creation labels (`assistOpp`
  potential-assisted vs self-created), on-ball time, per-zone frequency
  priors, rebound-opportunity mechanisms.
- Minutes uncertainty propagates: sample minutes from the allocator's
  distribution, then counts conditional on minutes.

## Binding evaluation

- Primary: mean pinball loss over the 10/25/50/75/90 quantiles, per stat,
  versus a per-game-average heuristic baseline (trailing-10 mean with
  minutes scaling) on the shadow season.
- Calibration: PIT uniformity per stat; 90 percent interval coverage within
  [0.87, 0.93] on the shadow season.
- Prop-odds scoring (only if odds captured): implied-probability log-loss
  versus the close.

## Fences

- Ratings inputs are as-of; no market inputs into the projection itself.
- The prop pricing layer alone may see the market.
- No proprietary third-party ratings anywhere in the model.

## Amendments

(none)
