# WNBA Daily Net-DECOMP Alpha Tuning

Status: development research, not promoted to the live WNBA board  
Last updated: 2026-08-29

## Decision

The imported single-season `5000 / 40000` offense/defense penalties and fixed
550-day decay should not be carried into the multi-year WNBA daily model.

On the canonical WNBA net-continuation substrate, the rolling-origin grid
selects:

- base offense alpha: `2000`
- base defense alpha: `5000`
- half-life: no decay over the available five-season window

The old `5000 / 40000 / 550` configuration scores `1.225812334` mean
game-weighted future-possession MSE.  The selected pure net model scores
`1.225305899`, an improvement of `506.435e-6`.  The old configuration is not
within the paired one-standard-error set.  The unconstrained grid minimum is
the same `2000 / 5000` base with a 1100-day half-life at `1.225265655`; no
decay is the simpler longest-history candidate within one paired SE.

This is a development result.  The 2026 season has already been inspected, so
the next untouched promotion gate is 2027.  The August 29 dev-only rerun below
confirms the no-decay selection on 2024/2025 folds but adds a strong
2026-specific decay reversal; see "August 29, 2026 rerun".

## Canonical target contract

The tuning surface uses the current `FIRST_CHANCE` and
`SECOND_CHANCE_CLEAN` net-continuation rows.  It separates the neutral scoring
baseline from direct turnover value rather than leaving both inside one
completion atom:

1. rim frequency
2. rim conversion
3. midrange frequency
4. midrange conversion
5. three-point frequency
6. three-point conversion
7. FT value
8. direct turnover value
9. neutral net baseline
10. net second chance

The ten row targets sum exactly to actual possession points.  Defense retains
the raw solver points-allowed convention inside the fit and is flipped only
for positive-good display.

The tuner is `wnba_test/tune_wnba_daily_net_hyperparams.py`.  It uses exact
`game_date`, hashes every source file, checks lineup/key alignment across
first- and second-chance rows, and rejects any selection input from 2027.

## Validation design

Rows cover 2022-2026 regular seasons and playoffs.  The non-overlapping
season-forward folds are:

- train 2022-2023, validate 2024
- train 2022-2024, validate 2025
- train 2022-2025, validate 2026

Each solve uses all completed earlier seasons with candidate time decay.  The
score is the mean of game-level possession MSE, first averaged within season
and then equally across seasons.  The paired one-standard-error calculation
clusters at game level.

Completed-season folds are deliberate.  The current processed files estimate
zone continuation value (`kappa`) from the whole source season.  Reusing one
of those files for an earlier same-season historical cutoff would leak future
continuations.  A literal daily backfill must instead rebuild kappa from games
available through the cutoff or freeze prior-season kappa for the next season.

The current lane is pure zero-centered RAPM.  The annual box priors are
full-season artifacts and are not safe inputs to an arbitrary historical
daily cutoff.  A later prior-centered challenger needs true preseason or
historical-as-of prior snapshots and must be tuned separately.

## ShotQuality target split

`wnba_test/build_wnba_sq_net_targets.py` maps the chronological first FGA to
each terminal possession row using raw action order.  It does not attempt the
invalid direct shot-event to ending-event join.  For first-chance midrange and
three-point attempts:

```text
net actual = actual points + miss * kappa(zone)
net SQ EV  = shot EV + (1 - p(make)) * kappa(zone)
residual   = net actual - net SQ EV
```

The adapter replaces the old net conversion child with SQ EV and carries the
residual as a new child.  Rim and every non-shot target remain bit-identical.
The rebuilt 2022-2026 target files close to the old net shot tree at machine
precision.

Two season-forward EV fields were evaluated:

- `shot_context_ev`: context-only, no player outcome history
- `shot_quality_prior_ev`: context plus strictly prior-season shooter residual

A common alpha cannot reveal any SQ benefit: ridge is linear, so solving
`EV + residual` with the same penalty is algebraically identical to solving
actual conversion.  The grouped test therefore holds the selected base at
`2000 / 5000` with no decay and tunes the two actual-minus-SQ residuals
separately.

### Context-only SQ result

The conservative one-SE candidate is:

- base: `2000 / 5000`, no decay
- mid/three residual offense alpha: `40000`
- mid/three residual defense alpha: `160000`

Its MSE is `1.225188586`, a `117.313e-6` gain over the selected unsplit net
model.  The paired game-cluster standard error is `82.434e-6`; the result wins
in 2025 and 2026 but loses in 2024.  The 95% interval therefore crosses zero.
Treat this as a promising low-sample target-denoising candidate, not a
promotion.

The unconstrained residual optimum is `10000 / 20000` at MSE `1.225160967`.
The much larger conservative defense penalty applies only to the noisy
actual-minus-EV residual; it does not imply that the base defensive RAPM
should use heavy shrinkage.

### Prior-shooter SQ result

The prior-shooter target scores `1.225183675` after its own residual-alpha
selection, a `122.224e-6` gain over unsplit net.  It is only `4.911e-6` better
than context-only with paired SE `23.598e-6`; the two are statistically tied,
and their season ordering is mixed.  Prefer context-only as the cleaner
development candidate because the shooter-history layer overlaps
conceptually with future box priors, not because it won or lost this tiny
point-estimate difference.

Missing SQ rows safely retain the original actual target and a zero residual,
but a production daily pipeline must refresh SQ before emitting the current
board.  The shot-level parquet and both SQ target families were refreshed on
August 29, 2026 (294 games through 08-28, frozen `150/75/250` smoothing,
closure gap `4.4e-16`); rerun both builders whenever `Processed/` moves.

## August 29, 2026 rerun — dev-only selection, decay mandate, refreshed SQ

The original selection above used 2024-2026 validation folds.  This rerun
re-selects on the 2024/2025 dev folds only (matching the predictive-VPM lane
protocol), holds 2026 as report-only, and runs every family twice: once on
the full half-life grid and once with the owner-mandated fixed 550-day decay.
Artifacts: `wnba_daily_net_tuning/sq_gate_{hl550,fullgrid}/` and
`sq_gate_report2026_{hl550,nodecay}/` (pinned singleton grids for the report
passes, so nothing is selected on 2026).

Dev selection (2024+2025, game-clustered paired deltas vs the `net` family
at the same decay spec):

- Full grid: the honest one-SE selection stays `2000 / 5000`, no decay, for
  all three families.  Decay is monotonically worse on the dev folds for
  `net`: none `1.221254`, 1100d `1.221330`, 730d `1.221339`, 550d
  `1.221368`, 365d `1.221468`.  `net_sq_context` (residuals `80000/80000`)
  is `-92.6e-6 [-296.6, +118.4]` vs net, 2025-driven (2024 `+79.0`, 2025
  `-264.1`); `net_sq_prior_shooter` (residuals `40000/160000`) is
  `-45.1e-6 [-239.8, +150.4]`.
- Mandated 550: every family re-selects `1000 / 5000` (decay shrinks the
  effective sample, so the less-shrunk offense wins).  `net_sq_context`
  (residuals `80000/20000`) is `-129.2e-6 [-353.4, +94.6]` vs net and
  negative in both dev years (2024 `-61.8`, 2025 `-196.7`);
  `net_sq_prior_shooter` is `-92.8e-6 [-307.3, +124.4]` with a positive
  2024.  Context-only remains the cleaner SQ candidate, and under decay its
  gain becomes year-consistent instead of 2025-only.

2026 report-only (spent/descriptive, paired by game, n=294):

- The decay mandate wins 2026 decisively: `net@550` vs `net@none` is
  `-422.1e-6 [-656.0, -188.0]` — the only interval in this batch excluding
  zero.  Per-year, no-decay is better in 2024 (`-203e-6`) and 2025
  (`-149e-6`) but much worse in 2026.  This matches the fresh-blend phase-2
  finding that 2026 carries unusually high recency value (its pooled vertex
  `5.40` exceeded the dev seasons'), and the spent-2026 recency observation
  behind the current board's `decay550` sensitivity column.
- SQ on 2026 is spec-dependent and inconclusive: `+99.7e-6` (context vs net
  at 550), `-117.4e-6` at no decay, `-242.1e-6 [-488.2, +11.9]` for
  prior-shooter at no decay.  Best-vs-best (`ctx@550` vs `net@none`) is
  `-322.4e-6 [-693.6, +32.1]`, driven by the decay term.

Verdict: no promotion.  Every dev-fold SQ interval crosses zero, the only
decisive interval lives on spent 2026, and the pre-registered rule keeps
`net`, `2000 / 5000`, no decay as the frozen candidate.  The 550-decay
SQ-context configuration (`1000 / 5000 / 550`, residuals `80000/20000`) is
recorded as the owner-preferred exploratory config; adopting it into any
surface is an explicit owner decision, and 2027 is the untouched
confirmation for the decay reversal, the SQ split, and their combination.

## Commands and artifacts

Build isolated SQ target families:

```bash
.venv/bin/python wnba_test/build_wnba_sq_net_targets.py \
  --years 2022-2026 --ev-column shot_context_ev

.venv/bin/python wnba_test/build_wnba_sq_net_targets.py \
  --years 2022-2026 --ev-column shot_quality_prior_ev
```

Tune the pure net and SQ families:

```bash
.venv/bin/python wnba_test/tune_wnba_daily_net_hyperparams.py \
  --target-family net

.venv/bin/python wnba_test/tune_wnba_daily_net_hyperparams.py \
  --target-family net_sq_context

.venv/bin/python wnba_test/tune_wnba_daily_net_hyperparams.py \
  --target-family net_sq_prior_shooter
```

Generated target and tuning receipts are gitignored under:

- `wnba_test/Results/wnba_net_sq_targets/`
- `wnba_test/Results/wnba_daily_net_tuning/`

Each tuning directory contains `report.json`, `frozen_candidate.json`,
`grid_scores.csv`, `fold_audit.csv`, and selected game-level scores.  SQ runs
also contain `residual_grid_scores.csv`.

The small committed handoff spec is
`wnba_test/config/wnba_daily_net_development_candidate.json`.  It keeps SQ
disabled and labels the base configuration as a development candidate rather
than a live promotion.

## Production seam

Andrew's daily orchestration and reader remain useful, but its WNBA adapter
currently consumes a different gross target family and borrowed NBA alphas.
Do not paste these numbers into that engine without changing its input
contract.  The safe integration is:

1. rebuild causal-as-of net targets in this repo;
2. refresh the repo-owned SQ table when the SQ candidate is enabled;
3. solve the exact structural target family documented here;
4. emit a source-hashed daily board/spec;
5. let the existing daily site/history shell ingest that board.

Promotion requires a preregistered 2027 future-possession result.  Until then,
the live single-season prior-centered WNBA board remains unchanged.
