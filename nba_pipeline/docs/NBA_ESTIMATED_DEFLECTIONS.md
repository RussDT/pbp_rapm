# Estimated deflections: frozen development protocol

Owner correction: **next-game per-100 accuracy is now primary**. See
[the next-game protocol and results](./NBA_DEFLECTIONS_NEXT_GAME.md).
The remaining-season experiment below is preserved as the named
longer-horizon comparator; its objective is not the current product target.

Registered September 12, 2026, before fitting or opening stress results.

This extends the estimated-skills research with a **deflections per 100
defensive possessions** filter. Deflections measure activity, not defensive
impact. It uses observed hustle counts only; no backfilled PA/DFGA or complete
impact ratings enter. Missing hustle is never converted to zero.

## Data and split

- Gabriel `player_sheets/game_report/year_files` supplies dated hustle counts.
- The existing PBP player-game panel supplies actual defensive possessions.
  Match exact player/date/season/season-type, reject ambiguous duplicate keys,
  require matching assists and FGA and minutes within 0.5, and audit exclusions.
- This first version deliberately uses the established **2019–2026** exposure
  panel. Deflections exist from 2017, but 2017–2018 are not silently mixed in
  with an estimated possession denominator.
- **2019**: warm-up and population initialization/bounds only.
- **2020–2022**: tuning and model selection; development forever.
- **2023–2025**: frozen historical stress, no refitting or winner switching.
- **2026**: state updates/export only; no accuracy scoring or tuning.
- **2027**: protected future confirmation. Historical stress is not new,
  untouched evidence for the overall estimated-skills research program.

## Models and objective

Compare a scalar random-walk Kalman filter with the existing two-component
OU engine (slow player rate plus mean-reverting short-run deviation), using
the same forward objective. Both are uncentered and use only the warm-up
population mean. No age/debut corrections are admitted for this new skill.
This is a v3-style core extension, not the full 17-skill v3 recipe.

Tune the **postgame state** against the next 41 played games within the
season, excluding the current game and crossing neither season boundaries
nor into stress years. Require at least ten future games, 200 observed
defensive possessions, and 95% coverage of future source minutes. Each
development season gets equal objective weight; exposure weights rows within
seasons. Differential evolution uses seed 20260912, 60 iterations, population
multiplier 15, tolerance 1e-7, no polish, and logarithmic positive bounds.
Report convergence honestly; hitting the budget is not convergence.

## Evaluation and selection

At Dec 1, Jan 1, Feb 1, and Mar 1, take each player's last state strictly
before the cutoff and predict the rest-of-season observed deflection rate.
Skip dates before the season begins. Require 200 future observed defensive
possessions and at least 95% tracking/exposure coverage of future source
minutes. All candidates and baselines use identical player/cutoff targets.

Baselines: trailing ten observed games, season-to-date, career-to-date, and
the independently tuned random walk. Primary score is per-100 squared error,
exposure-weighted within cutoff and equal-weight across cutoffs. Report
weighted correlation secondarily and next-game error as a diagnostic.

Use 2,000 paired **player-clustered** bootstrap draws, retaining a player's
overlapping cutoff targets together. Select OU only if its development MSE
improvement over the random walk exceeds one bootstrap standard error;
otherwise retain the simpler random walk. Write and hash the frozen parameter
receipt before the separate stress/export command. Never choose based on
stress performance. Player clustering handles repeated players/cutoffs, not
all shared-game/team dependence; this is not the game-prediction VPM gate.

The posteriors' latent variance is an uncalibrated filter diagnostic, not a
validated player confidence interval or a props distribution. No live
minutes/availability forecast is included.

## Relationship to the existing harness

`build_nba_estimated_skills_v3.py` tunes on 2020–2025 using a next-41-game
target. Its implementation scores the **pregame** estimate against games
after the current game, whereas its cutoff evaluation uses the latest
postgame state. This lane explicitly aligns postgame state with the next
41 games. It does not rewrite the old experiment or change its results.

`eval_estimated_skills_forward.py` implements the four-cutoff rest-of-season
test. `confirm_estimated_skills_2026.py` compared frozen v1/v3 on 2026 with
player-clustered intervals; that season is spent. The tracking-backfill
RMSE is a different, postgame reconstruction check that can use same-game
assists; it is not evidence of pregame skill prediction.

## Completed run

Source panel: 217,163 played rows, 205,519 usable observed rows, 1,261 players,
and 64,892 valid zero-deflection observations. Usable source-minute coverage
is 98.1%–99.3% by season. Twenty ambiguous PBP player/date exposure rows were
excluded before joining; box/minute disagreements are also excluded.

Development selected **OU**: +0.564% MSE improvement over the tuned random
walk, with a 95% player-clustered interval [-0.403%, +1.512%]. The absolute
improvement 0.002362 exceeds its one-SE threshold 0.002093, satisfying the
registered rule despite not clearing a 95% significance threshold. Both DE
runs reached their iteration budgets without formal convergence; this is a
fixed-budget comparison, not a claim of globally optimal parameters.

Frozen 2023–2025 stress, 4,276 player/cutoff cells and 12 cutoffs:

- OU MSE **0.47531** versus random-walk **0.48332** in per-100 rate units.
- **1.656% lower MSE**, player-clustered 95% interval **[0.682%, 2.650%]**;
  wins 9/12 cutoffs.
- By season versus random walk: 2023 +3.892%, 2024 +1.807%, 2025 -0.130%.
  The 2025 difference is inconclusive; do not say OU won every season.
- Weighted correlation with future observed rates: **0.810**.

Raw naive-baseline gains (roughly 80%–83%) are inflated by extremely small
prior samples. DeJon Jarreau's two prior defensive possessions produce an
unshrunk 100-per-100 career rate. Preserve the registered result, but use
`audit_estimated_deflections.py` for the explicitly post-freeze diagnostic
requiring 200+ prior defensive possessions. It retains 4,145 cells and gives:

| Comparator | OU MSE improvement | Player-clustered 95% interval |
| --- | ---: | ---: |
| Trailing ten observed games | 56.33% | 52.99%–59.35% |
| Season to date | 47.66% | 41.27%–54.05% |
| Career to date | 15.33% | 10.97%–19.52% |
| Tuned random walk | 1.70% | 0.66%–2.73% |

This sensitivity did not refit or select a model. It supports the direction
of the result without allowing tiny-history naive estimates to dominate the
reader-facing comparison.

## Estimates and reproduction

`current_states.csv` contains 1,261 last-observed player states, including
581 players last observed in season 2026. Each row has its actual observation
date; the archive ends June 13, 2026. These are last-game posteriors, not
September forecasts advanced through the offseason. Filter `season == 2026`
for the current-season board. The exported latent variance is uncalibrated.

Among 2026 players with at least 1,000 historical observed defensive
possessions: Thybulle 8.56, Dru Smith 7.26, Cason Wallace 7.20, Ausar Thompson
6.90, and Caruso 6.81 estimated deflections per 100 defensive possessions.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  nba_pipeline/scripts/build_estimated_deflections.py prepare
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  nba_pipeline/scripts/build_estimated_deflections.py tune
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  nba_pipeline/scripts/build_estimated_deflections.py stress
.venv/bin/python nba_pipeline/scripts/audit_estimated_deflections.py
.venv/bin/python -m pytest nba_pipeline/tests/test_estimated_deflections.py -q
```

Generated files live under `results/estimated_skills/deflections/` and stay
out of git: source panel/audit, tuning progress, frozen parameters with source
and harness hashes, development/stress cells, stress report, exposure
sensitivity, row predictions, and the player-state CSV. The stress command
refuses changed panel or harness hashes. Preserve existing receipts before
running a revised experiment; do not overwrite the historical result to
silently change selection.

Six regression tests cover target timing, season boundaries, missing coverage,
future-data invariance, measured-only history, cutoff alignment, and the
paired-bootstrap identical-model control. The backfill's separate seven tests
are not the predictive-skill evaluation.

Nothing in this run changes the existing 17-skill v3 exports or any RAPM/VPM
solve. Estimated deflections are available as an isolated research panel;
predictive-impact admission and calibrated props uncertainty remain separate.
