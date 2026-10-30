# Estimated play-type scoring RAPM

Owner-requested exploratory build, September 13, 2026. The initial receipt is
an actual five-season lineup ridge decomposition, not the illustrative numbers
in the supplied screenshot. This bounded diagnostic is complete; it does not
replace the active predictive VPM work or change any published series.

## What was built

`build_playtype_scoring_rapm.py` fits independent regular seasons 2022–2026
(season ending year), using 1,091,329 FGA with resolved ten-player lineups.
Every FGA appears in every regression. A child's response is actual shot points
when the shot belongs to its category and zero otherwise. The parent response
is actual shot points on all those same attempts. Misses remain zero outcomes
in the shared denominator.

| Display family | Frozen historical category | Interpretation |
| --- | --- | --- |
| Creation-type scoring | SELF_CREATED_2, SELF_CREATED_3 | Team scoring from description-tagged drives, pull-ups, hooks, fadeaways and related shots |
| Finishing-type scoring | CUT_2, OTHER_RIM_2 | Team scoring from cuts and remaining rim attempts |
| Transition scoring | TRANSITION | Team scoring on the frozen transition classifier's labeled attempts |
| Spacing-type shooting | OTHER_3 | Remaining threes after transition and creation classification |
| Other twos | OTHER_2 | Explicit unclassified two-point scoring; not forced into a role |

Transition takes priority. Spacing-type shooting does not establish catch-and-
shoot status, off-ball movement, screens, or gravity. Creation-type scoring
includes teammates' outcomes and does not isolate passing or personal creation.
Finishing-type scoring likewise is not the player's personal finishing talent.
All are lineup-adjusted **team scoring contributions by shot family**.

## Estimator and accounting

Use ten player indicators, a freely fitted intercept, zero priors, and fixed
ridge penalties of offense 2000 / defense 4000. These are initial research
settings, not selected optimal play-type penalties. There is no time decay,
rubberband, or home adjustment. Each season is fitted separately. The sparse
normal matrix is factored once and applied to five children and a separately
supplied parent response. Offense and defense are separately centered by their
on-court FGA exposure; display defense flips once to positive-good.

For each player and each side:

```
creation + finishing + transition + spacing + other_twos = total_fg
```

Units are **points per 100 team FGA**, not per 100 possessions. This is an
FGA-only scoring parent, not full ORAPM, oTS, Net DECOMP, or VPM. FT scoring,
turnover cost, and possession-continuation accounting are absent. Offensive
rebound shots enter as shots without a separate rebound value. Do not add these
coefficients to existing DECOMP children with different denominators.

Targets use exact raw make/miss outcomes, joined one-to-one by normalized game
ID and event number to existing `PLAYTYPE_PROXY_PTS_BACKCAST` categories. The
category pricing column is not the regression target. Unknown labels, missing
outcomes, non-FGA events, unresolved lineups, or duplicate lineup players fail.
Raw corrected duplicate event keys use the existing last-row convention.
Source file SHA256 hashes are recorded in the manifest.

Historical categories are PBP estimates, not observed Synergy history. Earlier
transition labels use retrospective backcasts; this is not an as-of forecast.
See [Historical mechanism backfills](HISTORICAL_MECHANISM_BACKFILLS.md).

## Validation receipt

All five annual builds completed. Largest raw coefficient closure gap:
`9.55e-15`. Three tests cover exclusive allocation, rejection of unknown labels,
and independently solved parent closure after exposure centering. The builder
also checks normal-equation residuals and row-level point closure.

A deterministic split of sorted game IDs into alternating halves tests
same-season repeatability, with at least 1,000 offensive team FGA in each half.
These halves are not a future holdout, and their persistent team/role context
can contribute to correlation. No alphas were selected with this diagnostic.

| Season ending | Creation | Finishing | Transition | Spacing | FG parent |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2022 | .490 | .588 | .460 | .403 | .305 |
| 2023 | .374 | .578 | .396 | .527 | .321 |
| 2024 | .321 | .577 | .323 | .394 | .304 |
| 2025 | .507 | .655 | .411 | .437 | .365 |
| 2026 | .501 | .566 | .466 | .487 | .297 |

These correlations establish some repeatability, not causal identification,
accuracy of the labels, or predictive admission. The direct fit rates Jarrett
Allen highly for creation-type team scoring and places Jokic's 2026 creation
component below zero. That is possible for team shot-family effects and must
not be presented as evidence Allen creates more personally than Jokic.

## Reproduce and inspect

```bash
.venv/bin/python nba_pipeline/scripts/build_playtype_scoring_rapm.py
.venv/bin/python -m pytest nba_pipeline/tests/test_playtype_scoring_rapm.py -q
```

Ignored output directory: `nba_pipeline/results/playtype_scoring_rapm/`.
It contains `annual_rapm.csv`, its parquet companion, source/solver manifest,
`split_game_repeatability.csv`, and a Jokic-only CSV extraction. The main board
contains every player and both offensive and positive-good defensive children.

The next methodological decision is whether the owner wants team play-type
impact (this direct model's estimand) or player role attribution. The latter
requires actor-aware shooter/passer contrasts and a defensible spacing signal;
relabeling a teammate residual as gravity is not sufficient. A full offensive
possession decomposition additionally requires FT, TOV and continuation targets
on a common possession contract. Neither extension has been built here. Keep
2026–27 protected and do not promote these estimates into VPM priors.
