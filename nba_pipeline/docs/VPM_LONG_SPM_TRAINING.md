# Long-sample statistical VPM prior

Owner authorized October15,2026. Immutable sibling experiment
`research_harness_v44_long_spm.json` / `vpm_long_spm_v1` implements the
[declared plan](VPM_LONG_SPM_PLAN.md). Existing v18, repaired v39, zero-prior
v42 and failed direct-game-loss v43 remain separate controls. No2027 outcomes
or model promotion.

## Result and decision

Owner-requested follow-up: [five previous/new VPM blends and seven-test results](VPM_LONG_SPM_BLENDS.md), separate immutable v46.

**Player-board dates:** the v44 evaluation board dated **October 1, 2025**
excludes2025–26 and must not be presented as a current2026 ranking. A separate
current emission now exists: immutable sibling **v45**, `long-spm-current`,
cut **September 1, 2026**. It refits the frozen six-season calibration on
2021–2026, keeps v44's selected core/4000/8000/TD550/scale1, and uses the exact
dated September1 skills, original entrant priors/blend and repaired SQ atoms.
No hyperparameter selection, future evaluation or2026–27 outcomes enter it.
The original monthly board supplies entrant metadata/priors; its old SQ target
values are not imported. This is not the separate September14 frozen-parent
variant quoted in earlier comparisons.

Wembanyama is **+2.5515 offense, +6.9966 defense, +9.5481 total**, rank2 among
recent NBA players. His own prior is+1.8264/+6.3271. Matched repaired zero-prior
at this same cut is+1.5776/+2.5801; reusing the previous dated own prior with
repaired SQ gives+2.4495/+3.8074. These are exact-cut comparisons, unlike the
older October2025 backtest list. The current estimate is a research board,
not a new performance confirmation or certified next-season roster.

Current artifacts live in sibling
`vpm/results/harness/vpm_long_spm_current_2026_09_01_v1/`:
`ratings.parquet`, `player_rankings.md`, `manifest.json`, and the saved
calibration/adapter validation receipts. Reproduce with:

```bash
.venv/bin/python ../vpm/vpm/run_harness.py \
  --config ../vpm/vpm/configs/research_harness_v45_long_spm_current.json \
  long-spm-current
```

**Retain as a research challenger; do not replace current VPM.** All seven
2023–2026 stress point estimates improve over current VPM. Six paired error
intervals cross zero; the low-continuity team win-percentage gain is positive
but has only13 team-seasons across four season blocks. This is not evidence
of a general improvement across seven independent tests. EPM still has lower
all-game and team-net errors. Compared with zero-prior VPM, all seven stress
paired error intervals favor the new model. The common-native possession
check also favors it over zero-prior, while its difference from current VPM
is unresolved.

|2023–2026 stress test|Long SPM|Current VPM|EPM|Lagged DARKO|
|---|---:|---:|---:|---:|
|Game margin R²|0.1866|0.1856|0.1889|0.1752|
|Low-continuity game R²|0.2957|0.2949|0.2981|0.3044|
|Team win% R²|0.6691|0.6609|0.6890|0.6490|
|Low-continuity team win% R²|0.6201|0.5606|0.4860|0.5936|
|Team net RMSE (lower)|3.2352|3.2892|3.1200|3.4102|
|Veteran game R²|0.1763|0.1753|0.1795|0.1653|
|Veteran team win% R²|0.6457|0.6365|0.6665|0.6215|

Game RMSE is13.9604 vs current13.9659 and EPM13.9415. Long-SPM minus current
performance, expressed as reference MSE minus candidate MSE, is+0.1528
[−0.8705,+1.1982]. Team net RMSE is3.2352 vs current3.2892 and EPM3.1200;
its paired gain over current is+0.3522[−0.7561,+1.1818]. The low-continuity
team-win error gain is+0.002499[+0.000744,+0.003646]. These errors share one
model and overlapping games, so they are not independent replications.

The legacy2021–2026 mixed table has game RMSE13.9990 vs current13.9928;
team net RMSE3.1256 vs current3.1861 and EPM3.0839. Do not substitute that
mixed block for the declared stress comparison.

On423,644 common-native stress possessions from4,273 games, the new model's
actual-point MSE gain over current is−54.06e−6[−147.17,+38.80]e−6, unresolved;
over zero-prior it is+205.80e−6[+30.38,+386.17]e−6. This diagnostic excludes
many rookie-involving lineups; all-player future-minute game tests remain
primary. No feature or strength changes were made after inspecting stress.

All44 focused tests pass. Fourteen real calibration fits pass independent
raw-count, preprocessing and full-covariance reconstruction; current-control
deployment ratings reproduce exactly, and every incumbent field forecast is
preserved. Mutating all post-cut targets/counts changes calibration
coefficients by exactly0. Independent scripts reproduce the selection,
all seven scores in development/stress/mixed periods,49 stress paired MSE
point estimates, source hashes and direct actual-point diagnostics.

## Architecture

The mean calibration uses the preceding six complete NBA seasons, one pooled
O/D coefficient per player, free season intercepts, actual FG+FT targets and
uniform possession weights. The2020 emission calibrates on2014–2019;
subsequent windows roll one year, through2020–2025 for the2026 emission.
Each window has1.48–1.51million rows and974–1088 players. Four incoherent
2014/2015 games are excluded from both lineups and matched player statistics;
see the [source audit](VPM_LONG_SPM_SOURCE_AUDIT.md). Player-side exposure in
those matched statistics equals lineup exposure exactly for every player.

For lineup matrix X, free season effects N, observed-stat design F and
residual player penalties Lambda=(2000 offense,4000 defense), minimize:

`||y - X beta - N gamma||² + (beta - F theta)' Lambda (beta - F theta) + 0.1 ||theta||²`.

The mapper profiles the player and season coefficients jointly. It retains
full O/D and teammate covariance. Saved pure zero-prior six-year RAPM is an
audit artifact; the mapper does not regress on independently weighted shrunken
point estimates. This is a covariance-aware long-period SPM candidate, not a
reproduction of undisclosed EPM fitting details or a calibrated Bayesian
posterior. Box evidence and lineup outcomes share games.

Calibration covariates are count sums divided by their matching exposure
sums on exactly the admitted games, RS+PS. The twelve offensive inputs match
the current parent; defense compares blocks/steals/DREB with the existing
seven-field extension. At least2000 side possessions are required for a
player's calibration feature direction; everyone remains in the lineup
likelihood. Missing rim/DFGA measurements stay missing. Rim saved points use
the same strictly earlier-game-day cumulative league reference as the dated
skill estimator. PF uses total O+D exposure; usage is events per100 offensive
possessions, despite its `pct` suffix. Makes equal attempts times percentage.

Observed-window means/scales, coherent shooting imputation and missingness
indicators are fitted only inside the calibration window. Features are
centered on its side-specific exposure reference, with free lineup season
intercepts. There is no additional EPM-style league-stat normalization. The
same coefficients are applied to existing dated neutral skill forecasts,
centered across the full current fitted universe, at output scale1. Raw
long-period statistics and estimated current skills have different errors
and context; coefficient transfer is an empirical assumption tested forward.

Entrant models and `max(1-career_possessions/2000,0)` blending stay fixed.
The existing actual-point/SQ-parent deployment is retained, including dated
shooter talent inside repaired SQ targets, actual FT, pre-cut continuation
prices, mid/three residual penalties12000/48000, fills and home adjustments.

## Development selection

Only2020–2022 game margins select the feature family and deployment evidence
settings. Loss is the mean of each season's mean squared game error. For each
candidate, paired SE against the minimum is
`sqrt(sum_s(var(game_loss_difference_s)/n_games_s) / n_seasons²)`.
Choose core defense when it is within its paired one SE of the extension.
Then compare27 combinations of offense2000/4000/8000, defense4000/8000/16000,
and decay275/550/1100 days. Among candidates within their paired one SE of
the minimum, prefer the smallest sum of absolute log2 distance from
4000/8000/550, then lexical candidate ID. No prior-scale or mapper-alpha grid.

The family minimum was extended defense (MSE189.3938), but core was within
one SE (difference0.4606, SE0.4750), so **core** was retained. The grid minimum
was4000/4000/275 (189.6337); **4000/8000/550** remained within one SE and was
retained. Core MSE was189.8544. The fixed entrant-only diagnostic was188.4697:
the own-skill SPM did not demonstrate a development increment beyond that arm.
It is a disclosed diagnostic, not a post hoc replacement or new selection arm.

Selection is saved before stress scoring. Annual coefficients refit on the
then-completed six seasons; family, ridge and deployment settings remain
frozen through2023–2026. Those years are report-only historical stress for
this candidate, already spent by the wider project, not a fresh holdout.
The legacy2021–2026 table mixes development and stress. All field systems
share actual future minutes and pace; these measure conditional player quality,
not a complete pregame availability forecast. Historical model publication
and source availability vintages remain uncertified.2026–27 stays untouched.

## Reproduction

From the RAPM root, using its numerical environment:

```bash
.venv/bin/python ../vpm/vpm/run_harness.py \
  --config ../vpm/vpm/configs/research_harness_v44_long_spm.json \
  long-spm --stage all

.venv/bin/python ../vpm/vpm/scripts/validate_long_spm.py \
  ../vpm/vpm/results/harness/vpm_long_spm_v1 \
  --benchmark-root ../vpm/vpm/results/harness/vpm_long_spm_v1

.venv/bin/python ../vpm/vpm/scripts/audit_long_spm_possessions.py \
  ../vpm/vpm/results/harness/vpm_long_spm_v1
```

Stages are `prepare`, `calibrate`, `tune`, `boards`, `evaluate`, `audit`, `all`.
Input/code/config locks prevent resuming with changed evidence. Generated
private calibration matrices, boards, predictions and model JSONs stay out of
git. The small curated evidence summary is tracked separately. See the sibling
output's `benchmark_report.md`, `scorecard.csv`, `paired_comparisons.csv`,
`selection.json`, `chronology_audit.json`, `independent_validation.json`, and
`actual_point_diagnostics.json` for exact results and receipts.
