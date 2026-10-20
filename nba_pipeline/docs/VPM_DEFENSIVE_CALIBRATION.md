# Current Parent VPM: defensive calibration and uncertainty

Owner authorized October20,2026: execute the proposed defense-only mapper
regularization, feature-block removals, role/minute/teammate-change calibration,
prior-confidence tests and uncertainty inspection. Then make current VPM fully
decomposable into Net DECOMP lanes. This document freezes the defensive tests
before fitting; the separate lane protocol follows source/identity inspection.

## Frozen defensive experiment

Control: current v53 forward_expanded_mpg_a10p0. Preserve dated neutral skills,
exact original imputation/scales/column centering, mature future-label cohorts,
entrant values/weights, actualFT, both SQ corrections12000/48000, TD550,
common game/minute census and original benchmarks. No2027 outcomes.

Fix each emission's original offensive mapper coefficients. Use the same
profiled future-lineup likelihood; refit only defensive coefficients with
mapper ridge10/30/100/300. This changes defensive prior means, not final lineup
penalties. At ridge10, original coefficients/priors must reproduce exactly.

At ridge10 also refit three separate block-removal candidates, removing both
numeric and missingness columns: contestedDREB; steals+blocks; rim savings+
rim contests+DFGA. Preserve the original remaining-feature preprocessing.
These are conditional fits with the same fixed offensive prior, not new
shooting feature definitions or position recentering.

Stage1 selects literally minimum equal-season2020–22 gameMSE among control,
three stronger penalties and three removals, lexical ties. No rating-spread or
player-name criterion. Stage2 freezes that selected prior and compares final
D precision8000/4000/2000 (O4000 unchanged), still on2020–22 only. Display all
nine unique deployed candidates and all seven paired tests on spent2023–26.
No large factorial search, repeat selection on later targets or automatic
promotion. Existing historical choices make this retrospective development,
not a new untouched confirmation. Jokić/Reed are diagnostic examples only.

## Diagnostic contract

Use the existing423644 actual future possessions/4273 native games and
incumbent-filled forecasts. Report native actual-point and prior-only errors
alongside future game/team tests. Segment future errors using strictly pre-cut
MPG, dated role composition and observed pre-cut team-change metadata; missing
metadata remains an explicit category. Use all eligible players/lineups,
not only the two inspected names. Calibration regressions are descriptive
held-forward diagnostics, never refitted player ratings or causal attribution.

For current/selected current-cut defense, report game-cluster wild-bootstrap
intervals from the exact joint deployment solve, including jointly perturbed
actual/mid/three target residuals and the weighted-centering correction.
Freeze learned mapper/entrant parameters and source measurements for this
bootstrap; label its intervals **conditional on those priors/parameters**,
not full model uncertainty or calibrated Bayesian confidence. Shared game
multipliers preserve correlated target errors. Report all recent players plus
Reed/Jokić, selected paired differences, and normal-equation checks. Do not
claim an absolute prior-influence percentage from prior/posterior proximity.

Independent audits must reconstruct the conditional mapper, original-control
parity, raw joint solves, score/paired points, dates/group metadata and bootstrap
linearity. Keep generated boards/models/caches ignored and commit curated
summary/source hashes and tests. Preserve all old readers; create a single
clearly named defense comparison rather than replacing current ratings.

## Status

Complete: all nine comparisons, context diagnostics, conditional uncertainty, independent audit and reader. See the completed receipt below.

## Completed result — October 20, 2026

Both development stages select `dmap_a100`, with final O/D precision
4000/8000 and TD550 unchanged. Stage two rejects weaker defensive precision.
Offensive prior coefficients stay fixed; final offense can move through the
joint solve. No output scaling, player-specific correction or promotion.

| Arm | Development game RMSE | Stress game RMSE | Stress team-net RMSE |
|---|---:|---:|---:|
| Current | 13.740987 | 13.895289 | 3.091360 |
| Defensive mapper 30 | 13.733133 | 13.892724 | 3.093214 |
| Defensive mapper 100 | 13.726933 | 13.888081 | 3.095796 |
| Defensive mapper 300 | 13.727752 | 13.888862 | 3.110697 |
| Remove contested DREB | 13.732930 | 13.891230 | 3.079643 |
| Remove steals + blocks | 13.756071 | 13.887233 | 3.077940 |
| Remove rim block | 13.735505 | 13.879656 | 3.096593 |
| Selected prior, D precision 4000 | 13.735852 | 13.889817 | 3.095990 |
| Selected prior, D precision 2000 | 13.769199 | 13.899152 | 3.105038 |

All seven paired stress comparisons against current are reported. Positive
MSE gain favors the candidate; game intervals use season-stratified game
clusters and team intervals use whole-season blocks.

| Test | MSE gain | 95% interval |
|---|---:|---:|
| Game | +0.200258 | [-0.029018, +0.443310] |
| Low-continuity game | +0.012799 | [-0.670265, +0.747457] |
| Team win percentage | -0.000012 | [-0.000092, +0.000067] |
| Low-continuity team win percentage | -0.000364 | [-0.000540, -0.000130] |
| Team net rating | -0.027445 | [-0.113719, +0.112517] |
| Veteran game | +0.201764 | [-0.028408, +0.441436] |
| Veteran team win percentage | -0.000006 | [-0.000089, +0.000078] |

The 423,644 native future possessions/4,273 games give final actual-point
MSE gain +14.70e-6 [-6.65e-6, +36.06e-6], also unresolved. Low-continuity
team win percentage worsens. Removing the rim block has the best later game
point estimate but was not the development choice; do not select it now.

At the September 1 cut, Jokić D moves 3.017349→2.594514 and Reed
3.594742→3.430329. Their own-skill D priors move 2.620965→2.056429 and
4.168528→3.960766. This reduces the concern modestly; it does not establish
which player estimate is correct.

### Calibration and conditional uncertainty

The pre-cut MPG, dated role-composition and observed prior-year team-change
segments produce 24 diagnostic rows. Overall held-forward D calibration
slope is current 0.9710 [0.8564, 1.0856], candidate 1.0380 [0.9155, 1.1604],
controlling for offense and season intercepts. This does not support a blanket
claim of defensive inflation. Group differences are diagnostics, not causal
role effects or new fitted player corrections.

Four hundred game-cluster wild draws perturb actual/mid/three residuals
jointly over 14,128 training games. Conditional Jokić intervals are current
[2.4134,3.6239], candidate [1.9906,3.2011]; Reed [3.0428,4.1605] and
[2.8784,3.9961]. Learned priors, skills, preprocessing and selection are fixed;
these omit their uncertainty. Identical design/precision gives identical
conditional noise widths, so these are not confidence intervals for the
prior-change benefit. Direct perturbation solves match within 1.7e-13.

Independent reconstruction verifies 679 sources, 56 conditional mappers,
72 boards, 154 scores and 168 paired rows. Prior/rating discrepancies are
below 7e-15; all nine control forecasts reproduce. Four focused tests pass.
Reader: http://127.0.0.1:8767/defensive-calibration.html; 582 recent/1,556 full
players, priors/finals, seven tests, groups and conditional intervals. Browser
search, group controls and downloaded CSV checked against saved artifacts.

### Reproduction and source-clean follow-up

From the sibling VPM repo, run its `research_harness_v59_defensive_calibration.json`
with `defensive-calibration --stage all`, then
`vpm/scripts/audit_defensive_calibration.py`,
`vpm/scripts/diagnose_defensive_calibration.py` and
`vpm/scripts/render_defensive_calibration.py` on the experiment root. The
curated receipt is `vpm/evidence/defensive_calibration_summary.json`.

These v59 comparisons preserve the original entrant estimator. The subsequent
exact-Net build discovered its two quarantined 2015 training games; v61 cleans
that source for both current and the candidate, freezes the above choices,
and reports changed forecasts separately in VPM_NET_LANE_DECOMPOSITION.md.
Do not mix the archived v59 scores with the cleaned-board claim of parity.

## Owner selection — October 20, 2026

After reviewing the ratings and mixed forecast evidence, the owner explicitly
selected `dmap_a100`: “Let’s do the 100 mapper.” Adopt the already-audited v61
clean-source arm as the local default. This is an owner decision; do not claim
new untouched confirmation or a resolved later predictive win. No new fitting,
manual player adjustment, public deployment or 2027 outcomes.

Sibling `vpm/configs/current_model.json` is the selection pointer. Rebuild local
views with `vpm/scripts/render_selected_ratings.py`: ratings.html, skills.html,
Net decomposition, current_vpm_full.csv and current_vpm_recent.csv all use the
same selected clean board. Offense mapper10, defense mapper100, parent4000/8000,
TD550 and SQ residual12000/48000 remain fixed. The skills renderer independently
reconstructs all1556 own priors with zero error; input forecasts/feature sets
are unchanged.582-row browser CSV matches the selected saved board;582/1556
controls, selected defaults and Jokić row checked. Earlier four-model ratings
remain at ratings-forward-archive.html; prior mapper10 remains a named comparison.

Selected current Jokić O6.837923/D2.594525/VPM9.432448; Reed
O0.218392/D3.430304/VPM3.648697. The source-repair/Net audits remain applicable.
Curated selection receipt: sibling vpm/evidence/selected_mapper100_summary.json.
