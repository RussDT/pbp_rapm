# Role-aware defensive prior versus current forward Parent VPM

Owner authorized October19,2026: test a role baseline plus individual skill
deviations against the retained current forward Parent, and show ratings and
predictive results. Same NBA VPM research slot. Current means v53
`forward_expanded_mpg_a10p0`, not the unselected rim-makes addition or Component
VPM. Preserve all existing boards and the no2027 boundary.

## Frozen before fitting

Role source is the cached, exact-ID completed-season position archive2017–2026.
Read position, minutes and availability metadata only; BPM/OBPM/DBPM, EPM and
impact outputs never enter role construction. Historical publication vintages
remain uncertified. For each dated own-skill cut, take at most two latest
available seasons, cap source age730 days, and average guard/wing/big membership
with weights minutes×0.5^(latest_source_year−source_year). PG/SG map guard,
SF/PF wing, C big; hyphenated hybrids share membership equally. A player can
therefore have soft shares across seasons. These are position-history roles,
not optical defensive assignments or matchup-derived archetypes. Before2017,
new entrants and uncovered players have missing roles, not zero-valued roles.

Use the original O/D feature set, original coherent imputation and standard
scales, and exactly the original matured365-day future-lineup training cohorts.
Fit predictor-only role means on eligible (>=500 future side possessions)
training rows, weighted by defensive exposure, with5000-possession shrinkage
toward the pooled predictor mean. Unknown roles use training-population shares
for this decomposition and a separate missingness indicator. No current/future
rating labels are used to construct roles, predictor means, or scales.

Compare two fixed role architectures plus the exact current control:

1. **Role baseline + shared skill deviations:** learned role baseline plus
   shared coefficients on predictors minus their role-expected values.
2. **Partially pooled role slopes:** the same baseline/shared terms plus
   role-weighted slope deviations, ridge100 (ten times the shared penalty).
   Unknown roles receive the shared slope only. Role baseline/shared/O blocks
   retain ridge10. Baseline shares/missing flag are training-standardized;
   skill residuals and interactions retain original physical feature scales.

Center full bases on cohort/deployment defensive exposure, never within each
position at emission. Thus between-role defensive value remains learnable.
Full-universe O/D centering and free lineup intercepts retain the existing
reference convention. Wholly unmeasured own-skill sides remain neutral.
All O/D mapper coefficients are fitted jointly through the full profiled
future-lineup likelihood; no player-specific haircut or desired ranking.

Freeze residual2000/4000, outputscale1, final4000/8000/TD550, actualFT, repaired
SQ/residual12000/48000, entrants, census, minutes/fills and all seven tests.
No rim-makes addition, lineup role interaction, role grid or penalty search.
Select literal minimum equal-season2020–22 gameMSE among control/two roles;
lexical ties. If null wins, still show the better role candidate as research.
Freeze choices through2023–26 spent stress. Report paired gains versus current,
previous repaired and EPM, plus native future actual-point/prior-only errors
and a defensive-only crossed forecast using control offense. Conditional actual
minutes/pace and retrospective source availability remain limitations.

Use a shared-coefficient reconstruction to prove the role design spans the
original prior exactly; show role-baseline/within-role/slope-deviation prior
contributions. Independently audit source roles, chronology, profile algebra,
null reproduction, player-minute/season scores and paired points before
presenting ratings. Keep generated artifacts ignored and curated evidence small.

## Completed results, October19

Development selects **partially pooled role slopes** within this three-way
comparison. Retain the current forward Parent after the later diagnostics.
This does not revise the earlier v53 family selection or choose Component VPM.

|Prior|2020–22 game RMSE|2023–26 game RMSE|2023–26 team-net RMSE|
|---|---:|---:|---:|
|Current split features|13.740987|13.895289|3.091360|
|Role baseline/shared deviations|13.734791|13.894658|3.092136|
|Partially pooled role slopes|13.734402|13.897586|3.105902|

Selected-role paired MSE gains against current (positive favors role-aware):

|Later test|Gain|95% interval|
|---|---:|---|
|Game margin|−0.063844|[−0.326068,+0.197560]|
|Low-continuity game|+0.934634|[+0.154403,+1.729268]|
|Team win percentage|−0.000067|[−0.000222,+0.000028]|
|Low-continuity team win|+0.000383|[+0.000041,+0.000725]|
|Team net rating|−0.090121|[−0.241401,+0.009377]|
|Veteran game margin|−0.026681|[−0.283703,+0.222678]|
|Veteran team win|−0.000057|[−0.000218,+0.000049]|

On423,644 native future actual possessions/4,273 games, selected-role final
MSE is1.414960351 versus current1.414933460. Paired gain is
−26.89e−6[−53.58,−0.60]e−6. Prior-only gain is
−38.01e−6[−82.43,+4.82]e−6. With current offense and its past-fitted intercept
fixed, selected-role defense gives−30.97e−6[−57.20,−5.04]e−6: worse. The shared
variant's final and defense-only intervals cross zero. These unadjusted,
overlapping historical diagnostics support no replacement; low-continuity
gains are not an overall win. Actual future minutes/pace and retrospective
architecture/source availability limit the interpretation. No2027 outcomes.

At the September1 cut,580/582 positive-exposure recent players have roles,
12 have mixed shares. The full1556-player model has871 missing roles; missing
coverage among retired/early-era players is expected, not zero membership.
Source roles are absent before2017. Current selected-role examples:

|Player|Current own prior D|Role own prior D|Current final D|Role final D|Current total|Role total|
|---|---:|---:|---:|---:|---:|---:|
|Nikola Jokic|2.621|2.431|3.017|2.897|9.875|9.743|
|Victor Wembanyama|5.228|4.669|6.380|6.053|9.083|8.767|
|Giannis Antetokounmpo|1.284|1.703|2.198|2.483|7.036|7.312|

Jokic's centered prior decomposition is1.318728 role baseline+0.932175 shared
skill deviation+0.180539 role slope deviation=2.431443. These are conditional
fitted contributions, not causal values or a role-specific zero reference.

## Verification and operation

Immutable sibling v55 output is `vpm/results/harness/vpm_role_aware_prior_v1`.
Independent checks pass:684 source files,16 fitted mappers,70 score rows,
42 paired points; maximum role-transform gap2.00e−15, profile gap0, original
prior span gap2.78e−17, score gap7.11e−15. Source reconstruction checks5,489
archived player-seasons,6,516 literal player mixtures and prefix invariance;
only position/minutes/availability fields enter, never archived BPM values.
All8 control boards reproduce; game forecasts exactly, team forecasts within
1.78e−15. Three source tests and three role tests pass, plus reused filter
tests. Generated fits, sources, logs, boards and CSV/HTML stay ignored;
curated sibling evidence is `vpm/evidence/role_aware_prior_summary.json`.

From RAPM, build the source with `.venv/bin/python
nba_pipeline/scripts/export_vpm_dated_roles.py`. From sibling VPM:

```sh
../rapm/.venv/bin/python -m vpm.harness.cli --config vpm/configs/research_harness_v55_role_aware_prior.json role-aware --stage all
../rapm/.venv/bin/python vpm/scripts/audit_role_aware.py vpm/results/harness/vpm_role_aware_prior_v1
../rapm/.venv/bin/python vpm/scripts/render_role_aware.py vpm/results/harness/vpm_role_aware_prior_v1
```

Use [role details](http://127.0.0.1:8767/role-aware.html) for both role variants
and prior decomposition, or the [combined comparison](http://127.0.0.1:8767/prior-comparisons.html)
for current/role/all-input ratings. Readers show582 positive-exposure recent
players or1556 full-universe players, search/sort and downloadable views.
Role model/breakdown/universe checks and combined search/sort/582-row CSV
values pass in browser; original ratings and numerical artifacts are preserved.
See the separate [unsplit-feature test](VPM_UNSPLIT_FEATURE_PRIOR.md).
