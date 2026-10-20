# All current skills in both O/D prior mappers

Owner authorized October19,2026 while the separate role-aware comparison was
being finalized. Test the retained forward Parent’s existing offensive and
defensive inputs in both mappers. This is a separate matched experiment;
role-aware priors and the unselected rim-makes addition are not bundled into it.

## Fixed protocol before fitting

Control is `forward_expanded_mpg_a10p0` from v53, with its original saved
preprocessing, coefficients, priors, boards and forecasts preserved. New
candidate `all_features` gives both sides the same ordered union of current
numeric inputs (24 distinct fields at current cut, including MPG once per
side) and their missing indicators. No raw totalDREB/totalPF, extra derived
percentages, rim-DFGM or role shares are added. Shooting probabilities support
coherent makes imputation in BOTH mappers without independent coefficients.

Each historical prefix uses the active input union from that prefix’s saved
control; fields with fewer than25 observed eligible training rows remain
unavailable. Fit training-only side-specific imputation/standardization with
original >=500 future side-possession eligibility and capped5000 exposure
weights. A side may now receive information from the opposite skill family;
wholly unmeasured union rows remain neutral. This coverage change and the
refitted preprocessing are part of the explicitly unsplit specification.

Use the same fully matured365-day future-lineup cohorts, profiled residual
2000/4000, mapper10, scale1 and free cohort intercept. No extra mapper/penalty
search. Final4000/8000/TD550, actualFT, repaired SQ/residual12000/48000,
entrant models/blend, census, minutes/fills and seven-test definitions stay
fixed. Compare literal minimum equal-season2020–22 gameMSE against control;
freeze through2023–26 spent stress. Do not select using Jokic movement or
later scores. No2027, release promotion or existing-board overwrite.

Independently verify profile equations, original-control coefficient/rating
reproduction, coherent D-side shooting imputation, seven-test/paired points
and native future total/prior/defense-only errors. Present this comparison
beside the completed role-aware test on the existing8767 review surface, with
plain names and clear controls. Keep generated artifacts ignored.

## Completed results, October19

The all-input candidate wins its fixed2020–22 development comparison, then
loses all seven later game/season comparisons against current. **Retain the
current split-feature forward Parent.** These are Parent prior experiments;
Component VPM and the older repaired control remain distinct.

|Prior|2020–22 game RMSE|2023–26 game RMSE|2023–26 team-net RMSE|
|---|---:|---:|---:|
|Current split features|13.740987|13.895289|3.091360|
|Role-aware, selected pooled slopes|13.734402|13.897586|3.105902|
|All inputs in both mappers|13.711644|13.913997|3.163449|
|Previous repaired Parent|13.724173|13.965886|3.289191|
|EPM benchmark|13.784609|13.941515|3.120003|

All-input paired MSE gains against current, positive favors all-input:

|Later test|Gain|95% interval|
|---|---:|---|
|Game margin|−0.520266|[−0.943424,−0.104865]|
|Low-continuity game|−1.712041|[−3.192368,−0.245486]|
|Team win percentage|−0.000289|[−0.000475,−0.000059]|
|Low-continuity team win|−0.000906|[−0.002317,−0.000136]|
|Team net rating|−0.450899|[−0.741669,−0.213925]|
|Veteran game margin|−0.639365|[−1.061705,−0.242128]|
|Veteran team win|−0.000330|[−0.000522,−0.000090]|

Intervals are unadjusted and tests overlap; these are not seven independent
replications. The2023–26 outcomes were already spent. Forecasts use actual
future minutes/pace; historical publication vintages/architecture remain
uncertified. No2027 outcomes or automatic release choice.

On423,644 native future actual possessions/4,273 games, all-input final MSE
is1.414941831 versus current1.414933460. Paired gain is
−8.37e−6[−48.88,+31.50]e−6, unresolved. Prior-only MSE is1.415634445 versus
1.415538267, gain−96.18e−6[−180.69,−13.32]e−6, worse.

One narrower crossed diagnostic is favorable: hold current offense and its
past-fitted intercept fixed, use all-input defensive ratings. MSE becomes
1.414885211, gain+48.25e−6[+18.75,+78.33]e−6. This is a spent diagnostic,
not a separately trained/selected defensive mapper or evidence that the
combined unsplit model wins. A conditional defensive fit remains untested.

At the September1 cut:

|Player|Current prior O/D|All-input prior O/D|Current final O/D|All-input final O/D|Current total|All-input total|
|---|---|---|---|---|---:|---:|
|Nikola Jokic|4.863 /2.621|4.206 /3.027|6.857 /3.017|6.715 /3.062|9.875|9.777|
|Victor Wembanyama|2.544 /5.228|2.007 /4.909|2.703 /6.380|2.433 /6.092|9.083|8.525|
|Giannis Antetokounmpo|4.430 /1.284|4.232 /0.993|4.839 /2.198|4.787 /2.031|7.036|6.818|

## Verification and operation

Immutable sibling v56 output is `vpm/results/harness/vpm_unsplit_feature_prior_v1`.
Independent audit passes:683 source files,8 fitted mappers,56 score rows,
21 paired points; maximum preprocessing gap8.88e−16, profile gap0, null
coefficient gap1.22e−15, scoring gap7.11e−15. All8 original control boards
reproduce, game forecasts exactly and team forecasts within1.78e−15.
Two unsplit tests pass, alongside three role and three reused filter tests;
RAPM's three role-source tests also pass (11 focused checks total).

Current union24 numeric inputs per side means96 mapper columns including
flags. Historical2020 has22 active inputs;2021–22 have23;2023 onward24.
Feature lists, original and refitted preprocessors and coefficients are saved
by prefix. Curated sibling evidence is
`vpm/evidence/unsplit_feature_prior_summary.json`; generated artifacts remain
ignored. From sibling VPM:

```sh
../rapm/.venv/bin/python -m vpm.harness.cli --config vpm/configs/research_harness_v56_unsplit_feature_prior.json unsplit-prior --stage all
../rapm/.venv/bin/python vpm/scripts/audit_unsplit_prior.py vpm/results/harness/vpm_unsplit_feature_prior_v1
../rapm/.venv/bin/python vpm/scripts/render_prior_comparisons.py vpm/results/harness/vpm_unsplit_feature_prior_v1
```

The [combined reader](http://127.0.0.1:8767/prior-comparisons.html) displays
total/O/D ratings and O/D priors for all three models,582 recent players or
1556 full-universe players, search/sort/CSV, all seven comparisons and native
diagnostics. Browser checks verify player search,582/1556, defensive sorting,
Jokic values and582-row sorted CSV. Original readers/numerical outputs remain
preserved. [Role-aware receipt](VPM_ROLE_AWARE_PRIOR.md) covers the other test.
