# Parent VPM: fixed shooting fouls and contested-DREB replacement

Owner requested October19,2026: rerun corrected shooting fouls, try contested
DREB instead of total DREB, compare with previous repaired VPM and show ratings.
This is **Parent VPM**, not the ten-component product. No public deployment.

## Frozen comparison before new fits

Keep the six-completed-season actual-point SPM calibration, frozen baseline
offensive mapper slopes, own-prior scale1, mapper ridge.1, incumbent entrant
blend, repaired chronological SQ, residual12000/48000, and actual FT unchanged.
Use the corrected independently recounted shooting-foul source with327958
events; never overwrite the invalid earlier source or its fitted ratings.

Fit two defensive recipes:

- **Fixed fouls + total DREB:** blocks, steals, total DREB, rim savings,
  rim contests, shooting PF and non-offensive PF.
- **Fixed fouls + contested DREB:** identical, with contested DREB replacing
  total DREB, including its missingness flag. Total DREB must be absent from
  the defensive model design, not merely renamed or supplemented.

Fix parent penalties4000/8000. Emit both at the earlier expanded candidate's
TD275; also emit both atTD550 as a registered sensitivity. No new model family,
feature search, penalty tuning, blend selection, or post-stress retuning.
The previous repaired parent remains the main reference; include the existing
core and the original six-active-field expanded parent as unchanged controls.
The previously used prior scales differ: previous own mapper x1.5 versus new
long-SPM x1. These differences remain explicit rather than implying a pure
rebound-only change versus previous. The within-repaired comparison isolates
the rebound replacement at matched settings.

Run the unchanged seven game/team tests for2020–22 development and2023–26
spent historical stress; paired game/season intervals versus previous/core/EPM
and direct matched total-versus-contested comparisons. Add neutral future
actual-point diagnostics. All systems share actual minutes/pace and fills;
no availability or untouched superiority claim.2027 remains protected.

Refit current priors on2021–26 and show the September1,2026 ratings on the same
1556-player universe,585 recent appearances, with O/D/total and changes versus
the exact-cut previous repaired parent. This is not an October roster check.

## Status

Completed; the results below supersede the pre-repair feature-validity claim, while preserving all original files.

## Reproduction

From the sibling VPM repository:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ../rapm/.venv/bin/python -m vpm.harness.cli \
  --config vpm/configs/research_harness_v52_fixed_fouls_contested.json \
  defense-repair --stage all
../rapm/.venv/bin/python vpm/scripts/audit_defense_repair.py \
  vpm/results/harness/vpm_fixed_fouls_contested_dreb_v1
../rapm/.venv/bin/python vpm/scripts/render_defense_repair.py \
  vpm/results/harness/vpm_fixed_fouls_contested_dreb_v1
```

The fixed module consumes stabilized dated player summaries, never raw event or
matchup rows. The independent auditor verifies all locked bytes,24 conditional
mapper equations, chronology, exclusion of both total-DREB columns from the
contested design, unchanged core forecasts, and player-to-game/team accounting.


## Results

Both new recipes have seven active numeric defensive features, plus seven missingness indicators. Corrected shooting PF enters every calibration. No tuning or selection was performed.

2023–26 spent stress:4,920 games and120 team-seasons. Equal-season RMSE:

|Model|Game margin|Team net|
|---|---:|---:|
|Previous repaired VPM|13.965886|3.289191|
|Fixed fouls + total DREB|13.969975|3.214106|
|Fixed fouls + contested DREB|13.960006|3.207770|
|Fixed total, TD550 check|13.975816|3.237095|
|Fixed contested, TD550 check|13.969377|3.234685|
|Existing core parent|13.960415|3.235205|
|EPM|13.941515|3.120003|

Contested DREB gives lower error point estimates on all seven tests versus previous repaired VPM; all seven paired MSE-gain intervals cross zero. Versus fixed total DREB atTD275, game and team-net estimates improve, while low-continuity errors worsen; all seven paired MSE-gain intervals cross zero. Its game correlation-squared gain excludes zero, but that is distinct from direct forecast error. Existing core has lower low-continuity errors; contestedTD275 versus core low-continuity team-win loss excludes zero. EPM has lower team-net error than both new recipes, with paired intervals excluding zero.

|TD275 paired comparison|Test|MSE gain [95% interval]|
|---|---|---:|
|fixed_total_td275 vs original_expanded|game_margin|+0.014949 [-0.087363, +0.112745]|
|fixed_total_td275 vs original_expanded|team_net_rating|-0.010273 [-0.044916, +0.030587]|
|fixed_contested_td275 vs fixed_total_td275|game_margin|+0.278418 [-0.048411, +0.594284]|
|fixed_contested_td275 vs fixed_total_td275|team_net_rating|+0.040691 [-0.054140, +0.118437]|
|fixed_contested_td275 vs VPM|game_margin|+0.164212 [-0.959732, +1.256018]|
|fixed_contested_td275 vs VPM|team_net_rating|+0.528993 [-0.187070, +1.177775]|

The shooting-foul repair alone changes game RMSE13.970510→13.969975 at fixed totalDREB/TD275; that change is unresolved. Swapping total for contested then reaches13.960006. The registered TD550 check also favors contested over total (13.975816→13.969377), unresolved. Development2020–22 game RMSE is13.724173 previous,13.724588 fixedtotal275 and13.710380 fixedcontested275; these are reported without a new selection.

### Neutral future actual points

Common-native423,644 future RS possessions in4,273 games, no future score/clock adjustments. Equal-season MSE: previous1.415015163; fixedtotal2751.415005811; fixedcontested2751.414948025. Positive paired gain favors the first model; game-resampled within season,4,000 replicates:

|Comparison|MSE gain ×10⁻⁶ [95% interval]|
|---|---:|
|fixed_contested_td275 vs previous|+67.14 [-39.02, +170.28]|
|fixed_contested_td550 vs previous|-22.58 [-115.48, +70.40]|
|fixed_contested_td275 vs fixed_total_td275|+57.79 [+25.04, +88.81]|
|fixed_contested_td550 vs fixed_total_td550|+50.35 [+22.90, +78.60]|
|fixed_contested_td275 vs core_control|+121.20 [+45.23, +195.41]|

The contested-versus-total possession increment excludes zero at both decay settings. It is small and does not establish a game-level win or superiority to previous repaired VPM. These are unadjusted research intervals across multiple reported comparisons.

### Current ratings

September1,2026;585 recent appearances/1,556 full universe. The contested recipe does not remove the large big-player shifts from the prior-mapping change. In particular, Jokic rises another1.037 versus the fixed-total recipe. These are comparative estimates, not validated positional calibration.

|Player|New contested O|New contested D|New contested VPM|Fixed total VPM|Previous repaired VPM|
|---|---:|---:|---:|---:|---:|
|Nikola Jokic|7.900|2.866|10.765|9.728|7.413|
|Victor Wembanyama|2.939|6.578|9.517|9.450|6.257|
|Shai Gilgeous-Alexander|7.110|0.658|7.768|7.761|6.813|
|Giannis Antetokounmpo|4.133|2.712|6.845|6.859|5.832|
|Kawhi Leonard|5.917|0.718|6.635|6.624|6.030|
|Luka Doncic|5.827|0.662|6.489|6.492|5.129|
|Robert Williams III|0.754|4.845|5.599|5.470|1.238|
|Donovan Clingan|2.353|3.068|5.421|5.362|1.849|
|Ty Jerome|5.803|-0.554|5.249|5.065|2.502|
|Paul Reed|0.921|4.169|5.090|4.909|1.953|
|LaMelo Ball|5.372|-0.286|5.087|5.058|3.682|
|Isaiah Hartenstein|0.944|4.065|5.010|4.798|3.100|
|Neemias Queta|1.199|3.678|4.877|4.671|2.480|
|Karl-Anthony Towns|2.688|2.132|4.821|4.648|2.742|
|Joel Embiid|3.583|1.227|4.810|4.280|3.528|

Private local reader: [ratings comparison](http://127.0.0.1:8766/ratings.html). It has exactly three plain-name main options, O/D/total, previous ratings and deltas, search, sorting, experience/universe filters and CSV. The older8765 page still contains the flagged pre-repair experiment.

Generated, gitignored artifacts in sibling `vpm/results/harness/vpm_fixed_fouls_contested_dreb_v1`: `ratings.html`, `ratings_comparison.csv` (585 rows), `ratings_comparison_full_universe.csv` (1,556), four current candidate boards, all historical boards, `scorecard.csv`, `paired_comparisons.csv`, `actual_point_game_errors.csv`, and `independent_audit.json`.

### Validation and disposition

28 focused source/calibration/FT/long-SPM tests pass. Independent audit verifies185 locked sources,24 conditional mapper equations,40 emitted O/D/total boards,154 score rows and105 paired MSE point estimates,8,289 total evaluation games/210 team-seasons, source chronology, correct feature exclusion, weighted centering, SQ closure, and player-minute reconstruction. Maximum game/team reconstruction gap7.11e-15; score gap3.56e-15; mapper residual4.27e-14. Existing core and six benchmark forecasts remain equal to saved controls. Browser verified three model options,585-player list, search and exact Clingan values across all three models; screenshot inspected.

Keep as Parent VPM research. ActualFT is fixed for this comparison; expectedFT was separately tested in the preceding receipt and is not silently included here. No component-model promotion, public deployment, fresh-season claim or2027 outcomes.

## Owner prior inspection view, October19

[Player priors](http://127.0.0.1:8766/priors.html) now shows saved prior O/D/total
beside final O/D/VPM. Default is fixed-fouls/contestedDREB skill-only SPM at
scale1; the menu also includes fixed-totalDREB and previous repaired. A second
selector shows the exact combined prior passed into the solve. Its background
weight is `max(1 - observed NBA possessions / 2000, 0)`; skill-only and combined
are identical for established players. The older model's skill-only display
includes its original scale1.5. These are prior means, not standalone lineup
ratings or an additional model fit.

Current contested skill-only examples: JokicO7.056507/D2.577892/total9.634399;
WembanyamaO1.826360/D6.004094/total7.830455; SGAO6.046403/D1.233709/total7.280112;
ReedO0.967311/D4.632707/total5.600018. Joan Beringer demonstrates the distinction:
skill-only total1.872418, combined−2.173458 at background weight.635.

From sibling VPM, reproduce this presentation with:

```bash
../rapm/.venv/bin/python vpm/scripts/render_defense_priors.py \
  vpm/results/harness/vpm_fixed_fouls_contested_dreb_v1
```

Generated files: `priors.html`, `<model>_priors.csv` (585 recent rows per model),
`priors_full_universe.csv` (three models ×1,556 players), and
`priors_view_manifest.json`. CSVs contain both prior types and the final ratings;
the reader's download button exports the current sorted/filtered view.

The renderer verifies exact player alignment, finite values, all prior/final
sum identities, and the saved blend identity for all three models. Browser
checks verified585/1556 universes, default values, search, model/basis changes,
Beringer's actual blend and the screenshot. Model, calibration, rating and
forecast artifacts remain unchanged. The final-ratings reader now links to
priors; its original HTML and renderer are preserved under gitignored
`reader_before_priors/`, and the curated summary records the UI-only hash change
and new prior-view manifest. No refit, new evaluation or2027 outcomes.

### Predictive scope of these priors

The six-season SPM mapper uses statistics and lineup outcomes from the same
calibration games. Its in-period fit does not establish forward predictiveness;
transfer of coefficients from pooled observed counts to dated current skill
forecasts is an empirical assumption. The final Parent VPM is a hybrid of this
prior mean and pre-cut lineup evidence, with SQ residual corrections; the prior
inspection page shows the means before that lineup update. Previous repaired
priors instead mapped pre-cut skill snapshots onto subsequently observed,
fully matured forward RAPM labels. Those methods should not be equated.

The latest final metric has chronological historical forward tests, but game
improvement versus previous is unresolved and that test does not establish
standalone SPM-prior superiority or validate the wider prior scale. The
[earlier old/new blend](VPM_LONG_SPM_BLENDS.md) combined previous and long-SPM
core final ratings; its50/50 variant is not used in these fixed-foul priors or
ratings. Its development selection retained previous, and the later50/50 gain
is spent research evidence. No new blend or predictive calibration is run here.
