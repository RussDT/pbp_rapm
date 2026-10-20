# Current Parent VPM: three offensive creation feature tests

Owner authorized October20,2026: separately compare predictive rim/potential
assists, assisted/unassisted makes by zone, and playtype difficulty against
retained current forward Parent, then show predictive results and ratings.
Same VPM research slot; no2027 outcomes or automatic public/component release.

## Frozen before fitting

Control is v53 `forward_expanded_mpg_a10p0`. Three independent challengers:

1. **Assist creation:** add predictive rim assists and potential assists/100.
2. **Assistedness:** retain total zone makes plus an unassisted-makes contrast
   for rim/mid/three. This spans separate assisted/unassisted coefficients,
   with no redundant third total. Predict each split as existing total makes
   times a bounded predicted composition; preserve exact total closure.
3. **Playtype difficulty:** add `2*TSA100*difficulty_pp/100` using forecast
   six-category shot mix and lag-priced category difficulty. Derived approximate
   predicted TSA100 is sum of predicted zone attempts+.44*predicted FTA. No extra raw difficulty
   term alongside its own point-valued transform.

No combined arm, feature/mapper/parent penalty grid, physical metadata or
player override. The added columns enter offense only. Original preprocessing,
valid-row rules, skill states and defensive feature set are retained; appended
imputation/scales are fitted solely on eligible matured training rows. Mapper
O/D coefficients refit jointly through the same profiled future-lineup
likelihood, residual2000/4000, ridge10, outputscale1. New coefficients zero
must exactly recover the old prior span and the original null solution.
Original entrant vectors/weights and final4000/8000/TD550, actualFT, repaired
SQ12000/48000 for both zones, census, home/minutes/fills stay fixed. D3 remains
48000 per owner clarification.

### Dated skill inputs

Construct2014–26 player-game counts on exact canonical own-skill game/player
exposure. Deduplicate raw PBP, use identical three/rim/mid definitions and
reconcile FGA/FGM counts by player-game. Parse made-shot assist tags, never
player2_id or an assumed assist label on misses. Resolve passers with exact
game/side name anchors and same-side on-court IDs; unresolved or unreconciled
rim-assist games stay missing. Assistedness of a made shot does not require
identifying the passer. Preserve explicit zero versus missing coverage.
Potential assists use only measured tracking fields, with exact date/player/
phase, box and minutes/exposure reconciliation; filled estimates are excluded.

Fit annual past-only two-state OU count filters for potential assists and
bounded proportion filters for rim share of total assists and unassisted
share of each zone's made shots. Predict rim assists as the unchanged own-AST
forecast times the bounded rim-assist share, preserving rim<=total assists.
Use existing6-parameter OU bounds, DE seed20260928, maxiter60, popsize15,
no polishing; objective is exposure-weighted next-appearance MSE after the
first observed season and first player observation. Report convergence and
fast/slow algebra, hold latest postgame state, maximum age730 days. Fit at
each2015–25 preseason vintage (December1 for2020), then replay only pre-cut
observations for each relevant historical/current feature cut. All parameter
and observation dates must precede the cut; no later frozen fit backfill.

Forecast six-category shot composition with decayed Dirichlet counts, same
outcome-invariant taxonomy as the existing PBP difficulty study. Select
half-life180/365/730/1500 days and prior50/200/1000 FGA by past-only weighted
next-appearance multiclass Brier loss, same annual vintages. The prior is the
training-prefix category distribution. Source shot mix never uses make/miss
or made-only assist tags. At a cut whose last completed season is yearY,
price categories only from RS yearY−1, shrink100 FGA toward that year's
league price. Difficulty is50*(league points/FGA−predicted mix points/FGA).
This deliberately lags even the latest completed season's prices. No raw
shot rows/provider assignments enter VPM; only dated stabilized summaries.

### Matched evaluation and audits

Emit seven2020–26 preseason cuts and September1,2026 current board. Compare
each challenger with exact current, previous repaired and EPM on all seven
common tests; show all three even if null wins. Select among current/three
challengers only by literal minimum equal-season2020–22 gameMSE, lexical
ties; freeze through spent2023–26. No stress/player-based selection. Also
report native future actual-possession, prior-only and crossed defensive
diagnostics using the identical accepted game/player census.

Independently audit raw counts, coverage, identity, split/category closure,
annual parameter/observation fences, truncated-source invariance, source
hashes, profile algebra, original null/board/forecast parity, seven test
definitions and paired error points before presenting results. Intervals are
unadjusted and overlapping historical tests are not independent replications.
Actual future minutes/pace and uncertified historical source/publication and
architecture vintages remain limitations. Source data and generated outputs
stay ignored; code/docs/tests and small curated evidence are committed.

## Completed results — October20,2026

Retain current forward Parent VPM. All three requested blocks are reported
separately; no combined arm or model promotion. Difficulty wins the declared
four-way early-development comparison but does not improve later predictions.

| Model | Development game RMSE2020–22 | Spent game RMSE2023–26 | Spent team-net RMSE |
|---|---:|---:|---:|
| Current forward Parent |13.740987|13.895289|3.091360|
| Rim/potential assists |13.741917|13.894258|3.086709|
| Assisted/unassisted zone makes |13.759713|13.907115|3.094985|
| Playtype difficulty points |13.715609|13.897795|3.110006|

Positive paired MSE gain favors the addition over current; intervals are
unadjusted95% and tests overlap. Overall game gains: assists+.028644
[−.014824,+.072941], assistedness−.328806[−.554117,−.105929], and
playtype points−.069651[−.211070,+.068916]. Assists also worsens
low-continuity games:−.157309[−.305078,−.014793]. Its team-net gain
+.028736[−.001112,+.058584] remains unresolved. Full seven-test comparisons
against current, previous repaired and EPM are in the reader and curated
sibling evidence. Previous repaired game/team-net13.965886/3.289191;
matched EPM13.941515/3.120003, with identical future minutes/pace.

On423644 native future actual possessions in4273 games, equal-season
weighting, assist final-MSE gain is+.358749e−6[−4.499333,+5.205717]e−6,
while its prior-only gain is+12.941574e−6[+.891684,+24.727901]e−6.
The prior increment therefore does not establish an overall final-model gain.
Assistedness final/prior gains remain unresolved. Playtype points worsens
both final (−23.662412e−6[−37.469857,−9.826265]e−6) and prior-only
(−49.921820e−6[−80.339168,−18.120572]e−6) predictions. Crossed
D-only diagnostics hold incumbent offense/intercept fixed; they are not
separately fitted or selected defensive replacements.

### TSA and point conversion clarification

The fitted input is already point-valued:
`own_role_added100 = 2 * predicted_TSA100 * difficulty_pp / 100`.
Predicted TSA is the sum of the existing three predicted zone FGA rates plus
`.44 * predicted_FTA100`. This is the conventional **approximate TSA**, not
an exact forecast of possession-ending FT trips. It does not distinguish
and-ones, technicals or two-/three-shot FT trips. This denominator limitation
was clarified to the owner after results; no silent denominator substitution
or refit. It is separate from actual-FT lineup targets, which remain unchanged.
Jokic difficulty+1.001073 percentage points becomes+.565745 points/100
at forecast volume, before the learned mapper coefficient. Do not call the
TSA denominator exact or raw difficulty itself the fitted feature.

### Ratings and source coverage

[Four-way ratings, priors, predictive inputs and results](http://127.0.0.1:8767/offensive-creation.html)
default to582 positive-recent-exposure players, with all1556 available.
Current/assist/assistedness/difficulty total VPM: Jokic9.874686/9.902489/
9.424180/9.873225; Wembanyama9.083126/9.080094/8.847461/9.185861.
The original ratings and skills readers are preserved. New inputs cover
574/582 rim assists,582 potential assists,576/564/534 unassisted rim/mid/three
makes and582 difficulty forecasts. Missing values are explicitly imputed in
training/deployment and shown missing in the raw-input view, never zero-filled.

Source package contains387962 canonical player-games and2913812 raw shots
from26 RS/PS files,13627 forecast rows at12 dated cuts and63043 provenance
rows. Independently reconstructed measured counts match exactly. Fifty-five
annual OU fits use the frozen finite DE budget:3 converge,52 stop at budget;
no claim of global optimization. Fast/slow filter agreement2.11e−15.
Eleven shot-mix fits select180-day half-life/prior50 FGA from the fixed grid.
Historical source/publication vintages remain uncertified; latest source
measurements need not have been publicly available at the historical cut.
All2023–26 targets are spent development evidence, with no2027 outcomes read.

### Validation and reproducibility

Independent source audit passes all38 original pins, counts, date fences,
training objectives, full-source fixed-parameter prefix checks, lag prices and
mix closure. Maximum state/objective/price gaps are1.43e−14/4.17e−17/
2.31e−14. A separate dependency_closure.json pins helper-loaded common.py
(matches committed pre-fit code) and the vintage-cut helper (matches its
previously frozen monthly source package). These supplemental hashes preserve
original locks and numerical outputs; the latter helper is a pre-existing
untracked dependency, not newly authored in this experiment.

Independent model audit verifies629 source files,24 mappers/24 boards, eight
reproduced controls,84 score rows and63 paired rows. Maximum profile gap
1.43e−14; original prior span gap0; null coefficients1.23e−15; direct
raw-lineup ratings5.33e−15; player-minute scores7.11e−15. Original forecasts
are reproduced, independent seven-test/paired-point reconstructions pass,
and six focused source/model tests pass. Browser views/search/sort/universe
and582-row downloaded ratings CSV were verified. Curated receipt:
`sibling vpm/vpm/evidence/offensive_creation_summary.json`; generated large
artifacts stay ignored.

From RAPM root (existing monthly own-skill/baseline packages are prerequisites):

```sh
.venv/bin/python nba_pipeline/scripts/export_vpm_offensive_creation.py --stage all --workers 3
.venv/bin/python nba_pipeline/scripts/audit_vpm_offensive_creation_source.py
```

From sibling VPM root:

```sh
../rapm/.venv/bin/python -m vpm.harness.cli --config vpm/configs/research_harness_v58_offensive_creation.json offensive-creation --stage all
../rapm/.venv/bin/python vpm/scripts/audit_offensive_creation.py vpm/results/harness/vpm_offensive_creation_v1
../rapm/.venv/bin/python vpm/scripts/render_offensive_creation.py vpm/results/harness/vpm_offensive_creation_v1
```

No source/model parameters were changed after inspecting outcomes. The reader
now explicitly discloses approximate TSA; the numerical tests are unchanged.

## Owner decision: retain the predictive stats library — October20,2026

Keep these forecasts as predictive box-score/advanced-stat outputs independently
of VPM feature selection. The skill-filter source package, annual parameters,
dated states and provenance remain retained. Failure to add after existing
VPM inputs does not mean the stat must be discarded, nor does this impact
comparison establish superiority of each individual-stat forecast.

The existing [predictive skills page](http://127.0.0.1:8767/skills.html) now
includes a **Creation & shot difficulty** group. Box score also shows rim and
potential assists; Advanced includes approximate TSA and playtype difficulty
in both TS percentage points and points/100. Player details label all15 added
outputs/support fields **Retained · not in VPM**, with no mapper coefficient.
This includes three unassisted shares and assisted/unassisted zone makes,
rim-assist share/count, potential assists, TSA, difficulty and difficulty points.
Assistedness describes makes, not all shot attempts. The exact original
current O/D mapper view, priors and ratings are preserved.

These fields enter the full/recent predictive_inputs CSVs and field dictionary,
plus the downloadable current table. Preserve raw missing cells without mapper
imputation; each zone's assisted and unassisted forecasts sum to its existing
predicted makes. Approximate TSA remains FGA+.44FTA, explicitly labeled. The
retained difficulty points use raw observed forecast support, not training
imputation. Display dates and method/provenance remain attached. This extends
the dated skill library; it does not replace the older standalone next-game
tracking-registry incumbent or claim a newly validated next-game recipe.

Rebuild with sibling `vpm/scripts/render_predictive_inputs.py` on the existing
v53 root. It checks the independent source audit, pre-cut dates, split closure,
unchanged original packet,1556 identities/582 recent exposures and exact
current prior reconstruction (gap0). Browser views and retained-status/date
labels passed; the downloaded582-row creation CSV matches saved values.
Curated receipt `vpm/evidence/predictive_creation_library_summary.json` records
retained fields, units, source/code/output hashes and coverage. No new fits,
lineup inputs, VPM predictions or2027 outcomes.
