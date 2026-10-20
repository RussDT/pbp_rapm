# Previous repaired VPM: expanded forward priors without shrunken-label calibration

Owner authorized October19,2026 after inspecting the six-year SPM priors.
Preserve previous repaired Parent VPM and corrected six-year SPM controls.
This experiment keeps dated skills predicting subsequent outcomes, rather than
fitting same-window observed statistics. No public deployment or2027 outcomes.

## Registered before new fits

Use the previous repaired365-day forward cohorts, beginning2015-10-01. A cohort
can train an emission only after its full365-day label window has matured.
Skill observations/parameter fits precede the cohort start; actual FG+FT lineup
outcomes follow it. Match the old timing, including the COVID date and formal
label maturity; current September1,2026 excludes the not-yet-matured2025-10-01
cohort. Overlapping COVID windows remain a disclosed limitation.

Compare two changes independently:

1. **Feature-only old mapper:** old2000/4000 shrunken future RAPM labels,
   mapper.1, own scale1.5, expanded-contested defensive recipe below.
2. **Forward covariance calibration:** fit shared feature coefficients through
   the complete likelihood of future lineups, jointly profiling cohort-specific
   O/D player residuals and intercepts. Residual penalties2000/4000. This avoids
   using zero-shrunk RAPM point estimates as regression responses, without
   assuming weak/noisy OLS player labels are exact. Output scale1; no manually
   chosen target spread or multiplier.

Covariance-calibration recipes: old seven defensive fields; expanded with
both rebound measures; expanded with contested replacing totalDREB; and that
replacement recipe plus dated trailing365-day RS minutes per appearance in
both O/D priors. The MPG lane tests coach/role information explicitly; it is
not presumed to be a skill. Counts/exposure and games remain metadata.

Expanded defense includes blocks, steals, rim savings, rim contests, DFGA,
corrected shootingPF, non-offensivePF, deflections, offensive fouls drawn,
contested defensive rebounds and contested shots; the total-DREB variant
adds totalDREB. Drop old totalPF in favor of the two foul fields, and omit the
charges subset of offensive fouls drawn. All features have missing flags.
Offense retains the old12 numeric inputs, with MPG only in its named lane.

Fit training-only coherent imputation/scaling on fully matured cohorts,
using >=500 future side possessions for the feature direction; all players
remain in the future lineup likelihood. Preserve absent fields as missing,
not zero; wholly unmeasured fields in an early prefix are unavailable.
Center the feature directions within cohort on side exposure. Sum cohort
profile information/RHS; the mean mapper is shared, player residuals are not.
Mapper ridge grid.1/1/10; alpha units are the summed native-point profile
objective, not the old normalized player-label objective. A feature-only
old-label control retains its original objective and.1 penalty.

Freeze lineup O4000/D8000/TD550, actualFT, repaired SQ, residual12000/48000,
entrant models/blend, census, fills, benchmark and seven tests. No new context,
scale, decay or lineup-penalty search. Select literal minimum equal-season
2020–22 gameMSE across the13 declared candidates, lexical ties. Freeze that
choice through2023–26 spent stress. Report all fixed candidates and paired
comparisons; no inference that wider priors are necessarily better forecasts.

Show selected current priors, ratings and prior-spread diagnostics beside
previous repaired and corrected six-year SPM. Current cut remains Sep1,2026,
not a certified October roster. Independently verify forward chronology,
profile equations, original control reconstruction, missingness, seven tests,
paired error points and current prior/lineup sums before displaying results.

## Status

Completed; results and interpretation follow below.


## Completed result

The requested wider forward priors are implemented. Their amplitude is learned from future lineup likelihood, with no extra output multiplier. The best forward-profile candidate on2020–22 is **expanded contested defense + MPG, mapper ridge10**. It has13 numeric O inputs and12 D inputs (50 columns with missing flags). On2023–26 it improves all seven paired error estimates versus previous repaired; all seven unadjusted intervals exclude zero. This is historical discovery on spent seasons, not untouched confirmation or a public release.

The overall13-way development winner is **the feature-only old-label extension**, RMSE13.721851, versus best forward13.740987 and previous13.724173. Do not call the forward candidate the overall development winner. The separate review names the minimum among the12 predeclared profile candidates using only original development losses; it leaves selection.json and the overall winner unchanged. The review and all candidate stress results are disclosed.

|Model|Development game RMSE|Stress game RMSE|Stress team-net RMSE|
|---|---:|---:|---:|
|Previous repaired|13.724173|13.965886|3.289191|
|More features, old label calibration|13.721851|13.950464|3.262441|
|Expanded forward calibration + MPG|13.740987|13.895289|3.091360|
|Same forward calibration without MPG|13.757133|13.934831|3.184765|
|Forward calibration, old features|13.759852|13.956330|3.212099|
|Six-year SPM, fixed fouls/contested DREB|13.710380|13.960006|3.207770|
|EPM|13.784609|13.941515|3.120003|

The matched MPG addition improves stress game RMSE13.934831→13.895289 and team-net3.184765→3.091360. Paired gameMSE gain+1.100448[+.719329,+1.504605] and team-net+.586220[+.333707,+.792198] favor MPG. This does not isolate all possible role/availability mechanisms or establish a causal minutes effect. Current conditional MPG slopes are+0.043504 O and+0.035140 D points/100 per extra MPG; no manual role bonus or multiplication by minutes is applied.

### Seven tests and paired uncertainty

Same4,920 stress games/120 team-seasons; low-continuity492 games/13 team-seasons. Positive MSE gain favors the forward+MPG candidate. Game resampling is paired within season; team uncertainty resamples whole seasons;4,000 replicates. Intervals are unadjusted across multiple candidates/tests and overlapping tests are not independent replications.

|Test|Forward+MPG RMSE|Gain versus previous [95% interval]|
|---|---:|---:|
|game_margin|13.895289|+1.966933 [+0.983951, +2.945840]|
|game_margin_low_continuity|14.467058|+6.019858 [+2.695128, +9.447358]|
|team_win_pct|0.084049|+0.000774 [+0.000170, +0.001379]|
|team_win_pct_low_continuity|0.108639|+0.002025 [+0.000544, +0.003456]|
|team_net_rating|3.091360|+1.262272 [+0.255729, +1.953802]|
|game_margin_veterans|13.975881|+2.215513 [+1.198861, +3.192426]|
|team_win_pct_veterans|0.087156|+0.000810 [+0.000130, +0.001490]|

Versus EPM, gameMSE gain+1.286792[−.065454,+2.655023] and team-net+.177913[−.063680,+.419506] remain unresolved. Do not report an established EPM win. All forecasts condition on common actual future minutes and pace; this is not a full pregame availability forecast.

### Prior-only and final future actual possessions

The independent native-player diagnostic uses423,644 actual future RS possessions in4,273 games. Prior-only forecasts use the combined prior and an intercept estimated solely from past TD550 actual scoring minus past prior lineup means. No future score/clock or future baseline fit is used.

- Prior-only actual-point MSE: previous1.415998229, feature-only1.415994700, forward+MPG1.415538267. Gain versus previous459.96e−6[304.35,616.33]e−6; versus feature-only456.43e−6[312.40,602.96]e−6. The wider learned priors improve this historical forward diagnostic rather than only their training fit.
- Final VPM actual-point MSE: previous1.415015163, forward+MPG1.414933460. Gain81.70e−6[−6.14,173.69]e−6 remains unresolved; final versus six-year SPM is also unresolved. Keep this distinct from the favorable game-level evidence.

### Current prior spread and ratings

Unweighted standard deviations across the same585 recent players:

|Model|Skill prior O SD|Skill prior D SD|Final total SD|
|---|---:|---:|---:|
|forward|1.494675|1.310118|2.459476|
|features_only|0.955633|0.556550|2.086397|
|previous|0.955633|0.498760|2.050424|
|sixyear|1.603604|1.300266|2.453170|

The forward defensive prior is2.63× as spread out as previous, with output scale1 rather than the old1.5. No target spread, positional recentering, manual player adjustment, extra smoothing or post-stress scaling was used. Wider remains a learned behavior, not a success criterion.

|Player|Prior O|Prior D|Final O|Final D|Final VPM|Previous VPM|
|---|---:|---:|---:|---:|---:|---:|
|Nikola Jokic|4.863|2.621|6.857|3.017|9.875|7.413|
|Victor Wembanyama|2.544|5.228|2.703|6.380|9.083|6.257|
|Giannis Antetokounmpo|4.430|1.284|4.839|2.198|7.036|5.832|
|Shai Gilgeous-Alexander|4.901|0.929|6.529|0.291|6.820|6.813|
|Kawhi Leonard|3.762|0.620|5.669|0.893|6.563|6.030|
|Luka Doncic|5.592|1.538|5.605|0.776|6.382|5.129|
|Donovan Mitchell|2.669|0.256|4.183|0.647|4.829|4.696|
|Ausar Thompson|-1.518|4.540|0.214|4.573|4.787|3.137|
|Stephen Curry|3.860|-0.478|5.570|-0.872|4.698|4.579|
|Chet Holmgren|-0.279|3.295|0.918|3.644|4.562|3.688|
|Joel Embiid|3.751|0.638|2.968|1.536|4.504|3.528|
|Bam Adebayo|2.236|1.461|1.382|3.114|4.496|3.149|
|Alex Caruso|-0.420|3.573|0.022|4.426|4.448|3.577|
|Anthony Davis|1.035|3.659|0.581|3.815|4.396|2.239|
|LaMelo Ball|4.302|-0.045|5.108|-0.823|4.285|3.682|

Current estimate date is September1,2026. [Interactive priors and ratings](http://127.0.0.1:8767/ratings.html) has four plain-name models, both skill-only and combined priors, final O/D/VPM, previous deltas, search, sorting,585/1556 universes and CSV downloads. Existing8766 readers remain unchanged.

### Validation and source limits

20 focused tests pass, including a synthetic case proving that profiling the future lineup likelihood matches an independent joint block solve and avoids attenuation inherited from shrunken point labels. This verifies the algebra, not a guarantee of empirical improvement.

RAPM-owned adapter reconstructs167,846 additional defensive cells across12 dated prefixes (max3.55e−15); older eight-prefix snapshots reproduce. Corrected shooting source remains327,958 independently recounted events. Independent model audit checks111 locked sources,10 raw future cohorts,96 profile models,112 emitted boards,280 score rows and105 paired MSE points across original/review comparisons. It independently rebuilds the raw future lineup normals, profile responses/information, mapper equations,23,978 MPG cells, future-label maturity, final sums, player-minute game/team forecasts, original control parity and exact development winner. Raw future geometry and MPG gaps are zero; mapper residual max4.55e−13. All old six-system benchmark forecasts and previous repaired ratings are preserved.

The previous forward label solver fixes its intercept at the target mean; new covariance calibration profiles a cohort intercept jointly. An initial parity preflight incorrectly compared those two coefficient conventions and was corrected to reproduce old labels with their original solver. A second preflight encountered absent early deflections; features with fewer than25 observed eligible rows now remain unavailable in that prefix, preserving missingness. Both failed code/log/source-lock snapshots are retained under preflight_*; no forecast scores were inspected before those corrections.

The ten365-day training windows repeat318 games across COVID-overlapping cohorts. Player residuals are cohort-specific; overlap is a disclosed dependence limitation. Labels retain formal365-day maturity: the Sep2026 emission uses2015–24 feature cohorts, latest label endOct1,2025, with current dated skill inputs. Source/publication vintages and the retrospective method search remain uncertified. No2027 outcomes, public deployment, EPM imitation, ten-component model promotion or automatic release choice.

## Reproduction

From RAPM root, export dated summaries:

```bash
.venv/bin/python nba_pipeline/scripts/export_vpm_forward_features.py
```

From sibling VPM:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ../rapm/.venv/bin/python -m vpm.harness.cli \
  --config vpm/configs/research_harness_v53_repaired_forward_expanded.json \
  repaired-forward --stage all
PYTHONPATH=. ../rapm/.venv/bin/python vpm/scripts/prepare_repaired_forward_review.py \
  vpm/results/harness/vpm_repaired_forward_expanded_v1
../rapm/.venv/bin/python vpm/scripts/audit_repaired_forward.py \
  vpm/results/harness/vpm_repaired_forward_expanded_v1
../rapm/.venv/bin/python vpm/scripts/render_repaired_forward.py \
  vpm/results/harness/vpm_repaired_forward_expanded_v1
```

Generated packets/models/boards/CSV/HTML/logs stay gitignored. Curated small receipt is sibling `vpm/evidence/repaired_forward_expanded_summary.json`.


## Predictive input inspection (October 19)

[Predictive skills](http://127.0.0.1:8767/skills.html) exposes the exact current
forward+MPG input packet: box-score rates, zone shooting probabilities and
makes/attempts, derived advanced summaries, defensive tracking and all24
unique numeric mapper fields (MPG is shared by O/D). Search/sort, group tabs,
player details, full-universe switching and CSV exports are available. The
ratings reader links directly to this view. No calibration, rating or forecast
was changed.

Distinguish three methods: annual past-only neutral skill states; the extra
tracking fields' fixed TD550/K1000 stabilized estimates; and observed trailing
365-day RS MPG. The tracking fields are not the separate optimized next-game
OU tracking models. Derived coherent points, FG%,2P%,eFG%,TS%,3P attempt share,
FTA/FGA and AST/TOV are inspection summaries, not additional coefficients.
TS% uses0.44; usage events/100 is not conventional USG%. The independent points
state, totalDREB, totalPF and charges are labeled auxiliary/nonmapped. Never
sum OREB/100 offense and DREB/100 defense as though their denominators matched.

Player details show raw values, mapper-side values after coherent imputation,
missingness flags, last observations, past parameter-fit endpoints and source
exposure. A wholly unmeasured side reports neutral rather than representing
its imputation means as used. Missing stays missing; actual zero remains zero.
Variances are not displayed as calibrated confidence intervals. Percentages
are fractions in CSV and percent units in the reader.

The inherited585-player recent-event flag includes Kevin Garnett, JJ Redick
and Tiago Splitter: those source rows have zero recent playing possession
exposure. This view defaults to the582 with positive measured O+D exposure in
the preceding365 days, retaining the inherited flag and all1556 ratings in
the full export. It changes the display filter only. The original recent flag
is not a certified active-player/roster field; upstream event-only rows must
not establish a playing appearance. Existing rating comparison cohorts and
saved results remain unchanged.

Standalone reproduction from the sibling checkout:

```sh
python vpm/scripts/render_predictive_inputs.py \
  vpm/results/harness/vpm_repaired_forward_expanded_v1 --rapm-root ../rapm
```

All1556 O/D own priors reconstruct exactly (maximum absolute error0) from the
shown source packet and original calibration. Coherent makes and rim-saved
identities pass with missingness preserved. Browser checks verify group
switches, player search/details, sorting,582/1556 filters, missing/zero cells
and CSV download. `skills_view_manifest.json` pins seven source artifacts,
renderer/template hashes, missing-cell counts, the three event-only rows and
generated exports. Original ratings HTML/current_review/renderer/evidence are
archived under ignored `reader_before_skills`; curated evidence records only
the reader navigation revision and new view. Existing numeric export, model,
selection and forecast hashes remain identical.

Owner has authorized considering the BAM matchup features, but requested this
input inspection first. That increment remains pending; no matchup input was
silently admitted to this model.
