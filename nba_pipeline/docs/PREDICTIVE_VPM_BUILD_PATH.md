# Predictive VPM: one build path

September 20 planning update: [One VPM product, two model versions](VPM_CONSOLIDATION_PLAN.md)
is the current consolidation plan. Extend the existing sibling VPM repository,
preserve v18 as the exact Parent control, and build a separate Components
challenger using shared dated evidence, lower-league priors, uncertainty and
evaluation. The K=500 fallback trials below remain historical experiments, not
the proposed baseline. This plan has not changed model artifacts or the HTML.

Owner priority, September 13, 2026: connect our predictive skills to useful
offensive and defensive priors, then improve component allocation and compete
with predictive EPM. This occupies the existing VPM research slot. The plan
below is an implementation recommendation, not a newly selected model.

## First deliverable

One reproducible estimate-date build that emits our own skill features, an
offensive prior, a defensive prior, and a prior-centered parent RAPM, with a
manifest and a same-sample benchmark receipt. Establish that working baseline
before changing component penalties or adding another feature family.

September 14 completion: the neutral exporter, own-skills harness adapter,
O/D parent priors and current 582-player build now run end to end. The first
fixed development replay does not establish an EPM win or a defensive-feature
increment. See the sibling [integration receipt](../../../vpm/vpm/docs/OWN_SKILL_PARENT_INTEGRATION.md)
for commands, files, fitted coefficients and evidence. The parent preserves
canonical shooter-expected FT scoring; literal actual-points and pregame tests
remain outstanding. Do not create another solver or public reader.

## What already exists

| Asset | Evidence / current state | Use now |
| --- | --- | --- |
| Intern-era EV/component VPM and subsequent Net lane | Config-driven harness, priors, target adapters, boards, possession evaluation | Preserve as named controls and reuse infrastructure |
| Direct O/D parent with EPM skills | Corrected v14 parent uses makes plus attempts; 4000/8000 solve, mapper 0.1, prior scale 1.5 | Strong simplicity control; EPM-skill dependence explicit |
| Own estimated skills | 17-skill PBP panel; v1/v3 and next-game v4 artifacts exist | Audit version, timing, units and neutral-state export before wiring |
| Own estimated deflections | Separate next-game pipeline and measured-observation contract | First optional disruption feature after basic integration |
| Broader parent prior bank | Next-season O/D research with tracking, fouls and assignment features | Candidate inventory; not an admitted prior or ready daily snapshot |
| Shot-quality work | Frozen EV target infrastructure plus separate richer challengers | Retain the established EV control; do not restart SQ research |
| SQ closest-defender dTS prior | Failed replication gate | Park; no shortcut to an admitted defensive prior |

The corrected parent beat complete EPM on the documented common 2023-2026
historical possession sample by 391.71e-6 MSE, interval [247.64, 536.17]e-6.
Corrected decomposition versus that parent was unresolved: +13.95e-6,
[-116.41, 140.69]e-6. These are historical player-quality tests using realized
lineups, not proof of better pregame forecasts or an independent-skill model.
See the sibling [counting-rate correction](../../../vpm/vpm/docs/EPM12_COUNTING_RATE_CORRECTION.md)
and [parent baseline](../../../vpm/vpm/docs/EPM12_PARENT_BASELINE.md).

## The integration gap

`../vpm/vpm/harness/engine.py::build_prior_feature_panel` currently dispatches
`completed_season_box`, `epm_predictive_snapshot`, and `epm_box_tracking`.
September 14: `own_estimated_skills` is now the fourth source, wired through
the immutable v15 integration config. It reads exact-date checksummed neutral
snapshots and explicitly labels fixed-parameter historical replay development-only.

The own-skill files are not interchangeable. v3 optimized a forward aggregate
and its objective was subsequently rejected by the owner. v4 next-game JSONs
exist and include opponent/team/rest context, but their presence does not prove
a selected, causal historical snapshot export. Some optimization receipts mark
non-convergence. Check the complete run/selection receipt rather than choosing
the highest version number. The skills documentation also records a
contemporaneous-season centering caveat that must be resolved before claiming
an as-of VPM replay.

## Feature contract

Use one row per `(nba_id, estimate_date)` in the wide model table, with a
companion per-feature provenance/coverage table. Preserve skill model version,
source hash, latest observation date, parameter-fit cutoff, units, exposure,
missingness, and available uncertainty. Record historical source availability
limitations; re-filtering old games is not automatically an as-of-trained model.

| Initial family | Own source fields | Model representation |
| --- | --- | --- |
| Shooting | rim/mid/three/FT percentages and attempt rates | Expected makes/100 = attempts/100 times percentage; retain attempts/100 |
| Passing and security | `ast_rate`, `tov_rate`, `usg_pct` | Explicit units and denominators; avoid stacking redundant scoring summaries |
| Offensive rebounding | `oreb_rate` | Offensive-possession rate |
| Defensive shot-value backbone | Owner-prioritized rim points saved, DFGA/100, foul rates, plus blocks | Recover exact existing definitions/source fields; stabilize suppression by defended attempts and activity by defensive possessions; compare rim saved versus blocks and their joint increment |
| Disruption and completion | `stl_rate`, `dreb_rate` | Defensive-possession rates; keep turnover creation and second-chance completion distinct from dTS |
| Optional next candidate | Estimated deflections | Separate admitted measured-data source; test conditional increment |

Verify whether stored rates are fractions per possession before multiplying by
100. Percentages remain fractions for the makes identity. Export a neutral
player-skill state separately from next-game context adjustments; do not feed
opponent/rest/availability effects into a portable player prior and add them
again in the game forecast. Any new neutral-state construction needs its own
validation; deleting a context column does not establish neutral talent.

Fail or explicitly flag missing required inputs. Preserve measured zero versus
missing observation. Do not use filled tracking observations as independent
measurements. Keep complete EPM and eWINS out of feature inputs.

## Build sequence and stop rules

1. **Connect the backbone.** Audit/export the dated skills and add the adapter.
   Receipt: unique keys, correct units, strictly earlier observations, training
   cutoffs, explicit missingness, and correct makes/attempt identities for one
   estimate date plus a small historical replay.
2. **Emit O/D priors and solve the parent.** Reuse matured future-parent labels
   and prior-centered solver machinery. Fit coefficients for our feature source;
   do not reuse EPM-feature coefficients by renaming columns. Use the corrected
   EPM parent settings as a fixed initial control, not proof they are optimal
   for our skills. Receipt: paired future actual-possession results against
   plain RAPM, corrected EPM-skill parent, and complete EPM.
3. **Test component value.** Feed the same audited source into the existing
   component recipes and compare the summed prediction against our parent.
   Keep the established SQ/Net targets fixed during this comparison. Retain
   exact component accounting. Do not distribute a parent score across children
   by arbitrary weights; the earlier defensive projection already failed.
4. **Compare defensive feature contributions.** The initial inventory must
   include rim points saved, DFGA/100, and fouls alongside blocks, steals and
   rebounds; the owner explicitly identified these as existing useful priors.
   Recover the canonical rim-points-saved source and formula before creating a
   replacement. Do not confuse it with the failed SQ closest-defender geometry
   experiment. The existing v5 config already has DFGA/foul features for dFT.
   Compare blocks-only, rim-saved-only and both on shared dated rows, then
   conditional additions for shot exposure and foul discipline. Deflections
   chiefly target disruption. Keep inclusion in the feature bank separate from
   final coefficient/recipe selection, which requires the summed forecast gate.
5. **Freeze the game comparison.** Shared pregame minutes/availability, games,
   context, and development-only calibration; primary game-margin error and
   paired game uncertainty. Freeze before opening 2026-27 outcomes.

Keep feature smoothing, mapper regularization, prior scale, and lineup-solve
penalties separately named. They act at different stages. The existing
"750/3000" config is actually six shot/FT O/D penalties at 750/750, TOV/SC at
3000/3000, and mid/three residuals at 12000/48000, with TD550 and prior scale
x2. Preserve that exact control; do not interpret it as one O/D alpha pair.

## Evidence boundaries

Own-skill hyperparameters used 2020-2025 development. Feeding those fixed
parameters into a 2020-2022 VPM replay would not restore a pristine historical
test. Use nested chronological skill fitting for a date-correct replay, or
label the fixed-parameter replay development-only. Audit source availability,
normalization and fully matured mapper labels as well as observation dates.
All previously inspected 2023-2026 evidence remains spent; 2027 remains the
untouched confirmation. Historical improvements guide work without proving
the whole new pipeline beats EPM.

The observed jumbo-defense issue is a diagnostic slice, not authorization to
rescale every big or add a position correction. Failed defensive rescaling,
parent projection and block-overlap experiments stay parked. A dTS prior gap
is plausible, but it has not been established as the sole cause of model error.

## Work-in-progress boundary

The completed-game attribution prototype is parked while this integration is
the active research task. Beyond the owner-requested fixed SQ-parent bridge below, new SQ variants, public
boards, synergy models and alpha searches wait. Existing ship and maintenance obligations retain their
portfolio slots. No live model or publication changes follow from this plan.

Related receipts: [estimated skills](NBA_ESTIMATED_SKILLS.md),
[deflections](NBA_DEFLECTIONS_NEXT_GAME.md),
[failed SQ defender gate](SQ_DEFENDER_DTS_PRIOR.md),
[VPM reproduction](VPM_REPRODUCTION.md), and the sibling
[broader parent bank](../../../vpm/vpm/docs/PARENT_RAPM_PRIORS_2018_2023.md).

## Local HTML research report (2026-09-14)

Read [Building our predictive VPM](../results/estimated_skills/own-skill-vpm.html)
for the own-skill parent explanation, all 582 current player estimates, both
fixed arms, exact interactive prior calculations, five paired development
comparisons, feature definitions, limitations, and the next component gate.
It is a portable single-file snapshot; generated data and HTML remain ignored.
The report clearly separates this O/D parent from the 750/3000 component family.

Authoring source: `results/estimated_skills/own_vpm_report/src/content/report/`.
Reviewed data can be prepared with `scripts/build_own_vpm_report_snapshot.py`;
this reads the existing frozen receipts and does not rerun models. Preserve
the app ID when replacing reviewed data in the existing report. Rebuild and
export using the report project's `AGENTS.md`; serve its `dist` directory for
a local preview. The completed HTML lives one level above the app project.

## Fixed SQ-parent bridge (2026-09-14)

The owner-requested first component increment is complete in sibling immutable
v18: keep own O/D priors ×1.5, parent 4000/8000 and TD550; split out midrange
and three-point residuals with the other VPM’s fitted 12000/48000 penalties.
The bridge improves the matching own-defense parent by +158.84e-6
[+44.52,+270.50] on the same spent 2024–25 expected-FT replay. Exact joint
solves enforce equal-penalty reconstruction; archived SQ targets match the
original manifest hashes. No alpha search, full component allocation or model
promotion occurred. See the sibling [SQ-parent receipt](../../../vpm/vpm/docs/OWN_SKILL_SQ_PARENT.md).

The existing HTML now includes the SQ comparison and a separate 582-player
EV-parent + mid-residual + three-residual board. Refresh its data with
`scripts/update_own_vpm_report_sq.py` after preparing the original snapshot;
it preserves report identity and reads saved outputs only.

The fixed-prediction actual-points sensitivity also passes: +164.60e-6
[+48.21,+277.64] versus the same parent, +373.56e-6 [+57.09,+693.39] versus
complete EPM, same 93,034 rows and 1,054 games. Raw actual-FT targets retain
exact canonical possession identity. This closes the scoring-convention check
for this development replay, not the untouched or full-game forecast gate.

## Low-exposure prior diagnostic (2026-09-17)

The Sep-14 board’s +1.24 Antetokounmpo row is Alex (`1630828`), not
Thanasis (`203648`, −1.81). Exact-ID names in `autocomplete_map.csv` resolve
the report’s ambiguous raw-PBP surname labels. Alex has six games, 20.6333
PBP minutes, and 45 offensive plus 45 defensive possessions. His extended
own prior is +1.2372 offense / +0.0133 defense; SQ finishes +1.2154 / +0.0252.
The matched SQ no-prior estimate is −0.0165 net. This is prior domination,
not a positive lineup-sample finding.

There is no minutes/rotation-based prior mean or exposure-dependent VPM
penalty in v18. Exposure informs upstream skill updates, mapper fit weights
and centering, and lineup evidence, but exported skill variances do not enter
the impact-prior mapper or player-specific precision. The low-exposure
current population is broader than the mapper’s >=500 future-possession
training-label population.

Alex’s 15 FTA in 45 offensive possessions contribute to an estimated 9.60
FTA/100 and 7.11 FTM/100; his 7 three-point attempts yield estimated 10.85
3FGA/100 and 3.57 3FGM/100. These smoothed rates feed the positive offense
prior. Also, his missing rim percentage makes rim makes missing; independent
mean imputation fills 3.7067 rim makes/100 against 3.6601 estimated attempts.
This breaks makes<=attempts after imputation. Fixing percentage-before-product
imputation and testing uncertainty-aware new-player priors are bounded next
candidates, not permission to hand-adjust Alex or add an unvalidated minutes
penalty. No model or frozen output changed in this diagnostic.

## Implemented low-minute challenger (2026-09-17)

Owner authorized prior-mean shrinkage to zero. Sibling v20
`research_harness_v20_own_low_exposure_coherent.json` / `own-low-exposure`
uses strictly pre-cut observed own-skill PBP minutes / (minutes + 500), with no
recentring afterward. It also refits coherent shooting imputation (probability
then makes product), flooring invalid negative attempt estimates and recording
counts. v19 stopped on that validity check before any evaluation.

Alex moves +1.24 to +0.05, but the fixed combined challenger is inconclusive
and slightly worse on 103,378 returning-player actual-point rows:
−14.77e-6 [−71.55,+44.56]; low-minute lineups −30.20 [−227.73,+172.15].
Keep the original control; no fitted optimum or predictive promotion. The
HTML includes both versions, full names and minute/prior weights. Reproduce
its data with `scripts/update_own_vpm_report_low_exposure.py` after the SQ
updater. Canonical [low-exposure receipt](../../../vpm/vpm/docs/OWN_LOW_EXPOSURE_PRIOR.md).

## Learned fallback challenger (2026-09-17)

Owner then requested learning the fallback. Sibling v21 profiles two O/D
prior centers jointly with pre-cut SQ EV parent coefficients, keeping K=500
and coherent inputs fixed. This avoids averaging already-shrunken labels.
The 2024 center is −3.34 O / −1.18 D = −4.52; current is −3.93 / −2.67 =
−6.59. Those are model intercepts at zero experience, not universal replacement
levels. Actual-point replay versus zero improves +42.87e-6 [−125.48,+207.86]
on 103,378 returning-player rows; low-minute subset +176.99 [−418.54,+780.97].
All direct comparisons remain inconclusive. Current Alex −6.24 illustrates
the aggressive extrapolation; preserve original and zero controls, do not
promote. Canonical [learned fallback receipt](../../../vpm/vpm/docs/OWN_LEARNED_FALLBACK.md);
HTML data updater `scripts/update_own_vpm_report_learned_fallback.py`.
