# Predictive VPM prior training: results

**Recommendation: retain the owner-selected v61 VPM.** Preserve the new annual-snapshot + season/home +365-day calibration-decay model as a research candidate. The registered experiment has not established an overall forecasting improvement or an EPM win. The selected model pointer remains unchanged.

The registered timing, decay, rolling, annual and native comparisons and September 1, 2026 research boards are complete and independently verified. A separate workflow audit confirms all 32 progression stages and three base stages exited successfully. No registered main comparison is missing. The report records the evidence for retaining current VPM; it does not promote the research candidate.

## What we tested

The proposed change was to train the stats-to-impact prior with each player's **predictive box-score/advanced stats immediately before each historical game**, rather than holding a cohort-start snapshot fixed across the following 365 days. Both designs train against actual lineup scoring with regularized offensive and defensive player residuals. Neither regresses onto precomputed RAPM labels.

We compared annual-snapshot and game-by-game training, each with and without season/home controls, on matched sources, preprocessing, cohorts and outcomes. We then tested no additional calibration decay and half-lives 365/730/1460 days. Only 2020–22 rolling development results selected the recipe. For rolling forecasts, all arms update their player inputs and ratings daily at deployment; “annual snapshot” describes prior-training timing, not stale deployed ratings.

Three separate memories remain distinct: each predictive stat's filter, historical games learning the stats-to-impact mapping, and final lineup evidence updating the prior. The selected 365-day half-life weights **calibration training games**; it changes neither stat-filter memory nor final Parent TD550, and retains all eligible earlier history with diminishing weight.

## Development chose the annual-snapshot design

The complete 455-date development replay contains 3,068 common regular-season games. Selection averages season MSEs equally; the RMSE below is the square root of that criterion.

| Prior training | Context | Equal-season RMSE |
|---|---|---:|
| Annual snapshot | Season/home |13.441051|
| Annual snapshot | None |13.442517|
| Pregame before each game | Season/home |13.452292|
| Pregame before each game | None |13.453081|

The matched timing difference is unresolved: annual+context's MSE gain over pregame+context is+0.302303, conditional 95% game interval[-0.314380,+0.939561]. The registered rule chose the literal minimum; it did not force the proposed pregame design to win.

For the winning timing/context arm,365-day calibration decay gives RMSE13.435545, versus no extra decay 13.441051,730 days 13.437714 and 1460 days 13.439264. Its gain over no extra decay is only 0.005506 RMSE points. The paired MSE gain is+0.147994, game interval[+0.011405,+0.286076]. These are post-selection development contrasts with fits and choices held fixed, not independent confirmation or an optimal-decay estimate. Original no-decay predictions were reused bit-for-bit.

## Later rolling games: a small unresolved gain

The frozen candidate was replayed on all 651 registered dates in 2023–26. On 4,772 common games, pooled RMSE is **13.586407 versus 13.593466** for the matched annual-snapshot/no-context/no-extra-decay control. The paired MSE gain is+0.191881,95% game interval[-0.006710,+0.396485]; the whole-season interval is[+0.058315,+0.325448]. The game comparison remains unresolved.

This contrast combines season/home controls and calibration decay within the annual-snapshot design. It is not a comparison against selected v61 or EPM, and does not isolate game-by-game training timing.

## Annual benchmark: current VPM remains competitive

The established seven-test annual-cut benchmark uses 4,782 common spent games and 120 team-seasons. It measures a different horizon from daily rolling forecasts and never reselects the recipe.

| Annual 2023–26 test: RMSE, lower is better | Candidate | Current v61 | EPM |
|---|---:|---:|---:|
| Game margin, points |13.927453|13.892012|13.934471|
| Low-continuity game margin, points |14.097343|14.415333|14.478896|
| Team win proportion |0.086824|0.085435|0.086303|
| Low-continuity team win proportion |0.089663|0.104282|0.113002|
| Team net rating, points/100 |3.186538|3.114328|3.118215|
| Veteran game margin, points |13.962388|13.976767|14.019490|
| Veteran team win proportion |0.088443|0.088651|0.089425|

Current v61 has better overall game/team point estimates. The candidate's overall game MSE gain versus v61 is-0.985941, game interval[-2.486479,+0.514865], season interval[-3.674267,+1.241431]. Versus EPM it is+0.195523, game[-1.771295,+2.214024], season[-2.424028,+2.815075]. Both headline differences remain unresolved.

Low-continuity game and team-win errors improve conditionally versus both references. These subgroup findings are useful, but do not establish a broad replacement case. New annual boards rate 100% of retained minutes; references cover approximately 89–92%, with the same registered fallback and veteran override. Source, universe, admission and entrant-coverage changes contribute to the comparison. Team targets use the common retained schedule rather than full 82-game projections.

## Native lineup diagnostics do not establish a final-rating gain

Independent audits reconstruct 609,665 development and 947,367 spent actual-point native rows. Native MSE scores and gains average the seasonal possession MSEs equally; units are squared points per native possession. For the candidate versus the matched static control, spent prior-only MSE gains under the registered **common past-mean baseline** are +0.001204072 for own priors and +0.001010498 for combined own/background priors.

The separately declared **past residual-centered secondary baseline** removes each prior's historical average scoring drift. Under this convention, own-prior gain becomes -0.000045162, game interval[-0.000078435,-0.000009984]; combined-prior gain is -0.000015896, interval[-0.000040357,+0.000009125]. The primary prior-only improvement therefore depends strongly on baseline-level calibration; it is not robust evidence of better player separation.

Final native MSE is 1.408876877 for the candidate versus 1.408865947 for the control: gain -0.000010929, game interval[-0.000025473,+0.000003834], season[-0.000022590,+0.000000732]. Final native improvement is unresolved. Both baseline conventions remain reported; neither was selected after seeing its outcome.

## Current research boards and exact decomposition

The September 1, 2026 candidate and matched static-control boards each contain **1,785 historical-universe players**, including 582 with positive validated 2026 playing exposure. This is a dated research board, not a certified current NBA roster or live refresh. Two players with unresolved exposure weights remain omitted; common whole-history quarantine removes 44 games/9,076 rows and retains 13,898 games/2,735,920 rows. Their weights were not guessed.

Independent reconstruction verifies the dated states, reviewed entrant priors, target and pricing digests, TD550 exposure, all own/background/combined priors and full/standard/SQ/group decompositions. Persisted lane sums, raw/centered ratings and prediction-preserving intercepts close to **1.94e-12**. A separate direct own-coefficient versus Net-sum check closes to 8.61e-11, within the unchanged 1e-9 guard. Original Parent/SQ normal equations pass without fitting or factorization: maximum relative residual 7.60e-13. All 6,613 source pins verify before and after. Current totals remain research outputs; none replaced the selected ratings.

## What was verified, and what remains limited

Independent checks cover the timing/decay locks, actual calibration preprocessing and solves, dated source/census eligibility, native prices, scorecards and paired intervals. The annual audit reproduces both development and spent benchmark artifacts within 1e-12. All seven annual cuts and 35 new-arm boards close for persisted standardNet/SQ lanes, own/background/combined priors, raw/centered ratings and scalar prediction-preserving intercepts to 1.05e-11. Annual outputs do not separately persist full 11 raw-lane arrays or per-lane intercepts; those nonexistent artifacts are not claimed audited.

Historical publication vintages are uncertified. Actual minutes/pace make forecasts conditional rating-quality tests, not availability forecasts. All 2023–26 results are spent historical diagnostics. Intervals hold learned fits and selection fixed, rather than representing full-model or selection-adjusted uncertainty. Only 7 of 187 upstream CORE fits converged; 180 exhausted the frozen 60-iteration budget. Defensive parameters and original tracking denominators stayed frozen. These are finite-budget fitted recipes, not optimal stat decays.

Six unsafe source games remain unresolved, with common date-specific whole-history Parent quarantine. No exposure weights or zero measurements were invented. Career-exposure/entrant prior-spread groups are audited; team-changer metadata is unavailable, and long-absence and calibration/reliability-by-exposure diagnostics were not performed. Full subgroup validation and live readiness are not established. No 2027 outcomes were read.

## Evidence

Operator commands and preserved numerical/source history are in the [canonical build status](VPM_PREGAME_BUILD_STATUS.md). Generated audit receipts remain excluded from Git:

- [Timing](../results/estimated_skills/pregame_actual_timing_selection_audit_v1/receipt.json) and [decay](../results/estimated_skills/pregame_actual_decay_selection_audit_v1/receipt.json).
- [Rolling stress](../results/estimated_skills/pregame_actual_rolling_stress_audit_v1/receipt.json) and [annual seven-test benchmark](../results/estimated_skills/pregame_actual_annual_benchmark_audit_v1/receipt.json).
- [Completed native development/spent audit](../results/estimated_skills/pregame_actual_native_diagnostics_audit_v1/completed_receipt.json).
- [Five current mapper fits](../results/estimated_skills/pregame_current_adjoint_map_audit_v2/summary.json).
- [Current-board algebra and source audit](../results/estimated_skills/pregame_current_adjoint_board_audit_v3/summary.json) and [all-stage workflow audit](../results/estimated_skills/pregame_actual_workflow_audit_v1/receipt.json).

The research candidate and all failed numerical versions remain preserved. No selected-model switch, publication or further outcome-driven tuning is recommended from these results.
