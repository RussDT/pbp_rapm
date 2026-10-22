# Expanded annual VPM: feature study

**Recommendation from the verified annual results: retain selected v61.** The broader annual feature search did not establish an improvement. The owner authorized this study on October 22, 2026; v72 is the bounded-feature experiment, `vpm_expanded_annual_bounded_v1`. Possession-level and current-Net independent checks both pass. The research candidate is archived; the selected model remains unchanged.

## What was trained

We tested 12 models: the unchanged clean annual control; a small alternate-representation/overlap addition; broad banks assigned to their native sides or both offense and defense at four regularization settings; and two engineered variants with six interactions/four training-only hinges. The source library has 129 canonical concepts/166 source heads, plus derived inventory fields. These counts describe available evidence, not independent coefficients or features automatically admitted to the winning model.

Each training example associates an annual, neutral player snapshot with the following 365 days of actual lineup scoring. The model fits that lineup likelihood with the full offensive/defensive covariance and regularized player/cohort residuals with offense/defense precisions 2000/4000. It does not regress onto shrunken RAPM labels. The baseline conditional mapper, Parent4000/8000/TD550, actualFT/SQ, entrants and historical whole-game quarantine stay fixed. H365 weights historical calibration games; it is not a newly optimized skill-filter decay.

Additional features use only earlier training snapshots for physical admission, domain transforms, robust support/scale, missing indicators and rank screening. Their coordinates are bounded by 10; the legacy prefix is unchanged. Joint group ridge handles correlation with old features. Total and every Net response share the same basis and penalties. New direct linear solves are checked numerically; upstream transported skill filters retain their finite-iteration qualifications.

Selection is prequential on 2020–22 annual conditional game-margin errors: outer 2020 has a declared strong-regularization default, outer 2021 tunes on 2020 only, and outer 2022 on 2020–21. We choose the family by equal-season outer MSE, then freeze its final precision before later-year diagnostic scoring. All these historical years have been used in prior research; this procedure does not create an untouched holdout.

## Development result

All 36 development fits and 37,668 predictions on 3,139 common games passed independent audits. Game counts are 1,055/909/1,175 for 2020/2021/2022. The frozen winner is `transport_overlap`, the small alternate-representation addition. At the current cut it has 34 offense and 31 defense basis coordinates, versus 26 per side in the clean control. These include missing indicators and transformations; they are not counts of independent physical skills. The larger native, both-side and engineered banks were actually fitted and compared, rather than silently excluded.

| Prequential family | Equal-season game RMSE |
| --- | ---: |
| Small overlap addition — selected | 13.583652 |
| Clean annual control | 13.584409 |
| Engineered bank | 13.586293 |
| Both-side broad bank | 13.596840 |
| Native-side broad bank | 13.613669 |

The interval reported here is the conditional interval saved by the selection stage; the separately seeded seven-test diagnostic interval differs slightly. The selected-versus-control MSE gain is +.020559 squared margin points, conditional 95% game interval from the selection stage [−.066247, +.104438]. The gain is unresolved and extremely small. This was a declared 12-candidate screen, not exhaustive optimization. Larger banks did not win this bounded screen; that does not establish that their underlying statistics are useless. Precision/family choice and fitted coefficients are held fixed in these retrospective intervals.

## Later-year annual results

The frozen expansion was evaluated on 4,782 common regular-season games in 2023–26 (1,218/1,188/1,226/1,150), 120 team-seasons and the same seven registered tests. The low-continuity subset has 470 games and 13 team-seasons. It uses average home/away returning-minute fraction below 50%. The veteran-only test replaces players with fewer than 250 prior NBA minutes by the declared replacement rating. All forecasts, scores, paired intervals, source identities, fallbacks and exposure accounting passed independent reconstruction.

| Test | Expanded | Clean control | Current v61 | EPM |
| --- | ---: | ---: | ---: | ---: |
| Game margin (points) | 13.929040 | 13.927453 | 13.892012 | 13.934471 |
| Team net rating (points/100) | 3.189882 | 3.186538 | 3.114328 | 3.118215 |
| Team wins (percentage points) | 8.674967 | 8.682426 | 8.543543 | 8.630276 |
| Low-continuity game margin | 14.100122 | 14.097343 | 14.415333 | 14.478896 |
| Low-continuity team wins (pp) | 8.908097 | 8.966265 | 10.428206 | 11.300223 |
| Veteran-only game margin | 13.962626 | 13.962388 | 13.976767 | 14.019490 |
| Veteran-only team wins (pp) | 8.834162 | 8.844297 | 8.865106 | 8.942453 |

These annual-snapshot forecasts use a different input schedule from the previous rolling replay, so the two headline RMSE values should not be compared directly. RMSE uses equal-season weighting. Team win-percentage errors above are converted from fractions to percentage points; game and net-rating rows retain their stated units. Lower is better. The small addition is marginally worse than its clean control on overall game and team-net errors; none of its seven paired gains over that control is resolved.

For overall game MSE, positive gain favors the expansion:

| Reference | MSE gain | Conditional 95% game interval | Whole-season sensitivity interval |
| --- | ---: | --- | --- |
| Clean control | −0.044194 | [−0.101449, +0.017365] | [−0.10237, +0.02420] |
| Current v61 | −1.030135 | [−2.532066, +0.479590] | [−3.77669, +1.28160] |
| EPM | +0.151330 | [−1.806417, +2.125988] | [−2.49446, +2.79712] |

All overall differences remain unresolved. Current v61 has the better overall game, team-net and team-win point estimates. Low-continuity game gains against v61/EPM clear the registered game-cluster intervals, but whole-season sensitivity intervals cross zero. The matched clean control is slightly better than the expansion on low-continuity game error, so that contrast cannot be credited to the new features.

The expansion and clean control have native ratings covering 100% of realized retained minutes; v61 covers 89.075–91.906% and EPM 88.683–91.697% by season. This describes rating coverage, not observation of every library field. All systems use the identical retained schedule, fallback and veteran-override rules. Comparisons against v61/EPM consequently change source, player universe, admission and entrant coverage as well as the learned prior. The clean-control contrast is the appropriate test of this declared feature expansion.

These are conditional retrospective intervals: realized minutes and pace, learned fits and the selected specification are held fixed. The added whole-season game check exhausts the 4^4 resamples of the four spent seasons. It is sensitivity evidence, not a new selection rule.

## Current ratings and Net decomposition

The September 1, 2026 board independently passes source, chronology and direct-equation checks for both models. Each emits 1,785 players from 1,787 queries, with 582 players having positive 2026 exposure; IDs 2610/1629741 remain omitted under unchanged source gates. This is a dated research board, not a certified October roster.

The same historical whole-game quarantine removes 44 games/9,076 rows; retained history is 13,898 games/2,735,920 rows. Own, entrant/background and combined priors; full and standard Net lanes; TS/TOV/SC and SQ; raw/centered ratings; lane intercepts; and prediction-preserving scalar intercepts all reproduce. Maximum saved closure is 1.94e−12, independent direct-own reconstruction gap 1.97e−11 and relative Parent/SQ normal residual 7.59e−13. All 8,868 canonical source pins verify before and after. The clean control's final ratings match the previous H365 board within 2.54e−11.

Unlike the annual benchmark products, these current products persist the full lane arrays and lane intercepts, and those arrays were audited. The current calibration uses only matured 2015–24 snapshots; the latest training target is June 22, 2025. The 2025 snapshot is correctly unmatured at this cut. No selected-model pointer was changed.

## Possession-level prediction

Scientific native diagnostics are complete on the same annual retained schedules: 623,624 native rows/3,139 development games and 949,371 rows/4,782 later-year games. Both periods pass independent reconstruction of all saved endpoints and paired intervals.

Every entry below is the expansion's MSE gain over the clean control, in units of 10⁻⁶ squared points per native possession. Positive favors the expansion. The statistic is the equal-season average of seasonal possession MSE, rather than game-margin MSE. Intervals hold learned filters, fits and selection fixed.

| Endpoint / intercept | Development gain | Later-year gain | Later-year 95% game interval | Later-year whole-season interval |
| --- | ---: | ---: | --- | --- |
| Own prior / common past mean | -64.484 | -116.463 | [-128.187, -104.713] | [-168.344, -64.581] |
| Own prior / residual centered | +21.532 | +13.686 | [+5.499, +21.692] | [-3.512, +26.482] |
| Combined prior / common past mean | -48.074 | -90.597 | [-101.848, -78.993] | [-137.831, -43.363] |
| Combined prior / residual centered | +1.414 | -5.556 | [-13.827, +2.425] | [-25.649, +7.876] |
| Final Parent + SQ | +1.486 | +1.494 | [-2.317, +5.272] | [-1.990, +5.262] |

The prior-only conclusion depends materially on the intercept convention. Under the primary common-past-mean baseline, both own and combined priors worsen; under the secondary past-residual-centered baseline, the own-prior point estimate improves, but its later-year whole-season interval still crosses zero. The combined secondary gain is unresolved. Final Parent+SQ improvement is tiny and unresolved under both game and season intervals. Neither a favorable secondary prior result nor a changed baseline calibration establishes a final predictive win.

The independent native audit checks actual FG+FT scoring, whole-game identity, both intercept conventions, source/TD exposure, saved annual totals and all 30 O/D/both endpoints and 15 paired contrasts per period. It does not claim to audit full lane arrays absent from the annual products; the separately persisted current lanes are covered above.

## Reproduction and completion

All 11 registered scientific stages exit successfully. Independent workflow verification checks every command, recipe, checkpoint and worker, selection-before-diagnostics chronology, and a fresh canonical closure of 9,805 files/4,215 JSON nodes/5.33 GB. The selected-v61 pointer remains SHA08e5882c463d2ac10eccc8161afd92f14650afd8585634b4008227fde75d05d5. The frozen controller intentionally ends at `registered_science_complete_pending_independent_report`, with `full_experiment_finished=false`; that marker must not be rewritten. The separate ignored `finishing_queue_bounded_v1/independent_completion.json` records final source/stage/report verification and the retain-v61 recommendation. No additional model selection or retuning is permitted from these results.

## Why v71 was rejected

The previous feature transform regressed additions on unrestricted legacy standardized fields. Rare missing-data flags then amplified the selected model's defensive priors to about −786,293 and−393,150 for two physically usable sparse-history players. The clean control stayed normal and reproduced the older control within 9.19e−12. A strict Net guard stopped the first later-year replay before any later-year scores were produced; compensated arithmetic confirmed a real extrapolation problem.

V71's models, development selection and failed outputs are preserved. V72 removes that auxiliary projection, uses a bounded raw added bank and retrains the entire development grid. It does not clip old player ratings, relax tolerances or add a remainder to force Net closure. The new rule was chosen from input-support evidence, with no later-year performance results used.

## Evidence and limitations

The ten later/current calibration maps independently pass normal-equation, exact response/Net, support and past-only chronology checks. Maximum added coordinate is below 10; maximum independent coefficient gap is 6.19e−15, normal residual 1.14e−12 and Net gap 4.26e−12. On the formerly failing 1,785-player packet, original missingness and exposure are unchanged, the largest absolute prior is 22.948 and Net closure is 5.90e−12 under the unchanged 1e−9 guard.

Annual products persist total priors, final ratings, centers and scalar intercepts. They do not separately persist all 11 raw lane arrays or lane intercepts; full annual lane-array auditing is not claimed.


- Seven-test reconstruction: `expanded_bounded_actual_seven_test_audit_v1/spent.json`, SHA23590a0787734f5e03287337612acef08628972f137985805fdebced7a20d90d. Exact control metadata and supplementary season sensitivity: `spent_parity_and_season_sensitivity.json`, SHA36efab330a8f9a632461ad04e505849b19dcaf85076f315c54bf58ff44bc6c03.
- Ten later/current map checks: `expanded_bounded_actual_calibration_audit_v1/spent_current_summary.json`, SHA1339408e03fabd93e715682af1ac0184d7a36faebcf311ea97e5062a5759ccac.
- Former failure packet: `expanded_bounded_former_failure_packet_audit_v1/receipt.json`, SHA3b068b113418e8ed7cddec234c722f3915d3a41fb8f62c2c47e916519f209204.
- Complete workflow/source verification: `expanded_bounded_workflow_audit_v1/receipt.json`, SHA6d29be7bdf4b7901d0c2acd7defb767b9611e5b5a7e8b444410d8e9bb3f8b9ad.
- [Compact machine-readable evidence](evidence/vpm_expanded_annual_bounded_v1.json) contains score values, units and hashes.
- Both independent native periods: `expanded_bounded_native_audit_v1/completed_receipt.json`, SHAc3a9a5c11a016c7710fd9e7c9f492b489d2bf40965794f5d82b969faede758ec. Development SHA1d9004248a31bcd168bcc479ff561eab2d6c8af32ef5bc324c6f7a9cda7ebff0; later-year SHA6a6a507d88a6986f454b07515ef5b1600fa2ed0b890bff2d3013ab1efd490efa. All actual FG+FT whole-game gaps are zero; maximum game-SSE gap5.69e−14, paired-interval gap1.92e−17.
- Current exact equations and source history: `expanded_bounded_current_adjoint_board_audit_v2/receipt.json`, SHAaf09a2015bd1a0815cd0598aa344f4d1bade6bdcb7af5f245f4afef27fc94e9e. Compact summary SHA97d44938c153dd8f36867c58a5082a9a24691ea6b66c401376b482586e89d7d9.
- Methodology and continuation: [expanded annual prior](VPM_EXPANDED_ANNUAL_PRIOR.md).
- All 36 independent fit/support checks: ignored `expanded_bounded_actual_calibration_audit_v1` receipts. Maximum independent normal residual2.11e−12, Net gap3.11e−12; clean-control coefficients agree with old H365 within4.87e−12.
- Raw source/operator reconstruction: `expanded_bounded_actual_source_audit_v1/receipt.json`, SHA00ac3ad6378a48769d9834611c4c15b52a5c6c95642578918f0cccb1e3a3e632.993,138 actual rows/5,104 games/3,252 earlier references; direction difference0, full-covariance relative residual1.61e−15.
- Complete forecast audit: `expanded_bounded_development_forecast_audit_v1/receipt.json`, SHA026d1decbe57403c8c0fcbdd0ead527d701ec82020dc2109f762be10aa2aa35a. All predictions reproduce within 1.07e−14; control IDs, metadata and TD exposures match.
- Independent selection: `expanded_bounded_actual_selection_audit_v1/receipt.json`, SHA6804dc1e856bfc8f1383faad7c1393de7c9faedb70bf1223b6e2c92312f9221a. Choices and 4,000-draw interval endpoints reproduce exactly.
- Failure packet: `expanded_failed_packet_diagnostic_v1/receipt.json`, SHAae4c9084aeb0e786f10943dc696ba98854617c280c553fb6697088c6464dc796. Bounded-rule final review: SHA4c48507d60e29c638376de10543a58de14e41c84d524fcf59b9d75a18507a993.166 focused VPM checks and 9 independent RAPM audit checks pass.

Game forecasts condition on realized minutes/pace. Historical provider publication vintages remain uncertified. Only 7 of 187 clean CORE fits converged; 180 exhausted 60 iterations. In the broader history library, 38 of 2,504 fitted recipes converged, 2,466 exhausted 30 iterations and 2,476 other annual slots used bootstrap recipes. These upstream filters have finite-optimizer-budget limits; additional library recipes are transported to masked observations, not newly refitted hyperparameters. Selected-v61/EPM comparisons also change source/universe/coverage, whereas the clean-control contrast isolates this declared feature recipe. This study uses the independent full-history profile; the older cross-stat-corrected profile was not silently admitted under changed observation masks. Annual products are not a daily replay or a certified live roster. No 2027 outcomes, selected-pointer switch or publication occurred.
