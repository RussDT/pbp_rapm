# Daily-trained VPM versus EPM

The next-game margin-error difference between daily-trained VPM and EPM remains unresolved.

This run tests the prior trained on **predictive stats immediately before each historical game, paired with that same game's actual lineup scoring**. It uses the existing `pregame_season_home` mapper, with annual past-only coefficient fits and fresh player stats and lineup ratings before each game day. The sample is **4,772 regular-season games on 651 dates**, ending seasons 2023–2026.

## Next-game results

| Completed season | Games | Daily-trained VPM | Annual-trained, matched weighting | Previous annual-trained H365 winner | Same-date EPM |
|---|---:|---:|---:|---:|---:|
| 2022–23 | 1,218 | 12.576031 | 12.587304 | 12.587762 | 12.802573 |
| 2023–24 | 1,188 | 13.736464 | 13.652505 | 13.649901 | 13.767560 |
| 2024–25 | 1,181 | 13.939058 | 13.965827 | 13.956101 | 13.938990 |
| 2025–26 | 1,185 | 14.181776 | 14.113414 | 14.100266 | 14.139078 |
| Equal-season headline | 4,772 | 13.622290 | 13.592865 | 13.586407 | 13.671691 |
| 2023–24 onward, equal-season | 3,554 | 13.953620 | 13.911910 | 13.903357 | 13.949369 |

Values are RMSE in final-score margin points; lower is better. The headline is the square root of mean seasonal game MSE, giving each season equal weight.

Daily-trained VPM versus EPM has paired MSE gain **+1.348366** squared points, defined as EPM error² minus VPM error². Its conditional 95% paired-game interval is **[-0.687286, +3.421812]**; the separate whole-season interval is **[-0.693116, +4.311532]**. Positive favors daily-trained VPM.

## What daily training changes

In the annual-snapshot approach, one dated player-stat snapshot is used throughout its following 365-day training cohort. In the daily-trained approach, each historical game uses that player's predictive stats from immediately before that game. Both learn the stat-to-impact mapping through a regularized lineup likelihood with player/cohort residuals, season effects and home context; neither regresses stats onto independently shrunken RAPM labels.

The primary daily-trained arm and the matched annual control both have **no additional calibration-game half-life**. Their source data, training games, preprocessing, context, residual penalties and deployment are matched; their training input timing differs. The previous annual-trained winner has the extra H365 calibration-game weighting, so that contrast changes both timing and weighting. H365 concerns mapper calibration games, while TD550 concerns the downstream lineup-history weighting. Neither is the memory of every predictive-stat filter.

The mapper coefficients are refit at annual past-only cuts, not every day. At deployment, every arm receives fresh pregame stats, entrant priors, exact daily decayed Parent ratings and the same SQ corrections. Two shared sparse factorizations per date serve the three arms. Existing upstream filters and mappers were reused; this task performed a new 651-date rating replay.

| Reference for daily-trained VPM | MSE gain, positive favors daily-trained | 95% paired-game interval | 95% season interval |
|---|---:|---:|---:|
| Annual snapshot, same weighting | -0.800804 | [-1.501712, -0.085392] | [-2.116930, +0.515323] |
| Previous annual-trained H365 winner | -0.976330 | [-1.660931, -0.290319] | [-2.337965, +0.385306] |

Annual-snapshot training improves the matched daily-training arm at the conditional paired-game 95% level. The whole-season interval still spans zero.

The daily-trained season/home arm had lower 2020–22 development error than the daily-trained arm without context. It was fixed for this requested replay before inspecting its 2023–26 results. At the owner's request, a provisional side score of the first 339 completed dates was preserved separately; it did not change the registered full replay, arms or scoring. The earlier overall development winner remained annual-trained H365; the new comparison does not select a replacement from spent results.

## Fixed supplementary samples

| Sample | Games | MSE gain versus EPM | 95% paired-game interval | 95% season interval |
|---|---:|---:|---:|---:|
| All games; veteran-only ratings | 4,772 | +0.935693 | [-1.013431, +2.849903] | [-1.622755, +4.061934] |
| 2023–24 onward | 3,554 | -0.118626 | [-2.621380, +2.329697] | [-1.209240, +0.855256] |
| 2023–24 onward; veteran-only ratings | 3,554 | -0.656741 | [-2.910356, +1.624398] | [-2.354256, +1.275286] |
| Low continuity | 474 | -8.677360 | [-18.707676, +1.064525] | [-23.871482, +6.516762] |
| Low continuity; veteran-only ratings | 474 | -9.390083 | [-17.599281, -0.935500] | [-24.104367, +5.324200] |

All paired procedures use 4,000 resamples with fits and model choices held fixed. Season resampling has only four seasons, or three in the later subset. Supplementary intervals are descriptive and not adjusted for multiple comparisons.

## Scoring and limits

Both metrics use identical games, actual player minutes, game duration, actual pace and a 3-point home advantage added once. The forecast is the home-minus-away difference in minute-weighted rating sums, divided by game duration and multiplied by pace/100, plus home advantage. Both systems have native ratings for every retained played minute; no missing-player fallback is used. The veteran sensitivity replaces players with fewer than 250 prior NBA minutes by −2 in both systems.

EPM is the identical saved neutral APIv2 (`game_optimized=0`) same-date series used in the previous comparison, preserved bit-for-bit. Provider `asof_dt` represents predictions excluding games on that date. These are refreshed historical reconstructions with uncertified original publication vintages. The 2023–24 onward sensitivity lies beyond the provider's stated SPM training windows through 2022–23, as recorded in the previous source audit. This is a conditional player-rating-quality test using realized minutes and pace, not a full availability/minutes forecast or the provider's complete game-prediction product.

These seasons were already explored in earlier VPM research and remain spent retrospective diagnostics. Prior/filter parameters and choice of arm are held fixed in the intervals. Only 7 of 187 upstream CORE fits reported convergence; 180 exhausted their frozen iteration budget. No optimal-decay or live predictive superiority claim follows. Selected v61, the historical library and 2027 outcomes remain unchanged.

## Verification

The previous annual-trained winner reproduces its saved game forecasts to 0 points. Independent player aggregation agrees to 1.07e-14; maximum saved rolling normal-equation error is 7.17e-15. The independent audit verifies all game/player inputs, dated boards, score rows and paired intervals. New full raw eleven-lane arrays are not persisted in this replay; no separate full-lane/intercept audit is claimed.

An inherited data receipt pins an older API TypeScript aggregation file whose working version has since changed. The replay verified the exact historical Git bytes in a separate archived copy, restricted to that one provenance-only path and SHA256. Scientific data, executable Python, coefficients, numerical objectives and tolerances retain their original pins. Frozen source/model receipts and the current API working file were preserved.

Artifacts: `/Users/russ/Documents/Codex/basketball/nba/research/vpm/vpm/results/harness/vpm_daily_trained_epm_provenance_v2`. `comparison/` contains game predictions, player inputs, scorecards, EPM paired comparisons, timing controls and their source/output receipt. `independent_audit.json` records the independent checks. The registration preserves the primary arm and controls. The [earlier annual-trained comparison](/Users/russ/Documents/Codex/basketball/nba/research/rapm/nba_pipeline/docs/VPM_DAILY_EPM_COMPARISON.md) remains separate.
