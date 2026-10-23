# Daily VPM versus same-date EPM

Daily VPM has lower game-margin error in this conditional retrospective comparison.

The comparison uses **4,772 regular-season games on 651 dates**, ending seasons 2023–2026. The VPM arm is the frozen `static_season_home_h365` research recipe, with player inputs and lineup-based ratings updated before each game day. Its prior mapper is trained on annual snapshots; H365 weights historical calibration games. This is neither the selected v61 series nor the separate 1997–2026 annual-history reconstruction.

## Next-game results

| Completed season | Games | Daily VPM RMSE | Same-date EPM RMSE |
|---|---:|---:|---:|
| 2022–23 | 1,218 | 12.587762 | 12.802573 |
| 2023–24 | 1,188 | 13.649901 | 13.767560 |
| 2024–25 | 1,181 | 13.956101 | 13.938990 |
| 2025–26 | 1,185 | 14.100266 | 14.139078 |
| Equal-season headline | 4,772 | 13.586407 | 13.671691 |
| Pooled games | 4,772 | 13.579479 | 13.665705 |
| 2023–24 onward, equal-season | 3,554 | 13.903357 | 13.949369 |

RMSE is in final-score margin points; lower is better. Equal-season RMSE is the square root of the mean seasonal game MSE, not the mean of seasonal RMSEs.

The headline paired MSE gain, defined as **EPM error² minus VPM error²**, is **+2.324695** squared points. Its conditional 95% paired-game interval is **[+0.505651, +4.150987]**; the separate whole-season interval is **[+0.309370, +4.364613]**. Positive favors VPM. Both procedures use 4,000 resamples and hold the learned fits and earlier recipe selection fixed.

## Fixed supplementary comparisons

| Sample | Games | MSE gain, positive favors VPM | 95% paired-game interval | 95% season interval |
|---|---:|---:|---:|---:|
| All games; veteran-only ratings | 4,772 | +1.277146 | [-0.528527, +3.105735] | [-1.108857, +3.663150] |
| 2023–24 onward | 3,554 | +1.281548 | [-0.942616, +3.460224] | [-0.477303, +3.225904] |
| 2023–24 onward; veteran-only ratings | 3,554 | +0.000324 | [-2.166525, +2.126179] | [-1.295418, +2.218687] |
| All required players have native EPM | 4,772 | +2.324695 | [+0.505651, +4.150987] | [+0.309370, +4.364613] |
| All required players have native EPM; veteran-only ratings | 4,772 | +1.277146 | [-0.528527, +3.105735] | [-1.108857, +3.663150] |
| Low-continuity games | 474 | -3.887480 | [-12.017724, +4.097094] | [-17.825513, +10.050554] |
| Low-continuity games; veteran-only ratings | 474 | -6.511882 | [-14.426341, +1.438790] | [-18.879952, +5.923163] |

The 2023–24 onward and veteran-only gains remain unresolved. Low-continuity point estimates favor EPM, with intervals spanning zero. These secondary samples were fixed before scoring and do not select a replacement model. Multiple subgroup intervals are descriptive and not adjusted for multiple comparisons.

## Fair scoring and coverage

Both metrics use the identical retained games, actual player minutes, game duration, pace and 3-point home advantage. Each team’s minute-weighted player-rating sum is divided by game duration, then the home-minus-away difference is multiplied by pace/100 and home advantage is added. The existing VPM predictions were independently replayed; no VPM filter, mapper or RAPM model was refitted.

The headline uses native player ratings where present. Missing ratings alone receive the established draft/replacement fallback using only minute accounting and draft metadata available before that game. The veteran-only sensitivity assigns the same −2 replacement value to all players with fewer than 250 prior minutes in either system. It is a separate comparison, not the headline.

| Completed season | Native EPM share of played minutes | Missing EPM player-game rows | All-native games |
|---|---:|---:|---:|
| 2022–23 | 100.000000% | 0 | 1,218 |
| 2023–24 | 100.000000% | 0 | 1,188 |
| 2024–25 | 100.000000% | 0 | 1,181 |
| 2025–26 | 100.000000% | 0 | 1,185 |

Both systems have native ratings for 100% of retained played minutes; no fallback was actually needed. Actual minutes and pace make this a conditional player-rating-quality test. It does not evaluate projected availability, expected minutes, or the provider’s complete proprietary game-prediction product. EPM is its neutral `game_optimized=0` player metric; home advantage is added once, identically to VPM.

## Source timing and interpretation

All EPM dates were freshly retrieved from the same current APIv2 model (`epm_source=v2`, `scope=all`). The [provider API documentation](https://dunksandthrees.com/api/v2/documentation) specifies that `asof_dt` predictions exclude games played on that date. `game_dt` describes an attached game and is not the last training observation. Exact duplicate neutral ratings on trade days are collapsed only when all scored identity/date/rating fields agree; original responses remain pinned. Old EPM cache files are not mixed into this comparison.

The [current EPM methodology](https://dunksandthrees.com/about/epm) lists SPM training windows through 2022–23. The 2023–24 onward summary was therefore specified before scoring as a source-based sensitivity. These API responses are refreshed historical reconstructions, not certified snapshots of what was published at the time. Provider design choices may also reflect later history.

VPM outcomes through 2026 have already been explored in earlier research. These are spent retrospective diagnostics, not a fresh independent promotion holdout. Only 7 of 187 upstream CORE fits reported convergence; 180 exhausted their frozen iteration budget. No optimal-decay claim follows. The selected model pointer, historical library, prior comparisons and 2027 outcomes remain unchanged.

## Verification and artifacts

All 4,772 game forecasts and 102,012 player-game inputs were independently reconstructed from pinned raw EPM, actual-minute sources and saved VPM forecasts. Independent aggregation gap is 1.42e-14 points. All 84 MSE/RMSE score rows and 8 paired estimates and intervals reproduce. 33 focused/shared checks pass.

Artifacts: `/Users/russ/Documents/Codex/basketball/nba/research/vpm/vpm/results/harness/vpm_daily_epm_v3`. The `comparison` directory contains game predictions, player inputs, coverage, scorecards, paired comparisons and a source/output receipt. `independent_audit.json` records independent verification. `scoring_plan.json` and `owner_model_choice.json` preserve the specification and owner’s choice of the frozen winner. No selected-pointer change or public deployment is made.
