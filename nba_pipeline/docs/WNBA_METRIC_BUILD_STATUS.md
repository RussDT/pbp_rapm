# WNBA Metric Build Status

Last updated: August 24, 2026

This is the canonical status page for the WNBA player-metric work in this
repository. It answers four questions:

1. What data do we have?
2. What metric have we actually built?
3. What has been validated?
4. What should happen next?

The short answer is that we now have a reproducible, lineup-aware 2022-2026 WNBA
RAPM pipeline, a season-forward-selected factor-specific box-prior version, and
a promising held-out benchmark. The box-prior true-prior result is now the
preferred current-season research metric. It is not yet a finished
DARKO/EPM-style product: the frozen preseason version still lacks rookie and
other no-history coverage, and the current live prior necessarily uses
current-season box production. A season-forward play-by-play shot-quality model
now exists as a research input, while a VPM-style factor-specific alpha search
failed its 2026 confirmation and was not promoted.

## Status at a Glance

| Layer | Status | Current scope |
| --- | --- | --- |
| Public classic PBP archive | Downloaded and audited | 2010-2026 |
| Public modern PBP archive | Downloaded, normalized, and lineup-validated | 2022-2026 |
| Conservative clean modern surface | Built | 1,270 games, 590,466 event rows |
| Standard processed surfaces | Built | 2022-2026 RS; 2022-2025 PS |
| Primary local metric | Built | Pure actual-score offensive/defensive RAPM |
| Legacy WNBA six-factor display | Built | TS-like scoring, TOV, and REB explanation layer |
| Clean point-value decomposition | Built | First chance plus direct second chance |
| Proprietary comparison data | Audited locally | DPM, PI RAPM, and NPI RAPM daily histories |
| Frozen 2026 validation | Built | 202 games, 33,745 possessions |
| Historical 2010-2021 RAPM | Not ready | Event lineups still need reconstruction |
| LUMA-compatible historical RAPM | Built, retrospective candidate | 2003-2026 aggregate stints plus exact 2009+ PBP Stats component priors; 5,623 games and 3,922 player-seasons |
| Historical-state prior metric | Built and walk-forward selected | Annual 2022-2025 RAPM state plus live 2026 update |
| Factor-specific box priors | Built and walk-forward selected | Public 2022-2026 box/PBP Stats features |
| Single-season box-prior RAPM/decomp | Built | Reader-ready 2022-2026 player-season history |
| Net-continuation single-season VPM | Built, research only | 2022-2025 prior labels plus current 2026 RS board |
| Joint component-alpha coordinate descent | Built, not promoted | Improved development; worse than uniform alphas on 2026 |
| PBP shot-quality expected points | Built, research only | 112,383 modern mid/three FGA; historical expansion rejected |
| RTSS-to-PBP ShotQuality student | Built, research only | 61,521 exact teacher shots; complementary blend positive on 2025 development |
| Frozen rookie/no-history prior | Not built | Current live model can use 2026 box stats; preseason model cannot |

## Metric Taxonomy

There are three local outputs. They must not be treated as interchangeable.

### 1. Primary Local Metric: Pure WNBA RAPM

This is the metric called `ours_native_frozen` in the benchmark.

For possession \(j\):

```text
actual points_j - training league mean
  = sum(offensive player effects)
  - sum(defensive player effects)
  + error_j
```

The model:

- uses one row per completed standard possession;
- estimates separate offensive and defensive coefficients;
- adjusts every player for the other nine players on court;
- uses ridge regression with the current fixed alpha of `3000`;
- centers offense and defense by possession-weighted league averages;
- reports `off`, positive-good `def`, and `net_rapm = off + def`;
- reports all three values in points per 100 possessions;
- uses no box-score prior, age curve, rookie prior, or current-season update in
  the frozen 2026 test;
- uses no time decay in the completed 2022-2025 reference build.

The primary solve uses actual `Net_Diff` scoring. Although `RAPM*.csv` also
contains luck-adjusted `Off_Diff` and `Def_Diff` targets, the master build runs
the total RAPM with `pure=True`, which explicitly replaces those targets with
`Net_Diff` and `-Net_Diff`.

This is a credible RAPM baseline. It is not yet the finished WNBA metric.

### 2. Legacy WNBA Six-Factor Display

`master_wnba_rapm.py` also runs three auxiliary lineup regressions:

| Display columns | Underlying WNBA target |
| --- | --- |
| `oTS`, `dTS` | Scoring on non-turnover possessions |
| `oTOV`, `dTOV` | Turnover indicator on standard possessions |
| `oREB`, `dREB` | Offensive-rebound indicator on rebound chances |

A possession-weighted second-stage regression maps those six auxiliary
coefficients back to total net RAPM. The 2022-2025 reconstruction has
in-sample `R² = 0.9982`.

That `R²` is a reconstruction diagnostic, not predictive accuracy. This WNBA
implementation is also not identical to the current production NBA six-factor
definitions: its `TS` surface is non-turnover scoring per possession rather than
literal player or lineup TS%, and its public columns are point-valued by the
second-stage reconstruction.

Use `net_rapm` as the overall player rating. Use the six columns only to describe
where the fitted RAPM appears to come from.

### 3. Clean Point-Value Decomposition

`clean_decomp_wnba.py` is a parallel experimental build:

```text
FIRST_CHANCE + SECOND_CHANCE_CLEAN = total scoring layer
```

Its current display includes:

- rim, midrange, and three-point frequency and conversion value;
- free-throw value, split into frequency and severity;
- first-chance turnover value;
- direct second-chance value as `oSC` and `dSC`;
- offensive, defensive, and net totals with exact displayed closure.

The completed 2022-2025 clean reference uses `off_alpha=1750` and
`def_alpha=4000`. It is useful for mechanism-level interpretation, but it did
not beat the simpler native RAPM in the first held-out 2026 test.

### 4. Historical-State Prior Clean RAPM

`build_prior_informed_wnba.py` implements the first true-prior prototype. For
season \(y\), it solves the clean possession likelihood around a dated
historical RAPM state:

```text
minimize
  ||W^(1/2) (target_y - X beta_y)||²
  + alpha_off ||beta_off,y - prior_off,y||²
  + alpha_def ||beta_def,y - prior_def,y||²

prior_y = persistence ^ year_gap * prior_state
```

This is a true prior-centered ridge solve. It is not a post-hoc blend and it is
not a multiplier applied to make the output resemble another metric.

The selection protocol was:

- build one annual state at a time from 2022 onward;
- evaluate each completed state only on the next season;
- use 2023, 2024, and 2025 as selection folds;
- search 180 offense-alpha, defense-alpha, and persistence combinations;
- treat candidates within `0.005` game-margin RMSE of the minimum as tied;
- choose the tied candidate with the best mean forward game-margin correlation;
- open the 2026 benchmark only after selection.

| Selected parameter | Value |
| --- | ---: |
| Offensive alpha | 1,750 |
| Defensive alpha | 2,000 |
| Annual persistence | 0.90 |
| Pre-2026 mean game-margin RMSE | 12.889 |
| Pre-2026 mean game-margin correlation | 0.418 |

The proprietary DPM, PI, and NPI files were not used as targets, priors,
features, or hyperparameter-selection inputs.

For the live clean decomposition, every component receives its own frozen
2022-2025 component prior. The selected dynamic total prior is reconciled with
the atomic component priors through the existing first-chance
completion/baseline bucket. Because the targets, alphas, lineups, and reconciled
priors are additive, the 2026 offense, defense, and net display residuals all
close to `0.000`.

The live file exposes the prior (`prior_*`), the current-season update
(`delta_*`), and `has_prior`. It is a descriptive current-season output and must
not be substituted into the frozen 2026 benchmark.

### 5. Factor-Specific Box-Prior Single-Season RAPM

`build_wnba_box_priors.py` is the preferred current research build. It replaces
the historical-RAPM centers above with public, factor-specific box predictions.
For each clean component and side:

1. stabilize the relevant current-season box rates;
2. train a Ridge box model only on seasons earlier than the season being scored;
3. select the box-model penalty on nested forward-year folds;
4. insert that prediction as the exact center of the single-season lineup
   ridge objective;
5. derive redundant children algebraically so the prior and fitted displays
   remain additive.

The feature families follow the mechanism being estimated. Examples include
shot-zone attempt shares and conversion for the six EFG atoms, free-throw
attempts/percentage/fouls drawn for FT, turnover types and assists for turnover
value, offensive rebounds for offensive second chance, and steals/blocks/fouls/
defensive rebounds for defensive components. Proprietary DPM, PI, and NPI
ratings are never used as features, targets, priors, or tuning inputs.

The final pre-2026 selection uses only the 2024 and 2025 next-season folds:

| Selected parameter | Value |
| --- | ---: |
| Offensive lineup alpha | 5,000 |
| Defensive lineup alpha | 40,000 |
| Box-prior scale | 1.50 |
| Direct-TOV explanatory defense alpha | 10,000 |
| Annual box-target offensive alpha | 750 |
| Annual box-target defensive alpha | 1,000 |
| Pre-2026 mean game-margin RMSE | 12.844 |
| Pre-2026 mean game-margin correlation | 0.439 |

After the deterministic lineup repair and full pre-2026 rebuild, `40,000` was
selected on the prespecified rule of best correlation within `0.005` RMSE of
the minimum. Both selected alphas originally sat on the edge of the audited
`true_prior_grid.csv` sweep — offense `5,000` was its lowest tested value and
defense `40,000` its highest — so neither edge had been probed. Both are now
resolved on development folds only, scored with the joint component-sum
future-possession objective and 2026 excluded:

| Extension | Grid | Result |
| --- | --- | --- |
| Defense (`extend_wnba_defense_alpha_grid.py`) | 40k, 60k, 80k, 120k, 160k | `40,000` held; best challenger `60,000` lost by `6.33e-6` |
| Offense (`extend_wnba_offense_alpha_grid.py`) | 1k-5k, then 6k-20k | `5,000` held; `4,000` lost by `2.85e-6`, `6,000` by `3.56e-6` |

The offense sweep had to be two-sided. A downward-only grid cannot close the
edge, because pooled MSE falls monotonically as alpha rises toward `5,000`;
the upward arm is what shows `5,000` is an interior minimum rather than a new
boundary. Pooled development MSE rises monotonically in both directions away
from it (`1.22016` at `5,000` versus `1.22057` at `1,000` and `1.22027` at
`30,000`), so the live pair is confirmed rather than merely unfalsified.

Note the scale: every gap here is in the `1e-6` range on possession MSE. The
original game-margin grid was similarly flat — nine of 27 combinations fell
inside the `0.005` RMSE tie-break band — so these decisions are correctly read
as "no evidence to move", not as sharp optima.

For 2026, `221` players receive stabilized current-season box priors and one
unresolved player receives zero. Current eligibility requires positive PBP
Stats offensive and defensive possessions; `Minutes` is retained only as a
diagnostic because `55` current rows have nonpositive normalized minutes with
valid exposure. The manifest lists those conflicts. The prior is additive by
construction:

```text
box total
  = six shot atoms
  + FT
  + first-chance completion
  + second chance

FT severity = FT - FT frequency
scoring TOV = total TOV value - bad-pass TOV value
```

The live output exposes `prior_*`, `delta_*`, and final `off`, positive-good
`def`, and `net_rapm`. It uses only the 2026 possession likelihood, so it is a
single-season RAPM with a genuine box prior rather than a multi-season RAPM or
post-hoc blend.

The Databallr publication surface no longer labels the legacy
first-chance-completion baseline as TOV. It publishes the direct
`ALT_TOV_LOSS_VALUE` family at defense alpha `10000` and reallocates
`completion - direct TOV` into the midrange/base shot-value child. The
transformation is exact and leaves headline impact unchanged. The direct alpha
was selected on 2023->2024 and 2024->2025 turnover-possession MSE; 2026 is
report-only. Canonical runner: `wnba_test/tune_wnba_direct_tov_alpha.py`.

### 6. Joint Factor-Alpha Coordinate Descent

`tune_wnba_component_alphas.py` ports the private Alt3/VPM coordinate-descent
method to the WNBA clean decomposition. This is not nine independent
cross-validations. Each candidate:

1. solves all nine additive components around their factor-specific box priors;
2. sums the resulting player coefficient maps;
3. predicts future actual possession points with the summed model;
4. changes one component-side alpha at a time;
5. repeats until no grid move improves the joint future-possession MSE.

Selection used `2023 -> 2024` and `2024 -> 2025`. The `2025 -> 2026` fold was
kept out of coordinate descent and used only as confirmation.

| Candidate | Development MSE | 2026 confirmation MSE |
| --- | ---: | ---: |
| Factor-specific coordinate descent | 1.219882 | 1.226979 |
| Uniform `5000/40000` | 1.220159 | 1.226560 |
| Maximum shrinkage | 1.220369 | 1.226976 |

The selected configuration improved development MSE by `276.984e-6` versus the
uniform board but lost `418.799e-6` on 2026. Therefore it failed confirmation
and is not promoted. The live WNBA rating remains `off_alpha=5000`,
`def_alpha=40000`, and `prior_scale=1.5`.

The marginal max-shrink ablations are still useful for deciding which lineup
components appear to contain repeatable signal. Requiring more than
`1.0e-6` MSE improvement in both development and confirmation:

| Trust label | Component sides |
| --- | --- |
| Supported diagnostic | Rim frequency O/D; rim conversion O; mid conversion D; three conversion D; first-chance completion D |
| Mixed | Mid conversion O; three conversion O; FT D; first-chance completion O |
| Prior-only at this sample | Rim conversion D; mid frequency O/D; three frequency O/D; FT O; second chance O/D |

These labels are marginal diagnostics around a jointly failed configuration,
not permission to publish the selected alpha vector. The report lives under
`wnba_test/Results/box_prior_true_prior/component_alpha_tuning/`.

### 7. Net-Continuation Single-Season VPM

The isolated `wnba_test/build_wnba_net_decomp.py` lane now ports the NBA
net-continuation accounting to the WNBA single-season box-prior model. It adds
zone-specific expected continuation to first-chance misses, removes the same
amount from second chance, reprices direct turnovers at the full net
non-turnover counterfactual, and refits all factor box priors on 2022-2025 net
labels. The live gross board and its frozen selection are not overwritten.

August 24, 2026 receipt:

| Field | Result |
| --- | ---: |
| Current games | 281 |
| Players | 229 |
| 1,000-possession qualifiers | 153 |
| 2026 rim / mid / three kappa | 0.4009 / 0.2960 / 0.3105 |
| Max additive board residual | `1.71e-13` |
| Correlation with same-date gross VPM | 0.99698 |
| Median / mean absolute VPM movement | 0.115 / 0.131 |
| Maximum absolute VPM movement | 0.623 |

Top qualified values are A'ja Wilson `+7.455`, Olivia Miles `+5.336`, Jackie
Young `+4.636`, Aliyah Boston `+4.495`, Kelsey Mitchell `+4.425`, and Veronica
Burton `+4.297`. These are descriptive current-season estimates, not a forward
validation result.

The main interpretation gain is composition. Miss-forcing no longer receives
the full value of a dead possession unless the defense secures the miss, TOV
deletes the full expected possession, and `SECOND_CHANCE_NET` reads as
opportunity-adjusted conversion. A'ja's same-date split illustrates it:

```text
gross: oTS 3.425 + oTOV 0.802 + oSC 0.646 = +4.875 offense
net:   oTS 2.984 + oTOV 0.914 + oSC 1.004 = +4.902 offense

gross: dTS 1.600 + dTOV 0.439 + dSC 0.338 = +2.378 defense
net:   dTS 1.226 + dTOV 0.500 + dSC 0.828 = +2.553 defense
```

Promotion state: research only. The row algebra, prior provenance, and current
board close, but the net-prior lane must beat the gross-prior lane on a
predeclared future actual-possession sample before publication is considered.

### 8. WNBA Play-by-Play Shot Quality

`build_wnba_pbp_shot_quality.py` builds expected points for midrange and
three-point attempts from the normalized Gabriel play-by-play. Rim is excluded
from model tuning, validation, and downstream impact scope. It uses only
pre-outcome information: shot value, distance, zone, shot family, period-clock
context, and made/missed-neutral shot-form modifiers.

The raw made/missed `actiontype`, actual outcome, actual points, full
description, and made-only assist tags are forbidden model inputs. Smoothing
was selected on the 2023-2025 season-forward folds; 2026 was opened only after
selection.

| 2026 model | RMSE | R² | MSE gain vs value baseline |
| --- | ---: | ---: | ---: |
| Shot-value baseline | 1.255962 | 0.005799 | — |
| Zone/geometry | 1.252714 | 0.010935 | 0.517% |
| Full context | 1.250690 | 0.014127 | 0.838% |
| Context + prior shooter | 1.249239 | 0.016414 | 1.068% |

The full modern model improves MSE over the shot-value baseline in every scored
year: `0.875%` in 2023, `0.863%` in 2024, `0.882%` in 2025, and `1.068%` in
2026.
That is a stable, modest PBP signal. It is not yet part of the impact metric.
Promotion is restricted to MID and THREE and requires an exact
shot-quality-plus-shotmaking target identity with RIM unchanged, followed by a
fresh season-forward component-sum gate. See
[WNBA Play-by-Play Shot Quality](./WNBA_PBP_SHOT_QUALITY.md).

The historical sample audit found 239,829 additional MID/THREE attempts from
2010-2021, but blind pooling did not pass the current-domain gate. A 2018-2025
window expanded the surface to 185,460 eligible attempts and seven development
folds, but its 2026 RMSE was `1.249319` versus `1.249239` for modern-only. The
full 2010 expansion was worse at `1.251520`. Historical windows remain opt-in
comparisons; the canonical output stays modern-only until a larger window wins
on an independent future season.

#### RTSS-to-PBP teacher/student result

The separate `train_wnba_rtss_pbp_student.py` lane uses calibrated RTSS `qSQ`
as a soft expected-points teacher on 61,521 exact 2023-2025 game/event matches,
then scores every MID/THREE shot using only PBP-observable inputs. The student
beats an identical same-row outcome learner, but it does not beat the mature
PBP model standalone. A blend selected on 2024 and frozen for 2025 improves the
context-only incumbent by `+0.001247 [+0.000151,+0.002282]` MSE; the
shooter-inclusive blend is inconclusive. Keep the result development-only,
leave RTSS 2026 sealed, and require exact EV/residual possession closure plus a
future summed-factor gate before VPM admission. See
[WNBA RTSS-to-PBP ShotQuality Student](./WNBA_RTSS_PBP_SHOT_QUALITY.md).

### Daily Current-Season Update

The current board can now be refreshed end to end without reopening model
selection:

```bash
.venv/bin/python wnba_test/update_wnba_metrics.py --as-of "$(date +%F)"
```

The command stages the official WNBA liveData CDN pull, requires a fully clean
current-season lineup file, force-refreshes the WNBA PBP Stats box totals,
rebuilds all six downstream CSV surfaces, and runs only the frozen selected
box-prior solve. It validates all published additive identities before
promoting:

- `wnba_test/Results/current/wnba_current_metrics.csv`
- `wnba_test/Results/current/latest_manifest.json`

The July 29, 2026 run added five games for a `207`-game, `34,586`-possession
surface and a `222`-player board. All 207 games cleared the conservative lineup
gate. See [WNBA Daily Metric Update](./WNBA_DAILY_UPDATE.md) for the operator
and failure-recovery contract.

### Five-Season Player History

`build_wnba_metric_history.py` applies that frozen architecture one season at a
time and writes a single reader-ready 2022-2026 table. Completed seasons use
regular season plus playoffs; 2026 uses the regular season through the current
data cutoff.

- 2023-2026 each use only that season's lineup likelihood.
- Each box-to-component prior is trained only on seasons earlier than the
  season being rated.
- 2022 is explicitly labeled `pure_warmup_no_prior`: no earlier reliable modern
  WNBA lineup season exists for a leakage-safe box prior.
- `rank` includes every player; `qualified_rank` requires 1,000 possession
  slots.
- `prior_*` is the public box prediction, `delta_*` is the lineup update, and
  `wnba_impact = wnba_offense + wnba_defense`.
- The consolidated output has 882 player-seasons, no duplicate season/player
  keys, no missing core ratings, and zero fitted display residual.

Build or refresh the history:

```bash
PYTHONPATH=. .venv/bin/python wnba_test/build_wnba_metric_history.py
```

Rebuild only the consolidated file and manifest from existing season solves:

```bash
PYTHONPATH=. .venv/bin/python wnba_test/build_wnba_metric_history.py \
  --reuse-season-outputs
```

Primary artifacts:

- `wnba_test/Results/box_prior_true_prior/history/wnba_single_season_leaderboard_2022_2026.csv`
- `wnba_test/Results/box_prior_true_prior/history/wnba_single_season_metrics_2022_2026.csv`
- `wnba_test/Results/box_prior_true_prior/history/manifest.json`

## Source Inventory

### Gabriel Public WNBA Repository

- Upstream: `https://github.com/gabriel1200/wnba`
- Ignored checkout: `nba_pipeline/external/gabriel_wnba/`
- Audited commit: `04c6bed5c87c8e43a70c7621c4062a3cbc6bb5b9`
- Commit time: `2026-07-23T21:51:32-05:00`

| Surface | Files | Rows | Coverage |
| --- | ---: | ---: | --- |
| Classic PBP | 3,969 | 1,483,539 | 2010-2026 RS/PS |
| Modern PBP | 1,273 | 591,776 | 2022-2026 RS/PS |
| Five-player lineup aggregates | 642 | 75,595 | 2010-2025 |

Important limitations:

- Modern PBP is a strict subset of classic PBP by game id.
- Classic PBP has 61 exact duplicate rows but no conflicting duplicate event
  keys in the audited checkout.
- Modern PBP has 550 header variants. It must be normalized by column name.
- The 2026 classic directory contains 126 zero-row future schedule
  placeholders.
- Modern game `1022400138` ends in the third quarter. Its complete classic
  counterpart is retained, but the incomplete modern game is excluded.
- `lineup_data/` contains season-team five-player aggregates. It is not an
  event-level paired-opponent lineup source and cannot be used directly for
  RAPM.

### Proprietary Daily Rating Archive

- Local zip: `databallr_share (2).zip`
- SHA-256:
  `254a8b4b34da1dd2803e8ecfe77ac7e33fc7a66d898a43068ae2c3b5b9310a01`
- Ignored extraction: `nba_pipeline/external/wnba_darko_daily/`

| Table | Rows | Players | Date range |
| --- | ---: | ---: | --- |
| `dpm_cols.parq` | 95,948 | 738 | 2009-06-06 to 2026-07-30 |
| `pi_rapm_ratings.parq` | 255,649 | 729 | 2009-06-07 to 2026-07-23 |
| `npi_rapm_ratings.parq` | 244,147 | 727 | 2009-09-04 to 2026-07-23 |

The tables have unique player-date keys and no nonfinite rating values.
`dpm = o_dpm + d_dpm` closes to floating-point precision. The DPM file contains
206 rows dated July 28-30, after the July 23 audit cutoff; they are excluded from
validation.

Until the provider confirms whether a date-D rating is pregame or postgame, the
only permitted live join is:

```text
latest rating_date strictly less than game_date
```

The proprietary zip, extracted tables, and player-level contents must remain
local and gitignored.

## Normalized and Processed Data

### Modern Normalization

The normalizer reconstructs event lineups from boxscore starters plus ordered
substitution blocks.

| Surface | Games | Event rows |
| --- | ---: | ---: |
| Full normalized archive | 1,272 | 591,467 |
| Conservative clean archive | 1,270 | 590,466 |

The clean surface:

- excludes the one truncated modern game before normalization;
- has no duplicate-side or cross-side player collisions;
- classifies technical/ejection actors separately from live lineup conflicts;
- repairs seven games deterministically by replaying chained same-player
  in/out blocks in source order and assigning corrected same-clock actions the
  nearest actor-compatible snapshot without changing persistent lineup state;
- excludes only `1022300075` and `1022500163`, which retain incomplete fives.

The full normalized archive is preserved for audit. Solves use the conservative
clean archive.

### Processed CSV Surfaces

The current WNBA downstream surface is CSV-based:

| Family | Files | Total rows | Available windows |
| --- | ---: | ---: | --- |
| `RAPM` | 9 | 208,082 | 2022-26 RS; 2022-25 PS |
| `TS` | 9 | 172,476 | 2022-26 RS; 2022-25 PS |
| `TOV` | 9 | 208,082 | 2022-26 RS; 2022-25 PS |
| `REB` | 9 | 100,406 | 2022-26 RS; 2022-25 PS |
| `FIRST_CHANCE` | 9 | 208,082 | 2022-26 RS; 2022-25 PS |
| `SECOND_CHANCE_CLEAN` | 9 | 208,082 | 2022-26 RS; 2022-25 PS |

There is no 2026 playoff surface yet. Do not run or label `22 26 ALL`; use
`22 26 RS`.

## Current Player Artifacts

Generated CSVs are local artifacts and remain out of git.

| Purpose | Artifact | Players |
| --- | --- | ---: |
| Frozen native training reference | `wnba_test/Results/rapm_22_23_24_25_all/weighted_factors.csv` | 287 |
| Frozen clean training reference | `wnba_test/Results/clean_decomp_22_25_all_a1750_4000/weighted_factors_alt3_efg_value_22_25_all_a1750_4000.csv` | 288 |
| Current native 2022-26 RS | `wnba_test/Results/rapm_22_23_24_25_26_rs/weighted_factors.csv` | 355 |
| Current clean 2022-26 RS | `wnba_test/Results/clean_decomp_22_26_rs_a1750_4000/weighted_factors_alt3_efg_value_22_26_rs_a1750_4000.csv` | 357 |
| Walk-forward grid and report | `wnba_test/Results/prior_informed_robust/` | 180 candidates |
| Frozen selected 2025 annual result | `wnba_test/Results/prior_informed_robust/frozen_2025_prior_informed_total.csv` | 182 |
| Frozen 2026 preseason state | `wnba_test/Results/prior_informed_robust/preseason_2026_prior_state.csv` | 288 |
| Live historical-state-prior 2026 clean decomp | `wnba_test/Results/clean_decomp_26_rs_a1750_2000_prior1_walkforward_selected/weighted_factors_alt3_efg_value_26_rs_a1750_2000_prior1_walkforward_selected.csv` | 221 |
| Box-model targets, priors, grid, and report | `wnba_test/Results/box_prior_true_prior/` | 27 final grid candidates |
| Frozen selected 2025 box-prior result | `wnba_test/Results/box_prior_true_prior/frozen_2025_box_true_prior_total.csv` | 182 |
| Live single-season box-prior 2026 decomp | `wnba_test/Results/current/wnba_current_metrics.csv` | 222 |
| Compact historical leaderboard | `wnba_test/Results/box_prior_true_prior/history/wnba_single_season_leaderboard_2022_2026.csv` | 882 player-seasons |
| Consolidated single-season history | `wnba_test/Results/box_prior_true_prior/history/wnba_single_season_metrics_2022_2026.csv` | 882 player-seasons |

The current 2022-26 files include the test season and are appropriate for
current-rating inspection only. They must never be substituted into the frozen
2026 evaluation.

## First Held-Out Benchmark

### Design

- Training: clean 2022-2025 regular season plus playoffs.
- Test: 202 clean 2026 regular-season games through July 23.
- Test possessions: 33,745.
- Local metric timing: frozen after 2025.
- Proprietary preseason timing: latest available rating before the first 2026
  game.
- Proprietary live timing: latest player rating strictly before each game date.
- Missing player ratings: league-average impact (`0`).
- Uncertainty: paired 4,000-draw game-cluster bootstrap.

This is conditional on the observed ten-player lineup and observed possession
count. It tests player-rating signal. It is not a deployable pregame forecast of
injuries, rotations, minutes, or pace.

The table is the original held-out design rerun after deterministic lineup-data
repair. Selection remains confined to pre-2026 folds, but the 2026 outcomes are
no longer untouched during this repair cycle; a later season or cutoff is still
required for fresh confirmation.

### Game-Margin Results

| Metric | Timing | RMSE | MAE | Correlation | Rated slots |
| --- | --- | ---: | ---: | ---: | ---: |
| League scoring mean | Frozen baseline | 13.807 | 10.981 | 0.281 | 100.0% |
| Local native RAPM | Frozen after 2025 | 13.319 | 10.291 | 0.374 | 80.0% |
| Local clean RAPM | Frozen after 2025 | 13.408 | 10.335 | 0.365 | 80.0% |
| Local box-prior true RAPM | Frozen after 2025 | 13.118 | 10.197 | 0.403 | 77.2% |
| DARKO DPM | Preseason frozen | 13.645 | 10.504 | 0.341 | 81.2% |
| DARKO DPM | Strictly prior day | 13.348 | 10.455 | 0.423 | 99.3% |
| PI RAPM | Preseason frozen | 13.593 | 10.426 | 0.363 | 80.5% |
| PI RAPM | Strictly prior day | 13.236 | 10.456 | 0.433 | 99.7% |
| NPI RAPM | Preseason frozen | 13.402 | 10.363 | 0.389 | 80.5% |
| NPI RAPM | Strictly prior day | 13.037 | 10.276 | 0.448 | 99.7% |

### What the Result Supports

- The frozen local native RAPM is directionally better than the league-mean
  baseline: RMSE improvement `0.488`, with a game-cluster 95% interval of
  `[-1.202, 0.205]`.
- The point estimates favor local native RAPM over preseason-frozen DPM.
- Local native RAPM and live DPM are effectively tied on RMSE:
  local-minus-DPM `-0.029`, 95% interval `[-0.577, 0.522]`.
- Live DPM has materially higher game-margin correlation than the local RAPM.
- Live NPI RAPM has the best point estimates in this first test.
- The factor-specific box-prior model improves the frozen clean point estimate
  by `0.291` RMSE and correlation from `0.365` to `0.403`.
- Against strictly-prior-day DARKO DPM, the box-prior model's RMSE difference is
  `-0.230`; the paired 95% interval is `[-0.833, 0.359]`, with a `78.3%`
  bootstrap probability that the local metric is better. The point estimate
  favors the local metric, but the interval crosses zero.
- Against strictly-prior-day NPI RAPM, the local-minus-NPI RMSE difference is
  `+0.081`, with a 95% interval of `[-0.532, 0.678]`; those models are
  indistinguishable in this sample.

### What the Result Does Not Support

- It does not prove that local RAPM “beats DARKO.”
- It does not establish a statistically conclusive head-to-head winner.
- It does not test real pregame lineup, minutes, injury, or pace forecasts.
- It does not justify tuning any choice on the 2026 test set.

Coverage is the clearest frozen-model weakness. The selected frozen box-prior
rating covers 77.2% of 2026 player slots and only 10.3% of possessions have all
ten players rated. The live proprietary series cover roughly 99% of player
slots. The live 2026 box-prior file can cover active newcomers after they record
current-season box production, but that is not a preseason solution and must
not be inserted into the frozen 2026 test.

### Scale Audit

Among 63 players with at least 200 minutes in the current 2026 PBP Stats file:

| Metric | Offense SD | Defense SD | Net SD | Net correlation with box-prior RAPM |
| --- | ---: | ---: | ---: | ---: |
| Local historical-state prior | 2.162 | 1.678 | 2.907 | 0.696 |
| Local box-prior true RAPM | 1.927 | 0.810 | 2.233 | 1.000 |
| DARKO DPM | 2.082 | 1.349 | 2.701 | 0.774 |
| PI RAPM | 2.296 | 1.407 | 3.021 | 0.799 |
| NPI RAPM | 2.189 | 1.560 | 2.995 | 0.668 |

The comparison uses the latest proprietary snapshot strictly before July 24,
2026 and is external validation only. The historical-state row has the widest
local defense. The forward-selected box-prior model is intentionally much more
conservative on defense (`0.810` SD); its net scale is therefore narrower than
DPM/PI/NPI. Future folds should continue to monitor whether this reflects real
defensive uncertainty or excessive shrinkage.

Benchmark artifacts:

- `wnba_test/Results/benchmarks/holdout_2026/summary.csv`
- `wnba_test/Results/benchmarks/holdout_2026/report.json`

## Reproducible Workflow

```bash
# 1. Sync and audit sources
.venv/bin/python wnba_test/sync_gabriel_wnba.py
.venv/bin/python wnba_test/audit_wnba_source_inputs.py \
  --as-of-date 2026-07-23

# 2. Normalize completed modern games
.venv/bin/python wnba_test/import_gabriel_modern_pbp.py \
  --seasons 2022-2026 \
  --season-types RS PS \
  --workers 8 \
  --overwrite \
  --exclude-incomplete

# 3. Create the conservative solve surface
.venv/bin/python wnba_test/validate_wnba_normalized_lineups.py --write-clean

# 4. Process every clean RS/PS input
.venv/bin/python wnba_test/process_rapm_wnba.py \
  wnba_test/GabrielNormalizedClean/WNBA22.csv
.venv/bin/python wnba_test/clean_decomp_wnba.py process \
  wnba_test/GabrielNormalizedClean/WNBA22.csv

# 5. Build frozen references
.venv/bin/python wnba_test/master_wnba_rapm.py 22 25 ALL
.venv/bin/python wnba_test/clean_decomp_wnba.py run 22 25 ALL \
  --off-alpha 1750 \
  --def-alpha 4000

# 6. Run the untouched 2026 benchmark
.venv/bin/python wnba_test/benchmark_wnba_metrics.py

# 7. Select the dynamic prior model using only 2023-2025 forward folds
.venv/bin/python wnba_test/build_prior_informed_wnba.py \
  --off-alphas 250,500,750,1000,1750,3000 \
  --def-alphas 250,500,750,1000,2000,4000 \
  --persistences 0.5,0.65,0.75,0.9,1.0 \
  --skip-current-components \
  --output-dir wnba_test/Results/prior_informed_robust

# 8. Reuse the frozen selection to build the live 2026 component file
.venv/bin/python wnba_test/build_prior_informed_wnba.py \
  --reuse-selection-dir wnba_test/Results/prior_informed_robust \
  --output-dir wnba_test/Results/prior_informed_robust

# 9. Build and select genuine factor-specific box priors, then solve 2026
.venv/bin/python wnba_test/build_wnba_box_priors.py \
  --off-alphas 5000,7500,10000 \
  --def-alphas 20000,30000,40000 \
  --prior-scales 1.25,1.5,1.75

# 10. Build the separate aggregate-stint historical bridge
.venv/bin/python wnba_test/build_luma_historical_metrics.py \
  --luma-root /path/to/LUMA/WNBA \
  --start-year 2003 \
  --end-year 2026 \
  --season-type ALL \
  --output-dir wnba_test/Results/luma_historical_our_priors
```

Repeat processing for all nine available clean RS/PS files before building the
full reference windows.

## Script Ownership

| Script | Responsibility |
| --- | --- |
| `wnba_test/sync_gabriel_wnba.py` | Safe clone or fast-forward of the public source |
| `wnba_test/audit_wnba_source_inputs.py` | Public/proprietary source audit |
| `wnba_test/import_gabriel_modern_pbp.py` | Modern column normalization and lineup reconstruction |
| `wnba_test/fetch_wnba_cdn.py` | Cached liveData PBP/boxscore fallback and normalization |
| `wnba_test/validate_wnba_normalized_lineups.py` | Lineup-quality report and clean-game exclusion |
| `wnba_test/process_rapm_wnba.py` | Standard RAPM, TS-like, TOV, and REB processed CSVs |
| `wnba_test/clean_decomp_wnba.py` | First-/second-chance processing and clean point-value solve |
| `wnba_test/master_wnba_rapm.py` | Pure RAPM plus legacy six-factor reconstruction |
| `wnba_test/benchmark_wnba_metrics.py` | Frozen/live held-out comparison and cluster bootstrap |
| `wnba_test/build_prior_informed_wnba.py` | Annual prior-centered solve, walk-forward selection, and current clean component build |
| `wnba_test/build_wnba_box_priors.py` | Stabilized factor-specific box priors, nested forward selection, and single-season true-prior clean build |
| `wnba_test/build_luma_historical_metrics.py` | Strict LUMA stint ingest, exact 2009+ modern-contract PBP Stats component priors, annual pure/true-prior RAPM, and aggregate-compatible component history |
| `wnba_test/wnba_names.py` | Shared player-id/name fallback |

## Verified Guardrails

- Normalize game ids as stable 10-character strings.
- Normalize modern source by column name.
- Apply consecutive substitution blocks at their actual event order. Do not
  retroactively apply them to every event at the same clock.
- If the same player is both in and out within one consecutive team block,
  replay that block sequentially; otherwise retain outs-before-ins grouping.
- A corrected non-substitution action may use the nearest same-clock snapshot
  containing its team actor, but must not mutate the persistent lineup state.
- Preserve the full normalized archive; exclude whole ambiguous games only in
  the conservative clean solve surface.
- Never treat aggregate `lineup_data` as event lineups.
- Never use same-day or future proprietary ratings in validation.
- Never tune alpha, priors, or feature choices on the frozen 2026 test.
- Never use proprietary ratings as a local prior or training target; they are
  external benchmarks only.
- Train a season's box prior only on earlier seasons. Current-season box
  features are allowed for the live descriptive build but never for a frozen
  preseason evaluation.
- Preserve prior-centered ridge algebra: solve with the penalty-weighted prior
  on the right-hand side, not a post-hoc possessions blend.
- Reconcile the selected total prior with atomic clean priors through the
  completion/baseline bucket before solving; offense, defense, and net display
  residuals must close exactly.
- Keep proprietary inputs and all generated player artifacts out of git.
- Rebuild all six processed CSV families after possession-logic changes.
- Keep native `oREB`/`dREB` distinct from clean direct `oSC`/`dSC`.
- Keep LUMA aggregate `OREB_OPPORTUNITY_VALUE` distinct from event-sequenced `SECOND_CHANCE_CLEAN`; aggregate stints cannot recover first-chance, ShotQuality, or net-continuation targets.

## Next Modeling Gates

1. **Frozen coverage priors:** add NCAA/international/rookie features and roster
   availability without using test-season outcomes.
2. **Factor-specific precision:** replace the shared total-level prior scale
   with component-level uncertainty only if forward folds improve and the
   additive identity remains exact.
3. **Rolling cutoffs:** add within-season frozen cutoffs, not only end-of-season
   states predicting the next year.
4. **Box/skill extensions:** test age, prior-season retention, role, and
   non-WNBA translation features inside the existing prior-centered penalty.
5. **Pregame evaluation:** add a separate rotation/minutes/pace layer if the
   desired product is a true pregame forecast.
6. **Historical lineups:** reconstruct and validate 2010-2021 event lineups
   before extending the RAPM training window.
7. **Provider contract:** confirm whether date-D proprietary ratings include
   games played on D; retain strict prior-date joins for the main benchmark.

The next useful milestone is not a more elaborate public decomposition. It is a
leakage-safe preseason no-history prior that raises player-slot coverage above
the current frozen `77.2%` without weakening the forward result.
