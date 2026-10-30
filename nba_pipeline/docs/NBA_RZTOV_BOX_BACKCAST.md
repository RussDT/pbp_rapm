# NBA box-estimated rZTOV backcast

## Same-sample RS correction — 2026-09-05

The prior reader-rate join was invalid for full-season aggregation: its 2026 snapshot has 64 Derrick White games versus 77 in the raw/processed PBP. Both are RS, but cover different games. The current builder counts FGA plus nontechnical two/three-shot FT trips directly from `002`-prefix raw games and divides by exposure from the same RS processed PBP. And-ones add no extra TSA. It verifies cached FGA, FTA, and OffPoss against those sources and fails on drift. No reader TSA rate is imported. Role predictions retain their existing feature contract, then the predicted share partitions the same-sample TSA count. `canonical_tsa_count` and `tsa_source` identify this contract. Older numerical receipts below are historical; use the regenerated manifest. See `RZTOV_SAME_SAMPLE_CORRECTION.md`.

**Status:** research-only retrospective rZTOV ledger, 1997-2026. The canonical
historical series is `box_estimated_rztov` for 1997-2013. Observed 2014-2026
rZTOV remains a separate reference and is never overwritten.

## 2026-09-05 canonical TSA correction

The emitted self-created and assisted volumes now partition canonical PBP TSA100,
not `FGA + 0.44*FTA`. The role models retain their existing predictor definitions
and fitted share/potential-assist estimates; only the shot-volume basis changes.
Turnover allowances, involvement denominators, lane centers, and derived rZTOV
are rebuilt on the corrected volumes. Earlier numerical receipts below describe
the pre-correction build; use the regenerated manifest and CSVs for current values.
This does not promote the research series or update a public database.

NBA uses the exact `TSA100` keyed by `nba_id + year` from
`pipelines/core/csvs/merged_rs_full.csv.gz` wherever present. Outside that panel
(including 1997-2000 and uncovered players), it counts raw FGA plus the first FT
of each two/three-shot trip, excluding technicals and and-ones, over resolved
OffPoss. The role board carries `tsa_source`; reconstructed rows are not claimed
to be exact matches to an unavailable reader row. The model-feature `tsa100`
remains the legacy box proxy until prediction, then the output `tsa100` is
replaced by the canonical volume. Observed modern role labels remain unchanged. The completed rebuild contains 14,595 rows: 12,298 exact reader-source joins and 2,297 explicitly labeled raw-PBP reconstructions. The six-decimal CSV export matches reader TSA100 within `5e-7`, and self plus assisted closes within `1.1e-6`; internal arithmetic remains exact to floating-point precision. Both historical and all-era diagnostic turnover ledgers were rebuilt successfully.

## Question and boundary

Observed NBA rZTOV uses three workload inputs:

- self-created true-shooting attempts per 100 offensive possessions;
- potentially assisted true-shooting attempts per 100; and
- potential assists created per 100.

The observed role files begin in 2013-14. `build_nba_rztov_box_backcast.py`
learns the two missing mappings on those observed seasons and applies them to a
repo-owned PBP/box feature contract available from 1996-97 onward. It then fits
historical turnover prices from each era's own count-basis RAPM targets. The
result is a complete retrospective ledger, not a claim that tracking was
observed before 2014. No modern turnover price is copied into the historical era.

## Sources and coverage

| Surface | Coverage | Use |
| --- | --- | --- |
| `raw_data/NBA{yy}.parquet` | 1997-2026 RS | FGA, FTA, made-shot assist tags, shot descriptors, turnovers, bad-pass turnovers, passer attribution |
| `processed/RAPM{yy}.parquet` | 1997-2026 RS | resolved offensive possessions and player IDs |
| `results/rztov_seasons_2014_2026.csv` | 2014-2026 RS | observed self-created TSA, assisted TSA, and potential-assist labels only |
| `processed/TOV{yy}.parquet` | 1997-2026 RS | count-basis total, scoring, and bad-pass turnover outcomes |
| `results/{tov,scoring_tov,badpass_tov}_*_rs_se_a1500_1500_results.csv` | rolling 5-season windows | independent offense RAPM targets for the joint price solve |

The local sibling `vpm/boxscore.csv` starts in 2014 and therefore is not used as
the historical source. Every 1997-2026 season has both required repo-owned
parquets. The current build contains 14,595 player-seasons, of which 5,369
observed-label player-seasons enter model development after the 500-possession
floor.

Older raw PBP carries the assister in `player2_id`. Modern fetched PBP does not,
so the builder parses the official `(... N AST)` tag and resolves its name only
when it maps uniquely to a player in the shooter's five-man lineup. It never
keep-firsts an ambiguity. Resolution is 98.7%-100.0% by season and at least
99.1% in all ten chronological validation seasons; the unresolved share
is recorded in `feature_source_audit.csv`.

## Stable box feature contract

All model inputs are rebuilt with one definition across the full era:

- `TSA100 = 100 * (FGA + 0.44 * FTA) / resolved offensive possessions`;
- three, rim, and residual-mid attempt frequencies from official descriptions;
- FTA, assists, assisted points, rim assists, turnovers, and bad-pass turnovers
  per 100;
- `usage_proxy = TSA100 + turnovers100`; and
- official assisted-make share.

Official assisted-make share is a useful predictor but is not the target. It is
conditional on makes, whereas the target labels potential assistance on made and
missed attempts. The feature is stabilized within each season:

```text
official share smoothed
  = (official assisted FGM + 40 * season league assisted-FGM share)
    / (FGM + 40)
```

The same-season league rate is ordinary completed-season box information. No
observed role label enters that calculation.

## Model and leakage contract

Two standardized linear ridges are fit:

1. potential-assisted attempt share, clipped to `[0, 1]`;
2. potential assists per 100, clipped at zero.

The assisted shot estimate is constrained, rather than independently fit:

```text
estimated assisted TSA100 = canonical TSA100 * estimated assisted share
estimated self TSA100     = canonical TSA100 - estimated assisted TSA100
```

This makes `self + assisted = TSA` exact; the current maximum floating-point
gap is `3.6e-15`.

Validation is expanding chronological season validation. The first scored
season is 2017: it is trained only on 2014-2016. Each later season is trained
only on earlier observed-label seasons, through the final 2026 test trained on
2014-2025. Alpha and feature family are selected on the pooled chronological
predictions. The final backcast model is then refit on all 2014-2026 labels.
That final refit is appropriate for a retrospective backcast, but means all
2014-2026 seasons are development evidence; there is no untouched promotion
holdout in this build.

## Development receipt

Possession-weighted expanding-season validation, test seasons 2017-2026:

| Target / feature family | Alpha | RMSE | R2 | Correlation |
| --- | ---: | ---: | ---: | ---: |
| Assisted-attempt share, rich all-era | 0.01 | 8.29 percentage points | 0.845 | 0.922 |
| Assisted-attempt share, official-share contract without rim fields | 0.01 | 8.45 pp | 0.838 | 0.917 |
| Assisted-attempt share, official share only | 0.01 | 10.22 pp | 0.763 | 0.881 |
| Assisted-attempt share, reduced without official share | 1000 | 12.69 pp | 0.636 | 0.803 |
| Potential assists/100, rich all-era | 0.01 | 1.05 | 0.961 | 0.984 |
| Potential assists/100, reduced | 0.01 | 1.26 | 0.944 | 0.982 |

The smoothed official share supplies the largest assisted-share improvement:
`12.69 pp -> 8.45 pp`. Rim frequency and rim-assist context then improve it to
`8.29 pp`. The selected rich contract is still all-era; “rich” here means the
additional description-derived rim fields, not optical tracking or a modern-only
box source.

## Historical pricing and exact ledger algebra

Every complete trailing-five-season window from 1997-2001 through 2022-2026 is
fit independently. The targets are regular-season, season-effect, count-basis
RAPM solves with offense/defense alpha `1500/1500`:

```text
SCORING_TOV ~ k_sc  * (scoring TOV/100 - p_self*self TSA100 - p_ast*ast TSA100)
BADPASS_TOV ~ k_bp  * (bad-pass TOV/100 - p_potast*potential assists100)
TOV         ~ k_tot * (total TOV/100 - all three priced role volumes)
```

Prices and kappas use the same joint three-target, two-lane alternating
weighted least-squares solve as the observed rZTOV explorer. Equation blocks
are inverse-residual-variance weighted; players need 2,000 possessions in a fit
window. Player-cluster bootstrap bands quantify identification. The 1997-2000
rows use the earliest complete 1997-2001 fit and carry
`early_window_inherited = 1`; they do not use a modern price.

After pricing, scoring and bad-pass allowances are possession-weighted centered
separately within every season. Exact identities include:

```text
sc_excess + bp_excess = excess
self TSA100 + assisted TSA100 = canonical TSA100
tov_saved_100 = -excess
tov_saved = tov_saved_100 * offensive possessions / 100
rZTOV = 100 * excess / (self TSA100 + assisted TSA100 + potast100 + TOV100)
```

Negative rZTOV is good. Rows qualify for rate display at 1,000 offensive
possessions and 10 involvement plays per 100.

## Modern overlap audit

For 2014-2026 the build creates two non-production audit surfaces on exactly
the same player-seasons, possessions, turnover counts, rolling target solves,
and windows. One refits prices with observed roles; the other refits prices
with box-estimated roles. The outputs report price deltas,
possession-weighted rZTOV correlation/RMSE, and Spearman rank stability. This
isolates role-estimation error from player coverage and turnover outcomes.

Completed-build receipt:

| Window | Self price | Assisted price | Potential-assist price | Total / SC / BP target R2 |
| --- | ---: | ---: | ---: | ---: |
| 1997-2001 | 13.73% `[12.38, 15.11]` | 5.99% `[5.38, 6.63]` | 14.77% `[14.42, 15.19]` | 0.721 / 0.652 / 0.351 |
| 2009-2013 | 11.13% `[10.50, 11.82]` | 6.50% `[6.04, 6.90]` | 13.69% `[13.25, 14.07]` | 0.710 / 0.581 / 0.298 |
| 2022-2026 | 10.92% `[10.28, 11.77]` | 4.53% `[3.95, 4.94]` | 13.44% `[13.03, 13.92]` | 0.657 / 0.533 / 0.227 |

Brackets are player-cluster bootstrap 5th/95th percentiles from 100 draws per
window. Across all 26 windows, the median 90% band width is 1.66 percentage
points for self, 1.11 for assisted, and 0.85 for potential-assist price. The
modern common-player 2014-2026 rollup has weighted rZTOV RMSE `0.433`,
correlation `0.982`, and Spearman rank correlation `0.979` across 1,077
qualified players. Maximum absolute estimated-minus-observed price deltas over
the nine modern windows are 0.66, 0.52, and 0.49 percentage points respectively.

## Build and outputs

```bash
.venv/bin/python nba_pipeline/scripts/build_nba_rztov_box_backcast.py

# Refit/re-score without rebuilding the raw feature cache
.venv/bin/python nba_pipeline/scripts/build_nba_rztov_box_backcast.py \
  --reuse-feature-cache
```

Generated artifacts live under `results/nba_rztov_box_backcast/` and remain
gitignored:

| Path | Contents |
| --- | --- |
| `nba_rztov_box_role_backcast_1997_2026.csv` | estimated roles, box TSA, smoothed official share, observed labels where available, status flag |
| `nba_box_estimated_rztov_1997_2013.csv` | canonical research-only historical rZTOV backcast |
| `nba_box_estimated_rztov_1997_2026_diagnostic.csv` | estimated-role method across all seasons for diagnostics |
| `nba_observed_rztov_2014_2026_reference.csv` | modern observed series retained and labeled separately |
| `historical_box_price_path_1997_2026.csv` | rolling prices, fit diagnostics, and bootstrap 5th/95th percentiles |
| `historical_box_lane_centres.csv` | exact per-season scoring and bad-pass centers |
| `modern_overlap_price_comparison.csv` | common-window observed-role versus estimated-role price deltas |
| `modern_overlap_player_seasons.csv` | common-player rZTOV comparison |
| `modern_overlap_summary.csv` | weighted RMSE/correlation and rank stability |
| `all_era_box_features_1997_2026.parquet` | reproducible all-era feature cache |
| `feature_source_audit.csv` | per-season row counts and assist-ID resolution |
| `chronological_model_validation.csv` | every alpha/family summary |
| `chronological_selected_predictions.csv` | selected expanding-season predictions |
| `manifest.json` | status, coverage, validation, identities, and paths |

Focused tests:

```bash
.venv/bin/python -m pytest -q \
  nba_pipeline/tests/test_nba_rztov_box_backcast.py
```

## Supabase publication

The site-facing results hub publishes the two canonical reader surfaces as
separate datasets:

- `nba_box_estimated_rztov_1997_2013` — 6,186 historical player-seasons;
- `nba_observed_rztov_2014_2026` — 5,370 modern player-seasons.

Sites read both from `analytics_current_rows` and join or label them explicitly
by dataset slug. Never replace this seam with the all-era estimated diagnostic.
The raw role board, observed-label training fields, price diagnostics, and
overlap audit rows are not published through the generic hub.

## Interpretation constraints

- Label every 1997-2013 rate as the `box_estimated_rztov` research series.
- Do not call the official assisted-make percentage the potential-assisted
  attempt percentage. The former is one input to the estimator.
- Do not overwrite observed 2014-2026 rZTOV workload with the estimates.
- The historical series uses era-specific rolling prices. Only 1997-2000 inherit
  the earliest complete historical 1997-2001 fit; no modern price is backfilled.
- The historical feature denominator is resolved offensive possessions. That
  contract is consistent across all seasons but is not a claim of full optical
  possession coverage.
