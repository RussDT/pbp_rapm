# WNBA Gabriel Ingest and Initial Metric Benchmark

For the consolidated project status, metric taxonomy, artifact inventory, and
next modeling gates, see
[WNBA Metric Build Status](./WNBA_METRIC_BUILD_STATUS.md).

This workflow ingests Gabriel's public WNBA play-by-play archive, reconstructs
modern event lineups from boxscore starters plus substitutions, builds the
existing WNBA possession surfaces, and benchmarks frozen local metrics against
strictly prior-date proprietary ratings.

## Source Provenance

Public source:

- repository: `https://github.com/gabriel1200/wnba`
- local ignored checkout: `nba_pipeline/external/gabriel_wnba/`
- audited commit: `04c6bed5c87c8e43a70c7621c4062a3cbc6bb5b9`
- commit time: `2026-07-23T21:51:32-05:00`

Proprietary input:

- local source zip: `databallr_share (2).zip`
- SHA-256: `254a8b4b34da1dd2803e8ecfe77ac7e33fc7a66d898a43068ae2c3b5b9310a01`
- ignored extraction directory: `nba_pipeline/external/wnba_darko_daily/`
- tables: `dpm_cols.parq`, `pi_rapm_ratings.parq`, and
  `npi_rapm_ratings.parq`

Never commit the zip, extracted parquet files, cached WNBA boxscores, normalized
raw CSVs, processed CSVs, solver outputs, or benchmark outputs.

## Reproducible Commands

Sync the public checkout without resetting or overwriting local source changes:

```bash
.venv/bin/python wnba_test/sync_gabriel_wnba.py
```

Audit the public and proprietary inputs:

```bash
.venv/bin/python wnba_test/audit_wnba_source_inputs.py \
  --as-of-date 2026-07-23
```

Normalize every completed modern game. `curl_cffi` is required because Akamai
returns an HTML block page to plain HTTP clients. Boxscores are cached locally.
The `--exclude-incomplete` flag is explicit because one 2024 modern source file
ends during the third quarter; its complete classic feed remains preserved.

```bash
.venv/bin/python wnba_test/import_gabriel_modern_pbp.py \
  --seasons 2022-2026 \
  --season-types RS PS \
  --workers 8 \
  --overwrite \
  --exclude-incomplete
```

Validate every lineup and write the conservative solve surface:

```bash
.venv/bin/python wnba_test/validate_wnba_normalized_lineups.py --write-clean
```

Process each clean raw season file through both the native four-metric path and
the clean first-/second-chance path:

```bash
.venv/bin/python wnba_test/process_rapm_wnba.py \
  wnba_test/GabrielNormalizedClean/WNBA22.csv
.venv/bin/python wnba_test/clean_decomp_wnba.py process \
  wnba_test/GabrielNormalizedClean/WNBA22.csv
```

Repeat for each RS/PS file through the desired endpoint, then build the completed
2022-2025 references:

```bash
.venv/bin/python wnba_test/master_wnba_rapm.py 22 25 ALL
.venv/bin/python wnba_test/clean_decomp_wnba.py run 22 25 ALL \
  --off-alpha 1750 \
  --def-alpha 4000
```

Run the held-out 2026 benchmark:

```bash
.venv/bin/python wnba_test/benchmark_wnba_metrics.py
```

Do not request `22 26 ALL` before a 2026 playoff surface exists. Use `22 26 RS`
for a current regular-season-only run and label it accordingly.

## Audited Coverage

The public checkout contains:

| Surface | Files | Rows | Coverage |
| --- | ---: | ---: | --- |
| Classic PBP | 3,969 | 1,483,539 | 2010-2026 RS/PS |
| Modern PBP | 1,273 | 591,776 | 2022-2026 RS/PS |
| Five-player lineup aggregates | 642 | 75,595 | 2010-2025 |

Important source findings:

- Modern PBP is a strict subset of classic PBP by game id.
- Classic PBP has one stable header and 61 exact duplicate rows. Those duplicate
  event keys have identical payloads; conflicting duplicate keys were not found.
- The modern files have 550 header variants because liveData fields are emitted
  dynamically. Always normalize by column name.
- The 2026 classic directory includes 126 zero-row future schedule placeholders.
  Require a terminal final event before ingest.
- `pbp_modern/1022400138.csv` stops in the third quarter. Its classic counterpart
  contains the complete game. The current modern clean solve excludes this game
  rather than inventing missing modern events.
- `lineup_data/` is season-team five-player aggregate data, not event-level
  paired opponent stints. It cannot be joined directly onto PBP as RAPM lineups.

Modern normalization produced:

- 1,272 games and 591,467 event rows after explicitly excluding the truncated
  modern game.
- zero duplicate-side or cross-side player collisions.
- deterministic repair for chained same-clock substitutions and corrected
  same-clock actor ordering recovered seven previously excluded games.
- two games remain ambiguous and excluded: `1022300075` and `1022500163`.
- a conservative clean surface of 1,270 games and 590,466 event rows.

The completed 2022-2025 native build contains 175,506 RAPM possessions and 287
players. The native six-factor reconstruction has in-sample `R² = 0.9982`.
That is an algebra/reconstruction diagnostic, not a predictive score.

Current through-July-23, 2026 regular-season artifacts are:

- native six-factor: `wnba_test/Results/rapm_22_23_24_25_26_rs/weighted_factors.csv`
  with 355 rated players;
- clean point-value decomposition:
  `wnba_test/Results/clean_decomp_22_26_rs_a1750_4000/weighted_factors_alt3_efg_value_22_26_rs_a1750_4000.csv`
  with 357 rated players and exact displayed component closure.

These current files include 2026 results and are for current ratings and
inspection. They are not inputs to the frozen 2026 benchmark below.

## Proprietary Ratings Audit

| Table | Rows | Players | Date range |
| --- | ---: | ---: | --- |
| `dpm_cols.parq` | 95,948 | 738 | 2009-06-06 to 2026-07-30 |
| `pi_rapm_ratings.parq` | 255,649 | 729 | 2009-06-07 to 2026-07-23 |
| `npi_rapm_ratings.parq` | 244,147 | 727 | 2009-09-04 to 2026-07-23 |

Primary player-date keys are unique and numeric rating columns have no nulls.
`dpm = o_dpm + d_dpm` closes to floating-point precision. Sixteen player ids
have multiple historical name strings, while no name string maps to multiple
ids.

`dpm_cols.parq` includes 206 rows dated after the audit cutoff, on July 28-30,
2026. The benchmark never uses them. Until the provider confirms whether a
date-D value is pregame or postgame, every live validation join uses the latest
player rating with `rating_date < game_date`.

## Rebuilt Held-Out Result

Training inputs are the clean 2022-2025 RS+PS surfaces. The holdout is 33,745
possessions across 202 clean 2026 regular-season games through July 23. Every
processed possession is resolved to a home or away offense.

This is the original held-out design rerun after a deterministic source-data
repair. Model selection still uses only pre-2026 folds, but the 2026 results
have now been inspected during the repair cycle. Treat them as repaired
benchmark evidence, not a second untouched confirmation.

This is a conditional-on-observed-lineup test: each held-out possession is
scored using the ten players actually recorded on court, and predicted
possession points are summed using the observed possession counts. It measures
the incremental signal in player ratings. It is not a pregame forecast of
rotations, injuries, or pace.

Game-margin results:

| Metric | Timing | RMSE | Correlation | Rated player slots |
| --- | --- | ---: | ---: | ---: |
| League scoring mean | Frozen baseline | 13.807 | 0.281 | 100.0% |
| Local native RAPM | Frozen after 2025 | 13.319 | 0.374 | 80.0% |
| Local clean point-value RAPM | Frozen after 2025 | 13.408 | 0.365 | 80.0% |
| Local factor-specific box-prior RAPM | Frozen after 2025 | 13.118 | 0.403 | 77.2% |
| DARKO DPM | Preseason frozen | 13.645 | 0.341 | 81.2% |
| DARKO DPM | Latest strictly prior day | 13.348 | 0.423 | 99.3% |
| PI RAPM | Preseason frozen | 13.593 | 0.363 | 80.5% |
| PI RAPM | Latest strictly prior day | 13.236 | 0.433 | 99.7% |
| NPI RAPM | Preseason frozen | 13.402 | 0.389 | 80.5% |
| NPI RAPM | Latest strictly prior day | 13.037 | 0.448 | 99.7% |

The local native metric improves game-margin RMSE by `0.488` points relative to
the league-mean baseline. Its 4,000-draw game-cluster interval is
`[-1.202, 0.205]`, so this remains directional rather than conclusive.

The local frozen metric and live DARKO DPM are effectively tied:
local-minus-DPM RMSE difference `-0.029`, 95% interval `[-0.577, 0.522]`.
Live NPI RAPM is directionally better than the local frozen metric by `0.282`
RMSE, but that paired interval also crosses zero. Do not call these
head-to-head differences conclusive from 202 games.

The later season-forward-selected factor-specific box-prior model improves the
local point estimate to `13.118` RMSE and `0.403` correlation. Relative to live
strictly-prior-day DARKO DPM, its RMSE difference is `-0.230`, with a paired
95% interval of `[-0.833, 0.359]` and a `78.3%` bootstrap probability that the
local metric is better. Relative to live NPI RAPM, the difference is `+0.081`
with a 95% interval of `[-0.532, 0.678]`. The point estimates are encouraging,
but neither head-to-head result is conclusive.

Coverage is the most actionable gap. Frozen 2025 ratings cover only about 80% of
2026 player slots because of rookies, returners, expansion, and roster churn;
the selected box-prior rating covers `77.2%` of slots and `10.3%` of
possessions have all ten players rated. Missing ratings are
set to league average (`0`). The live proprietary series cover approximately
99% of player slots. The first metric-development priority is therefore a
leakage-safe prior for unseen and low-possession players, not a more elaborate
public decomposition.

## Next Modeling Gates

1. Add preseason priors for rookies, expansion players, and returners, then rerun
   the same frozen 2026 holdout without changing the test set.
2. Tune offense and defense shrinkage only with grouped-by-game,
   season-forward folds ending in 2025. Do not use 2026 to select alpha.
3. Evaluate native, clean, and prior-centered versions on several rolling
   season holdouts, not only 2026.
4. Confirm the proprietary date-label contract. Keep strict prior-date joins
   even after confirmation for the main benchmark.
5. Reconstruct and audit 2010-2021 historical event lineups separately. The
   aggregate `lineup_data` files are useful positive constraints, but they are
   not sufficient event lineups.
6. Report game-cluster uncertainty and player-slot coverage with every metric
   comparison.
