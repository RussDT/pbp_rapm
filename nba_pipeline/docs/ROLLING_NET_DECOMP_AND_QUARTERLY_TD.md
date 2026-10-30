# Rolling net DECOMP and quarterly TD550 RAPM

This workflow owns the completed historical backbone for the selected forward
Net DECOMP product plus a separate quarterly research RAPM family. The existing
five-year outputs do not by themselves replace the gross daily files: the
1Y-5Y net release and consumer migration must land first. Quarterly TD550
remains research-only and does not replace daily production RAPM.

## Five-season net-continuation DECOMP

The source lane is `nba_pipeline/processed_netcont_lane/`. Build both required
parquet families for every RS and playoff season before solving:

```bash
.venv/bin/python nba_pipeline/scripts/reprocess_metric.py FIRST_CHANCE 1997 2026 \
  --season-types all --workers 6 \
  --processed-dir nba_pipeline/processed_netcont_lane
.venv/bin/python nba_pipeline/scripts/reprocess_metric.py SECOND_CHANCE_CLEAN 1997 2026 \
  --season-types all --workers 6 \
  --processed-dir nba_pipeline/processed_netcont_lane
```

The rolling runner covers all 26 complete five-season windows from 1997-2001
through 2022-2026 by default:

```bash
.venv/bin/python nba_pipeline/scripts/run_net_decomp_rolling.py \
  --workers 4 --cores-per-rapm 2
```

The forward current release is the five-window 1Y-5Y family:

```bash
.venv/bin/python nba_pipeline/scripts/run_net_decomp_rolling.py \
  --current-release --release-end-year 2026 \
  --workers 2 --cores-per-rapm 2
```

The owner-expanded archive uses all 140 available rolling 1Y-5Y windows from
1997 through 2026:

```bash
.venv/bin/python nba_pipeline/scripts/run_net_decomp_rolling.py \
  --all-one-to-five-season-windows --release-end-year 2026 \
  --workers 2 --cores-per-rapm 2
```

Every solve uses ALL games, rubberband, RS/PS-aware season effects, and fixed
offense/defense alphas `2000/4000`. It does not use `--ps-prior`: that
descriptive playoff variant failed the forward gate. Components use the
five-child `DECOMP_NET_*` tree, unchanged FT value, net-priced turnover value,
and `SECOND_CHANCE_NET = SC_NET_CONT_Diff`. Outputs live under:

- component solves: `nba_pipeline/results/net_decomp_rolling/components/`
- assembled CSV/parquet: `nba_pipeline/results/net_decomp_rolling/`
- master copies: `nba_pipeline/master_results/net_decomp_rolling/`

The net family remains isolated and must not overwrite gross
`weighted_factors_decomp_*`; public migration uses explicit
`weighted_factors_decomp_net_*` artifacts and a reversible consumer cutover.

## ShotQuality-v2 explanatory and TD550 variants

`scripts/run_net_sq_decomp_rolling.py` operates on the isolated output of
`build_vpm_target_swap.py --net-lane`. It emits two intentionally separate
families:

- `--profile explanatory`: common `2000/4000` component alphas, current 1Y-5Y,
  and explicit Russell ShotQuality v2 EV plus realized residuals;
- `--profile vpm --timedecay --half-life 550`: frozen component-specific
  penalties and TD550, labeled predictive research rather than DECOMP parent.

The historical bridge supplies outcome-safe 1997-2015 EV and native v2 supplies
2016-2026. Once the target attachment stage completes, either profile can run
the complete 140-window grid with `--all-one-to-five-season-windows`; the
VPM-only 26-window five-season compatibility mode remains available through
`--all-five-season-windows`.

The restartable umbrella workflow, logs, and API-oriented artifact inventory
are documented in [RAPM Table Registry](./RAPM_TABLE_REGISTRY.md).

## Quarterly trailing-six-season TD550 RAPM

The quarterly runner produces four historical as-of estimates within every
season for which six season labels exist in the 1997-2026 data. That is 25
trailing windows, 1997-2002 through 2021-2026, and 100 snapshots total.

```bash
.venv/bin/python nba_pipeline/scripts/run_quarterly_td_rapm.py \
  --workers 4 --cores-per-rapm 1
```

Checkpoint dates are data-defined: the dates on which at least 25%, 50%, 75%,
and 100% of that season's combined RS+PS games have been completed. Same-date
games stay together. Each estimate includes the previous five complete
seasons plus the current season through the checkpoint. The solve is pure RAPM,
ALL, rubberband, RS/PS-aware season effects, and 550-day half-life time decay,
with the standard `3000/3000` RAPM penalties. Earlier-season rows remain fully
available and decay relative to the latest game in that checkpoint.

Outputs and their date/player-count receipt live under
`nba_pipeline/results/quarterly_td550/`. Partial current-season parquets are
temporary and are removed after each solve; canonical processed files are
read-only inputs.

## Validation contract

- Net parquet construction must pass the child, kappa, SC payback, and grand
  row identities at `1e-9`.
- Every expected RS/PS file must exist; partial season rebuilds are failures.
- Every net weighted-factor window must contain all requested component solves
  and report bounded parent/split coefficient gaps. Source-row tolerance is
  `1e-9`; independently solved two-decimal component CSVs may accumulate up to
  `0.075` points/100 on either fitted offense or defense side of a parent/split
  identity, or `0.30` points/100 against the separately solved RAPM total,
  before the assembler fails. Reader-facing net gaps are derived by combining
  the two guarded sides and can therefore be slightly larger than `0.075`.
  Every gap stays visible rather than being hidden in a balancing component.
- Every quarterly snapshot must produce a nonempty player result and appear in
  `quarterly_td550_manifest.csv` with its cutoff date and current-season game
  count.

## Calendar-date no-prior net-continuation VPM

`scripts/run_vpm_netcont_asof.py` builds October 1, January 1, May 1, and August 30
snapshots for season-end years 2020-2026. It uses the isolated net-continuation
EV target lane, no priors, TD550, smooth rubberband, RS/PS-aware season
effects, and the frozen ten component-specific VPM alphas. October 1 is the
preseason date in the prior calendar year; January 1 and May 1 are in the
season-end year, and August 30 is the post-season checkpoint.

For a partial season the runner filters raw rows by eligible game id and
rebuilds FIRST_CHANCE and SECOND_CHANCE_CLEAN before attaching EV targets.
This keeps the file-level continuation kappa and shot-tree baselines strictly
pre-cut. Filtering a completed-season net target parquet is not valid for an
in-season estimate. Outputs, component receipts, rubberband coefficients, and
season effects live under `results/vpm_netcont_asof_td550/`.

### Completed 2020-2026 calendar build receipt (2026-08-30)

- Built 28 boards: four checkpoints for each season-end year 2020-2026.
- Every board contains all ten component results plus ten rubberband
  coefficient files, ten rubberband effect files, ten rubberband surfaces,
  and ten RS/PS season-effect files.
- Player counts range from 823 (2020 October preseason) to 1,556 (2026 May
  and August). Every processed input ends strictly before its cutoff; the
  August 30, 2026 board ends with the available June 13, 2026 PBP.
- The maximum component or three-factor identity gap across all manifests is
  `1.78e-15`. The top-level CSV/JSON receipt contains all 28 snapshots.

### Supabase publication

`scripts/publish_no_prior_vpm_snapshots.py` validates and consolidates the 28
boards, adds calendar lookup fields and explicit model metadata, and publishes
them to the dedicated typed table
`public.nba_no_prior_vpm_td550_snapshots`. The table is research-only and does
not replace the accepted prior-centred VPM board.

```bash
.venv/bin/python nba_pipeline/scripts/publish_no_prior_vpm_snapshots.py
.venv/bin/python nba_pipeline/scripts/publish_no_prior_vpm_snapshots.py --publish
```

The primary key is `(player_id, snapshot_date)`. Indexed access paths cover
player/date, case-insensitive player name/date, calendar year/month, and
season-end year/checkpoint. Every row includes the ten offense/defense atomic
components, VPM O/D/net, the standard three-factor O/D summaries, possessions,
TD550, smooth clock-aware rubberband, RS/PS season effects, prior scale zero,
and the frozen component-alpha JSON contract.

The August 31, 2026 publication contains 34,075 unique player-date rows across
28 dates from 2019-10-01 through 2026-08-30. Database checks found zero
duplicate keys and zero nonzero-prior rows; the current Supabase publishable
key returned the August 30 Shai row through the public Data API.

## Completed 1997-2026 build receipt (2026-08-18)

- Net-continuation prep rebuilt and paired 60 RS/PS parquet sets containing
  7,157,906 rows. The maximum row gaps were `6.93e-14` for the five-child
  identity, `5.68e-14` for the kappa bridge, `1.78e-14` for the second-chance
  payback bridge, and `5.68e-14` for grand closure.
- The net runner completed 312 component solves and assembled all 26 five-year
  windows (21,618 player-window rows) in both results and master-results as CSV
  and parquet. Maximum exported coefficient gaps were `0.065` for the shot
  parent, `0.041` for the turnover split, and `0.192` against the separately
  solved offensive RAPM total; all are inside the documented guards and remain
  explicit in the tables.
- The quarterly runner completed all 100 snapshots across 25 trailing-six-year
  windows. Each window has four checkpoints and all four expected artifacts:
  player results, rubberband coefficients, rubberband effects, and RS/PS season
  effects. Player counts range from 794 to 1,096. The receipt begins with
  1997-2002 Q1 as of 2001-12-14 and ends with 2021-2026 Q4 as of 2026-06-13.
