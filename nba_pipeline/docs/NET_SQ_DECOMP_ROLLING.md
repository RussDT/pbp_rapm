# Net SQ DECOMP rolling archive (1Y-5Y)

Operator surface for the rolling Net DECOMP boards built by
`scripts/run_net_sq_decomp_rolling.py` and published by
`scripts/publish_td550_rolling_archive.py`. Methodology for the underlying
net-continuation tree is in [NET_CONTINUATION_DECOMP.md](./NET_CONTINUATION_DECOMP.md);
this file covers only what runs, what it writes, and how the season rolls over.

## Two lineages, deliberately distinct

`master_results/net_sq_decomp_rolling/` holds both, 420 files each
(140 windows x csv/parquet/manifest.json):

| Profile | `spec_id` | Alphas | Filenames |
|---|---|---|---|
| `explanatory` | `net_sq_resid_a2000_4000` | uniform 2000/4000 for every component | `weighted_factors_decomp_net_sq_resid_{window}_all_rb_se_a2000_4000.*` |
| `vpm` | `net_sq_resid_vpmalpha` (`_td{half_life}` when time-decayed) | the ten frozen component-specific no-prior VPM penalties in `VPM_ALPHAS` | `weighted_factors_vpm_net_sq_resid_{window}_all_rb_se_td550.*` |

The uniform-alpha lineage exists so the ten-component split can be compared
with the parent Net DECOMP accounting. The component tune is the predictive
board lineage and must not be described as the parent RAPM. They are not
interchangeable: the series the Databallr site reads is
`net_sq_resid_vpmalpha_td550`, so a window computed under
`a2000_4000` cannot stand in for it.

## Published archive

`publish_td550_rolling_archive.py` concatenates every
`weighted_factors_vpm_net_sq_resid_*_td550.csv`, asserts an exact window set
and one shared `component_alphas` contract across windows, stages
`results/td550_rolling_publication/{rows.parquet,sources.json}` with a
per-window sha256 receipt, and with `--publish` writes the Supabase table
`nba_no_prior_vpm_td550_rolling`. As of October 30, 2026 the archive is
92,125 rows across 140 windows, end years 1997-2026. That table is live and
feeds the site: validate without `--publish` first.

## Season rollover (October 30, 2026)

Two literals blocked a 2027 (2026-27) run, both fixed:

- The producer's `expected_input_paths` required an RS and a PS half of
  `FIRST_CHANCE`/`SECOND_CHANCE_CLEAN`/`RAPM` for every season in the window,
  so a season that has played no playoffs yet failed all five windows on
  `*27_PS.parquet`. `--allow-missing-playoffs` exempts **the window's own end
  year only**; every earlier season still requires both halves, because a
  historical season missing its playoffs is a broken target build rather than
  an in-season state. A partial end-year playoff half (some but not all three
  families) is a hard error for the same reason. Affected windows record
  `end_year_playoffs_included: false` and `missing_playoff_inputs` in their
  manifest, so an in-season board is never silently compared with the
  historical RS+PS windows.
  This precheck was the only blocker: `rapm.py` warns and skips a missing
  input file, and its sole RS+PS row requirement is inside the `ps_prior`
  branch, which this runner does not use.
- The publisher hardcoded `expected = {... for end in range(1997, 2027) ...}`
  and then asserted exact set equality in both directions, so nothing with a
  2027 window could validate — and simply bumping the literal to 2028 would
  have broken the *current* archive by demanding 2027 windows that do not
  exist. The bound is now read off the newest end year present on disk, with
  completeness still exact, so a partially computed newest season fails rather
  than publishing half of it. `--max-end-year` pins it for a deliberate check.

Offline coverage: `tests/test_td550_rolling_season_rollover.py` (10 tests).
`run_net_sq_decomp_rolling.py --release-end-year` still defaults to `2026`;
the nightly chain passes it explicitly, so this is a known follow-up rather
than a live bug.

## 2027 state: blocked upstream

The compute cannot run yet. Chain, from the bottom:

1. `master_results/net_sq_decomp_rolling` has no 2027 window — expected,
   nothing has produced one.
2. The producer's net-lane inputs
   `results/vpm_target_swap_netcont/processed/{FIRST_CHANCE,SECOND_CHANCE_CLEAN,RAPM}27.parquet`
   are absent. That directory holds 180 parquets covering 1997-2026, both
   halves each. With `--allow-missing-playoffs` the five 2027 windows report
   `missing_inputs=3 exempt_playoff_inputs=3`; the three are this RS half.
   Canonical `processed/` does have 2027 RS (`FIRST_CHANCE27`, `RAPM27`,
   `SECOND_CHANCE_CLEAN27`, built October 26, 2026) and no `_PS`.
3. Building them with `build_vpm_target_swap.py --net-lane --years 2027`
   requires an `--ev-parquet` of per-shot expected values for 2027. None
   exists. The modern lane is the sibling VPM repo's
   `results/harness/vpm_clean_sq_preseason_v2/sq/clean_ev.parquet`
   (1,614,212 rows, seasons ending 2016-2026); the only `ev_pts` table in this
   repo is the 1997-2015 historical bridge.
4. Producing 2027 EV means running the sibling `harness/chronological_sq.py`,
   whose `clean_sq.shot_source` is
   `results/pbp_shot_quality/pbp_shot_quality_1997_2026.parquet` — max
   `season_end_year` 2026. The repo-owned PBP shot-quality table itself has no
   2027 rows and would have to be extended first. Its `prepare()` also reads
   `RAPM{yy}` and `RAPM{yy}_PS` unconditionally, so the no-playoffs-yet
   problem recurs there.
5. `clean_sq.score_seasons` ends at 2026 and `clean_sq.protected_seasons` is
   `[2027]`; 2027 is also `confirmation_seasons` for `offseason_alpha` and
   `parent_baseline` and `protected_seasons` for `entrant_priors` and
   `monthly`. Scoring 2027 through that lane touches the reserved confirmation
   season the root instructions protect. Extending the PBP shot-quality table
   and scoring 2027 is an owner decision, not a mechanical next step.

Do not substitute a different lineage's output for the missing window:
`a2000_4000` is a different model, and the published series name is part of
the contract.
