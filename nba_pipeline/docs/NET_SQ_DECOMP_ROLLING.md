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

## 2027 state: runnable up to one owner decision

Nothing has produced a 2027 window, and the chain below is why. The first
three links are stale rather than broken; the fourth is a decision.

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
   `season_end_year` 2026. That table is merely **stale**, not blocked: it was
   last built July 22, 2026, three months before `raw_data/NBA27.parquet`
   landed on October 26 (815,808 bytes, 26,523 events), and
   `build_pbp_shot_quality.py --years 2027` parses it cleanly — 9,298 shots
   through all four estimation stages in about two seconds, no warnings.
   Rebuild it **beside** the existing artifact, never over it: `--output-name`
   defaults to `pbp_shot_quality_all`, so the `pbp_shot_quality_1997_2026`
   name was passed explicitly and a rerun under a new name cannot clobber it.
   Nothing should repoint a consumer off the 2026-named file as part of the
   rebuild. 9,298 new shots against 6,327,218 existing is 0.15% of the corpus,
   so the empirical-Bayes global prior re-fit shifts history only slightly —
   measure that drift on the 1997-2026 overlap rather than assuming it.
   `prepare()` also reads `RAPM{yy}` and `RAPM{yy}_PS` unconditionally, so the
   no-playoffs-yet problem recurs there.
5. `clean_sq.score_seasons` ends at 2026 and `clean_sq.protected_seasons` is
   `[2027]`; 2027 is also `confirmation_seasons` for `offseason_alpha` and
   `parent_baseline` and `protected_seasons` for `entrant_priors` and
   `monthly`. **This is the one genuine blocker.** Scoring 2027 through that
   lane touches the reserved confirmation season the root instructions
   protect, so it is an owner decision rather than a mechanical step. The
   nightly below gates on exactly this and reports the stage BLOCKED instead
   of editing a research config to get past it.

Do not substitute a different lineage's output for the missing window:
`a2000_4000` is a different model, and the published series name is part of
the contract.

## Nightly refresh

`scripts/nightly_net_sq_td550_refresh.sh` runs the whole chain on a schedule,
one gate per stage. It exists because nothing in this repo was scheduled: no
launchd plist, no caller for `build_pbp_shot_quality.py`,
`run_net_sq_decomp_rolling.py` or `publish_td550_rolling_archive.py`. That one
gap is why the three surfaces are each stale at a different date — shot
quality at July 22, the net-lane targets at September 6, the published archive
at end year 2026 — while the raw 2027 season file landed October 26.

A tick costs a handful of `stat()` calls and one JSON read, and runs nothing
unless an upstream actually moved:

| Stage | Gate |
| --- | --- |
| A shot quality | newest `raw_data/NBA*.parquet` is newer than `pbp_shot_quality_1997_{SEASON_YEAR}.parquet`, or that file is absent. Built beside the 2026-named artifact under a season-specific `--output-name`. |
| B per-shot EV | `clean_ev.parquet` older than stage A's output, **and** the clean_sq config covers the season: target year in `score_seasons`, absent from `protected_seasons`. A season that fails the coverage half reports BLOCKED and takes C and D with it. |
| C net lane | the three `{FIRST_CHANCE,SECOND_CHANCE_CLEAN,RAPM}{yy}.parquet` targets missing from the net-lane processed dir, or older than `clean_ev.parquet`. |
| D rolling solve | any of the five td550 window CSVs for the release end year missing from `master_results`, or older than stage C's targets. Runs with `--allow-missing-playoffs`. |
| E publish | **off by default.** Always validates without `--publish`; writes only when `PUBLISH=1` is set explicitly. |

The gates are mtime-derived and so self-calibrating: a stage that fails leaves
its output old, and the next tick retries rather than waiting out an interval.
Stage E's success stamp (the sha256 of the staged `sources.json`, under
`logs/nightly_net_sq/`) is the one piece of state, written only after the
publish returns 0. A failure anywhere in A-D skips the publish entirely, since
a failed C skips D and leaves the windows unresolved.

The publish default is off because `nba_no_prior_vpm_td550_rolling` is live and
feeds the site, and an unattended write to it has not been authorised. Env
overrides: `SEASON_YEAR`, `FORCE`, `PUBLISH`, `ONLY`, `NIGHTLY_STUB`,
`NIGHTLY_STUB_FAIL`, `NIGHTLY_PIPELINE_DIR`, `VPM_REPO`, `VPM_SQ_CONFIG`.
`NIGHTLY_STUB=1` logs each stage's command instead of running it, which is how
the gate and stamp logic was verified.

It is deliberately **not installed** — no launchd plist, no cron entry. To
start it:

```bash
# ~/Library/LaunchAgents/com.russellthomas.nba-net-sq-td550.plist
# ProgramArguments: /bin/bash <repo>/nba_pipeline/scripts/nightly_net_sq_td550_refresh.sh
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.russellthomas.nba-net-sq-td550.plist
```
