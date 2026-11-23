# Scoring Impact + rZTOV: Daily Build for the Season in Progress

One command builds the Scoring Impact board and the rZTOV explorer for the
current season and promotes the result only when it passes its checks. It was
added 2026-11-23 for 2026-27 (`2027`).

It is a local stopgap. The plan of record moves this job to Cloudflare; until
then nothing here is scheduled, nothing is written to Supabase, and the
publication registry is unchanged.

## Run it

```bash
cd nba_pipeline/scripts
../../.venv/bin/python run_scoring_rztov_daily.py              # season from NBA_RAPM_SEASON_YEAR, default 2027
../../.venv/bin/python run_scoring_rztov_daily.py --season 2027
../../.venv/bin/python run_scoring_rztov_daily.py --offline     # no Databallr read; reuses the last promoted snapshot
../../.venv/bin/python run_scoring_rztov_daily.py --no-promote  # build and check, leave the result in work/
```

It needs the season's raw play-by-play (`raw_data/NBA27.parquet`) to be current,
so run it after the daily fetch. A full run takes about 1 minute 45 seconds.

`--season` must be after the frozen fit. `--season 2026` is refused.

## What it runs

| Step | Script | Seconds |
| --- | --- | --- |
| `transition` | `score_historical_transition.py` (current season, frozen calibration) | 9 |
| `box_backcast` | `build_nba_rztov_box_backcast.py --last-season` (workload estimator) | 63 |
| `explorer` | `build_rztov_explorer.py --frozen-payload --last-season` | 2 |
| `playtype_mix` | `build_td_playtype_mix_board.py --seed-dir --last-feature-season` | 6 |
| `lineup_tov` | `build_lineup_tov_board.py --seed-board`, role alphas fixed at 300 | 2 |
| `scoring_board` | `build_historical_scoring_security.py --last-season --refresh-anchor-from-databallr` | 18 |
| `scoring_page` | `build_historical_scoring_impact_html.py` | 1 |
| `audit` | `audit_scoring_rztov_reconciliation.py --fetch-live` | 3 |

Every child runs under the interpreter that launched the runner. The exact
arguments are in `step_commands()`; each step's output is in
`work/logs/<step>.log`.

## Freeze

`scoring_season.FROZEN_FIT_SEASON = 2026`. Nothing is fit or tuned on a later
season:

- Break-evens and lane point prices: the 2022-26 window. 2026-27 rows carry
  `source = box_estimated`.
- Self-created / assisted / potential-assist estimator: labels and CV through
  2026.
- Transition calibration, playtype role calibration, lineup role alphas: as
  frozen.

Completed seasons are reused from the frozen `_2026` artifacts, never rebuilt.
The builders' defaults still end at 2026, so every documented command keeps
reproducing the frozen files. The runner never overwrites them, and it never
writes to `results/sq_transition_backfill/` or `docs/`.

When 2026-27 finishes, moving the freeze is a deliberate owner step: rebuild the
count solves and price targets, refit, and re-audit. It is not something the
daily job does.

## Rules for a season in progress

These are declared rules. None was tuned on 2026-27.

- **Floors.** A season after the freeze with fewer than 1,230 regular-season
  games is `in_progress`. Its row floor is `max(100, 500 x progress)`
  possessions and its qualifying floor is `max(200, 1000 x progress)`, where
  progress is games played over 1,230. At 247 games the floors are 100.4 and
  200.8. Seasons inside the fit keep 500 / 1,000 whatever their length.
- **Undated newest games.** A game with no date whose id is above every dated
  game of its season is dated the day after the last dated game. This applies
  only past the freeze and is counted in the mix and lineup manifests
  (`trailing_undated_games`).
- **One snapshot for scoring context.** Past the freeze, TSA100, rTS and role
  come together from the Databallr `player_zts` snapshot when that row is
  complete (`scoring_context_source = databallr_zts_observed`). Mixing local
  rTS with snapshot role broke the zTS identity mid-season. The local values
  stay on the row as `local_rts` and `local_tsa100`, and the scoring manifest
  reports the gap under `coverage.post_fit_scoring_context`.
- **Known limit of that rule.** Possessions and the turnover ledger still come
  from local play-by-play, so they can sit a game ahead of or behind the zTS
  snapshot. It closes when one job builds both.

## Checks and promotion

Two checks gate promotion:

1. **Reconciliation audit.** The 34 frozen checks plus 8 for estimated rows
   (`board_vs_rztov_estimated`: rZTOV, both excesses, both lane prices and
   turnover value on the board against the explorer CSV, and coverage both
   ways). 42 in total.
2. **Frozen rows.** Every row through 2026 in the scoring board, explorer
   seasons, playtype-mix board and lineup board must equal the frozen artifact
   in every frozen column, blanks included. New columns are ignored.

Layout under `results/scoring_rztov_daily/<season>/` (gitignored):

| Path | Meaning |
| --- | --- |
| `work/` | the build in progress; wiped at the start of every run |
| `current/` | the last build that passed; the only directory consumers read |
| `previous/` | the build `current` replaced |
| `last_failure.json` | present only while the most recent run failed |

A failed step, a failed audit or a changed frozen row leaves `current/`
untouched. Yesterday's board keeps serving.

`current/daily_manifest.json` records the build id (`<UTC time>-g<RS games>`),
season state and floors, git commit and dirty flag, raw PBP hash, step timings,
audit result, frozen-rows result and current-season row counts.

First promoted build, `20261123T213052Z-g247`: 42/42 audit checks; frozen rows
identical on 11,605 scoring, 5,370 explorer, 14,565 mix and 14,583 lineup rows;
389 current-season rows on the board, 341 qualified.

## Outputs in `current/`

| File | Contents |
| --- | --- |
| `scoring/historical_scoring_security_1997_2027.csv` | scoring board with `season_status` |
| `scoring/historical-scoring-impact.html` | reader page; live floors shown for the season in progress |
| `explorer/rztov_seasons_2014_2027.csv` | explorer season rows |
| `explorer/rztov-explorer.html` | explorer page |
| `box_backcast/nba_rztov_box_role_backcast_1997_2027.csv` | estimated workload: self-created, assisted, potential assists |
| `mix/td_playtype_mix_board_1997_2027.csv` | playtype-mix board |
| `lineup/lineup_tov_board_1997_2027.csv` | lineup turnover board |
| `audit/reconciliation_receipt.json` | the 42 checks |

## Not done

- **`--solves run` is untested end to end.** It adds the three 2023-27 count
  solves and the price targets. They are not consumed while the freeze holds,
  so the default skips them.
- **No schedule.** No launchd job was installed.
- **No publish.** The four `_2026` Supabase datasets are unchanged and no
  `_2027` dataset is registered.
- **A live failing run has not been observed.** The failure path is covered by
  unit tests only.

## Tests

```bash
.venv/bin/python -m pytest -q nba_pipeline/tests/test_scoring_rztov_daily.py
```

17 tests: season tags, the fit cap, partial-season floors, the snapshot rule,
the estimated-row checks, the frozen-rows comparison, the step list, promotion
and the frozen-season refusal.

## Related

- [Databallr integration handoff](./SCORING_RZTOV_DATABALLR_HANDOFF.md)
- [Historical scoring and ball security](./HISTORICAL_SCORING_SECURITY.md)
- [rZTOV](./RZTOV.md) and the [box backcast](./NBA_RZTOV_BOX_BACKCAST.md)
- [Lineup turnover board](./LINEUP_TOV_BOARD.md)
- [Playtype-mix board](./TD_OVERALL_PLAYTYPE_MIX_OTS.md)
