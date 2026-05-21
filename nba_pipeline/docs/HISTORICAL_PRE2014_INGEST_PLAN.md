# Historical Pre-2014 Ingest Plan

## Goal

Get `1997+` NBA play-by-play into a raw parquet shape close enough to the current
`NBA{YY}.parquet` contract that the existing RAPM processors can run with minimal
or no downstream changes.

The important constraint is not "can we download old play-by-play text?".
It is "can we produce per-event rows with stable game order, scores, player IDs,
and reconstructed `home_player1..5` / `away_player1..5` lineups?".

## Current Conclusion

The implemented production path for the current pre-2014 backfill is Gabriel's
`merged_playbyplay/old_data` team-split parquet archive:

```bash
python nba_pipeline/scripts/build_gabriel_old_pbp.py \
  --years 1997-2013 \
  --season-types all \
  --metrics RAPM,TOV,REB,TS \
  --report nba_pipeline/validation/gabriel_old_pbp/build_1997_2013_report.json
```

That command downloads/checks out `old_data`, merges the team files into season
raw parquets, converts them into this repo's raw schema, and writes both regular
season and playoff processed parquets:

- raw RS: `nba_pipeline/raw_data/NBA{YY}.parquet`
- raw playoffs: `nba_pipeline/raw_data/NBA{YY}_PS.parquet`
- processed RS: `nba_pipeline/processed/{RAPM,TOV,REB,TS}{YY}.parquet`
- processed playoffs: `nba_pipeline/processed/{RAPM,TOV,REB,TS}{YY}_PS.parquet`

The 2026-05-01 full run built all `1997-2013` RS/PS raw files plus all 136
processed metric parquets with no skipped metric entries and no game-side
inference failures. Solver smokes passed for `RAPM 13 13 RS --pure`,
`RAPM 13 13 PS --pure`, and `TOV` / `REB` / `TS` 2013 RS.

The older ESPN / SportsDataverse plus Basketball-Reference plan remains useful
as an independent fallback or comparison source, but it is no longer the primary
path for the user's requested 1997-2013 ingest.

## Gabriel Old Data Conversion Notes

Gabriel files are team-split (`{TEAM}_{YEAR}_{rs|ps}.parquet`), so the builder
dedupes repeated team-view rows before writing season files. The source
`actionNumber` is preserved as `source_actionNumber`, but converted `event_num`
is reassigned chronologically by game because source action numbers are not
always time ordered.

Lineups come from the source home/away player snapshot columns when present. The
converted raw files include `allow_partial_lineups=True`; the processors preserve
known slots, fill unknown slots as `0`, keep rows with at least four known
offensive and defensive players, and the solver excludes player id `0` from the
coefficient universe.

For Gabriel historical `RAPM` and `TS` processing, missing shooter FT% falls
back to the observed FT result rather than the generic `0.75` expectation. This
is intentionally wired through `build_gabriel_old_pbp.py` only, so the normal
daily pipeline keeps its default expected-FT behavior unless explicitly changed.
The `1997-2000` RS/PS `RAPM` and `TS` parquets were reprocessed with this
fallback on 2026-05-01.

## Historical Core Weighted Factors

The isolated historical runner builds the regular-season 1997-2026 core
weighted-factor file without invoking the daily master runner:

```bash
python nba_pipeline/scripts/run_historical_core_weighted_factors.py 97 26 RS --cores 8
```

This solves `RAPM`, `TS`, `TOV`, `REB`, `BADPASS_TOV`, and `SCORING_TOV` with
season fixed effects, then runs the second-stage weighted-factor regressions.
Outputs from the 2026-05-01 run live in:

- `nba_pipeline/results/historical_core_97_26_rs_se/`
- `nba_pipeline/results/weighted_factors_core_97_26_rs_se.csv`
- `nba_pipeline/master_results/weighted_factors_core_97_26_rs_se.csv`

The run completed with `2,882` players in the final joined table. Core
second-stage fit quality was high: net R^2 `0.983928`, offense R^2 `0.975755`,
defense R^2 `0.968840`; TOV child decomposition R^2 was `0.999909` offense and
`0.999889` defense.

## Prior Fallback Conclusion

Before the Gabriel archive was adopted, the viable fallback path was not a
single-source story:

1. `2002-03` through `2013-14`:
   Use SportsDataverse / `hoopR` bulk ESPN season files for play-by-play and player
   box scores.
2. `2001-02`:
   Use Basketball-Reference as a one-season fallback because the bulk ESPN release
   for `2002` is only partial.

This means we should target `2001-02` through `2013-14`, but treat `2001-02`
as a special loader instead of pretending the same source covers everything.

## Source Audit

### 1. SportsDataverse / hoopR ESPN season releases

Primary docs:

- [load_nba_pbp](https://hoopr.sportsdataverse.org/reference/load_nba_pbp.html)
- [load_nba_player_box](https://hoopr.sportsdataverse.org/reference/load_nba_player_box.html)

What the docs say:

- `load_nba_pbp(seasons)` minimum season is `2002`
- `load_nba_player_box(seasons)` minimum season is `2002`

What was verified locally:

- `load_nba_pbp(2002)` returned `222,565` rows across `528` games, with dates
  from `2002-02-14` to `2002-06-12`
- `load_nba_pbp(2003)` returned `461,933` rows across `1052` games, with dates
  from `2002-11-01` to `2003-06-15`
- `load_nba_pbp(2014)` returned `591,570` rows across `1315` games, with dates
  from `2013-10-29` to `2014-06-15`
- `load_nba_pbp(2002)` includes substitution rows (`type_id = 584`, `type_text = "Substitution"`)
- `espn_nba_player_box(game_id)` includes a `starter` flag, `athlete_display_name`,
  `team_abbreviation`, `home_away`, and minutes

Operational implication:

- This is a strong primary source for `2002-03` through `2013-14`
- It is not enough for the full `2001-02` season because the `2002` release is partial

### 2. Official NBA `stats.nba.com` V2 / V3 endpoints

What was verified locally:

- direct `stats.nba.com/stats/playbyplayv2` requests timed out with zero bytes returned
- this was true even with browser-like headers

Operational implication:

- do not build the historical path around `playbyplayv2`
- even if the endpoint still exists somewhere, it is not reliable enough from this
  environment to be the foundation

### 3. Official NBA `data.nba.com` historical JSON

Candidate surface:

- `https://data.nba.com/data/v2015/json/mobile_teams/nba/{season}/scores/pbp/{game_id}_full_pbp.json`

What was verified locally:

- `hoopR::nba_data_pbp(game_id)` targets this exact family
- local requests failed with HTTP/2 transport errors or fell through to "no data"

Operational implication:

- this is not a dependable primary implementation path from this machine right now
- if we later get a stable client for it, it is worth revisiting as an official fallback

### 4. Basketball-Reference

Verified example pages:

- [2002 Finals Game 4 PBP](https://www.basketball-reference.com/boxscores/pbp/200206120NJN.html)
- [2002 Finals Game 4 box score](https://www.basketball-reference.com/boxscores/200206120NJN.html)

What was verified locally:

- the play-by-play page loads cleanly and contains a parseable HTML table with
  time, text, and score state
- the box score page contains parseable team tables such as `box-LAL-game-basic`
  and `box-NJN-game-basic`
- the first five player rows in each team box are the starters for that game

Operational implication:

- Basketball-Reference is a workable fallback for `2001-02`
- use it only where ESPN/SportsDataverse coverage is missing, because it requires
  more custom parsing and name-to-ID reconciliation

## Recommended Build Strategy

Current repo prototype:

- `nba_pipeline/scripts/01_fetch_historical_pbp_espn.py`

This script now implements the ESPN / SportsDataverse historical raw-parquet path
for seasons where the bulk releases exist.

### Phase 1: 2002-03 through 2013-14 via SportsDataverse

Inputs:

- `play_by_play_{season}.rds`
- `player_box_{season}.rds`

Use `load_nba_pbp()` and `load_nba_player_box()` or download the underlying
release files directly.

Why this is the best primary path:

- bulk season files already exist
- event order, score state, game IDs, team IDs, and player names are already structured
- player box data gives starters without per-game HTML scraping
- substitution rows exist in the play-by-play feed

### Phase 2: 2001-02 via Basketball-Reference

Inputs:

- season schedule/result pages for 2001-02
- per-game box score pages
- per-game play-by-play pages

Why this is the best fallback:

- the ESPN `2002` season release starts on `2002-02-14`
- BRef exposes both the game box score and full play-by-play pages for the missing season segment

### Phase 3: Normalize both sources into the repo's current raw contract

Target output should remain a V2-style raw parquet with at least:

- `game_id`
- `event_num`
- `event_type`
- `event_action_type`
- `period`
- `time_quarter`
- `time_remaining`
- `home_description`
- `visitor_description`
- `neutral_description`
- `home_score`
- `away_score`
- `score_margin`
- `player1_id`, `player1_name`
- `player2_id`, `player2_name`
- `player3_id`, `player3_name`
- `home_player1..5`
- `away_player1..5`

This keeps `02_process_rapm.py` and the `process_rapm_blocks` logic as close as
possible to the current path.

## Lineup Reconstruction Plan

### ESPN / SportsDataverse seasons

Per game:

1. seed each team's on-court five from `load_nba_player_box(..., starter == TRUE)`
2. order events by `game_id`, then `game_play_number` or `sequence_number`
3. on each substitution row, parse "X enters the game for Y"
4. update the active five for the correct team
5. stamp the current `home_player1..5` and `away_player1..5` onto every event row

This is the key reason the ESPN source is usable. Without the `starter` flag and
explicit substitution events, it would not be good enough for RAPM.

### Basketball-Reference 2001-02 season

Per game:

1. parse starters from the two basic box score tables
2. order play-by-play rows exactly as shown on the page
3. parse substitution text such as "T. MacCulloch enters the game for J. Collins"
4. update the active five for the correct side
5. stamp `home_player1..5` / `away_player1..5`

## Player ID Strategy

Do not keep the historical side in raw ESPN athlete IDs or BRef slugs.

Reason:

- `rapm.py` only needs stable player keys, but multi-year windows become useless if
  a player changes identifier families when we cross from the pre-2014 historical
  path into the current NBA-ID era

Recommended normalization:

1. map player names to `autocomplete_map.csv` `nba_id` values whenever possible
2. use season/team context to resolve ambiguous names
3. maintain a small manual override file for true collisions

This looks practical:

- `autocomplete_map.csv` currently has `2850` rows, `2837` unique player names,
  and only `13` duplicated names

That duplicate count is small enough that a curated override table is realistic.

## Event-Type Normalization

The current processors rely on current-style `event_type` strings and side-specific
descriptions. Historical normalization needs a shim layer.

At minimum, the shim must:

- map ESPN or BRef event categories into current-style values like `MAKE`, `MISS`,
  `FreeThrow`, `Turnover`, `Rebound`, `Foul`, `Substitution`, `Timeout`,
  `JumpBall`, `StartOfPeriod`, `EndOfPeriod`
- split unified event text into `home_description`, `visitor_description`,
  or `neutral_description`
- preserve score state and event order
- carry the reconstructed lineups on every row

This is the main implementation task. Once the historical rows look like current
raw parquet, the existing possession logic becomes reusable.

## Recommended Implementation Order

1. build an ESPN historical loader for `2003` first
2. normalize one season into `NBA03.parquet`-style output
3. run `02_process_rapm.py` on that season and fix schema gaps until it works
4. backfill `2004` through `2014`
5. build a BRef-only `2002` loader for the full `2001-02` season
6. run a cross-season name/ID reconciliation pass

This order is deliberate:

- `2003+` is the clean bulk path
- `2002` is the annoying exception
- prove the common case before spending time on the hardest season

## What I Would Not Do

- I would not wait on `playbyplayv2` to start working again
- I would not build the whole historical stack around `data.nba.com` until we have
  a stable client path from this machine
- I would not store pre-2014 data in ESPN athlete IDs and hope to reconcile later
- I would not scrape all pre-2014 seasons from Basketball-Reference if `2003+`
  can come from bulk ESPN files

## Bottom Line

Yes, we should be able to get `2001+` into a similar enough raw format for RAPM.

The practical plan is:

- `2002-03` through `2013-14`: SportsDataverse / ESPN bulk season files
- `2001-02`: Basketball-Reference fallback
- unify both into current V2-style raw parquet with reconstructed lineups and
  normalized player IDs

That is the path with the least risk and the least custom scraping.
