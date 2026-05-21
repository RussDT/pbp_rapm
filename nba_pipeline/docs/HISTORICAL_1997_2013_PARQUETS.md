# 1997-2013 Historical Parquets

This document is the canonical operating note for the pre-2014 Gabriel `old_data` ingest. The implemented source for 1997-2013 is Gabriel's `merged_playbyplay/old_data` archive; ignore the other older local experiments unless a future task explicitly revives them.

## Current State

The 1997-2013 regular-season and playoff data has been converted into this repo's standard raw and processed parquet layout.

- Source repo: `https://github.com/gabriel1200/merged_playbyplay/tree/master/old_data`
- Local source clone/cache: `nba_pipeline/external/merged_playbyplay`
- Builder: `nba_pipeline/scripts/build_gabriel_old_pbp.py`
- Build report: `nba_pipeline/validation/gabriel_old_pbp/build_1997_2013_report.json`
- FT fallback reprocess report: `nba_pipeline/validation/gabriel_old_pbp/reprocess_1997_2000_actual_missing_ft_report.json`

The completed core build produced 34 raw season-type files and 136 core processed metric parquets, with no skipped metric entries and no game-side inference failures in the report.

On 2026-05-18 the regular-season production files were refreshed from Gabriel's
updated `old_data` source and rebuilt through the full non-ShotQuality parquet
dependency set. The validation run wrote:

```text
nba_pipeline/validation/historical_full_parquet_rebuild_20260518_100028/build_1997_2013_all_metrics_report.json
```

That run produced all 306 expected regular-season processed metric entries
(`17` seasons x `18` metrics), with no RS skips. The same run checked for
playoff source files under the RS validation source directory and skipped all
17 PS raw entries with `reason = no_source_files`; do not treat the PS parquets
as refreshed by that validation run.

On 2026-05-18 the playoff production files were repaired from the existing
historical PS raw files, then rebuilt through the same 18-metric
non-ShotQuality dependency set. Because the refreshed Gabriel validation source
directory only contained RS source files, the PS repair uses
`repair_existing_historical_lineup_sides.py` against `NBA97_PS.parquet` through
`NBA13_PS.parquet` rather than a fresh Gabriel PS source checkout.

The regular-season factor rebuild covers every 1997-2013 non-ShotQuality metric used by the modern factor surfaces:

```text
RAPM
LA_RAPM
TOV
REB
TS
RIM_FREQ
RIM_FG_PCT
ASSIST_POINTS
RIM_ASSIST
THREE_FREQ
THREE_FG_PCT
MIDRANGE_FREQ
MIDRANGE_FG_PCT
FIRST_CHANCE
FIRST_CHANCE_CLEAN
FT_PREMIUM
SECOND_CHANCE
SECOND_CHANCE_CLEAN
```

The alternate clean 3-factor flow is covered by the same physical parquets:
`FIRST_CHANCE` and `SECOND_CHANCE_CLEAN`. `ALT_TS`, `ALT_EFG`, `ALT_FT`,
`ALT_TOV`, `ALT_BADPASS_TOV`, and `ALT_SCORING_TOV` are solver aliases derived
from `FIRST_CHANCE`, not separate physical `ALT_*` parquet files.

The refreshed audit file is:

```text
nba_pipeline/results/processed_rs_parquet_averages_per100_97_26.csv
```

Regenerate the full regular-season processed parquet audit CSV family with:

```bash
python nba_pipeline/scripts/build_processed_rs_parquet_averages.py
```

## File Locations

Raw regular-season files:

```text
nba_pipeline/raw_data/NBA97.parquet
...
nba_pipeline/raw_data/NBA13.parquet
```

Raw playoff files:

```text
nba_pipeline/raw_data/NBA97_PS.parquet
...
nba_pipeline/raw_data/NBA13_PS.parquet
```

Processed regular-season files:

```text
nba_pipeline/processed/RAPM97.parquet
nba_pipeline/processed/LA_RAPM97.parquet
nba_pipeline/processed/TOV97.parquet
nba_pipeline/processed/REB97.parquet
nba_pipeline/processed/TS97.parquet
nba_pipeline/processed/ASSIST_POINTS97.parquet
nba_pipeline/processed/FIRST_CHANCE97.parquet
nba_pipeline/processed/FIRST_CHANCE_CLEAN97.parquet
nba_pipeline/processed/FT_PREMIUM97.parquet
nba_pipeline/processed/MIDRANGE_FG_PCT97.parquet
nba_pipeline/processed/MIDRANGE_FREQ97.parquet
nba_pipeline/processed/RIM_ASSIST97.parquet
nba_pipeline/processed/RIM_FG_PCT97.parquet
nba_pipeline/processed/RIM_FREQ97.parquet
nba_pipeline/processed/SECOND_CHANCE97.parquet
nba_pipeline/processed/SECOND_CHANCE_CLEAN97.parquet
nba_pipeline/processed/THREE_FG_PCT97.parquet
nba_pipeline/processed/THREE_FREQ97.parquet
...
nba_pipeline/processed/RAPM13.parquet
nba_pipeline/processed/LA_RAPM13.parquet
nba_pipeline/processed/TOV13.parquet
nba_pipeline/processed/REB13.parquet
nba_pipeline/processed/TS13.parquet
nba_pipeline/processed/ASSIST_POINTS13.parquet
nba_pipeline/processed/FIRST_CHANCE13.parquet
nba_pipeline/processed/FIRST_CHANCE_CLEAN13.parquet
nba_pipeline/processed/FT_PREMIUM13.parquet
nba_pipeline/processed/MIDRANGE_FG_PCT13.parquet
nba_pipeline/processed/MIDRANGE_FREQ13.parquet
nba_pipeline/processed/RIM_ASSIST13.parquet
nba_pipeline/processed/RIM_FG_PCT13.parquet
nba_pipeline/processed/RIM_FREQ13.parquet
nba_pipeline/processed/SECOND_CHANCE13.parquet
nba_pipeline/processed/SECOND_CHANCE_CLEAN13.parquet
nba_pipeline/processed/THREE_FG_PCT13.parquet
nba_pipeline/processed/THREE_FREQ13.parquet
```

Processed playoff files use the same metric prefixes with `_PS`, for example:

```text
nba_pipeline/processed/RAPM97_PS.parquet
nba_pipeline/processed/TOV97_PS.parquet
nba_pipeline/processed/REB97_PS.parquet
nba_pipeline/processed/TS97_PS.parquet
```

## How To Rebuild

Full 1997-2013 RS/PS rebuild:

```bash
python nba_pipeline/scripts/build_gabriel_old_pbp.py \
  --years 1997-2013 \
  --season-types all \
  --metrics RAPM,TOV,REB,TS \
  --report nba_pipeline/validation/gabriel_old_pbp/build_1997_2013_report.json
```

Full 1997-2013 regular-season dependency rebuild from an already-refreshed
Gabriel source directory:

```bash
python nba_pipeline/scripts/build_gabriel_old_pbp.py \
  --years 1997-2013 \
  --season-types all \
  --skip-download \
  --source-dir nba_pipeline/validation/historical_eop_flow_1997_2013_rs/merged_playbyplay_updated \
  --raw-dir nba_pipeline/raw_data \
  --processed-dir nba_pipeline/processed \
  --metrics RAPM,LA_RAPM,TOV,REB,TS,RIM_FREQ,RIM_FG_PCT,ASSIST_POINTS,RIM_ASSIST,THREE_FREQ,THREE_FG_PCT,MIDRANGE_FREQ,MIDRANGE_FG_PCT,FT_PREMIUM,SECOND_CHANCE,FIRST_CHANCE,FIRST_CHANCE_CLEAN,SECOND_CHANCE_CLEAN \
  --workers 3 \
  --report nba_pipeline/validation/historical_full_parquet_rebuild_YYYYMMDD_HHMMSS/build_1997_2013_all_metrics_report.json
```

If the source directory only contains RS files, `--season-types all` is still
safe but the report will mark PS raw entries as skipped with `no_source_files`.

Existing playoff raw files can be side-repaired and rebuilt in isolated
validation directories with:

```bash
python nba_pipeline/scripts/repair_existing_historical_lineup_sides.py \
  --years 1997-2013 \
  --season-types ps \
  --raw-dir nba_pipeline/raw_data \
  --output-dir nba_pipeline/validation/historical_ps_lineup_repair_1997_2013/raw \
  --report nba_pipeline/validation/historical_ps_lineup_repair_1997_2013/raw_repair_report.json

python nba_pipeline/scripts/reprocess_metric.py RAPM 1997 2013 \
  --season-types ps \
  --workers 4 \
  --raw-dir nba_pipeline/validation/historical_ps_lineup_repair_1997_2013/raw \
  --processed-dir nba_pipeline/validation/historical_ps_lineup_repair_1997_2013/processed
```

Repeat the `reprocess_metric.py` command for the full metric list before
production promotion. `--raw-dir` / `--processed-dir` are intended for this
validation-first workflow. Plain `RAPM` uses the no-luck, actual-missing-FT
registry entry; use `LA_RAPM` for luck-adjusted output.

If the source clone already exists and you only need to regenerate local parquets, avoid redownloading:

```bash
python nba_pipeline/scripts/build_gabriel_old_pbp.py \
  --years 1997-2013 \
  --season-types all \
  --skip-download \
  --metrics RAPM,TOV,REB,TS \
  --report nba_pipeline/validation/gabriel_old_pbp/build_1997_2013_report.json
```

1997-2000 `RAPM` / `TS` FT fallback reprocess:

```bash
python nba_pipeline/scripts/build_gabriel_old_pbp.py \
  --years 1997-2000 \
  --season-types all \
  --skip-download \
  --metrics RAPM,TS \
  --report nba_pipeline/validation/gabriel_old_pbp/reprocess_1997_2000_actual_missing_ft_report.json
```

Targeted regular-season non-ShotQuality factor rebuild:

```bash
python nba_pipeline/scripts/reprocess_metric.py FIRST_CHANCE 1997 2013 --season-types rs --workers 6
python nba_pipeline/scripts/reprocess_metric.py ASSIST_POINTS 1997 2013 --season-types rs --workers 6
python nba_pipeline/scripts/reprocess_metric.py LA_RAPM 1997 2013 --season-types rs --workers 6
```

`reprocess_metric.py` accepts full years for the Gabriel range and writes normal
two-digit processed filenames, such as `FIRST_CHANCE97.parquet` through
`FIRST_CHANCE13.parquet`.

After targeted parquet rebuilds, refresh the processed parquet average audit
files:

```bash
python nba_pipeline/scripts/build_processed_rs_parquet_averages.py
```

## Conversion Behavior

Gabriel's archive is team-file based. The builder stitches team views into season files, dedupes duplicated team-view rows, infers game sides, and writes repo-style raw season parquets.

Important conversion details:

- Raw season files are written into `nba_pipeline/raw_data/` using the normal `NBA{YY}.parquet` and `NBA{YY}_PS.parquet` naming.
- Missing source `team` abbreviations are repaired from stable `teamId` mappings before descriptions are routed into `home_description` / `visitor_description`. This prevents valid actions from being treated as neutral rows when one team-file view has `team = None`.
- Converted rows keep Gabriel's source action number where available, but repo-facing event order is normalized for processing.
- Lineups are reconstructed from the source home/away player snapshots where available.
- Unknown lineup slots are represented as player id `0`.
- Converted rows intentionally mark partial-lineup mode, because the historical source can know four players on one side and five on the other.
- Use `nba_pipeline/scripts/audit_lineup_side_conflicts.py` to check converted raw files for impossible lineup assignments. The audit expands each row to ten lineup-slot records, then flags (a) row-level cases where the same `player_id` appears on both home and away and (b) game/player IDs that are overwhelmingly on one side but appear on the opposite side in a small number of rows.

The processor keeps partial-lineup rows when each side has at least four known players. Those rows receive full weight in the RAPM surface; missing slots remain `0`.

The solver must exclude player id `0` from the coefficient universe. `nba_pipeline/scripts/rapm.py` already does this for player coefficients and home/away player sets.

## Supplemental NBA Archive

A separate local NBA stats archive exists at:

```text
/Users/russellthomas/Docs/2026_NBA_PIPELINE/archive (1)
```

It is useful as a validation and repair source for historical starter/lineup
issues, but only some files contain player-level evidence:

- `PlayByPlay.parquet` covers roughly 1996-11-01 through 2026-04-19, has
  event rows and substitution descriptions, and includes `gameId`, `actionType`,
  `description`, `personId`, `playerFullName`, and `teamTricode`. It does not
  contain lineup snapshots or direct starter flags. In 2008-09, substitution
  rows use text like `SUB: Weaver FOR Mason`; `personId` is the outgoing player
  and the incoming player must be parsed from the description and resolved
  against game/team player metadata.
- `PlayerStatistics.csv` covers 1946-11-26 through 2026-05-06 and includes
  player-game rows, DNP rows, `numMinutes`, `comment`, `personId`, `gameId`,
  `playerteamId`, and `opponentteamId`. Use it for active/DNP validation and
  player-team-game membership.
- `PlayerStatisticsExtended.csv` covers 1996-11-01 through 2026-05-06 and has
  similar player-game IDs plus advanced box-score fields. It usually omits DNP
  rows and should not be treated as the complete roster source.
- `Players.csv` maps `personId` to name, active years, and coarse position.
- `TeamHistories.csv` maps franchise eras. Trim padded `teamAbbrev` values and
  resolve by `(teamId, season)`, not by city alone; for example
  `teamId = 1610612760` is `SEA` through 2007 and `OKC` from 2008 onward.
- `Games.csv`, `TeamStatistics.csv`, `TeamStatisticsExtended.csv`, and
  `LeagueSchedule*.csv` are useful for game/team/date/score context, but they
  do not provide starter or lineup evidence.

## Evidence-Driven Lineup Repair Prototype

The current validation-first historical lineup repair harness is:

```bash
python nba_pipeline/scripts/repair_historical_lineups_evidence.py --year 2009 --disable-player-stats
```

By default it writes isolated validation artifacts under:

```text
nba_pipeline/validation/historical_lineup_evidence_repair_2009/
```

The repair uses the supplemental archive event stream as hard on-court evidence:

- game-local player/team membership from `PlayerStatistics.csv`
- compact substitution rows in `PlayByPlay.parquet`, where `personId` is the outgoing player and the incoming player is parsed from `SUB: ... FOR ...`
- first-substitution clusters to repair bad opening lineups, using the last plausible pre-cluster lineup and the first complete post-cluster lineup rather than intermediate sub-row snapshots
- event actors, assist/block/steal/foul-drawn/jump-ball participants as mandatory on-court players for their game-local side

Do not promote these artifacts blindly. The 2009 validation run eliminated the
structural overlap corruption but did not yet fully match PBPStats/WOWY
possession counts:

```text
Raw same-player home/away rows: 2051 -> 0
Processed same-player O/D rows: 58 -> 0
Processed rows with unknown lineup slots: 2885 -> 33
Kyle Weaver, all-actual-FT repaired RAPM: 2228 OffPoss, 2219 DefPoss, 2378 PF, 2424 PA, -2.51 net
Kyle Weaver, Supabase wowy_stats_two OKC 2009 RS aggregate: 2302 OffPoss, 2304 DefPoss, 2418 PF, 2460 PA, -1.73 net
```

Use `--disable-player-stats` when comparing to PBPStats/WOWY actual scoring.
Without it, the processor fetches Supabase FT% for known shooters and only uses
actual FT makes/misses for missing FT% rows, so the output is not an
apples-to-apples actual-scoring on/off comparison.

## 2009 Standard EOP Flow Validation

The shared standard possession path now owns RAPM EOP/offense assignment. For
Gabriel historical rows it:

- assigns direct shot, rebound, turnover, foul, and FT offense markers before
  player propagation
- matches neutral team turnovers from `neutral_description` using game-local
  side labels plus NBA abbreviation/nickname aliases
- runs `propagate_player_values_py` and `ft_off_check_py`
- fills blank terminal lineups from the prior in-game row
- creates `poss_group` from shifted EOP, fills `poss_offense` inside the
  possession, and maps O/D players from `poss_offense`
- infers one-row terminal possessions with no direct text marker as the
  opposite side of the previous completed possession in that game/period

The 2009 validation was run from a freshly fetched Gabriel
`merged_playbyplay/old_data` source into isolated directories only:

```text
nba_pipeline/validation/historical_eop_flow_2009/raw/NBA09.parquet
nba_pipeline/validation/historical_eop_flow_2009/processed/RAPM09.parquet
nba_pipeline/validation/historical_eop_flow_2009/processed/RAPM09_prefinal_eop_debug.parquet
nba_pipeline/validation/historical_eop_flow_2009/validation_summary.csv
nba_pipeline/validation/historical_eop_flow_2009/validation_meta.json
```

The first 2009 validation did not promote production parquets. It used
all-actual FT mode for PBPStats/WOWY comparison. Current 2009 RS results:

```text
League final RAPM: 226,180 possessions, 245,894 points
League WOWY: 227,074 off / 227,059 def possessions, 245,686 PF / 245,631 PA
Final dropped blank-lineup EOP rows: 0
Final same-player O/D overlap rows: 0
Final incomplete-lineup rows: 0

OKC final RAPM: 7,743 off / 7,760 def possessions, 7,950 PF / 8,454 PA, -6.27 net
OKC WOWY: 7,758 off / 7,766 def possessions, 7,952 PF / 8,452 PA, -6.33 net

Kyle Weaver final RAPM: 2,296 off / 2,294 def possessions, 2,417 PF / 2,461 PA, -2.01 net
Kyle Weaver WOWY: 2,302 off / 2,304 def possessions, 2,418 PF / 2,460 PA, -1.73 net
```

The remaining league possession gap is no longer from finalization dropping
blank EOP rows. Treat the next audit as a source-definition / raw-event coverage
comparison before promoting 2009 or generalizing the rebuild to 1997-2013.

For full-range validation against already-built historical raw parquets, use:

```bash
SUPABASE_URL= SUPABASE_SERVICE_ROLE_KEY= SUPABASE_KEY= \
PYTHONPATH=nba_pipeline/scripts:. \
python nba_pipeline/scripts/validate_historical_eop_flow.py \
  --years 1997-2013 \
  --raw-dir nba_pipeline/raw_data \
  --processed-dir nba_pipeline/validation/historical_eop_flow_1997_2013_rs/processed_existing_raw \
  --report-dir nba_pipeline/validation/historical_eop_flow_1997_2013_rs \
  --workers 3 \
  --write-prefinal
```

The validation harness disables player stats by default so `missing_ft_fallback=actual`
is a true all-actual-FT comparison, including 1997-2000 where the normal loader
can otherwise find the local BRef FT% lookup. The May 17, 2026 all-actual run
over existing raw parquets produced:

```text
1997-2013 RS final RAPM rows: 3,642,966
Pre-final EOP rows: 3,642,972
Final dropped EOP rows: 6
Final dropped points: 0
Final incomplete-lineup rows: 0
Final same-player O/D overlap rows: 2,752
Largest overlap seasons: 2009 (965), 2002 (315), 2008 (194), 2004 (162), 2013 (160)
```

Interpretation: the shared EOP/offense fix generalizes across the historical
regular-season files, but existing production raw parquets still contain stale
lineup-side overlap artifacts. Do not promote a full 1997-2013 rebuild from the
existing raw files as final; promotion should use a refreshed Gabriel raw rebuild
with the source/converter side-sanitization path, then rerun this validation.

## 1997-2013 RS Production EOP/Lineup Promotion

The full regular-season historical RAPM pass was rebuilt from refreshed Gabriel
old-data raw files with converter side guardrails, duplicate-slot dedupe, and
the shared standard EOP/offense flow. The production files promoted from the
validated staging area are:

```text
nba_pipeline/raw_data/NBA97.parquet ... NBA13.parquet
nba_pipeline/processed/RAPM97.parquet ... RAPM13.parquet
```

The previous production raw/RAPM files were backed up before promotion:

```text
nba_pipeline/validation/historical_eop_flow_1997_2013_rs/production_backup_20260518_021105
```

All-actual-FT validation for the refreshed build:

```text
Report: nba_pipeline/validation/historical_eop_flow_1997_2013_rs/dedup_common_validation/historical_eop_flow_summary.csv
Final RAPM rows: 3,642,803
Pre-final EOP rows: 3,642,938
Final removed rows: 135
Final removed points: 102.0
Final same-player O/D overlap rows: 0
Final incomplete-lineup rows: 0
Final points: 3,857,710.0
```

Production verification after promotion:

```text
Production RAPM rows: 3,642,803
Production O/D overlap rows: 0
Production final incomplete rows: 0
Production same-side duplicate slots: 0
Production points_abs total: 3,857,682.397
```

The production points total differs slightly from the all-actual validation
because production processing still uses the normal historical expected-FT path
where the 1997-2000 local BRef FT% lookup is available.

2009 all-actual validation after the full rebuild:

```text
Final rows: 226,179
Pre-final rows: 226,180
Removed rows: 1
Removed points: 0
O/D overlap rows: 0
Incomplete-lineup rows: 0
Final points: 245,894.0
```

Promoted production Kyle Weaver line from `nba_pipeline/processed/RAPM09.parquet`:

```text
Player id: 201602
OffPoss / DefPoss: 2,296 / 2,294
PF / PA: 2,398.627 / 2,470.253
ORtg / DRtg / Net: 104.47 / 107.68 / -3.21
```

The later full dependency rebuild overwrote `RAPM09` through the normal
historical expected-FT production path. The all-actual validation line above is
still the right WOWY/PBPStats comparison; the production line is the one to use
for downstream RAPM solves.

## 1997-2013 RS Full Metric Rebuild

The 2026-05-18 full RS dependency rebuild wrote all requested regular-season
metric parquets from the refreshed Gabriel source and then refreshed the
processed RS average audit files.

```text
Backup directory: nba_pipeline/validation/historical_full_parquet_rebuild_20260518_100028
Report: nba_pipeline/validation/historical_full_parquet_rebuild_20260518_100028/build_1997_2013_all_metrics_report.json
Processed RS entries: 306 / 306
Processed RS skipped entries: 0
Raw RS skipped entries: 0
Raw PS skipped entries: 17, reason = no_source_files
RAPM97-13 rows: 3,642,803
RAPM O/D overlap rows: 0
RAPM incomplete side rows: 0
RAPM same-side duplicate rows: 0
FIRST_CHANCE FT split max row gap: 7.105427357601002e-15
FIRST_CHANCE EFG-value split max row gap: 1.3766765505351941e-14
```

The rebuilt regular-season metrics are:

```text
RAPM, LA_RAPM, TOV, REB, TS, RIM_FREQ, RIM_FG_PCT, ASSIST_POINTS,
RIM_ASSIST, THREE_FREQ, THREE_FG_PCT, MIDRANGE_FREQ, MIDRANGE_FG_PCT,
FT_PREMIUM, SECOND_CHANCE, FIRST_CHANCE, FIRST_CHANCE_CLEAN,
SECOND_CHANCE_CLEAN
```

This rebuild covers the historical RS raw files, core RAPM files, and dependent
factor parquets. Weighted-factor solve outputs still need to be rerun from
these refreshed parquets before downstream alt3 bundles should be considered
fully refreshed.

## 1997-2013 PS Lineup-Side Repair and Full Metric Rebuild

The 2026-05-18 playoff pass repaired the existing historical PS raw files with
game-local lineup-side guardrails and rebuilt the full 18-metric dependency set
into production. This fixed the known playoff same-player O/D overlap issue and
filled the missing playoff factor parquets.

```text
Raw repair report: nba_pipeline/validation/historical_ps_lineup_repair_1997_2013/raw_repair_report.json
Processed validation summary: nba_pipeline/validation/historical_ps_lineup_repair_1997_2013/processed_ps_validation_summary.json
Processed validation CSV: nba_pipeline/validation/historical_ps_lineup_repair_1997_2013/processed_ps_validation_summary.csv
Production backup: nba_pipeline/validation/historical_ps_lineup_repair_1997_2013/production_backup_20260518_175307
Processed PS entries: 306 / 306
Processed PS skipped entries: 0
Processed PS zero-row files: 0
Processed PS O/D overlap cell hits: 0
RAPM97-13_PS rows: 239,952
RAPM PS incomplete side rows: 2,721
FIRST_CHANCE FT split max row gap: 7.105427357601002e-15
FIRST_CHANCE EFG-value split max row gap: 9.769962616701378e-15
```

Before this promotion, production had all 17 `RAPM*_PS.parquet` files but only
140 of the 306 expected playoff metric parquets. Production playoff RAPM also
had 1,723 same-player O/D overlap cell hits and 5,914 incomplete side rows. The
promoted rebuild has zero O/D overlaps across every processed playoff metric
file and reduces RAPM incomplete side rows to 2,721. Remaining incomplete rows
are unknown-slot historical partial lineups (`0`) and should be handled by the
solver's existing player-id-0 exclusion.

The rebuilt playoff metrics are:

```text
RAPM, LA_RAPM, TOV, REB, TS, RIM_FREQ, RIM_FG_PCT, ASSIST_POINTS,
RIM_ASSIST, THREE_FREQ, THREE_FG_PCT, MIDRANGE_FREQ, MIDRANGE_FG_PCT,
FT_PREMIUM, SECOND_CHANCE, FIRST_CHANCE, FIRST_CHANCE_CLEAN,
SECOND_CHANCE_CLEAN
```

This rebuild covers the historical PS raw files, core RAPM files, and dependent
factor parquets. Weighted-factor solve outputs still need to be rerun from the
refreshed RS and PS parquet surfaces before downstream historical all-season
bundles should be considered fully refreshed.

Do not trust `startingPosition` in the player-stat files as an old-game starter
flag. In 2009 it behaves like active-player position metadata: game `20800639`
has more than five nonblank `startingPosition` values per team. For 2017 onward
it is often closer to a real starter flag, but historical starter repair should
not depend on it without a per-season/team-game count audit.

For bad Gabriel opening lineups, use the supplemental archive as follows:

1. Validate game/team/player IDs with `Games.csv`, `Players.csv`,
   `TeamHistories.csv`, and `PlayerStatistics.csv`.
2. Parse `PlayByPlay.parquet` substitution descriptions in the target game.
3. Treat a first-period `SUB: InPlayer FOR OutPlayer` as evidence that
   `OutPlayer` was on the floor before the cluster and `InPlayer` was not.
4. If Gabriel's opening lineup already contains the incoming player, treat the
   opening snapshot as suspect and repair the starter set before running normal
   substitution replay or stable-neighbor lineup fill.

Concrete canary: in `20800639`, the supplemental PBP has `SUB: Weaver FOR
Mason` at Q1 `09:49` and the jump ball is `Jordan vs. Green: Tip to Collison`.
That supports Mason and Collison being on the opening OKC court and Weaver not
starting. Gabriel's opening OKC lineup for that game includes Weaver and omits
Collison, which creates the bogus `Mason / Durant / Green / Westbrook / Weaver`
lineup in the current validation surface.

## Processed Metrics

The pre-2014 Gabriel path builds these core RAPM-ready surfaces:

- `RAPM`: possession point differential surface
- `TS`: scoring-efficiency surface
- `TOV`: turnover surface
- `REB`: offensive rebounding surface

This is enough for the historical core weighted-factor build: TS, TOV, ORB, plus the TOV bad-pass split.

The 1997-2013 regular-season non-ShotQuality factor surface is now filled out
for:

- `LA_RAPM`: luck-adjusted point surface from the RAPM processor
- `ASSIST_POINTS`: assisted field-goal points per standard possession
- `RIM_ASSIST`: assisted rim makes per standard possession
- `RIM_FREQ` / `RIM_FG_PCT`
- `MIDRANGE_FREQ` / `MIDRANGE_FG_PCT`
- `THREE_FREQ` / `THREE_FG_PCT`
- `FT_PREMIUM`
- `FIRST_CHANCE`
- `SECOND_CHANCE`
- `SECOND_CHANCE_CLEAN`

The alt3 aliases are derived from `FIRST_CHANCE`:

- `ALT_TS`: same-sample non-turnover `Net_Diff` baseline, with turnover rows imputed to that baseline in the solver
- `ALT_EFG`: same-sample non-turnover `FC_EFG_Diff` baseline
- `ALT_FT`: same-sample non-turnover `FC_FT_Diff` baseline
- `ALT_TOV`: first-chance `Is_Turnover`
- `ALT_BADPASS_TOV`: first-chance `Is_BadPass_TOV`
- `ALT_SCORING_TOV`: first-chance `Is_Turnover - Is_BadPass_TOV`

ShotQuality-dependent surfaces remain unavailable from Gabriel old data:
`INITIAL_EV`, `TRANSITION_FREQ`, `TRANSITION_RIM`, `SQ_POSS`, `CONTEST`,
`SPECIAL_RAPM`, `EV_RAPM`, `INITIAL_EV_BETA`, `PLAYTYPE_TS_MIX`, and
`PLAYTYPE_PROXY_PTS`.

The 1997-2013 `TOV` parquets include `Is_BadPass_TOV`. This is derived from event description text containing `Bad Pass` in the TOV processor. The solver aliases use:

- `BADPASS_TOV`: `Is_BadPass_TOV`
- `SCORING_TOV`: `Is_Turnover - Is_BadPass_TOV`

Spot checks found nonzero bad-pass coverage across the historical range, including:

- 1997 RS: 35,699 turnover rows, 17,138 bad-pass rows
- 2013 RS: 34,279 turnover rows, 15,789 bad-pass rows

## Free-Throw Fallback

For Gabriel historical `RAPM` and `TS`, missing shooter FT percentage can use actual FT outcomes instead of the generic expectation fallback. This is intentionally scoped to the Gabriel historical build path and does not change the daily pipeline default.

The 1997-2000 RS/PS `RAPM` and `TS` parquets were reprocessed with this actual-outcome fallback because player FT% coverage was known to be incomplete there.

For 1997-2000 Basketball Reference regular-season stat CSVs, build a local
player-season FT% lookup with:

```bash
python nba_pipeline/scripts/build_bref_ft_pct_lookup.py
```

The generator reads root-level `1997.csv` through `2000.csv`, joins
`Player-additional` to Gabriel `site_Data/index_master.csv` via `bref_id`, and
writes:

```text
nba_pipeline/external/bref_ft_pct_1997_2000.csv
```

When a player has multiple Basketball Reference rows in a season, the generator
prefers the aggregate row (`TOT`, `2TM`, `3TM`, etc.) before falling back to the
largest-games team row. The output is keyed by `nba_id` and `year`; blank BRef
`FT%` values remain blank so downstream processors can apply their normal
missing-FT% fallback.

The shared player-stat loader uses this regular-season lookup as the local
fallback for both RS and PS 1997-2000 processing when Supabase stats are
missing/unavailable. This affects processors that consume `FTPerc`: `RAPM`,
`LA_RAPM`, `TS`, and `FT_PREMIUM`. It does not supply historical `ThreePerc`.

The RS and PS 1997-2000 `RAPM`, `LA_RAPM`, `TS`, and `FT_PREMIUM` parquets were
rewritten with this lookup using:

```bash
PYTHONUNBUFFERED=1 python nba_pipeline/scripts/build_gabriel_old_pbp.py \
  --years 1997-2000 \
  --season-types all \
  --skip-download \
  --metrics RAPM,LA_RAPM,TS,FT_PREMIUM \
  --report nba_pipeline/validation/gabriel_old_pbp/reprocess_1997_2000_bref_rs_ft_all_report.json
```

Relevant implementation:

- `nba_pipeline/scripts/process_rapm_blocks/common.py`
- `nba_pipeline/scripts/process_rapm_blocks/process_rapm.py`
- `nba_pipeline/scripts/process_rapm_blocks/process_ts.py`
- `nba_pipeline/scripts/build_gabriel_old_pbp.py`
- `nba_pipeline/scripts/build_bref_ft_pct_lookup.py`

## Validation

Solver smoke checks passed after the historical build:

```bash
python nba_pipeline/scripts/rapm.py RAPM 13 13 RS --pure
python nba_pipeline/scripts/rapm.py RAPM 13 13 PS --pure
python nba_pipeline/scripts/rapm.py TOV 13 13 RS
python nba_pipeline/scripts/rapm.py REB 13 13 RS
python nba_pipeline/scripts/rapm.py TS 13 13 RS
python nba_pipeline/scripts/rapm.py FIRST_CHANCE 13 13 RS --pure
python nba_pipeline/scripts/rapm.py ASSIST_POINTS 13 13 RS
python nba_pipeline/scripts/rapm.py THREE_FREQ 13 13 RS
```

The full isolated 1997-2026 regular-season core weighted-factor run completed with estimated season dummy effects:

```bash
python nba_pipeline/scripts/run_historical_core_weighted_factors.py 97 26 RS --cores 8
```

Outputs:

- `nba_pipeline/results/historical_core_97_26_rs_se/`
- `nba_pipeline/results/weighted_factors_core_97_26_rs_se.csv`
- `nba_pipeline/master_results/weighted_factors_core_97_26_rs_se.csv`
- downstream published copy: `/Users/russellthomas/Docs/rapms/master_results/weighted_factors_core_97_26_rs_se.csv`

The final weighted-factor table had 2,882 players. The second-stage fit quality was strong for a long historical single-core build:

- Net R^2: `0.983928`
- Offense R^2: `0.975755`
- Defense R^2: `0.968840`
- TOV split R^2 offense/defense: `0.999909` / `0.999889`

## Known Caveats

The historical Gabriel parquets are RAPM-usable, not a perfect modern-play-by-play reconstruction.

- Partial lineups are expected. Use the available player signal rather than dropping every row with one unknown slot.
- Unknown lineup slots are `0`, and player id `0` must stay excluded from regression coefficients.
- Pre-2014 files do not have ShotQuality enrichment. Do not expect ShotQuality-specific metrics such as `INITIAL_EV`, `TRANSITION_FREQ`, `TRANSITION_RIM`, or playtype-value surfaces from this source.
- The May 2026 RS factor backfill ran with the local Supabase key rejected by Supabase, so processors that request player FT% / 3P% used their existing default/fallback behavior. `RAPM` / `LA_RAPM` used actual FT outcomes for missing FT% rows through the Gabriel wrapper; clean-chance and FT-premium processors used their standard defaults where stats were unavailable.
- Historical raw/player names may be less complete than modern NBA API data. NBA player id is the primary key.
- Legacy processed parquets do not all share a canonical embedded `Season` encoding. For season summaries and intercept harnesses, prefer filename-derived season keys and audit parquet `Season` values separately.
- The historical core run is intentionally isolated from the daily master runner. Do not wire 1997-2013 into `03_run_rapm_analysis.py` unless the daily pipeline impact is explicitly reviewed.

## Operator Notes

Use `--skip-download` when the local Gabriel source cache already exists. A full redownload is only needed if the source clone/cache is missing or intentionally being refreshed.

For 1997-2026 core weighted factors, use the isolated runner:

```bash
python nba_pipeline/scripts/run_historical_core_weighted_factors.py 97 26 ALL \
  --rubberband \
  --fixed-season-effects \
  --age-poly-coefficients nba_pipeline/results/rapm_14_26_all_rb_agefe/age_curve_poly3_coefficients.csv \
  --cores 14
```

That path solves `RAPM`, `TS`, `TOV`, `REB`, `BADPASS_TOV`, and `SCORING_TOV`, then writes the core historical weighted-factor CSVs without changing the normal daily pipeline.

The current preferred long historical run uses fixed constants instead of estimated season and age nuisance columns:

- fixed season effects: compute each metric's regular-season raw target mean by season, subtract that season's RS value from both RS and PS rows, and save the audit table as `fixed_rs_season_baselines.csv`
- fixed age polynomial: use the smoothed `14-26` age curves, subtract `curve(age) - curve(27)` for every known offensive and defensive lineup slot before centering the target, and save per-metric `*_fixed_offsets.csv` summaries
- estimated dummy columns remain available through `--season-effects` and `--age-dummies`, but they are not used in the fixed-constant historical run

Algebraically, the fixed constants are part of the regression target:

```text
y_adjusted = y - fixed_rs_season_baseline - fixed_age_poly_offset
```

This is equivalent to including known columns with coefficient fixed at `1`, but implemented by moving the known offset to the left-hand side before the ridge solve.
