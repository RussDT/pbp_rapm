# Rim contest savings and matchup burden

## October 18, 2026 source inventory: game logs versus season pairs

### Full2018–2026 request census resolved

The authorized backfill now resolves all12,214 completed NBA-team games in
the official census.11,521 games contain2,169,558 native matchup pair rows;
the693 empty responses are22 regular-season games and671 preseason games.
No request errors remain. Raw JSON, hashes, game IDs, exact dates and normalized
pairs live privately at `cache/matchup_games_v1/`; the independent full archive
identity/count-fidelity audit passes all12,214 games with no count defects. This supersedes the pilot-only availability finding below.

The defensive feature and predictive test contract is
[VPM matchup-defense research](VPM_MATCHUP_DEFENSE_FEATURES.md). A successful
download alone does not establish reliable field measurement or predictive
value. Neutral-site official rows can share `AWAY @ HOME` text, so the new
census resolves exact abbreviation tokens rather than using the row's team ID
as its venue slot. The native outer-player attacker convention remains fixed.

### Live per-game retrieval confirmed

Owner correctly identified the next step: use `BoxScoreMatchupsV3` directly
and retain its game rows before the core BAM season aggregation. A live pilot
on October18 retrieved and cached four exact game IDs successfully:

|Game ID|Game date|Phase|Matchup rows|
|---|---|---|---:|
|0021700002|2017-10-17|RS|184|
|0022400001|2024-11-12|RS|155|
|0022500004|2025-10-22|RS|260|
|0042500121|2026-04-18|PS|146|

The745 raw pair rows preserve game ID and the date from the exact PBP game.
They include matchup time, partial possessions, field goals/three-pointers/free
throws, points, assists/potential assists, turnovers, blocks, switches and fouls.
Cache: `nba_pipeline/cache/matchup_game_probe_v1/{game_id}.json`,
`raw_player_pairs.parquet`, `pilot_manifest.json`. The pilot is a source-access
check, not a complete archive, accepted feature or model change. No2027 outcomes
were requested. Historical publication/extraction vintages remain uncertified.

**Use the native response hierarchy and validate direction.** In all four live
responses, the outer team/player is the attacker, and that player's nested
`matchups` list names opposing defenders. Grouping points/shots by the outer
player aligns much better with actual PBP box totals than grouping by the
nested opponent. The installed nba_api flattening labels the outer player
`personIdOff` and the opponent `personIdDef`. The legacy core BAM code reverses
those IDs; do not copy that inversion into a new game-log adapter. This pilot
does not silently repair any existing BAM or VPM outputs.

Coverage/definitions vary by era. The2024/2025/2026 examples omit1–2 points
and1–6 FGA relative to observed PBP. The2017 game differs materially: tracked
points211 versus actual243, FGA184 versus177, and FT attempts0 versus40.
Thus old FT fields and assignment totals cannot be assumed complete, exclusive
or equivalent across eras. Cache raw values, retain field-specific coverage
and validate before building dated player-level candidate skills. Do not use
this pilot to assert that every game from2017 onward is available.

The failed dated checkpoint described below is a separate source failure;
it does not block fetching proper game-ID-specific responses.

The local VPM BAM input remains annual: `vpm/vpm/data/matchup/matchups_two_rs_2018_2026.parquet`
in the sibling repo has722,575 rows, keyed by season/offensive player/defensive
player, with no game ID or game date. Its matchup minutes, partial possessions,
shooting, assists, turnovers, blocks and fouls are season aggregates.2026 remains
an incomplete archived snapshot with no verified extraction date.

A separate local shot-defender archive exists in the NBA workspace at
`data-sources/matchup-fga/defender/{2021,2022,2023,2024,2025}/*.csv`.
There are537/594/534/569/568 player files by ending-year folder. Sampled files
contain `GAME_ID`, `GAME_EVENT_ID`, `GAME_DATE`, shooter `PLAYER_ID`, `DEF_ID`,
shot coordinates/type and make/miss. These can support game-level defended-shot
summaries after validation, but do not contain full matchup assignment minutes
or partial-possession exposure. Sampled files in every folder contain pipe-joined
multiple defender IDs, so uniqueness of closest-defender attribution must never
be assumed. Folder/year/sample schema is verified; whole-archive coverage and
attribution are not newly certified. Preserve the existing NBA-versus-SQ
provider distinction, and do not import these raw event rows directly into VPM.

### Recovered dated checkpoint: invalid game-log grain

Owner's follow-up led to recovering tracked files absent from the working tree
of `nba/data-sources/matchup-fga`. Source repository HEAD at recovery:
`8f68216687b7432df883257bbfde10029e19043d`. The Git index contains:

- `.ipynb_checkpoints/matchups_by_date_defense-checkpoint.csv`, blob
  `73f04b9f7c0e9d23fea4a27e625ca6e259b69571`,88,187,792 bytes;
- `.ipynb_checkpoints/matchups_by_date_offense-checkpoint.csv`, blob
  `8b7fd0b4bc47f851bfadf2175c31a246ce90271a`,351 bytes, header only.

Exact blob copies and a source/checksum/validation receipt now live under
`nba_pipeline/external/matchup_daily_recovered_v1/`; keep these private generated
artifacts out of git. The earlier working-folder search missed these recoverable
tracked checkpoints. This supersedes any broad statement that no dated output
could be recovered locally.

**The recovered defense file is not valid daily evidence.** Its671,490 rows
cover only five labels, October16–20,2018. Every date has the same134,298
complete row values, allowing row reordering and excluding `game_date`.
Full sorted-frame equality is verified across all five dates.336,515 rows
have `GP>1` (maximum7). Example: Andre Drummond versus Myles Turner has
`GP=4`,220.4 partial possessions and32 FGA on every date, including October16,
when neither player appears in the positive-exposure PBP game logs. Thus the
archive contains repeated multi-game aggregate responses with scrape dates
attached. The exact NBA endpoint/filter failure cause is not established.
It cannot be used for before-game matchup features or differenced into daily
counts; doing so would introduce future information and duplicate exposures.

The repo's `scrape.py` and `new_scrape.ipynb` request `LeagueSeasonMatchups`
with equal DateFrom/DateTo and also reference
`matchup_outputs/matchups_{mode}_{year}.csv`; those larger per-year outputs
have not been located in the checked working trees. The core pipeline at
`nba/pipelines/core/databallr/Bam Grabber_playoffs.py` has a real
`fetch_boxscore_matchups_for_game` fallback, but
`fetch_boxscore_matchups_for_season` calls `aggregate_boxscore_matchups` before
saving BAM CSVs. The checked core files therefore retain season-pair grain,
not per-game rows. A header/schema scan of134,578 pipeline/core/linked-data
CSV/parquet candidates and62,581 player-sheet/merged-PBP CSV candidates found
no other full possession-pair game-log table. These are scoped local checks,
not a claim about every historical machine or remote archive.

Separate validated defense measurement files already have game IDs/dates:
`results/estimated_skills/monthly_defense_v1/rim_counts.parquet` has314,552 rows,
and `dfga_counts.parquet` has340,877 rows, for ending seasons2014–2026
(2013-10-29 through2026-06-13). DFGA retains explicit validity/reason fields;
coverage is not universal. These are player-game contest measurements, not
attacker-defender possession pair logs. The older recovered-rim table below
covers only2019–2026 and should not be confused with this expanded monthly source.

This inventory does not admit any new predictive feature or change a model.

## September 20 cumulative-count repair

The 2020–2022 RS unprefixed fields are season-to-date counts, not unusable
single-game counts. The older malformed `DateFrom{date_str}=` query explains
the format. `build_rim_defense_matchup_skills.py` now differences snapshots
only across consecutive positive-minute PBP appearances, separately by season,
phase and player. First snapshots require the first known appearance; missing
snapshots, invalid increments and ambiguous appearance dates remain missing.
Actual PBP team history gates trades: post-team-change cumulative observations
remain excluded because the 2023 overlap exposed a frozen post-trade series.
The archive's own TEAM_ID is not reliable historical team identity. Recovered
rows also use PBP team identity for the whole-team/date zero-contest gate.

Provider normal percentages are NEVER differenced or reused as game baselines.
Recovered counts feed the lagged-league predictive model, not reconstructed
opponent-normalized game savings. Ordinary per-game prefixed counts remain
unchanged. The old artifact is preserved; new files are under
`results/estimated_skills/rim_defense_matchups_recovered_v3/`.

| Season | Games with recovered observations / total | Player-games recovered | Defensive-possession coverage |
| --- | --- | ---: | ---: |
| 2020 | 1054 / 1059 | 20,987 | 95.21% |
| 2021 | 1069 / 1080 | 20,832 | 92.01% |
| 2022 | 1230 / 1230 | 24,104 | 94.51% |

These are games with at least one valid player observation, not complete-game
coverage. Post-trade exclusions are 861/1535/1420 positive-exposure rows. Other
exclusions include missing increments, exposure mismatches and whole-team zero
reports. Remaining 16 games have no accepted observations. The 2023 independent
prefixed-count check is exact on 21676/21679 eligible rows; three discrepancies
on 2022-12-27 remain recorded in `overlap_discrepancies.csv`. This is strong
format validation, not proof every historical provider snapshot is perfect.
All 159461 source rows outside the repaired seasons/phase retain identical
counts and validity. Regression tests cover gaps, season resets, invalid
increments, trade freezes and future-append invariance.

Reproduce with:

```bash
.venv/bin/python nba_pipeline/scripts/build_rim_defense_matchup_skills.py --out nba_pipeline/results/estimated_skills/rim_defense_matchups_recovered_v3
```

The historical source-rejection account below describes the pre-repair baseline.


Research measurement layer, September 14, 2026. Owner requested both statistics
as candidates for the homegrown predictive box-score bank. This adds source
measurements, not a promoted predictive model, VPM prior, or main-table override.

Run `.venv/bin/python nba_pipeline/scripts/build_rim_defense_matchup_skills.py`.
Outputs are ignored under `results/estimated_skills/rim_defense_matchups/`:
`player_games.parquet`, `rim_seasons.csv`, `matchup_seasons.csv`,
`current_research.csv`, searchable `index.html`, `coverage.csv`, and source-hashed
`manifest.json`.

## Rim definition and data contract

Preserve core `nba_rapm/pbp_pipeline/normalization.py`'s positive-good formula:
`rim_points_saved = -2 * rim_dfga * rim_dif% / 100`.
Here calculate its count-based equivalent `2*(FGA*normal_pct - FGM)` to avoid
additional rounding of the percentage difference. The NBA endpoint is
`leaguedashptdefend`, `DefenseCategory=Less Than 6Ft`. This is not the PBP
layup/dunk rim taxonomy, restricted area, or a total rim-deterrence measure.

Select a single complete field family per source file: repaired `lt6ft_totals_`,
then `less_6ft_def_`. The original unprefixed defensive fields are rejected: the
2020-2022 archive repeats season totals in daily rows (194-213 attempts per 100
possessions). The original scraper's `url14` omits the season/type/total parameters.
Endpoint identity alone did not establish daily grain; the initial audit missed
this. Missing rejected fields stay missing, with an explicit source-family label. Never mix an endpoint's baseline with another's
counts. Input files cover 2019-2026 RS and PS separately. Join exact player/date/
season/phase to PBP, exclude ambiguous PBP dates, and require FGA/AST agreement
and minutes agreement within 0.5. Require nonnegative attempts/makes, makes no
larger than attempts, positive defensive exposure, and a valid baseline for
positive attempts. Entire team/date zero-contest panels are invalid. Missing
rows remain missing; no inferred counts are fed to a model.

The recovered provider normal percentage reflects the shooting baseline of the
opponents faced; its historical availability is NOT verified. These saved-point
labels are retrospective and must not be called date-safe pregame predictors.
Individual attribution is provider closest-defender assignment, not causal
credit or a complete accounting of team defense. No saved free throws, deterrence,
rebound continuation, or generic matchup-burden bonus is added.

Export attempts/100 covered defensive possessions, allowed FG%, baseline FG%,
saved points/100 contests, and saved points/100 covered defensive possessions.
Always retain total/covered defensive possessions and coverage. Summing rates
is prohibited; aggregate numerators and denominators first. Rounded provider
baselines mean this is a rounded-baseline estimate.

After rejecting the corrupted fields, the build yields 126,474 usable player-games
from 236,816 unambiguous PBP rows. Coverage is 97.86% of defensive possessions
in supported 2019 and 2023-2026 seasons, or 63.08% across all eight seasons.
The earlier 196,922 / 96.04% claim was invalid and is superseded. The current RS export has 580
rows including matchup-only players; the reader shows 359 with at least 1,000
covered defensive possessions and 80% rim coverage. This is not a verified
active-roster inventory.

## Matchup difficulty

Reuse the BAM source matrix already recovered from core/cloud into sibling VPM:
`vpm/vpm/data/matchup/matchups_two_rs_2018_2026.parquet`. There are 722,575 unique
season/offensive-player/defensive-player pairs. Require unique pairs and finite
nonnegative partial possessions. Do not use BAM's outcome-derived score as
assignment difficulty or equate NBA matchup assignments with rim contests.

For assignment season Y, freeze each opponent's last prior-season uncentered
homegrown `pts_rate` and `ast_rate` posterior. Multiply by 100 and average them
using matchup PARTIAL_POSS. Export scoring and passing burden separately, plus
each difference from the league covered-assignment weighted mean. No arbitrary
combined weight is fitted. Positive means assignments with higher scoring or
assist volume; this is offensive load, not total offensive impact. It can miss
high-impact low-volume players. The two measures are not additive points saved.

Unknown opponents are excluded from the strength-weighted mean, with their
exposure retained in the coverage denominator. Strength observations precede
the assignment season; legacy filter hyperparameters were tuned across history
through 2025, so this is not a pristine as-of-trained validation. No current
assignment outcome enters the difficulty computation. 2018-2019 have no usable
prior own-skill snapshot and stay missing.

2026 matchup data is an incomplete season snapshot without a verified extraction
date. It must not be described as the same through-date as the rim table, and
must not be used to emit 2027 preseason priors. Annual BAM rows cannot support
daily matchup-difficulty estimates without dated assignment data.

## Predictive follow-up contract

The next experiment needs a date-safe baseline independent of the provider's
retrospective normal percentage. Estimate contest activity per defensive
possession and finishing suppression per defended attempt separately; combine
only afterward. Test against blocks-only and no-suppression controls on identical
future games, score made shots/points as well as percentages, and cluster by game.
Height/wingspan can enter the existing explicitly retrospective candidate lane;
measurement dates remain unknown. Do not add assignment burden mechanically to
savings, tune on 2027, or promote any feature without a future gate.

Three tests cover savings algebra/sign, partial-possession weighting and missing
coverage, and invariance of difficulty to observed matchup scoring. Full-build
count algebra was also checked to numerical precision.

Rolling league-relative estimates are now available: see [PREDICTIVE_RIM_SAVED.md](PREDICTIVE_RIM_SAVED.md). They deliberately exclude the provider normal-percentage field.
