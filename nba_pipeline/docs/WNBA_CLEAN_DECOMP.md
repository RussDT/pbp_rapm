# WNBA Clean EFG-Value Decomposition

`wnba_test/clean_decomp_wnba.py` ports the NBA clean first-/second-chance EFG-value approach to the WNBA CSV workflow.

## Outputs

Processed clean possession files are written to:

- `wnba_test/Processed/FIRST_CHANCE{YY}.csv`
- `wnba_test/Processed/FIRST_CHANCE{YY}_PS.csv`
- `wnba_test/Processed/SECOND_CHANCE_CLEAN{YY}.csv`
- `wnba_test/Processed/SECOND_CHANCE_CLEAN{YY}_PS.csv`

The current WNBA workflow is CSV-based. No WNBA parquet surfaces exist in
`pbp_rapm` or downstream `rapms`; when rebuilding the WNBA downstream possession
surface, rebuild the processed `RAPM`, `TS`, `REB`, `TOV`, `FIRST_CHANCE`, and
`SECOND_CHANCE_CLEAN` CSV families for both RS and PS.

Weighted-factor style display files are written to:

- `wnba_test/Results/clean_decomp_{start}_{end}_{season_type}/weighted_factors_alt3_efg_value_{start}_{end}_{season_type}.csv`

When publishing these files to the downstream `rapms` repo, use the `wnba_`
prefix so V3 can distinguish the WNBA schema from NBA Alt3 EFG-value files:

- `master_results/wnba_weighted_factors_alt3_efg_value_{start}_{end}_{season_type}.csv`

The display columns mirror the newest NBA EFG-value bundle:

- six first-chance shot-value pieces: rim/mid/three frequency and FG value
- `ALT_FT`
- `ALT_FT_FREQ` and `ALT_FT_SEVERITY`, with `ALT_FT_FREQ + ALT_FT_SEVERITY = ALT_FT` before three-decimal export rounding
- display `ALT_TOV_VALUE`, sourced from `ALT_FC_COMPLETION`
- diagnostic turnover-loss columns: `ALT_TOV_LOSS_VALUE`, bad-pass loss, scoring/handling loss
- direct `SECOND_CHANCE_CLEAN`
- `oFC`, `dFC`, display sums, and residual checks

## Commands

Fetch the current completed WNBA regular season from the static liveData CDN:

```bash
python wnba_test/fetch_wnba_cdn.py --season 2026 \
  --season-type "Regular Season" \
  --update-player-map
```

The season scanner checks CDN ids in sequence (`102YYxxxxx` for regular season,
`104YYxxxxx` for playoffs), keeps final games by default, appends new games to
`wnba_test/WNBA{YY}.csv`, and updates `wnba_test/player_index_map.csv` with new
boxscore player names. Use `--overwrite` to rebuild the raw season file from
scratch, `--include-live` for in-progress games, and `--end-seq N` for bounded
smoke tests.

Fetch one or more explicit WNBA raw games from the current static liveData CDN:

```bash
python wnba_test/fetch_wnba_cdn.py 1022500001 \
  --output wnba_test/WNBA25_cdn_sample_1022500001.csv
```

This path uses:

- `https://cdn.wnba.com/static/json/liveData/playbyplay/playbyplay_{GAME_ID}.json`
- `https://cdn.wnba.com/static/json/liveData/boxscore/boxscore_{GAME_ID}.json`

It is the current fallback when the old stats API endpoints time out. The fetcher
reconstructs row lineups from boxscore starters and grouped substitution events,
then writes the same raw CSV columns consumed by `process_rapm_wnba.py`.

Process one or more raw WNBA files:

```bash
python wnba_test/clean_decomp_wnba.py process wnba_test/WNBA25.csv
```

Run a clean weighted-factor file from already-processed clean files:

```bash
python wnba_test/clean_decomp_wnba.py run 25 25 RS
```

Process and run a window:

```bash
python wnba_test/clean_decomp_wnba.py process-and-run 24 25 ALL
```

Run the current five-year no-decay alpha `1750/4000` window:

```bash
python wnba_test/clean_decomp_wnba.py process-and-run 21 25 ALL \
  --off-alpha 1750 \
  --def-alpha 4000
```

Run the current five-year decayed production-style window:

```bash
python wnba_test/clean_decomp_wnba.py process-and-run 21 25 ALL \
  --time-decay-days 700 \
  --off-alpha 1750 \
  --def-alpha 4000
```

Run the current 2026 regular-season clean bundle after a CDN pull:

```bash
python wnba_test/process_rapm_wnba.py wnba_test/WNBA26.csv
python wnba_test/clean_decomp_wnba.py process wnba_test/WNBA26.csv
python wnba_test/clean_decomp_wnba.py run 26 26 RS \
  --off-alpha 1750 \
  --def-alpha 4000
```

`process-and-run` should be used when changing FT logic because the expected FT
values and FT frequency/severity trip values are baked into the processed
`FIRST_CHANCE` and `SECOND_CHANCE_CLEAN` CSV rows before the solve.

## FT Source

Expected free-throw value uses the shared WNBA shooting-stats loader:

- `wnba_test/wnba_stats_source.py`

- `/Users/russellthomas/Docs/wnba/wnba-stats.csv`
- `/Users/russellthomas/Docs/wnba/data/YYYY_pbp.csv` or `YYYYps_pbp.csv`
- `https://api.pbpstats.com/get-totals/wnba`

The loader keeps WNBA expected shooting stats tied to the WNBA stats backend:
historical rows come from `/Users/russellthomas/Docs/wnba/wnba-stats.csv`.
Current seasons that have not been rebuilt into that final CSV use the WNBA
backend's PBP Stats player-totals cache under `/Users/russellthomas/Docs/wnba/data/`;
if the cache is missing, the loader pulls the same source and writes the cache.
It derives `FTPerc` from `FtPoints / FTA` and `ThreePerc` from `Fg3Pct` or
`FG3M / FG3A`. Raw CDN play-by-play is not used to infer FT priors. If all
stats sources are missing, rows fall back to `0.75` for FT and `0.35` for 3P.

The WNBA FT split mirrors the NBA clean trip-accounting layer:

- `FC_FT_Diff` is the first-chance FT point value
- `FC_FT_FREQ_Diff` prices each first-chance FT trip at the processed-file average FT-trip value
- `FC_FT_SEVERITY_Diff` is the trip's value above or below that average
- `FC_FT_FREQ_Diff + FC_FT_SEVERITY_Diff = FC_FT_Diff` on the processed CSV rows

The WNBA clean TOV value uses the average non-turnover first-chance scoring
baseline, not average EFG shot value. True first-chance TOV rows now require no
prior first-chance FGA/FT completion event in the same possession, matching the
NBA clean first-chance rule, so TOV rows are scoreless and the point value is
exactly `-avg(non-turnover FIRST_CHANCE Net_Diff)`.

## Partial Lineups

The clean WNBA path supports partial raw lineup rows when there are at least four
known home players and four known away players. Missing lineup slots are filled
with player id `0`, final rows must still have at least four known offensive and
four known defensive players, and player id `0` is excluded from the regression
and exposure counts.

In the 2021-2025 WNBA source files checked on 2026-05-05, all raw rows with
missing lineup slots had fewer than four known players on at least one team, so
no partial rows survived into the final 5-year possession surface.

## Robust Possession Builder

`wnba_test/process_rapm_wnba.py` exposes `prepare_wnba_standard_possession_df`,
and the clean WNBA path uses that helper before first-/second-chance splitting.
The standard WNBA `RAPM`, `TS`, and `TOV` processors should also route through
that helper, while `REB` should use the same inferred pending miss-side/rebound
side context before assigning offensive rebound chances.
The WNBA raw feed differs from the NBA parquet path in two important ways:

- team rebounds and team turnovers can appear only in generic `description`
  text, with `teamid = 0` and the team id stored in `personid`
- blocked-shot, replay, substitution, or other admin rows can sit between a
  missed shot and the rebound row

The WNBA helper now infers home/away team ids by game, resolves generic team
events from `personid`, tracks the pending missed-shot side through intervening
admin rows, marks defensive-rebound possession ends from rebound side versus
miss side, and carries a live possession owner into generic period-end rows.
This avoids the older immediate-previous-row rebound rule that undercounted
defensive-rebound-ended possessions and inflated raw on-court ORTG/DRTG levels.

## Current Validation

The 2026 RS CDN pull completed on 2026-05-29:

- command: `python wnba_test/fetch_wnba_cdn.py --season 2026 --season-type "Regular Season" --update-player-map`
- raw output: `wnba_test/WNBA26.csv`
- `55` final games from `1022600001` through `1022600055`
- `27,558` raw rows, no duplicate `game_id` / `event_num` rows, and no missing home/away lineup slots
- standard processed outputs: `RAPM26.csv` `9,164` rows, `TS26.csv` `7,587`, `REB26.csv` `4,367`, `TOV26.csv` `9,164`
- clean outputs: `FIRST_CHANCE26.csv` `9,164` rows and `SECOND_CHANCE_CLEAN26.csv` `9,164`, each with `0.00000000` identity, FT split, and EFG component gaps
- clean bundle output: `wnba_test/Results/clean_decomp_26_rs_a1750_4000/weighted_factors_alt3_efg_value_26_rs_a1750_4000.csv`, `193` players, max offense/defense/net display residual `0.0000`
- smoke solve: `python wnba_test/rapm_wnba.py RAPM 26 26 RS --pure --cores 4` wrote `wnba_test/Results/rapm_26_rs_pure_results.csv`
- caveats: `stats.wnba.com` still disconnected for league/player-stat calls, so 2026 expected FT/3P values now come from the shared WNBA stats source, which wrote/read `193` current WNBA player rows through `/Users/russellthomas/Docs/wnba/data/2026_pbp.csv`; the old NBA schedule CSV has no WNBA 2026 dates, so WNBA clean time decay still relies on the existing date proxy.

The 2024-2025 ALL WNBA run completed at:

- `wnba_test/Results/clean_decomp_24_25_all/weighted_factors_alt3_efg_value_24_25_all.csv`

Validation from that run:

- 2024 RS: `35,248` clean rows
- 2024 PS: `3,086` clean rows
- 2025 RS: `41,261` clean rows
- 2025 PS: `3,418` clean rows
- every clean first-/second-chance processed file reported `0.00000000` max row identity gap before final output
- the final weighted-factor display reported max offense, defense, and net residuals of `0.0000` after rounding

The 2021-2025 ALL `td700_a1750_4000` WNBA run completed at:

- `wnba_test/Results/clean_decomp_21_25_all_td700_a1750_4000/weighted_factors_alt3_efg_value_21_25_all_td700_a1750_4000.csv`
- downstream publish name: `master_results/wnba_weighted_factors_alt3_efg_value_21_25_all_td700_a1750_4000.csv`

Validation from that run:

- `307` players
- max offense, defense, and net display residuals of `0.0000` after rounding
- rerun on 2026-05-05 after rebuilding the WNBA processed CSV surface with robust EOP logic; A'ja Wilson: `off 3.914`, `def 0.722`, `net_rapm 4.636`, `nALT_FT 0.615`
- rerun on 2026-05-06 after adding the FT frequency/severity split and the no-prior-completion first-chance TOV guard; A'ja Wilson: `off 3.914`, `def 0.722`, `net_rapm 4.636`, `nALT_FT 0.614`, `nALT_FT_FREQ 0.571`, `nALT_FT_SEVERITY 0.043`

The 2021-2025 ALL no-decay `a1750_4000` WNBA run completed at:

- `wnba_test/Results/clean_decomp_21_25_all_a1750_4000/weighted_factors_alt3_efg_value_21_25_all_a1750_4000.csv`
- downstream publish name: `master_results/wnba_weighted_factors_alt3_efg_value_21_25_all_a1750_4000.csv`

Validation from that run:

- `307` players
- `186,377` clean possessions across `1,278` games
- no player id `0` in the output
- no partial rows survived this source window after the 4-and-4 raw lineup rule
- max offense, defense, and net display residuals of `0.0000` after rounding
- A'ja Wilson: `off 3.418`, `def 0.658`, `net_rapm 4.076`, `nALT_FT 0.774`
- rerun on 2026-05-06 after adding the FT frequency/severity split and the no-prior-completion first-chance TOV guard; A'ja Wilson: `off 3.418`, `def 0.658`, `net_rapm 4.076`, `nALT_FT 0.777`, `nALT_FT_FREQ 0.817`, `nALT_FT_SEVERITY -0.040`

The 2026-05-06 written-CSV audit checked all 2021-2025 RS/PS WNBA
`FIRST_CHANCE` files: `10` files, `202,401` rows, no missing
`FC_FT_FREQ_Diff` / `FC_FT_SEVERITY_Diff`, max FT split gap `6.19e-15`, true
first-chance TOV rows `31,480`, and TOV rows with nonzero `Net_Diff` `0`.

The 2021-2025 RS/PS processed CSV rebuild on 2026-05-05 produced these standard
surface row counts after the robust EOP fix:

- 2021: `RAPM 30,114`, `TS 24,829`, `REB 15,267`, `TOV 30,114`
- 2021 PS: `RAPM 2,662`, `TS 2,201`, `REB 1,346`, `TOV 2,662`
- 2022: `RAPM 34,223`, `TS 27,996`, `REB 16,993`, `TOV 34,223`
- 2022 PS: `RAPM 3,581`, `TS 3,031`, `REB 1,810`, `TOV 3,581`
- 2023: `RAPM 38,249`, `TS 31,481`, `REB 19,141`, `TOV 38,249`
- 2023 PS: `RAPM 3,139`, `TS 2,631`, `REB 1,651`, `TOV 3,139`
- 2024: `RAPM 38,524`, `TS 31,674`, `REB 19,188`, `TOV 38,524`
- 2024 PS: `RAPM 3,362`, `TS 2,811`, `REB 1,710`, `TOV 3,362`
- 2025: `RAPM 44,780`, `TS 36,820`, `REB 22,677`, `TOV 44,780`
- 2025 PS: `RAPM 3,767`, `TS 3,130`, `REB 1,927`, `TOV 3,767`

After the robust WNBA possession helper change, raw regular-season EOP counts
versus PBPStats team `OffPoss` are:

- 2021: `30,115` WNBA EOP rows vs `30,427` PBPStats possessions
- 2022: `34,224` vs `34,511`
- 2023: `38,249` vs `38,278`
- 2024: `38,524` vs `37,970`
- 2025: `44,780` vs `44,724`

The same rebuilt 2021-2025 regular-season surface gives A'ja Wilson on-court
ratings of `111.71` offense, `100.09` defense, and `+11.62` net. The PBPStats
WOWY reference shown by the user was `111.8` offense, `99.7` defense, and
`+12.1` net.

The robust helper was also used to rebuild `RAPM21_PS.csv` through
`RAPM25_PS.csv`. Postseason row counts versus PBPStats team `OffPoss` are:

- 2021 PS: `2,662` rebuilt rows vs `2,720` PBPStats possessions
- 2022 PS: `3,581` vs `3,583`
- 2023 PS: `3,139` vs `3,154`
- 2024 PS: `3,362` vs `3,401`
- 2025 PS: `3,767` vs `3,810`

A'ja Wilson's rebuilt 2021-2025 all-games line is `111.32` offense, `100.68`
defense, and `+10.64` net on court; off court for Las Vegas it is `99.84`
offense, `102.65` defense, and `-2.81` net.

## Caveats

- As of 2026-05-07, `stats.wnba.com/stats/playbyplayv3`,
  `stats.nba.com/stats/playbyplayv3` with WNBA game ids, and
  `stats.wnba.com/stats/gamerotation` timed out from the local environment.
  `cdn.wnba.com/static/json/liveData/playbyplay/playbyplay_{GAME_ID}.json`
  returned a complete sample game (`1022500001`, `519` rows), and the matching
  boxscore endpoint returned starters/player metadata. The direct `cdn.nba.com`
  path returned access denied for the same WNBA game id.
- The WNBA finalizer still inherits the older WNBA script's game-date merge against the NBA schedule CSV, so `game_date` is currently missing in these WNBA clean outputs. For `--time-decay-days`, the clean runner attaches a stable date proxy by WNBA season and sorted `game_id` order using season date bounds. Replace this with real WNBA schedule dates when the stats API is reachable or a local WNBA schedule table is available.
- WNBA FT values are player-season adjusted through `wnba_test/wnba_stats_source.py`; use `/Users/russellthomas/Docs/wnba/wnba-stats.csv` when it has the shooter/year/playoff split, then `/Users/russellthomas/Docs/wnba/data/YYYY_pbp.csv` or `YYYYps_pbp.csv`, and PBP Stats WNBA player totals only to populate that backend cache when needed. Rows without a matched FT% still use the `0.75` fallback.
- This workflow intentionally solves the clean total as `FIRST_CHANCE + SECOND_CHANCE_CLEAN` on the same rows, not from the older `RAPM{YY}.csv` files, because the older WNBA possession rows do not exactly match the new clean split surface.
- The 2024 raw WNBA feed still has a positive total-possession gap versus PBPStats
  after the robust helper (`+554` EOP rows), concentrated in period-end rows.
  Keep validating that season separately before treating raw totals as exact.
