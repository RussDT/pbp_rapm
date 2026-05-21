# CLAUDE.md

This file complements `AGENTS.md` for coding agents working in `/Users/russellthomas/Docs/pbp_rapm`.

## Agent Ops Workflow

- Treat `nba_pipeline/docs/` as this repo's FreshDocs surface.
- Use `nba_pipeline/docs/Codex_Scratchpad.md` as the repo-versioned short-term memory file.
- At session start: read `AGENTS.md`, this file, `nba_pipeline/CLAUDE.md`, `nba_pipeline/docs/Codex_Scratchpad.md`, run `git status --short`, then open the relevant docs before editing.
- When behavior, methodology, or operator workflow changes, update the matching docs in `nba_pipeline/docs/` in the same pass and keep `nba_pipeline/docs/README.md` current.
- At session end: append one short scratchpad entry, promote durable lessons into `AGENTS.md` / `CLAUDE.md`, and keep only the latest 5 scratchpad entries.
- Git backup policy: commit code, docs, tests, and small curated reference files; keep generated CSV/parquet/log/model artifacts out of git unless explicitly force-added with a documented reason. See `nba_pipeline/docs/GIT_BACKUP_STRATEGY.md`.
- When publishing the standalone ShotQuality RAPMs to the downstream `rapms` repo, source the canonical solver outputs `nba_pipeline/results/td/transition_freq_24_26_all_td700_results.csv`, `nba_pipeline/results/td/transition_rim_24_26_all_td700_results.csv`, and `nba_pipeline/results/td/initial_ev_24_26_all_td700_results.csv`, then rename them to the published filenames `transitionfreq_24_26_all_td700_results.csv`, `transitionrim_24_26_all_td700_results.csv`, and `initialev_24_26_all_td700_results.csv`; copy `SPECIAL_RAPM` into `master_results/` before syncing downstream.
- `rapm.py --cv-alpha` is opt-in; it tunes offense and defense alphas separately with grouped-by-`gameid` folds and writes a companion `*_alpha_cv.csv` artifact, while `--publish-to-rapms` publishes only the main result CSV to the downstream `rapms` repo.

## Project Snapshot

- Product: NBA play-by-play RAPM, six-factor, and DRL/Shapley research repo
- Primary code surface: `nba_pipeline/`
- Canonical docs surface: `nba_pipeline/docs/`
- Scratchpad: `nba_pipeline/docs/Codex_Scratchpad.md`

## Non-Negotiables

1. Do not conflate the state attribution layer with the event attribution layer.
2. Preserve the additive contract documented in `nba_pipeline/docs/ADDITIVE_OUTPUT_SCHEMA.md` when touching decomposition outputs.
3. Update `nba_pipeline/docs/DRL_SHAPLEY_BUILD_STATUS.md` whenever project status or modeling direction changes materially.
4. Keep durable methodology rules in `AGENTS.md` / `CLAUDE.md`, not in the scratchpad.

## Legacy PBP Notes

- Raw fetched PBP does not populate `player2_*` assist fields, so assist-derived metrics need to parse the `(... N AST)` description tag.
- Raw fetched PBP can contain duplicated event rows for the same game; normalize `game_id` to a stable 10-character string and dedupe exact rows before writing or processing seasonal raw parquet files.
- In the raw lineup/on-off parser, made FG rows should keep their shooting team as `TeamOnOffense` even when the next event is a shooting foul; otherwise `and-1` points can disappear or flip teams during possession-side fill.
- In the raw lineup/on-off parser, foreign-side technical/admin FT rows should be emitted as zero-possession scoring rows instead of being folded into the terminal possession owner.
- In the raw lineup/on-off parser, split any contiguous non-terminal offense segment inside one `poss_group` into its own possession row instead of letting the terminal owner inherit its scoring.
- In the standard possession definition, offensive foul / charge rows without a companion turnover row should still mark `End_of_Possession` and keep `TeamOnOffense` on the fouling side.
- In the standard possession definition, missed final FTs should mark `End_of_Possession` unless the next row is a live defensive rebound by the other team or a live offensive rebound by the shooting team.
- In the shared standard possession flow, fill `poss_offense` after FT correction and before O/D mapping. Historical neutral-description team turnovers should be matched with game-local side labels plus NBA abbreviation/nickname aliases; blank terminal `EndOfPeriod` or neutral-turnover rows can inherit the opposite side of the previous completed possession in the same game/period.
- For Gabriel historical raw rebuilds, apply game-local lineup-side guardrails before processing: infer player side from non-overlap lineup evidence, strip impossible same-player home/away collisions, dedupe repeated positive player ids within one side, and drop only rows that remain below the partial-lineup minimum after dedupe.
- Rare continued possessions after a missed `1 of 1` FT can legitimately produce multi-make possession totals in legacy assist-based numerators.
- Turnover-ended possessions are not guaranteed to be scoreless in the raw possession layer; dead-ball or pre-turnover scoring can still survive onto a terminal turnover row. Clean first-chance TOV flags must require no prior first-chance FGA/FT completion event in the same possession, so scored possessions stay in first-chance scoring and do not also become first-chance turnover rows.
- New modern RAPM metrics are not complete until the raw enrichment path, the standard parquet builder, and `scripts/reprocess_metric.py` all know about them.
- Point/value-style processed metrics need the standard parquet sign convention `Def_Diff = -Off_Diff`; same-sign offense/defense outputs will corrupt downstream additive decompositions.
- For any decomposition change, prove the processed parquet/CSV row algebra before interpreting RAPM coefficient sums: check max row gap, sum gap, and child-parent row identity counts. Separate ridge solves, centering, and rounded exports can leave tiny coefficient gaps even when the row algebra is exact.
- Legacy processed parquets do not share a canonical embedded `Season` encoding; use filename-derived season keys for season-level summary harnesses and treat parquet `Season` as an auditable field, not the source of truth.
- Production EFG-value display bundles should use six atomic `FIRST_CHANCE` shot-value aliases (`ALT_EFG_RIM_FREQ`, `ALT_EFG_RIM_FG`, `ALT_EFG_MID_FREQ`, `ALT_EFG_MID_FG`, `ALT_EFG_THREE_FREQ`, `ALT_EFG_THREE_FG`) plus `ALT_FT`, display `TOV Value = ALT_TOV_VALUE`, and `SECOND_CHANCE_CLEAN`; derive rim/mid/three/shot totals in the bundle layer. `ALT_FT` can be split further with `ALT_FT_FREQ + ALT_FT_SEVERITY = ALT_FT`, where frequency is average first-chance FT-trip value and severity is trip value above/below that average. `ALT_EFG_RIM/MID/THREE` broad zone solves are audit-only. `ALT_TOV_VALUE` is the point-valued first-chance turnover complement and should equal `ALT_BADPASS_TOV_VALUE + ALT_SCORING_TOV_VALUE`; `ALT_FC_COMPLETION` is legacy audit/glue for the older `FIRST_CHANCE - ALT_EFG_VALUE - ALT_FT` construction. Weighted-factor exports from this bundle should use the matching legacy alt3 weighted-factors file as the player universe / `Latest_Year` source and sign-flip all defensive component columns so positive means good defense.
- RAPM / factor `--age-dummies` uses `dpm_history.csv` only as a player-season age lookup (`nba_id`, `season`, `age`); the age effects themselves are still estimated inside the possession regression, and DARKO `o_dpm` / `d_dpm` are not age-effect targets. `03_run_rapm_analysis.py --age-dummies` applies this to the full daily weighted-factor metric set and writes `_agefe` outputs.
- Backend processors that read Supabase player stats should prefer `SUPABASE_SERVICE_ROLE_KEY` over `SUPABASE_KEY`; the latter may be a publishable key and can fail in the local processing environment.
- Gabriel `merged_playbyplay/old_data` is the implemented pre-2014 source. Use `nba_pipeline/scripts/build_gabriel_old_pbp.py` to build `1997-2013` RS/PS raw files and core processed parquets; converted rows intentionally use partial-lineup mode with unknown slots as `0`, and the solver must exclude player id `0`. The converter repairs missing source `team` abbreviations from stable `teamId` mappings before routing descriptions to home/visitor fields. Regular-season and playoff 1997-2013 non-ShotQuality factor parquets exist for `LA_RAPM`, shot-location frequency/FG%, assist, FT premium, and clean first-/second-chance metrics; `FIRST_CHANCE_CLEAN` is a supported Gabriel builder metric, not just a targeted reprocess path. Use `nba_pipeline/scripts/reprocess_metric.py METRIC 1997 2013 --season-types rs|ps --workers N` for targeted historical rebuilds; use `--raw-dir` and `--processed-dir` for validation-first rebuilds outside production paths. If PS source files are unavailable, repair existing PS raw files with `nba_pipeline/scripts/repair_existing_historical_lineup_sides.py` before reprocessing. The alt3 aliases (`ALT_TS`, `ALT_EFG`, `ALT_FT`, `ALT_TOV`, `ALT_BADPASS_TOV`, `ALT_SCORING_TOV`) are derived from `FIRST_CHANCE` rather than separate physical `ALT_*` parquets, and `build_processed_rs_parquet_averages.py` emits them into the audit CSVs from that source. Gabriel historical `RAPM` / `TS` processing uses actual FT outcomes when shooter FT% is missing, without changing the daily pipeline default; `LA_RAPM` is the luck-adjusted surface. For 1997-2000 FT replacement, `nba_pipeline/external/bref_ft_pct_1997_2000.csv` is the local Basketball Reference `nba_id`/FT% lookup, and the shared stats loader uses that regular-season FT% lookup for both RS and PS `RAPM`, `LA_RAPM`, `TS`, and `FT_PREMIUM` processing when Supabase stats are unavailable. Canonical docs: `nba_pipeline/docs/HISTORICAL_1997_2013_PARQUETS.md`.
- For Gabriel historical starter repairs, the supplemental NBA stats archive at `/Users/russellthomas/Docs/2026_NBA_PIPELINE/archive (1)` can validate games, teams, players, and substitutions. Use `PlayByPlay.parquet` substitution descriptions plus `PlayerStatistics.csv` active/DNP rows, `Players.csv`, and `TeamHistories.csv`; do not treat old `PlayerStatistics*.csv startingPosition` values as starter flags without auditing counts, because 2009 rows can have more than five nonblank values per team.
- Preferred long historical core runs use fixed constants, not estimated season/age nuisance columns: `run_historical_core_weighted_factors.py --fixed-season-effects` subtracts each metric's RS raw target mean by season from both RS and PS rows, and `--age-poly-coefficients` subtracts fixed `curve(age) - curve(27)` row offsets before centering and solving.
- The BKN 2020 friend-PBP ingest fixture lives at `nba_pipeline/raw_data/BKN_2020_rs.parquet`; use `nba_pipeline/scripts/validate_friend_pbp_ingest.py` for side-by-side conversion/process/solve checks. Current best state: shot-zone metrics are exact, TOV flags are exact on matched rows, RAPM has 14,541/14,542 official rows matched with one official-only goaltend-and-one event, remaining friend-only rows are mostly zero-value team rebounds around blocked-shot jump-ball/admin sequences, and the bigger solve-level gap is lineup parity (`740` of `14,541` matched RAPM rows differ in O/D player sets).
- WNBA clean EFG-value decomposition lives in `wnba_test/clean_decomp_wnba.py`; use it to write WNBA clean `FIRST_CHANCE` / `SECOND_CHANCE_CLEAN` CSVs and `weighted_factors_alt3_efg_value_*` display outputs. Its clean total is `FIRST_CHANCE + SECOND_CHANCE_CLEAN` on matching rows, not the older WNBA `RAPM{YY}.csv` surface. WNBA mirrors the NBA `ALT_FT_FREQ + ALT_FT_SEVERITY = ALT_FT` trip split and first-chance TOV guard: true TOV rows require no prior first-chance FGA/FT completion and use the average non-turnover first-chance TS/scoring baseline.
- WNBA raw CSV possession parsing should use `prepare_wnba_standard_possession_df` in `wnba_test/process_rapm_wnba.py`; generic team rebounds/turnovers can have `teamid = 0` with the team id in `personid`, and admin rows can separate a miss from its rebound. Avoid immediate previous-row MISS/REBOUND-only logic for WNBA defensive-rebound possession ends.
- Current WNBA PBP fetch fallback is the static WNBA liveData CDN, not the old stats endpoint: `https://cdn.wnba.com/static/json/liveData/playbyplay/playbyplay_{GAME_ID}.json` plus `boxscore_{GAME_ID}.json`; reconstruct lineups from boxscore starters and substitution events when `stats.wnba.com/stats/playbyplayv3` / `gamerotation` time out.
- The current WNBA downstream possession surface is CSV-based, not parquet-based. After WNBA possession-logic changes, rebuild `RAPM`, `TS`, `REB`, `TOV`, `FIRST_CHANCE`, and `SECOND_CHANCE_CLEAN` CSVs for both RS and PS before rerunning five-year clean EFG-value solves; `REB` should use inferred pending miss-side/rebound-side context.
- `SHOOTER_OREB` is built on missed-FGA rows so `Shooter` can be a first-class design column separate from `O1-O4` non-shooter offensive players; it exposes team `Offensive_Rebound` and `Self_Offensive_Rebound` targets for shooter miss-recoverability models.

## Key Docs

- `nba_pipeline/docs/README.md`
- `nba_pipeline/docs/PIPELINE_QUICKSTART.md`
- `nba_pipeline/docs/RAPM_METHODOLOGY.md`
- `nba_pipeline/docs/ADDITIVE_OUTPUT_SCHEMA.md`
- `nba_pipeline/docs/DRL_SHAPLEY_BUILD_STATUS.md`
- `nba_pipeline/docs/HISTORICAL_1997_2013_PARQUETS.md`
- `nba_pipeline/docs/tov-decomposition.md`

## Definition of Done

- requested change implemented
- matching docs updated when behavior or methodology changed
- validation run or explicitly skipped
- `nba_pipeline/docs/Codex_Scratchpad.md` updated if new durable context was learned
