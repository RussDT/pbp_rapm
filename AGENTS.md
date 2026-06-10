# AGENTS.md

## Agent Ops Workflow

- Treat `nba_pipeline/docs/` as this repo's FreshDocs surface.
- Use `nba_pipeline/docs/Codex_Scratchpad.md` as the repo-versioned short-term memory file.
- At session start: read `AGENTS.md`, `CLAUDE.md`, `nba_pipeline/CLAUDE.md`, `nba_pipeline/docs/Codex_Scratchpad.md`, run `git status --short`, then open the relevant docs before editing.
- When behavior, methodology, or operator workflow changes, update the matching docs in `nba_pipeline/docs/` in the same pass and keep `nba_pipeline/docs/README.md` current.
- At session end: append one short scratchpad entry, promote durable lessons into `AGENTS.md` / `CLAUDE.md`, and keep only the latest 5 scratchpad entries.
- Git backup policy: commit code, docs, tests, and small curated reference files; keep generated CSV/parquet/log/model artifacts out of git unless explicitly force-added with a documented reason. See `nba_pipeline/docs/GIT_BACKUP_STRATEGY.md`.
- Repo cleanup/onboarding policy: start with `nba_pipeline/docs/CONTRIBUTOR_ONBOARDING.md`, `nba_pipeline/docs/ARTIFACT_MANIFEST.md`, and `nba_pipeline/docs/REPO_CLEANUP_PLAN.md`; use `nba_pipeline/scripts/audit_repo_inventory.py` for dry-run inventory before moving, deleting, or reclassifying artifacts/scripts.
- When publishing the standalone ShotQuality RAPMs to the downstream `rapms` repo, source the canonical solver outputs `nba_pipeline/results/td/transition_freq_24_26_all_td700_results.csv`, `nba_pipeline/results/td/transition_rim_24_26_all_td700_results.csv`, and `nba_pipeline/results/td/initial_ev_24_26_all_td700_results.csv`, then rename them to the published filenames `transitionfreq_24_26_all_td700_results.csv`, `transitionrim_24_26_all_td700_results.csv`, and `initialev_24_26_all_td700_results.csv`; `SPECIAL_RAPM` should be copied into `master_results/` before the downstream sync.
- `rapm.py --cv-alpha` is opt-in; it tunes offense and defense alphas separately with grouped-by-`gameid` folds and writes a companion `*_alpha_cv.csv` artifact, while `--publish-to-rapms` publishes only the main result CSV to the downstream `rapms` repo.

## Repo-Level DRL/Shapley Context

This repo now has an active DRL/Shapley build effort inside:
- [train_drl_shapley.py](/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/scripts/train_drl_shapley.py)
- [nba_pipeline/docs/DRL_SHAPLEY_BUILD_STATUS.md](/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/docs/DRL_SHAPLEY_BUILD_STATUS.md)
- [nba_pipeline/docs/ADDITIVE_OUTPUT_SCHEMA.md](/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/docs/ADDITIVE_OUTPUT_SCHEMA.md)
- [nba_pipeline/autoresearch_drl_shapley/README.md](/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/autoresearch_drl_shapley/README.md)
- [nba_pipeline/autoresearch_drl_shapley/program.md](/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/autoresearch_drl_shapley/program.md)
- [nba_pipeline/autoresearch_drl_shapley/results.tsv](/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/autoresearch_drl_shapley/results.tsv)

## What Future Agents Should Know

The DRL/Shapley project is trying to implement the Sloan paper's core ideas:
- state value modeling from play-by-play
- event value via `delta_V = reward + gamma * V(next) - V(current)`
- Shapley-style player attribution
- additive event decomposition
- separate pair synergy diagnostics

This work is intended to complement, not replace, the repo's six-factor RAPM core.

## Legacy PBP Notes

- In fetched raw PBP, `player2_id` / `player2_name` are not populated for assists; assist-derived metrics must parse the `(... N AST)` tag from the shot description text.
- Fetched raw PBP can contain duplicated event rows for the same game; normalize `game_id` to a stable 10-character string and dedupe exact raw rows before writing or processing `NBAXX.parquet`.
- For per-FGA shot-clock estimation, sort raw events by `game_id`, `period`, `clock_sec` descending, then `event_num` ascending; corrected NBA PBP events can be appended with late event numbers at an earlier clock.
- In the raw lineup/on-off parser, assign `TeamOnOffense` on made FG rows even when the next row is a shooting foul; `and-1` made baskets still belong to the shooting team even if the possession-end rule treats the sequence separately.
- In the raw lineup/on-off parser, foreign-side technical/admin FT rows should be preserved as zero-possession scoring rows instead of being folded into the live-ball possession owner.
- In the raw lineup/on-off parser, any contiguous non-terminal offense segment inside one `poss_group` should be split into its own possession row; otherwise the terminal owner absorbs carried-over `and-1`, technical/admin, or pre-change scoring.
- In the standard possession definition, offensive foul / charge rows without a companion turnover row should still end the possession and keep `TeamOnOffense` on the fouling side.
- In the standard possession definition, a missed final FT (`1 of 1`, `2 of 2`, `3 of 3`) should end the possession unless the next row is a live defensive rebound by the other team or a live offensive rebound by the shooting team.
- In the shared standard possession flow, fill `poss_offense` after FT correction and before O/D mapping. Historical neutral-description team turnovers should be matched with game-local side labels plus NBA abbreviation/nickname aliases; blank terminal `EndOfPeriod` or neutral-turnover rows can inherit the opposite side of the previous completed possession in the same game/period.
- For Gabriel historical raw rebuilds, apply game-local lineup-side guardrails before processing: infer player side from non-overlap lineup evidence, strip impossible same-player home/away collisions, dedupe repeated positive player ids within one side, and drop only rows that remain below the partial-lineup minimum after dedupe.
- Under the legacy standard possession definition, rare continued possessions after a missed `1 of 1` FT can contain multiple assisted makes before the possession ends.
- Turnover-ended possessions are not guaranteed to be scoreless in the raw possession layer; dead-ball or pre-turnover scoring can survive onto a terminal turnover row. Clean first-chance TOV flags must require no prior first-chance FGA/FT completion event in the same possession, so scored possessions stay in first-chance scoring and do not also become first-chance turnover rows.
- When adding a new modern RAPM metric, wire all three surfaces together: `01b_enrich_pbp_shotquality.py` if new raw SQ fields are needed, `02_process_rapm.py` for the standard parquet build, and `scripts/reprocess_metric.py` for fast historical/backfill runs.
- When adding a new solver alias or published metric family, update `rapm.py` prefix allowlists, `03_run_rapm_analysis.py` run/organize allowlists, and any `build_season_intercepts.py` first-chance special cases in the same pass; otherwise solve-time success can still fail during result organization or intercept generation.
- For point/value-style processed metrics, keep the standard sign convention in the parquet surface: `Def_Diff = -Off_Diff`. Writing same-sign offense and defense values will break downstream additive comparisons.
- For any decomposition change, prove the processed parquet/CSV row algebra before interpreting RAPM coefficient sums: check max row gap, sum gap, and child-parent row identity counts. Separate ridge solves, centering, and rounded exports can leave tiny coefficient gaps even when the row algebra is exact.
- Legacy processed parquets do not share a canonical embedded `Season` encoding; for season-level summaries or intercept harnesses, prefer filename-derived season keys and audit parquet `Season` values separately.
- Treat `weighted_factors_alt3_efg_value_*` as the active public Alt3 process. Older `weighted_factors_alt3_*` files are legacy/audit artifacts; they rebuild residual `oSC` / `dSC` buckets and should not be used as the current public interpretation path. Active player-facing EFG-value weighted-factor builds should write CSV and parquet siblings.
- Treat RAPM DECOMP scposs as the current player-facing `csvs/scposs` surface. The daily pipeline should rebuild `csvs/scposs/{nba_id}.csv` from the active `weighted_factors_alt3_efg_value_*_all_rb_se_a2000_4000.csv` files via `/Users/russellthomas/Docs/REPLIT_NBA_RAPM/scripts/sync_alt3_decomp_to_scposs.py`; `update_2026_scposs.py` is legacy six-factor output and should not be used as the current decomp path. The current metrics export should read current 1Y/2Y/3Y/4Y/5Y RAPM values from that scposs surface before writing the explicit `one_year_*` through `five_year_*` columns in `current_comp.csv` and the `1Y RAPM` through `5Y RAPM` rows in `structuredtest.csv`; keep `TimedecayRAPM` and `*_timedecay` columns on their original timedecay source, not as aliases for 1Y DECOMP. `PureRAPM.csv` is still regenerated for legacy compatibility and `PureRAPMPeaks.csv` should be regenerated from it before pushing the `csvs` repo. The app reads `RussDT/csvs@test-branch`, so daily `csvs` publishes should require and push `test-branch`.
- RAPM DECOMP era peak rows should use the native decade files in `/Users/russellthomas/Docs/rapms/master_results`: `weighted_factors_alt3_efg_value_00_09_all_rb_se_a2000_4000.csv`, `weighted_factors_alt3_efg_value_10_19_all_rb_se_a2000_4000.csv`, and `weighted_factors_alt3_efg_value_20_26_all_rb_se_a2000_4000.csv`. Build non-rolling era/support windows with `nba_pipeline/scripts/run_alt3_efg_value_rolling.py --windows ... --force-base --force-components ALL --publish-to-rapms`.
- Production EFG-value display bundles should use six atomic `FIRST_CHANCE` shot-value aliases (`ALT_EFG_RIM_FREQ`, `ALT_EFG_RIM_FG`, `ALT_EFG_MID_FREQ`, `ALT_EFG_MID_FG`, `ALT_EFG_THREE_FREQ`, `ALT_EFG_THREE_FG`) plus `ALT_FT`, display `TOV Value = ALT_TOV_VALUE`, and display direct `SECOND_CHANCE_CLEAN` as `oSC` / `dSC`; derive rim/mid/three/shot totals in the bundle layer. `ALT_FT` can be split further with `ALT_FT_FREQ + ALT_FT_SEVERITY = ALT_FT`, where frequency is average first-chance FT-trip value and severity is trip value above/below that average. `ALT_EFG_RIM/MID/THREE` broad zone solves are audit-only. `ALT_TOV_VALUE` is the point-valued first-chance turnover complement and should equal `ALT_BADPASS_TOV_VALUE + ALT_SCORING_TOV_VALUE`; `ALT_FC_COMPLETION` is legacy audit/glue for the older `FIRST_CHANCE - ALT_EFG_VALUE - ALT_FT` construction. In the active EFG-value bundle, `SECOND_CHANCE_CLEAN` is not the balancing bucket; `ALT_EFG_BASELINE` absorbs the residual needed for the public display to close to total RAPM. Weighted-factor exports from this bundle should use the matching legacy alt3 weighted-factors file only as the player universe / `Latest_Year` source and sign-flip all defensive component columns so positive means good defense.
- For the 2023-24 through 2025-26 first-chance transition/half-court split, `FIRST_CHANCE` carries `Is_FC_Transition_Possession` and `Has_FC_FGA`. The placeholder aliases `FC_TRANSITION_SCORING` and `FC_HALFCOURT_SCORING` use same-sample mode PPP baselines on the opposite mode; the additive aliases satisfy `FC_MODE_MIX + FC_TRANSITION_VALUE + FC_HALFCOURT_VALUE = FIRST_CHANCE.Net_Diff` at row level. Same-sample baselines are computed inside `rapm.py` on the loaded run and respect `--timedecay`.
- RAPM / factor `--age-dummies` uses `dpm_history.csv` only as a player-season age lookup (`nba_id`, `season`, `age`); the age effects themselves are still estimated inside the possession regression, and DARKO `o_dpm` / `d_dpm` are not age-effect targets. `03_run_rapm_analysis.py --age-dummies` applies this to the full daily weighted-factor metric set and writes `_agefe` outputs.
- Backend processors that read Supabase player stats should prefer `SUPABASE_SERVICE_ROLE_KEY` over `SUPABASE_KEY`; the latter may be a publishable key and can fail in the local processing environment.
- Gabriel `merged_playbyplay/old_data` is the current implemented pre-2014 source. Use `nba_pipeline/scripts/build_gabriel_old_pbp.py` to build `1997-2013` RS/PS raw files and the core processed parquets; converted rows intentionally use partial-lineup mode with unknown slots as `0`, and the solver must exclude player id `0`. The converter repairs missing source `team` abbreviations from stable `teamId` mappings before routing descriptions to home/visitor fields. Regular-season and playoff 1997-2013 non-ShotQuality factor parquets exist for `LA_RAPM`, shot-location frequency/FG%, assist, FT premium, and clean first-/second-chance metrics; `FIRST_CHANCE_CLEAN` is a supported Gabriel builder metric, not just a targeted reprocess path. Use `nba_pipeline/scripts/reprocess_metric.py METRIC 1997 2013 --season-types rs|ps --workers N` for targeted historical rebuilds; use `--raw-dir` and `--processed-dir` for validation-first rebuilds outside production paths. If PS source files are unavailable, repair existing PS raw files with `nba_pipeline/scripts/repair_existing_historical_lineup_sides.py` before reprocessing. The alt3 aliases (`ALT_TS`, `ALT_EFG`, `ALT_FT`, `ALT_TOV`, `ALT_BADPASS_TOV`, `ALT_SCORING_TOV`) are derived from `FIRST_CHANCE` rather than separate physical `ALT_*` parquets, and `build_processed_rs_parquet_averages.py` emits them into the audit CSVs from that source. Gabriel historical `RAPM` / `TS` processing uses actual FT outcomes when shooter FT% is missing, without changing the daily pipeline default; `LA_RAPM` is the luck-adjusted surface. For 1997-2000 FT replacement, `nba_pipeline/external/bref_ft_pct_1997_2000.csv` is the local Basketball Reference `nba_id`/FT% lookup, and the shared stats loader uses that regular-season FT% lookup for both RS and PS `RAPM`, `LA_RAPM`, `TS`, and `FT_PREMIUM` processing when Supabase stats are unavailable. Canonical docs: [1997-2013 Historical Parquets](/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/docs/HISTORICAL_1997_2013_PARQUETS.md).
- For Gabriel historical starter repairs, the supplemental NBA stats archive at `/Users/russellthomas/Docs/2026_NBA_PIPELINE/archive (1)` can validate games, teams, players, and substitutions. Use `PlayByPlay.parquet` substitution descriptions plus `PlayerStatistics.csv` active/DNP rows, `Players.csv`, and `TeamHistories.csv`; do not treat old `PlayerStatistics*.csv startingPosition` values as starter flags without auditing counts, because 2009 rows can have more than five nonblank values per team.
- For the preferred long historical core run, use `run_historical_core_weighted_factors.py` with fixed constants instead of estimated season/age nuisance columns: `--fixed-season-effects` subtracts each metric's RS raw target mean by season from both RS and PS rows, and `--age-poly-coefficients` subtracts fixed `curve(age) - curve(27)` row offsets before centering and solving. The usual output suffix is `_rb_fse_agepoly3`.
- The BKN 2020 friend-PBP ingest fixture lives at `nba_pipeline/raw_data/BKN_2020_rs.parquet`; validate schema conversion with `nba_pipeline/scripts/validate_friend_pbp_ingest.py`, which writes isolated artifacts under `nba_pipeline/validation/friend_ingest/`. Current best state: shot-zone metrics are exact, TOV flags are exact on matched rows, RAPM has 14,541/14,542 official rows matched with one official-only goaltend-and-one event, remaining friend-only rows are mostly zero-value team rebounds around blocked-shot jump-ball/admin sequences, and the bigger solve-level gap is lineup parity (`740` of `14,541` matched RAPM rows differ in O/D player sets).
- WNBA clean EFG-value decomposition now lives in `wnba_test/clean_decomp_wnba.py`; it writes clean `FIRST_CHANCE` / `SECOND_CHANCE_CLEAN` CSVs under `wnba_test/Processed/` and weighted-factor display files under `wnba_test/Results/clean_decomp_*`. The clean WNBA total should be solved from `FIRST_CHANCE + SECOND_CHANCE_CLEAN` on matching rows, not from older `RAPM{YY}.csv` files. WNBA mirrors the NBA `ALT_FT_FREQ + ALT_FT_SEVERITY = ALT_FT` trip split and first-chance TOV guard: true TOV rows require no prior first-chance FGA/FT completion and use the average non-turnover first-chance TS/scoring baseline.
- WNBA raw CSV possession parsing should use `prepare_wnba_standard_possession_df` in `wnba_test/process_rapm_wnba.py`; generic team rebounds/turnovers can have `teamid = 0` with the team id in `personid`, and admin rows can separate a miss from its rebound. Do not use only immediate previous-row MISS/REBOUND text for WNBA defensive-rebound possession ends.
- Current WNBA PBP fetch fallback is the static WNBA liveData CDN, not the old stats endpoint: `https://cdn.wnba.com/static/json/liveData/playbyplay/playbyplay_{GAME_ID}.json` plus `boxscore_{GAME_ID}.json`; reconstruct lineups from boxscore starters and substitution events when `stats.wnba.com/stats/playbyplayv3` / `gamerotation` time out. For current-season pulls, use `python wnba_test/fetch_wnba_cdn.py --season YYYY --season-type "Regular Season" --update-player-map`; it scans CDN ids (`102YYxxxxx` RS, `104YYxxxxx` PS), writes `wnba_test/WNBA{YY}.csv`, and appends new player names from boxscores into `wnba_test/player_index_map.csv`. Keep expected WNBA FT/3P stats cohesive with the WNBA stats backend via `wnba_test/wnba_stats_source.py`: it reads `/Users/russellthomas/Docs/wnba/wnba-stats.csv` first, then the backend PBP Stats cache at `/Users/russellthomas/Docs/wnba/data/YYYY_pbp.csv` or `YYYYps_pbp.csv`, and only then pulls `https://api.pbpstats.com/get-totals/wnba` player totals for `FtPoints`, `FTA`, and `Fg3Pct` while writing that backend cache. Do not infer expected FT priors from the CDN raw PBP unless explicitly doing a validation-only audit.
- The current WNBA downstream possession surface is CSV-based, not parquet-based. After WNBA possession-logic changes, rebuild `RAPM`, `TS`, `REB`, `TOV`, `FIRST_CHANCE`, and `SECOND_CHANCE_CLEAN` CSVs for both RS and PS before rerunning five-year clean EFG-value solves; `REB` should use inferred pending miss-side/rebound-side context.
- `SHOOTER_OREB` is a missed-FGA row surface, not a rebound-event surface: keep `Shooter` separate from `O1-O4` non-shooter offensive players, use `D1-D5` for defenders, and expose both team `Offensive_Rebound` and `Self_Offensive_Rebound` targets for shooter miss-recoverability work.
- Repo-owned PBP shot quality lives in `nba_pipeline/scripts/build_pbp_shot_quality.py` and is documented in `nba_pipeline/docs/PBP_SHOT_QUALITY.md`. It parses raw PBP field-goal descriptions for distance/action descriptors, estimates `shot_context_ev` without external ShotQuality `initial_ev`, adds career and player-season shooter talent residuals, then adds a defender-on-court residual allowed adjustment for `shot_quality_with_defense_ev`. The possession-level `RUSSELL_SHOTQUALITY` RAPM surface uses `shot_quality_season_ev` for every FGA, expected FT value for FTs, `0` for other events, and `Def_Diff = -Off_Diff` so the defensive side estimates allowed shot-quality suppression instead of reusing the on-court defender residual.
- `BLOCK_RECOVERY` is a blocked-FGA row surface: denominator is blocked field goals, target is `Block_Recovered_By_Defense`, solver target convention is `Off_Diff = -target`, `Def_Diff = target`, and exported CSVs keep the standard `net_rapm = off - def` display convention so lower `def` values are better defensive block-recovery impact.
- `DUNK` and `DUNK_ASSIST` are made-field-goal event-count standard-possession surfaces built from the shared `is_dunk` flag in `process_rapm_blocks/common.py`; `DUNK_ASSIST` additionally requires the made dunk to have an assist tag. Their parquet targets are `Is_Dunk` and `Is_Dunk_Assist`, with solver convention `Off_Diff = target`, `Def_Diff = -target`.
- Career teammate summaries should combine raw `NBA*.parquet` event-clock deltas for shared minutes with processed `RAPM*.parquet` possessions for ORTG/DRTG/net rating. Generated career-teammate CSVs under `nba_pipeline/results/career_teammates/` are artifacts unless explicitly curated for git.
- Databallr WOWY / lineup net-rating work should use the duplicate raw action-score possession path in `nba_pipeline/scripts/build_clean_lineup_net_rating.py`, not `RAPM*.parquet`; period boundaries are hard splits, foreign-side technical/admin FT points are zero-possession scoring rows for the shooting team, and empty admin-only period-end rows should not become phantom possessions.

## Critical Guardrails

Do not conflate these two layers:
- state attribution layer: `sum_i phi_i(s) = V(s)`
- event attribution layer: `sum_i credit_i(e) = delta_V(e)`

Player state ratings and player event-credit totals are related, but they are not the same total.

If you touch decomposition logic, preserve the additive contract documented in:
- [nba_pipeline/docs/ADDITIVE_OUTPUT_SCHEMA.md](/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/docs/ADDITIVE_OUTPUT_SCHEMA.md)

If you touch overall project status or modeling direction, update:
- [nba_pipeline/docs/DRL_SHAPLEY_BUILD_STATUS.md](/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/docs/DRL_SHAPLEY_BUILD_STATUS.md)

## Current Research State

Current autoresearch work only optimizes the value-model harness in:
- [nba_pipeline/autoresearch_drl_shapley/train.py](/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/autoresearch_drl_shapley/train.py)

It does not yet optimize the full decomposition layer.

The next intended harness is for the additive decomposition contract, using:
- `reconciliation_report.json`
- `state_values.parquet`
- `state_player_phi.parquet`
- `event_player_credit.parquet`
- `player_bucket_totals.csv`
- `player_totals.csv`

## Automation Note

The DRL autoresearch automation is supposed to operate in the real workspace at:
- `/Users/russellthomas/Docs/pbp_rapm`

It should not rely on a detached worktree for this project.

Automation config:
- [automation.toml](/Users/russellthomas/.codex/automations/drl-iterate/automation.toml)
