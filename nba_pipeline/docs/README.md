# NBA Pipeline Docs

- [Predictive VPM field benchmark](VPM_FIELD_BENCHMARK.md) - seven preseason game/season, continuity and veterans-only tests against dated EPM and lagged DARKO/LEBRON/MAMBA; owner-requested 1.10×/1.20× scale diagnostics

This directory is the canonical FreshDocs surface for `/Users/russellthomas/Docs/pbp_rapm`.

## Start Here

- [Full-feature VPM rebuild — audited current preview first](VPM_FULL_FEATURE_HISTORY_REBUILD.md) - 453 development dates independently audited; `full_split_raw` independently frozen, with unresolved paired game gain versus total-raw. One 2026-10-01 mapper and the 2,856-player current board pass independent source, prior, Net/SQ, all168-column/20-array and fresh full-prefix normal checks (`873fc59d`). Show the separate same-date OLD v74/new current O/D/Total and own-prior preview before the 6,108-date rebuild. Spent fits/diagnostics, annual/native adapters, measured storage budget and 19 early-release fits remain pending. Reuse the 2026 mapper; the old 20-fit history phase remains unauthorized. Preserve all successful/failed roots and the 15 GiB reserve. Reader8800 still OLD v74; selected v61/2027/public/live unchanged.

- [Game and season VPM impact architecture](VPM_ACTUAL_IMPACT_ARCHITECTURE.md) - player-game actual-impact ledger, exact Net DECOMP at both grains, season aggregation, rate and total value, pooled game-stat/season-lineup model and publication gates

- [Daily VPM history from1996–97](VPM_DAILY_1997_HISTORY.md) -6,108 dates/9,416,143 estimates/2,856players, independently checked source-adapted annual-trained prior with daily updates and exact components; [browse all dates/players](http://127.0.0.1:8800/?coherent=2#date=2026-10-01), explicit initial backcasts and preserved selectedv61.

- [Daily-trained VPM versus EPM](VPM_DAILY_TRAINED_EPM_COMPARISON.md) - completed same-game pregame-input training replay: 4,772 games, RMSE13.622290 versus EPM13.671691; overall difference unresolved, annual-trained controls better. [Earlier annual-trained daily-deployment comparison](VPM_DAILY_EPM_COMPARISON.md) remains separate.
- [Annual VPM history from1996–97](VPM_1997_HISTORY.md) -30 completed-season boards/46,322 rows, independently verified forward Parent/Net history; explicit1997–99 backcasts, era missingness and uniform own-only priors; selectedv61 unchanged
- [Expanded annual VPM results](VPM_EXPANDED_ANNUAL_RESULTS.md) - completed broad-feature/engineering search and independent audits; no established gain, retain selected v61; [method and reproduction](VPM_EXPANDED_ANNUAL_PRIOR.md)
- [Pregame VPM results and recommendation](VPM_PREGAME_RESULTS.md) - completed registered comparisons and independent audits; retain selectedv61, overall gain unresolved; [operator history and receipts](VPM_PREGAME_BUILD_STATUS.md)
- [Full predictive stat history 1996–2026](FULL_PREDICTIVE_SKILL_HISTORY.md) - completed787558-row/30-season research library; chronological audits, coherent exports, modest cross-stat/FT-association gains and fixed-budget optimizer qualification

- [Databallr merged game-log source audit](DATABALLR_GAME_LOG_SOURCE_AUDIT.md) - working team/player API reads with report=true, 30 Denver seasons, 1997 player counts, modern field inventory, tracking-vintage/FT flags and saved merged samples

- [Pregame VPM prior training plan](VPM_PREGAME_PRIOR_PLAN.md) - owner-authorized game-date predictive stats to actual lineup points; pregame replay, regularized dynamic mapper, separate timing/context/decay tests and historical validation. Implementation underway; selected v61 unchanged

- [Predictive VPM visual three-page explainer](../../../output/documents/predictive_vpm_three_pager.docx) ([PDF](../../../output/pdf/predictive_vpm_three_pager.pdf)) - court and scoreboard fundamentals; central forward-training distinction from EPM's published same-window SPM; recency, exact decomposition, EPM-relative errors and paired uncertainty. Both systems are predictive; training direction alone is not proven to explain gains. No exact feature inventory. Rebuild with `nba_pipeline/scripts/create_predictive_vpm_explainer.py` (python-docx, matplotlib and numpy) and the bundled document renderer

- [Quarterly predictive VPM from 2014](VPM_QUARTERLY_HISTORY.md) - completed 52 quarterly snapshots / 105,268 player-date rows, missing-tracking fallbacks, MPG decision, exact Net priors/finals and audited predictive results
- [Matched selected-model MPG decision](VPM_MPG_DECISION.md) - retain MPG after mapper100 ablation, with audited prediction differences and appearance-count limitations
- [Coherent selected VPM overview](VPM_NET_LANE_DECOMPOSITION.md#coherent-odtotal-overview--october20) - one Offense / Defense / Total leaderboard, grouped Net summary, three-column player lanes and separately labeled priors; same saved selected model

- [Selected VPM team-win projection bridge](VPM_TEAM_WIN_PROJECTIONS.md) - unified donor method, tested minutes/Net aggregation and conditional win engine; current roster/minutes and missing identities remain required before forecasts

- [Defensive Parent calibration](VPM_DEFENSIVE_CALIBRATION.md) - owner-selected mapper100, nine matched tests, conditional intervals and pre-cut groups; selected ratings/skills/Net align, later evidence remains mixed
- [Exact Net lanes for Parent VPM](VPM_NET_LANE_DECOMPOSITION.md) - learned own/entrant/combined priors and final O/D into Net TS/TOV/SC and shot/SQ atoms; clean-source parent, independent closure and forecast audit

- [Estimated-skills daily operations](ESTIMATED_SKILLS_DAILY.md) - one inventory of current/standalone/retained outputs, immutable recipe pins and daily-output preflight; live ingestion/state adapters and scheduler remain unwired

- [Current Parent offensive creation tests](VPM_OFFENSIVE_CREATION_TEST.md) - separate predicted assist, assistedness and playtype-points additions; retained current, approximate TSA disclosure, predictive evidence, audited ratings/priors and owner-retained auxiliary stats in the predictive skills library

- [Current Parent mid/three SQ ablation](VPM_CURRENT_SQ_ABLATION.md) - fixed-prior no SQ/mid/three/both forecasts, positive overall game increments, unchanged D3 48000 and audited four-way results

- [Role-aware defensive prior](VPM_ROLE_AWARE_PRIOR.md) and [all skills in both O/D mappers](VPM_UNSPLIT_FEATURE_PRIOR.md) - matched current-Parent tests, retained split control, seven-test/native evidence and one combined ratings/prior reader

- [Rim defended makes in forward VPM](VPM_DEFENDED_MAKES.md) - annual DFGA-style OU fits, matched single-feature test, tiny spent forecast change, higher Jokic D, retained control and audited comparison

- [Expanded forward priors for repaired VPM](VPM_REPAIRED_FORWARD_EXPANDED.md) - dated skills predict future lineup scoring; wider learned priors, explicit MPG test, old-label feature control, seven-test evidence, audited ratings and predictive box/advanced input inspection

- [Fixed fouls and contested-DREB replacement](VPM_FIXED_FOULS_CONTESTED_DREB.md) - completed Parent VPM rerun, exact replacement, previous-repaired comparison, audited ratings and skill-only/blended prior views, plus small future-possession gain; game improvements remain inconclusive

- [VPM defensive overlap and contested rebounds](VPM_DEFENSIVE_FEATURE_OVERLAP.md) - measured feature correlations, contested-DREB candidate scope, and corrected all-zero shooting-foul extraction; original saved ratings remain pre-repair; see the separate corrected rerun above

- [VPM model map: component versus parent](VPM_MODEL_MAP.md) - the two architectures, what the current ratings page actually contains, and correction of the parent-as-component launch recommendation

- [VPM launch defensive bank and ratings review](VPM_LAUNCH_DEFENSE.md) - completed v51: separate rim/disruption/foul/rebound and component tests, current585-player comparison, dated-FT launch-review recommendation and independent seven-test audits; no public deployment

- [Extended-defense VPM with dated FT and context](VPM_EXPECTED_FT_CONTEXT.md) - completed v50: expected FT gives a tiny actual-point lineup gain; expanded defense and home/rubberband do not improve game forecasts; minimum-MSE tuning and independent seven-test audits

- [Per-game matchup defensive VPM research](VPM_MATCHUP_DEFENSE_FEATURES.md) - completed2018–26 backfill, audited assignment/fresh-BAM features and seven-test comparisons; small unresolved assignment gain, BAM selects null
- [Matchup game-log inventory and live pilot](RIM_DEFENSE_MATCHUP_SKILLS.md) - 11,521 games/2,169,558 native rows, field-specific coverage and rejected dated-checkpoint audit

- [Previous/new VPM blends](VPM_LONG_SPM_BLENDS.md) - v46 five fixed weights, seven tests;50/50 improves historical stress game errors while development retains previous VPM

- [Long-sample SPM implementation and results](VPM_LONG_SPM_TRAINING.md) - v44 six-year calibration, forward selection, seven stress tests, and independently validated research challenger
- [Long-sample SPM build plan](VPM_LONG_SPM_PLAN.md) and [source audit](VPM_LONG_SPM_SOURCE_AUDIT.md)

- [VPM versus EPM cohort audit](VPM_EPM_COHORT_AUDIT.md) - repaired-control paired diagnostics across52 cells; team changers are the leading weakness, while entrant and movement-shooter substitutions favor VPM

- [Forward-trained prior implementation](VPM_FORWARD_PRIOR_TRAINING.md) - v43 exact training through the lineup solve, nested chronological game forecasts, seven-test comparison and rejected first candidate

- [Forward-trained VPM architecture](VPM_FORWARD_TRAINED_ARCHITECTURE.md) - recommended skill means, calibrated precision and lineup residual model; train through the solve against future actual games, with preserved controls

- [Zero-prior VPM field benchmark](VPM_NO_PRIOR_FIELD_BENCHMARK.md) - matched repaired SQ model with zero player-impact priors; seven tests and retained shooter-talent definition
- [VPM ShotQuality temporal repair](VPM_SQ_TEMPORAL_REPAIR.md) - owner-authorized past-only EV reconstruction, cutoff invariance and paired field rerun; original artifacts preserved

- [Monthly VPM history from 2019](VPM_MONTHLY_HISTORY.md) - completed 93-cut research history, 110,458 player-months, chronological skill fits, entrant-only priors and prediction-preserving reference

- [Rookie and G League VPM priors](VPM_ENTRANT_PRIORS.md) - dated source recovery, identity gates, draft/statistical prior challengers, and own-SQ integration audit

- [SQ extended own-defense VPM board](https://russ-vpm-sq-board.russdt.chatgpt.site) - private ChatGPT Site: 582-player September14 board, 25 estimated skills, and separately labeled repaired rim diagnostics; source checkout is sibling `vpm-sq-board-site`

- [Rim cumulative-count repair](RIM_DEFENSE_MATCHUP_SKILLS.md) - recovered 2020–2022 observations with appearance/trade gates and versioned frozen-parameter feature refresh
- [Shared skills + TD factors](../../../vpm/vpm/docs/OWN_FACTOR_FEATURE_PARENT.md) - parent challenger with53 inputs per side and shared33 control
- [All33 own-skill components](../../../vpm/vpm/docs/OWN_ALL_COMPONENTS.md) - owner-requested unrestricted feature union across all20 component-side prior maps
- [Component VPM reader](VPM_COMPONENT_SITE.md) - component boards, exact-ID full names, and Expanded + fouls vs predictive EPM comparison
- [Foul-drawing SQ parent](../../../vpm/vpm/docs/OWN_FOULS_PARENT.md) - owner-requested personal/offensive foul and charge inputs, measured-only dated integration
- [Expanded own-skill SQ parent](../../../vpm/vpm/docs/OWN_TRACKING_PARENT.md) - 15 offense / 9 defense inputs, dated tracking states and matched development comparison
- [VPM Feature Coverage Audit](VPM_FEATURE_COVERAGE_AUDIT.md) - actual 19 prior features, later-starting histories, rejected rim archive fields, stale states and chronological rebuild requirements

- [xPM Research Notes](./XPM_RESEARCH_NOTES.md) - George Berry methods, benchmark scorecard, entrant-prior gaps, and owner-requested leakage-free 2020–2026 simpler SQ-parent VPM comparison

- [VPM Consolidation Plan](./VPM_CONSOLIDATION_PLAN.md) - September 20 plan: one product repository, frozen Parent baseline and rebuilt Components challenger, shared college/G-League priors, uncertainty, evaluation and reader; implementation not yet started

- [Own-Skill O/D Parent Integration](../../../vpm/vpm/docs/OWN_SKILL_PARENT_INTEGRATION.md) - working neutral-feature adapter, 582-player prior-centered parent, fixed development comparison and remaining actual-points/game gates

- [Predictive VPM Build Path](./PREDICTIVE_VPM_BUILD_PATH.md) - current research priority: own dated skills, O/D priors, parent control, component increment, and future-game EPM comparison
- [Learned low-experience fallback](../../../vpm/vpm/docs/OWN_LEARNED_FALLBACK.md) - jointly learned O/D centers, fixed minute weights, current board and inconclusive forward comparison
- [Own-skill low-minute prior](../../../vpm/vpm/docs/OWN_LOW_EXPOSURE_PRIOR.md) - implemented fixed zero-fallback challenger, coherent shooting, current board and inconclusive development comparison
- [Own-skill SQ parent](../../../vpm/vpm/docs/OWN_SKILL_SQ_PARENT.md) - frozen SQ EV plus mid/three residuals, borrowed penalties, paired development results and current board
- [Own-skill VPM HTML report](../results/estimated_skills/own-skill-vpm.html) - local research reader: O/D parent, 582-player board, fitted priors, development comparisons, and component next steps

- [Next-Game Estimated Deflections](./NBA_DEFLECTIONS_NEXT_GAME.md) - owner-selected next-game per-100 objective, with the original longer-horizon filter preserved as a comparator

- [Next-Game Tracking Filters](./NBA_TRACKING_NEXT_GAME.md) - next-game per-100 OU/random-walk filters for potential assists, drives, possession minutes, contested shots, defended FGA, and contested DREB on the measured-only tracking panel; same split as deflections

- [Next-Game Fouls-Drawn Filters](./NBA_FOULS_DRAWN_NEXT_GAME.md) - CDN live-feed foul ledger naming the fouled player (2019-20 to 2025-26), three-way box-confirmed reconciliation, and next-game OU/random-walk filters for offensive fouls drawn, charges drawn, and personal fouls drawn
- [NBA Predictive Box-Score Stats Registry](./NBA_PREDICTIVE_BOX_STATS_REGISTRY.md) - one curated registry (`config/estimated_skills_registry.json`) of the 28 predictive box-score stats across all 20 estimated-skills lanes: what is on the table, the dev-selected model, receipt-backed evidence, readable state notes, and a generated status/values/evidence snapshot with a self-contained reader

- [Estimated Deflections and Testing Harness](./NBA_ESTIMATED_DEFLECTIONS.md) - observed-hustle predictive rates, frozen development/stress split, and exact forward-target contract

- [NBA Tracking Game-Log Backfill](./NBA_TRACKING_BACKFILL.md) - conservative potential-assist and DFGA repair for estimated-skills preparation; preserves observed data and labels estimates/omitted-zero inferences

- [zTS Year-to-Year Stability](./ZTS_STABILITY.md) - matched 2006–2025 production-rate audit: smaller annual movement, mixed correlation, and role-change diagnostics
- [EPM Contract-Aging Data Handoff](./CONTRACT_AGING_HANDOFF.md) - private normalized RS/EPM inputs, census gates, existing aging research, and proposed chronological forecast validation
- [NBA Analytics Project Portfolio](./PROJECT_PORTFOLIO.md) - current priorities, project states, and work-in-progress limits
- [Model Assurance Ledger](./MODEL_ASSURANCE_LEDGER.md) - one-page status, evidence, and promotion gates for every major model surface
- [RAPM Table Catalog and Calculation Queue](./RAPM_TABLE_CATALOG.md) - one source of truth for current, research, diagnostic, and legacy RAPM offerings plus missing rebuilds
- [RAPM Table Registry](./RAPM_TABLE_REGISTRY.md) - checked-in API-oriented series/mechanism definitions, current no-prior TD550 mechanism board, generated artifact status, and restartable queues
- [Mechanism RAPM Future-Alpha Validation](./MECHANISM_FUTURE_ALPHA_VALIDATION.md) - trailing-five TD550 fits scored on later non-overlapping two-season possession targets, with paired one-SE selection
- [Historical Mechanism Backfills](./HISTORICAL_MECHANISM_BACKFILLS.md) - exact modern STOP history plus separately labeled 1997-2026 transition and playtype-compatible backcasts
- [Codex Scratchpad](./Codex_Scratchpad.md) - short-term session memory only
- [Contributor Onboarding](./CONTRIBUTOR_ONBOARDING.md)
- [Artifact Manifest](./ARTIFACT_MANIFEST.md)
- [Repo Cleanup Plan](./REPO_CLEANUP_PLAN.md)
- [Local Supabase Database Catalog](./SUPABASE_DATABASE_CATALOG.md) - searchable local inventory of the Databallr schemas, tables, columns, relationships, sizes, and access controls
- [Supabase Results Publishing](./SUPABASE_RESULTS_PUBLISHING.md) - versioned, RLS-protected result registry and one-command site publishing workflow
- [Pipeline Quickstart](./PIPELINE_QUICKSTART.md)
- [RAPM Methodology](./RAPM_METHODOLOGY.md)
- [21st Century RAPM](./THIRTY_YEAR_AGE_ADJUSTED_RAPM.md) - age-adjusted
  1997-2026 career O/D parent, negative-LAKER fringe prior, and validated
  adaptive young-player precision challenger
- [Clock-Aware Rubberband Adjustment](./RUBBERBAND_ADJUSTMENT.md)
- [Rubberband Era Stability](./RUBBERBAND_ERA_STABILITY.md)
- [Home-Court RAPM Adjustment](./HOMECOURT_ADJUSTMENT.md)
- [Playoff Handling In Multi-Year RAPM](./PLAYOFF_HANDLING.md)
- [NBA Playoff Droppers: Translation, Archetypes, and Exposure](./PLAYOFF_DROPPERS.md) — walk-forward player-history, offensive/defensive archetype, environment-change, and rotation-survival baseline
- [NBA Event-Enrichment Contract](./NBA_EVENT_ENRICHMENT_CONTRACT.md)
- [Dunk and Dunk Assist Handoff](./dunk.md)
- [PBP Shot Quality](./PBP_SHOT_QUALITY.md)
- [ShotQuality Three-Point Difficulty](./SQ_THREE_POINT_DIFFICULTY.md)
- [Visual Shot Quality Explainer](./shot-quality-explainer.html)
- [Second-Chance Opportunity Prior](./SECOND_CHANCE_OPPORTUNITY_PRIOR.md)
- [Forward SQ + Second-Chance Release Value](./SQ_SECOND_CHANCE_FORWARD.md)
- [Shot-Clock SQ and Grenade Burden](./SQ_SHOT_CLOCK_CHALLENGER.md)
- [Season-Forward SQ: EPM and Closest-Defender Features](./SQ_WALK_FORWARD_CHALLENGER.md)
- [Block-Aware Rim ShotQuality Challenger](./SQ_RIM_BLOCK_CHALLENGER.md)
- [SQ Defender Providers: NBA Assignments vs Rich Geometry](./SQ_DEFENDER_PROVIDER_COMPARISON.md)
- [Primary-Defender Net-DECOMP Split](./PRIMARY_DEFENDER_NET_DECOMP.md) —
  five-year role diagnostic and compact rounded reader export
- [ShotQuality-Era Challenger Models](./SHOTQUALITY_CHALLENGER_MODELS.md)
- [Historical Transition Backfill](./HISTORICAL_TRANSITION_BACKFILL.md)
- [Public Shot-Quality Methods](./SHOT_QUALITY_PUBLIC_METHODS.md)
- [Historical Shot-Quality Transfer](./HISTORICAL_SHOT_QUALITY_TRANSFER.md)
- [Russell ShotQuality v2 Historical Bridge](./RUSSELL_V2_HISTORICAL_BRIDGE.md) —
  gated 1997-2015 EV targets for a historical no-prior VPM, with native v2
  unchanged from 2016 onward
- [Retrospective Historical ShotQuality v4](./HISTORICAL_SHOT_QUALITY_RETROSPECTIVE_V4.md)
- [Historical ShotQuality Safe Crossfit](./HISTORICAL_SHOT_QUALITY_SAFE_CROSSFIT.md)
- [Shot-Clock Estimation](./SHOT_CLOCK_ESTIMATION.md)
- [Time-Decay Overall Playtype-Mix RAPM](./TD_OVERALL_PLAYTYPE_MIX_OTS.md)
- [rZTOV: Mix-Adjusted Turnovers](./RZTOV.md)
- [NBA Box-Estimated rZTOV Backcast](./NBA_RZTOV_BOX_BACKCAST.md)
- [Historical Own-Scoring and Ball-Security Value](./HISTORICAL_SCORING_SECURITY.md) — observed Databallr/Synergy role for 2006-2026, calibrated PBP backcast only for 1997-2005
- [Historical Scoring Impact Leaderboard](./historical-scoring-impact.html) — scoring-first view excludes bad-pass turnover from the headline while retaining it and full scoring security as sortable columns
- [rZTOV Explorer](./rztov-explorer.html) — player-first reader: verdict, lane bullets, season chart, sticky leaderboard/map, URL-shareable state
- [Scoring Impact + rZTOV: Databallr Integration Handoff](./SCORING_RZTOV_DATABALLR_HANDOFF.md) — dataset contract, display rules, the 34-check reconciliation receipt (`audit_scoring_rztov_reconciliation.py`), rebuild order, and the one research column `playtype_mix_rapm` (full own + teammate lineup diet, two-season 550-day half-life, 1997-2026; descriptive, never added to SV; stability receipt from `audit_playtype_mix_stability.py`)
- [Lineup Turnover Board](./LINEUP_TOV_BOARD.md) — two-season TD550 lineup turnovers with the player on offense, split exactly into own / teammates / unassigned team turnovers per lane, priced at a kept possession's net first-chance value; transparency view beside the box rZTOV turnover values (never added), with per-season pass-through slopes; team turnovers absent from 1997-2013 source rows
- [Prior-Informed Time-Decay TOV RAPM](./TOV_PRIOR_RAPM.md)
- [Scoring at 20 Playground](./scoring-at-20-playground.html)
- **Impact to Winning Player Skill Lab** — the sibling
  `research/from-impact-to-winning` routes `/skills/rztov`, `/skills/zts`, and
  `/skills/scoring` provide the themed public organization. Refresh its checked-in
  rZTOV, scoring, and Databallr production zTS snapshots with
  `bun run refresh:metrics` from that repo.
- [Possession Start Report](./POSSESSION_START_REPORT.md) — relative-PPP chart with actual rates/counts, made rim/mid/three splits, separate blocked/unblocked rim misses, one missed-three bar, and own missed-FGA continuation companion with shooter-position filters and paired own-continuation and opponent-PPP comparisons, make/miss weighting, and a first-chance-only filter.
- [End-Game ATO PPP](./ENDGAME_ATO_PPP.md)
- [Career Teammate Summary](./CAREER_TEAMMATE_SUMMARY.md)
- [DARKO Teammate Help and The Help Index](./DARKO_TEAMMATE_HELP.md)
- [Title Lift over Contract](./TITLE_LIFT_OVER_CONTRACT_DATA.md) — salary-peer
  title simulation, era-local EPM models, and the 1998-2026 lagged-DARKO
  research sensitivity
- [Playoff injury title lift](./PLAYOFF_INJURY_TITLE_LIFT.md) — year-by-year participation-scaled lift, nested leave-one-season-out nonplayoff estimates, roster coverage, and separate injury-counterfactual proposal
- [EPM Game and Season Value Calibration](./EPM_GAME_VALUE.md)
- [Alternate Clean 3-Factor](./ALTERNATE_CLEAN_3_FACTOR.md)
- [Public Eight-Component DECOMP](./PUBLIC_DECOMP.md)
- [Net SQ DECOMP rolling archive](./NET_SQ_DECOMP_ROLLING.md) - the two rolling 1Y-5Y lineages, the live td550 Supabase archive, the season-rollover fixes, and why 2027 is blocked upstream
- [TD550 Rolling Supabase Archive](./TD550_ROLLING_SUPABASE.md) - the 1997-2026 rolling Net SQ DECOMP publication and consumer contract
- [FT Fallback Repair](./FT_FALLBACK_REPAIR.md) - targeted correction of affected rolling Net tables and cutoff-safe no-prior snapshots
- [VPM Reproduction and Recovery Status](./VPM_REPRODUCTION.md)
  - includes the exact annual RS parent ORAPM/DRAPM target recipe and the
    sibling 2018-2023 season-forward parent-prior/stress receipts
- [Alt3 EFG-Value Weighted-Factor Intern Explainer](./alt3_efg_weighted_factor_explainer.html)
- [WNBA Clean EFG-Value Decomposition](./WNBA_CLEAN_DECOMP.md)
- [WNBA Gabriel Ingest and Initial Metric Benchmark](./WNBA_GABRIEL_INGEST_AND_BENCHMARK.md)
- [WNBA Metric Build Status](./WNBA_METRIC_BUILD_STATUS.md)
- [WNBA Daily Metric Update](./WNBA_DAILY_UPDATE.md) (includes the atomic Supabase-backed Databallr `wnba_vpm` publish, direct-TOV semantic reallocation, and selected explanatory TOV alpha via `wnba_test/prepare_wnba_vpm_publish.py --upload`)
- [WNBA Play-by-Play Shot Quality](./WNBA_PBP_SHOT_QUALITY.md)
- [WNBA Predictive Parent](./WNBA_PREDICTIVE_PARENT.md) (actual-FT, past-only own skills, direct O/D prior, SQ challenger, and forward rating-spread audit)
- [WNBA Daily Net-DECOMP Alpha Tuning](./WNBA_DAILY_NET_ALPHA_TUNING.md)
- [WNBA RTSS-to-PBP ShotQuality Student](./WNBA_RTSS_PBP_SHOT_QUALITY.md)
- [WNBA Playtype / Shot-Role Difficulty](./WNBA_PLAYTYPE_DIFFICULTY.md)
- [WNBA RTSS Rich-Tracking Feature Audit](./WNBA_RTSS_FEATURE_AUDIT.md)
- [WNBA RTSS Component-Prior Research](./WNBA_RTSS_PRIOR_RESEARCH.md)
- [WNBA RTSS Shooter, Passer, and Presence RAPM](./WNBA_RTSS_PLAYMAKING_RAPM.md)
- [WNBA Box-Estimated rZTOV](./WNBA_RZTOV.md)
- [WNBA Predictive Program Plan of Record](./WNBA_BETTING_PROGRAM.md)
- [WNBA Pre-Tip Forecast Log Operations](./WNBA_FORECAST_LOG.md)
- [WNBA Predictive Box Panel (Kalman + DE)](./WNBA_KALMAN_BOX.md)
- [WNBA Estimated Skills (zone Kalman + fusion gate)](./WNBA_ESTIMATED_SKILLS.md)
- [WNBA Predictive Net-DECOMP VPM (Kalman-prior research lane)](./WNBA_PREDICTIVE_NET_VPM.md)
- [NBA Estimated Skills (PBP Kalman panel; v4 next-game OU + context candidate; next-game lab rounds 1-3; family map of all 20 NBA estimated-skill lanes)](./NBA_ESTIMATED_SKILLS.md)
- [WNBA RTSS One-Shot Confirmation Prereg](./WNBA_RTSS_CONFIRMATION_PREREG.md)
- [WNBA Player-Props Prereg](./WNBA_PROPS_PREREG.md)
- [WNBA Cold-Start / Coverage Prereg](./WNBA_COLD_START_PREREG.md)
- [Age-Curve RAPM](./AGE_CURVE_RAPM.md)
- [Age-Dummy RAPM](./AGE_DUMMY_RAPM.md)
- [21st Century RAPM: 1997-2026 Age-Adjusted O/D](./THIRTY_YEAR_AGE_ADJUSTED_RAPM.md)
- [Season Intercepts](./SEASON_INTERCEPTS.md)
- [1997-2013 Historical Parquets](./HISTORICAL_1997_2013_PARQUETS.md)
- [Modern Raw-Lineup Repair (2014+)](./MODERN_LINEUP_REPAIR.md)
- [Historical Pre-2014 Ingest Plan](./HISTORICAL_PRE2014_INGEST_PLAN.md)
- [Historical Video RAPM Paper Notes](./HISTORICAL_VIDEO_RAPM_PAPER_NOTES.md)
- [Friend PBP Ingest Validation](./FRIEND_PBP_INGEST_VALIDATION.md)
- [Clean Lineup Net-Rating Possessions](./LINEUP_NET_RATING_PARQUET.md)
- [Additive Output Schema](./ADDITIVE_OUTPUT_SCHEMA.md)
- [DRL Shapley Build Status](./DRL_SHAPLEY_BUILD_STATUS.md)
- [Git Backup Strategy](./GIT_BACKUP_STRATEGY.md)
- [TOV Decomposition](./tov-decomposition.md)

## Additional References

- [Estimated play-type scoring RAPM](PLAYTYPE_SCORING_RAPM.md): creation/finishing/transition/spacing FGA-only additive research estimates, 2022–2026.

- [RAPM Methodology Review](./RAPM_Methodology_Review.md)
- [Best Public Impact Metric 2026](./BEST_PUBLIC_IMPACT_METRIC_2026.md)
- [Alt3 EFG oRAPM Breakdown Infographic](./visuals/alt3_efg_orapm_breakdown.md)
- RAPM DECOMP Paper: [TeX](./papers/additive_rapm_decomposition.tex), [HTML](./papers/additive_rapm_decomposition.html), [PDF](./papers/additive_rapm_decomposition.pdf)
- RAPM DECOMP Legacy Draft: [TeX](./papers/additive_rapm_decomposition_legacy.tex), [HTML](./papers/additive_rapm_decomposition_legacy.html), [PDF](./papers/additive_rapm_decomposition_legacy.pdf)
- [2026 Lineup Parser Investigation](./LINEUP_PARSER_INVESTIGATION_2026.md)
- [Time Decay RAPM Handoff](./timedecay_rapm_handoff.md)
- [FT Premium Handoff](./FT_PREMIUM_HANDOFF.md)
- [Weighted Factors Galaxy](./WEIGHTED_FACTORS_GALAXY.md)
- [Five-Man Fit Lab Recovery](./FIVE_MAN_FIT_LAB.md)
- [DTOV / RFTOV Investigation](./DTOV_RFTOV_Investigation_Report.md)

## Maintenance Rules

- Update docs in the same pass as behavior or methodology changes.
- Keep this index current when adding or renaming canonical docs.
- Keep durable rules in root `AGENTS.md` / `CLAUDE.md`; keep only short-term notes in `Codex_Scratchpad.md`.

Recent methodology notes:
- `scripts/build_epm_game_value.py` joins the final 2025-26 EPM snapshot to
  exact raw-PBP minutes and lineup-possession exposure, calibrates season
  points-to-wins on 2014-2025 regular seasons, and fits the neutral
  expected-margin win curve on 2014-2022 with 2023-2025 held out. The current
  conversion is `0.0297496` wins per point and logistic `k=0.144482`; probit is
  statistically indistinguishable, while a clipped global linear probability
  loses held-out log loss. Keep Game +/- linear, use the bounded curve for
  NET WIN%, and label `-2 EPM` as an uncalibrated replacement assumption. See
  [EPM Game and Season Value Calibration](./EPM_GAME_VALUE.md).
- `scripts/train_sq_rim_block_challenger.py` builds a season-forward immediate
  rim-EV challenger with a separate block-probability diagnostic. It joins raw
  NBA block labels/opposing fives to ShotQuality pre-release geometry, uses
  whole-game-shifted shooter/assigned-defender histories plus date-aware EPM
  rim and block skills, and keeps current outcomes out of features. The
  `initial_ev`-anchored direct model lowers held-out points MSE by `1.09%` in
  2024-25 and `3.38%` in 2025-26; block features add within both unanchored
  forward ablations. These inspected folds are development evidence only.
  Current output is immediate shot value and does not yet price the elevated
  offense-retention/OREB value after blocks; see
  [Block-Aware Rim ShotQuality Challenger](./SQ_RIM_BLOCK_CHALLENGER.md).
- The sibling ShotQuality repo now exposes a derived 2025-26 NBA PBP coordinate
  backup under `court_analysis/coordinate_repair/`. Fresh provider downloads
  reproduce the collapsed `shot_x` / `shot_y` defect. The layer preserves both
  raw sources and match diagnostics, repairs 107,002 of 120,316 FGA in 673
  flagged games, and leaves ambiguous/unmatched shots spatially unusable.
  Consecutive-date regular-season candidates require a unique final-score
  match with clock-first PBP ordering; four Play-In records stay outside the
  regular-season index by design. Use it for zones and court maps, not as
  optical defender geometry.
- `scripts/train_sq_walk_forward_challenger.py` freezes model coefficients on
  prior seasons and shifts shooter/defender outcome priors by whole game before
  testing 2024-25 and 2025-26. ShotQuality closest-defender geometry clearly
  improves forward midrange and three-point loss where present; the complete
  context+EPM+geometry+identity model has lower point-MSE than `initial_ev` in
  all four segment-season cells, with three game-cluster intervals entirely
  below zero. The NBA matchup archive is a separate 2016-17-through-2024-25
  identity lane and can contain multiple assigned defenders per shot. Its
  strict 2024-25 three-point arm is neutral versus context+EPM and clearly
  worse than rich SQ geometry. Opposing five-man as-of D-EPM also fails to
  improve three-point loss despite 97.80% complete test coverage; see
  [Season-Forward SQ](./SQ_WALK_FORWARD_CHALLENGER.md) and
  [SQ Defender Providers](./SQ_DEFENDER_PROVIDER_COMPARISON.md).
- `scripts/fetch_epm_data.py` restores gitignored, date-stamped Dunks & Threes
  `epm-all` snapshots from `EPM_API_KEY`, validates the predictive-rate schema,
  and writes a checksum/provenance manifest with separate requested and
  provider-effective dates. EPM `p_fg3pct` / `p_fgpct_mid` are candidate
  as-of shooter-talent priors, not context-only SQ fields; raw provider
  snapshots remain outside public DECOMP and direct VPM inputs. See
  [VPM Reproduction and Recovery Status](./VPM_REPRODUCTION.md).
- fetch_monthly_epm_title.py maintains the private 2001-present monthly
  EPM-only archive for The Help Index championship model. The current title
  surface freezes the last snapshot before each postseason, weights it by the
  actual normalized playoff rotation, and fits one conditional title
  distribution across the playoff field. Its common neutral baseline is the
  season-average playoff rotation, not five 0-EPM teammates. See
  [EPM Championship Help](./EPM_CHAMPIONSHIP_HELP.md) and
  [Title Lift over Contract: data and method](./TITLE_LIFT_OVER_CONTRACT_DATA.md).
  The research module also retains a date-correct dynamic prior-informed
  playoff-residual prototype; its one-standard-error selection remains the
  null, so it does not replace the public eight-year sensitivity model.
- NBA matchup defender assignments and ShotQuality closest-defender assignments are separate providers. `matchup_fga` currently covers 2017-2025; the recovered rich SQ defender/geometry store is 2024-25 only, with primary identity on about 76% of FGA and no native secondary ID. The transitional SQ enrichment writes `sq_primary_defender_id` plus provider/match diagnostics, collapses only exact duplicate payloads, and leaves provider/raw-key conflicts explicitly ambiguous instead of using keep-first joins. Centered primary-role models must publish exposure-weighted on-ball/off-ball contributions that close exactly to the defensive parent. See [NBA Event-Enrichment Contract](./NBA_EVENT_ENRICHMENT_CONTRACT.md).
- `scripts/compare_sq_second_chance_forward.py` now has two strict rolling shot-release tests for the shooter/zone OREB layer. In 2024-25, adding shooter × zone to identical shot-context/exact-lineup continuation improves combined points MSE by `0.000142` for context+EPM and `0.000140` for geometry+identity, with intervals just below zero; the full geometry lane is tied with the fair provider benchmark. In the 2025-26 replication (`130,229` releases), the pooled shooter-specific gains remain directional but cross zero, while the full geometry lane beats provider `initial_ev` plus the same continuation by `0.004628` MSE, interval `[-0.006233, -0.003059]`. Contextual continuation is robust in both seasons. The pooled held-out zone baselines now precede player interpretation: long twos carry `23.7%` OREB probability after a miss and `0.171` expected second-chance points per initial attempt, versus `0.208` for short midrange and `0.183` for threes; these are descriptive zone averages, not causal same-state zone effects or exact-zone post-rebound PPP. The Embiid long-two role-vacancy effect shrinks to `-0.13` OREB percentage points and `-0.0010` points per attempt in 2025-26, so it is descriptive rather than a stable confirmed player penalty. See [Forward SQ + Second-Chance Release Value](./SQ_SECOND_CHANCE_FORWARD.md).
- `scripts/train_sq_shot_value_challenger.py` builds leakage-controlled 2024-25 midrange/three challengers against `initial_ev`; the nested midrange logit and nonlinear three model both win on game-group OOF loss with game-cluster intervals below zero. `scripts/train_sq_transition_classifier.py` builds historical-safe transition labels from ordinary PBP and retains strong no-refit 2025-26 discrimination. `scripts/score_historical_transition.py` scores the frozen 97-feature contract over 6,328,268 RS/PS shots from 1997-2026, uses game-cross-fitted calibration for labeled 2025-26 rows and explicitly retrospective pooled calibration for earlier unlabeled seasons, normalizes both Gabriel total-second and modern second-component clocks, and writes per-file/era/feature drift audits under `results/sq_transition_backfill/`. `scripts/build_historical_shot_quality_transfer.py` distills the rich/modern teacher into historical PBP features; the frozen portable v3 now has a resumable 60/60-file 1997-2026 RS+PS backfill over 4,267,890 midrange/threes plus 100%-coverage identical-row competition audits. V3 wins every 2024-26 era/bucket comparison but loses 14/40 historical comparisons overall. `scripts/build_russell_v2_historical_bridge.py` uses a later native-v2 overlap gate to emit 2,583,850 historical 1997-2015 targets: the v2-distilled stack passes for midrange, while threes retain the portable-v3 fallback after the stack fails held-out Brier. Native v2 remains unchanged for 2016-2026, safe-crossfit remains audit-only, and a pooled 1997-2026 solve requires season effects because the 2015/2016 raw probability seam is nonzero. The separate retrospective v4 stack has the best point Brier in all eight era/bucket cells and paired intervals excluding zero in seven, while its failed temporal-transfer cells are retained as a regime-drift warning rather than hidden. The stricter safe-crossfit harness rebuilds every outcome-derived encoding inside each outer game fold; over all history it beats v3 by `0.002931` Brier overall and beats canonical defense on threes by `0.000325`, but still trails canonical defense on midrange by `0.002417`. See [ShotQuality-Era Challenger Models](./SHOTQUALITY_CHALLENGER_MODELS.md), [Historical Transition Backfill](./HISTORICAL_TRANSITION_BACKFILL.md), [Historical Shot-Quality Transfer](./HISTORICAL_SHOT_QUALITY_TRANSFER.md), [Russell ShotQuality v2 Historical Bridge](./RUSSELL_V2_HISTORICAL_BRIDGE.md), [Retrospective Historical ShotQuality v4](./HISTORICAL_SHOT_QUALITY_RETROSPECTIVE_V4.md), and [Historical ShotQuality Safe Crossfit](./HISTORICAL_SHOT_QUALITY_SAFE_CROSSFIT.md).
- `scripts/build_shot_clock_estimates.py` estimates per-FGA shot-clock time from raw PBP by sorting events by `game_id`, `period`, `clock_sec` descending, then `event_num` ascending. This clock-first ordering is required because corrected NBA PBP events can be appended with late event numbers at an earlier clock. `scripts/analyze_turnover_by_shot_clock.py` reuses that state machine for FGA/turnover/FT-trip decision endpoints and reports event location, survival-style turnover hazard, actual TSx2 by clock bin, and continuous-grid remaining possession value split into direct first-chance and second-chance continuation value. `scripts/analyze_shot_clock_difficulty_ots.py` cross-fits clock expectations by game and produces player shot-clock difficulty plus exactly adjusted TS%; it also exports the four zTS role buckets and tests clock incrementally over playtype difficulty. The two-transition evidence says zTS absorbs most of the clock gain, so neither clock surface is a production promotion. `scripts/analyze_pbp_playtype_difficulty_ots.py` builds a six-bucket, outcome-invariant description proxy, prices each player's current shot mix from the immediately prior season, and compares it with Synergy on exact-common future-oTS rows. `scripts/analyze_td_overall_playtype_mix_ots.py` adds the usage-centered overall lineup-mix parent: its exact own/teammate split closes, but its `-0.0040 [-0.0233,+0.0136]` RMSE increment after own difficulty does not clear the future-oTS gate. Raw-versus-calibrated timing sensitivity remains explicit because PBP time is whole-second and the calibrated clock is an estimate. The generated artifacts live under `results/shot_clock/`, `results/turnover_shot_clock/`, `results/shot_clock_difficulty/`, `results/pbp_playtype_difficulty/`, and `results/td_overall_playtype_mix/`; methodology is documented in [Shot-Clock Estimation](./SHOT_CLOCK_ESTIMATION.md), [Shot-Clock Difficulty and Clock-Adjusted TS](./SHOT_CLOCK_DIFFICULTY_OTS.md), [PBP-Only Playtype Difficulty and Future oTS](./PBP_PLAYTYPE_DIFFICULTY_OTS.md), and [Time-Decay Overall Playtype-Mix RAPM](./TD_OVERALL_PLAYTYPE_MIX_OTS.md).
- `scripts/build_rztov_explorer.py` builds rZTOV, `(actual turnovers - allowance) / plays` per 100 offensive possessions, where self-created attempts, assisted attempts, and potential assists each earn a break-even rate. The three prices come from one joint no-intercept alternating-least-squares solve per trailing window against three count-basis solves (`{tov,scoring_tov,badpass_tov}_{lo}_{hi}_rs_se_a1500_1500`): attempt volumes buy scoring-turnover allowance, potential assists buy bad-pass allowance, and the shared price vector must also satisfy the total solve, which identifies the self-price spillover (`+1.7` over the scoring-lane anchor, inside the within-player CI `[+1.0, +4.6]`). Kappas have a known physical value of `-1.00` and fitted recovery reads as ridge shrinkage; break-evens are ratios and scale-invariant to it. Two season adjustments apply: prices refit on each trailing five-season window, then each lane's season excess re-centered on its own possession-weighted mean so league saved is zero inside both lanes every year. Putback FGAs move from the self-created to the assisted bucket with a clamped transfer. Multi-season boards are possession-weighted roll-ups of separately priced season rows. Board 2022-26 `n=568`: excess correlates `-0.761` with TOV-value RAPM versus `-0.071` for raw TOV/100, year-over-year `r +0.802`. Generated CSVs are gitignored artifacts under `results/`. See [rZTOV: Mix-Adjusted Turnovers](./RZTOV.md).
- `scripts/build_nba_rztov_box_backcast.py` produces the research-only `box_estimated_rztov` ledger for 1997-2013 and a 1997-2026 diagnostic on one raw-PBP/resolved-possession feature contract. Role validation is `8.29` percentage-point RMSE (`R2 0.845`) for potentially assisted attempt share and `1.05` per-100 RMSE (`R2 0.961`) for potential assists. Smoothed official assisted-make share remains an input, not the target. Historical prices come from independent rolling five-season joint TOV/SCORING_TOV/BADPASS_TOV RS+SE a1500/1500 fits, with 1997-2000 using the earliest 1997-2001 fit; 100-draw player bootstraps report identification bands. Modern observed rZTOV remains separate. On 1,077 common qualified 2014-2026 player rollups, estimated-role rZTOV has weighted RMSE `0.433`, correlation `0.982`, and rank correlation `0.979` versus observed-role rZTOV. The build preserves exact shot, lane, centering, and saved-turnover identities. See [NBA Box-Estimated rZTOV Backcast](./NBA_RZTOV_BOX_BACKCAST.md).
- `scripts/research_prior_informed_tov_rapm.py` is the predictive follow-up: a walk-forward trailing-5 time-decay TOV RAPM ridge whose offense side is optionally centered on the mix-adjusted allowance prior (`|kappa_y|` times trailing decayed rZTOV excess). On the frozen 2025/2026 folds the mix prior beats the plain TD solve `-18.9e-6` pooled with the game-clustered interval clear of zero on both folds, while a persistence prior is neutral and is subsumed by the mix prior in the combo grid. `scripts/research_predictive_ztov_prior.py` extends this with an EPM predictive-ledger prior built entirely from date-gated `p_*` skills (predicted TOV, FGA+FTA mapped onto the tracking TSA volume basis, and assists mapped to potential assists, all priced through the same lane model): the pure predictive prior beats plain TD `-23.4e-6` pooled and edges the observed tracking prior (`-4.5e-6`, interval grazes zero) with no tracking data at all, and blending the observed excess back in adds nothing detectable (`-0.4e-6` vs pure predictive). `scripts/research_predictive_ztov_daily.py` is the daily extension: `evaluate` is a pre-declared within-season rolling test (refresh points on the 1st of Dec-Apr, 2019-2026, every possession from Dec 1 scored once) and `board --as-of D [--fetch]` writes the as-of research board `results/tov_predictive_ztov_board_{D}.csv` plus manifest. Refreshing the possession solve inside the season is worth `-89.7e-6` pooled over the static annual map (P 1.00 in every season and window, growing from ~55 in December to ~105-117 by March/April), the pz prior still beats plain refresh (`-15.1e-6`), and swapping in the latest EPM snapshot at full weight ties the season-start prior with coverage carry-forward (`-0.04e-6 [-3.7,+3.5]`). `evaluate --diagnose` (2026-08-18) explains the tie: the fresh snapshot's movement predicts the season-start solve's forward errors (slope `+0.26`, vs `+0.57` for the static annual map and `-0.07` for the full swap), but at full weight it is double counted against the live rows — the MSE-optimal weight on the swap is `0.50 [0.36,0.64]` — so the pre-specified half blend beats the season-start prior in 8/8 seasons (`-3.2e-6 [-5.0,-1.4]`) while the full swap ties, and `board` now defaults to `--fresh-weight 0.5` in season (`0` = season-start only, `1` = full swap; `off_pz_stale`/`off_pz_fresh` side columns). Frozen-fold numbers are descriptive; the untouched confirmation is 2026-27. Research lane only. See [Prior-Informed Time-Decay TOV RAPM](./TOV_PRIOR_RAPM.md).
- The Help Index defaults to combined career context and exposes both single-season and possession-weighted career leaderboards. Career rankings, profile rank/percentile, and shared-floor teammate summaries can be filtered to regular season, playoffs, or combined; combined pools RS and playoff numerators and denominators instead of averaging schedule summaries. The canonical descriptive build applies each player's matching same-season DARKO retrospectively to the entire season; it is season-level talent context, not an as-of forecast. The navy-and-yellow Lineup Company card opens with 10 clickable teammates and expands by 10, prioritizing frequency rank, exposure share, and the teammate's same-season DARKO impact while retaining weighted help contribution and rating coverage as supporting detail. Its Share action preserves the selected player, time context, schedule, and metric in the URL. The league board opens with 25 qualified players and expands by 25. Selecting a year, RS/PS schedule, or career-arc bar synchronizes the league board and lineup-company panel to that season, while Career restores both surfaces to the selected career schedule. Player season selectors show only seasons available for that player and schedule. Career pair data is published in 64 player-ID shards per schedule so one selection does not download the full career table. See [DARKO Teammate Help and The Help Index](./DARKO_TEAMMATE_HELP.md).
- The separate 2002-2026 championship panel uses monthly pre-postseason EPM
  and actual normalized playoff rotations. The playoff-field conditional
  softmax sums to one; non-playoff teams have zero odds. Every player on one
  team receives identical actual odds, split exactly from the season-average
  playoff rotation into the player's accounting share, shared-floor teammate,
  and off-floor lift. The public page leads with Title Lift over Contract as a
  realized salary-peer swap: on every other historical playoff team, remove the
  closest salary-cap-share rotation player and insert the focal player at that
  peer's actual regular-season workload to simulate qualification and seed,
  then use the peer's playoff workload in the seeded four-round series bracket.
  The headline is average swapped-roster
  odds minus original salary-peer roster odds. Raw title lift uses the same
  placements against a 0-EPM insertion. Regular-season win totals use 2,048
  shared schedule-based draws from the fitted win distribution; conditional
  seeded bracket probabilities are exact, using a leave-one-season-out-selected
  playoff game model and best-of-seven series calculation. The public breakdown
  retains playoff-only lift and seeding plus qualification lift, which add
  exactly to the full-season headline. The cost input is annual
  salary share, not cap hit; missing annual salary remains missing.
  Contract EPM v1 remains a prototype sensitivity, not the headline hurdle. The old
  normalized-self field is labeled player share of team lift and remains only
  in the exact team-accounting equation. Season spikes default to salary-peer
  lift and toggle to context, raw, or teammate lift. The
  page keeps season spikes and teammate lifts side by side and collapses league
  rankings, the DARKO archive, and deeper methodology until requested. The
  league comparison opens as one sortable table with simultaneous salary-peer
  title lift, context title lift, raw title lift, teammate title lift,
  and team title odds columns under compact two-line labels.
  Salary-peer and raw lift cover every annual-salary-covered player-season with
  a frozen pre-playoff EPM rating, including players whose own teams missed the
  postseason; actual team, context, teammate, and additive fields remain null
  outside the playoffs. Multi-team salary stints sum to one annual salary share.
  The August 2020 snapshot is explicitly labeled season 2020, restoring full
  EPM coverage for the bubble playoff field.
  Championship detail is published in 64 player-ID shards. See
  [EPM Championship Help](./EPM_CHAMPIONSHIP_HELP.md). The research-only
  [scalar contract-title challenger](./CONTRACT_TITLE_CURVE_CHALLENGER.md)
  tests the simpler expected-contract-EPM shortcut out of season: EPM surplus
  alone reaches `0.902` correlation with full placements, while multiplying by
  expected salary-slot workload reaches `0.987`. It is a compression test
  trained against the full result, not independent validation or a production
  replacement. Its realized regular-season points-over-contract audit now uses
  exact possession exposure and the identity `surplus EPM * possessions / 100`;
  never treat the sum of annual EPM rates as accumulated points. LeBron's
  corrected total is `4,328.3` points across 114,936 averaged possessions, while
  `81.744` is retained only as an EPM-season sum. The build also retains an
  era-local, budget-neutral full-placement
  challenger: it uses playoff teams within four seasons of the focal year and
  debits signed salary mismatches at `0.067` regular-season and `0.078` playoff
  team-rating points per cap percentage point. Its career totals put LeBron at
  `1.057` titles over contract and Curry at `0.672`, versus `1.417` and `1.235`
  in the full-history metric. A third entering-playoffs EPM-prior challenger
  adds a ridge-shrunk player translation learned only from the previous eight
  postseasons to every player in the bracket, while leaving RS qualification
  and seeding on pre-playoff EPM. Its 2007-2025 held-out playoff MSE improves by
  `31.53e-6`, but the clustered interval crosses zero; it remains research-only.
  The public reader shows all three models simultaneously in the player header
  and league comparison. A URL-persisted selector controls only the active
  season-spike series and detailed model explanation. LeBron is `1.206` career
  titles over contract under the playoff prior versus `1.057` under v2. The
  player page also shows every available non-overlapping five-year peak for the
  selected contract model: rank the complete consecutive five-season windows by
  summed lift, take the best, then repeat among windows sharing no selected
  season.
- `scripts/build_rapm_noft.py` writes audit parquets such as `processed/RAPMnoFT26_PS.parquet` using the standard RAPM possession parser and lineups but actual made/missed FT outcomes instead of shooter expected FT%. This is for raw ORtg/DRtg comparison work; the default `RAPM` processor behavior remains expected-FT unless `ft_value_mode="actual"` is passed.
- `scripts/build_pbp_shot_quality.py` builds a repo-owned shot-quality table from raw PBP descriptions, independent of external ShotQuality `initial_ev`. It writes per-FGA `shot_context_ev`, career and player-season shooter talent, defender-on-court residual allowed, `shot_quality_season_ev`, and `shot_quality_with_defense_ev`; the current full-history artifact is `results/pbp_shot_quality/pbp_shot_quality_1997_2026.parquet` with `6,327,218` unique 1997-2026 RS+PS shots. `RUSSELL_SHOTQUALITY` converts that table into standard-possession RAPM parquets by using `shot_quality_season_ev` for every FGA, expected FT value for FTs, and `0` for other events. The [Visual Shot Quality Explainer](./shot-quality-explainer.html) now separates this production immediate-SQ contract from strict forward defender-geometry, shot-clock, grenade-burden, and expected second-chance continuation challengers, with worked Jokić and Embiid examples. See [PBP Shot Quality](./PBP_SHOT_QUALITY.md).
- The promoted 2014-2026 modern-lineup repair was followed by a 644-file processed rebuild and a fresh 2022-26 ALL RB+SE DECOMP comparison. The processed validation has zero bad saved lineups and exact five-child row algebra; the fresh/published net RAPM correlation is `0.99875` on 1,020 common players. See [Modern Raw-Lineup Repair (2014+)](./MODERN_LINEUP_REPAIR.md).
- `/Users/russellthomas/Docs/2026_NBA_PIPELINE/archive (1)` is a supplemental NBA stats archive for historical validation. For 2009 starter repair, use `PlayByPlay.parquet` substitution text plus `PlayerStatistics.csv` active/DNP validation, `Players.csv` ID/name metadata, and `TeamHistories.csv` franchise-era mapping. Do not trust old `PlayerStatistics*.csv startingPosition` values as starter flags; in game `20800639` that field has more than five nonblank values per team. See [1997-2013 Historical Parquets](./HISTORICAL_1997_2013_PARQUETS.md).
- `scripts/audit_lineup_side_conflicts.py` audits converted historical raw PBP files for impossible lineup-side assignments in linear time by expanding each event row to ten lineup-slot records. Use it on Gabriel old-data repairs to catch same-player home/away collisions and game/player IDs that are almost entirely on one side but briefly appear on the other.
- `wnba_test/fetch_wnba_cdn.py` fetches WNBA raw games from the current WNBA static liveData CDN when the old stats API endpoints time out. It reads `playbyplay_{GAME_ID}.json` plus `boxscore_{GAME_ID}.json`, reconstructs lineups from boxscore starters and substitution events, and writes the raw CSV columns expected by `wnba_test/process_rapm_wnba.py`. For current-season pulls, prefer `python wnba_test/fetch_wnba_cdn.py --season 2026 --season-type "Regular Season" --update-player-map`; it scans CDN ids, writes `wnba_test/WNBA26.csv`, and appends new boxscore names to `wnba_test/player_index_map.csv`. A verified 2026-05-29 pull wrote `27,558` rows across `55` final games through `1022600055`. See [WNBA Clean EFG-Value Decomposition](./WNBA_CLEAN_DECOMP.md).
- `scripts/build_gabriel_old_pbp.py` now fills missing Gabriel source `team` abbreviations from stable `teamId` mappings before converting old-data rows and applies game-local lineup-side guardrails before writing refreshed old-data raw files. This keeps valid scoring/rebound/turnover descriptions on the correct home/visitor side and removes same-player home/away collisions before RAPM processing.
- `process_rapm_blocks/common.py` now owns the shared standard EOP/offense flow used by RAPM. It keeps neutral-description team turnovers, blank EndOfPeriod rows, and FT-correction EOP rows by filling `poss_offense` before O/D mapping, and dedupes same-side player slots before finalization.
- `scripts/validate_historical_eop_flow.py` validates the shared EOP/offense fix across historical RAPM raw parquets. The refreshed 1997-2013 RS production promotion wrote `NBA97` through `NBA13` and `RAPM97` through `RAPM13`; production verification has `3,642,803` RAPM rows, `0` O/D overlap rows, `0` incomplete rows, and `0` same-side duplicate-slot rows. The previous production files were backed up under `validation/historical_eop_flow_1997_2013_rs/production_backup_20260518_021105`. After the full dependency rebuild, current `RAPM09` has Kyle Weaver at `2,296/2,294` off/def possessions, `2,398.627/2,470.253` PF/PA, and `-3.21` net. See [1997-2013 Historical Parquets](./HISTORICAL_1997_2013_PARQUETS.md).
- The full 1997-2013 RS historical parquet dependency rebuild from Gabriel's refreshed source completed under `validation/historical_full_parquet_rebuild_20260518_100028/`. It rebuilt all 306 expected RS processed entries across 18 metrics, including `FIRST_CHANCE_CLEAN`, refreshed the processed RS average audit files, and kept RAPM lineup integrity at `0` O/D overlaps, `0` incomplete side rows, and `0` same-side duplicates. The report also marks all PS raw entries skipped with `no_source_files` because that validation source directory only contained RS source files.
- The 1997-2013 PS historical parquets were repaired from existing PS raw files with `scripts/repair_existing_historical_lineup_sides.py`, then rebuilt through the same 18-metric non-ShotQuality dependency set using `scripts/reprocess_metric.py --raw-dir ... --processed-dir ...`. Production now has all 306 expected PS processed files; validation summary `validation/historical_ps_lineup_repair_1997_2013/processed_ps_validation_summary.json` shows `0` O/D overlap cell hits across every PS metric file, `239,952` RAPM PS rows, and `2,721` remaining RAPM partial-lineup rows with unknown slots as `0`.
- `scripts/build_team_alt3_efg_value_weighted_factors.py` builds team-level Alt3 EFG-value weighted factors from the same processed possession/component targets as the player bundle. It solves team offense/team defense indicators directly, defaults to low team ridge alpha `25`, sign-flips defense columns so positive means good defense, and writes both CSV and parquet outputs such as `master_results/team_weighted_factors_alt3_efg_value_24_26_all_a25.{csv,parquet}`.
- `scripts/analyze_offensive_impact_portability.py` tests whether player offensive impact changes differentially against stronger actual five-man defensive personnel. The date-safe 2019-2026 primary design freezes player O-EPM/role on December 1, uses latest-pregame five-defender D-EPM, player-season fixed effects, teammate/home/rest/score controls, game-clustered uncertainty, playoff prediction, mechanism decomposition, player slope shrinkage, and falsifications. The result runs opposite the motivating mediocre-player-collapse hypothesis: the O-EPM × lineup D-EPM interaction is `-0.060 [-0.076,-0.044]`, a modest extra decline for higher-O-EPM players driven by rim and three-point shotmaking. Generated tables/figures are gitignored under `results/offensive_impact_portability/`; see [Offensive Impact Portability](./OFFENSIVE_IMPACT_PORTABILITY.md).
- `SHOOTER_OREB` is a missed-FGA rebound-opportunity parquet built on the shot row rather than the rebound row. It stores `O1-O4` as the four non-shooter offensive players, `Shooter` as the missed-FGA shooter, `D1-D5` as defenders, and targets `Offensive_Rebound` plus `Self_Offensive_Rebound`; this is the intended surface for separating non-shooter offensive rebounding impact from shooter miss recoverability. Use `scripts/run_shooter_oreb_rapm.py` for the matching three-block ridge solve.
- `BLOCK_RECOVERY` is a blocked-FGA parquet built on the blocked shot row. It stores the shooting lineup as `O1-O5`, the defending lineup as `D1-D5`, and targets `Block_Recovered_By_Defense`; `rapm.py` sign-converts it internally as `Off_Diff = -target` and `Def_Diff = target`, while exported CSVs keep the standard `net_rapm = off - def` convention, so lower `def` values are better for defensive block-recovery impact.
- `STOP` is a standard-possession parquet whose binary target is the union of an offensive turnover and a blocked FGA recovered by the defense. The 2024-2026 RS+PS prepared build has 779,798 possessions and 133,921 STOPs; game-grouped CV selected 2,000/2,000 offense/defense alphas. The generated canonical zero-prior solve is `results/stop_24_26_all_se_cv_a2000_2000_results.csv`; `scripts/build_stop_display.py` writes the reader-ready companion with `oSTOP`, positive-good `dSTOP`, and `STOP = oSTOP + dSTOP`. The preferred predictive prototype is now `scripts/tune_stop_td_epm_prior.py`: TD700 possessions with a date-correct prior equal to `1.25 × p_stl_100 + 0.50 × expected OFFD_100 + 0.10 × p_blk_100 × expected recovery%`. OFFD is forecast with 550-day decay from the total offensive-fouls-drawn rate; charges are only an auxiliary subtype so moving-screen drawers remain represented. The 2,000/10,000 candidate records mean future MSE 0.14185727 versus 0.14185948 for expected steals alone and 0.14189368 for optimized TD700 zero-prior, winning all three 2024-2026 folds. Component weights were selected on those spent folds, so `results/stop_td_epm_prior_2027.csv` is a research prior pending untouched confirmation. See [RAPM Methodology](./RAPM_METHODOLOGY.md#4d-stop-guaranteed-zero-possession-rate).
- `wnba_test/clean_decomp_wnba.py` ports the newest clean EFG-value first-/second-chance decomposition to WNBA CSVs. The current five-year WNBA runs use shared WNBA player-season shooting stats from `wnba_test/wnba_stats_source.py`, which reads `/Users/russellthomas/Docs/wnba/wnba-stats.csv` first, then the backend PBP Stats cache under `/Users/russellthomas/Docs/wnba/data/`, and falls back to fetching/writing that same PBP Stats player-totals cache when the final WNBA stats file has not been rebuilt for the season. The no-decay `21-25 ALL a1750/4000` output is `wnba_test/Results/clean_decomp_21_25_all_a1750_4000/weighted_factors_alt3_efg_value_21_25_all_a1750_4000.csv`, and the decayed `td700` companion is under `clean_decomp_21_25_all_td700_a1750_4000`. Downstream `rapms` publishes use the `wnba_` prefix, such as `wnba_weighted_factors_alt3_efg_value_21_25_all_a1750_4000.csv`, so V3 can distinguish WNBA schema from NBA Alt3 EFG-value files. The WNBA processed surface is currently CSV-based, not parquet-based; rebuild `RAPM`, `TS`, `REB`, `TOV`, `FIRST_CHANCE`, and `SECOND_CHANCE_CLEAN` CSVs for RS and PS after possession-logic changes. The clean path supports raw rows with at least four known players on each team while excluding player id `0` from solves, uses the shared WNBA robust possession helper for generic team rebounds/turnovers plus period-end possession owners, mirrors the NBA `ALT_FT_FREQ + ALT_FT_SEVERITY = ALT_FT` split, and uses the no-prior-completion first-chance TOV guard so true TOV rows are scoreless before applying the average non-turnover first-chance TS/scoring baseline. See [WNBA Clean EFG-Value Decomposition](./WNBA_CLEAN_DECOMP.md).
- `wnba_test/build_wnba_net_decomp.py` is the isolated WNBA kappa/net-continuation research lane. It grants a missed first-chance FGA its zone-specific expected continuation, pays the same value out of `SECOND_CHANCE_NET`, reprices turnovers at the full net non-turnover counterfactual, and refits factor box priors on pre-2026 net labels. The August 24 board covers 281 games / 229 players, closes at `1.71e-13`, and correlates `0.99698` with same-date gross VPM; it remains research-only pending a predeclared future-possession gate. Artifacts live under `wnba_test/Results/wnba_net_decomp/`. See [WNBA Clean EFG-Value Decomposition](./WNBA_CLEAN_DECOMP.md#net-continuation-single-season-research-board).
- `wnba_test/build_prior_informed_wnba.py` retains the historical-RAPM-state true-prior prototype (`1750/2000`, `0.90` persistence). The preferred current research layer is `wnba_test/build_wnba_box_priors.py`: it trains factor-specific public box priors on earlier seasons only, selects the single-season prior-centered solve on sealed 2024-2025 forward folds, and builds the live 2026 clean decomposition at offense alpha `5000`, defense alpha `40000`, and prior scale `1.5`. `wnba_test/build_wnba_metric_history.py` applies that frozen architecture to the reader-ready 2022-2026 history: 2022 is a labeled pure-RAPM warm-up, 2023-2026 use year-forward box priors, and the consolidated CSV contains 882 player-seasons with full and 1,000-possession-qualified ranks. Atomic priors, FT/TOV children, and fitted offense/defense/net displays remain exactly additive. The frozen 2025 box-prior board records 13.118 game-margin RMSE and 0.403 correlation on the repaired 2026 benchmark; the standard benchmark includes that board alongside prior-day DARKO/PI/NPI comparisons. Proprietary DPM/PI/NPI ratings remain external benchmarks only. See [WNBA Metric Build Status](./WNBA_METRIC_BUILD_STATUS.md).
- `wnba_test/build_luma_historical_metrics.py` extends the WNBA lineup history to 2003-2026 from LUMA aggregate stints: 5,623 games, 292,446 strictly valid positive-duration team-side observations, 3,922 player-seasons, annual pure RAPM, and an exact eight-child TS/TOV/OREB-opportunity decomposition. LUMA supplies only the lineup likelihood; the corrected prior uses the exact component/side PBP Stats feature architecture from the 2022-2026 WNBA build, archived back to 2009, with expanding-season mapper-alpha selection and explicit 2003-2009 warm-ups. The rebound child remains `OREB_OPPORTUNITY_VALUE`, not event-sequenced `SECOND_CHANCE_CLEAN`. Generated boards and a source-hashed manifest live under `wnba_test/Results/luma_historical_our_priors/`; the old generic-LUMA-rate prototype is deprecated. See [WNBA LUMA Historical RAPM Bridge](./WNBA_LUMA_HISTORICAL_BRIDGE.md).
- `wnba_test/tune_wnba_component_alphas.py` ports the VPM-style joint component-sum coordinate descent to the nine WNBA clean-decomposition atoms. It tunes on 2023-to-2024 and 2024-to-2025 future actual-possession MSE and keeps 2026 out of selection. The July 29 candidate improved development MSE by `276.984e-6` versus uniform `5000/40000`, but lost `418.799e-6` on 2026, so it is diagnostic only and the live alphas remain unchanged.
- `wnba_test/tune_wnba_daily_net_hyperparams.py` replaces the borrowed single-season alpha assumption with a pure multi-year net-DECOMP rolling-origin grid over completed 2022-2026 seasons. On the exact ten-target structural net family it selects base offense/defense alphas `2000/5000` with no decay over the five-season window; the unconstrained grid minimum uses a 1100-day half-life, while the old `5000/40000/550` row is outside the paired one-SE set. `wnba_test/build_wnba_sq_net_targets.py` adds an isolated raw-action-mapped MID/THREE SQ EV plus residual split. Context-only SQ gains `117.313e-6` on the summed future-possession point estimate with a game-cluster interval crossing zero and wins 2/3 seasons; the prior-shooter target is statistically tied and conceptually overlaps future box priors. Both are development-only because 2026 is spent, current 2026 SQ coverage is stale, and within-season historical daily cuts still require causal-as-of kappa rebuilding. See [WNBA Daily Net-DECOMP Alpha Tuning](./WNBA_DAILY_NET_ALPHA_TUNING.md).
- `wnba_test/build_wnba_rtss_prior_features.py` and `evaluate_wnba_rtss_priors.py` build and season-forward test compact RTSS mechanisms as factor-specific box-prior additions. The current research surface has 497 mapped player-seasons and 75 stabilized features, including exact on-ball time percentage from touch time over offensive chance floor time. Fourteen of 24 component-side challengers cleared 2024 development; nine improved both RMSE and correlation on the once-opened 2025 confirmation, led by defensive THREE frequency, defensive FT, and offensive second chance. This is underpowered screening: the 2024 selection fit has only the 2023 RTSS season, while the box backbone can use 2022 onward. Raw `qSP - qSQ` did not earn a standalone promotion and has a visible 2023-to-2024 calibration shift. Keep the lane research-only and the live box prior unchanged until compact RIM/MID/THREE and other factor-specific additions are built from multi-year estimated skills and clear fresh RTSS plus future-possession confirmation. See [WNBA RTSS Component-Prior Research](./WNBA_RTSS_PRIOR_RESEARCH.md).
- `wnba_test/train_wnba_rtss_q_models.py` audits provider qSQ/qSP and fits current-game-excluded season-forward challengers on 2023-2025 RTSS while hard-fencing 2026. Raw q empirically uses an expected-eFG scale, so calibrated make probability is `2*q/shot_value`. Rich release context improves full-court qSQ/qSP primarily by correcting persistent rim overprediction, but does not consistently beat the provider on the exact 66,886-shot sanctioned MID/3 sample; no live WNBA target changes. See [WNBA RTSS qSQ/qSP Challengers](./WNBA_RTSS_Q_MODELS.md).
- The ignored WNBA RTSS mirror is refreshed from the cross-account prefix `s3://rtss-data/wnba/`. Verify with `aws s3 ls`, inspect `api/portal/schedules.json`, dry-run `aws s3 sync`, then sync into `downloads/rtss-data/wnba/`. On August 6, 2026, the remote regular-season manifest had 330 games, 229 ended, with paired 2D/3D markings available for ended games. The old `rtss-data` IAM user no longer exists locally; manual reads currently work through the authenticated account and bucket policy, while unattended updates should use a newly created least-privilege profile. See [WNBA RTSS Rich-Tracking Feature Audit](./WNBA_RTSS_FEATURE_AUDIT.md).
- `wnba_test/cross_validate_wnba_rtss_priors.py` treats 2023-2025 as development data and runs symmetric leave-one-season-out RTSS residual CV over a properly trained non-held-out 2022-2025 box backbone. Use `shots.assistOpp`, not made-only `shots.assisted`, for potential-assisted versus self-created attempts; pass-level `assistOpp` supplies potential assists created. The player-season surface identifies 56,899 potential-assisted shots and 56,750 potential assists. Compact tracking lowers pooled RMSE and raises correlation for 15/24 component-sides. For offensive shot priors, it helps RIM frequency, MID frequency, MID finishing, and THREE frequency; retain box-only for RIM and THREE finishing. MID role structure is cleanest: role-only finishing improves `0.73%` in all three folds, and touch/creation context reaches `0.79%`. This is post-holdout development, not live validation. See [WNBA RTSS Component-Prior Research](./WNBA_RTSS_PRIOR_RESEARCH.md).
- `wnba_test/cross_validate_wnba_rtss_total_priors.py` tests WNBA RTSS directly against annual clean-total low-alpha ORAPM/DRAPM targets using orthogonal offense/defense target-alpha sweeps, symmetric 2023-2025 season CV, nested compact selection, and held-out permutation importance. `off_alpha=250` is the lowest defensible exploratory offense target: target SD `3.54`, mean YoY correlation `0.318`, and RTSS-only `R2=0.292` / correlation `0.541`. Low-alpha defense fails: even `def_alpha=1000` has mean YoY correlation `0.159` and RTSS-only `R2=0.007`. Creation role and playmaking lead offense; only rebound positioning is remotely stable on defense. Box stays stronger, and compact RTSS additions worsen pooled box RMSE at every tested target alpha, so this is a diagnostic parent-prior experiment rather than a live or additive-prior promotion. See [WNBA RTSS Component-Prior Research](./WNBA_RTSS_PRIOR_RESEARCH.md).
- `wnba_test/build_wnba_rtss_playmaking_roles.py` fits a centered joint shooter/potential-assist-passer/presence shot-value RAPM on 111,951 exact RTSS shot lineups and prices pass turnovers separately. A single pooled one-SE role alpha (`300`) keeps the three annual targets comparable. In symmetric development CV, on-ball time plus potential assists predicts passer shot value at `0.597 R2` and `0.773` correlation; adding rim assists is slightly worse because the inputs are strongly collinear. The same compact lane has only `0.102 R2` after pass-turnover cost and no explanatory value for presence. Keep scoring, passer creation, passing turnover cost, and presence as distinct prior children. See [WNBA RTSS Shooter, Passer, and Presence RAPM](./WNBA_RTSS_PLAYMAKING_RAPM.md).
- `wnba_test/build_wnba_rztov.py` estimates the missing WNBA rZTOV workload from PBP Stats-style box/PBP fields after learning assisted-shot share and potential assists from the 2023-2025 RTSS labels. LOSO development reaches `0.558 R2` for assisted share and `0.950` for potential assists. Estimated self-created plus assisted TSA is constrained to equal box TSA exactly; WNBA-specific joint break-evens are `14.88%`, `6.71%`, and `18.59%` for self-created TSA, assisted TSA, and potential assists. The build emits both the 229-row current 2026 board and an 890-row 2022-2026 descriptive history. The history uses one fixed 2023-2025 retrospective role/price fit, labels 2022 as pre-training backcast, 2023-2025 as training-period estimates, and 2026 as current unlabeled; it is not date-safe prediction or observed tracking history. Preserve negative-good rZTOV and do not replace the live WNBA VPM board. See [WNBA Box-Estimated rZTOV](./WNBA_RZTOV.md).
- `wnba_test/build_wnba_rtss_interim_snapshot.py` builds a dated descriptive RTSS ratings board without firing or weakening the season-end promotion harness. The August 10 artifact applies the frozen pre-2026 residual corrections to the August 6 season-2026 RTSS feature mirror, solves their linear coefficient change on the preserved 242-game possession snapshot, and adds that change to the preserved August 10 box-only board. It performs no 2026 outcome scoring or selection, leaves the live metric and confirmation sentinel untouched, and re-derives all additive parents after rounded-artifact composition. See [WNBA RTSS Component-Prior Research](./WNBA_RTSS_PRIOR_RESEARCH.md).
- NBA data should bootstrap WNBA box/component priors through hierarchical partial pooling, not naive row pooling: standardize within league-season, use aligned atomic targets, shrink WNBA slopes toward NBA slopes with component-specific transfer strength selected only on WNBA folds, and compare against WNBA-only and NBA-only baselines. The local repo does not have men's RTSS markings with the identical `assistOpp`/`qSQ` schema, so creation-role and rich-tracking residual coefficients remain WNBA-only. See [WNBA RTSS Component-Prior Research](./WNBA_RTSS_PRIOR_RESEARCH.md).
- The WNBA predictive program plan of record is [WNBA Predictive Program Plan of Record](./WNBA_BETTING_PROGRAM.md); its Phase 0/1 game-day lane shares the `wnba_test/wnba_gameday_common.py` contract module. `build_wnba_game_logs.py` builds the finals table and player game logs from the CDN boxscore cache with US/Eastern game dates; `project_wnba_minutes.py` allocates 200 team minutes per game (EWMA half-life tuned on 2025 only); `calibrate_wnba_forecast_sigma.py` fits HCA/sigma on 2022-2025 prior-season history boards and replays 2026 pregame; `emit_wnba_pretip_forecasts.py`, `capture_wnba_odds.py`, and `score_wnba_forecasts.py` run the hash-chained pre-tip forecast/odds logs with fail-closed joins, a strict `emitted_at < tipoff` fence, and CRPS-vs-realized next to closing-line value. `build_wnba_kalman_box.py` is the DARKO-style per-stat Kalman panel (DE-tuned on ≤2025, 2026 frozen holdout; all 14 stats beat trailing-10 both places); `build_wnba_zone_game_lines.py` + `build_wnba_estimated_skills.py` extend it to E-Skills-style zone estimated skills (rim/mid/three/FT percentages on attempts, per-minute volumes on minutes; all 8 beat all baselines on dev and the frozen 2026 holdout, and the LightGBM fusion stage is a documented null) — see [WNBA Estimated Skills](./WNBA_ESTIMATED_SKILLS.md). The guarded one-shot RTSS confirmation harness is `run_wnba_rtss_confirmation.py` (fires once at season end per [the prereg](./WNBA_RTSS_CONFIRMATION_PREREG.md)); `extend_wnba_defense_alpha_grid.py` retained `def_alpha=40000` on development folds. Offline tests: `nba_pipeline/tests/test_wnba_gameday_lane.py`. See [WNBA Pre-Tip Forecast Log Operations](./WNBA_FORECAST_LOG.md) and [WNBA Predictive Box Panel](./WNBA_KALMAN_BOX.md).
- `wnba_test/build_wnba_playtype_difficulty.py` ports the frozen six-family NBA PBP shot-role taxonomy to WNBA regular-season play-by-play, priced from the immediately prior season, giving the WNBA its first role/playtype difficulty value in TS percentage points without any Synergy-equivalent feed. The one required change is season centering: WNBA scorers kept adding descriptors (`CUT_2` share `0.001` in 2013 to `0.066` in 2026), so the NBA's prior-season league reference reads pure labeling drift as up to `3` TS points of league-wide difficulty; pre-2018 rows are emitted only as `descriptor_regime = sparse_descriptions` and are unusable. Second Spectrum confirms the labels without seeing a description (`SELF_CREATED_3` is `94.2%` off-dribble at `3.79` dribbles, `OTHER_3` is `94.9%` catch-and-shoot), family values correlate `0.99` with recalibrated tracking values, and player-seasons correlate Pearson `0.760`. Raw `qSQ` cannot be used on the WNBA points scale: it runs `+15` points hot on threes in all of 2023-2025 while sitting within `2.4` points on twos. The scale is mildly over-dispersed, so add `0.81 x playtype_diff` to `rTS`, never `1.0x` — the NBA's exact substitute-adjusted-for-raw spec *loses* to the box baseline in 6/6 seasons because role persists (`rho=0.757`) about twice as strongly as shooting efficiency (`lambda=0.397`), and `beta(lambda-rho)` predicts that sign. Development only; 2026 is the untouched confirmation season. Offline tests: `nba_pipeline/tests/test_wnba_playtype_difficulty.py`. See [WNBA Playtype / Shot-Role Difficulty](./WNBA_PLAYTYPE_DIFFICULTY.md).
- `wnba_test/build_wnba_pbp_shot_quality.py` builds a leakage-safe, season-forward WNBA expected-points model from Gabriel play-by-play distance, shot family, shot-form modifiers, and period-clock context. Impact scope is explicitly limited to the existing MID and THREE components; rim is excluded. Smoothing is selected on 2023-2025 only; on 2026 midrange/three attempts, context plus strictly prior-season shooter talent improves MSE by `1.068%` over a two-/three-point baseline. Gabriel classic PBP can add 239,829 historical MID/THREE attempts, but 2018-plus and full-history windows did not beat modern-only on 2026, so historical expansion remains opt-in and the canonical research artifact stays modern-only. See [WNBA Play-by-Play Shot Quality](./WNBA_PBP_SHOT_QUALITY.md).
- `wnba_test/train_wnba_rtss_pbp_student.py` distills RTSS `qSQ` into the outcome-safe PBP MID/THREE feature contract. The exact `game_id + pbpId` join validates 61,521 teacher shots with one shot-value conflict excluded. The standalone student clearly beats a same-row binary-outcome learner on matched 2024 and 2025 shots but does not replace the mature PBP model. A blend weight selected on all 2024 PBP shots gives the qSQ student `0.41935` weight and improves 2025 context-only MSE by `0.001247 [+0.000151,+0.002282]`; the shooter-inclusive blend remains inconclusive. This is spent development evidence, RTSS 2026 stays sealed, and the live WNBA VPM is unchanged. See [WNBA RTSS-to-PBP ShotQuality Student](./WNBA_RTSS_PBP_SHOT_QUALITY.md).
- `scripts/build_alt3_efg_value_display_bundle.py` and `scripts/build_alt3_efg_value_weighted_factors.py` preserve the immediate pre-DECOMP Alt3 audit family. These exports sign-flip defense columns, include six EFG-value pieces plus FT/TOV/SC drilldowns, and close through `ALT_EFG_BASELINE`. Do not treat them as the forward public interpretation; use `weighted_factors_decomp_net_*` after the audited 1Y-5Y migration, with gross `weighted_factors_decomp_*` retained for compatibility.
- `scripts/build_vpm_target_swap.py` builds isolated EV-target `FIRST_CHANCE` research parquets from a leakage-safe per-shot EV table. It swaps mid/three make targets, carries actual-minus-EV residuals in the registered VPM aliases, uses clock-first possession assignment, and refuses output if the row-level target identity fails. It changes neither forward Net DECOMP nor gross compatibility DECOMP and does not copy private licensed shot data. See [VPM Reproduction and Recovery Status](./VPM_REPRODUCTION.md).
- New VPM experiments run from the private sibling
  `vpm/vpm/run_harness.py`: the config-driven harness checks 2016-2026 target
  closure, builds matured forward component labels and compact stabilized
  player priors, protects holdouts, scores summed future actual possessions
  against as-of EPM, and writes paired game-clustered uncertainty artifacts.
  Its `estimate-board` command applies the accepted standalone v2 x2 model to
  the full as-of player pool and emits exact ten-component plus
  oTS/oTOV/oSC estimates; current-player status comes from latest-season PBP,
  while current box coverage remains a separate diagnostic. It never
  publishes public DECOMP or consumes raw provider/shot rows directly.
- The sibling VPM harness also owns the preregistered preseason full-season
  alpha lane (`research_harness_v6_offseason_netcont.json` and
  `offseason-alpha`). It uses explicit preseason snapshots—October 1 except
  December 1, 2020 for the delayed 2020-21 offseason—selects a five-parameter
  structured alpha family and a separate plain-RAPM pair on 2019-2022 future
  regular seasons, treats 2023-2026 as report-only stress evidence, and
  protects 2027 as the untouched confirmation. Training can use only complete
  source seasons earlier than the target; this is why completed-season
  net-continuation baselines are date-safe for this lane even though
  whole-file baselines are not automatically safe for midseason cuts. The
  completed run retained incumbent v4: the selected challenger tied v4 in
  development (`+0.65e-6 [-25.15,+26.73]`) and remained inconclusive versus
  v4 x2 in stress (`+28.50e-6 [-14.50,+71.65]`). Selected x2 clearly beat
  TD550 RAPM and complete EPM on common realized lineups, but this is not a
  full game forecast or a promotion. Exact requested-date EPM snapshots are
  mandatory and 2027 remains untouched. Its companion position-allocation
  audit finds a specific jumbo-defense warning: selected x2 clearly improves
  zero-center defenses, is inconclusive with one center, and loses to its
  no-prior counterpart in all four stress seasons for two-plus-center defense;
  the offensive two-plus-center bucket improves. This is diagnostic evidence,
  not a blanket center penalty or permission to tune on spent 2023-2026 rows.
  The follow-up scale and parent-projection experiments reject both simple
  repairs: defense x1 loses to x2 in aggregate historical stress and the
  exact player-level parent projection loses in development. Treat the
  remaining problem as lineup-dependent rim-protection redundancy and keep
  it separate from additive VPM. See
  [VPM Reproduction](VPM_REPRODUCTION.md#defensive-prior-scale-and-allocation-follow-up).
  See [VPM Reproduction](VPM_REPRODUCTION.md#preseason-position-allocation-audit).
- The hardened EPM12 challenger is
  `vpm/configs/research_harness_v3_epm12.json`. It maps only checksummed
  date-safe D&T predictive `p_*` box-score skills into all 12 component-side
  priors; complete EPM and eWINS are forbidden features. EPM12 x2 adds
  `+153.02e-6` beyond EV no-prior on the four development cuts (4/4; 95% CI
  `[+73.38,+233.87]`) and is the new research baseline for residual stacking.
  The public reader now displays this EPM12 x2 research board; that display
  does not promote the method or spend the untouched 2026-27 confirmation.
- The sibling EPM12 direct-parent simplicity baseline maps the same date-safe
  `p_*` union to matured future pure O/D RAPM before one parent solve. On
  frozen 2023-2026 stress it beats plain RAPM by `+254.11e-6
  [171.85,336.95]`; decomposed selected VPM adds an unresolved `+41.55e-6
  [-82.46,165.29]`, and the extra 25% post-solve pull clearly loses to the
  parent. Keep decomposition pending untouched 2027, but require future
  component claims to clear this parent baseline. See
  [VPM Reproduction](VPM_REPRODUCTION.md#epm12-direct-parent-simplicity-baseline).
- The EPM12 shooting-input audit corrects the original additive
  percentage-plus-attempt recipes by constructing three, rim, midrange, and FT
  makes per 100 first. The corrected lane remains tied with its parent and
  loses retrospectively to the old representation on spent stress, so the
  board stays frozen and 2027 remains the decision set. See
  [VPM Reproduction](VPM_REPRODUCTION.md#epm12-counting-rate-correction).
- The EPM12 x2 research board is available in the public
  [VPM Lab](https://vpm-lab-2026.russdt.chatgpt.site): an interactive current/
  full-history ranking table in a bounded scroll frame, first-class
  ten-component architecture, and an integrated player-impact hierarchy that
  connects identity directly to offense, defense, and total VPM before a
  `3 x 2` factor map with large `oTS/oTOV/oSC` and `dTS/dTOV/dSC` totals.
  Each TS tile condenses its eight underlying inputs into four exact display
  values: Rim (rim frequency + rim FG), Mid (mid EV + residual), 3P (frequency
  + EV + residual), and FT value. The full ten-component engine remains
  unchanged. Desktop keeps the connected player header and both factor rows in
  a compact single-screen hierarchy while retaining its explanatory copy.
  Mobile preserves the six-tile map in one compact frame and hides only
  secondary prose. The player card defaults to a warm-white/black Geist
  editorial treatment with chartreuse reserved for aggregate totals, while an
  in-card switch preserves the original dark blue/green comparison. Turnover
  tiles use explicit offensive/defensive turnover value wording; in the dark
  alternate, blue and green consistently distinguish offense and defense.
  The product also includes
  a ten-component five-player additive study, and a separately labeled catalog
  of non-additive research drilldowns. Its `/about` guide uses the product name
  **Impact Monster** for the ten-factor breakdown: RAPM is the parent total,
  the ten factor types are estimated on offense and defense and add back to
  their RAPM sides, and `TOV/TS/SC` is only the friendly possession-stage
  organizer. The nested `vpm-explorer/` project
  contains a frozen copy of the displayed research board and must be deliberately
  refreshed after a new harness estimate; see
  [VPM Reproduction and Recovery Status](./VPM_REPRODUCTION.md).
- `FIRST_CHANCE_CLEAN` is the direct first-chance scoring parquet for the newer FC accounting: it keeps the same standard possession rows, uses ExpFT-adjusted first-chance points on non-turnovers, and sets true first-chance turnover rows to `0`. A true first-chance turnover requires no prior first-chance FGA/FT completion event in the same possession. It should solve the same concept as `ALT_TS + ALT_TOV_VALUE`, while `ALT_BADPASS_TOV_VALUE + ALT_SCORING_TOV_VALUE = ALT_TOV_VALUE` remains the turnover drilldown.
- `FIRST_CHANCE` now carries ShotQuality-era `Is_FC_Transition_Possession` and `Has_FC_FGA` labels for 2023-24 through 2025-26. `rapm.py` exposes same-denominator first-chance mode aliases: `FC_TRANSITION_SCORING`, `FC_HALFCOURT_SCORING`, `FC_MODE_MIX`, `FC_TRANSITION_VALUE`, and `FC_HALFCOURT_VALUE`. The additive trio satisfies `FC_MODE_MIX + FC_TRANSITION_VALUE + FC_HALFCOURT_VALUE = FIRST_CHANCE.Net_Diff` at the processed-row level; the placeholder scoring aliases use same-sample mode PPP baselines computed inside the loaded RAPM run and respect `--timedecay`.
- `FIRST_CHANCE` now also carries `FC_FT_FREQ_Diff` and `FC_FT_SEVERITY_Diff`, an exact two-part split of `FC_FT_Diff`. `ALT_FT_FREQ` prices each first-chance FT trip at the same processed-file average FT-trip value; `ALT_FT_SEVERITY` is the trip's value above or below that average, combining trip type mix and FT shooting. Validate `FC_FT_FREQ_Diff + FC_FT_SEVERITY_Diff = FC_FT_Diff` before solving or publishing the child RAPMs.
- Decomposition changes must verify processed-surface row algebra first: check max row gap, sum gap, and child-parent identity counts in the parquet/CSV before relying on RAPM coefficient sums. Separate ridge solves and rounded CSV exports can leave tiny coefficient gaps even when the row algebra is exact.
- The production 24-26 ALL RB+SE a2000/4000 EFG-value display bundle is `results/alt_efg_value_display_bundle_24_26_all_rb_se_a2000_4000_results.csv`. It uses six atomic shot-value solves (`rim/mid/three` frequency and FG), `ALT_FT`, clean first-chance `ALT_TOV_VALUE`, `SECOND_CHANCE_CLEAN`, point-adjusted bad-pass/scoring turnover drilldowns, `ASSIST_POINTS`, and `RIM_ASSIST`; broad `ALT_EFG_RIM/MID/THREE` zone solves are audit-only. Regenerate published bundles after this correction before treating their TOV columns as final.
- `FIRST_CHANCE` / `rapm.py` now support a zero-residual `ALT_EFG_VALUE` shot-location decomposition: `ALT_EFG_RIM + ALT_EFG_MID + ALT_EFG_THREE = ALT_EFG_VALUE`, with a deeper six-value split into `ALT_EFG_RIM_FREQ`, `ALT_EFG_RIM_FG`, `ALT_EFG_MID_FREQ`, `ALT_EFG_MID_FG`, `ALT_EFG_THREE_FREQ`, and `ALT_EFG_THREE_FG`. `ALT_EFG` remains the `ALT_TS - ALT_FT` accounting parent; `ALT_EFG_VALUE` removes the fixed replacement-shot baseline from the player-facing shot-location solve. See [Alternate Clean 3-Factor](./ALTERNATE_CLEAN_3_FACTOR.md).
- `03_run_rapm_analysis.py` now includes seven auxiliary shot-profile solves in each master weighted-factor run: `RIM_FREQ`, `RIM_FG_PCT`, `THREE_FREQ`, `THREE_FG_PCT`, `MIDRANGE_FREQ`, `MIDRANGE_FG_PCT`, and `ASSIST_POINTS`. Both `weighted_factors_*` and `weighted_factors_alt3_*` export them as pass-through diagnostic columns outside the additive factor reconstruction.
- `03_run_rapm_analysis.py` now passes fixed offense/defense alpha overrides through to every metric solve with `--off-alpha` and `--def-alpha`; non-default alpha weighted-factor outputs and result folders include an `a{off}_{def}` suffix such as `_a2000_4000`.
- `rapm.py` and `03_run_rapm_analysis.py` now handle cross-century two-digit season windows; for example, `97 0` expands to 1997-2000 and writes `97_00` outputs. The downstream publish helper was used for the full 4-year rolling `all_rb_se` weighted-factor set from `97_00` through `23_26`.
- `scripts/run_historical_core_weighted_factors.py` now supports the preferred long historical fixed-constant run: use `--fixed-season-effects` to subtract each metric's RS raw mean by season from both RS and PS rows, and `--age-poly-coefficients` to subtract fixed `curve(age) - curve(27)` age offsets before centering and solving. This writes `_fse_agepoly{degree}` outputs plus `fixed_rs_season_baselines.csv` and per-metric `*_fixed_offsets.csv` audits. See [1997-2013 Historical Parquets](./HISTORICAL_1997_2013_PARQUETS.md).
- `scripts/build_processed_rs_parquet_averages.py` regenerates the regular-season processed parquet average audit family, including `processed_rs_parquet_averages_per100_97_26.csv`, row counts, sources, and skipped-file reasons. It also derives the alt3 alias audit columns (`ALT_TS`, `ALT_EFG`, `ALT_FT`, `ALT_TOV`, `ALT_BADPASS_TOV`, `ALT_SCORING_TOV`, plus SQ/make where present) from `FIRST_CHANCE`.
- `daily_rapm_update.sh` fetches/processes RS+PS, rebuilds the active 1Y-5Y `weighted_factors_decomp_*` CSV/parquet family, and syncs it downstream. It deliberately skips the old Alt3-to-scposs mapper until that mapper is audited for the new 97-column schema; current-metric boards therefore stay on the last audited scposs surface. The `csvs` publish step still requires `test-branch`.
- `scripts/season_windows.py` is the shared season-year window helper. `scripts/upload_six_factor.py` and `scripts/upload_timedecay_rapm.py` no longer carry hardcoded year tokens: both derive their `weighted_factors_*` filenames from `--year` (default `NBA_RAPM_SEASON_YEAR`, else `2027`) the same way `daily_rapm_update.sh` derives its own, so a season rollover no longer points the no-argument nightly calls at files that do not exist. Both take `--dry-run`, which reports the per-window row counts and touches Supabase not at all; the six-factor uploader also takes repeatable `--source-file INTERVAL=PATH` for backfills. Each refuses to upload when `Latest_Year` in the CSVs disagrees with the requested year, because the upload deletes the target year before inserting.
- The `scposs` 2027 publish is blocked on a missing mapper rather than on stale literals; see [Public DECOMP](./PUBLIC_DECOMP.md) for the measured schema gap and input coverage. `update_2026_scposs.py` now resolves the real `rapm_{token}_all_se` layout (falling back to the legacy `_all` layout) and refuses to write when the existing `scposs` files carry a wider schema than it produces.
- The rolling Net SQ DECOMP producer/publisher no longer carry hardcoded season literals. `scripts/run_net_sq_decomp_rolling.py --allow-missing-playoffs` lets each window's own end year have no PS target half, for a season still in progress, while every earlier season still requires both halves and a partial end-year playoff half is a hard error; affected windows record `end_year_playoffs_included: false` in their manifest. `scripts/publish_td550_rolling_archive.py` now derives the newest required end year from the archive on disk instead of a `range(1997, 2027)` literal, keeping completeness exact so a partially computed newest season fails rather than publishing half of it (`--max-end-year` pins it). `scripts/nightly_net_sq_td550_refresh.sh` runs the four-stage chain on a schedule with one staleness gate per stage, and keeps the Supabase publish off unless `PUBLISH=1` is set explicitly, because `nba_no_prior_vpm_td550_rolling` is live. It is written but deliberately not installed. The 2027 compute is runnable up to one owner decision: the repo-owned shot-quality table is merely stale, but the clean_sq config marks 2027 a protected confirmation season, and the nightly reports that stage BLOCKED rather than editing the config around it; see [Net SQ DECOMP rolling archive](./NET_SQ_DECOMP_ROLLING.md).
- `rapm.py --age-dummies` now uses `dpm_history.csv` as the player-season age source, covering 1997-2026 via `nba_id`, `season`, and `age`. This uses DARKO only as an age lookup; `o_dpm` / `d_dpm` are not used as RAPM targets. `03_run_rapm_analysis.py --age-dummies` passes age fixed effects through the full daily weighted-factor metric set and writes `_agefe` weighted-factor outputs. `--age-curve` remains the separate RAPM-history prior mode backed by `player_stats_with_metrics`.
- `scripts/run_historical_core_weighted_factors.py` has an older isolated 1997-2026 regular-season estimated-season-effects build that solved `RAPM`, `TS`, `TOV`, `REB`, `BADPASS_TOV`, and `SCORING_TOV` before writing `weighted_factors_core_97_26_rs_se.csv`. Keep this path separate from the daily master runner.
- `scripts/build_gabriel_old_pbp.py` now builds the Gabriel `merged_playbyplay/old_data` archive into 1997-2013 RS/PS raw parquets plus core `RAPM`, `TOV`, `REB`, and `TS` processed parquets. The regular-season 1997-2013 non-ShotQuality factor backfill now covers `LA_RAPM`, assist, shot-location frequency/FG%, FT premium, and clean first-/second-chance surfaces; `scripts/reprocess_metric.py` supports full-year historical ranges and `--season-types` for targeted rebuilds. Converted raw files mark `allow_partial_lineups=True`, processors keep rows with at least four known O/D players, and `rapm.py` excludes unknown player id `0` from coefficients. Gabriel historical `RAPM` / `TS` processing uses actual FT outcomes when shooter FT% is missing, without changing the daily pipeline default. `scripts/build_bref_ft_pct_lookup.py` builds the 1997-2000 Basketball Reference `nba_id`/FT% lookup; the shared stats loader now uses that RS lookup for both RS and PS 1997-2000 `RAPM`, `LA_RAPM`, `TS`, and `FT_PREMIUM` rebuilds. See [1997-2013 Historical Parquets](./HISTORICAL_1997_2013_PARQUETS.md).
- `rapm.py` now supports `--strength-splits` for RAPM-only DARKO strong/not-strong opponent interactions, using the sibling `2026_NBA_PIPELINE/databallr/darko/dpm_history.csv` by default and writing `_strength` result files with `off_vs_*`, `def_vs_*`, exposure counts, and exposure-weighted `overall_*` columns. Add `--strength-legacy-scale` to write `_strength_scaled` outputs with the legacy target standard-deviation scaling (`1.48` offense, `1.36` defense).
- `rapm.py` also supports `--strength-sensitivity` for continuous DARKO opponent-strength slopes, writing `_strengthsens` outputs with separate offense/defense slopes plus `strong_minus_weak` and `weak_minus_strong_drop` diagnostics.
- `process_rapm_blocks/common.py` now keeps schedule seasons as canonical labels like `2025-26` so processed parquet finalization can merge `game_date`; `scripts/backfill_processed_game_dates.py` backfills older processed files if solver logs show flat decay weights or processed surfaces have null dates, with `--all-files` for full processed-surface scans and `--infer-legacy-id-dates` for optional pre-2014 legacy ID date inference.
- `01_fetch_pbp_data.py` now normalizes raw `game_id` values and removes exact duplicate PBP rows before writing parquet; the shared raw reader in `process_rapm_blocks/common.py` also dedupes exact rows defensively on read.
- `scripts/01_fetch_historical_pbp_espn.py` can now build historical raw parquets from SportsDataverse / `hoopR` ESPN season files, including lineup reconstruction from starter flags plus substitution rows and repo-stable historical player ID mapping.
- `process_rapm_blocks/common.py` now repairs missing raw lineup IDs with in-game lineup fills before dropping rows, and `scripts/lineup_stats.py` falls back to the FT shooting side when offense assignment is blank on free-throw rows.
- `rapm.py` and `03_run_rapm_analysis.py` support optional season fixed effects via `--season-effects`; in `ALL` runs these now split regular season and playoffs into separate `(season, RS/PS)` nuisance coefficients instead of blending them. See [Pipeline Quickstart](./PIPELINE_QUICKSTART.md) and [RAPM Methodology](./RAPM_METHODOLOGY.md).
- `rapm.py` now also supports `--age-dummies` for RAPM and factor solves, adding separate offense and defense age-bin fixed effects (ages 18-38, 38+ collapsed into 38) and writing a companion `*_age_effects.csv`; usage lives in [Age-Dummy RAPM](./AGE_DUMMY_RAPM.md).
- `scripts/build_season_intercepts.py` now materializes raw column means plus solver-facing seasonal intercept artifacts from the processed parquet surface, using filename-derived season keys and an audit file to expose legacy parquet `Season` mismatches.
- `02_process_rapm.py` and `rapm.py` now support `ASSIST_POINTS`, `RIM_ASSIST`, `DUNK`, `DUNK_ASSIST`, `BLOCK_RECOVERY`, `STOP`, `THREE_FREQ`, `THREE_FG_PCT`, and `MIDRANGE_FG_PCT`; definitions live in [RAPM Methodology](./RAPM_METHODOLOGY.md).
- `ASSIST_POINTS_RECEIVER` and `ASSIST_POINTS_PASSER` expand each standard possession to five focal-player opportunity rows. `scripts/run_assist_points_hierarchy.py` fits their passer-minus-receiver contrast on the common resolved sample, then anchors it to parent offensive `ASSIST_POINTS` so the final passer and receiver RAPMs add exactly to the parent.
- `SCORING_ONBALL` and `SCORING_OFFBALL` provide an analogous shooter/non-shooter hierarchy. Scoring shots and FTs are on-ball; each non-shooter gets one quarter of the off-ball target. `scripts/run_scoring_role_hierarchy.py` anchors the role contrast to `SCORING_PARENT` so the two player components add exactly to offensive scoring RAPM.
- `TS_SELF_SHOT` and `TS_TEAMMATE_SHOT` split the existing turnover-free TS attempt sample by who takes the shot. `scripts/run_ts_shot_role_hierarchy.py` anchors the conditional self-minus-teammate contrast to parent offensive TS RAPM and exports exact self-shot, teammate-shot, and shot-mix Shapley components. The conditional teammate-shot field is the preferred nonshooter diagnostic; it is not optical off-ball tracking.
- `scripts/run_joint_ts_shooter_interaction.py` is the preferred oTS role model. It fits one attempt-level regression with ordinary offense/defense effects plus a player-season-centered shooter interaction, selects the role alpha with grouped game CV and the one-SE rule, and exports exact direct self-shot plus teammate-shot contributions. The older two-solve Shapley hierarchy remains an audit prototype.
- `01b_enrich_pbp_shotquality.py`, `02_process_rapm.py`, and `rapm.py` now support `PLAYTYPE_TS_MIX`, a descriptor-based playtype-mix expected scoring RAPM for ShotQuality seasons.
- `01b_enrich_pbp_shotquality.py`, `02_process_rapm.py`, and `rapm.py` now support `PLAYTYPE_PROXY_PTS`, a Synergy-proxy shot-mix RAPM that assigns every FGA to `transition`, `cut`, `spot_up`, `unassisted_3`, `unassisted_2`, or `other_assisted_2` and scores each shot by the season-relative actual-points value of that bucket.
- `scripts/analyze_pbp_playtype_difficulty_ots.py` is the distinct own-shooter difficulty test: its PBP-only categories ignore shot outcome and made-only assist tags, use immediately prior-season prices, and predict annual future oTS. The primary comparison is the decompressed floating-three model—raw `pts100`, `TSA100`, and volume-weighted role difficulty enter separately—with feature-season on-ball percentage, potential assists/100, and rim assists/100 in every model. On 735 chronological player-seasons the floating baseline is `1.4940` RMSE; PBP reaches `1.4744` (`-0.0197 [-0.0383,-0.0024]`), while Synergy reaches `1.4860` and is inconclusive. PBP's role-to-raw-points multiplier is `0.767` and `0.779`; its `0.0116` RMSE point-estimate edge over Synergy has an interval crossing zero. This is spent-window development evidence, not promotion; see [PBP-Only Playtype Difficulty and Future oTS](./PBP_PLAYTYPE_DIFFICULTY_OTS.md).
- `rapm.py` now supports optional grouped alpha cross-validation via `--cv-alpha`, selecting offense and defense alphas separately over a configurable grid and writing a companion `*_alpha_cv.csv` summary artifact.
- `rapm.py` now supports optional direct publishing via `--publish-to-rapms`, copying the main result CSV into local `master_results/` and the downstream `rapms` repo before committing/pushing that repo when the published file changes.
- `03_run_rapm_analysis.py` supports orchestration tuning via `--rapm-workers` for concurrent RAPM subprocesses and `--cores-per-rapm` for the `rapm.py --cores` value passed to each subprocess; defaults preserve the existing 4 concurrent RAPM jobs and auto-selected per-RAPM load workers.
- `03_run_rapm_analysis.py --publish-to-rapms` copies the completed run's standard and alt3 weighted-factor CSVs into `/Users/russellthomas/Docs/rapms/master_results/`, commits only those files, and pushes the downstream `rapms` repo if they changed.
- `build_master_frequency_rapm.py` now builds a combined rim/midrange/three frequency + FG% RAPM export with readable column labels in `master_results/`.
- `02_process_rapm.py`, `rapm.py`, and `03_run_rapm_analysis.py` support the legacy alternate clean 3-factor path: `FIRST_CHANCE`, `SECOND_CHANCE_CLEAN`, plus the `ALT_TS` / `ALT_TOV` aliases and both Alt3 output families. These are retained for audit and migration work.
- Net-continuation is the selected forward public decomposition. The owner-expanded artifact contract is all 140 available rolling 1Y-5Y windows from 1997 through 2026 for actual Net, common-alpha SQ-v2 plus residual Net, and TD550 component-alpha SQ-v2 research VPM. The restartable `run_rapm_overnight.py --full-window-grid` mode fills missing cuts while preserving separate filenames and product labels. Reversible consumer migration follows completion. Gross `weighted_factors_decomp_*` remains compatibility/rollback and both Alt3 output families remain legacy/audit. See [Net-Continuation DECOMP](./NET_CONTINUATION_DECOMP.md) and [Gross Eight-Component DECOMP](./PUBLIC_DECOMP.md).
- The separate historical/as-of net workflows remain reproducible: `run_net_sq_decomp_rolling.py --all-one-to-five-season-windows` targets the complete 140-cut SQ grid while preserving the completed 26-window five-season backbone; `run_quarterly_td_rapm.py` builds four game-count checkpoint estimates for every complete trailing-six-season window at TD550; and `run_vpm_netcont_asof.py` builds no-prior net-VPM boards on October 1, January 1, May 1, and August 30 for season-end years 2020-2026 with cut-local partial-season targets. `publish_no_prior_vpm_snapshots.py` publishes the 34,075 player-date rows to the indexed, RLS-protected Supabase table `nba_no_prior_vpm_td550_snapshots`. Those VPM and quarterly artifacts retain their research labels; selecting Net DECOMP for explanation does not promote predictive VPM. See [Rolling net DECOMP and quarterly TD550 RAPM](./ROLLING_NET_DECOMP_AND_QUARTERLY_TD.md).
- `scripts/build_team_game_net_decomp.py` rolls the isolated possession rows into a descriptive team-vs-team game ledger. It exports an eight-component net-continuation view and a shooter-specific `RUSSELL_SHOTQUALITY v2` expected-versus-realized view sourced from `shot_quality_season_ev`—never provider `initial_ev`—then adds realized FT conversion versus expectation so both close exactly to the final score at game and series grain. This is retrospective team accounting, not a one-game player RAPM or causal attribution; see [Team Game-Log Net DECOMP](./TEAM_GAME_NET_DECOMP.md).
- `FIRST_CHANCE` now carries first-chance FT and non-FT scoring splits on all seasons (`FC_FT_Diff`, `FC_EFG_Diff`), and ShotQuality seasons also add the deeper `FC_SQ_Diff` / `FC_MAKE_Diff` split.
- `rapm.py` and `03_run_rapm_analysis.py` now expose `ALT_EFG` and `ALT_FT` across all alt3 windows while keeping `oALT_TS` / `dALT_TS` in the published alt3 CSVs for backwards-compatible schema; ShotQuality-era files still add `oALT_SQ` / `oALT_MAKE` as a deeper drilldown.
- `03_run_rapm_analysis.py` recenters legacy `weighted_factors_alt3_*` factor columns in place by player possession weights before rebuilding `oFC` / `dFC` and residual `oSC` / `dSC`. Use `scripts/center_published_alt3_weighted_factors.py` only for older legacy-file repairs.
- `FIRST_CHANCE.Is_Turnover` now excludes turnovers that happen after the first offensive miss, so `ALT_TOV` is a true first-chance turnover surface.
- `rapm.py` now also exposes `ALT_BADPASS_TOV` and `ALT_SCORING_TOV`; for published rolling `weighted_factors_alt3_*_all_rb_se_a2000_4000.csv` repairs, use `scripts/patch_alt3_tov_value_splits.py` to populate `oALT_TOV_bp`, `oALT_TOV_sc`, `dALT_TOV_bp`, and `dALT_TOV_sc` from the point-valued `ALT_BADPASS_TOV_VALUE` / `ALT_SCORING_TOV_VALUE` aliases and write fixed three-decimal CSVs.
- `daily_rapm_update.sh` historically published curated Alt3 master-results exports into the downstream `rapms` GitHub repo. Both Alt3 families are now audit outputs. Gross `weighted_factors_decomp_*` remains the live compatibility publish until the audited 1Y-5Y `weighted_factors_decomp_net_*` cutover.
- The downstream `rapms` publish set now also syncs `rim_assist_24_26_{all,rs}_results.csv` from `nba_pipeline/results` alongside the weighted-factors files in `master_results`.
- `daily_rapm_update.sh` now maps the solver's canonical `transition_freq_24_26_all_td700_results.csv`, `transition_rim_24_26_all_td700_results.csv`, and `initial_ev_24_26_all_td700_results.csv` outputs onto the published downstream filenames `transitionfreq_24_26_all_td700_results.csv`, `transitionrim_24_26_all_td700_results.csv`, and `initialev_24_26_all_td700_results.csv` so `rapms` does not keep re-copying stale pre-rename files.
- `daily_rapm_update.sh` now copies `special_rapm_24_26_all_results.csv` into `master_results` and publishes it into the downstream `rapms` repo alongside the standalone ShotQuality metric exports.
- `scripts/lineup_stats.py` now materializes a 2026 lineup WOWY table from raw PBP counts in `processed/LINEUP_STATS26.parquet` and supports exact/include/exclude lineup queries plus player on/off queries keyed by NBA IDs; usage lives in [Pipeline Quickstart](./PIPELINE_QUICKSTART.md).
- `scripts/upload_wowy_team_player_presence.py` builds the Supabase `wowy_team_player_presence` table from raw `home_player1..5` / `away_player1..5` lineup columns. Databallr joins this table to `player_alt3_efg_factors` for WOWY team RAPM DECOMP, so `timedecay_rapm` keeps its current-strength meaning.
- `scripts/build_clean_lineup_net_rating.py` starts the duplicate raw action-score possession parquet path for Databallr WOWY/net rating. It treats every `EndOfPeriod` row as a hard segment boundary, drops empty admin-only period-end segments, uses action-score points to avoid replay score-correction artifacts, and preserves foreign-side technical/admin FT scoring as zero-possession rows for the shooting team. See [Clean Lineup Net-Rating Possessions](./LINEUP_NET_RATING_PARQUET.md).
- `scripts/build_weighted_factors_galaxy.py` now builds reusable 3D galaxy payloads from arbitrary `weighted_factors*.csv` inputs, with built-in 6-factor and 8-factor presets plus derived `oEFG/dEFG` support.

- 2026-10-19: rZTOV / scoring-security lane point prices now regress on near-unpenalized (alpha 20/40) lane TOV RAPM from `scripts/build_tov_price_targets.py`; the old alpha-2000/4000 target shrank them by about 40%. Observed 2022-26 is now `0.902` scoring / `1.020` bad pass, up from `0.535` / `0.767`. See [rZTOV points conversion](RZTOV.md#points-conversion).
- 2026-09-05: NBA/WNBA estimated rZTOV shot-role volumes now partition canonical PBP TSA100; see [NBA](NBA_RZTOV_BOX_BACKCAST.md#2026-09-05-canonical-tsa-correction) and [WNBA](WNBA_RZTOV.md#2026-09-05-canonical-tsa-correction).

- [rZTOV team reconciliation](RZTOV_TEAM_RECONCILIATION.md): turnover coverage, source gaps, and squad-level workload-estimation effects.

- [Team workload calibration](RZTOV_TEAM_CALIBRATION.md): nested chronological NBA and WNBA held-out role diagnostics; no automatic rating adjustment.

- [Complete team-page rZTOV accounting](RZTOV_TEAM_PAGE_ACCOUNTING.md): all-player coverage and a separate team-turnover baseline.

- [Same-sample RS rZTOV correction](RZTOV_SAME_SAMPLE_CORRECTION.md): replaces the mixed-snapshot reader TSA join; corrected NBA team numbers and validation.

- [Genius/rZTOV label audit](RZTOV_GENIUS_LABEL_AUDIT.md): confirmed defensive-rebound false positives in putback reclassification; current team interpretation requires repair.

- [Team shot-quality distributions](./TEAM_SHOT_QUALITY_DISTRIBUTION.md) — descriptive team offense/defense opportunity distributions, FT TSA prototype, and home-run threshold.

- [Deflections playoff context](NBA_DEFLECTIONS_PLAYOFF_CONTEXT.md): prior-season matched-player context test; retain next-game OU control.

- [Player game net points](PLAYER_GAME_NET_POINTS.md): research six-factor game credit, historical priors, observed actor signals, and explicit opponent-three conversion bridge.

- [Game box coefficient review](PLAYER_GAME_BOX_NET_POINTS.md): traditional box → learned six-factor coefficients → explicit team adjustment; PBP difficulty/zTS proxy and prior weights shown separately.

- Player attribution event audit: see [PLAYER_GAME_BOX_NET_POINTS.md](PLAYER_GAME_BOX_NET_POINTS.md#possession-attribution-audit-and-featuretarget-mismatch) for the first/second-chance steal mapping and one-game reader.

- [Tracking playoff quartiles](TRACKING_PLAYOFF_QUARTILES.md): seven-stat exploratory audit with coverage gates and player-cluster intervals.

- [Tracking next-game filters](NBA_TRACKING_NEXT_GAME.md): six-stat next-game OU vs random-walk harness, frozen 2023-2025 stress, dated 2026 states.

- [Fouls-drawn next-game filters](NBA_FOULS_DRAWN_NEXT_GAME.md): CDN `foulDrawnPersonId` ledger with reconciliation against repo PBP and the official box, three-stat next-game OU vs random-walk harness (2020 warm-up, 2021-2022 tune, frozen 2023-2025 stress), dated 2026 states.

- [Predictive box-score stats registry](NBA_PREDICTIVE_BOX_STATS_REGISTRY.md): curated registry of all 28 live stats and 20 lanes with plain-English state notes; `build_estimated_skills_registry.py` validates it, reads the frozen receipts, and writes the status/values/evidence snapshot and reader under `results/estimated_skills/registry/`.

- [Current skill comparison](CURRENT_SKILL_COMPARISON.md): local homegrown-v1 versus EPM matching-category table.

- [Three-point volume prior](THREE_PCT_VOLUME_PRIOR.md): 3PA/100-informed Kalman challenger and next-game results.

- [Shooting fusion protocol](SHOOTING_FUSION.md) and [results](SHOOTING_FUSION_RESULTS.md): chronological cross-skill prior, linear and boosted challengers with joint makes gate.

- [Rim blocked-shot feature](RIM_BLOCKED_PRIOR.md): BLKA/FGA incremental test and derived combined 2P%.

- [Next-game attempt rates versus EPM](ATTEMPT_RATES_NEXT_GAME_EPM.md): matched snapshot-date and next-appearance tests.

- [Deflections-to-steals increment](DEFLECTIONS_STEALS_INCREMENT.md): positive historical next-game increment beyond steals-only calibration.

- [Descriptive league seasonality](LEAGUE_SEASONALITY.md) and [predictive team-game seasonality](TEAM_GAME_SEASONALITY.md): separate observed calendar patterns from validated game context.

- [Physical features for estimated skills](ESTIMATED_SKILLS_PHYSICAL_FEATURES.md): height/wingspan adapter, coverage and historical-date restrictions.

- [Rim contest savings and matchup burden](RIM_DEFENSE_MATCHUP_SKILLS.md): six-foot defense game logs, BAM assignment load, coverage and predictive-admission boundaries.

- [Rolling predictive rim points saved](PREDICTIVE_RIM_SAVED.md): dated contest/suppression estimates, corrected source grain, and next-game/block-control validation.

- [Shooter and Teammate ShotQuality Split](./SQ_SHOOTER_TEAMMATE_SPLIT.md) — owner-requested 2024–2026 RS FGA-only own/teammate SQ and residual contributions, with talent-aware provider and improved-hybrid comparisons.

- [Five-Year Russell v2 Six-Part Shooting Split](./V2_SIX_SHOOTING_SPLIT.md) — current own/teammate SQ-rim, mid/three residual and restored DECOMP expected-FT premium; retuned six offensive alphas, chronological validation, standard oTS comparison, clickable per-player gain accounting and downloadable conditional rates.

- [Burden-adjusted teammate SQ](./BURDEN_ADJUSTED_TEAMMATE_SQ.md) — cross-fitted role/talent baseline and research teammate surplus; split-game reliability and creation checks.

- [Own and teammate turnover decomposition](./OWN_TEAMMATE_TOV.md) — parent-anchored player-attributed turnover cost/suppression, point-valued display with preserved raw rates, separate possession units, CV and coverage.
