# NBA Predictive Box-Score Stats Registry

`nba_pipeline/config/estimated_skills_registry.json` is the one curated list of every
predictive box-score stat the NBA estimated-skills family produces, what model each one is
on the table with, what the best known model is, what evidence backs it, and what happens
next. `nba_pipeline/scripts/build_estimated_skills_registry.py` validates that file, pulls
the evidence out of the frozen receipts, joins the current per-player values into one
table, and writes a self-contained reader. This page is the plain-English companion: it
carries the same status notes as the registry so a reader can see the state of every stat
without opening JSON.

Registry as of 2026-09-14. Owner: Russell Thomas. Twenty lanes, 28 stats on
the values table, all with the status `live`.

## Contract

- **Role:** research/forecast lane only; nothing here is a rating prior, a solve prior, or a VPM input
- **Objective:** next-game exposure-weighted squared error (per 100 possessions for rates, per attempt for percentages); rest-of-season aggregates are secondary
- **Pre-tip rule:** every feature must be constructible before tip from information available live (completed-game rosters, schedule, prior-game residuals)
- **Observations:** measured observations only; *_filled values, inferred zeros and estimates are never observations; an unmeasured game is NaN, never zero
- **Holdout:** 2026 is export-only or spent limited evidence; never next-game scored by v4 or any lab recipe; v1 was scored once against naive baselines on 2026-08-27 and the v3 forward confirmation ran once; 2027 is untouched.
- **Providers:** EPM, DARKO, DPM and eWINS are never features, targets or tuning inputs, and provider values are never copied into published or committed artifacts
- **Artifacts:** everything under nba_pipeline/results/estimated_skills/ is generated and stays out of git; this registry and its builder are the checked-in source

## Status vocabulary

- **live**: a dated current per-player estimate is exported and appears on the values table; the model is frozen and 2026 is export-only
- **development**: a model is selected on development seasons but has no current-state export, so the table still shows the incumbent
- **increment**: an increment that passed development and frozen stress but is not exported to any table
- **measurement**: a measurement layer only (retrospective baseline); not a forecast
- **imputation**: a postgame repair layer that prepares inputs; its filled values are never observations
- **diagnostic**: descriptive or comparison output only; nothing is estimated for the next game
- **failed**: tested against a pre-declared gate and rejected; nothing on any table changed

Every stat on the values table is `live`. The other statuses describe lanes that support,
diagnose, or failed to improve those stats.

## Stats on the values table

Evidence is the headline line from the 2026-09-14 snapshot: paired next-game WMSE gain in
percent with a game-clustered 95% interval and per-season wins where the receipt has them.
Percent gains are small by construction because game rows are mostly binomial noise; read the
interval and the season wins alongside the point estimate.

| stat_id | label | lane | side | units | on the table | dev-selected | headline evidence |
|---|---|---|---|---|---|---|---|
| `rim_pct` | Rim FG% | 1 | offense | % | v1 | v4 + lab recipe G | v1 (table) vs trail10 +8.930% (dev 2020-2025); v4 (dev-selected) vs v1 (table) +0.500% (dev 2020-2025); lab recipe G (ctxps+ctxmult+xusg+jump) vs re-tuned v4 +0.053% [+0.034, +0.075] (dev 2020-2025) |
| `mid_pct` | Mid-range FG% | 1 | offense | % | v1 | v4 (no lab gain) | v1 (table) vs trail10 +9.410% (dev 2020-2025); v4 (dev-selected) vs v1 (table) +0.161% (dev 2020-2025) |
| `three_pct` | 3P% | 1 | offense | % | v1 + three-point volume prior (lane 11, strength 0.5) | v4 (no lab gain) | v1 (table) vs trail10 +9.672% (dev 2020-2025); v4 (dev-selected) vs v1 (table) +0.194% (dev 2020-2025) |
| `ft_pct` | FT% | 1 | offense | % | v1 | v4 + lab recipe D | v1 (table) vs trail10 +8.095% (dev 2020-2025); v4 (dev-selected) vs v1 (table) +0.151% (dev 2020-2025); lab recipe D (ctxplus+ctxmult+xusg+jump) vs re-tuned v4 +0.021% [+0.003, +0.040] (dev 2020-2025) |
| `rim_fga_rate` | Rim FGA /100 | 1 | offense | per 100 offensive possessions | v1 | v4 + lab recipe F | v1 (table) vs trail10 +6.334% (dev 2020-2025); v4 (dev-selected) vs v1 (table) +1.158% (dev 2020-2025); lab recipe F (ctxplus+ctxmult+xall+jump) vs re-tuned v4 +0.285% [+0.215, +0.357] (dev 2020-2025) |
| `mid_fga_rate` | Mid FGA /100 | 1 | offense | per 100 offensive possessions | v1 | v4 + lab recipe F | v1 (table) vs trail10 +5.583% (dev 2020-2025); v4 (dev-selected) vs v1 (table) +2.229% (dev 2020-2025); lab recipe F (ctxplus+ctxmult+xall+jump) vs re-tuned v4 +0.705% [+0.567, +0.831] (dev 2020-2025) |
| `three_fga_rate` | 3PA /100 | 1 | offense | per 100 offensive possessions | v1 | v4 + lab recipe F | v1 (table) vs trail10 +5.462% (dev 2020-2025); v4 (dev-selected) vs v1 (table) +1.336% (dev 2020-2025); lab recipe F (ctxplus+ctxmult+xall+jump) vs re-tuned v4 +0.301% [+0.227, +0.375] (dev 2020-2025) |
| `fta_rate` | FTA /100 | 1 | offense | per 100 offensive possessions | v1 | v4 + lab recipe F | v1 (table) vs trail10 +6.669% (dev 2020-2025); v4 (dev-selected) vs v1 (table) +0.802% (dev 2020-2025); lab recipe F (ctxplus+ctxmult+xall+jump) vs re-tuned v4 +0.299% [+0.223, +0.379] (dev 2020-2025) |
| `pts_rate` | PTS /100 | 1 | offense | per 100 offensive possessions | v1 | v4 + lab recipe F | v1 (table) vs trail10 +6.328% (dev 2020-2025); v4 (dev-selected) vs v1 (table) +0.830% (dev 2020-2025); lab recipe F (ctxplus+ctxmult+xall+jump) vs re-tuned v4 +0.504% [+0.426, +0.582] (dev 2020-2025) |
| `ast_rate` | AST /100 | 1 | offense | per 100 offensive possessions | v1 | v4 + lab recipe D | v1 (table) vs trail10 +5.617% (dev 2020-2025); v4 (dev-selected) vs v1 (table) +1.255% (dev 2020-2025); lab recipe D (ctxplus+ctxmult+xusg+jump) vs re-tuned v4 +0.439% [+0.358, +0.519] (dev 2020-2025) |
| `tov_rate` | TOV /100 | 1 | offense | per 100 offensive possessions | v1 | v4 + lab recipe F | v1 (table) vs trail10 +7.685% (dev 2020-2025); v4 (dev-selected) vs v1 (table) +0.838% (dev 2020-2025); lab recipe F (ctxplus+ctxmult+xall+jump) vs re-tuned v4 +0.321% [+0.235, +0.400] (dev 2020-2025) |
| `oreb_rate` | OREB /100 | 1 | offense | per 100 offensive possessions | v1 | v4 + lab recipe F | v1 (table) vs trail10 +8.266% (dev 2020-2025); v4 (dev-selected) vs v1 (table) +0.338% (dev 2020-2025); lab recipe F (ctxplus+ctxmult+xall+jump) vs re-tuned v4 +0.111% [+0.062, +0.160] (dev 2020-2025) |
| `usg_pct` | Usage % | 1 | offense | per 100 offensive possessions | v1 | v4 + lab recipe F | v1 (table) vs trail10 +5.426% (dev 2020-2025); v4 (dev-selected) vs v1 (table) +1.343% (dev 2020-2025); lab recipe F (ctxplus+ctxmult+xall+jump) vs re-tuned v4 +0.270% [+0.195, +0.343] (dev 2020-2025) |
| `dreb_rate` | DREB /100 | 1 | defense | per 100 defensive possessions | v1 | v4 + lab recipe G | v1 (table) vs trail10 +7.007% (dev 2020-2025); v4 (dev-selected) vs v1 (table) +0.675% (dev 2020-2025); lab recipe G (ctxps+ctxmult+xusg+jump) vs re-tuned v4 +0.063% [+0.029, +0.099] (dev 2020-2025) |
| `stl_rate` | STL /100 | 1 | defense | per 100 defensive possessions | v1 | v4 + lab recipe D | v1 (table) vs trail10 +8.779% (dev 2020-2025); v4 (dev-selected) vs v1 (table) +0.409% (dev 2020-2025); lab recipe D (ctxplus+ctxmult+xusg+jump) vs re-tuned v4 +0.041% [+0.014, +0.066] (dev 2020-2025) |
| `blk_rate` | BLK /100 | 1 | defense | per 100 defensive possessions | v1 | v4 + lab recipe D | v1 (table) vs trail10 +7.845% (dev 2020-2025); v4 (dev-selected) vs v1 (table) +0.817% (dev 2020-2025); lab recipe D (ctxplus+ctxmult+xusg+jump) vs re-tuned v4 +0.364% [+0.277, +0.460] (dev 2020-2025) |
| `pf_committed_rate` | PF committed /100 | 1 | both | per 100 possessions (both sides) | v1 | v4 + lab recipe F | v1 (table) vs trail10 +7.575% (dev 2020-2025); v4 (dev-selected) vs v1 (table) +0.551% (dev 2020-2025); lab recipe F (ctxplus+ctxmult+xall+jump) vs re-tuned v4 +0.291% [+0.231, +0.346] (dev 2020-2025) |
| `deflections` | Deflections /100 def | 2 | defense | per 100 defensive possessions | next_ou (plain OU, no corrections, no context) | same as table (frozen) | OU vs re-tuned random walk (selection) +0.191% [+0.118, +0.263] 3/3 (dev 2020-2022); next_ou vs trail10 +9.056% [+7.187, +12.270] 3/3 (frozen stress 2023-2025); next_ou vs next_random_walk +0.129% [+0.057, +0.202] 3/3 (frozen stress 2023-2025) |
| `potential_assists` | Potential assists /100 off | 19 | offense | per 100 offensive possessions | next_ou (plain OU) | same as table (frozen) | OU vs re-tuned random walk (selection) +0.593% [+0.430, +0.756] 3/3 (dev 2020-2022); next_ou vs trail10 +6.303% [+5.639, +7.265] 3/3 (frozen stress 2023-2025); next_ou vs next_random_walk +0.858% [+0.706, +1.015] 3/3 (frozen stress 2023-2025) |
| `drives` | Drives /100 off | 19 | offense | per 100 offensive possessions | next_ou (plain OU) | same as table (frozen) | OU vs re-tuned random walk (selection) +0.998% [+0.775, +1.212] 3/3 (dev 2020-2022); next_ou vs trail10 +5.865% [+5.391, +6.368] 3/3 (frozen stress 2023-2025); next_ou vs next_random_walk +1.144% [+0.952, +1.341] 3/3 (frozen stress 2023-2025) |
| `possession_minutes` | Possession minutes /100 off | 19 | offense | per 100 offensive possessions | next_ou (plain OU) | same as table (frozen) | OU vs re-tuned random walk (selection) +2.502% [+2.020, +2.964] 3/3 (dev 2020-2022); next_ou vs trail10 +10.702% [+9.924, +11.489] 3/3 (frozen stress 2023-2025); next_ou vs next_random_walk +2.480% [+2.088, +2.871] 3/3 (frozen stress 2023-2025) |
| `contested_shots` | Contested shots /100 def | 19 | defense | per 100 defensive possessions | next_ou (plain OU; frozen despite a stress reversal) | same as table (frozen) | OU vs re-tuned random walk (selection) +0.211% [+0.122, +0.293] 3/3 (dev 2020-2022); next_ou vs trail10 +5.959% [+5.329, +6.634] 3/3 (frozen stress 2023-2025); next_ou vs next_random_walk -0.098% [-0.189, -0.010] 0/3 (frozen stress 2023-2025) |
| `dfga` | Defended FGA /100 def | 19 | defense | per 100 defensive possessions | next_ou (plain OU) | same as table (frozen) | OU vs re-tuned random walk (selection) +0.339% [+0.247, +0.426] 3/3 (dev 2020-2022); next_ou vs trail10 +7.340% [+6.878, +7.823] 3/3 (frozen stress 2023-2025); next_ou vs next_random_walk +0.269% [+0.184, +0.359] 3/3 (frozen stress 2023-2025) |
| `contested_dreb` | Contested DREB /100 def | 19 | defense | per 100 defensive possessions | next_ou (plain OU) | same as table (frozen) | OU vs re-tuned random walk (selection) +0.084% [+0.026, +0.145] 3/3 (dev 2020-2022); next_ou vs trail10 +8.894% [+7.948, +10.077] 3/3 (frozen stress 2023-2025); next_ou vs next_random_walk +0.127% [+0.062, +0.189] 3/3 (frozen stress 2023-2025) |
| `off_fouls_drawn` | Offensive fouls drawn /100 def | 20 | defense | per 100 possessions | next_ou (plain OU; near-equivalent to the random walk) | same as table (frozen) | OU vs re-tuned random walk (selection) +0.062% [-0.007, +0.131] 2/2 (dev 2021-2022); next_ou vs trail10 +7.767% [+7.195, +8.366] 3/3 (frozen stress 2023-2025); next_ou vs next_random_walk +0.116% [+0.038, +0.194] 3/3 (frozen stress 2023-2025) |
| `charges_drawn` | Charges drawn /100 def | 20 | defense | per 100 possessions | next_ou (plain OU; near-equivalent to the random walk) | same as table (frozen) | OU vs re-tuned random walk (selection) +0.032% [-0.024, +0.091] 2/2 (dev 2021-2022); next_ou vs trail10 +8.465% [+7.432, +9.533] 3/3 (frozen stress 2023-2025); next_ou vs next_random_walk +0.116% [+0.040, +0.202] 3/3 (frozen stress 2023-2025) |
| `pfd` | Personal fouls drawn /100 off | 20 | offense | per 100 possessions | next_ou (plain OU) | same as table (frozen) | OU vs re-tuned random walk (selection) +0.287% [+0.171, +0.400] 2/2 (dev 2021-2022); next_ou vs trail10 +6.549% [+6.157, +6.959] 3/3 (frozen stress 2023-2025); next_ou vs next_random_walk +0.348% [+0.244, +0.446] 3/3 (frozen stress 2023-2025) |
| `rim_points_saved` | Rim points saved /100 def | 9 | defense | points per 100 defensive possessions | exponential decay (730-day half-life, K=100 prior attempts), not the OU engine | same as table (frozen) | decay model vs prior-mean baseline, points saved +1.220% [+0.986, +1.450] 2/2 (frozen stress 2024-2025); rim saved vs blocks alone +0.091% [-0.092, +0.252] 2/2 (frozen stress 2024-2025); blocks + rim saved vs blocks alone +0.249% [+0.163, +0.330] 2/2 (frozen stress 2024-2025) |

### State notes

- **Rim FG%** (`rim_pct`, lane 1): On the table as v1. The only percentage the lab moves: the playoff term is worth about +0.05% on dev. The rim blocked-shot feature (lane 10) failed and changed nothing. *Next:* Promote with the rest of the panel when a v5 builder exists; no stat-specific work pending.
- **Mid-range FG%** (`mid_pct`, lane 1): On the table as v1. No lab recipe: every extra feature is null for mid-range percentage, so v4 is the ceiling under the current feature set. *Next:* None beyond the panel-wide v5 build.
- **3P%** (`three_pct`, lane 1): On the table as v1 blended with the owner-authorized volume prior (low-volume shooters are pulled toward the population curve for their attempt rate). The plain v1 value is kept beside it. No lab recipe helps 3P%; the untested idea is a volume-dependent attractor inside the filter. *Next:* Test a volume-dependent attractor as a lab option; keep the volume prior on the table until then.
- **FT%** (`ft_pct`, lane 1): On the table as v1. Recipe D is worth about +0.02% on dev (mostly the team-change jump); effectively at the ceiling. *Next:* None beyond the panel-wide v5 build.
- **Rim FGA /100** (`rim_fga_rate`, lane 1): On the table as v1. Recipe F (all-skill form + jump) adds +0.29% on dev over re-tuned v4. Same-day 2026 comparison vs EPM was inconclusive; next appearance within 7 days favoured v1 by +1.5% (lane 15, spent). *Next:* Build v5 with recipe F; do not rerun the 2026 EPM comparison until the owner pre-registers one pass.
- **Mid FGA /100** (`mid_fga_rate`, lane 1): On the table as v1. The biggest lab gainer: v4 context is worth +1.64% over OU and recipe F a further +0.71% on dev. The gap between the table (v1) and the best known model is largest here. *Next:* First stat to promote once a v5 builder exists.
- **3PA /100** (`three_fga_rate`, lane 1): On the table as v1. v4 context +1.06% over OU; recipe F +0.30% more on dev. The minutes trend is worth keeping here (+0.03%). *Next:* Build v5 with recipe F.
- **FTA /100** (`fta_rate`, lane 1): On the table as v1. Recipe F +0.30% on dev; usage form and minutes trend both matter for free-throw rate. Personal fouls drawn (lane 20) is the CDN-measured companion stat. *Next:* Build v5 with recipe F; later test PFD as a cross-family form feature.
- **PTS /100** (`pts_rate`, lane 1): On the table as v1. Recipe F +0.50% on dev, 6/6 seasons; the all-skill form block is worth +0.17% of that. *Next:* Build v5 with recipe F.
- **AST /100** (`ast_rate`, lane 1): On the table as v1. Recipe D +0.44% on dev; the playoff term carries most of it and the full form block does not help assists. Potential assists (lane 19) is the tracking companion. *Next:* Build v5 with recipe D.
- **TOV /100** (`tov_rate`, lane 1): On the table as v1. Recipe F +0.32% on dev; the all-skill form block alone is +0.23%. *Next:* Build v5 with recipe F.
- **OREB /100** (`oreb_rate`, lane 1): On the table as v1. Recipe F +0.11% on dev but only 4/6 seasons; D, F and G are within each other's intervals, so one rebound recipe (G) is acceptable. *Next:* Build v5; use the rebound recipe.
- **Usage %** (`usg_pct`, lane 1): On the table as v1. v4 adds pre-tip teammate absence (+0.30 pp fair increment; the withdrawn realised-roster version was inflated) and recipe F a further +0.27% on dev. Team-change jump is worth +0.12% here. *Next:* Build v5 with recipe F.
- **DREB /100** (`dreb_rate`, lane 1): On the table as v1. Recipe G +0.06% on dev; absence features are null for rebounds. Contested DREB (lane 19) is the tracking companion. *Next:* Build v5 with the rebound recipe.
- **STL /100** (`stl_rate`, lane 1): On the table as v1. Recipe D +0.04% on dev. The bigger known gain is outside the panel: pregame deflections add +0.41% on stress (lane 5) and are not exported. *Next:* In v5, feed deflections into steals as a context feature inside the filter instead of the post-hoc ridge.
- **BLK /100** (`blk_rate`, lane 1): On the table as v1. Recipe D +0.36% on dev; the all-skill form block hurts blocks (-0.09%), so usage form only. Rim points saved (lane 9) is not yet clearly better than blocks alone. *Next:* Build v5 with recipe D; keep the blocks-vs-rim-saved pair reported on every rim rerun.
- **PF committed /100** (`pf_committed_rate`, lane 1): On the table as v1. Recipe F +0.29% on dev; the playoff term is worth +0.20%. Fouls committed is the repo-PBP side; fouls drawn (lane 20) is the CDN side. *Next:* Build v5 with recipe F.
- **Deflections /100 def** (`deflections`, lane 2): Live and dated. OU beat the re-tuned random walk on frozen stress and beats every naive baseline by 5-11%. The longer-horizon comparator (lane 3) and the playoff offset (lane 4) are preserved but not used. Stress is spent, so any lab work must declare 2020-2025 as development and lean on 2027. *Next:* Direct port of the lab winners (multiplicative opponent/own-team residuals, playoff term, jump, cross-skill form) with 2020-2025 declared as development.
- **Potential assists /100 off** (`potential_assists`, lane 19): Live and dated. Held on stress 3/3 seasons vs the random walk. Uses only measured game logs; the backfill's repaired values (lane 6) are never observations. *Next:* Playoff term first (top-quartile PA drops -1.68 in the playoffs per lane 7), then the rest of the lab winners, with 2020-2025 declared as development.
- **Drives /100 off** (`drives`, lane 19): Live and dated. Second-largest OU gain in the tracking family; held on stress 3/3. *Next:* Same lab port as potential assists.
- **Possession minutes /100 off** (`possession_minutes`, lane 19): Live and dated. Largest OU gain in the family (+2.5% on stress vs the random walk) because on-ball time is the most persistent tracking stat. *Next:* Same lab port; this is the stat most likely to benefit from the usage-form feature.
- **Contested shots /100 def** (`contested_shots`, lane 19): Live and dated, with a caveat: OU won on dev but lost to the random walk on stress (about -0.1%, 0/3 seasons). It was frozen rather than switched, so treat OU and the random walk as equivalent here. Still beats the naive baselines. *Next:* First tracking stat to re-examine in a lab (opponent shot-volume context is the obvious pre-tip feature).
- **Defended FGA /100 def** (`dfga`, lane 19): Live and dated. Held on stress 3/3. DFGA nulls mix omitted zeros with outages; only measured games are observations (inferred zeros from lane 6 are excluded). *Next:* Same lab port as the other tracking stats.
- **Contested DREB /100 def** (`contested_dreb`, lane 19): Live and dated. Smallest positive OU gain in the family but held 3/3 on stress. *Next:* Same lab port; the playoff decline for top-quartile contested rebounders (-0.44 per lane 7) makes the playoff term the first candidate.
- **Offensive fouls drawn /100 def** (`off_fouls_drawn`, lane 20): Live and dated. This is the superset: charges plus illegal screens and push-offs (the feed cannot split screens from push-offs). The dev gate barely cleared, so OU and the random walk are near-equivalent; both beat the naive baselines by 7-8%. Measured from the NBA live CDN feed, 2019-20 onward only. *Next:* Multiplicative opponent term (the fouling team's tendency is known before tip), then the playoff term; 2021-2025 declared as development.
- **Charges drawn /100 def** (`charges_drawn`, lane 20): Live and dated. The charge-descriptor subset of offensive fouls drawn; rare (about 0.13 per 100 in the warm-up season), so estimates are heavily shrunk. OU and the random walk are near-equivalent. *Next:* Same lab port as offensive fouls drawn.
- **Personal fouls drawn /100 off** (`pfd`, lane 20): Live and dated. The clearest fouls-drawn result: OU beat the random walk on dev and held 3/3 on stress. Reconciles with box PFD on every measured game. *Next:* Same lab port; also a candidate cross-family feature for the FTA-rate filter.
- **Rim points saved /100 def** (`rim_points_saved`, lane 9): Live and dated, but the weakest live lane: one development season (2023, because pre-2023 source fields were season totals), a different estimator from every other lane, and it is not yet clearly better than blocks alone (interval crosses zero). Blocks plus rim saved does beat blocks alone. Saved points combine a contest rate and a finishing suppression, exported beside the headline. *Next:* Re-express contest rate and suppression as OU states on the shared engine before any context work; report the blocks-only pair on every rerun.

## Lanes

| lane | name | status | stats on the table | state | doc |
|---|---|---|---|---|---|
| 1 | PBP Kalman panel (17 skills) | live | `t`, `h`, `e`, ` `, `1`, `7`, ` `, `p`, `b`, `p`, `_`, `*`, ` `, `s`, `t`, `a`, `t`, `s`, ` `, `a`, `b`, `o`, `v`, `e` | Table shows v1 (Aug 26, 2026 build). v4 is dev-selected (17/17 skills, 101/102 seasons vs OU, median +0.56%) but has no current-state export. Lab round 3 fixes a per-skill recipe worth a further median +0.29% (15/17 CI above zero); there is no v5 builder. 2026 is partially exposed (v1 naive-baseline holdout scored once; v3 forward confirmation scored once) and has never been next-game scored by v4 or a recipe. | nba_pipeline/docs/NBA_ESTIMATED_SKILLS.md |
| 2 | Next-game deflections OU | live | `deflections` | Live; stress spent. | nba_pipeline/docs/NBA_DEFLECTIONS_NEXT_GAME.md |
| 3 | Longer-horizon estimated deflections | diagnostic | none on the values table | Rest-of-season objective, rejected by the owner in favour of next-game error. Kept only as the comparator for lane 2 (its forward-horizon OU beats the random walk by 1.66% on stress). Not on any table. | nba_pipeline/docs/NBA_ESTIMATED_DEFLECTIONS.md |
| 4 | Deflections playoff offset | failed | none on the values table | An additive prior-season playoff offset (-0.18 to -0.39 per 100) did not clear development and went slightly negative on stress. Lane 2 unchanged. The lab's multiplicative playoff term is a different form and is still a candidate. | nba_pipeline/docs/NBA_DEFLECTIONS_PLAYOFF_CONTEXT.md |
| 5 | Deflections-to-steals increment | increment | none on the values table | Pregame deflections add a real increment to next-game steals (dev and stress both positive, coefficient 0.23-0.28 extra steals per deflection). Not exported to any table. The same-day 2026 EPM comparison was inconclusive and is spent. | nba_pipeline/docs/DEFLECTIONS_STEALS_INCREMENT.md |
| 6 | Tracking game-log backfill | imputation | none on the values table | Postgame repair of potential assists and DFGA (355,995 audited rows). Preparation only: lanes 19 and 20 read measured values and never the *_filled columns. | nba_pipeline/docs/NBA_TRACKING_BACKFILL.md |
| 7 | Tracking playoff quartiles | diagnostic | none on the values table | Descriptive RS-to-playoff rate changes by quartile (top-quartile potential assists -1.68, deflections -0.48 per 100). Regression to the mean is not removed; the playoff term must be tested inside the filters instead. | nba_pipeline/docs/TRACKING_PLAYOFF_QUARTILES.md |
| 8 | Rim contest savings and matchup burden | measurement | none on the values table | Measurement layer for lane 9. The provider season-normal baseline is retrospective, so this is not a forecast and is not on the values table; 476 ambiguous PBP rows are excluded and the 2026 matchup source is incomplete. | nba_pipeline/docs/RIM_DEFENSE_MATCHUP_SKILLS.md |
| 9 | Rolling predictive rim points saved | live | `rim_points_saved` | Live; weakest live lane (see the stat row). | nba_pipeline/docs/PREDICTIVE_RIM_SAVED.md |
| 10 | Rim blocked-shot prior | failed | none on the values table | Prior BLKA/FGA as an extra rim FG% feature fails development and stress. Nothing changed. | nba_pipeline/docs/RIM_BLOCKED_PRIOR.md |
| 11 | Three-point volume prior | live | `three_pct` | Owner-authorized on the table for 3P% only (strength 0.5). Small overall gain, large for low-volume shooters. Exported through 2026 observations without scoring 2026. | nba_pipeline/docs/THREE_PCT_VOLUME_PRIOR.md |
| 12 | Shooting fusion | failed | none on the values table | Cross-skill ridge/boosted 3P% corrections win conditional log loss but fail the joint makes gate; the selected volume+FT prior has stress gains of a few hundredths of a percent and no confirmed makes increment over the volume prior alone. Not on the table. | nba_pipeline/docs/SHOOTING_FUSION_RESULTS.md |
| 13 | Physical features | diagnostic | none on the values table | Height/wingspan adapter (509 of 582 current players covered, undated source). Candidate predictors only; the fusion test with physical features failed the joint gate. | nba_pipeline/docs/ESTIMATED_SKILLS_PHYSICAL_FEATURES.md |
| 14 | Team-game and league seasonality | failed | none on the values table | The game-number cubic selects zero on development. Descriptive calendar drift exists but is not a validated context term. Low priority. | nba_pipeline/docs/TEAM_GAME_SEASONALITY.md |
| 15 | Attempt rates vs EPM (2026) | diagnostic | none on the values table | v1 rim/mid/three attempt rates against 12 cached EPM dates on 2026: same-day inconclusive, next appearance within 7 days rim +1.5%. Spent 2026 evidence for v1, not v4. Provider rows stay private. | nba_pipeline/docs/ATTEMPT_RATES_NEXT_GAME_EPM.md |
| 16 | EPM E-Skills benchmark (2026) | diagnostic | none on the values table | Aggregate-only correlation and level-adjusted MSE receipts for v1 vs the same 12 EPM snapshots. Spent 2026 evidence; provider values never copied. | nba_pipeline/docs/NBA_ESTIMATED_SKILLS.md |
| 17 | Current skill comparison table | diagnostic | none on the values table | The local reader table the values export is built from: v1 for every PBP skill except 3P% (volume prior) plus next-game deflections. Its EPM/DARKO columns are private and local only; this registry copies none of them. | nba_pipeline/docs/CURRENT_SKILL_COMPARISON.md |
| 18 | 2026 forward confirmation (v3) | diagnostic | none on the values table | One-shot rest-of-season confirmation of v1 vs v3 (15/17, median +4.81%) on a rejected objective. This is why 2026 counts as partially exposed for lane 1; v3 must not be selected. | nba_pipeline/docs/NBA_ESTIMATED_SKILLS.md |
| 19 | Next-game tracking filters (six stats) | live | `potential_assists`, `drives`, `possession_minutes`, `contested_shots`, `dfga`, `contested_dreb` | Live; stress spent; contested shots reversed on stress but stays frozen. | nba_pipeline/docs/NBA_TRACKING_NEXT_GAME.md |
| 20 | Next-game fouls-drawn filters (three stats) | live | `off_fouls_drawn`, `charges_drawn`, `pfd` | Live; stress spent; 8,543 of 8,879 games measured from the CDN ledger. | nba_pipeline/docs/NBA_FOULS_DRAWN_NEXT_GAME.md |

## PBP panel models

- **Table model (v1):** `v1 scalar Kalman random walk (build 2026-08-26, receipts eval_report.json and hyperparams.json)`
- **Dev-selected (v4):** `v4: OU state with population attractor + debut/age/early-season corrections + pre-tip context (build_nba_estimated_skills_v4.py; v4_nextgame.json)`
- **Lab round 3 recipes:** D = `{'variant': 'ctxplus+ctxmult+xusg+jump', 'description': 'playoff term + minutes trend, multiplicative context, usage-form cross-skill feature, team-change jump'}` (ft_pct, ast, stl, blk), F = `{'variant': 'ctxplus+ctxmult+xall+jump', 'description': 'same as D with the full 16-skill cross-skill form block'}` (attempt, scoring, usage, tov, pf rates), G = `{'variant': 'ctxps+ctxmult+xusg+jump', 'description': 'D without the minutes trend (playoff term only)'}` (rebounds, rim_pct). mid_pct and three_pct have no recipe.
- v4 has no current-state export and there is no v5 builder; assemble() in research_skills_nextgame_lab.py is the specification of the recipe

## Generated snapshot

`build_estimated_skills_registry.py` writes to `nba_pipeline/results/estimated_skills/registry/`
(gitignored, like every other generated artifact under `results/`):

| file | contents |
|---|---|
| `estimated_skills_registry.json` | the curated registry plus per-stat evidence rows and value coverage |
| `registry_status.csv` / `registry_status.md` | one row per stat: models, coverage, latest state date, headline evidence, notes, next step; the MD adds the lane table |
| `registry_lanes.csv` | one row per lane with status, stats, docs, scripts, receipts and evidence |
| `registry_evidence.csv` | every evidence row: stat or lane, comparison, window, gain, interval, season wins, source receipt |
| `current_values.csv` | wide table, one row per player in the 582-player current-PBP universe: value, state date and (PBP) observation count per stat, plus companions (plain v1 3P%, rim contest rate and suppression, tracking observed-game counts) |
| `current_values_long.csv` | the same values in long form with units, status and source |
| `index.html` | self-contained reader: contract, status table, lane inventory, filterable values table with per-cell state dates, and the evidence rows |
| `manifest.json` | build time, registry hash, input receipt hashes, coverage, missing receipts and adapter errors |

Values come only from the lanes' own exports (the v1 provenance table, the next-game
`current_2026.csv` states, the rim-saved current file). A player with no measured state for a
stat stays blank; nothing is zero-filled, nothing is imputed, and the builder refuses to write
if a value column is all zeros or a receipt named in the registry is missing. No provider value
(EPM, DARKO, DPM, eWINS) is read or written, and the comparison-table provider columns are not
copied.

## Rebuild

```bash
.venv/bin/python nba_pipeline/scripts/build_estimated_skills_registry.py
```

`--validate-only` checks the registry without touching receipts. Tests:
`nba_pipeline/tests/test_estimated_skills_registry.py`.

## Editing the registry

- Add a stat when a lane exports dated current states; give it a family, lane, side, `table_model`,
  `evidence` adapter, `export` (path, id column, value column, date column) and readable `notes` /
  `next_step`.
- Change `table_model` only when the table actually changes (for the PBP panel that means a v5
  builder and a new provenance export). The registry describes what is on the table, not what won
  in the lab.
- Never put a 2026 next-game score in the registry: 2026 is the frozen, partially exposed holdout
  and 2027 is untouched. Evidence adapters read development and frozen-stress receipts only.
- Keep proprietary ratings out of every field; the validator rejects export columns that name a
  provider.
