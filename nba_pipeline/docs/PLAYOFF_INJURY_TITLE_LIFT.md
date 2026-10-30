# Playoff injury availability and title lift

Owner request: 2026-10-11. Status: source feasibility and six-case absence audit complete; injury-ledger ingestion and probability counterfactual are not implemented or published.

## What the existing model measures

`build_championship_help.py::summarize_retrospective_team_season` uses actual playoff possession exposure with entering-playoffs player ratings. Injured players therefore contribute fewer exposures to their own team's retrospective strength. A player missing the entire postseason has no row in the appearance-derived playoff roster; this is not proof that he was absent from the team roster.

The primary salary-peer Title Lift over Contract differs: `playoff_title_simulation.py::add_salary_peer_title_lift` inserts focal quality at the destination salary peer's actual RS/playoff workload. It does not measure the focal player's own availability. Keep those existing fields intact. Injury loss should first be a separate actual-team title-chance sensitivity; a standardized salary-peer availability metric requires its own placement contract.

The current bracket uses a constant team rating through all rounds and a constant rating difference through each series. A postseason-average rating cannot distinguish missing a first-round game from missing Finals Game 7. Do not claim that multiplying current title lift by games missed measures injury loss.

## Audited pilot counts

| Season ending | Player | Team | Team playoff games | Appeared | No appearance |
| --- | --- | --- | ---: | ---: | ---: |
| 2015 | Kyrie Irving | CLE | 20 | 13 | 7 |
| 2015 | Kevin Love | CLE | 20 | 4 | 16 |
| 2017 | Kawhi Leonard | SAS | 16 | 12 | 4 |
| 2019 | Kevin Durant | GSW | 22 | 12 | 10 |
| 2021 | Kawhi Leonard | LAC | 19 | 11 | 8 |
| 2024 | Giannis Antetokounmpo | MIL | 6 | 0 | 6 |

These are audited *whole-game absence counts*, independently matched between raw PBP lineup IDs and processed O/D lineup IDs, then joined to `results/championship_help/playoff_games.csv`. They are not inferred injury diagnoses. Primary injury evidence and its coverage limits are in [Playoff injury sources](./PLAYOFF_INJURY_SOURCE_RESEARCH.md).

Local artifacts: `nba_pipeline/results/playoff_injury_pilot/summary.csv`, `absent_games.csv`, and `audit.json` (six cases, 51 absent-game rows, input SHA256 hashes). Generated outputs stay out of git. Read absence dates from the ledger, not from a team's total games minus a player-season count after a trade. Partial injury games are additional lost minutes, not whole missed games; Durant's brief 2019 Finals Game 5 appearance and Kawhi's 2017 WCF Game 1 appearance are not counted as absences.

## Source and classification contract

Build one row per eligible player/team/game, including players with zero playoff appearances. Require dated playoff roster membership, exact NBA player ID and game ID, date, series, round, appearance/minutes, status, medical reason, primary source URL and timestamp, and evidence confidence. Inactive/DNP alone is insufficient. Separate injury, illness, suspension, coach decision, personal absence and unresolved status. Preserve report revisions and use actual participation to resolve a pregame status that changed before tip. Link a cited injury episode to every counted absence; an episode headline by itself is not a complete league ledger.

Modern official injury PDFs plus official final scorer reports are the preferred backbone; older years require individually verified scorer reports and team/NBA announcements. Do not claim a complete historical medical-reason archive until season-by-season source coverage is measured. No undocumented zero-minute player becomes injured by default.

## Proposed calculation

Hold regular-season qualification and seeds fixed for a playoff-only loss. Freeze one common player-quality vector before the relevant injury (entering-playoffs EPM is the first reproducible sensitivity). Use the same ratings in both paths, rather than letting missed games shrink active quality.

For each affected game, restore a declared healthy workload and remove exactly those minutes from the observed fill-in players. Require a feasible 240-minute regulation rotation and report the healthy-minute and fill-in assumptions. At the team's game-rating scale:

```text
healthy_rating[g] - observed_rating[g]
  = sum_j restored_minutes_from_fill_in_j[g] / regulation_minutes[g]
          * (injured_player_EPM - fill_in_j_EPM)
```

Account for overtime consistently, or normalize both scenarios to 240 regulation minutes. Retain observed opponent health when restoring only the focal team; label an all-teams-healthy scenario separately. Do not use a universal replacement rating when observed fill-ins are available.

Extend the series recursion to game-varying ratings/availability. For an injury-date view, condition on the series score at injury and simulate the remaining bracket. For an entering-playoffs retrospective view, use a documented availability/recovery scenario across all simulated paths. Observed dates alone cannot specify health for counterfactual extra games, different opponents, or rounds never reached: require explicit recovery assumptions and sensitivity ranges. Never count eliminated-team games as actual missed games.

```text
injury_title_chance_loss_pp
  = 100 * (P(title | healthy workload scenario)
           - P(title | observed injury/recovery scenario))
```

This is a modeled difference in championship probability, not proof that an injury cost a championship. Keep signed results rather than flooring them at zero. A separate player-over-alternative lift comparison can use the same healthy and injured paths with an identical baseline roster and salary treatment; that baseline must be explicit before naming the result title lift lost.

With overlapping injuries, report joint team loss first. One-at-a-time restoration effects are not additive; use Shapley allocation over injury episodes only if an exact attributed team-loss total is needed.

## Reader and validation contract

Proposed fields: playoff games on eligible roster, games appeared, confirmed injury games missed, unresolved absences, healthy workload assumption, modeled healthy title odds, modeled injury-path title odds, injury title-chance loss in percentage points, evidence coverage, and scenario range. Entire-postseason absentees must remain visible.

Validate no-injury identity, restore/remove minute closure, series and bracket probability closure, constant-availability reproduction of the existing solver, hypothetical-game recovery rules, exact injury-date conditioning, and attribution closure if used. Do not publish numeric injury title loss from the current static solver without implementing the game-varying counterfactual.

## Implemented participation sensitivity (2026-10-11)

The owner selected the simpler participation operator after the source audit:

```text
actual participation[y] = games appeared[y] / team's actual playoff games[y]
estimated participation[y] = (other-career games appeared + k * other-year league rate)
                            / (other-career team playoff games + k)
participation-scaled lift[y] = existing season lift[y] * participation used[y]
career participation-scaled lift = sum of covered adjusted season lifts
```

Implemented in `scripts/playoff_availability.py`, integrated into the normal championship-help build, and available as a saved-cut refresh through `scripts/refresh_playoff_participation.py`. The latter patches only new app fields, preserving separately merged DARKO rows and metadata. Original title-lift columns are unchanged. V1, V2 and EPM playoff-prior adjustments are exported; the reader shows V2 beside its original value, a sortable comparison column, and a season participation percentage with an actual/estimated/inferred label. The reader's missing-context wording now refers to no player playoff appearance, rather than wrongly claiming every zero-appearance player's team missed the playoffs.

For a target year, remove that entire year from player history and the game-weighted league baseline. Select `k` using an inner leave-one-season-out loop across the remaining years, minimizing equal-season average player-season rate MSE over `[0,1,2,4,8,16,32,64,128,256]` equivalent games. Exact ties favor the larger prior. Within every inner fold, remove both the outer target year and the inner validation year. Players without other playoff history receive the fold's league mean. This is retrospective cross-fitting and may use later career seasons; it is not a past-only forecast. The current archive is 2002–2026, so earlier playoff history is outside this estimate.

Each of the 25 outer folds selects `k=32`. Outer equal-season MSE is `0.0748783` versus `0.0842690` for the other-year league mean alone, an 11.1% reduction. These are held-out participation comparisons within this retrospective universe, not evidence of improved title prediction. The fallback league rate is approximately 78%; it covers rotation and coaching selection as well as health. Training includes only observed playoff-team participants plus source-verified zero-appearance cases; exclusion of uncertified absent-roster players creates a selection limitation, especially for the fallback.

Raw lineup IDs are deduplicated at player/team/game grain against the existing audited playoff schedule. Existing playoff appearances establish team assignment. The cited override in `config/playoff_participation_roster_overrides.json` admits Giannis 2024 at exactly `0/6`, even though the existing appearance-derived `is_playoff_participant` is false. Do not change that older flag or manufacture unavailable teammate accounting. A no-appearance player's matching latest regular-season team and focal team can supply an explicitly labeled roster proxy: if that team reached the playoffs, use inferred zero participation; if it did not, use the career estimate. Unresolved assignments remain missing, not estimated nonplayoff. Uncertified zero-participation proxies are excluded from model fitting.

Current source coverage: 11,613 focal player-seasons; 5,116 direct rates, 4,911 estimated nonplayoff rates, 1,028 inferred playoff-roster zeros, and 558 unresolved assignments. Availability fields may be populated when salary/title-lift fields are missing; career adjustment coverage counts only finite adjusted V2 seasons. Career sums never use a single career availability multiplier, and no-data totals remain missing. The remaining roster proxies need dated roster/transaction certification before calling the whole board certified actual availability.

Examples: Kyrie 2015 is `13/20=65%`, Love 2015 `4/20=20%`, Durant 2019 `12/22=54.5%`, Kawhi 2021 `11/19=57.9%`, and Giannis 2024 `0/6=0%`. Kyrie 2023 (nonplayoff) is estimated at `87.8%`; Durant 2025 at `89.1%`.

This operator scales the entire signed seasonal lift, including its seeding component, as the owner-requested descriptive sensitivity. It does not rerun championship odds, identify injury-only loss, preserve an economic contract-cost debit, or count partial-game lost minutes. Negative lift moves toward zero under lower participation; call it participation-scaled lift, not an injury penalty or corrected contract efficiency. The game-varying injury probability counterfactual above remains a separate unimplemented extension.

Reproduce without refitting existing title models:

```bash
.venv/bin/python nba_pipeline/scripts/refresh_playoff_participation.py
.venv/bin/python -m pytest nba_pipeline/tests/test_playoff_availability.py nba_pipeline/tests/test_championship_help.py -q
cd teammate-help-app
npm run build
node --test tests/rendered-html.test.mjs
```

Saved-cut refreshes preserve external app fields. A full `build_championship_help.py` rebuild exports the EPM base as before; reapply `publish_darko_playoff_title_lift.py` afterward to merge the separate DARKO challenger. Generated CSV/CV files remain gitignored; only curated roster evidence, source code, tests and documentation belong in this change. No deployment is included.
