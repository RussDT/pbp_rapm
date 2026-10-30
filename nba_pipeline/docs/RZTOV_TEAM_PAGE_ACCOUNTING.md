# Complete NBA team-page accounting

> Source correction, 2026-09-05: the initial NBA results below mixed a partial-season reader TSA rate with complete-season PBP exposure. They are superseded by the same-sample RS rebuild documented in `RZTOV_SAME_SAMPLE_CORRECTION.md`. Do not quote the original NBA numerical receipts as current.

2026-09-05 diagnostic, 2025-26 regular season. Built by
`nba_pipeline/scripts/build_rztov_team_page_accounting.py`; all 30 team outputs
are in `results/rztov_team_page_accounting/nba_2026_teams.csv`. No public tables
or player ratings were changed.

The comparison includes all players, not just those above the historical
500-possession ledger cutoff. It uses existing player-season workload estimates
and 2022-2026 prices, allocated to teams by resolved offensive possessions.
Player expected scoring and bad-pass turnovers are centered on the complete
league player sample. This is a distinct full-coverage display diagnostic, not
the unchanged filtered player board.

Team/unassigned turnovers have no modeled player workload. Add a separate
league-average baseline of 0.716850 team turnovers per 100 resolved offensive
possessions. Total expected turnovers = player workload expectation + this
explicit team-turnover baseline. Never describe the entire expectation as
coming solely from player workload, or force actual team residuals to zero.

Team-turnover events frequently put the franchise ID in `player1_id`, while
`player1_team_id` is zero. Recover that ID rather than leaving all such events
in the unknown-team bucket. Offensive team identity uses player-game mappings,
expanded with observed venue lineups for players who author no events, and
requires at least three agreeing mapped offensive players and no conflicting
mapped teams on every possession. All possessions resolve in this build.

## Boston and Portland

| Quantity | Boston | Portland |
| --- | ---: | ---: |
| Resolved offensive possessions | 7,754 | 8,294 |
| Player turnovers, including low-minute players | 943 | 1,351 |
| Of those, below prior ledger cutoff | 28 | 9 |
| Team/unassigned turnovers | 71 | 66 |
| Total actual turnovers | 1,014 | 1,417 |
| Player-workload expected turnovers | 1,132.95 | 1,122.50 |
| Separate team-turnover expectation | 55.58 | 59.46 |
| Total expected turnovers | 1,188.54 | 1,181.96 |
| Actual / 100 possessions | 13.08 | 17.08 |
| Expected / 100 possessions | 15.33 | 14.25 |
| Fewer than expected / 100 | +2.25 | -2.83 |

The expectation is estimated, not causal credit or a calibrated prediction
interval. Team-stint role differences remain unobserved in this allocation.
Source counts and denominator come from local raw PBP and resolved RAPM
possessions, not independent official team box-score certification.

The build hard-checks role coverage, player exposure totals, unambiguous
possession teams, turnover-team attribution, total event-count closure, and
league expected-versus-actual closure. The manifest records all-player coverage,
prices, centering, baseline, and source limitations. Full totals retain
team-specific residuals; only league aggregates are centered.
