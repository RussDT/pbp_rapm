# rZTOV team accounting and expectation audit

> Source correction, 2026-09-05: the initial NBA results below mixed a partial-season reader TSA rate with complete-season PBP exposure. They are superseded by the same-sample RS rebuild documented in `RZTOV_SAME_SAMPLE_CORRECTION.md`. Do not quote the original NBA numerical receipts as current.

Completed 2026-09-05. This is a read-only diagnostic of the current NBA estimated
1997-2026 and WNBA estimated 2022-2026 boards. No scores, prices, or public tables
were changed. Reproduce with `nba_pipeline/scripts/audit_rztov_team_totals.py`.
Generated CSVs remain under `nba_pipeline/results/rztov_team_audit/`.

## Main finding

Player arithmetic closes, but estimated role errors do not cancel within teams.
Do not force a team's expected turnovers to equal its actual turnovers: that
would erase the team residual. Keep actual-count accounting separate from
calibration of the model's estimated workload.

At fixed canonical TSA and fixed prices, estimated-minus-observed allowance is:

```
(self_price - assisted_price) * (estimated_self - observed_share_implied_self)
+ potential_assist_price * (estimated_potential_assists - observed_potential_assists)
```

Observed-share-implied self volume uses the same canonical TSA, so the comparison
isolates estimated role mix and passing volume rather than differing TSA bases.
Positive differences mean a more forgiving estimated allowance.

## Team workload-estimation effect

For the NBA 2014-2026 common observed-label sample, the median absolute team
allowance change is 1.42%, the 90th percentile is 3.41%, and the largest is 6.97%.
For 2026 alone, the range is -42.4 to +43.1 turnover allowances, or -4.28% to
+4.75% of the common-sample estimated allowance.

WNBA observed RTSS comparison is available for 2023-2025 only. The median
absolute team shift is 1.39%. Season ranges are:

| Season | Allowance count difference | Percent of common estimated allowance |
| --- | ---: | ---: |
| 2023 | -17.54 to +25.19 | -3.20% to +4.43% |
| 2024 | -17.53 to +16.92 | -3.42% to +3.07% |
| 2025 | -24.01 to +18.28 | -4.44% to +3.06% |

These differences remove the common-sample league-wide mean by possession
weight, mirroring the lane-centering principle. They are retrospective fit
checks, not held-out validation: the final estimators were trained on these
observed-label seasons. They may understate future estimation error.

Team attribution allocates each player's season quantities using their exact
local PBP offensive-possession shares across teams. It does not assign traded
players entirely to a last-listed team. These are season-rate allocations,
not separately estimated team-stint roles. Unresolved team slots stay in team 0
rather than disappearing; NBA 2026 has 1,680 such slots, WNBA 2026 has zero.
The role-effect percentages use only players with both estimated and observed
labels, not a claim of complete team coverage.

## Count accounting

Across all NBA seasons, each included player's recovered turnover count matches
local PBP within CSV rounding (maximum absolute player gap 0.000033 turnovers).
Scoring plus bad-pass rates close within 0.0000011. Allowances sum to actual
turnovers across the included league sample because of season centering; this
is an enforced identity, not evidence that the model is calibrated within teams.

NBA 2026 source accounting:

| Bucket | Turnovers |
| --- | ---: |
| Included estimated-rZTOV players | 33,280 |
| Player turnovers outside the 500-possession ledger cutoff | 656 |
| Team/unassigned turnovers | 1,759 |
| All local PBP turnover events | 35,695 |

WNBA 2026 includes all 7,709 player turnovers in its box source and local PBP;
458 team/unassigned events bring the PBP total to 8,167. WNBA board arithmetic
matches its own box source for all five seasons to floating-point precision.

The separate WNBA PBP mirror is not perfectly identical to the box panel:

| Season | Box player turnovers minus PBP player turnovers | Sum of absolute player differences |
| --- | ---: | ---: |
| 2022 | -7 | 19 |
| 2023 | +28 | 32 |
| 2024 | +2 | 54 |
| 2025 | +30 | 30 |
| 2026 | 0 | 0 |

The largest individual historical difference is five turnovers. This local
source discrepancy is not an allowance-model arithmetic failure; its cause
was not established by this audit. The source reference here is local PBP,
not a separately fetched official team box-score archive. Event totals are
therefore a source reconciliation, not independent official certification.

## Next decision

A team-season workload calibration gate is justified. First require accurate
team-stint inputs and report estimated-versus-observed team allowance residuals
on held-out seasons. A prospective correction must preserve legitimate team
workload differences and pass a future target, rather than forcing each team
residual to zero. Do not tune a correction on the spent comparison seasons.

The nested held-out-workload follow-up is complete; see [Team workload calibration](RZTOV_TEAM_CALIBRATION.md).

The subsequent [complete team-page accounting](RZTOV_TEAM_PAGE_ACCOUNTING.md) recovers franchise IDs stored in `player1_id` for team turnovers, resolves every current possession team, and includes low-minute player expectations. The original audit kept these team events in its team-0 bucket; they are not inherently impossible to attribute.
