# Scalar Title Lift over Contract challenger

Status: **research diagnostic; not a production replacement**.

## Question

Can the realized salary-peer placement model be compressed to the simpler
intuition below?

```text
salary share -> expected contract EPM
player EPM - expected contract EPM -> one title-lift curve
```

The test is deliberately a surrogate. It learns its title curve against the
full placement model, so it measures how compressible that model is. It does
not independently validate the championship probabilities.

## Design

`scripts/analyze_contract_title_curve.py` scores every complete salary-covered
player-season out of season. For each ending-year season from 2002 through 2025,
both curves are fit without any rows from the held-out season.

The contract curve is a nondecreasing isotonic fit from salary share to average
realized salary-peer EPM. The literal challenger is:

```text
expected_contract_EPM = contract_curve(salary_share)
surplus_EPM = player_EPM - expected_contract_EPM
shortcut_title_lift = zero_anchored_title_curve(surplus_EPM)
```

The smallest role-aware version also learns the expected playoff workload of
the salary slot and converts rate surplus to a team-rating change:

```text
expected_role = workload_curve(salary_share)
surplus_team_rating = surplus_EPM * expected_role
role_shortcut_title_lift = zero_anchored_title_curve(surplus_team_rating)
```

Both title curves are nondecreasing and forced to return exactly zero at zero
estimated surplus. A separate fitted-intercept diagnostic is retained only to
show the cost of that conceptual zero anchor.

## Out-of-season result

The common sample contains 11,241 player-seasons across 24 salary-covered
seasons.

| Challenger | Pearson | Spearman | MAE (pp) | RMSE (pp) | Bias: full − shortcut (pp) |
|:---|---:|---:|---:|---:|---:|
| EPM surplus only, zero anchored | 0.902 | 0.885 | 0.673 | 0.881 | -0.539 |
| EPM surplus × expected role, zero anchored | 0.987 | 0.958 | 0.432 | 0.504 | -0.429 |
| Role-adjusted, fitted intercept diagnostic | 0.987 | 0.958 | 0.207 | 0.264 | +0.000 |

The literal EPM-only shortcut is not adequate. It substantially overstates
cheap high-EPM seasons because it gives a minimum-salary slot the title leverage
of the much larger playoff role carried by a max slot. Expected salary-slot
workload is not optional.

### Realized scoreboard points over contract

Do not interpret a career sum of annual EPM-surplus rates as accumulated
points. The corrected linear accounting uses exact regular-season lineup
exposure for each player-season:

```text
realized RS points over contract
  = (player EPM - held-out expected contract EPM)
    * average(offensive possessions, defensive possessions)
    / 100
```

The offensive and defensive possession counts are averaged so the same team
possession is not counted twice. Career totals sum the annual point values.
`surplus_epm_season_sum` retains the old rate sum only as an audit field; its
unit is EPM-seasons, not points. LeBron's 22 covered seasons sum to `81.744`
EPM-seasons over contract, but the correct realized regular-season total is
`4,328.3` points over 114,936 averaged possessions.

Once that one role term is included, a scalar curve explains most of the
placement model. The remaining fitted-intercept gap is also informative: a
player equal to the **mean EPM** of a heterogeneous salary-peer distribution is
not guaranteed to have zero mean title lift after a nonlinear bracket
calculation. Mean EPM is not necessarily the title-equity certainty equivalent.

## LeBron and Curry

| Estimate | 2009 LeBron | LeBron career | Curry career | LeBron − Curry |
|:---|---:|---:|---:|---:|
| Full placements | +15.08 pp | 1.417 | 1.235 | 0.182 |
| EPM-only shortcut | +10.21 pp | 1.085 | 1.017 | 0.068 |
| EPM × expected-role shortcut | +15.32 pp | 1.538 | 1.279 | 0.259 |
| Role shortcut, fitted intercept diagnostic | +14.89 pp | 1.444 | 1.210 | 0.233 |

The role-aware shortcut preserves LeBron as the career leader and creates a
larger LeBron-Curry separation than the full placement result. It also moves
Chris Paul between them under the zero-anchored career sum. This is a useful
ranking sensitivity, not a promotion decision.

## Era-local, budget-neutral full-placement follow-up

The requested full-placement sensitivity is now implemented beside production.
It keeps only destination postseasons within four years of the focal season and
charges non-exact salary matches at `0.067` regular-season and `0.078` playoff
team-rating points per signed cap percentage point. This is not the scalar
shortcut: it preserves the actual team contexts and performs the averaging
after each qualification/seed/bracket calculation.

| Estimate | 2009 LeBron | LeBron career | Curry career | LeBron − Curry |
|:---|---:|---:|---:|---:|
| Full-history placements | +15.08 pp | 1.417 | 1.235 | 0.182 |
| Era-local, budget-neutral v2 | +17.53 pp | 1.057 | 0.672 | 0.385 |

The v2 and full-history career totals correlate `0.960` across 2,275 common
careers. The larger LeBron-Curry separation is driven by the joint era-field
and cap-gap change; it should not be attributed to either ingredient without a
separate ablation.

## Salary-match caveat

The production placement uses the closest salary-share playoff rotation player
on every destination team with no maximum gap. Across the 11,241 focal rows,
the average of player-season median gaps is 0.93 cap percentage points and the
median is 0.57. The average player-season p90 gap is 2.54 points and the median
p90 is 1.64. For 2009 LeBron, the median and p90 gaps are much wider: 2.41 and
7.41 cap points.

The v2 export now retains the signed mean gap, median and p90 absolute gaps, the
fraction beyond five cap points, and the mean rating debit. It deliberately
keeps one real closest-salary slot per team instead of interpolating between two
players, because interpolation would invent a workload and roster role that no
player occupied. Large one-sided max matches remain visible in the diagnostics
and are priced rather than presented as exact equivalence.

## Reproduce

```bash
PYTHONPATH=. .venv/bin/python \
  nba_pipeline/scripts/analyze_contract_title_curve.py

PYTHONPATH=. .venv/bin/pytest -q \
  nba_pipeline/tests/test_contract_title_curve.py
```

Generated row, career, JSON, and Markdown receipts are written under
`nba_pipeline/results/contract_title_curve/` and remain gitignored.
