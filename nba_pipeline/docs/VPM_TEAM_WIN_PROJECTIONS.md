# Selected VPM → team season-win projections

Owner request October20,2026: learn the embedded unified RAPM repo's season
projection machinery and apply the useful forecast layer to selected mapper100.
This is the same VPM research/product integration slot, not a Plan40/43 donor
campaign. Donor inspection was read-only. No2027 outcomes were accessed.

## What unified actually ships

`unified-rapm/daily_v5/deploy_site.sh` selects owner/current minutes and
`p2_0_production`, invoking `project_2027.py` and `build_projection_teams.py`.
The older `simulate_standings.py` is descriptive actual-minute2025–26 work;
it is not the forward season path.

The forward path uses `w_i=active_MPG_i*expected_GP_i/82` and
`team_strength=5*sum(w_i*rating_i)/sum(w_i)`. Its current allocation preserves
**sum(active_MPG*availability)=240**, while raw active MPG can exceed240.
Multiplying availability twice or renormalizing raw active MPG changes the
meaning. Roster and allocator membership must match in both directions.
Newer active-contract minutes separate active, season and replacement minutes.

Production simulates10,000 seasons on one legal synthetic82-game/41-home slate.
It draws season-long team strength once per simulation and then independent
Student-t game noise. The noise variance subtracts the strength-draw variance
before adding game noise, with a36-point-squared floor. Rest/travel effects are
not applied to the synthetic season slate. Older docstrings claiming a new
slate each simulation are superseded by the executed code.

Do not transfer donor player aging/XGB values or its arbitrary redistribution
of aggregate changes into six factors. Keep our selected learned Net lanes.
Donor cold-player/SE inflation and Student-t parameters are calibrated to its
own player system, not VPM. Its locally linear additive player-win allocation
sums as accounting; it is not exact causal marginal wins or nonlinear lane
counterfactual value.

## Implemented VPM bridge

Sibling `vpm/harness/team_projection.py` and harness command `team-wins`:

1. Accept explicit prospective `franchise,player_id,season_mpg`. Optional
   active_MPG and expected_GP must satisfy `season_mpg=active_MPG*GP/82` once.
   Every team must sum240 after explicit redistribution/replacement. Reject
   missing/duplicate identities, missing selected ratings and hidden fills.
2. Aggregate selected O/D/Net and every standard Net lane with
   `team_lane=sum(season_MPG_i*player_lane_i)/48`. The five-player slot factor
   is therefore explicit. Team lanes still sum exactly to team O/D/net.
3. For each matchup,
   `margin=(home_team_VPM-away_team_VPM)*expected_pace/100 + HCA`.
   Neutral games receive no HCA. Carry the existing VPM benchmark logistic
   `P(home_win)=sigmoid(0.14*margin)` and HCA3 as explicit assumptions;
   default expected pace100 is a scenario assumption, not a fitted forecast.
   These are not donor calibration parameters or a newly validated preseason
   probability model.
4. Sum game probabilities for expected wins. Conditional simulations draw
   one winner per game; each draw has exactly1,230 league wins. Displayed
   10th/50th/90th percentiles include **game-result variation only**, freezing
   ratings, minutes and team strengths. They omit rating/availability/rotation
   uncertainty and are not full season forecast intervals. No playoff/title
   probabilities are implemented or claimed.

A supplied schedule must have only outcome-free allowed fields,30teams,
1,230games and82games/team. Alternatively explicitly request `--synthetic`:
reuse the donor's pure schedule helper, discard its cosmetic2025 dates, and
label the result schedule-neutral. No actual2027 game outcomes or realized
minutes are needed or admitted. A proper pregame validation later requires
historically predicted minutes rather than the oracle-minute quality benchmark.

## Source inventory and current missing inputs

The donor's migrated out tree is absent in this checkout, including the current
minutes allocator, GP forecast and rotation distributions. Available owner CSV
`unified-rapm/priors/depthcharts_2026_27.csv` is an older July membership/role
snapshot:538rows/30teams,79missing IDs, **all projected minutes blank**. Against
the selected VPM board90rows have no rating (including unresolved IDs). Ten
teams exceed18players. The adjacent provenance file explicitly describes a
July19 candidate rather than the adjacent live CSV; do not certify that live
file from the candidate hash.

The July3 candidate JSON has596rows and differs by59missing/116additional
membership keys; do not splice those minutes onto the owner CSV. July9 official
rosters are also stale. No current prospective actual schedule artifact was
found in the inspected roots. Therefore no current30-team win table was
produced. An async owner question asks which current/provisional roster/minutes
assumptions to use; elapsed time is not approval to invent them.

`team-wins --prepare` saves a538-row minutes template and readiness report to
sibling ignored `vpm/results/team_projection_2027_v1/`. Fields remain blank;
no fabricated GP or role-to-minutes defaults are promoted as current estimates.
The adapter can run once dated minutes and exact player/entrant admission are
provided. Source hashes and this readiness receipt are curated in sibling
`vpm/evidence/team_projection_bridge_summary.json`.

## Reproduction

From sibling VPM:

```
../rapm/.venv/bin/python -m vpm.harness.cli --config vpm/configs/current_model.json team-wins --prepare
../rapm/.venv/bin/python -m vpm.harness.cli --config vpm/configs/current_model.json team-wins \
  --minutes /path/to/projected_minutes.csv --input-as-of 2026-10-20 \
  --synthetic --out vpm/results/team_projection_2027_scenario
```

Use `--schedule /path/to/outcome_free_schedule.csv` instead of `--synthetic`
for an admitted actual slate. Outputs are team_wins.csv,
player_contributions.csv, game_probabilities.csv and manifest.json. Missing
inputs fail; preparation does not claim operational readiness.

Seven focused tests pass: availability once,240-minute conservation, slot
aggregation, missing identities/ratings, outcome-column rejection, equal-team
41-win conservation/common-offset invariance, and explicit pace conversion.
Tests run the pure donor schedule helper only, never donor model scripts.
No full preseason validation, public deployment or2027 evaluation is claimed.
