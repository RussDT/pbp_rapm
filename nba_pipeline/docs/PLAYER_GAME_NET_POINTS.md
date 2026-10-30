# Six-factor completed-game net points

**Allocation superseded:** the owner meant learned game-box coefficients, not
historical player-rating anchors. See [Box coefficient revision](PLAYER_GAME_BOX_NET_POINTS.md).
This v1 remains the source-audited team target/exposure adapter and an allocation audit.

Research prototype requested by the owner on 2026-09-13. Builder:
`scripts/build_player_game_net_points.py`. This is completed-game performance
attribution, not a forecast, one-game RAPM, or a promoted VPM method.

## Contract

Player columns, in points with positive-good defense:

- `oTS`, `oTOV`, `oSC`: shot value, turnover value, net second-chance value;
- `dTS`, `dTOV`, `dSC`: the corresponding opponent values prevented;
- `offense`, `defense`, `game_net_points`: exact sums of those children.

Offense retains actual results. On defense, remove `actual points - Russell v2
expected points` for **every opponent three**, including second-chance shots.
First-chance adjustments belong in dTS; second-chance adjustments belong in
dSC. Keep the observed rebound chain and original net-continuation payback.
This version does not neutralize rebound opportunities caused by misses.

At team-game grain:

```
player net points + unequal-possession baseline - opponent 3P variance
    = actual scoreboard margin
```

Opponent variance is positive when the opponent exceeds its three-point
expectation. Adjusted margins need not be antisymmetric: each offense retains
its actual conversion, while each defense uses expected opponent threes.
Neither side's adjusted margin is a replacement for the actual score.

The unequal-possession baseline is necessary for an above-average ledger;
it is an explicit accounting term, not another luck adjustment. Baselines are
the full selected season/type source's unconditional TS/TOV/SC means, before
game filtering. All prices and SQ are retrospective, not as-of forecasts.

## Priors and allocation

Default history for 2026 games is the actual Net DECOMP 2021-2025 board:

`master_results/net_decomp_rolling/weighted_factors_decomp_net_21_25_all_rb_se_a2000_4000.csv`.

Sum the five net shot children plus FT for each side's TS prior, and use the
published TOV and SC children directly. Preserve the positive-good defensive
sign. Do not import native percentage factors or gross DECOMP into this ledger.
The loader rejects target-season `Latest_Year`, duplicate players, incompatible
pricing labels and nonfinite factors. Missing players receive zero history and
an explicit `prior_missing` flag. The source filename/manifest must still be
audited: a table's Latest_Year field alone cannot authenticate its training cut.

For player i and factor k:

```
historical anchor = prior_scale * historical rate * side possessions / 100
player credit = historical anchor + observed actor signal + shared adjustment
shared adjustment = exposure share * (team target - sum(anchors + signals))
```

This is the minimum weighted squared displacement subject to the team sum,
using variance proportional to exposure. It is a **declared attribution rule**,
not a fitted posterior with empirically validated variances. Default prior scale
1 is provisional. `zero_prior_net_points` and `prior_effect` expose sensitivity
to removing history while retaining observed signals and identical team totals.
Prior effects sum to zero within each team-factor; history redistributes credit.
These differences are not confidence intervals.

Observed actor signals currently include:

- FGA actual-minus-Russell-v2 conversion, to the shooter in oTS or oSC;
- FT actual-minus-prior-season FT expectation, to the shooter in oTS; FT
  probabilities use 50 league-average pseudo-attempts, a provisional choice;
- personal turnovers on canonical turnover terminals, charged at the same
  non-turnover opportunity price used in the team target;
- steals resolved to exactly one game-local source name, at that turnover price;
- personal offensive rebounds priced at full-source gross second-chance points
  divided by personal offensive rebounds;
- personal defensive rebounds priced at the full-source average first-miss
  continuation value.

Rebound and steal prices are opportunity-value anchors, not established causal
credits. Their shared adjustment prevents component double counting but does
not prove correct individual attribution. Unresolved steal names, team
turnovers, passing/creation and shot-defense responsibility remain shared.
There is no provider-defender identity shortcut or position bonus.

## Source integrity

Use chronological raw PBP, exact game/event keys, and canonical aligned FC/SC
rows. Complementary split descriptions may be merged only if every populated
field agrees. Conflicting event records fail. Raw field goals plus actual free
throws must match the final score before producing a player board.

The processed substrate uses expected FTs. Its team-score bridge is included
in TS, with observed personal FT conversion plus a shared remainder. It is not
an independently reconstructed player-specific processed FT correction.

An isolated adapter restores omitted **scoreless end-period possessions**,
including same-offense missed-FGA/OREB chains. It recovers the dominant source
zone continuation tariff (at least 95% agreement among matched positive-kappa
samples), adds that tariff to TS and subtracts it from SC. It does not rewrite
production parquets. Mixed-side, scoring, or unrecognized live tails fail.

Every FGA must have valid SQ, match the possession's offense, and preserve the
source first-FGA flag. Actor participation and positive exposure are required.
Exposure is terminal-lineup possession exposure (first-shot lineup for restored
tails), not exact event-level exposure across within-possession substitutions.

Russell v2 is the existing shooter-specific, retrospective player-season SQ
surface. It is **not** pure defender contest quality and may lack relevant
geometry. The adjustment therefore replaces conversion with model expectation;
labeling the entire remainder as proven shooting luck would overstate it.

## Initial review and validation

Run the passing Knicks–Cavaliers sample (games 1, 3 and 4):

```bash
.venv/bin/python nba_pipeline/scripts/build_player_game_net_points.py \
  --game-id 0042500301,0042500303,0042500304
```

This yields 65 player-games, six team-games, and maximum scoreboard error below
`5e-14` points. Game 2 (`0042500302`, event 186) fails shooter/possession side
agreement and is not included. A broader NYK run additionally reveals
cross-period first-FGA flag disagreements in ATL games 2/4 and SAS game 3;
full-season coverage is not established. Do not relax those gates to fill a board.

Outputs under gitignored `results/player_game_net_points/`:

- `players.csv/.parquet`: six factors plus each factor's historical, observed,
  and shared contributions;
- `teams.csv/.parquet`: targets, opponent variance by lane, FT bridge, baseline,
  and scoreboard reconciliation;
- `observed_events.csv/.parquet`: actor-signal audit;
- `manifest.json`: source paths/stats, historical-prior and implementation
  hashes, prices, selection, timing, and limitations;
- `review.html`: local reader with all six factors and prior sensitivity.

Tests cover conversion isolation, defensive three-point adjustment in both
chance lanes, constrained attribution and prior sensitivity, prior timing and
substrate rejection, rebound-count corrections, invalid inputs, and raw event
conflicts. The existing team-game builder tests run alongside them.

## Next methodological gate

Before publishing a season board, resolve the possession-source failures,
audit game-level actor assignments and the FT bridge, compare reasonable prior
scales and rebound/steal prices on independently reviewed possessions, and add
creation evidence without double-counting shooting. Retain the zero-history
control. Team closure is an accounting check, not proof of player-level accuracy.
No production VPM input or protected 2026-27 predictive gate is changed.
