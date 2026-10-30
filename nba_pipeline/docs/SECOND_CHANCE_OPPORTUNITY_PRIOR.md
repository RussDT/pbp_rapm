# Second-Chance Opportunity Prior

`scripts/build_second_chance_opportunity_prior.py` measures the second-chance
opportunity cost created when a particular player takes a shot from a
particular zone. It bridges repo-owned Shot Quality and the existing
`SHOOTER_OREB` miss-recovery surface.

This is a retrospective research artifact. It does not replace or modify
`SECOND_CHANCE_CLEAN`, forward `weighted_factors_decomp_net_*`, gross
compatibility `weighted_factors_decomp_*`, or
any published player rating. Both Alt3 families remain legacy/audit surfaces
and are not targets or balancing buckets for this model.

## Question

A center taking a long two changes more than make probability. On a miss:

- the center is the shooter rather than one of the four potential crashers;
- a teammate occupies the fourth non-shooter slot;
- the miss has its own distance and shot-type geometry;
- the five defenders affect the chance of a recovery.

The model asks:

> If one of the four teammates took the same-zone shot and the actual shooter
> moved into that teammate's non-shooter role, how would offensive-rebound
> probability change?

It evaluates all four teammate swaps and averages them equally. This is a
role-vacancy counterfactual, not a claim that each teammate should literally
take the shot.

## Inputs

- `processed/SHOOTER_OREB{YY}[_PS].parquet`: one row per eligible missed-FGA
  rebound opportunity, with the shooter, four other offensive players, five
  defenders, and observed OREB result.
- `results/pbp_shot_quality/pbp_shot_quality_1997_2026.parquet`: zone, distance
  bin, shot family, action type, descriptor bundle, shooter, and repo-owned SQ
  make probabilities.
- `processed/SECOND_CHANCE_CLEAN{YY}[_PS].parquet`: observed physical
  second-chance points used only for the points-per-OREB conversion.

The join key is normalized `game_id + event_num`, with shooter identity checked
after the join.

## Model

The eligible-miss target is:

```text
P(OREB | missed FGA)
```

The base ridge model contains:

```text
season/sample
+ shot zone
+ distance bin
+ shot family
+ action type
+ descriptor bundle
+ four exact non-shooter identities
+ five exact defender identities
```

The enhanced model adds `shooter identity × canonical SQ shot zone`.

All reported predictions are out of fold under grouped cross-validation by
`game_id`. No game appears in both the fit and validation side of a fold. The
fold's points-per-OREB conversion is also computed from training games only.

This is a linear probability ridge with predictions clipped to `[0.01, 0.99]`.
The unbounded score is retained for exact attribution:

```text
actual configuration - role swap
= crash-group reassignment
+ shooter-by-zone reassignment
```

The 2022-26 role-swap identity closed to `1.5e-16`.

## EV Accounting

The second-chance branch is:

```text
SC EV per attempt
= (1 - SQ make probability)
 × P(OREB | miss)
 × points per OREB
```

Primary columns use `shot_quality_prior_make_prob`, whose shooter residual uses
only his earlier attempts. Parallel `_season_sq` columns use
`shot_quality_season_make_prob` for consistency with the current
`RUSSELL_SHOTQUALITY` possession surface.

This layer prices only the continuation after a miss. Ordinary Shot Quality
still prices the initial shot:

```text
decision value = initial shot EV + second-chance EV
```

Do not add this opportunity prior to `SECOND_CHANCE_CLEAN` as if they were
independent rating components. `SECOND_CHANCE_CLEAN` is the observed physical
downstream result; this artifact explains one pre-shot mechanism feeding it.

## Outputs

The default generated directory is
`results/second_chance_opportunity_prior/`:

- `miss_opportunity_predictions.parquet`: one row per eligible miss, with base,
  shooter-aware, and role-swap OOF probabilities; the two exact role
  contributions; and fold-safe SC conversion.
- `player_zone_accounting.csv`: canonical SQ zones (`rim`, `short_mid`,
  `long_mid`, `deep_two`, `three`).
- `player_shot_mix_accounting.csv`: reader-facing zones with
  `long_mid + deep_two = long_two`.
- `metrics.json`: coverage, folds, losses, validation gain, and reconciliation.
- `manifest.json`: inputs, model contract, outputs, and guardrails.

These are generated artifacts and remain outside git.

## Current 2022-26 Result

```bash
.venv/bin/python \
  nba_pipeline/scripts/build_second_chance_opportunity_prior.py \
  2022 2026 \
  --season-type ALL \
  --folds 3 \
  --alpha 300 \
  --output-dir \
  nba_pipeline/results/second_chance_opportunity_prior/22_26_all_a300
```

Audit:

- `607,019` eligible misses across `6,572` games;
- `100%` SQ join coverage and `0` shooter-ID mismatches;
- base/enhanced OOF Brier `0.197828 / 0.197688`;
- Brier gain `0.000140`;
- game-cluster bootstrap 95% interval for Brier gain
  `[0.000106, 0.000176]`;
- base/enhanced OOF log loss `0.583072 / 0.582760`;
- log-loss gain `0.000312`;
- game-cluster bootstrap 95% interval for log-loss gain
  `[0.000226, 0.000401]`;
- no train/validation game overlap.

The shooter-zone signal is modest at league scale but positive out of sample.
Its main value is explanatory accounting for high-volume player/zone cases.

### Joel Embiid long twos

The combined `long_two` row covers both canonical zones beyond 10 feet:

| Item | 2022-26 RS+PS |
|---|---:|
| Attempts | 1,877 |
| Eligible miss/rebound rows | 981 |
| Actual OREB rate | 15.49% |
| Context + exact other nine, shooter unidentified | 18.66% |
| Actual Embiid configuration, OOF | 16.88% |
| Teammate shoots / Embiid joins crash group | 18.46% |
| Role-vacancy difference | -1.58 percentage points |
| Prior-SQ SC EV difference per attempt | -0.0096 points |
| Prior-SQ total over the sample | -18.0 points |
| Season-SQ total over the sample | approximately -17.2 points |

The `18.66%` baseline is not “because he is a center.” It has no position
field. It already sees the exact remaining four offensive players and five
defenders, so it indirectly captures the center-shaped hole in the crash group.
The role swap then moves Embiid back into that group and lets an actual teammate
be the shooter.

For this sample, the direct shooter-by-zone term is the main cost. The
crash-group reassignment is slightly favorable to the actual Embiid-shoots
configuration, so the data do not support charging him an additional large
generic “center should have crashed” penalty. That result can reflect scheme,
transition-defense responsibilities, health, and teammate roles—not just
rebounding ability.

## Limits and Next Layer

The current data do not identify:

- optical locations at the release or rebound;
- rebound trajectory, side, and distance;
- box-outs, crash assignments, or transition-defense instructions;
- a causal coaching counterfactual for who would actually take the shot;
- fully forward-only context encodings in the repo-owned SQ table.

The current conversion is a league-window approximation: within each fit fold,
it divides all `SECOND_CHANCE_CLEAN` points by eligible FGA OREBs represented in
`SHOOTER_OREB`. The processed second-chance rows do not retain the originating
miss ID, so rare miss/rebound sequences excluded by the next-event
`SHOOTER_OREB` definition cannot be paired directly. Preserve that limitation
until the possession builder carries an origin-shot key.

An explicit position label would not solve those gaps. The safer next layer is
a continuous rebound-role prior learned from pre-shot/lineup behavior or
tracking data, with position used only as shrinkage. Test it against this
exact-player role-swap baseline and keep it outside public DECOMP until it wins
on held-out games and remains stable across seasons.

The first downstream integration now lives in
[Shot-Clock SQ and Grenade Burden](./SQ_SHOT_CLOCK_CHALLENGER.md). It attaches
this player-zone OREB prior to clock-conditioned and neutral-clock shot
probabilities, producing an expected-possession value at release and an exact
shot-burden/second-chance-offset bridge.

The first strict season-forward predictive test now lives in
[Forward SQ + Second-Chance Release Value](./SQ_SECOND_CHANCE_FORWARD.md).
It trains the opportunity and continuation layers only on 2022-24, evaluates
2024-25 first-FGA shot-plus-second-chance points, and compares the
shooter-specific continuation against both a shooter-hidden exact-lineup lane
and provider `initial_ev` given the same continuation.
