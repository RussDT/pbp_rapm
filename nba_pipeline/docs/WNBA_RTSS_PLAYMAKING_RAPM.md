# WNBA RTSS Shooter, Passer, and Presence RAPM

## Status

Research-only. This model does not change the frozen live WNBA rating or the
current `5000/40000/1.5` box-prior solve.

The first complete 2023-2025 build shows that a passer-attributed shot-value
RAPM is identifiable from the RTSS export, provided playmaking is defined
before fitting. The current definition is:

```text
passer shot value
+ pass-turnover cost
= net playmaking value
```

Presence is the adjusted offensive shot value left for a player while she is
neither the shooter nor the potential-assist passer. It is not automatically
called gravity, screening, or off-ball movement.

This is a field-goal shot-value model, not a complete oTS decomposition. It
does not yet include the full free-throw side of true shooting.

## Role construction

Each RTSS shot contributes one row with the exact five offensive and five
defensive players from `chance_players`.

- shooter: `shots.shooterId`;
- passer: a distinct, on-court `shots.passerId` when `shots.assistOpp` is true;
- presence: the remaining offensive players;
- defense: the exact five defenders.

The model uses `shots.assistOpp`, which labels potential-assisted attempts on
makes and misses. Made-only `shots.assisted` is not used to define the passer
role.

The 2023-2025 source contains `111,956` shot rows. The role build retains
`111,951`; only five fail exact-lineup or shooter validation. It retains
`56,740` shots with a valid potential-assist passer.

## Centered joint model

For player `i`, let:

- `s_i` be her share of on-court shots taken;
- `p_i` be her share of on-court shots on which she is the valid
  potential-assist passer;
- `b_i` be her exposure-averaged offensive parent coefficient;
- `delta_s_i` be the shooter-minus-presence contrast; and
- `delta_p_i` be the passer-minus-presence contrast.

The shot-row design adds two centered role interactions:

```text
is_shooter - s_i
is_passer - p_i
```

The recovered conditional values are:

```text
presence_i = b_i - s_i * delta_s_i - p_i * delta_p_i
shooter_i  = presence_i + delta_s_i
passer_i   = presence_i + delta_p_i
```

The additive contributions are:

```text
scoring contribution   = s_i * shooter_i
passer contribution    = p_i * passer_i
presence contribution  = (1 - s_i - p_i) * presence_i
```

Therefore:

```text
scoring + passer + presence = offensive shot-value parent
```

The executed output closes with a maximum absolute gap of
`4.44e-16`.

The primary target is actual points per FGA. Provider `qSQ` is fit only as a
separate MID/THREE benchmark and is not the primary playmaking target. Values
are normalized defensively because the export contains both `0-1` and `0-100`
q-value scales.

## Turnover cost

RTSS `passes.turnover` is priced at the season's average points per valid
shot and expressed per 100 offensive chances:

```text
pass-turnover cost
= -100 * pass turnovers / offensive chances * average points per shot
```

The second published identity is:

```text
net playmaking = passer shot value + pass-turnover cost

scoring + net playmaking + presence
= shot-value parent + pass-turnover cost
```

This identity also closes to `4.44e-16`.

## On-ball input

The prior feature builder now calculates actual on-ball time percentage:

```text
100 * RTSS touch time / exact offensive chance floor time
```

The denominator comes from each chance's game-clock duration and is assigned
through exact offensive `chance_players` rows. The rate is stabilized toward
the season league mean with an `1,800`-second prior.

## Common role shrinkage

Separate season-specific role alphas produced incomparable target variance.
The final build therefore pools game-grouped CV errors across 2023, 2024, and
2025 and applies the one-standard-error rule once.

- offensive/defensive parent alpha: `3000`;
- common shooter/passer role alpha: `300`.

All three annual role targets use that common alpha.

## Compact-prior development test

The prior test uses symmetric leave-one-2023/2024/2025-season-out
development. It includes `423` mapped player-seasons with at least `250`
offensive RTSS chances. This is not an historical as-of test and none of the
three seasons remains untouched confirmation.

Candidate inputs are:

- RTSS on-ball time percentage;
- RTSS potential assists per 100 chances; and
- PBP Stats rim assists per 100 possessions.

### Passer shot value

| Model | Pooled RMSE | R2 | Correlation | Winning folds |
|---|---:|---:|---:|---:|
| Train weighted mean | 0.4277 | -0.013 | -0.091 | — |
| On-ball only | 0.3462 | 0.336 | 0.580 | 3/3 |
| On-ball + potential assists | **0.2697** | **0.597** | **0.773** | **3/3** |
| On-ball + potential assists + rim assists | 0.2745 | 0.582 | 0.764 | 3/3 |

Potential assists materially improve the passer target. Rim assists do not
improve the two-feature model.

### Net playmaking after pass-turnover cost

| Model | Pooled RMSE | R2 | Correlation | Winning folds |
|---|---:|---:|---:|---:|
| Train weighted mean | 0.4281 | -0.011 | -0.080 | — |
| On-ball + potential assists | **0.4036** | **0.102** | **0.322** | **3/3** |
| On-ball + potential assists + rim assists | 0.4045 | 0.098 | 0.318 | 2/3 |

The compact creation inputs still help after turnover cost, but much less.
That is expected: the proposed three-input prior does not contain a passing
turnover variable, while pass-turnover cost is a large part of the net target.

### Presence

No creation-feature model improves pooled presence RMSE over the weighted-mean
baseline. The compact three-input model has `R2 = -0.012`.

This is useful negative evidence. On-ball time, potential assists, and rim
assists belong in the explicit passer prior. They should not be reused to
explain the remaining presence term.

## Collinearity

The three proposed inputs are strongly correlated on the mapped surface:

- on-ball time vs potential assists: `0.875`;
- on-ball time vs rim assists: `0.732`;
- potential assists vs rim assists: `0.822`.

Ridge regularization is required. The empirical result favors the more
complete opportunity label—potential assists—over made-rim-only assists once
on-ball share is known.

## Modeling decision

Use the following architecture for the next NBA/WNBA prior experiment:

1. scoring prior: keep the separate scoring block (`Points/100`, `TSA/100`,
   and role points) on the scoring contribution;
2. passer prior: use on-ball share plus potential assists, with rim assists as
   an optional redundant feature rather than a required third axis;
3. passing turnover: model as its own cost child rather than expecting the
   creation-volume inputs to recover it;
4. presence: leave zero-centered until spacing, screen, relocation, and
   defensive-attention features demonstrate incremental value against the
   centered presence target.

The WNBA evidence validates the role split and the potential-assist lane. It
does not validate a complete oTS prior, because this shot model does not yet
contain the full FT component and all three RTSS seasons are development data.

## Reproduction

Run:

```bash
.venv/bin/python wnba_test/build_wnba_rtss_playmaking_roles.py --workers 4
```

Use `--rebuild-features` after changing the on-ball feature definition.

Key ignored outputs:

- `wnba_test/Results/rtss_playmaking_roles/role_coefficients.csv`;
- `wnba_test/Results/rtss_playmaking_roles/player_season_role_values.csv`;
- `wnba_test/Results/rtss_playmaking_roles/role_alpha_cv.csv`;
- `wnba_test/Results/rtss_playmaking_roles/playmaking_prior_loso_folds.csv`;
- `wnba_test/Results/rtss_playmaking_roles/playmaking_prior_loso_summary.csv`;
- `wnba_test/Results/rtss_playmaking_roles/playmaking_prior_loso_predictions.csv`;
- `wnba_test/Results/rtss_playmaking_roles/source_diagnostics.csv`; and
- `wnba_test/Results/rtss_playmaking_roles/manifest.json`.

