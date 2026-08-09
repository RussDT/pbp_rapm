# Active DECOMP and ShotQuality-EV VPM

## Table of Contents

- [Model Families](#model-families)
- [Public Eight-Component DECOMP](#public-eight-component-decomp)
- [Source-Row Shot Algebra](#source-row-shot-algebra)
- [Ten-Component VPM](#ten-component-vpm)
- [Frozen VPM Alphas](#frozen-vpm-alphas)
- [Impact Monster Product Contract](#impact-monster-product-contract)
- [Current Evidence](#current-evidence)
- [Data Admission](#data-admission)
- [Repository Ownership](#repository-ownership)
- [Implementation Checklist](#implementation-checklist)

## Model Families

Keep these surfaces distinct:

- `weighted_factors_decomp_*`: active public actual-points decomposition.
- ShotQuality-EV VPM: predictive research evolution of DECOMP with a ten-part
  additive tree.
- `weighted_factors_alt3_efg_value_*`: legacy EFG-value display/audit family.
- `weighted_factors_alt3_*`: older legacy family.

Do not feed an Alt3 schema into a DECOMP consumer without an explicit mapper
audit. Do not call an Alt3 balancing residual a DECOMP component.

## Public Eight-Component DECOMP

Use these player-facing components:

1. `DECOMP_RIM_FREQ`
2. `DECOMP_RIM_FG`
3. `DECOMP_MID_FG`
4. `DECOMP_THREE_FREQ`
5. `DECOMP_THREE_FG`
6. `ALT_FT`
7. `ALT_TOV_VALUE`
8. `SECOND_CHANCE_CLEAN`

The first seven live on the shared `FIRST_CHANCE` possession universe;
`SECOND_CHANCE_CLEAN` is the direct second-chance surface.

Interpret the five shot children as:

- rim frequency: value from shifting attempts between rim and midrange;
- rim FG: actual rim points relative to the loaded-sample rim baseline;
- mid FG: the replacement/midrange base value and actual midrange points;
- three frequency: value from shifting attempts between three and midrange;
- three FG: actual three-point value relative to the three baseline.

`ALT_FT` is first-chance free-throw value. `ALT_TOV_VALUE` is the point-valued
first-chance turnover complement. `SECOND_CHANCE_CLEAN` is direct value after
the first chance; it is not a balancing bucket.

## Source-Row Shot Algebra

Let `r`, `m`, and `t` be the loaded sample's mean first-chance points per rim,
midrange, and three-point attempt.

On a rim FGA:

```text
DECOMP_RIM_FREQ = r - m
DECOMP_RIM_FG = actual - r
DECOMP_MID_FG = m
```

On a midrange FGA:

```text
DECOMP_MID_FG = actual
```

On a three-point FGA:

```text
DECOMP_MID_FG = m
DECOMP_THREE_FREQ = t - m
DECOMP_THREE_FG = actual - t
```

All unused shot children are zero. On an FT event, the foul-type-aware
replacement-shot baseline lives in `DECOMP_MID_FG`. Therefore:

```text
DECOMP_RIM_FREQ
+ DECOMP_RIM_FG
+ DECOMP_MID_FG
+ DECOMP_THREE_FREQ
+ DECOMP_THREE_FG
= FC_EFG_Diff
```

The processor must prove this at action and processed-possession grain. The
normal tolerance is `1e-9` or tighter.

For first-chance turnovers, compute each alias's imputation from the exact
loaded non-turnover sample. In time-decay runs, use the same time-decay weights
for those means. Preserve `Def_Diff = -Off_Diff` in point/value parquets.

## Ten-Component VPM

VPM keeps the public tree's structure but denoises the first midrange or
three-point attempt with a frozen walk-forward expected value.

Use:

1. `DECOMP_RIM_FREQ`
2. `DECOMP_RIM_FG`
3. `DECOMP_MID_FG`, with midrange EV substituted
4. `DECOMP_THREE_FREQ`
5. `DECOMP_THREE_FG`, with three-point EV substituted
6. `ALT_FT`
7. `ALT_TOV_VALUE`
8. `DECOMP_MID_VALUE`, midrange `actual - EV`
9. `DECOMP_EFG`, three-point `actual - EV`
10. `SECOND_CHANCE_CLEAN`

Target rules:

- Keep rim actual.
- Attach EV only to the chronological first first-chance midrange/three FGA.
- Use clock-first event ordering; corrected events can have late event numbers
  at earlier clocks.
- Reject conflicting EV keys and zone mismatches.
- Preserve original parent and reserved columns in explicit audit fields.
- Carry the two `actual - EV` residuals so the ten-part player sum still
  targets actual possession points.

The frozen shipped target uses `RUSSELL_SHOTQUALITY v2` walk-forward EV
parquets. Do not claim that it uses newer rich geometry, historical
safe-crossfit, shot-clock/grenade, defender-role, or continuation models.

## Frozen VPM Alphas

Use the checked-in ShotQuality-EV component alphas unless a new immutable
experiment config and forward selection receipt replace them:

| Component | Offense alpha | Defense alpha |
|---|---:|---:|
| `DECOMP_RIM_FREQ` | 750 | 750 |
| `DECOMP_RIM_FG` | 1,000 | 750 |
| `DECOMP_MID_FG` | 750 | 1,500 |
| `DECOMP_THREE_FREQ` | 1,000 | 2,000 |
| `DECOMP_THREE_FG` | 1,000 | 2,000 |
| `ALT_FT` | 750 | 750 |
| `ALT_TOV_VALUE` | 2,000 | 6,000 |
| `DECOMP_MID_VALUE` | 8,000 | 48,000 |
| `DECOMP_EFG` | 6,000 | 48,000 |
| `SECOND_CHANCE_CLEAN` | 2,000 | 3,000 |

Use TD550 for the accepted/current VPM estimate. Use offense/defense alphas
`2000/4000` for the plain-RAPM comparison lane. Do not confuse prior-mapper
ridge alphas with component solve alphas.

## Impact Monster Product Contract

Treat Impact Monster as the ten-factor offense/defense breakdown itself.
Present one player hierarchy:

```text
player -> offense + defense = total VPM
       -> TS + TOV + SC on each side
       -> ten atomic component types on each side
```

Within each TS tile, expose these exact rollups without renaming or dropping
the atomic inputs:

```text
Rim = rim frequency + rim FG
Mid = mid EV + mid actual-minus-EV residual
3P = three frequency + three EV + three actual-minus-EV residual
FT = FT value
```

Keep `TOV` and `SC` as direct single-component factors. Keep broader mechanism
drilldowns visibly separate and never stack them onto VPM total.

Use positive-good defense in reader-facing output. Distinguish additive player
sums from lineup/synergy forecasts.

## Current Evidence

Treat this as a dated July 2026 evidence snapshot:

- The 2016-2026 target surface contains 22 RS/PS files and 2,781,191
  first-chance possession rows.
- Its checked rebuild attached 1,468,174 EV rows and closed with maximum row
  gap `5.35e-14`.
- EV no-prior improved future actual-possession MSE versus plain RAPM on the
  registered development surface.
- The completed-season skill-prior x2 lane added `+8.24e-6` beyond EV
  no-prior, won 4/4 cuts, and had a paired 95% interval
  `[+2.79e-6, +13.62e-6]`.
- EPM12 x2 added `+153.02e-6` beyond EV no-prior, won 4/4 cuts, and had a
  paired interval `[+73.38e-6, +233.87e-6]`.
- EPM12 x2 beat the accepted box-prior x2 lane by `+144.70e-6` on that
  development surface, interval `[+67.83e-6, +223.85e-6]`.
- The earlier direct pure-VPM versus complete-EPM interval crossed zero. Do
  not describe that earlier comparison as an EPM victory.

EPM12 x2 is the leading research base, not silently promoted untouched
confirmation. The 2025-26 tiles are spent; reserve 2026-27 or another frozen
future window for the next promotion decision.

These possession-level results are necessary intermediate evidence. The
long-term product claim requires direct future-game prediction against EPM.

## Data Admission

| Source | Status | Allowed VPM use |
|---|---|---|
| Frozen walk-forward mid/three EV | Accepted | Target construction |
| Completed-season box/playtype skills | Accepted | Stabilized priors |
| Date-aware EPM `p_*` skills | Leading research | Candidate priors |
| Complete EPM | Benchmark only | Comparison or frozen ensemble |
| `eWINS` | Forbidden | Never a prior feature |
| Lagged repo-owned shooter talent | Candidate | Requires incremental forward gate |
| Shot-clock/grenade summaries | Blocked | Require multi-season reliability |
| Defender/role/continuation summaries | Candidate or blocked | Require forward gate |
| Raw provider shots/defender joins | Forbidden | Never direct VPM input |
| Shot/event fold predictions | Forbidden | Never direct VPM input |

## Repository Ownership

RAPM owns:

- canonical raw and processed PBP;
- possession construction and DECOMP algebra;
- solver helpers;
- `nba_pipeline/scripts/build_vpm_target_swap.py`.

The VPM research repo owns:

- frozen EV parquets;
- immutable configs and target-source routing;
- priors, evaluation, bootstrap, holdouts, evidence, and boards;
- `vpm/run_harness.py`, the collaborator interface.

Generated target parquets stay isolated under
`nba_pipeline/results/vpm_target_swap/` and out of git.

## Implementation Checklist

Before accepting a DECOMP/VPM change:

1. Confirm the exact parent and row universe.
2. Prove action and processed-row identities.
3. Confirm offense/defense signs.
4. Freeze source hashes, config ID, alphas, and date contracts.
5. Compare on identical future actual-outcome rows.
6. Report paired game-clustered uncertainty versus plain RAPM, EV no-prior,
   and EPM where applicable.
7. Keep component diagnostics subordinate to the summed forecast.
8. Record rejected challengers as carefully as accepted ones.
9. Never reopen a spent holdout for model selection.
10. Preserve the RAPM/VPM repository boundary.
