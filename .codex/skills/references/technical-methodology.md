# Technical Methodology: Six-Factor RAPM

## Table of Contents
- [Standard RAPM Baseline](#standard-rapm-baseline)
- [The Decomposition Move](#the-decomposition-move)
- [Regression Specification](#regression-specification)
- [The Units Problem](#the-units-problem)
- [Reconstruction Regression](#reconstruction-regression)
- [Why This Result Is Not Mechanical](#why-this-result-is-not-mechanical)
- [ShotQuality TS Decomposition Methodology](#shotquality-ts-decomposition-methodology)

## Standard RAPM Baseline

Regularized Adjusted Plus-Minus takes every stint of NBA play (continuous stretch where the same 10 players are on the court) and runs a ridge regression predicting point differential per 100 possessions from player indicator variables.

- **Design matrix**: One column per player (~775 active players). +1 for on-court offense, -1 for on-court defense. Each row is a stint.
- **Dependent variable**: Point differential per 100 possessions for the stint.
- **Ridge penalty**: Handles collinearity (starters play together constantly) and small-sample lineup combinations.
- **Output**: One coefficient per player = their isolated impact on point differential, controlling for all teammates and opponents.

Traditional split: ORAPM (offensive coefficient) + DRAPM (defensive coefficient) = total RAPM.

**Important**: This "no box priors" paragraph applies to the **pure** six-factor RAPM series. The repo now also contains a hybrid empirical-Bayes blend and a true prior-centered solve. Do not apply the pure-method caveat to those series.

## Current 2026 Methodology Variants

### Pure six-factor RAPM

- Lineup-based, time-decayed (`td700`) solve with no box priors in the actual regression.
- In current comparison files, this shows up as `actual_*` or `pure_*`.

### Hybrid six-factor RAPM

- Built by `rapm/russell/build_hybrid_2026_factors.py`.
- This is not a re-solved RAPM with nonzero priors.
- It is a post-solve empirical Bayes blend:

  `hybrid = (pure * factor_specific_possessions + box_prior * k) / (factor_specific_possessions + k)`

### True-prior six-factor RAPM

- Built by `rapm/russell/build_true_prior_factor_rapm.py`.
- This is a real prior-centered solve inside the regression:

  `min ||W^(1/2)(y - Xβ)||² + λ_off ||β_off - μ_off||² + λ_def ||β_def - μ_def||²`

- Use `prior_*` to refer to this series in current-player comparison exports.

## The Decomposition Move

Take the exact same regression machinery -- same player-in-lineup design matrix, same stint-level observations, same ridge regression -- and run it three separate times, each time predicting a different dependent variable:

1. **TS regression**: Predicts True Shooting Percentage. Produces oTS (offensive) and dTS (defensive) coefficients per player.
2. **TOV regression**: Predicts Turnover Rate. Produces oTOV (offensive) and dTOV (defensive) coefficients per player.
3. **REB regression**: Predicts Offensive Rebound Rate. Produces oREB (offensive) and dREB (defensive) coefficients per player.

Each regression uses the same +1/-1 design matrix, so offense and defense coefficients within each model are estimated simultaneously (they are NOT independent of each other). What IS independent is the TS model from the TOV model from the REB model.

## Regression Specification

- **Method**: Ridge regression
- **Ridge alpha**: 3000 (for the ShotQuality decomposition; six-factor may differ -- check pipeline)
- **Design matrix**: One column per player, +1 on-court offense, -1 on-court defense, alternating offense/defense blocks
- **Time-decay**: Half-life of 700 days. Recent games weighted more heavily. Allows using ~3 seasons of data for stability while keeping recent performance dominant.
- **Data scope**: ~650,000 non-turnover possessions from 2023-24 through 2025-26 seasons (for ShotQuality TS decomposition). Six-factor uses 2021-2026 time-decayed data (~1071 players).
- **Output**: Offensive and defensive coefficients per player, expressed as impact vs league average

## The Units Problem

Each factor-RAPM is in its own native units:
- oTS-RAPM is in True Shooting Percentage units
- oTOV-RAPM is in Turnover Rate units
- oREB-RAPM is in Offensive Rebound Rate units

You **cannot** just add them together and get RAPM back -- that would be like adding inches and kilograms.

## Reconstruction Regression

To test whether the six factors explain full RAPM, run a second-stage regression:

```
RAPM_i = B1(oTS_i) + B2(oTOV_i) + B3(oREB_i) + B4(dTS_i) + B5(dTOV_i) + B6(dREB_i) + e_i
```

This regression does three things:
1. **Finds beta weights** -- conversion factors from native units to points-per-100-possessions. B1 = how many points a 1% change in TS is worth. B2 = how many points a 1% change in TOV is worth. Etc.
2. **Reveals relative importance** -- which factors carry the most point-value weight in determining total impact.
3. **Tests explanatory power** -- how much of full RAPM these independently estimated factors explain.

**Result: R^2 = 0.985**

The six factors explain approximately 98.5% of the variance in full RAPM. The residual (RESID column in the data) is typically < 0.6 for any individual player.

## Why This Result Is Not Mechanical

This result is NOT guaranteed. The six factors were estimated in completely separate regressions with completely different dependent variables. There is no mathematical constraint forcing them to reconstruct RAPM. They could have explained 60%, or 75%, and the remaining variance would represent a mysterious residual -- impact that doesn't flow through shooting, turnovers, or rebounding.

But they don't leave much residual at all. Nearly everything about a player's point-differential impact can be accounted for by their influence on three pairs of mechanical levers: shooting efficiency (both ends), turnover creation/avoidance (both ends), and rebounding extension/denial (both ends).

That's not a decomposition -- it's a discovery. It tells you something about the structure of basketball itself.

## ShotQuality TS Decomposition Methodology

### The Exact-Sum Insight

Every non-turnover possession can be scored three ways that sum exactly to the TS value:

**For FGA possessions:**
- **TS** = Actual points scored
- **SQ** = ShotQuality pre-shot expected value (based on shot location, closest defender distance, shot type, shooter movement, touch time, and other contextual factors)
- **CONTEST** = Actual points - SQ expected value
- **FT** = 0

Sum: SQ + Contest + FT = Actual points = TS. Exact.

**For FT possessions:**
- **TS** = Expected FT value (shooter's expected FT conversion, removes luck)
- **SQ** = Foul-type-aware baseline for the shot value that the foul replaced
- **CONTEST** = 0
- **FT** = Expected FT value minus that foul-type-aware baseline

Sum: SQ + Contest + FT = Expected FT value = TS. Exact.

### Foul-Type-Aware FT Baselines

Let `avg_pts` be the average points scored on FGA possessions.

- **2-shot foul**: `SQ = avg_pts / 2` per FT, `FT = ExpFT - avg_pts / 2`
- **3-shot foul**: `SQ = avg_pts / 3` per FT, `FT = ExpFT - avg_pts / 3`
- **And-1**: `SQ = 0`, `FT = ExpFT`
- **Technical**: `SQ = 0`, `FT = ExpFT`

### Four Parallel Regressions

Same design matrix, same stints, same ridge alpha. Only the scoring of each possession changes:

1. **TS regression**: FGAs at actual points, FTs at expected FT value
2. **SQ regression**: FGAs at ShotQuality pre-shot EV, FTs at the foul-type-aware SQ baseline
3. **FT regression**: FGAs at 0, FTs at expected FT value minus the foul-type-aware SQ baseline
4. **CONTEST regression**: FGAs at actual minus SQ EV, FTs at 0

Each produces offensive and defensive coefficients per player.

### Units

Points per 100 **shots** (NOT possessions). Turnover possessions are excluded from the matrix entirely because they have no shot to decompose.

### Second-Stage WLS Confirmation

```
Off TS = B1(Off SQ) + B2(Off FT) + B3(Off Contest)
Def TS = B4(Def SQ) + B5(Def FT) + B6(Def Contest)
```

- Betas land at ~0.95-1.0
- R^2 > 0.9999
- Residual < 0.3 pts/100 shots (attributable to differential ridge shrinkage across regressions, not missing basketball signal)

### The Nested Decomposition

Level 1: RAPM -> six factors (R^2 = 0.985)
Level 2: TS factors -> SQ + Contest + FT (R^2 > 0.9999)

Two layers of independently estimated, mechanically grounded decomposition that both reconstruct nearly all of the parent signal. The game keeps reducing to identifiable mechanical levers.
