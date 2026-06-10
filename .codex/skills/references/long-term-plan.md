# Six-Factor RAPM Long-Term Plan

This note captures the intended long-term direction for the six-factor RAPM stack. It is not the same thing as the active production spec. Use it when the task is about:

- future modeling direction
- long-term roadmap for six-factor RAPM
- how to improve the current `2026` single-season true-prior beta
- EPM-inspired prior architecture
- estimated skills / stabilized stat priors
- replacing the current messy prior-testing lane with a cleaner single-season validation loop

## Goal

Build a `single-season impact metric` that:

- keeps the final lineup solve current-season only
- uses better priors than raw current-season box rows
- produces all 9 public metrics:
  - `ORAPM`
  - `DRAPM`
  - `RAPM`
  - `oTS`
  - `oTOV`
  - `oREB`
  - `dTS`
  - `dTOV`
  - `dREB`
- can compete with or beat `BPM`, `LEBRON`, and `EPM` as a current-season impact metric

The final product should be:

- single-season by lineup likelihood
- prior-informed, not pure
- factor-structured, not one monolithic all-in-one

## Current Problem

The current prior lane was inherited from a multi-year box-to-factor workflow.

What it currently does well:

- finds useful features
- produces stable priors
- avoids fully ad hoc box weights

What is currently misaligned:

- the legacy prior-testing target is a multi-year, time-decayed, rubberbanded factor snapshot
- the current product goal is single-season, no within-season time decay
- this means the prior validation target is cleaner than the product target, but not the same object

So the current lane is acceptable for feature screening, but not the final scientific justification for the single-season product.

## Product Direction

The long-term intended architecture is:

1. `Estimated Skills`
2. `Factor-Specific SPM Priors`
3. `Single-Season True-Prior RAPM`

Conceptually:

`raw stats -> estimated skill stats -> factor priors -> current-season RAPM solve`

This is the main EPM-inspired idea worth borrowing. The point is not to copy EPM exactly. The point is to use:

- stat-specific stabilization
- multi-year information for skill estimation
- factor-specific priors
- current-season lineup truth in the final solve

## Long-Term Architecture

### 1. Estimated Skill Layer

For each backend stat we care about, build a stabilized `skill` estimate rather than relying on the raw current-season box row.

Examples:

- `oTS` skills:
  - scoring efficiency skill
  - burden-adjusted scoring skill
  - rim-assist creation skill
  - spacing / gravity skill
- `oTOV` skills:
  - scoring-turnover skill
  - bad-pass turnover skill
  - offensive load skill
  - hub creation / decision burden skill
- `oREB` skills:
  - self-OREB skill
  - reboundable miss creation
  - positional burden-adjusted OREB skill
- `dTS` skills:
  - rim deterrence skill
  - rim contest skill
  - foul discipline skill
  - shot suppression / contest skill
- `dTOV` skills:
  - steal skill
  - deflection skill
  - forced-turnover skill
  - disruption / chaos skill
- `dREB` skills:
  - defensive rebound conversion skill
  - box-out skill
  - block recovery skill
  - cleanup / team-finish skill

These skills should be:

- date-aware
- age-aware where useful
- stat-specific in stabilization speed
- context-aware when the stat calls for it

The purpose is to separate `true talent` from `single-season noise`.

### 2. Factor-Specific SPM Priors

Do not use one giant all-in-one prior as the final target.

Train separate priors for:

- `oTS`
- `oTOV`
- `oREB`
- `dTS`
- `dTOV`
- `dREB`

Each factor should get its own:

- target
- feature family
- shrinkage profile
- validation scorecard

This is the core edge of the six-factor framework relative to monolithic all-in-ones.

### 3. Single-Season True-Prior Solve

The final metric should remain:

- current-season only in the lineup likelihood
- prior-centered in the regression
- solved separately for scoring margin and the three factor families

Long-term preference:

- no within-season time decay for the published single-season series unless a no-decay backtest clearly loses
- priors should be strong enough to stabilize, but not so strong that they pre-decide offensive stars or defensive specialists

## Validation Ladder

The intended validation ladder is:

### Stage 1: Feature Screening

Use the cleaner historical multi-year lane only for:

- feature ideation
- rough screening
- checking whether a new variable is directionally useful

Do not treat this as final validation for the single-season product.

### Stage 2: Single-Season Target Validation

Build pure single-season factor truth for each season.

Preferred target years:

- `2019`
- `2020`
- `2021`
- `2022`
- `2023`
- `2024`
- `2025`

This is noisy, but it is the correct object.

Use:

- possession weighting
- pooled multi-year evaluation
- role / minute bucket inspection

### Stage 3: Same-Season Cutoff Testing

Preferred primary validation:

- freeze a season at real cutoff dates
- fit the prior and true-prior solve using only information available at that cutoff
- predict the rest of the same season

This is the best direct test of whether the metric works in practice as an in-season tool.

### Stage 4: Year-Over-Year Testing

Preferred secondary validation:

- fit season `Y`
- test against season `Y+1`

This is useful for shrinkage calibration and talent-stability justification, but it should not be the only test because it can over-reward overly conservative priors.

### Stage 5: Competitor Benchmarking

Once the internal metric is stable, compare against:

- `BPM`
- `LEBRON`
- `EPM`

Do not train to match them. Use them only as external benchmarks.

## Alpha Philosophy

For the single-season true-prior solve:

- pure single-season RAPM is too noisy on its own
- priors should matter materially
- but the right alpha is not something to guess from multi-year pure RAPM defaults

Important principles:

- single-season PI-RAPM should usually shrink harder than multi-year pure RAPM
- prior-centered alpha is stronger than zero-centered alpha
- offensive and defensive alpha should be tuned separately
- factor priors should not all be trusted equally

Long-term preference:

- tune alpha on same-season cutoff tests
- use year-over-year as secondary support
- avoid picking alpha only from the inherited multi-year next-snapshot lane

## Current Modeling Priorities

The main next-step priorities are:

1. Replace raw current-season box rows with estimated skill rows where possible.
2. Keep the compact `oTS` backend formula philosophy, but feed it better stabilized skill inputs.
3. Build cleaner skill priors for `dTS`, `dTOV`, and `dREB`, where raw box inputs are especially noisy.
4. Introduce physical measurements where useful, especially:
   - `height_wo_shoes_in`
   - `wingspan_in`
5. Rebuild the validation harness around pure single-season targets.
6. Tune offense and defense shrinkage under the correct single-season spec.

## What Not To Do

- Do not treat the current multi-year box-prior holdout `R^2` as if it fully validates the single-season product.
- Do not collapse the six factors into one public training target.
- Do not let the priors become a broad “all offense halo” or “all defense halo”.
- Do not optimize only for year-over-year persistence.
- Do not mistake raw current-season box rows for the ideal final prior input layer.

## Desired End State

The desired end state is:

- a current-season true-prior RAPM product with all 9 metrics
- factor-specific priors built from stabilized skill stats
- validation aligned to single-season use cases
- clear evidence that the metric is more informative than pure RAPM and competitive with public all-in-ones

Short version:

`single-season lineup truth + multi-year skill estimation + factor-specific priors`

That is the long-term design target.
