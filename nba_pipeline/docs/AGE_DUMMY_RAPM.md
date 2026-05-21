# Age-Dummy RAPM / Factor Mode

## Summary

`rapm.py` supports an optional `--age-dummies` mode for RAPM and the factor
surfaces used by the daily weighted-factor build.

This mode adds separate age-bin fixed effects directly into the possession regression:
- offensive players at age `a` contribute `+1` to `age_off_fe_a`
- defensive players at age `a` contribute `+1` to `age_def_fe_a`
- ages use integer buckets `18` through `38`
- `38+` collapses into `38`

The resulting coefficients produce three curves in points per 100 possessions after controlling for:
- player offense coefficients
- player defense coefficients
- optional rubberband effects
- optional season fixed effects

For non-point factors, the output is still scaled the same way as the metric
solve's normal `off` / `def` columns: coefficient times `100`. Interpret each
age curve in that factor's native target units, not as total point value unless
the target itself is a point-differential surface.

Exported views:
- offense age curve
- defense age curve
- net age curve computed as `off - def`

This is different from [`AGE_CURVE_RAPM.md`](./AGE_CURVE_RAPM.md), which builds an age-based prior and does **not** add age features to the matrix.

## Identification

Because every possession has five offensive players and five defensive players, each side-specific age block still has a reference-category issue.

So the solver omits age `27` as the computational reference bucket for both offense and defense.

The exported `*_age_effects.csv` file then recenters the offense and defense curves separately to weighted mean zero across observed age slots, and reports net as `off_centered - def_centered`.

## Age Source

Age bins come from the DARKO history file using:
- default path: `/Users/russellthomas/Docs/2026_NBA_PIPELINE/databallr/darko/dpm_history.csv`
- override path: `--darko-history`
- `nba_id`
- `season`
- `age`

Only the `age` lookup is used from this file for age dummies. DARKO `dpm`,
`o_dpm`, and `d_dpm` are not used as regression targets for this mode.

Season assignment rules:
- possessions use the processed parquet `Season` year
- for each player-season, the solver looks up that player's age row at the same season end year
- if that exact season is missing, it falls back to the latest DARKO age row at or before that season
- if a player has no age row at all, that player's age features stay blank and the player is still estimated normally through the player coefficients

## Command

Single metric:

```bash
cd nba_pipeline/scripts
python rapm.py RAPM 14 26 ALL --pure --rubberband --season-effects --age-dummies
python rapm.py TS 14 26 ALL --rubberband --season-effects --age-dummies
python rapm.py ALT_TS 14 26 ALL --rubberband --season-effects --age-dummies
```

Daily weighted-factor suite with age dummies on every metric solve:

```bash
cd nba_pipeline/scripts
python 03_run_rapm_analysis.py 14 26 ALL --rubberband --age-dummies
```

That run includes the standard weighted-factor metrics (`RAPM`, `TS`, `TOV`,
`REB`, `BADPASS_TOV`, `SCORING_TOV`, `SECOND_CHANCE`, `FT_PREMIUM`) and the
alternate clean 3-factor metrics (`FIRST_CHANCE`, `ALT_TS`, `ALT_EFG`,
`ALT_FT`, `ALT_TOV`, `ALT_BADPASS_TOV`, `ALT_SCORING_TOV`,
`SECOND_CHANCE_CLEAN`, plus `ALT_SQ` / `ALT_MAKE` when the start year is 2024
or later). Add `--season-effects` if the run should also estimate season fixed
effects.

## Output

When `--age-dummies` is enabled:
- result filenames include `_agefe`
- the solver writes a companion `*_age_effects.csv`

Example outputs:
- `rapm_14_26_all_pure_rb_se_agefe_results.csv`
- `rapm_14_26_all_pure_rb_se_agefe_age_effects.csv`
- `ts_14_26_all_rb_se_agefe_age_effects.csv`
- `alt_ts_14_26_all_rb_se_agefe_age_effects.csv`
- `weighted_factors_14_26_all_rb_agefe.csv`
- `weighted_factors_alt3_14_26_all_rb_agefe.csv`

The age-effects file contains:
- `age`
- `off_raw_effect_per_100`
- `off_centered_effect_per_100`
- `def_raw_effect_per_100`
- `def_centered_effect_per_100`
- `net_centered_effect_per_100`
- `off_slots`
- `def_slots`
- `total_slots`
- `is_reference`

Main columns to inspect:
- `off_centered_effect_per_100`
- `def_centered_effect_per_100`
- `net_centered_effect_per_100`
