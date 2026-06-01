# Two-Stage Residual Strength RAPM

## Question

Can we fit ordinary six-year RAPM once, freeze those player values, then run a second pass with only opponent-strength variables?

Yes. The second pass is useful, but the validation result is very specific: the signal is mostly a global schedule-strength/environment adjustment, not a stable player-specific strong/weak skill split.

## Construction

Run scope:

- Seasons: `2020-21` through `2025-26` (`RAPM21` through `RAPM26`, season keys `2021-2026`)
- Season type: `ALL` (`RS + PS`)
- Rows: `1,514,781`
- Players: `1,096`
- Stage 1 base: `nba_pipeline/results/rapm_21_26_all_pure_a2000_4000_results.csv`
- Stage 1 alphas: offense `2000`, defense `4000`
- Validation: grouped holdout by `game_id`, 20%, seed `42`

Stage 1 fixed prediction:

```text
target_mean + sum(off_player_base_values) + sum(def_player_base_values)
```

Stage 2 residual target:

```text
raw_possession_target - target_mean - fixed_stage1_player_prediction
```

Stage 2 models tested:

- `global_binary`: two variables only
  - offense vs strong opposing defense
  - defense vs strong opposing offense
- `global_continuous`: two variables only
  - offense slope vs season-z-scored opposing defensive DARKO
  - defense slope vs season-z-scored opposing offensive DARKO
- `player_binary_delta`: player-specific residual strong deltas
- `player_continuous_slope`: player-specific residual continuous slopes
- `player_binary_after_global`: global binary first, then player residual deltas
- `player_continuous_after_global`: global continuous first, then player residual slopes

## Validation Results

Base-only validation RMSE: `1.171784723`

| Rank | Model | Alpha | Validation RMSE | Improvement vs Base |
|---:|---|---:|---:|---:|
| 1 | `global_continuous` | none | `1.171642454` | `+0.000142270` |
| 2 | `global_binary` | none | `1.171746617` | `+0.000038106` |
| 3 | `player_continuous_after_global` | `8000` | `1.171755821` | `+0.000028902` |
| 4 | `player_continuous_slope` | `8000` | `1.171765975` | `+0.000018748` |
| 5 | `player_binary_after_global` | `8000` | `1.171791712` | `-0.000006988` |
| 6 | `player_binary_delta` | `8000` | `1.171804333` | `-0.000019610` |

Lower player residual alphas overfit. The player-specific terms only avoid damage when heavily shrunk, and even then they do not beat the two-variable global continuous model.

## Best Two-Variable Adjustment

`global_continuous` coefficients per 100:

- `global_off_opp_def_z_slope`: `-1.4375`
- `global_def_opp_off_z_slope`: `+1.4766`

Interpretation:

- After fixed 2000/4000 base RAPM, a `+1 SD` stronger opposing defense lowers expected offense by about `1.44` points per 100.
- A `+1 SD` stronger opposing offense worsens defensive allowed-points coefficient by about `1.48` points per 100.

This is exactly the kind of effect we should expect if the original RAPM is averaging across uneven lineup environments. It is not proof that individual stars have stable, separable strong/weak resilience.

## Player-Level Read

The most honest player table is `player_continuous_slope_21_26_all_a8000.csv`: it keeps the fixed base RAPM and estimates only player residual slopes with heavy shrinkage.

Examples from that table:

| Player | Base Net | Observed Net | Net vs Weak 1SD | Net vs Avg | Net vs Strong 1SD | Strong - Weak 2SD |
|---|---:|---:|---:|---:|---:|---:|
| Nikola Jokic | `+9.12` | `+9.14` | `+9.01` | `+9.12` | `+9.24` | `+0.23` |
| Shai Gilgeous-Alexander | `+7.72` | `+7.54` | `+10.12` | `+7.72` | `+5.33` | `-4.79` |
| Giannis Antetokounmpo | `+7.24` | `+7.37` | `+6.27` | `+7.24` | `+8.21` | `+1.94` |
| Victor Wembanyama | `+6.71` | `+6.74` | `+6.23` | `+6.71` | `+7.19` | `+0.96` |

For player-specific interpretation, treat deltas as exploratory. The validated claim is weaker and cleaner:

> A two-stage residual pass finds real global opponent-strength compression, but stable player-specific strong/weak splits are not strongly supported by grouped holdout.
