# Randle Strong/Weak RAPM Notes

Run date: 2026-05-22

Window: `2021-26 ALL`, using processed `RAPMYY.parquet` and `RAPMYY_PS.parquet` rows.

## Construction

The prior production split used redundant player base plus two bucket features. This harness uses identifiable constructions:

- `binary_reference`: weak/not-strong is the base environment, strong is one player-side delta.
- `continuous_slope`: opponent strength is season-standardized DARKO lineup strength, and each player-side gets one slope around average opponent strength.

Interaction features are column-scaled so they receive stronger ridge shrinkage than base RAPM effects. The grid tested interaction penalty multipliers `2`, `4`, `8`, and `16` at base alpha `1000`.

## Validation

Grouped holdout is by `game_id`.

| Interaction mult | Baseline RMSE | Binary RMSE | Binary improvement | Continuous RMSE | Continuous improvement |
|---:|---:|---:|---:|---:|---:|
| 2 | 1.172601 | 1.172653 | -0.000052 | 1.172586 | +0.000016 |
| 4 | 1.172601 | 1.172609 | -0.000008 | 1.172502 | +0.000100 |
| 8 | 1.172601 | 1.172588 | +0.000014 | 1.172456 | +0.000145 |
| 16 | 1.172601 | 1.172583 | +0.000018 | 1.172454 | +0.000148 |

Read: the binary split is mostly diagnostic and only barely helps when shrunk hard. The continuous slope is the better research path, but the predictive gain is tiny. Treat this as a signal extractor, not a finished public metric.

## Recommended Outputs

- Best binary companion: `outputs/binary_reference_21_26_all_a1000_im16.csv`
- Best continuous model: `outputs/continuous_slope_21_26_all_a1000_im16.csv`
- Validation grid: `outputs/validation_grid.csv`
- Diagnostics: `outputs/diagnostics_21_26_all_a1000_im16.json`

## Current Read

The continuous model with interaction multiplier `16` is the sanest current attempt. It preserves stars at the top better than lower-shrink runs while improving held-out prediction more than the binary split.

For players with at least 10,000 possessions, the top continuous observed-net group starts:

| Player | Observed net | Net vs weak -1 SD | Net avg | Net vs strong +1 SD | 2 SD strong-minus-weak |
|---|---:|---:|---:|---:|---:|
| Nikola Jokic | 6.03 | 7.16 | 6.20 | 5.25 | -1.92 |
| Shai Gilgeous-Alexander | 5.90 | 8.65 | 6.13 | 3.60 | -5.05 |
| Franz Wagner | 5.63 | 6.76 | 5.57 | 4.39 | -2.37 |
| Victor Wembanyama | 5.61 | 5.92 | 5.67 | 5.42 | -0.50 |
| Dereck Lively II | 5.28 | 6.13 | 5.47 | 4.81 | -1.32 |

For the binary reference model with interaction multiplier `16`, the top qualified group starts:

| Player | Overall net | Off strong | Off weak | Def strong | Def weak | Net strong-minus-weak |
|---|---:|---:|---:|---:|---:|---:|
| Nikola Jokic | 9.40 | 7.49 | 8.29 | -1.22 | -1.47 | -1.05 |
| Victor Wembanyama | 8.66 | 1.25 | 0.93 | -7.44 | -7.75 | +0.01 |
| Shai Gilgeous-Alexander | 7.68 | 6.52 | 7.43 | -0.16 | -0.72 | -1.47 |
| Giannis Antetokounmpo | 7.54 | 3.71 | 4.10 | -3.64 | -3.46 | -0.21 |
| Kawhi Leonard | 7.33 | 5.70 | 6.52 | -0.73 | -1.24 | -1.33 |

## Next Work

- Add bootstrap intervals for player deltas/slopes.
- Add a minimum-exposure reliability score for strength response.
- Test playoff-only and recent-playoff-heavy weighting.
- Try team/opponent heldout splits, not just game heldout, to see whether the response terms generalize across opponent contexts.
