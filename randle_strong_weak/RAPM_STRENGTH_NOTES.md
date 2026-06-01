# RAPM-Strength Two-Stage Check

## Construction

This reruns the two-stage residual harness with the same frozen stage-one RAPM base:

- Seasons: `2020-21` through `2025-26`
- Stage-one base: `rapm_21_26_all_pure_a2000_4000_results.csv`
- Stage-one alphas: offense `2000`, defense `4000`
- Validation: 20% grouped holdout by `game_id`, seed `42`

Only the opponent-strength source changed.

For each possession:

- offensive opponent strength = `-sum(opposing defenders' frozen def RAPM)`
- defensive opponent strength = `sum(opposing offensive players' frozen off RAPM)`

The defensive sign is flipped because the exported defensive RAPM coefficient is an allowed-points coefficient, where lower is better.

## Results

Base-only validation RMSE: `1.171784723`

| Strength Source | Best Model | Validation RMSE | Improvement vs Base |
|---|---|---:|---:|
| DARKO DPM | `global_continuous` | `1.171642454` | `+0.000142270` |
| Frozen RAPM | `global_continuous` | `1.171709383` | `+0.000075340` |

Frozen RAPM lineup strength still helps globally, but less than DARKO DPM strength.

## Interpretation

The RAPM-strength result says the frozen RAPM model still leaves some global lineup-strength structure in its residuals. That is evidence of mild nonlinearity or underpricing in the additive base RAPM itself.

The DARKO-strength result being stronger says DARKO is adding information not fully present in the frozen RAPM values. That could be because DARKO carries box-score, aging, trajectory, and current-role information that the six-year RAPM base does not fully capture.

Player-specific residuals are weaker in the RAPM-strength version. The best player table at alpha `8000` still loses to the frozen base on holdout:

| Model | Validation RMSE | Improvement vs Base |
|---|---:|---:|
| `player_continuous_after_global` | `1.171847185` | `-0.000062462` |
| `player_continuous_slope` | `1.171851764` | `-0.000067041` |
| `player_binary_after_global` | `1.171894230` | `-0.000109507` |
| `player_binary_delta` | `1.171902553` | `-0.000117830` |

Best read: RAPM-based lineup strength is useful as an internal audit, but DARKO is the better opponent-strength prior for validated correction.

