# Shooting fusion results — September 14, 2026

Protocol: [SHOOTING_FUSION.md](SHOOTING_FUSION.md). Runner: `nba_pipeline/scripts/research_shooting_fusion.py`. Results: ignored `results/estimated_skills/shooting_fusion/`. This is an isolated chronological research comparison, not a change to the displayed model or VPM.

## Selection

Base filter selected on 2019 only: q_day=1e-7, r=1, initial variance .01. Six grid settings, no later labels in base selection. Nine candidate models receive the same pregame information boundary. Twenty ambiguous player-date rows were quarantined. Current shooting results never enter predictors; zero-attempt games contribute volume history.

2021–2022 initially selects linear ridge fusion alpha1000 by the one-SE primary rule. It improves binomial log loss by 0.0221% against the strongest simple model, but worsens next-game 3PM/100 MSE by 0.1021%; paired difference interval favors the simple control. It fails the joint gate. Boosting does not earn selection. The final choice is the half-strength volume+FT prior followed by Kalman residual updates.

## Historical stress

2023–2025 contains 83,680 player-games and 280,086 three-point attempts, with zero-attempt games included in the makes target. Versus the independently selected standalone Kalman control, the selected model improves:

| Objective | Relative loss reduction |
|---|---:|
| Binomial log loss per attempt | 0.0094% |
| Attempt-weighted squared percentage error | 0.0473% |
| Possession-weighted squared 3PM/100 error | 0.0554% |

Paired game-bootstrap intervals favor the selected model on all three. Against the matched volume-only prior, adding FT information improves log loss 0.0072% and percentage MSE 0.0410%; both intervals favor the addition. Joint makes improvement is only 0.0133% and its interval crosses zero. Thus FT information improves conditional percentage prediction here, but its incremental all-player makes gain is unresolved. Low-volume slices are diagnostic and do not override that result.

This volume-only control uses the new 2019-selected base and 180-day volume history, so it is not identical to the old v1-based model currently displayed. No claim of beating the existing board or EPM is warranted from these numbers. No 2026 observations were scored or used in this run; 2027 remains protected. Historical stress seasons have already been explored in related work and are not fresh confirmation.

## Deliverables and limits

- `development_gate.json` records selection before stress scoring.
- `report.json` contains objectives, slices, calibration-related receipts, comparisons and input hashes.
- `predictions.parquet` preserves every model's pregame predictions and features.
- `selected_dated_predictions.parquet` exports the chosen pregame 3P%, 3PA/100 and exact product 3PM/100, with season training cutoff. Historical 2021–2022 rows remain selection evidence, not independently held-out validation.
- `calibration.csv` reports prediction buckets and observed shooting rates.
- Three regression tests verify prediction timing, zero-attempt treatment, constant-anchor equivalence, and inclusion of zero-attempt games in joint loss.

These are game-conditional, context-free forecasts of observed shotmaking, not identified neutral talent under a different shot diet. No calibrated fused uncertainty or current-2026 state export is claimed. The initial prior strengths, decay choices and feature family are fixed modeling conventions rather than exhaustive optimization. The legitimate result is a useful simple cross-skill prior, with no admitted more complex fusion yet.
