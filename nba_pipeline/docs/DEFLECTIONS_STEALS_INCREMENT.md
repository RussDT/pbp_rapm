# Estimated deflections as a steals predictor

Protocol, September 14, 2026: compare a ridge residual correction to pregame v1 steals with the same correction plus pregame next-OU deflections. Ridge alpha10, standardized predictors, defensive-possession weights. The steals-only calibration control prevents crediting deflections for generic scale/intercept repairs. Both models predict next-game steals/100 on exactly the same rows, joined by player ID/game/date; use only the stored pregame predictions, never current deflections or postgame states. Require an earlier measured deflection observation. This observed-tracking sample is a coverage-conditioned test, not an all-player deployment simulation.

Fit each season using 2020 through the preceding season. Select by paired game-bootstrap one-SE MSE improvement on 2021–2022; report 2023–2025 as already-spent historical stress, with per-season checks and comparison against original steals predictions. No 2026 data scored or used in regression fits; 2027 untouched. Existing base-model hyperparameters have broader development selection histories, so this is retrospective research, not a fully as-of-trained base-model replay. No automatic board change.

Runner: `nba_pipeline/scripts/test_deflections_steals_increment.py`. Ignored receipts: `results/estimated_skills/deflections_steals_increment/`.

## Results

Development passes:0.2925% MSE improvement [0.1736,0.4122] on49,932 player-games. Spent2023–2025 improves0.4103% [0.3220,0.4983] versus steals-only calibration and0.4037% [0.3126,0.4895] versus original v1, on78,291 player-games /3,931 games. Gains by season:+0.1124%,+0.4247%,+0.6535%. Deflection coefficients are positive in every expanding fit (0.23–0.28 additional steals/100 per deflection/100, conditional on the steals estimate). This coefficient is a predictive association, not a conversion probability or causal credit.

There are177,586 matched rows with earlier deflection observations across2019–2025, versus188,344 available steals rows;10,758 rows are outside coverage or lack prior measured deflections. Join uniqueness, finite evaluated predictions, prior observation support and exclusion of2026 targets verified. Input hashes retained. Candidate merits a fully chronological base-model replay and export decision; displayed steals model remains unchanged.

## Same-day EPM comparison

`compare_enhanced_steals_epm.py` fits the fixed deflection residual mapper on2020–2025 and applies it to2026 pregame forecasts on cached EPM snapshot dates. All three models share1,549 player-games across76 games /10 dates with games, requiring previous measured deflections. Raw defensive-possession-weighted steals/100 MSE, no mean adjustment, paired game bootstrap. Enhanced vs EPM improves0.2854% [−0.5788,+1.1420]; original vs EPM−0.2886% [−0.6203,+0.0504]; enhanced vs original+0.5724% [−0.2237,+1.3632]. All comparisons inconclusive. RMSE: enhanced1.9183, EPM1.9210, original1.9238. This is limited-date, already-spent2026 evidence, not proof of EPM superiority/inferiority or a daily-coverage benchmark. No table change. Private receipts in estimated_skills/enhanced_steals_epm.
