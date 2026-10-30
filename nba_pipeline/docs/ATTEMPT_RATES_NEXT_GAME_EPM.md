# Next-game attempt rates versus EPM

Run `nba_pipeline/scripts/compare_attempt_rates_next_game_epm.py`. Uses saved homegrown v1 posteriors strictly before each of the 12 existing EPM requested dates from October15,2025 through April1,2026. Provider date-D snapshots exclude games on D; requested and provider-effective date retained, effective date cannot exceed requested date. Target is the player's first played game on/after the snapshot within seven days, with valid offensive possession exposure. Primary subset: played on the snapshot date, avoiding forecast-age differences. Secondary: the full next-appearance sample with both estimates frozen at the same snapshot. This is not an every-day next-game benchmark, nor an availability forecast. Source and fitting data have already been explored; 2026 is spent historical evidence. No model tuning or table changes.

Each target is attempts / offensive possessions *100. Compare raw predictions without mean adjustment, weight squared errors by offensive possessions across the sample, and bootstrap games jointly for both models (2,000 paired draws). Exclude ambiguous player-date source duplicates; retain only the latest snapshot for any repeated player-game-skill outcome. Join exact player IDs and require both predictions. These are conditional-on-playing rates, not total-attempt forecasts. PBP/provider zone definitions can differ. No 2027 data accessed.

| Rate | Same-day error reduction, ours vs EPM [95% CI] | Next appearance within7 days |
|---|---:|---:|
| Rim FGA/100 | +0.69% [−0.98,+2.26] | +1.53% [+0.49,+2.51] |
| Mid FGA/100 | −0.14% [−2.45,+2.21] | +0.34% [−1.15,+1.78] |
| Three FGA/100 | +0.08% [−0.59,+0.72] | +0.31% [−0.27,+0.88] |

Same-day: 1,655 player-games across76 games per skill. Secondary:4,693 player-games across521 games per skill. All primary comparisons inconclusive. Secondary supports a modest rim-volume advantage only; do not substitute its1.53% for a same-day win or the older remaining-season9.7% comparison. Single-season and limited dates restrict precision. Intervals are pointwise, not multiplicity-adjusted.

Ignored receipts: `results/estimated_skills/epm_next_game_attempts/report.json` and `private_comparison_rows.parquet`. Report includes hashes, coverage, game counts, raw calibration biases and paired intervals. Provider values remain private local artifacts.
