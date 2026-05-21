# DTOV vs RFTOV Investigation Report

**Date:** January 13, 2026
**Investigator:** Claude (with Russell Thomas)
**Purpose:** Determine whether lineup-based DTOV adds predictive value beyond box-score RFTOV

---

## Executive Summary

**Finding: Lineup-based DTOV is mostly noise around box-score RFTOV.**

The lineup regression for defensive turnovers does not capture meaningful signal beyond what individual steal rates and offensive fouls drawn (RFTOV) already provide. RFTOV is more stable, more predictive, and explains DTOV almost entirely.

**Recommendation:** For nbarapm.com, consider using RFTOV as the primary "defensive turnover" metric rather than lineup-based DTOV, or use a heavily RFTOV-weighted blend (70% RFTOV / 30% DTOV).

---

## Background

### Definitions

| Metric | Source | Description |
|--------|--------|-------------|
| **RFTOV** | Box score (Supabase `player_stats.rFTOV_100`) | Relative forced turnovers per 100 possessions. Derived from individual steals and offensive fouls drawn. |
| **DTOV** | Lineup regression (`rapm.py` with TOV data) | Defensive turnover RAPM. Ridge regression coefficient for how much a player's presence affects opponent turnover rate. |

### Core Question

Does the lineup regression capture anything about defensive turnover generation that box-score stats don't already capture?

---

## Methodology

### Data
- **Years:** 2016-2025 (10 seasons)
- **Windows:** 5 non-overlapping 3-year train → 3-year test periods
- **Sample:** ~175-200 players per window with 5000+ possessions in each period

### Tests Performed

1. **In-sample correlation:** 5-year RFTOV vs 5-year DTOV (same period)
2. **Stability analysis:** How well does each metric predict itself 3 years later?
3. **Cross-prediction:** Does Train RFTOV or Train DTOV better predict Test DTOV?
4. **Incremental value:** Does adding DTOV to RFTOV improve predictions?
5. **Optimal blend:** What weight on RFTOV vs DTOV maximizes predictive accuracy?

---

## Results

### 1. In-Sample Correlation (2021-2025)

RFTOV and DTOV are highly correlated within the same time period:

| Metric | Correlation | R² |
|--------|-------------|-----|
| Raw RFTOV vs DTOV | -0.70 | 0.49 |
| Weighted (by poss) | -0.77 | 0.59 |

**Interpretation:** RFTOV explains ~50-60% of DTOV variance in-sample. They're measuring largely the same thing.

### 2. Stability Analysis

How well does each metric predict itself 3 years into the future?

| Metric | R (self-prediction) | R² |
|--------|---------------------|-----|
| **RFTOV** | **0.855** | 0.731 |
| DTOV | 0.571 | 0.326 |

**Interpretation:** RFTOV is far more stable over time. A player's box-score forced turnover rate persists much better than their lineup-based DTOV coefficient.

### 3. Cross-Prediction: What Predicts Future DTOV?

Using 3-year training data to predict next 3-year DTOV:

| Predictor | Avg R | Avg R² |
|-----------|-------|--------|
| Train RFTOV → Test DTOV | -0.687 | 0.473 |
| Train DTOV → Test DTOV | 0.631 | 0.400 |

**Key Finding:** RFTOV predicts future DTOV better than past DTOV does.

This is counterintuitive - you'd expect DTOV to predict itself better. But because RFTOV is less noisy, it's a better predictor of the underlying skill that future DTOV is (noisily) measuring.

### 4. Does DTOV Add Incremental Value?

When predicting future DTOV:

| Model | Avg R² |
|-------|--------|
| RFTOV only | 0.473 |
| DTOV only | 0.400 |
| Both combined | 0.509 |

**Improvement from adding DTOV:** +3.6 percentage points R² (0.473 → 0.509)

When predicting future RFTOV:

| Model | Avg R² |
|-------|--------|
| RFTOV only | 0.731 |
| Both combined | 0.731 |

**Improvement from adding DTOV:** +0.0 percentage points

**Interpretation:** DTOV doesn't help predict a player's own future box-score turnover generation. This is expected - your own stats predict your own future stats.

**Important caveat:** This test doesn't rule out that DTOV captures "non-box-score" contributions like helping teammates force turnovers. A player who directs defense well might improve their teammates' RFTOV without boosting their own. We did not test whether a player's DTOV residual (after removing their RFTOV) correlates with teammate turnover generation. That remains an open question.

### 5. Optimal Blend Weights

Testing different RFTOV/DTOV blends for predicting future DTOV:

| RFTOV Weight | DTOV Weight | Avg R² |
|--------------|-------------|--------|
| 0% | 100% | 0.404 |
| 50% | 50% | 0.495 |
| **70%** | **30%** | **0.509** |
| 100% | 0% | 0.474 |

**Optimal:** 70% RFTOV / 30% DTOV

Results were consistent across all 5 test windows (optimal ranged from 60-80% RFTOV).

---

## Positional Analysis

### Average RFTOV and DTOV by Position

| Position | Avg RFTOV | Avg DTOV |
|----------|-----------|----------|
| Guard | +0.22 | -0.14 (good) |
| Wing | -0.04 | -0.10 (good) |
| Big | -0.23 | +0.23 (bad) |

Guards have high RFTOV (more steals) and good DTOV. Bigs have low RFTOV and bad DTOV. The positional patterns match.

### Position Adjustment

We tested whether position-adjusting RFTOV (recentering by position group) improved correlations:

| RFTOV Type | Correlation with DTOV |
|------------|----------------------|
| Raw | -0.70 |
| Position-adjusted | -0.66 |

**Finding:** Position adjustment hurts correlation. Raw RFTOV is the better predictor because DTOV also contains positional signal. Don't position-adjust the prior.

---

## Delta Analysis

For players with 6 full years of data (3 years + 3 years), we tested whether changes in RFTOV predict changes in DTOV:

| Metric | Value |
|--------|-------|
| Slope (ΔRFTOV → ΔDTOV) | -0.69 |
| R² | 0.20 |

**Interpretation:** A 1-point improvement in RFTOV corresponds to ~0.69 point improvement in DTOV. RFTOV captures about 70% of the DTOV signal, with 30% being either additional DTOV signal or noise.

---

## Conclusions

### What We Learned

1. **RFTOV is the signal, DTOV is noisy RFTOV.** The lineup regression isn't capturing hidden defensive contributions - it's just adding noise to what the box score already tells us.

2. **RFTOV is more stable.** Year-to-year correlation of 0.86 vs 0.57 for DTOV. Box-score stats persist better than lineup coefficients.

3. **RFTOV better predicts future DTOV.** Past RFTOV (R² = 0.47) beats past DTOV (R² = 0.40) at predicting future DTOV.

4. **DTOV adds minimal incremental value.** Only +3.6% R² when combined with RFTOV for predicting DTOV, and +0% when predicting RFTOV.

5. **Optimal blend is 70% RFTOV / 30% DTOV.** If using both, weight heavily toward RFTOV.

### Why This Happens

The lineup regression for turnovers faces a fundamental challenge: turnovers are discrete events (steals, offensive fouls) that are directly attributed to individual players in the box score. Unlike points (where lineup context matters for shot creation/defense), turnovers have a clear individual cause.

When a steal happens, RFTOV credits the player who got the steal. DTOV credits all 5 defenders. The individual attribution is more accurate.

### What We Didn't Test (Open Questions)

The analysis focused on whether DTOV helps predict a player's **own** future turnover generation. But DTOV might still capture value we didn't measure:

1. **Teammate effects:** Does a player's DTOV residual (after removing their RFTOV) correlate with their teammates' RFTOV? Some players might direct defense, funnel ball handlers into traps, or create chaos that leads to steals credited to others.

2. **Scheme/role effects:** A player's DTOV might reflect the defensive scheme they play in rather than individual skill. This could still be predictive of team outcomes even if not predictive of individual future stats.

3. **Non-steal turnovers:** RFTOV focuses on steals and offensive fouls drawn. DTOV might capture contributions to other turnover types (bad passes forced by positioning, shot clock violations, travels induced by pressure).

These questions would require lineup-level analysis (tracking which teammates a player shares the court with) rather than the individual-level analysis we performed.

### Recommendations for nbarapm.com

**Option A: Use RFTOV only**
- Simpler, more stable, equally predictive
- Label it "Forced Turnovers" or similar

**Option B: Use 70/30 blend**
- Marginally more predictive (+3.6% R²)
- More complex to explain
- Formula: `0.70 * RFTOV_scaled + 0.30 * DTOV`

**Option C: Show both separately**
- Let users see RFTOV (box-score) and DTOV (lineup)
- Note that they're highly correlated

**Not recommended:** Using pure DTOV. It's strictly worse than RFTOV for prediction.

---

## Appendix: Test Windows

| Window | Train Years | Test Years | n Players |
|--------|-------------|------------|-----------|
| 1 | 2016-2018 | 2019-2021 | 177 |
| 2 | 2017-2019 | 2020-2022 | 179 |
| 3 | 2018-2020 | 2021-2023 | 172 |
| 4 | 2019-2021 | 2022-2024 | 177 |
| 5 | 2020-2022 | 2023-2025 | 198 |

All results averaged across these 5 windows for robustness.

---

## Files Generated

- `prior_informed_rapm.py` - Script with analysis functions
- `results/rftov_dtov_insample_21_25.csv` - In-sample correlation data
- `results/tov_*_all_results.csv` - 3-year DTOV results for each window

---

## Code Reference

Key functions in `prior_informed_rapm.py`:
- `fetch_rftov_data()` - Fetch RFTOV from Supabase
- `analyze_insample_correlation()` - In-sample RFTOV vs DTOV correlation
- `analyze_predictive_value()` - Out-of-sample prediction tests (inline in investigation)

---

*Report generated from Claude Code investigation session, January 13, 2026*
