# RAPM Hyperparameter Optimization: Methodology Review Request

## Executive Summary

We are building a Regularized Adjusted Plus-Minus (RAPM) system for NBA player evaluation and seeking expert feedback on our hyperparameter optimization methodology. This document describes our approach, the challenges we've encountered, and specific questions where we'd like a second opinion.

---

## 1. Background: What is RAPM?

### 1.1 The Core Problem

We want to estimate each NBA player's impact on their team's scoring margin when they're on the court. The challenge: players don't play in isolation. When Stephen Curry scores, his four teammates are also on the court. How do we disentangle individual contributions?

### 1.2 The RAPM Model

For each possession `i`, we observe:
- `y_i` = points scored by offense - points scored by defense (typically -3 to +3)
- 5 offensive players (O1-O5) and 5 defensive players (D1-D5)

We solve a ridge regression:

```
y_i = β_O1 + β_O2 + β_O3 + β_O4 + β_O5 + β_D1 + β_D2 + β_D3 + β_D4 + β_D5 + ε_i

Minimize: Σ(y_i - ŷ_i)² + λ_off * Σβ_off² + λ_def * Σβ_def²
```

Where:
- `β_j` = player j's per-possession impact (positive = good for their team)
- `λ_off`, `λ_def` = regularization parameters for offense and defense
- Offense coefficients represent "points added when player is on offense"
- Defense coefficients represent "points prevented when player is on defense"

### 1.3 Why Regularization Matters

Without regularization, RAPM overfits to noise. Players with few possessions get extreme values. The regularization strength (λ) controls how much we shrink coefficients toward zero:
- Too low: Overfitting, unstable estimates
- Too high: Underfitting, everyone looks average
- Just right: Balances signal and noise

---

## 2. Our Hyperparameter Optimization Goal

We want to find the optimal regularization parameters through **forward-looking cross-validation**. The key principle: **test on future data that the model has never seen**.

### 2.1 Why Forward-Looking (Not K-Fold)?

Standard k-fold cross-validation would randomly split data across time. This causes data leakage:
- Training data from 2024 would help predict 2022 games
- This inflates apparent accuracy

Forward-looking validation mirrors real-world use:
- Train on years 1-3
- Evaluate predictions on year 4
- No information flows backward in time

### 2.2 Current Fold Structure

| Fold | Training Years | Evaluation Years |
|------|---------------|------------------|
| 1 | 2014-15, 2015-16, 2016-17 | 2017-18, 2018-19 |
| 2 | 2015-16, 2016-17, 2017-18 | 2018-19, 2019-20 |
| 3 | 2016-17, 2017-18, 2018-19 | 2019-20, 2020-21 |
| 4 | 2017-18, 2018-19, 2019-20 | 2020-21, 2021-22 |
| 5 | 2018-19, 2019-20, 2020-21 | 2021-22, 2022-23 |
| 6 | 2019-20, 2020-21, 2021-22 | 2022-23, 2023-24 |
| 7 | 2020-21, 2021-22, 2022-23 | 2023-24, 2024-25 |
| 8 | 2021-22, 2022-23, 2023-24 | 2024-25, 2025-26 |

- 3-year training window (rolling)
- 2-year evaluation window
- 8 folds total

---

## 3. Variants Being Tested

### 3.1 Baseline RAPM
Standard ridge regression with:
- `λ_off` = 3000 (fixed, not tuned)
- `λ_def` = tuned over grid [1500, 2500, 3000, 4500, 6000, 8000, 10000]

### 3.2 Time-Decay RAPM
Weights recent games more heavily:
```
weight_i = decay_base ^ (days_since_game)
```
- Decay grid: [0.997, 0.999, 0.9995]
- 0.997 ≈ 230-day half-life
- 0.9995 ≈ 1386-day half-life (3.8 years)

### 3.3 Possession-Weighted Ridge (Novel Approach)
Different regularization per player based on sample size:
```
λ_j = λ_base * multiplier_j
multiplier_j = clamp((N_ref / (N_j + N0))^p, m_min, m_max)
```

Where:
- `N_j` = possessions for player j in training data
- `N_ref` = 2000 (reference possession count)
- `N0` = 200 (smoothing constant)
- `p` = power parameter (tuned: 0, 0.25, 0.5, 0.75, 1.0)
- `m_min` = 0.7, `m_max` = 2.0 to 4.0 (tuned)

**Intuition**: High-possession players (stars) need less regularization because we have more data about them. Low-possession players (bench players, injured) need more regularization.

### 3.4 Combined: Time-Decay + Possession-Weighted
Both mechanisms combined.

---

## 4. Evaluation Methodology

### 4.1 Evaluation Metric: Stint-Level RMSE

For each possession in the evaluation period:
```
prediction_i = Σ(β_offense_players) + Σ(β_defense_players)
RMSE = sqrt(mean((y_i - prediction_i)²))
```

We predict at the possession level and compute RMSE.

### 4.2 The Coverage Problem (CRITICAL ISSUE)

**Current implementation**: If ANY of the 10 on-court players was not in the training data, we skip that possession in evaluation.

**Result**: Only ~25% of evaluation possessions are included.

**Why this happens**:
- Rookies weren't in the league during training years
- G-League callups, undrafted players, etc.
- International players arriving mid-season

**Concern**: We're only testing on veteran-heavy lineups, which may:
- Bias toward easier-to-predict scenarios
- Miss the hard cases (new player integration)
- Overstate model accuracy

**Proposed fix**: Treat unknown players as having 0 impact (league average) and include ALL possessions in evaluation. This gives 100% coverage but may increase RMSE.

---

## 5. Preliminary Results (2-Fold Quick Test)

With the current 25% coverage approach:

| Variant | Avg RMSE | Best λ_def | Other Params |
|---------|----------|-----------|--------------|
| possridge | 1.6014 | 10000 | p=1.0, m_max=4.0 |
| timedecay_possridge | 1.6016 | 10000 | decay=0.9995, p=1.0, m_max=3.0 |
| baseline | 1.6021 | 10000 | - |
| timedecay | 1.6024 | 10000 | decay=0.9995 |

**Key observations**:
1. All variants prefer high defense regularization (λ_def = 10000)
2. Possession-weighted ridge slightly outperforms baseline
3. Time decay alone doesn't help (or slightly hurts)
4. Full possession weighting (p=1.0) is optimal

---

## 6. Open Questions for Review

### Question 1: Is stint-level RMSE the right metric?

Each possession outcome is inherently noisy (scores are 0, 2, or 3 points typically). Should we instead:
- Aggregate to game-level predictions?
- Use a different metric like log-loss or calibration?
- Weight by possession importance (playoffs, clutch time)?

### Question 2: Is 3-year training window optimal?

We chose 3 years somewhat arbitrarily. Tradeoffs:
- More years → more data, but player skills change over time
- Fewer years → more current, but less sample size

Is there research on optimal window sizes for player evaluation?

### Question 3: Is the possession-weighted ridge approach sound?

The formula `λ_j = λ_base * (N_ref / (N_j + N0))^p` is novel. Is this a principled way to incorporate sample size into regularization? Are there established methods we should consider instead?

### Question 4: Should we tune offense lambda too?

We fixed `λ_off = 3000` based on the intuition that offensive signal is stronger and needs less regularization. But we haven't validated this. Should we:
- Tune both (larger grid search)
- Use a fixed ratio (e.g., λ_off = 0.5 * λ_def)
- Keep it fixed but validate the choice

### Question 5: Is our time decay grid too narrow?

We tested [0.997, 0.999, 0.9995], which corresponds to half-lives of ~0.6 to ~3.8 years. Should we test:
- No decay at all (already tested via baseline)
- More aggressive decay (0.99, 0.995)?
- Season-level decay instead of daily?

### Question 6: How should we handle the coverage problem?

Options:
a) Keep current approach (25% coverage, cleaner but biased)
b) Treat unknowns as 0 (100% coverage, harder test)
c) Report both metrics
d) Use priors for rookies based on draft position/college stats

### Question 7: Are there alternative hyperparameter search methods?

We're using grid search, which is exhaustive but slow. Should we consider:
- Bayesian optimization
- Random search
- More sophisticated methods (Hyperband, etc.)

### Question 8: Should year-to-year stability be a criterion?

Currently we only optimize for predictive RMSE. Should we also consider:
- Year-to-year correlation of player ratings
- Ranking stability (do the same players stay at top/bottom?)
- A combined objective?

---

## 7. Technical Details

### 7.1 Data Volume

| Training Window | Possessions | Unique Players |
|-----------------|-------------|----------------|
| 3 years | ~750,000 | ~800-900 |

| Evaluation Window | Possessions | Unique Players |
|-------------------|-------------|----------------|
| 2 years | ~500,000 | ~600-700 |

### 7.2 Design Matrix

- Sparse binary matrix (10 non-zeros per row)
- Dimensions: ~750,000 rows × ~1600 columns (800 players × 2 for off/def)
- Solved using conjugate gradient for per-coefficient penalties

### 7.3 Computational Cost

- Baseline (7 configs): ~5 min per fold
- Time decay (21 configs): ~15 min per fold
- Possridge (105 configs): ~1 hour per fold
- Combined (315 configs): ~3 hours per fold
- Full 8-fold backtest: ~6-8 hours total

---

## 8. Request for Feedback

We would appreciate expert feedback on:

1. **Methodological soundness**: Are there fundamental flaws in our approach?
2. **Missing considerations**: What important factors are we overlooking?
3. **Best practices**: What does the sports analytics literature recommend?
4. **Alternative approaches**: Are there better methods we should consider?
5. **Specific answers**: To any of the 8 questions above

---

## 9. References

For context, this work builds on:
- Rosenbaum (2004): Adjusted Plus-Minus in Basketball
- Sill (2010): Improved NBA Adjusted +/- Using Regularization and Out-of-Sample Testing
- Engelmann (2017): Possession-Based Player Performance Analysis
- Myers (2011): Points Created: A New Way to Look at Basketball
- NBA API play-by-play data (2014-2025)

---

## Appendix: Code Structure

```
nba_pipeline/scripts/rapm_variants/
├── base_rapm.py              # Data loading, design matrix, constants
├── possession_ridge.py       # Custom solver for per-coefficient penalties
├── backtest.py               # Forward-looking cross-validation
├── rapm_baseline.py          # Standard ridge
├── rapm_timedecay.py         # Time-weighted ridge
├── rapm_possridge.py         # Possession-weighted penalties
├── rapm_timedecay_possridge.py # Combined approach
└── run_all_variants.py       # Orchestrator
```

Results saved to: `nba_pipeline/results/rapm_variants/`
