# FT_PREMIUM — Design Handoff

**Date:** 2026-02-17
**Status:** Complete. All three components sum to TS per possession. WLS coefficients all ~1.0. `FT_PREMIUM` also runs in the standard pipeline and is copied into `weighted_factors` as raw `oFT` / `dFT`.

---

## What FT_PREMIUM Is Supposed to Measure

The *premium* a player generates by drawing free throws (and shooting them above a neutral baseline), holding all other aspects of scoring constant. It is one of three components of a dTS decomposition, and it can also be computed historically without ShotQuality:

| Component | FGA value | FT value | Captures |
|-----------|-----------|----------|----------|
| **SQ_POSS** | `initial_ev` | `0.5` | Shot quality forced/allowed |
| **FT_PREMIUM** | `0.0` | `ExpFT - 0.5` | FT drawing + FT shooting premium |
| **CONTEST** | `actual_pts - initial_ev` | `0.0` | Execution above/below shot quality |

**Per-possession verification:**
- FGA: SQ(`initial_ev`) + FT(`0`) + CONTEST(`actual-initial_ev`) = `actual_pts` = TS
- FT:  SQ(`0.5`) + FT(`ExpFT-0.5`) + CONTEST(`0`) = `ExpFT` = TS

All three use the **same TS denominator** (FGA + FT possessions, turnovers excluded) for consistent ridge shrinkage across metrics.

---

## Design Iterations (the drama)

| Version | FGA value | FT value | Problem |
|---------|-----------|----------|---------|
| v1 | `1.0` | `ExpFT` | FGA possessions dominate; FT drawing players don't stand out |
| v2 | `0.5` | `ExpFT` | User caught — "no shoot the FGA should be 1.0 not 0.5" |
| v3 | `1.0` | `ExpFT - 0.5` | Same problem as v1; Shai only +0.31 off |
| **v4 (current)** | `0.0` | `ExpFT - 0.5` | Shai +1.18 ✓, Harden +1.40 ✓, Giannis -0.10 ✓ |

**Why FGA=0 works:** FGA possessions contribute nothing to the FT_PREMIUM numerator, so the only way to generate positive off_ft RAPM is to have your lineup draw more FTs than average. FT possessions contribute `ExpFT - 0.5`, the excess over a neutral 0.5 baseline. A 90% FT shooter generates +0.40 per FT drawn; the league-average 75% shooter generates +0.25.

**Neutral baseline rationale:** The TS denominator treats FGA and FT possessions equally (1 possession each). A neutral FT is worth `0.5` in the same units as SQ (league avg ~1.0 per FGA). So `ExpFT - 0.5` is the true premium above a neutral possession.

---

## WLS Regression Results (Resolved)

After fixing SQ_POSS (FT=0.5) and CONTEST (FT=0.0), the three components sum to TS per possession. WLS regression confirms:

```
OFFENSIVE (R²=0.970, N=476):
  off_sq      +0.97  ***
  off_ft      +0.98  ***   ✓ positive
  off_contest +0.98  ***

DEFENSIVE (R²=0.960, N=476):
  def_sq      +0.96  ***
  def_ft      +1.02  ***   ✓ positive
  def_contest +0.99  ***
```

All coefficients ~1.0 as expected from the per-possession identity. The previous negative FT coefficient was caused by CONTEST using FT=1.0 (bug) and SQ_POSS using FT=1.0 (design error), which broke the additive decomposition.

---

## Sanity Check: Fixed-Coeff Decomposition (FT = TS - SQ - CONTEST)

```
                          oTS   oSQ   oFT  oCONT   dTS   dSQ   dFT  dCONT
Giannis Antetokounmpo    4.93  3.47  -0.46  1.92  -1.15 -0.33  0.16 -0.98   ✓ neg oFT (poor FT%)
Shai Gilgeous-Alexander  3.60 -0.07   0.72  2.95  -2.04 -0.74  0.75 -2.05   ✓ pos oFT (elite drawer)
James Harden             2.67  1.41   1.56 -0.30   1.45  0.06  0.05  1.34   ✓ highest oFT
Nikola Jokic             7.14  2.21  -1.36  4.29   0.27 -0.41 -0.07  0.75   ✓ neg oFT (rarely draws FTs)
```

---

## Standard Pipeline Integration

- `02_process_rapm.py` now builds `FT_PREMIUM` in the normal processing pass.
- `03_run_rapm_analysis.py` now runs `rapm.py FT_PREMIUM ...` as part of the standard analysis pass.
- `weighted_factors_*.csv` now includes raw `oFT` / `dFT` columns copied directly from the `FT_PREMIUM` RAPM results.
- In `weighted_factors`, `dFT` is sign-flipped so positive still means good defense, matching the rest of the defensive display columns.
- These `oFT` / `dFT` columns are informational TS subfactor outputs. They are **not** fed back into the six-factor weighting regressions, and `RESID` remains based on the six-factor core.

## Files

| File | Description |
|------|-------------|
| `scripts/process_rapm_blocks/process_ft_premium.py` | Processor (current: FGA=0, FT=ExpFT-0.5) |
| `processed/FT_PREMIUM*.parquet` | Processed possession data (historical support from 2014 onward if raw data exists) |
| `results/ftpremium_24_26_all_results.csv` | RAPM results (ALL, 24-26) |
| `results/td/ftpremium_24_26_all_td700_results.csv` | RAPM results (time-decay td700) |
| `results/ts_decomp_24_26_all/` | All component results + regression outputs |
| `scripts/ts_decomp_regression.py` | WLS factor regression script |

---

## How to Re-run From Scratch

```bash
# 1. Reprocess historical FT_PREMIUM parquets (from scripts/)
python reprocess_metric.py FT_PREMIUM 14 26

# 2. Run RAPM ridge regression
python rapm.py FT_PREMIUM 14 26 ALL
python rapm.py FT_PREMIUM 14 26 ALL --timedecay --half-life 700

# 3. Optional: for 24-26 windows, copy results to ts_decomp folder and re-run factor regression
cp ../results/ftpremium_24_26_all_results.csv ../results/ts_decomp_24_26_all/
python ts_decomp_regression.py 24 26 ALL

# 4. Standard weighted factors run (now includes oFT/dFT automatically)
python 03_run_rapm_analysis.py 14 26 ALL --rubberband
```
