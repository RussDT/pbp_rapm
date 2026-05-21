# TOV Decomposition: Bad Pass TOV + Scoring TOV

## Overview
TOV RAPM is decomposed into two sub-components that exactly sum to the parent:
- **BADPASS_TOV**: Turnovers from bad passes (playmaking errors)
- **SCORING_TOV**: All other turnovers (lost ball, offensive foul, traveling, step out of bounds)

**Identity**: `Is_Turnover = Is_BadPass_TOV + Is_Scoring_TOV` on every row, by construction.

## Classification
Bad Pass TOV is identified by checking if the play-by-play description contains "Bad Pass" (case-insensitive). Everything else is Scoring TOV.

Typical split: ~47% bad pass, ~53% scoring across all seasons.

## Implementation

### Single Parquet Approach
Instead of separate parquet files, `Is_BadPass_TOV` is stored as an extra column in the existing `TOV*.parquet` files alongside `Is_Turnover`. The `rapm.py` script uses prefix aliasing:

```python
PREFIX_FILE_MAP = {
    'BADPASS_TOV': 'TOV',
    'SCORING_TOV': 'TOV',
}
```

When you run `python rapm.py BADPASS_TOV 26 26 RS`, it loads `TOV26.parquet` and uses `Is_BadPass_TOV` as the metric column. For `SCORING_TOV`, it derives `Is_Turnover - Is_BadPass_TOV` at load time.

### Files Modified
- `process_rapm_blocks/process_tov.py`: Adds `Is_BadPass_TOV` column, passes `['Is_Turnover', 'Is_BadPass_TOV']` to `_finalize_df`
- `rapm.py`: `PREFIX_FILE_MAP` for aliasing, `prefix` parameter threaded through `detect_file_type_and_prepare` → `_load_single_file` → `_load_files_parallel/sequential` → `run_simplified_rapm`, argparse choices updated
- `03_run_rapm_analysis.py`: Runs BADPASS_TOV and SCORING_TOV in parallel, performs second-stage WLS decomposition, adds `oTOV_bp`, `oTOV_sc`, `dTOV_bp`, `dTOV_sc` to weighted_factors

### Decomposition Regression (Second Stage)
In `03_run_rapm_analysis.py`, after computing the standard 6-factor weighted factors:
1. Regress `off_tov` on `off_badpass_tov + off_scoring_tov` (WLS, possession-weighted)
2. Betas are ~1.0 each, R² ≈ 0.9999
3. Chain the coefficients: `oTOV_bp = off_badpass_tov × bp_beta × coef_off_tov`

This ensures `oTOV_bp + oTOV_sc ≈ oTOV` (within rounding).

## Usage
```bash
# Individual metric runs
python rapm.py BADPASS_TOV 21 26 ALL
python rapm.py SCORING_TOV 21 26 ALL

# Full pipeline (automatically included)
python 03_run_rapm_analysis.py 21 26 ALL
```

## Reprocessing TOV Parquets
To regenerate TOV parquets with the new `Is_BadPass_TOV` column:
```bash
python reprocess_metric.py TOV 14 25    # All historical seasons
```

## Interpretation
- **oTOV_bp** (offensive bad pass TOV): Playmaking error cost. High values = careful passer. Point guards and playmakers vary most here.
- **oTOV_sc** (offensive scoring TOV): Ball-handling/scoring error cost. High values = secure ball-handler. Wings and drivers vary most here.
- **dTOV_bp** (defensive bad pass TOV): Ability to force bad passes. Active hands, anticipation, help defense.
- **dTOV_sc** (defensive scoring TOV): Ability to force lost balls, charges, travels. Physical defense, positioning.

## Findings (21-26 ALL)
- TOV decomposition betas: ~1.0 each, R² = 0.9999
- In alt regression (replacing TOV with BP+SC in net RAPM prediction):
  - Scoring TOV slightly more valuable on offense (1.32 vs 1.24)
  - Bad Pass TOV slightly more valuable on defense (1.24 vs 1.22)
  - Differences are modest — both highly significant


# Second Chance RAPM

## Overview
SECOND_CHANCE measures second-chance points per possession — points scored after an offensive rebound within the same possession. Every possession gets a row; non-second-chance possessions get 0.

The processed `SECOND_CHANCE` and `SECOND_CHANCE_CLEAN` parquets should use the
same sign convention as TS and pure RAPM:
- `Off_Diff = second chance points/value`
- `Def_Diff = -Off_Diff`

If both are written with the same sign, the downstream defensive SC outputs stop
being comparable to the parent RAPM surface and FC/SC additivity checks break.

## Integration in weighted_factors
Added to `03_run_rapm_analysis.py` pipeline. Creates derived columns:
- **oFT / dFT**: Raw FT_PREMIUM RAPM outputs copied through from the standard FT run. `dFT` is sign-flipped in `weighted_factors` so positive still means good defense. Informational TS subfactor columns only; not part of the six-factor weighting regression.
- **o_sc**: Raw offensive second-chance RAPM coefficient
- **d_sc**: Raw defensive second-chance RAPM coefficient
- **o_fc**: First-chance offense = `off_rapm - off_sc2` (everything except second-chance)
- **d_fc**: First-chance defense = `def_rapm - def_sc2`

For the alternate `weighted_factors_alt3_*` export, `oSC` / `dSC` are balancing
residual buckets, not the direct `SECOND_CHANCE_CLEAN` RAPM coefficients. That
surface is forced to add back to `off` / `def` after the public first-chance
subfactor rollup.
- **ALT_TOV decomposition**: `03_run_rapm_analysis.py` also regresses `ALT_TOV` on `ALT_BADPASS_TOV + ALT_SCORING_TOV` and publishes `oALT_TOV_bp`, `oALT_TOV_sc`, `dALT_TOV_bp`, `dALT_TOV_sc` in `weighted_factors_alt3_*`. These are informational turnover-avoidance subcolumns built on the true first-chance turnover universe; the additive first-chance identity still uses parent `oALT_TOV` / `dALT_TOV`.
- **o_pval**: Offensive possession value = `oTOV + oREB` (weighted)
- **d_pval**: Defensive possession value = `dTOV + dREB` (weighted)

## Findings
- REB alone (R²=0.9847) > SECOND_CHANCE alone (R²=0.9637) for predicting RAPM
- Both together: R²=0.9869, marginal improvement
- REB dominates (coeff ~0.64) while SC2 adds smaller signal (~0.25)
- SC2 captures "what you do after the board" — scoring conversion, not just getting the rebound


# Weighted Factors Column Reference

## Current weighted_factors CSV columns
```
player_id, player_name, Latest_Year,
oTS, oFT, oTOV, oTOV_bp, oTOV_sc, oREB,     # offensive factors (+ FT passthrough)
dTS, dFT, dTOV, dTOV_bp, dTOV_sc, dREB,     # defensive factors (+ FT passthrough)
off, def, net_rapm, RESID,                   # overall RAPM
o_sc, o_fc, d_sc, d_fc,                     # second-chance / first-chance split
o_pval, d_pval,                              # possession value aggregates
possessions, off_poss, def_poss              # sample size
```

### Factor hierarchy
- **6-factor core**: oTS, oTOV, oREB, dTS, dTOV, dREB (from WLS regression on net_rapm)
- **FT passthrough**: oFT, dFT come directly from FT_PREMIUM RAPM results and do not change the six-factor core or RESID definition
- **TOV decomposition**: oTOV_bp + oTOV_sc ≈ oTOV, dTOV_bp + dTOV_sc ≈ dTOV
- **Derived aggregates**: o_pval = oTOV + oREB, d_pval = dTOV + dREB
- **SC/FC split**: o_fc + o_sc = off (raw), d_fc + d_sc = def (raw, note sign)
- **RESID**: net_rapm - sum(6 factors), captures unexplained variance


# Utility Script: reprocess_metric.py

Generic script to reprocess any metric's parquets across multiple years with parallelization.

```bash
python reprocess_metric.py METRIC [start_year] [end_year]

# Examples
python reprocess_metric.py TOV 14 25           # TOV for all historical seasons
python reprocess_metric.py SECOND_CHANCE 14 25  # Second chance
python reprocess_metric.py TS 23 26             # TS for recent years
```

Supports: RAPM, TS, TOV, REB, RIM_FREQ, RIM_FG_PCT, MIDRANGE_FREQ, MIDRANGE_FG_PCT, TRANSITION_FREQ, TRANSITION_RIM, INITIAL_EV, SPECIAL_RAPM, SQ_POSS, FT_PREMIUM, CONTEST, SECOND_CHANCE
Also supports: ASSIST_POINTS, RIM_ASSIST, THREE_FREQ, THREE_FG_PCT
