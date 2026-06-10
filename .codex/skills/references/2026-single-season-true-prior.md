# 2026 Single-Season True-Prior Beta

This reference captures the local March 2026 single-season true-prior beta build and where its artifacts live.

Use this file when the task is about any of the following:

- current `2026` single-season true-prior RAPM
- `ss_*` beta fields
- current `2026` priors
- `weighted_factors_26_rs_rb_trueprior.csv`
- `rapms/master_results` single-season true-prior artifacts
- the distinction between the old `td700` single-season test run and the current no-decay beta

## Current Spec

As of March 15, 2026, the active local single-season beta spec is:

- season window: `26 26 RS`
- lineup likelihood: no within-season time decay
- rubberband: on
- offense alpha: `3000`
- defense alpha: `5000`
- prior scale: `1.0`
- priors: current-season box rows for active `2026` players, scored by historically fit box-to-factor relationships
- fallback prior pool: six-season time-decayed box/tracking window for non-current players only
- `oTS` prior mode: `compact_formula`

Important distinction:

- the lineup solve is no-decay
- the active current-player prior is scored from the current `2026` row
- the six-season time-decayed box window is only a fallback for non-current players in the expanded solver pool

## Active `oTS` Prior

The current backend `oTS` prior is no longer the older broad ridge feature basket. The active default in the box-prior builder is a fixed compact scoring formula.

Builder details:

- file: `/Users/russellthomas/Docs/draft_shared/rapm/russell/estimate_2026_boxscore_factors.py`
- env toggle: `OTS_PRIOR_MODE`
- current default: `compact_formula`

Formula:

`box_oTS = -1.882 + 0.604 * pts100 + 0.703 * ptdiff_pts100 - 0.619 * TSA100 + 0.666 * rim_ast100 - 0.059 * onball_pct`

Definitions:

- `pts100 = 2 * TS_pct * TSA100`
- `ptdiff_pts100 = 2 * TSA100 * playtype_diff / 100`
- `rim_ast100 = PASSING_AtRimAssists/100`
- `onball_pct = PASSING_on-ball-time%`

Interpretation:

- this is the current preferred backend scoring-value prior for `oTS`
- `zTS` remains the preferred public scoring-efficiency stat
- the compact formula is meant to keep `box_oTS` from acting like a broad all-offense halo prior

## Builder Paths

- local beta harness:
  `/Users/russellthomas/Docs/2026_NBA_PIPELINE/nba_rapm/build_single_season_beta_local.py`
- true-prior builder:
  `/Users/russellthomas/Docs/draft_shared/rapm/russell/build_true_prior_factor_rapm.py`
- solver:
  `/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/scripts/rapm.py`

## Core Naming Rules

- `box_*` = readable prior estimate
- `pure_*` = pure single-season lineup solve
- `prior_*` = true-prior single-season solve
- `shift_* = prior_* - pure_*`
- `ss_*` = local public beta mapping for the current season surface

Do not confuse:

- `weighted_factors_26_rs_rb_trueprior.csv` = current no-decay true-prior single-season weighted factors
- `weighted_factors_26_rs_rb_td700_trueprior.csv` = older td700 single-season true-prior run

## Primary Artifacts

### Readable Priors

Readable priors for the full 2026 pool:

- `/Users/russellthomas/Docs/draft_shared/rapm/russell/results/2026_single_season_true_prior_box_priors_fullpool.csv`
- mirrored to:
  `/Users/russellthomas/Docs/rapms/master_results/2026_single_season_true_prior_box_priors_fullpool.csv`

Schema:

`nba_id, player_name, Latest_Year, box_prior_source, matched_current_box, matched_box_window, current_OffPoss, current_Minutes, current_GamesPlayed, OffPoss, Minutes, off_poss, def_poss, possessions, box_oTS, box_oTOV, box_oREB, box_dTS, box_dTOV, box_dREB, box_off, box_def, box_net`

Notes:

- `box_prior_source = current_single_season` for active `2026` players, `window_fallback` for non-current fallback rows, and `imputed_fallback` for players with no matched box row in either source
- `box_off = box_oTS + box_oTOV + box_oREB`
- `box_def = box_dTS + box_dTOV + box_dREB`
- `box_net = box_off + box_def`

### Joined Current-Player Export

Current-player joined export with priors, pure solve, and final true-prior solve:

- `/Users/russellthomas/Docs/draft_shared/rapm/russell/results/2026_single_season_true_prior_factor_rapm.csv`
- mirrored to:
  `/Users/russellthomas/Docs/rapms/master_results/2026_single_season_true_prior_factor_rapm.csv`

Key columns:

- context: `nba_id, player_name, Pos2, Defensive_Role, current_OffPoss, current_Minutes`
- prior source: `box_prior_source`
- readable priors: `box_off, box_def, box_net, box_oTS, ... , box_dREB`
- pure solve: `pure_off, pure_def, pure_net, pure_oTS, ... , pure_dREB`
- final true-prior solve: `prior_off, prior_def, prior_net, prior_oTS, ... , prior_dREB`
- shifts: `shift_off, shift_def, shift_net, shift_oTS, ... , shift_dREB`

### Weighted-Factor Outputs

Current no-decay single-season weighted factors:

- canonical local file:
  `/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/master_results/weighted_factors_26_rs_rb_trueprior.csv`
- mirrored publish target:
  `/Users/russellthomas/Docs/rapms/master_results/weighted_factors_26_rs_rb_trueprior.csv`

Older td700 single-season weighted factors:

- `/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/master_results/weighted_factors_26_rs_rb_td700_trueprior.csv`
- mirrored to:
  `/Users/russellthomas/Docs/rapms/master_results/weighted_factors_26_rs_rb_td700_trueprior.csv`

Schema:

`player_id, player_name, Latest_Year, oTS, oFT, oTOV, oTOV_bp, oTOV_sc, oREB, dTS, dFT, dTOV, dTOV_bp, dTOV_sc, dREB, off, def, net_rapm, RESID, o_sc, o_fc, d_sc, d_fc, o_pval, d_pval, possessions, off_poss, def_poss`

Interpretation:

- `off` = ORAPM
- `def` = DRAPM
- `net_rapm = off + def`
- `oTOV_bp / oTOV_sc` and `dTOV_bp / dTOV_sc` are decomposition diagnostics, not public-facing beta columns

### Local Public Beta Surface

Public beta export for the current season:

- `/Users/russellthomas/Docs/2026_NBA_PIPELINE/csvs/ss_beta_2026.csv`

Public fields:

- `ss_orapm, ss_drapm, ss_rapm`
- `ss_ots, ss_otov, ss_oreb, ss_dts, ss_dtov, ss_dreb`
- `ss_off_poss, ss_def_poss, ss_reliability, ss_model_variant, ss_snapshot_date`

## Current QC Snapshot

Latest local summary:

- `/Users/russellthomas/Docs/2026_NBA_PIPELINE/nba_rapm/data/outputs/single_season_beta/single_season_beta_summary.json`

Current checks from the no-decay build:

- `row_count = 500`
- `duplicate_nba_ids = 0`
- `factor_sum_r2 = 0.9890`
- max arithmetic error for `ss_rapm - (ss_orapm + ss_drapm)` is `0.01`
- latest summary also reports:
  - `orapm_rmse_vs_lebron = 1.2752`
  - `drapm_rmse_vs_lebron = 1.1647`
  - `rapm_rmse_vs_lebron = 1.8220`

## Alpha Justification Status

Current alpha choice for the no-decay beta:

- `off_alpha = 3000`
- `def_alpha = 5000`

Why:

- `def_alpha = 5000` has direct historical support from the existing multi-year next-snapshot defense-alpha tuning.
- `off_alpha = 3000` is the current conservative anchor because the solver is prior-centered, not zero-centered, and the offensive priors are already reasonably strong.

Important caveat:

- the exact year-over-year single-season alpha harness the user wants is not yet implemented in this reference state
- the older `20_25 -> 26` test is a multi-year next-snapshot proxy, not the final single-season `Y -> Y+1` validation loop

## When Explaining This Series

Say:

- `2026 single-season true-prior beta`
- `no-decay current-season true-prior RAPM`
- `single-season true-prior beta fields`

Do not say:

- `published hybrid board`
- `pure RAPM`
- `td700 single-season` unless the task is explicitly about the older test artifact
