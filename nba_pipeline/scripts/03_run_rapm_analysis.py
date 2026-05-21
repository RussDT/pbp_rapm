#!/usr/bin/env python3
"""
Master RAPM Analysis Script

Usage (from root):
    python nba_pipeline/scripts/03_run_rapm_analysis.py 26 26 ALL   # Single year (2025-26)
    python nba_pipeline/scripts/03_run_rapm_analysis.py 23 26 ALL   # Multi-year (2022-23 to 2025-26)
    python nba_pipeline/scripts/03_run_rapm_analysis.py 26 26 RS    # Regular Season only
    python nba_pipeline/scripts/03_run_rapm_analysis.py 26 26 PS    # Playoffs only

Runs the standard RAPM suite in parallel, then performs the six-factor regression analysis.
Auxiliary metrics like FT_PREMIUM, shot frequencies, shot FG%, and assist points are also
run and copied into the final CSV when available.
Results are organized into a dedicated subfolder with proper naming convention.

Naming: Uses start_end year format (e.g., _21_26_ instead of _21_22_23_24_25_26_)

Usage:
    python 03_run_rapm_analysis.py <start_year> <end_year> <season_type>
    
Examples:
    python 03_run_rapm_analysis.py 23 25 ALL
    python 03_run_rapm_analysis.py 24 26 RS
    python 03_run_rapm_analysis.py 25 25 PS
"""

import argparse
import logging
import subprocess
import sys
import os
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# Get the pipeline root directory (parent of scripts/)
SCRIPT_DIR = Path(__file__).parent
PIPELINE_ROOT = SCRIPT_DIR.parent
PROCESSED_DIR = PIPELINE_ROOT / "processed"
RESULTS_DIR = PIPELINE_ROOT / "results"
MASTER_RESULTS_DIR = PIPELINE_ROOT / "master_results"
PROJECT_ROOT = PIPELINE_ROOT.parent
RAPMS_DIR = PROJECT_ROOT.parent / "rapms"
RAPMS_MASTER_RESULTS_DIR = RAPMS_DIR / "master_results"

AUXILIARY_FACTOR_METRICS = [
    ('RIM_FREQ', 'rim_freq', 'RIM_FREQ'),
    ('RIM_FG_PCT', 'rim_fg_pct', 'RIM_FG_PCT'),
    ('THREE_FREQ', 'three_freq', 'THREE_FREQ'),
    ('THREE_FG_PCT', 'three_fg_pct', 'THREE_FG_PCT'),
    ('MIDRANGE_FREQ', 'midrange_freq', 'MIDRANGE_FREQ'),
    ('MIDRANGE_FG_PCT', 'midrange_fg_pct', 'MIDRANGE_FG_PCT'),
    ('ASSIST_POINTS', 'assist_points', 'ASSIST_POINTS'),
]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def get_year_range_str(start_year: int, end_year: int) -> str:
    """
    Generate year range string for file naming.
    
    Uses compact format: start_end (e.g., 21_26 instead of 21_22_23_24_25_26)
    For single year: just the year (e.g., 26)
    """
    if start_year == end_year:
        return f"{start_year:02d}"
    else:
        return f"{start_year:02d}_{end_year:02d}"


def normalize_season_end_year(year_value: int) -> int:
    """Convert two-digit season end year to four digits."""
    year_value = int(year_value)
    if year_value < 100:
        return 2000 + year_value if year_value <= 50 else 1900 + year_value
    return year_value


def build_result_suffix(season_suffix: str, pure: bool = False, rubberband: bool = False,
                        season_effects: bool = False, age_curve: bool = False,
                        age_dummies: bool = False, timedecay: bool = False, half_life=None,
                        off_alpha=None, def_alpha=None) -> str:
    """Build a results filename suffix consistent with rapm.py ordering."""
    parts = [season_suffix]
    if pure:
        parts.append("pure")
    if rubberband:
        parts.append("rb")
    if season_effects:
        parts.append("se")
    if age_dummies:
        parts.append("agefe")
    if timedecay:
        td_half = int(half_life) if half_life else 700
    actual_off_alpha = 3000 if off_alpha is None else float(off_alpha)
    actual_def_alpha = 3000 if def_alpha is None else float(def_alpha)
    if actual_off_alpha != 3000 or actual_def_alpha != 3000:
        parts.append(f"a{int(actual_off_alpha)}_{int(actual_def_alpha)}")
    if age_curve:
        parts.append("age")
    if timedecay:
        parts.append(f"td{td_half}")
    return "_".join(parts)


def run_rapm_analysis(rapm_type, start_year, end_year, season_type, timedecay=False, half_life=None,
                      rubberband=False, alpha_rb=None, season_effects=False, age_curve=False,
                      age_dummies=False, cores_per_rapm=None, off_alpha=None, def_alpha=None):
    """
    Run a single RAPM analysis as a subprocess.

    Args:
        rapm_type: One of RAPM, TS, TOV, REB, FT_PREMIUM, etc.
        start_year: Starting year (2-digit)
        end_year: Ending year (2-digit)
        season_type: RS, PS, or ALL
        timedecay: If True, apply time decay weighting
        half_life: Half-life in days for time decay
        rubberband: If True, enable rubberband effect
        alpha_rb: Rubberband regularization alpha
        season_effects: If True, enable season fixed effects
        age_dummies: If True, add age-bin fixed effects to this metric solve
        off_alpha: Optional offense alpha override
        def_alpha: Optional defense alpha override

    Returns:
        tuple: (rapm_type, success, output)
    """
    # Build command - rapm.py is now in the scripts folder
    rapm_script = SCRIPT_DIR / 'rapm.py'
    cmd = ['python', str(rapm_script), rapm_type, str(start_year), str(end_year), season_type]

    # Add --pure flag for RAPM
    if rapm_type == 'RAPM':
        cmd.append('--pure')
        if age_curve:
            cmd.append('--age-curve')
    if age_dummies:
        cmd.append('--age-dummies')

    # Add time decay flags
    if timedecay:
        cmd.append('--timedecay')
        if half_life:
            cmd.extend(['--half-life', str(half_life)])

    # Add rubberband flags
    if rubberband:
        cmd.append('--rubberband')
        if alpha_rb is not None:
            cmd.extend(['--alpha-rb', str(alpha_rb)])
    if season_effects:
        cmd.append('--season-effects')
    if cores_per_rapm is not None:
        cmd.extend(['--cores', str(cores_per_rapm)])
    if off_alpha is not None:
        cmd.extend(['--off-alpha', str(off_alpha)])
    if def_alpha is not None:
        cmd.extend(['--def-alpha', str(def_alpha)])
    
    logging.info(f"Starting {rapm_type} analysis: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        elapsed = time.time() - start_time
        logging.info(f"✓ {rapm_type} completed in {elapsed:.1f}s")
        return (rapm_type, True, result.stdout)
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        logging.error(f"✗ {rapm_type} failed after {elapsed:.1f}s: {e}")
        logging.error(f"  Error output: {e.stderr}")
        return (rapm_type, False, e.stderr)


def run_regression_analysis(year_range, season_suffix, output_dir, start_year, end_year,
                            timedecay=False, half_life=None, rubberband=False,
                            season_effects=False, age_curve=False, age_dummies=False,
                            off_alpha=None, def_alpha=None):
    """
    Run regression analysis on the generated RAPM files.

    Args:
        year_range: String like "21_26" or "26"
        season_suffix: "all", "rs", or "ps"
        output_dir: Directory to save results
        start_year: Starting year (2-digit, e.g., 25 for 2025)
        end_year: Ending year (2-digit, e.g., 26 for 2026)
        timedecay: If True, read from td/ subfolder
        half_life: Half-life in days (for filename matching)
        rubberband: If True, match _rb suffix in filenames
    """
    logging.info("=" * 80)
    logging.info("RUNNING REGRESSION ANALYSIS")
    logging.info("=" * 80)

    import pandas as pd
    import numpy as np
    import statsmodels.api as sm
    alt_ts_decomp_supported = normalize_season_end_year(start_year) >= 2024

    def center_alt3_public_columns(df):
        """Center published alt3 factor columns by possession weight in-place."""
        def weighted_mean(col, weight_col):
            if col not in df.columns or weight_col not in df.columns:
                return None
            values = pd.to_numeric(df[col], errors='coerce')
            weights = pd.to_numeric(df[weight_col], errors='coerce')
            mask = values.notna() & weights.notna() & (weights > 0)
            if not mask.any():
                return None
            return float(np.average(values[mask], weights=weights[mask]))

        def center_column(col, weight_col):
            mean = weighted_mean(col, weight_col)
            if mean is None:
                return False
            df[col] = (pd.to_numeric(df[col], errors='coerce') - mean).round(2)
            return True

        def set_column(col, expr):
            df[col] = expr.round(2)

        for side, weight_col, total_col, fc_col, sc_col in [
            ('o', 'off_poss', 'off', 'oFC', 'oSC'),
            ('d', 'def_poss', 'def', 'dFC', 'dSC'),
        ]:
            if weight_col not in df.columns:
                continue

            has_sq_make = all(f'{side}ALT_{part}' in df.columns for part in ['SQ', 'MAKE'])
            if has_sq_make:
                center_column(f'{side}ALT_SQ', weight_col)
                center_column(f'{side}ALT_MAKE', weight_col)
                set_column(f'{side}ALT_EFG', df[f'{side}ALT_SQ'] + df[f'{side}ALT_MAKE'])
            else:
                center_column(f'{side}ALT_EFG', weight_col)

            center_column(f'{side}ALT_FT', weight_col)
            if f'{side}ALT_EFG' in df.columns and f'{side}ALT_FT' in df.columns:
                set_column(f'{side}ALT_TS', df[f'{side}ALT_EFG'] + df[f'{side}ALT_FT'])
            else:
                center_column(f'{side}ALT_TS', weight_col)

            center_column(f'{side}ALT_TOV', weight_col)
            center_column(f'{side}ALT_TOV_bp', weight_col)
            center_column(f'{side}ALT_TOV_sc', weight_col)

            if f'{side}ALT_TS' in df.columns and f'{side}ALT_TOV' in df.columns:
                set_column(fc_col, df[f'{side}ALT_TS'] + df[f'{side}ALT_TOV'])
            else:
                center_column(fc_col, weight_col)

            if total_col in df.columns and fc_col in df.columns:
                set_column(sc_col, df[total_col] - df[fc_col])
            else:
                center_column(sc_col, weight_col)

        component_cols = [
            'oALT_TS', 'oALT_TOV', 'oSC',
            'dALT_TS', 'dALT_TOV', 'dSC',
        ]
        if 'net_rapm' in df.columns and all(col in df.columns for col in component_cols):
            set_column('RESID', df['net_rapm'] - df[component_cols].sum(axis=1))

    def save_regression_coefficients(results, output_path):
        coef_data = [
            {
                'variable': 'R_squared',
                'coefficient': results.rsquared,
                'std_error': np.nan,
                'p_value': np.nan,
                'conf_int_lower': np.nan,
                'conf_int_upper': np.nan
            },
            {
                'variable': 'Adjusted_R_squared',
                'coefficient': results.rsquared_adj,
                'std_error': np.nan,
                'p_value': np.nan,
                'conf_int_lower': np.nan,
                'conf_int_upper': np.nan
            },
            {
                'variable': 'Intercept',
                'coefficient': results.params['const'],
                'std_error': results.bse['const'],
                'p_value': results.pvalues['const'],
                'conf_int_lower': results.conf_int().loc['const', 0],
                'conf_int_upper': results.conf_int().loc['const', 1]
            }
        ]

        for col in [idx for idx in results.params.index if idx != 'const']:
            coef_data.append({
                'variable': col,
                'coefficient': results.params[col],
                'std_error': results.bse[col],
                'p_value': results.pvalues[col],
                'conf_int_lower': results.conf_int().loc[col, 0],
                'conf_int_upper': results.conf_int().loc[col, 1]
            })

        pd.DataFrame(coef_data).to_csv(output_path, index=False)

    rapm_suffix = build_result_suffix(
        season_suffix,
        pure=True,
        rubberband=rubberband,
        season_effects=season_effects,
        age_curve=age_curve,
        age_dummies=age_dummies,
        timedecay=timedecay,
        half_life=half_life,
        off_alpha=off_alpha,
        def_alpha=def_alpha,
    )
    other_suffix = build_result_suffix(
        season_suffix,
        pure=False,
        rubberband=rubberband,
        season_effects=season_effects,
        age_curve=False,
        age_dummies=age_dummies,
        timedecay=timedecay,
        half_life=half_life,
        off_alpha=off_alpha,
        def_alpha=def_alpha,
    )
    age_artifact_suffix = "_agefe" if age_dummies else ("_age" if age_curve else "")

    # Files are in the output_dir (organized folder) after organize_results moves them
    rapm_file = output_dir / f"rapm_{year_range}_{rapm_suffix}_results.csv"
    ts_file = output_dir / f"ts_{year_range}_{other_suffix}_results.csv"
    tov_file = output_dir / f"tov_{year_range}_{other_suffix}_results.csv"
    reb_file = output_dir / f"reb_{year_range}_{other_suffix}_results.csv"
    badpass_tov_file = output_dir / f"badpass_tov_{year_range}_{other_suffix}_results.csv"
    scoring_tov_file = output_dir / f"scoring_tov_{year_range}_{other_suffix}_results.csv"
    second_chance_file = output_dir / f"second_chance_{year_range}_{other_suffix}_results.csv"
    first_chance_file = output_dir / f"first_chance_{year_range}_{other_suffix}_results.csv"
    alt_ts_file = output_dir / f"alt_ts_{year_range}_{other_suffix}_results.csv"
    alt_efg_file = output_dir / f"alt_efg_{year_range}_{other_suffix}_results.csv"
    alt_sq_file = output_dir / f"alt_sq_{year_range}_{other_suffix}_results.csv"
    alt_make_file = output_dir / f"alt_make_{year_range}_{other_suffix}_results.csv"
    alt_ft_file = output_dir / f"alt_ft_{year_range}_{other_suffix}_results.csv"
    alt_tov_file = output_dir / f"alt_tov_{year_range}_{other_suffix}_results.csv"
    alt_badpass_tov_file = output_dir / f"alt_badpass_tov_{year_range}_{other_suffix}_results.csv"
    alt_scoring_tov_file = output_dir / f"alt_scoring_tov_{year_range}_{other_suffix}_results.csv"
    second_chance_clean_file = output_dir / f"second_chance_clean_{year_range}_{other_suffix}_results.csv"
    ft_premium_file = output_dir / f"ft_premium_{year_range}_{other_suffix}_results.csv"
    auxiliary_files = {
        metric_key: output_dir / f"{file_stem}_{year_range}_{other_suffix}_results.csv"
        for _, file_stem, metric_key in AUXILIARY_FACTOR_METRICS
    }

    # Check that all files exist
    missing_files = []
    for f in [
        rapm_file, ts_file, tov_file, reb_file, badpass_tov_file, scoring_tov_file,
        second_chance_file, first_chance_file, alt_ts_file, alt_efg_file, alt_ft_file, alt_tov_file,
        alt_badpass_tov_file, alt_scoring_tov_file,
        second_chance_clean_file, ft_premium_file
    ]:
        if not f.exists():
            missing_files.append(str(f))
    for f in auxiliary_files.values():
        if not f.exists():
            missing_files.append(str(f))
    if alt_ts_decomp_supported:
        for f in [alt_sq_file, alt_make_file]:
            if not f.exists():
                missing_files.append(str(f))

    if missing_files:
        logging.error(f"Missing required files for regression: {', '.join(missing_files)}")
        return False

    # Load data
    logging.info("Loading RAPM files for regression...")
    rapm_df = pd.read_csv(rapm_file)
    ts_df = pd.read_csv(ts_file)
    tov_df = pd.read_csv(tov_file)
    reb_df = pd.read_csv(reb_file)
    badpass_tov_df = pd.read_csv(badpass_tov_file)
    scoring_tov_df = pd.read_csv(scoring_tov_file)
    second_chance_df = pd.read_csv(second_chance_file)
    first_chance_df = pd.read_csv(first_chance_file)
    alt_ts_df = pd.read_csv(alt_ts_file)
    alt_efg_df = pd.read_csv(alt_efg_file)
    alt_sq_df = pd.read_csv(alt_sq_file) if alt_ts_decomp_supported else None
    alt_make_df = pd.read_csv(alt_make_file) if alt_ts_decomp_supported else None
    alt_ft_df = pd.read_csv(alt_ft_file)
    alt_tov_df = pd.read_csv(alt_tov_file)
    alt_badpass_tov_df = pd.read_csv(alt_badpass_tov_file)
    alt_scoring_tov_df = pd.read_csv(alt_scoring_tov_file)
    second_chance_clean_df = pd.read_csv(second_chance_clean_file)
    ft_premium_df = pd.read_csv(ft_premium_file)
    auxiliary_dfs = {
        metric_key: pd.read_csv(auxiliary_files[metric_key])
        for _, _, metric_key in AUXILIARY_FACTOR_METRICS
    }

    logging.info(f"  RAPM: {len(rapm_df)} players")
    logging.info(f"  TS: {len(ts_df)} players")
    logging.info(f"  TOV: {len(tov_df)} players")
    logging.info(f"  REB: {len(reb_df)} players")
    logging.info(f"  BADPASS_TOV: {len(badpass_tov_df)} players")
    logging.info(f"  SCORING_TOV: {len(scoring_tov_df)} players")
    logging.info(f"  SECOND_CHANCE: {len(second_chance_df)} players")
    logging.info(f"  FIRST_CHANCE: {len(first_chance_df)} players")
    logging.info(f"  ALT_TS: {len(alt_ts_df)} players")
    logging.info(f"  ALT_EFG: {len(alt_efg_df)} players")
    logging.info(f"  ALT_FT: {len(alt_ft_df)} players")
    if alt_ts_decomp_supported:
        logging.info(f"  ALT_SQ: {len(alt_sq_df)} players")
        logging.info(f"  ALT_MAKE: {len(alt_make_df)} players")
    logging.info(f"  ALT_TOV: {len(alt_tov_df)} players")
    logging.info(f"  ALT_BADPASS_TOV: {len(alt_badpass_tov_df)} players")
    logging.info(f"  ALT_SCORING_TOV: {len(alt_scoring_tov_df)} players")
    logging.info(f"  SECOND_CHANCE_CLEAN: {len(second_chance_clean_df)} players")
    logging.info(f"  FT_PREMIUM: {len(ft_premium_df)} players")
    for _, _, metric_key in AUXILIARY_FACTOR_METRICS:
        logging.info(f"  {metric_key}: {len(auxiliary_dfs[metric_key])} players")
    
    # Merge datasets - now track off_poss and def_poss separately if available
    merge_cols = ['player_id', 'player_name', 'off', 'def', 'net_rapm', 'possessions']
    if 'off_poss' in rapm_df.columns:
        merge_cols.append('off_poss')
    if 'def_poss' in rapm_df.columns:
        merge_cols.append('def_poss')
    
    merged = rapm_df[merge_cols].copy()
    merged.rename(columns={'off': 'off_rapm', 'def': 'def_rapm'}, inplace=True)
    
    # Merge TS
    ts_subset = ts_df[['player_id', 'off', 'def']].rename(
        columns={'off': 'off_ts', 'def': 'def_ts'}
    )
    merged = merged.merge(ts_subset, on='player_id', how='inner')
    
    # Merge TOV
    tov_subset = tov_df[['player_id', 'off', 'def']].rename(
        columns={'off': 'off_tov', 'def': 'def_tov'}
    )
    merged = merged.merge(tov_subset, on='player_id', how='inner')
    
    # Merge REB
    reb_subset = reb_df[['player_id', 'off', 'def']].rename(
        columns={'off': 'off_reb', 'def': 'def_reb'}
    )
    merged = merged.merge(reb_subset, on='player_id', how='inner')

    # Merge BADPASS_TOV
    bp_subset = badpass_tov_df[['player_id', 'off', 'def']].rename(
        columns={'off': 'off_badpass_tov', 'def': 'def_badpass_tov'}
    )
    merged = merged.merge(bp_subset, on='player_id', how='inner')

    # Merge SCORING_TOV
    sc_subset = scoring_tov_df[['player_id', 'off', 'def']].rename(
        columns={'off': 'off_scoring_tov', 'def': 'def_scoring_tov'}
    )
    merged = merged.merge(sc_subset, on='player_id', how='inner')

    # Merge SECOND_CHANCE
    sc2_subset = second_chance_df[['player_id', 'off', 'def']].rename(
        columns={'off': 'off_sc2', 'def': 'def_sc2'}
    )
    merged = merged.merge(sc2_subset, on='player_id', how='inner')

    # Merge FT_PREMIUM (reported directly in weighted_factors; not used in the six-factor regressions)
    ft_subset = ft_premium_df[['player_id', 'off', 'def']].rename(
        columns={'off': 'off_ft', 'def': 'def_ft'}
    )
    merged = merged.merge(ft_subset, on='player_id', how='inner')

    for _, file_stem, metric_key in AUXILIARY_FACTOR_METRICS:
        aux_subset = auxiliary_dfs[metric_key][['player_id', 'off', 'def']].rename(
            columns={'off': f'off_{file_stem}', 'def': f'def_{file_stem}'}
        )
        merged = merged.merge(aux_subset, on='player_id', how='inner')

    logging.info(f"Merged dataset: {len(merged)} players")
    
    # Feature columns
    feature_cols = ['off_ts', 'def_ts', 'off_tov', 'def_tov', 'off_reb', 'def_reb']
    
    # Run 3 regressions: Net, Offensive, Defensive
    results_data = {}
    
    for target_col, target_name in [('net_rapm', 'Net'), ('off_rapm', 'Offensive'), ('def_rapm', 'Defensive')]:
        logging.info(f"\nRunning {target_name} RAPM regression...")
        
        X = merged[feature_cols]
        y = merged[target_col]
        weights = merged['possessions']
        
        X_with_const = sm.add_constant(X)
        model = sm.WLS(y, X_with_const, weights=weights)
        results = model.fit()
        
        logging.info(f"  R² = {results.rsquared:.4f}")
        results_data[target_col] = results
    
    # Save regression coefficients
    for target_col, target_name in [('net_rapm', 'net'), ('off_rapm', 'off'), ('def_rapm', 'def')]:
        results = results_data[target_col]
        output_path = output_dir / f'regression_{target_name}_coefficients{age_artifact_suffix}.csv'
        save_regression_coefficients(results, output_path)
        logging.info(f"  Saved {target_name} coefficients to {output_path.name}")
    
    # TOV decomposition: regress TOV coefficients on BADPASS_TOV + SCORING_TOV
    logging.info("\nRunning TOV decomposition (BADPASS + SCORING)...")
    tov_decomp_data = {}
    for side, tov_col, bp_col, sc_col in [
        ('off', 'off_tov', 'off_badpass_tov', 'off_scoring_tov'),
        ('def', 'def_tov', 'def_badpass_tov', 'def_scoring_tov'),
    ]:
        X_tov = merged[[bp_col, sc_col]]
        y_tov = merged[tov_col]
        X_tov_c = sm.add_constant(X_tov)
        tov_model = sm.WLS(y_tov, X_tov_c, weights=merged['possessions']).fit()
        tov_decomp_data[side] = tov_model
        logging.info(f"  TOV {side} decomposition: R²={tov_model.rsquared:.6f}, "
                     f"bp_beta={tov_model.params[bp_col]:.4f}, "
                     f"sc_beta={tov_model.params[sc_col]:.4f}")

    # Save TOV decomposition coefficients
    tov_decomp_rows = []
    for side in ['off', 'def']:
        model = tov_decomp_data[side]
        for var in model.params.index:
            tov_decomp_rows.append({
                'side': side,
                'variable': var,
                'coefficient': model.params[var],
                'std_error': model.bse[var],
                'p_value': model.pvalues[var],
            })
    tov_decomp_df = pd.DataFrame(tov_decomp_rows)
    tov_decomp_path = output_dir / f'tov_decomposition_coefficients{age_artifact_suffix}.csv'
    tov_decomp_df.to_csv(tov_decomp_path, index=False)
    logging.info(f"  Saved TOV decomposition coefficients to {tov_decomp_path.name}")

    # Create weighted factors CSV (using net RAPM coefficients)
    results_net = results_data['net_rapm']

    coef_off_ts = results_net.params['off_ts']
    coef_def_ts = results_net.params['def_ts']
    coef_off_tov = results_net.params['off_tov']
    coef_def_tov = results_net.params['def_tov']
    coef_off_reb = results_net.params['off_reb']
    coef_def_reb = results_net.params['def_reb']

    # TOV decomposition coefficients (second-stage)
    off_tov_decomp = tov_decomp_data['off']
    def_tov_decomp = tov_decomp_data['def']

    weighted_df = merged[['player_id', 'player_name']].copy()

    # Calculate Latest_Year
    latest_year = normalize_season_end_year(end_year)
    weighted_df['Latest_Year'] = latest_year
    weighted_df['Latest_Year'] = weighted_df['Latest_Year'].astype(int)

    # Calculate weighted factors
    weighted_df['oTS'] = (merged['off_ts'] * coef_off_ts).round(2)
    weighted_df['oTOV'] = (merged['off_tov'] * coef_off_tov).round(2)
    weighted_df['oREB'] = (merged['off_reb'] * coef_off_reb).round(2)
    weighted_df['dTS'] = (merged['def_ts'] * coef_def_ts).round(2)
    weighted_df['dTOV'] = (merged['def_tov'] * coef_def_tov).round(2)
    weighted_df['dREB'] = (merged['def_reb'] * coef_def_reb).round(2)

    # TOV decomposition sub-factors:
    # oTOV_bp = (badpass_tov_coeff * bp_beta) * coef_off_tov_from_net_regression
    # This chains: BADPASS_TOV → TOV → Net RAPM
    weighted_df['oTOV_bp'] = (merged['off_badpass_tov'] * off_tov_decomp.params['off_badpass_tov'] * coef_off_tov).round(2)
    weighted_df['oTOV_sc'] = (merged['off_scoring_tov'] * off_tov_decomp.params['off_scoring_tov'] * coef_off_tov).round(2)
    weighted_df['dTOV_bp'] = (merged['def_badpass_tov'] * def_tov_decomp.params['def_badpass_tov'] * coef_def_tov).round(2)
    weighted_df['dTOV_sc'] = (merged['def_scoring_tov'] * def_tov_decomp.params['def_scoring_tov'] * coef_def_tov).round(2)

    # FT_PREMIUM is copied through directly as an auxiliary TS subfactor.
    weighted_df['oFT'] = merged['off_ft'].round(2)
    weighted_df['dFT'] = (-merged['def_ft']).round(2)

    for _, file_stem, metric_key in AUXILIARY_FACTOR_METRICS:
        weighted_df[f'o{metric_key}'] = merged[f'off_{file_stem}'].round(2)
        weighted_df[f'd{metric_key}'] = (-merged[f'def_{file_stem}']).round(2)

    weighted_df['off'] = merged['off_rapm'].round(2)

    # FIX: Multiply def by -1 so positive = good defense
    weighted_df['def'] = (-merged['def_rapm']).round(2)

    weighted_df['net_rapm'] = merged['net_rapm'].round(2)

    # RESID = net_rapm minus sum of 6 weighted factors
    weighted_df['RESID'] = (weighted_df['net_rapm'] - (
        weighted_df['oTS'] + weighted_df['oTOV'] + weighted_df['oREB'] +
        weighted_df['dTS'] + weighted_df['dTOV'] + weighted_df['dREB']
    )).round(2)

    # Second chance / first chance (raw, not regression-weighted)
    weighted_df['o_sc'] = merged['off_sc2'].round(2)
    weighted_df['d_sc'] = (-merged['def_sc2']).round(2)
    weighted_df['o_fc'] = (merged['off_rapm'] - merged['off_sc2']).round(2)
    weighted_df['d_fc'] = (-(merged['def_rapm'] - merged['def_sc2'])).round(2)

    # Possession value = oTOV + oREB, dTOV + dREB
    weighted_df['o_pval'] = (weighted_df['oTOV'] + weighted_df['oREB']).round(2)
    weighted_df['d_pval'] = (weighted_df['dTOV'] + weighted_df['dREB']).round(2)

    # Add possessions columns
    weighted_df['possessions'] = merged['possessions']
    if 'off_poss' in merged.columns:
        weighted_df['off_poss'] = merged['off_poss']
    if 'def_poss' in merged.columns:
        weighted_df['def_poss'] = merged['def_poss']

    weighted_df = weighted_df.sort_values('net_rapm', ascending=False)

    # Reorder columns
    aux_off_columns = [f'o{metric_key}' for _, _, metric_key in AUXILIARY_FACTOR_METRICS]
    aux_def_columns = [f'd{metric_key}' for _, _, metric_key in AUXILIARY_FACTOR_METRICS]
    column_order = ['player_id', 'player_name', 'Latest_Year', 'oTS', 'oFT', *aux_off_columns,
                    'oTOV', 'oTOV_bp', 'oTOV_sc', 'oREB',
                    'dTS', 'dFT', *aux_def_columns, 'dTOV', 'dTOV_bp', 'dTOV_sc', 'dREB',
                    'off', 'def', 'net_rapm', 'RESID',
                    'o_sc', 'o_fc', 'd_sc', 'd_fc', 'o_pval', 'd_pval', 'possessions']
    if 'off_poss' in weighted_df.columns:
        column_order.append('off_poss')
    if 'def_poss' in weighted_df.columns:
        column_order.append('def_poss')
    
    weighted_df = weighted_df[column_order]
    
    # Save with new naming convention
    weighted_filename = f'weighted_factors_{year_range}_{build_result_suffix(season_suffix, rubberband=rubberband, season_effects=season_effects, age_curve=age_curve, age_dummies=age_dummies, timedecay=timedecay, half_life=half_life, off_alpha=off_alpha, def_alpha=def_alpha)}.csv'

    output_path = output_dir / weighted_filename
    weighted_df.to_csv(output_path, index=False)
    logging.info(f"  Saved weighted factors to {output_path.name}")

    # Also save directly to results/ folder (non-timedecay only, to avoid duplicates)
    if not timedecay:
        results_output = RESULTS_DIR / weighted_filename
        weighted_df.to_csv(results_output, index=False)
        logging.info(f"  Saved weighted factors to results/{weighted_filename}")

    # Copy to master_results folder
    MASTER_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    master_output = MASTER_RESULTS_DIR / weighted_filename
    shutil.copy2(output_path, master_output)
    logging.info(f"  Copied weighted factors to master_results/")

    # Alternate clean 3-factor build
    logging.info("\nRunning alternate clean 3-factor regression analysis...")
    merged_alt = rapm_df[merge_cols].copy()
    merged_alt.rename(columns={'off': 'off_rapm', 'def': 'def_rapm'}, inplace=True)

    alt_merge_inputs = [
        (first_chance_df, 'first_chance', 'off_fc_raw', 'def_fc_raw'),
        (alt_ts_df, 'alt_ts', 'off_alt_ts', 'def_alt_ts'),
        (alt_efg_df, 'alt_efg', 'off_alt_efg', 'def_alt_efg'),
        (alt_ft_df, 'alt_ft', 'off_alt_ft', 'def_alt_ft'),
        (alt_tov_df, 'alt_tov', 'off_alt_tov', 'def_alt_tov'),
        (alt_badpass_tov_df, 'alt_badpass_tov', 'off_alt_badpass_tov', 'def_alt_badpass_tov'),
        (alt_scoring_tov_df, 'alt_scoring_tov', 'off_alt_scoring_tov', 'def_alt_scoring_tov'),
    ]
    if alt_ts_decomp_supported:
        alt_merge_inputs.extend([
            (alt_sq_df, 'alt_sq', 'off_alt_sq', 'def_alt_sq'),
            (alt_make_df, 'alt_make', 'off_alt_make', 'def_alt_make'),
        ])
    alt_merge_inputs.append((second_chance_clean_df, 'second_chance_clean', 'off_sc_clean', 'def_sc_clean'))
    for _, file_stem, metric_key in AUXILIARY_FACTOR_METRICS:
        alt_merge_inputs.append(
            (auxiliary_dfs[metric_key], metric_key, f'off_{file_stem}', f'def_{file_stem}')
        )
    for source_df, label, off_name, def_name in alt_merge_inputs:
        subset = source_df[['player_id', 'off', 'def']].rename(
            columns={'off': off_name, 'def': def_name}
        )
        merged_alt = merged_alt.merge(subset, on='player_id', how='inner')
        logging.info(f"  Merged {label}: {len(merged_alt)} players remain")

    if alt_ts_decomp_supported:
        alt_feature_cols = [
            'off_alt_sq', 'def_alt_sq',
            'off_alt_make', 'def_alt_make',
            'off_alt_ft', 'def_alt_ft',
            'off_alt_tov', 'def_alt_tov',
            'off_sc_clean', 'def_sc_clean'
        ]
    else:
        alt_feature_cols = [
            'off_alt_ts', 'def_alt_ts',
            'off_alt_tov', 'def_alt_tov',
            'off_sc_clean', 'def_sc_clean'
        ]
    alt_results_data = {}
    for target_col, target_name in [('net_rapm', 'Net'), ('off_rapm', 'Offensive'), ('def_rapm', 'Defensive')]:
        logging.info(f"Running alternate {target_name} RAPM regression...")
        X_alt = merged_alt[alt_feature_cols]
        y_alt = merged_alt[target_col]
        weights_alt = merged_alt['possessions']

        X_alt_const = sm.add_constant(X_alt)
        alt_model = sm.WLS(y_alt, X_alt_const, weights=weights_alt).fit()
        alt_results_data[target_col] = alt_model
        logging.info(f"  Alt {target_name} R² = {alt_model.rsquared:.4f}")

    for target_col, target_name in [('net_rapm', 'net'), ('off_rapm', 'off'), ('def_rapm', 'def')]:
        alt_output_path = output_dir / f'regression_alt3_{target_name}_coefficients{age_artifact_suffix}.csv'
        save_regression_coefficients(alt_results_data[target_col], alt_output_path)
        logging.info(f"  Saved alternate {target_name} coefficients to {alt_output_path.name}")

    logging.info("Running FIRST_CHANCE decomposition checks...")
    fc_decomp_rows = []
    alt_ts_decomp_data = {}
    for side, alt_ts_col, alt_efg_col, alt_ft_col in [
        ('off', 'off_alt_ts', 'off_alt_efg', 'off_alt_ft'),
        ('def', 'def_alt_ts', 'def_alt_efg', 'def_alt_ft'),
    ]:
        X_fc = merged_alt[[alt_efg_col, alt_ft_col]]
        y_fc = merged_alt[alt_ts_col]
        X_fc_const = sm.add_constant(X_fc)
        fc_model = sm.WLS(y_fc, X_fc_const, weights=merged_alt['possessions']).fit()
        alt_ts_decomp_data[side] = fc_model
        logging.info(
            "  ALT_TS %s from ALT_EFG+ALT_FT: R²=%.6f, alt_efg_beta=%.4f, alt_ft_beta=%.4f",
            side,
            fc_model.rsquared,
            fc_model.params[alt_efg_col],
            fc_model.params[alt_ft_col],
        )
        for var in fc_model.params.index:
            fc_decomp_rows.append({
                'model': 'alt_ts_from_efg_ft',
                'side': side,
                'variable': var,
                'coefficient': fc_model.params[var],
                'std_error': fc_model.bse[var],
                'p_value': fc_model.pvalues[var],
            })

    if alt_ts_decomp_supported:
        for side, alt_ts_col, alt_sq_col, alt_make_col, alt_ft_col in [
            ('off', 'off_alt_ts', 'off_alt_sq', 'off_alt_make', 'off_alt_ft'),
            ('def', 'def_alt_ts', 'def_alt_sq', 'def_alt_make', 'def_alt_ft'),
        ]:
            X_fc = merged_alt[[alt_sq_col, alt_make_col, alt_ft_col]]
            y_fc = merged_alt[alt_ts_col]
            X_fc_const = sm.add_constant(X_fc)
            fc_model = sm.WLS(y_fc, X_fc_const, weights=merged_alt['possessions']).fit()
            logging.info(
                "  ALT_TS %s decomposition: R²=%.6f, alt_sq_beta=%.4f, alt_make_beta=%.4f, alt_ft_beta=%.4f",
                side,
                fc_model.rsquared,
                fc_model.params[alt_sq_col],
                fc_model.params[alt_make_col],
                fc_model.params[alt_ft_col],
            )
            for var in fc_model.params.index:
                fc_decomp_rows.append({
                    'model': 'alt_ts_from_sq_make_ft',
                    'side': side,
                    'variable': var,
                    'coefficient': fc_model.params[var],
                    'std_error': fc_model.bse[var],
                    'p_value': fc_model.pvalues[var],
                })

    for side, fc_col, alt_ts_col, alt_tov_col in [
        ('off', 'off_fc_raw', 'off_alt_ts', 'off_alt_tov'),
        ('def', 'def_fc_raw', 'def_alt_ts', 'def_alt_tov'),
    ]:
        X_fc = merged_alt[[alt_ts_col, alt_tov_col]]
        y_fc = merged_alt[fc_col]
        X_fc_const = sm.add_constant(X_fc)
        fc_model = sm.WLS(y_fc, X_fc_const, weights=merged_alt['possessions']).fit()
        logging.info(
            "  FIRST_CHANCE %s reconstruction: R²=%.6f, alt_ts_beta=%.4f, alt_tov_beta=%.4f",
            side,
            fc_model.rsquared,
            fc_model.params[alt_ts_col],
            fc_model.params[alt_tov_col],
        )
        for var in fc_model.params.index:
            fc_decomp_rows.append({
                'model': 'fc_raw_from_alt_ts_tov',
                'side': side,
                'variable': var,
                'coefficient': fc_model.params[var],
                'std_error': fc_model.bse[var],
                'p_value': fc_model.pvalues[var],
            })

    fc_decomp_path = output_dir / f'alt3_first_chance_decomposition_coefficients{age_artifact_suffix}.csv'
    pd.DataFrame(fc_decomp_rows).to_csv(fc_decomp_path, index=False)
    logging.info(f"  Saved FIRST_CHANCE decomposition coefficients to {fc_decomp_path.name}")

    logging.info("Running ALT_TOV decomposition (BADPASS + SCORING)...")
    alt_tov_decomp_data = {}
    for side, alt_tov_col, bp_col, sc_col in [
        ('off', 'off_alt_tov', 'off_alt_badpass_tov', 'off_alt_scoring_tov'),
        ('def', 'def_alt_tov', 'def_alt_badpass_tov', 'def_alt_scoring_tov'),
    ]:
        X_alt_tov = merged_alt[[bp_col, sc_col]]
        y_alt_tov = merged_alt[alt_tov_col]
        X_alt_tov_c = sm.add_constant(X_alt_tov)
        alt_tov_model = sm.WLS(y_alt_tov, X_alt_tov_c, weights=merged_alt['possessions']).fit()
        alt_tov_decomp_data[side] = alt_tov_model
        logging.info(
            "  ALT_TOV %s decomposition: R²=%.6f, bp_beta=%.4f, sc_beta=%.4f",
            side,
            alt_tov_model.rsquared,
            alt_tov_model.params[bp_col],
            alt_tov_model.params[sc_col],
        )

    alt_tov_decomp_rows = []
    for side in ['off', 'def']:
        model = alt_tov_decomp_data[side]
        for var in model.params.index:
            alt_tov_decomp_rows.append({
                'side': side,
                'variable': var,
                'coefficient': model.params[var],
                'std_error': model.bse[var],
                'p_value': model.pvalues[var],
            })
    alt_tov_decomp_path = output_dir / f'alt_tov_decomposition_coefficients{age_artifact_suffix}.csv'
    pd.DataFrame(alt_tov_decomp_rows).to_csv(alt_tov_decomp_path, index=False)
    logging.info(f"  Saved ALT_TOV decomposition coefficients to {alt_tov_decomp_path.name}")

    alt_results_net = alt_results_data['net_rapm']
    alt_weighted_df = merged_alt[['player_id', 'player_name']].copy()
    alt_weighted_df['Latest_Year'] = latest_year
    alt_weighted_df['Latest_Year'] = alt_weighted_df['Latest_Year'].astype(int)

    if alt_ts_decomp_supported:
        alt_weighted_df['oALT_SQ'] = (merged_alt['off_alt_sq'] * alt_results_net.params['off_alt_sq']).round(2)
        alt_weighted_df['oALT_MAKE'] = (merged_alt['off_alt_make'] * alt_results_net.params['off_alt_make']).round(2)
        alt_weighted_df['oALT_FT'] = (merged_alt['off_alt_ft'] * alt_results_net.params['off_alt_ft']).round(2)
        alt_weighted_df['oALT_EFG'] = (
            alt_weighted_df['oALT_SQ'] +
            alt_weighted_df['oALT_MAKE']
        ).round(2)
        alt_weighted_df['oALT_TS'] = (
            alt_weighted_df['oALT_EFG'] +
            alt_weighted_df['oALT_FT']
        ).round(2)
        alt_weighted_df['oALT_TOV'] = (merged_alt['off_alt_tov'] * alt_results_net.params['off_alt_tov']).round(2)
        alt_weighted_df['oALT_TOV_bp'] = (
            merged_alt['off_alt_badpass_tov'] *
            alt_tov_decomp_data['off'].params['off_alt_badpass_tov'] *
            alt_results_net.params['off_alt_tov']
        ).round(2)
        alt_weighted_df['oALT_TOV_sc'] = (
            merged_alt['off_alt_scoring_tov'] *
            alt_tov_decomp_data['off'].params['off_alt_scoring_tov'] *
            alt_results_net.params['off_alt_tov']
        ).round(2)
        alt_weighted_df['dALT_SQ'] = (merged_alt['def_alt_sq'] * alt_results_net.params['def_alt_sq']).round(2)
        alt_weighted_df['dALT_MAKE'] = (merged_alt['def_alt_make'] * alt_results_net.params['def_alt_make']).round(2)
        alt_weighted_df['dALT_FT'] = (merged_alt['def_alt_ft'] * alt_results_net.params['def_alt_ft']).round(2)
        alt_weighted_df['dALT_EFG'] = (
            alt_weighted_df['dALT_SQ'] +
            alt_weighted_df['dALT_MAKE']
        ).round(2)
        alt_weighted_df['dALT_TS'] = (
            alt_weighted_df['dALT_EFG'] +
            alt_weighted_df['dALT_FT']
        ).round(2)
        alt_weighted_df['dALT_TOV'] = (merged_alt['def_alt_tov'] * alt_results_net.params['def_alt_tov']).round(2)
        alt_weighted_df['dALT_TOV_bp'] = (
            merged_alt['def_alt_badpass_tov'] *
            alt_tov_decomp_data['def'].params['def_alt_badpass_tov'] *
            alt_results_net.params['def_alt_tov']
        ).round(2)
        alt_weighted_df['dALT_TOV_sc'] = (
            merged_alt['def_alt_scoring_tov'] *
            alt_tov_decomp_data['def'].params['def_alt_scoring_tov'] *
            alt_results_net.params['def_alt_tov']
        ).round(2)
        alt_weighted_df['oFC'] = (
            alt_weighted_df['oALT_TS'] + alt_weighted_df['oALT_TOV']
        ).round(2)
        alt_weighted_df['dFC'] = (
            alt_weighted_df['dALT_TS'] + alt_weighted_df['dALT_TOV']
        ).round(2)
    else:
        alt_weighted_df['oALT_TS'] = (merged_alt['off_alt_ts'] * alt_results_net.params['off_alt_ts']).round(2)
        alt_weighted_df['oALT_EFG'] = (
            merged_alt['off_alt_efg'] *
            alt_ts_decomp_data['off'].params['off_alt_efg'] *
            alt_results_net.params['off_alt_ts']
        ).round(2)
        alt_weighted_df['oALT_FT'] = (
            alt_weighted_df['oALT_TS'] - alt_weighted_df['oALT_EFG']
        ).round(2)
        alt_weighted_df['oALT_TOV'] = (merged_alt['off_alt_tov'] * alt_results_net.params['off_alt_tov']).round(2)
        alt_weighted_df['oALT_TOV_bp'] = (
            merged_alt['off_alt_badpass_tov'] *
            alt_tov_decomp_data['off'].params['off_alt_badpass_tov'] *
            alt_results_net.params['off_alt_tov']
        ).round(2)
        alt_weighted_df['oALT_TOV_sc'] = (
            merged_alt['off_alt_scoring_tov'] *
            alt_tov_decomp_data['off'].params['off_alt_scoring_tov'] *
            alt_results_net.params['off_alt_tov']
        ).round(2)
        alt_weighted_df['dALT_TS'] = (merged_alt['def_alt_ts'] * alt_results_net.params['def_alt_ts']).round(2)
        alt_weighted_df['dALT_EFG'] = (
            merged_alt['def_alt_efg'] *
            alt_ts_decomp_data['def'].params['def_alt_efg'] *
            alt_results_net.params['def_alt_ts']
        ).round(2)
        alt_weighted_df['dALT_FT'] = (
            alt_weighted_df['dALT_TS'] - alt_weighted_df['dALT_EFG']
        ).round(2)
        alt_weighted_df['dALT_TOV'] = (merged_alt['def_alt_tov'] * alt_results_net.params['def_alt_tov']).round(2)
        alt_weighted_df['dALT_TOV_bp'] = (
            merged_alt['def_alt_badpass_tov'] *
            alt_tov_decomp_data['def'].params['def_alt_badpass_tov'] *
            alt_results_net.params['def_alt_tov']
        ).round(2)
        alt_weighted_df['dALT_TOV_sc'] = (
            merged_alt['def_alt_scoring_tov'] *
            alt_tov_decomp_data['def'].params['def_alt_scoring_tov'] *
            alt_results_net.params['def_alt_tov']
        ).round(2)
        alt_weighted_df['oFC'] = (alt_weighted_df['oALT_TS'] + alt_weighted_df['oALT_TOV']).round(2)
        alt_weighted_df['dFC'] = (alt_weighted_df['dALT_TS'] + alt_weighted_df['dALT_TOV']).round(2)
    alt_weighted_df['off'] = merged_alt['off_rapm'].round(2)
    alt_weighted_df['def'] = (-merged_alt['def_rapm']).round(2)
    for _, file_stem, metric_key in AUXILIARY_FACTOR_METRICS:
        alt_weighted_df[f'o{metric_key}'] = merged_alt[f'off_{file_stem}'].round(2)
        alt_weighted_df[f'd{metric_key}'] = (-merged_alt[f'def_{file_stem}']).round(2)

    # For the public additive decomposition, trust the larger-sample first-chance split
    # and let second chance absorb the remaining offense/defense.
    alt_weighted_df['oSC'] = (alt_weighted_df['off'] - alt_weighted_df['oFC']).round(2)
    alt_weighted_df['dSC'] = (alt_weighted_df['def'] - alt_weighted_df['dFC']).round(2)
    alt_weighted_df['net_rapm'] = merged_alt['net_rapm'].round(2)
    if alt_ts_decomp_supported:
        alt_weighted_df['RESID'] = (
            alt_weighted_df['net_rapm'] - (
                alt_weighted_df['oALT_TS'] + alt_weighted_df['oALT_TOV'] + alt_weighted_df['oSC'] +
                alt_weighted_df['dALT_TS'] + alt_weighted_df['dALT_TOV'] + alt_weighted_df['dSC']
            )
        ).round(2)
    else:
        alt_weighted_df['RESID'] = (
            alt_weighted_df['net_rapm'] - (
                alt_weighted_df['oALT_TS'] + alt_weighted_df['oALT_TOV'] + alt_weighted_df['oSC'] +
                alt_weighted_df['dALT_TS'] + alt_weighted_df['dALT_TOV'] + alt_weighted_df['dSC']
            )
        ).round(2)
    alt_weighted_df['possessions'] = merged_alt['possessions']
    if 'off_poss' in merged_alt.columns:
        alt_weighted_df['off_poss'] = merged_alt['off_poss']
    if 'def_poss' in merged_alt.columns:
        alt_weighted_df['def_poss'] = merged_alt['def_poss']
    center_alt3_public_columns(alt_weighted_df)
    logging.info("  Centered alt3 public factor columns by possession-weighted player mean")

    alt_weighted_df = alt_weighted_df.sort_values('net_rapm', ascending=False)
    if alt_ts_decomp_supported:
        alt_column_order = [
            'player_id', 'player_name', 'Latest_Year',
            *aux_off_columns,
            'oALT_TS', 'oALT_EFG', 'oALT_SQ', 'oALT_MAKE', 'oALT_FT', 'oALT_TOV', 'oALT_TOV_bp', 'oALT_TOV_sc', 'oSC',
            *aux_def_columns,
            'dALT_TS', 'dALT_EFG', 'dALT_SQ', 'dALT_MAKE', 'dALT_FT', 'dALT_TOV', 'dALT_TOV_bp', 'dALT_TOV_sc', 'dSC',
            'oFC', 'dFC',
            'off', 'def', 'net_rapm', 'RESID', 'possessions'
        ]
    else:
        alt_column_order = [
            'player_id', 'player_name', 'Latest_Year',
            *aux_off_columns,
            'oALT_TS', 'oALT_EFG', 'oALT_FT', 'oALT_TOV', 'oALT_TOV_bp', 'oALT_TOV_sc', 'oSC',
            *aux_def_columns,
            'dALT_TS', 'dALT_EFG', 'dALT_FT', 'dALT_TOV', 'dALT_TOV_bp', 'dALT_TOV_sc', 'dSC',
            'oFC', 'dFC',
            'off', 'def', 'net_rapm', 'RESID', 'possessions'
        ]
    if 'off_poss' in alt_weighted_df.columns:
        alt_column_order.append('off_poss')
    if 'def_poss' in alt_weighted_df.columns:
        alt_column_order.append('def_poss')
    alt_weighted_df = alt_weighted_df[alt_column_order]

    alt_weighted_filename = (
        f'weighted_factors_alt3_{year_range}_'
        f'{build_result_suffix(season_suffix, pure=False, rubberband=rubberband, season_effects=season_effects, age_curve=age_curve, age_dummies=age_dummies, timedecay=timedecay, half_life=half_life, off_alpha=off_alpha, def_alpha=def_alpha)}.csv'
    )
    alt_output_path = output_dir / alt_weighted_filename
    alt_weighted_df.to_csv(alt_output_path, index=False)
    logging.info(f"  Saved alternate weighted factors to {alt_output_path.name}")

    if not timedecay:
        alt_results_output = RESULTS_DIR / alt_weighted_filename
        alt_weighted_df.to_csv(alt_results_output, index=False)
        logging.info(f"  Saved alternate weighted factors to results/{alt_weighted_filename}")

    alt_master_output = MASTER_RESULTS_DIR / alt_weighted_filename
    shutil.copy2(alt_output_path, alt_master_output)
    logging.info("  Copied alternate weighted factors to master_results/")
    
    # Create predictions for all players
    for target_col, target_name, pred_col in [
        ('net_rapm', 'net', 'predicted_net_rapm'),
        ('off_rapm', 'off', 'predicted_off_rapm'),
        ('def_rapm', 'def', 'predicted_def_rapm')
    ]:
        results = results_data[target_col]
        X = merged[feature_cols]
        X_with_const = sm.add_constant(X)
        
        pred_df = merged[['player_name', target_col, 'possessions']].copy()
        pred_df[pred_col] = results.predict(X_with_const)
        pred_df['residual'] = pred_df[target_col] - pred_df[pred_col]
        pred_df = pred_df.sort_values(target_col, ascending=False)
        
        pred_df[[target_col, pred_col, 'residual']] = pred_df[[target_col, pred_col, 'residual']].round(2)
        
        output_path = output_dir / f'predictions_{target_name}_all{age_artifact_suffix}.csv'
        pred_df.to_csv(output_path, index=False)
        logging.info(f"  Saved {target_name} predictions to {output_path.name}")
    
    logging.info("\nRegression analysis complete!")
    return True


def organize_results(year_range, season_suffix, start_year, end_year, season_type,
                     timedecay=False, half_life=None, rubberband=False,
                     season_effects=False, age_curve=False, age_dummies=False,
                     off_alpha=None, def_alpha=None):
    """
    Organize all generated files into a dedicated folder.

    Args:
        year_range: String like "21_26" (new compact format)
        season_suffix: "all", "rs", or "ps"
        start_year: Starting year (for folder name)
        end_year: Ending year (for folder name)
        season_type: RS, PS, or ALL
        timedecay: If True, use td/ subfolder and _td suffix
        half_life: Half-life in days for time decay
        rubberband: If True, add _rb to folder/file names

    Returns:
        Path to the output directory
    """
    td_suffix = f"_td{int(half_life) if half_life else 700}" if timedecay else ""
    rb_suffix = "_rb" if rubberband else ""
    se_suffix = "_se" if season_effects else ""
    age_suffix = "_age" if age_curve else ""
    agefe_suffix = "_agefe" if age_dummies else ""
    alpha_suffix = ""
    actual_off_alpha = 3000 if off_alpha is None else float(off_alpha)
    actual_def_alpha = 3000 if def_alpha is None else float(def_alpha)
    if actual_off_alpha != 3000 or actual_def_alpha != 3000:
        alpha_suffix = f"_a{int(actual_off_alpha)}_{int(actual_def_alpha)}"

    # Create output directory with new naming
    if start_year == end_year:
        folder_name = f"rapm_{start_year:02d}_{season_type.lower()}{rb_suffix}{se_suffix}{agefe_suffix}{age_suffix}{td_suffix}{alpha_suffix}"
    else:
        folder_name = f"rapm_{start_year:02d}_{end_year:02d}_{season_type.lower()}{rb_suffix}{se_suffix}{agefe_suffix}{age_suffix}{td_suffix}{alpha_suffix}"

    if timedecay:
        output_dir = RESULTS_DIR / "td" / folder_name
    else:
        output_dir = RESULTS_DIR / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"\nOrganizing results into: {output_dir}")

    if timedecay:
        source_dir = RESULTS_DIR / "td"
    else:
        source_dir = RESULTS_DIR

    rapm_suffix = build_result_suffix(
        season_suffix,
        pure=True,
        rubberband=rubberband,
        season_effects=season_effects,
        age_curve=age_curve,
        age_dummies=age_dummies,
        timedecay=timedecay,
        half_life=half_life,
        off_alpha=off_alpha,
        def_alpha=def_alpha,
    )
    other_suffix = build_result_suffix(
        season_suffix,
        pure=False,
        rubberband=rubberband,
        season_effects=season_effects,
        age_curve=False,
        age_dummies=age_dummies,
        timedecay=timedecay,
        half_life=half_life,
        off_alpha=off_alpha,
        def_alpha=def_alpha,
    )

    # Move files to organized folder (to avoid duplicates)
    files_to_move = [
        f"rapm_{year_range}_{rapm_suffix}_results.csv",
        f"ts_{year_range}_{other_suffix}_results.csv",
        f"tov_{year_range}_{other_suffix}_results.csv",
        f"reb_{year_range}_{other_suffix}_results.csv",
        f"ft_premium_{year_range}_{other_suffix}_results.csv",
        f"badpass_tov_{year_range}_{other_suffix}_results.csv",
        f"scoring_tov_{year_range}_{other_suffix}_results.csv",
        f"second_chance_{year_range}_{other_suffix}_results.csv",
        f"first_chance_{year_range}_{other_suffix}_results.csv",
        *[
            f"{file_stem}_{year_range}_{other_suffix}_results.csv"
            for _, file_stem, _ in AUXILIARY_FACTOR_METRICS
        ],
        f"alt_ts_{year_range}_{other_suffix}_results.csv",
        f"alt_efg_{year_range}_{other_suffix}_results.csv",
        f"alt_sq_{year_range}_{other_suffix}_results.csv",
        f"alt_make_{year_range}_{other_suffix}_results.csv",
        f"alt_ft_{year_range}_{other_suffix}_results.csv",
        f"alt_tov_{year_range}_{other_suffix}_results.csv",
        f"alt_badpass_tov_{year_range}_{other_suffix}_results.csv",
        f"alt_scoring_tov_{year_range}_{other_suffix}_results.csv",
        f"second_chance_clean_{year_range}_{other_suffix}_results.csv",
    ]

    for filename in files_to_move:
        source = source_dir / filename
        if source.exists():
            dest = output_dir / filename
            shutil.move(source, dest)
            logging.info(f"  Moved {filename}")

    if season_effects:
        season_effect_files = [filename.replace("_results.csv", "_season_effects.csv") for filename in files_to_move]
        for filename in season_effect_files:
            source = source_dir / filename
            if source.exists():
                dest = output_dir / filename
                shutil.move(source, dest)
                logging.info(f"  Moved {filename}")

    if age_dummies:
        age_effect_files = [filename.replace("_results.csv", "_age_effects.csv") for filename in files_to_move]
        for filename in age_effect_files:
            source = source_dir / filename
            if source.exists():
                dest = output_dir / filename
                shutil.move(source, dest)
                logging.info(f"  Moved {filename}")

    # Move rubberband coefficient/effect files if present
    if rubberband:
        rb_prefixes = [
            f"rapm_{year_range}_{rapm_suffix}",
            *[
                filename.replace("_results.csv", "")
                for filename in files_to_move
                if not filename.startswith(f"rapm_{year_range}_")
            ],
        ]
        for rb_prefix in rb_prefixes:
            for rb_file_suffix in ['_rubberband_coefficients.csv', '_rubberband_effects.csv']:
                rb_filename = f"{rb_prefix}{rb_file_suffix}"
                source = source_dir / rb_filename
                if source.exists():
                    dest = output_dir / rb_filename
                    shutil.move(source, dest)
                    logging.info(f"  Moved {rb_filename}")

    return output_dir


def publish_weighted_factors_to_rapms(year_range, weighted_suffix):
    """
    Copy this run's weighted-factor outputs to the downstream rapms repo and push.

    Only the standard and alt3 weighted-factor CSVs for this exact run are staged.
    Existing staged changes in the downstream repo are treated as a blocker so this
    helper cannot accidentally commit unrelated work.
    """
    if not RAPMS_DIR.exists():
        raise FileNotFoundError(f"Downstream rapms repo not found: {RAPMS_DIR}")

    filenames = [
        f"weighted_factors_{year_range}_{weighted_suffix}.csv",
        f"weighted_factors_alt3_{year_range}_{weighted_suffix}.csv",
    ]
    missing = [name for name in filenames if not (MASTER_RESULTS_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Cannot publish missing master_results file(s): " + ", ".join(missing)
        )

    preexisting_staged = subprocess.run(
        ['git', '-C', str(RAPMS_DIR), 'diff', '--cached', '--name-only'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if preexisting_staged:
        raise RuntimeError(
            "Downstream rapms repo already has staged changes; refusing to mix commits:\n"
            + preexisting_staged
        )

    RAPMS_MASTER_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    copied_paths = []
    for filename in filenames:
        source = MASTER_RESULTS_DIR / filename
        dest = RAPMS_MASTER_RESULTS_DIR / filename
        shutil.copy2(source, dest)
        copied_paths.append(dest)
        logging.info("Published copy: %s -> %s", source, dest)

    rel_paths = [str(path.relative_to(RAPMS_DIR)) for path in copied_paths]
    subprocess.run(
        ['git', '-C', str(RAPMS_DIR), 'add', *rel_paths],
        check=True,
        capture_output=True,
        text=True,
    )

    diff_result = subprocess.run(
        ['git', '-C', str(RAPMS_DIR), 'diff', '--cached', '--quiet'],
        capture_output=True,
        text=True,
    )
    if diff_result.returncode == 0:
        logging.info("No downstream rapms changes detected after publish; skipping commit/push")
        return copied_paths, False
    if diff_result.returncode != 1:
        raise RuntimeError(f"git diff --cached --quiet failed: {diff_result.stderr}")

    commit_message = f"Publish weighted factors {year_range} {weighted_suffix}"
    subprocess.run(
        ['git', '-C', str(RAPMS_DIR), 'commit', '-m', commit_message],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ['git', '-C', str(RAPMS_DIR), 'push', 'origin', 'main'],
        check=True,
        capture_output=True,
        text=True,
    )
    logging.info("Published weighted factors to rapms and pushed origin/main")
    return copied_paths, True


def main():
    parser = argparse.ArgumentParser(
        description='Master RAPM Analysis - run weighted-factor RAPMs and regression',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Multi-year all seasons (2023-2025)
  python 03_run_rapm_analysis.py 23 25 ALL

  # Single year regular season (2024-25)
  python 03_run_rapm_analysis.py 25 25 RS

  # Multi-year playoffs (2024-2026)
  python 03_run_rapm_analysis.py 24 26 PS

  # Time decay with 700-day half-life (6 years)
  python 03_run_rapm_analysis.py 21 26 ALL --timedecay --half-life 700

  # Standard run with season fixed effects
  python 03_run_rapm_analysis.py 14 26 ALL --season-effects

  # Standard run with season and age fixed effects on every daily weighted-factor metric
  python 03_run_rapm_analysis.py 14 26 ALL --season-effects --age-dummies

  # Benchmark solver orchestration: 8 concurrent RAPMs, 4 load cores per RAPM
  python 03_run_rapm_analysis.py 1 4 ALL --rapm-workers 8 --cores-per-rapm 4

  # Three-year historical run with fixed offense/defense alphas
  python 03_run_rapm_analysis.py 97 99 ALL --rubberband --season-effects --off-alpha 2000 --def-alpha 4000
        """
    )

    parser.add_argument('start_year',
                       type=int,
                       help='Starting year (2-digit, e.g., 24 for 2024-25)')
    parser.add_argument('end_year',
                       type=int,
                       help='Ending year (2-digit, inclusive, e.g., 26 for 2025-26)')
    parser.add_argument('season_type',
                       choices=['RS', 'PS', 'ALL'],
                       help='Season type: RS (regular season), PS (playoffs), or ALL (both)')
    parser.add_argument('--timedecay', '-td',
                       action='store_true',
                       help='Apply time decay weighting (recent games weighted more)')
    parser.add_argument('--half-life',
                       type=float,
                       default=None,
                       help='Half-life in days for time decay (default: 700). Implies --timedecay.')
    parser.add_argument('--rubberband', '-rb',
                       action='store_true',
                       help='Enable rubberband effect (score-margin x period adjustment)')
    parser.add_argument('--alpha-rb',
                       type=float,
                       default=None,
                       help='Rubberband regularization alpha (default: 10)')
    parser.add_argument('--season-effects', '-se',
                       action='store_true',
                       help='Pass --season-effects through to every RAPM solve and suffix downstream outputs with _se.')
    parser.add_argument('--age-curve',
                       action='store_true',
                       help='Pass --age-curve to the RAPM solve only and suffix downstream outputs with _age.')
    parser.add_argument('--age-dummies',
                       action='store_true',
                       help='Pass --age-dummies through to every weighted-factor metric solve and suffix downstream outputs with _agefe.')
    parser.add_argument('--rapm-workers',
                       type=int,
                       default=4,
                       help='Number of RAPM subprocesses to run concurrently (default: 4).')
    parser.add_argument('--cores-per-rapm',
                       type=int,
                       default=None,
                       help='Optional --cores value passed to each rapm.py subprocess.')
    parser.add_argument('--off-alpha',
                       type=float,
                       default=None,
                       help='Optional offense alpha override passed to every RAPM solve.')
    parser.add_argument('--def-alpha',
                       type=float,
                       default=None,
                       help='Optional defense alpha override passed to every RAPM solve.')
    parser.add_argument('--publish-to-rapms',
                       action='store_true',
                       help='Copy this run\'s standard and alt3 weighted-factor CSVs into /Users/russellthomas/Docs/rapms/master_results, then commit/push rapms if changed.')

    args = parser.parse_args()

    # --half-life implies --timedecay
    if args.half_life is not None:
        args.timedecay = True
    if args.age_dummies and args.age_curve:
        parser.error("--age-dummies cannot be combined with --age-curve.")
    if args.rapm_workers < 1:
        parser.error("--rapm-workers must be at least 1.")
    if args.cores_per_rapm is not None and args.cores_per_rapm < 1:
        parser.error("--cores-per-rapm must be at least 1.")
    if args.off_alpha is not None and args.off_alpha <= 0:
        parser.error("--off-alpha must be positive.")
    if args.def_alpha is not None and args.def_alpha <= 0:
        parser.error("--def-alpha must be positive.")
    
    logging.info("=" * 80)
    logging.info("MASTER RAPM ANALYSIS (Pipeline Version)")
    logging.info("=" * 80)
    logging.info(f"Years: {args.start_year}-{args.end_year}")
    logging.info(f"Season Type: {args.season_type}")
    if args.timedecay:
        td_half = args.half_life if args.half_life else 700
        logging.info(f"Time Decay: ENABLED (half-life={td_half} days)")
    if args.rubberband:
        logging.info(f"Rubberband: ENABLED (alpha_rb={args.alpha_rb if args.alpha_rb else 10})")
    if args.season_effects:
        logging.info("Season Effects: ENABLED")
    if args.age_curve:
        logging.info("Age Curve: ENABLED (RAPM solve only)")
    if args.age_dummies:
        logging.info("Age Fixed Effects: ENABLED (all weighted-factor metric solves)")
    logging.info(f"RAPM subprocess workers: {args.rapm_workers}")
    if args.cores_per_rapm is not None:
        logging.info(f"Per-RAPM load cores: {args.cores_per_rapm}")
    if args.off_alpha is not None or args.def_alpha is not None:
        logging.info(f"Alpha overrides: off={args.off_alpha if args.off_alpha is not None else 3000}, def={args.def_alpha if args.def_alpha is not None else 3000}")
    if args.publish_to_rapms:
        logging.info("Publish to rapms: ENABLED")
    logging.info(f"Output: {RESULTS_DIR}")
    logging.info("=" * 80)

    # Step 1: Run all RAPM types in parallel (6-factor core + auxiliary standard outputs)
    rapm_types = [
        'RAPM', 'TS', 'TOV', 'REB',
        'BADPASS_TOV', 'SCORING_TOV',
        'SECOND_CHANCE', 'FT_PREMIUM',
        *[metric_type for metric_type, _, _ in AUXILIARY_FACTOR_METRICS],
        'FIRST_CHANCE', 'ALT_TS', 'ALT_EFG', 'ALT_FT', 'ALT_TOV', 'ALT_BADPASS_TOV', 'ALT_SCORING_TOV', 'SECOND_CHANCE_CLEAN'
    ]
    if normalize_season_end_year(args.start_year) >= 2024:
        rapm_types.extend(['ALT_SQ', 'ALT_MAKE'])

    logging.info("\nStep 1: Running RAPM analyses in parallel...")
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=args.rapm_workers) as executor:
        futures = {
            executor.submit(run_rapm_analysis, rapm_type, args.start_year, args.end_year, args.season_type,
                            args.timedecay, args.half_life, args.rubberband, args.alpha_rb,
                            args.season_effects, args.age_curve, args.age_dummies,
                            args.cores_per_rapm, args.off_alpha, args.def_alpha): rapm_type
            for rapm_type in rapm_types
        }
        
        results = {}
        for future in as_completed(futures):
            rapm_type, success, output = future.result()
            results[rapm_type] = success
    
    elapsed = time.time() - start_time
    
    # Check if all succeeded
    failures = [rt for rt, success in results.items() if not success]
    
    if failures:
        logging.error(f"\n✗ {len(failures)} analysis(es) failed: {', '.join(failures)}")
        logging.error("Cannot proceed with regression analysis.")
        return 1
    
    logging.info(f"\n✓ All {len(rapm_types)} RAPM analyses completed successfully in {elapsed:.1f}s")
    
    # Use new compact year range format
    year_range = get_year_range_str(args.start_year, args.end_year)
    
    if args.season_type == 'ALL':
        season_suffix = 'all'
    elif args.season_type == 'PS':
        season_suffix = 'ps'
    else:
        season_suffix = 'rs'
    
    # Step 2: Organize results into folder
    output_dir = organize_results(year_range, season_suffix, args.start_year, args.end_year, args.season_type,
                                  args.timedecay, args.half_life, args.rubberband, args.season_effects, args.age_curve, args.age_dummies,
                                  args.off_alpha, args.def_alpha)

    # Step 3: Run regression analysis
    logging.info("\nStep 2: Running regression analysis...")
    regression_success = run_regression_analysis(year_range, season_suffix, output_dir, args.start_year, args.end_year,
                                                 args.timedecay, args.half_life, args.rubberband, args.season_effects, args.age_curve, args.age_dummies,
                                                 args.off_alpha, args.def_alpha)
    
    if not regression_success:
        logging.error("Regression analysis failed.")
        return 1

    weighted_suffix = build_result_suffix(season_suffix, pure=False, rubberband=args.rubberband,
                                          season_effects=args.season_effects,
                                          age_curve=args.age_curve, age_dummies=args.age_dummies,
                                          timedecay=args.timedecay, half_life=args.half_life,
                                          off_alpha=args.off_alpha, def_alpha=args.def_alpha)
    if args.publish_to_rapms:
        try:
            copied_paths, pushed = publish_weighted_factors_to_rapms(year_range, weighted_suffix)
        except Exception as exc:
            logging.error("Publish to rapms failed: %s", exc)
            return 1
        logging.info("Published weighted-factor files to rapms: %s", ", ".join(str(path) for path in copied_paths))
        if pushed:
            logging.info("Downstream rapms repo committed and pushed.")
    
    # Summary
    logging.info("\n" + "=" * 80)
    logging.info("ANALYSIS COMPLETE!")
    logging.info("=" * 80)
    logging.info(f"Results saved to: {output_dir}")
    logging.info(f"Master results: {MASTER_RESULTS_DIR}")
    logging.info("\nGenerated files:")
    rapm_suffix = build_result_suffix(season_suffix, pure=True, rubberband=args.rubberband,
                                      season_effects=args.season_effects,
                                      age_curve=args.age_curve, age_dummies=args.age_dummies, timedecay=args.timedecay, half_life=args.half_life,
                                      off_alpha=args.off_alpha, def_alpha=args.def_alpha)
    other_suffix = build_result_suffix(season_suffix, pure=False, rubberband=args.rubberband,
                                       season_effects=args.season_effects,
                                       age_curve=False, age_dummies=args.age_dummies, timedecay=args.timedecay, half_life=args.half_life,
                                       off_alpha=args.off_alpha, def_alpha=args.def_alpha)
    artifact_suffix = "_agefe" if args.age_dummies else ("_age" if args.age_curve else "")
    logging.info(f"  - rapm_{year_range}_{rapm_suffix}_results.csv")
    logging.info(f"  - ts_{year_range}_{other_suffix}_results.csv")
    logging.info(f"  - tov_{year_range}_{other_suffix}_results.csv")
    logging.info(f"  - reb_{year_range}_{other_suffix}_results.csv")
    logging.info(f"  - ft_premium_{year_range}_{other_suffix}_results.csv")
    for _, file_stem, _ in AUXILIARY_FACTOR_METRICS:
        logging.info(f"  - {file_stem}_{year_range}_{other_suffix}_results.csv")
    logging.info(f"  - first_chance_{year_range}_{other_suffix}_results.csv")
    logging.info(f"  - alt_ts_{year_range}_{other_suffix}_results.csv")
    logging.info(f"  - alt_tov_{year_range}_{other_suffix}_results.csv")
    logging.info(f"  - alt_badpass_tov_{year_range}_{other_suffix}_results.csv")
    logging.info(f"  - alt_scoring_tov_{year_range}_{other_suffix}_results.csv")
    logging.info(f"  - second_chance_clean_{year_range}_{other_suffix}_results.csv")
    logging.info(f"  - regression_net_coefficients{artifact_suffix}.csv")
    logging.info(f"  - regression_off_coefficients{artifact_suffix}.csv")
    logging.info(f"  - regression_def_coefficients{artifact_suffix}.csv")
    logging.info(f"  - regression_alt3_net_coefficients{artifact_suffix}.csv")
    logging.info(f"  - regression_alt3_off_coefficients{artifact_suffix}.csv")
    logging.info(f"  - regression_alt3_def_coefficients{artifact_suffix}.csv")
    logging.info(f"  - alt3_first_chance_decomposition_coefficients{artifact_suffix}.csv")
    logging.info(f"  - tov_decomposition_coefficients{artifact_suffix}.csv")
    logging.info(f"  - weighted_factors_{year_range}_{weighted_suffix}.csv")
    logging.info(f"  - weighted_factors_alt3_{year_range}_{weighted_suffix}.csv")
    logging.info(f"  - predictions_net_all{artifact_suffix}.csv")
    logging.info(f"  - predictions_off_all{artifact_suffix}.csv")
    logging.info(f"  - predictions_def_all{artifact_suffix}.csv")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
