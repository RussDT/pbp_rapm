#!/usr/bin/env python3
"""
Master RAPM Analysis Script

Runs all 4 RAPM types (RAPM, TS, TOV, REB) in parallel, then performs regression analysis.
Results are organized into a dedicated subfolder.

Usage:
    python master_rapm_analysis.py <start_year> <end_year> <season_type>
    
Examples:
    python master_rapm_analysis.py 23 25 ALL
    python master_rapm_analysis.py 24 26 RS
    python master_rapm_analysis.py 25 25 PS
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def run_rapm_analysis(rapm_type, start_year, end_year, season_type):
    """
    Run a single RAPM analysis as a subprocess.
    
    Args:
        rapm_type: One of RAPM, TS, TOV, REB
        start_year: Starting year (2-digit)
        end_year: Ending year (2-digit)
        season_type: RS, PS, or ALL
    
    Returns:
        tuple: (rapm_type, success, output)
    """
    # Build command
    cmd = ['python', 'rapm.py', rapm_type, str(start_year), str(end_year), season_type]
    
    # Add --pure flag for RAPM
    if rapm_type == 'RAPM':
        cmd.append('--pure')
    
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

def run_regression_analysis(year_range, season_suffix, output_dir, start_year, end_year):
    """
    Run regression analysis on the generated RAPM files.
    
    Args:
        year_range: String like "23_24_25" or "25"
        season_suffix: "all", "rs", or "ps"
        output_dir: Directory to save results
        start_year: Starting year (2-digit, e.g., 25 for 2025)
        end_year: Ending year (2-digit, e.g., 26 for 2026)
    """
    logging.info("=" * 80)
    logging.info("RUNNING REGRESSION ANALYSIS")
    logging.info("=" * 80)
    
    # Import the necessary modules
    import pandas as pd
    import numpy as np
    import statsmodels.api as sm
    
    results_dir = Path('Results')
    
    # Determine file suffix based on season type
    if season_suffix == "all":
        suffix = "all_pure_results"
    elif season_suffix == "ps":
        suffix = "ps_pure_results"
    else:
        suffix = "rs_pure_results"
    
    # Build file paths
    # RAPM uses pure_results suffix, others use results suffix
    rapm_file = results_dir / f"rapm_{year_range}_{suffix}.csv"
    
    # For TS, TOV, REB - use "results" instead of "pure_results"
    other_suffix = suffix.replace("pure_results", "results")
    ts_file = results_dir / f"ts_{year_range}_{other_suffix}.csv"
    tov_file = results_dir / f"tov_{year_range}_{other_suffix}.csv"
    reb_file = results_dir / f"reb_{year_range}_{other_suffix}.csv"
    
    # Check that all files exist
    missing_files = []
    for f in [rapm_file, ts_file, tov_file, reb_file]:
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
    
    logging.info(f"  RAPM: {len(rapm_df)} players")
    logging.info(f"  TS: {len(ts_df)} players")
    logging.info(f"  TOV: {len(tov_df)} players")
    logging.info(f"  REB: {len(reb_df)} players")
    
    # Merge datasets
    merged = rapm_df[['player_id', 'player_name', 'off', 'def', 'net_rapm', 'possessions']].copy()
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
        
        coef_data = []
        coef_data.append({
            'variable': 'R_squared',
            'coefficient': results.rsquared,
            'std_error': np.nan,
            'p_value': np.nan,
            'conf_int_lower': np.nan,
            'conf_int_upper': np.nan
        })
        
        coef_data.append({
            'variable': 'Adjusted_R_squared',
            'coefficient': results.rsquared_adj,
            'std_error': np.nan,
            'p_value': np.nan,
            'conf_int_lower': np.nan,
            'conf_int_upper': np.nan
        })
        
        coef_data.append({
            'variable': 'Intercept',
            'coefficient': results.params['const'],
            'std_error': results.bse['const'],
            'p_value': results.pvalues['const'],
            'conf_int_lower': results.conf_int().loc['const', 0],
            'conf_int_upper': results.conf_int().loc['const', 1]
        })
        
        for col in feature_cols:
            coef_data.append({
                'variable': col,
                'coefficient': results.params[col],
                'std_error': results.bse[col],
                'p_value': results.pvalues[col],
                'conf_int_lower': results.conf_int().loc[col, 0],
                'conf_int_upper': results.conf_int().loc[col, 1]
            })
        
        coef_df = pd.DataFrame(coef_data)
        output_path = output_dir / f'regression_{target_name}_coefficients.csv'
        coef_df.to_csv(output_path, index=False)
        logging.info(f"  Saved {target_name} coefficients to {output_path.name}")
    
    # Create weighted factors CSV (using net RAPM coefficients)
    results_net = results_data['net_rapm']
    
    coef_off_ts = results_net.params['off_ts']
    coef_def_ts = results_net.params['def_ts']
    coef_off_tov = results_net.params['off_tov']
    coef_def_tov = results_net.params['def_tov']
    coef_off_reb = results_net.params['off_reb']
    coef_def_reb = results_net.params['def_reb']
    
    weighted_df = merged[['player_id', 'player_name']].copy()
    # Calculate Latest_Year: convert 2-digit end_year to 4-digit year (e.g., 26 -> 2026)
    latest_year = 2000 + end_year
    weighted_df['Latest_Year'] = latest_year
    weighted_df['Latest_Year'] = weighted_df['Latest_Year'].astype(int)
    weighted_df['oTS'] = (merged['off_ts'] * coef_off_ts).round(2)
    weighted_df['oTOV'] = (merged['off_tov'] * coef_off_tov).round(2)
    weighted_df['oREB'] = (merged['off_reb'] * coef_off_reb).round(2)
    weighted_df['dTS'] = (merged['def_ts'] * coef_def_ts).round(2)
    weighted_df['dTOV'] = (merged['def_tov'] * coef_def_tov).round(2)
    weighted_df['dREB'] = (merged['def_reb'] * coef_def_reb).round(2)
    weighted_df['off'] = merged['off_rapm'].round(2)
    weighted_df['def'] = merged['def_rapm'].round(2)
    weighted_df['net_rapm'] = merged['net_rapm'].round(2)
    weighted_df = weighted_df.sort_values('net_rapm', ascending=False)
    
    # Reorder columns to match reference format: player_id, player_name, Latest_Year, then rest
    column_order = ['player_id', 'player_name', 'Latest_Year', 'oTS', 'oTOV', 'oREB', 'dTS', 'dTOV', 'dREB', 'off', 'def', 'net_rapm']
    weighted_df = weighted_df[column_order]
    
    output_path = output_dir / 'weighted_factors.csv'
    weighted_df.to_csv(output_path, index=False)
    logging.info(f"  Saved weighted factors to {output_path.name}")
    
    # Create predictions for all players (for each regression)
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
        
        # Round values
        pred_df[[target_col, pred_col, 'residual']] = pred_df[[target_col, pred_col, 'residual']].round(2)
        
        output_path = output_dir / f'predictions_{target_name}_all.csv'
        pred_df.to_csv(output_path, index=False)
        logging.info(f"  Saved {target_name} predictions to {output_path.name}")
    
    logging.info("\nRegression analysis complete!")
    return True

def organize_results(year_range, season_suffix, start_year, end_year, season_type):
    """
    Organize all generated files into a dedicated folder.
    
    Args:
        year_range: String like "23_24_25"
        season_suffix: "all", "rs", or "ps"
        start_year: Starting year (for folder name)
        end_year: Ending year (for folder name)
        season_type: RS, PS, or ALL
    
    Returns:
        Path to the output directory
    """
    # Create output directory
    if start_year == end_year:
        folder_name = f"rapm_{start_year}_{season_type.lower()}"
    else:
        folder_name = f"rapm_{start_year}_{end_year}_{season_type.lower()}"
    
    output_dir = Path('Results') / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"\nOrganizing results into: {output_dir}")
    
    # Determine file suffix
    if season_suffix == "all":
        suffix = "all_pure_results"
    elif season_suffix == "ps":
        suffix = "ps_pure_results"
    else:
        suffix = "rs_pure_results"
    
    # Move RAPM output files to the folder
    results_dir = Path('Results')
    
    # RAPM uses pure_results suffix, others use results suffix
    other_suffix = suffix.replace("pure_results", "results")
    file_patterns = [
        f"rapm_{year_range}_{suffix}.csv",
        f"ts_{year_range}_{other_suffix}.csv",
        f"tov_{year_range}_{other_suffix}.csv",
        f"reb_{year_range}_{other_suffix}.csv"
    ]
    
    for pattern in file_patterns:
        source = results_dir / pattern
        if source.exists():
            dest = output_dir / pattern
            shutil.copy2(source, dest)
            logging.info(f"  Copied {pattern}")
    
    return output_dir

def main():
    parser = argparse.ArgumentParser(
        description='Master RAPM Analysis - Run all 4 types and regression',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Multi-year all seasons (2023-2025)
  python master_rapm_analysis.py 23 25 ALL
  
  # Single year regular season (2024-25)
  python master_rapm_analysis.py 25 25 RS
  
  # Multi-year playoffs (2024-2026)
  python master_rapm_analysis.py 24 26 PS
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
    
    args = parser.parse_args()
    
    logging.info("=" * 80)
    logging.info("MASTER RAPM ANALYSIS")
    logging.info("=" * 80)
    logging.info(f"Years: {args.start_year}-{args.end_year}")
    logging.info(f"Season Type: {args.season_type}")
    logging.info("=" * 80)
    
    # Step 1: Run all 4 RAPM types in parallel
    rapm_types = ['RAPM', 'TS', 'TOV', 'REB']
    
    logging.info("\nStep 1: Running RAPM analyses in parallel...")
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(run_rapm_analysis, rapm_type, args.start_year, args.end_year, args.season_type): rapm_type
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
    
    logging.info(f"\n✓ All 4 RAPM analyses completed successfully in {elapsed:.1f}s")
    
    # Determine year range and season suffix for file names
    years = list(range(args.start_year, args.end_year + 1))
    if len(years) == 1:
        year_range = f"{years[0]:02d}"
    else:
        year_range = '_'.join([f"{y:02d}" for y in years])
    
    if args.season_type == 'ALL':
        season_suffix = 'all'
    elif args.season_type == 'PS':
        season_suffix = 'ps'
    else:
        season_suffix = 'rs'
    
    # Step 2: Organize results into folder
    output_dir = organize_results(year_range, season_suffix, args.start_year, args.end_year, args.season_type)
    
    # Step 3: Run regression analysis
    logging.info("\nStep 2: Running regression analysis...")
    regression_success = run_regression_analysis(year_range, season_suffix, output_dir, args.start_year, args.end_year)
    
    if not regression_success:
        logging.error("Regression analysis failed.")
        return 1
    
    # Summary
    logging.info("\n" + "=" * 80)
    logging.info("ANALYSIS COMPLETE!")
    logging.info("=" * 80)
    logging.info(f"Results saved to: {output_dir}")
    logging.info("\nGenerated files:")
    logging.info(f"  - rapm_{year_range}_{season_suffix}_pure_results.csv")
    other_suffix = season_suffix.replace("pure_", "")
    logging.info(f"  - ts_{year_range}_{other_suffix}_results.csv")
    logging.info(f"  - tov_{year_range}_{other_suffix}_results.csv")
    logging.info(f"  - reb_{year_range}_{other_suffix}_results.csv")
    logging.info(f"  - regression_net_coefficients.csv")
    logging.info(f"  - regression_off_coefficients.csv")
    logging.info(f"  - regression_def_coefficients.csv")
    logging.info(f"  - weighted_factors.csv")
    logging.info(f"  - predictions_net_all.csv")
    logging.info(f"  - predictions_off_all.csv")
    logging.info(f"  - predictions_def_all.csv")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

