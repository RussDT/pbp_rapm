#!/usr/bin/env python3
"""
Prior-Informed RAPM Analysis Script

This script:
1. Fetches RFTOV (relative forced turnovers) data from Supabase
2. Analyzes how predictive RFTOV is of future DTOV (defensive turnover RAPM)
3. Runs a prior-informed RAPM for defensive turnovers using RFTOV as a prior

Usage:
    # Analyze RFTOV predictiveness
    python prior_informed_rapm.py analyze 15 25

    # Run prior-informed DTOV RAPM
    python prior_informed_rapm.py run 23 26 ALL --prior-weight 0.3
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.linear_model import Ridge
from dotenv import load_dotenv
from supabase import create_client, Client

# Get the pipeline root directory
SCRIPT_DIR = Path(__file__).parent
PIPELINE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = PIPELINE_ROOT.parent
PROCESSED_DIR = PIPELINE_ROOT / "processed"
RESULTS_DIR = PIPELINE_ROOT / "results"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load environment variables
load_dotenv()

###############################################################################
# POSITION GROUPING
###############################################################################

def get_position_group(pos2):
    """
    Map positions to 3 groups: Guard, Wing, Big.

    Args:
        pos2: Position string from Supabase (e.g., 'PG', 'SF', 'C', 'PG,SG')

    Returns:
        'Guard', 'Wing', 'Big', or None
    """
    if pd.isna(pos2) or pos2 == 'nan' or pos2 is None:
        return None
    # Take first position if combo (e.g., "PG,SG" -> "PG")
    primary = str(pos2).replace('-', ',').split(',')[0].strip()
    if primary in ['PG', 'SG']:
        return 'Guard'
    elif primary == 'SF':
        return 'Wing'
    elif primary in ['PF', 'C']:
        return 'Big'
    return None


###############################################################################
# SUPABASE DATA FETCHING
###############################################################################

def fetch_rftov_data(start_year: int, end_year: int, playoffs: int = 0) -> pd.DataFrame:
    """
    Fetch RFTOV data from Supabase for the specified year range.

    Args:
        start_year: Starting year (4-digit, e.g., 2015)
        end_year: Ending year (4-digit, e.g., 2025)
        playoffs: 0 for regular season, 1 for playoffs

    Returns:
        DataFrame with columns: nba_id, year, rFTOV_100, Pos2
    """
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")

    if not url or not key:
        logging.error("SUPABASE_URL or SUPABASE_KEY missing in .env")
        return None

    try:
        supabase: Client = create_client(url, key)

        all_data = []
        for year in range(start_year, end_year + 1):
            logging.info(f"Fetching RFTOV data for year {year}...")

            response = supabase.table("player_stats") \
                .select('nba_id, year, "rFTOV_100", "Pos2", playoffs') \
                .eq("year", year) \
                .eq("playoffs", playoffs) \
                .execute()

            if response.data:
                df_year = pd.DataFrame(response.data)
                all_data.append(df_year)
                logging.info(f"  Fetched {len(df_year)} players for year {year}")

        if all_data:
            df = pd.concat(all_data, ignore_index=True)
            logging.info(f"Total: {len(df)} player-years fetched")
            return df
        else:
            logging.warning("No data fetched from Supabase")
            return None

    except Exception as e:
        logging.error(f"Error fetching RFTOV data: {e}")
        return None


###############################################################################
# PREDICTIVENESS ANALYSIS
###############################################################################

def analyze_rftov_predictiveness(start_year: int, end_year: int):
    """
    Analyze how predictive RFTOV is of future DTOV.

    This creates year-over-year correlations and regression analysis.
    """
    logging.info("=" * 60)
    logging.info("RFTOV PREDICTIVENESS ANALYSIS")
    logging.info("=" * 60)

    # Fetch RFTOV data
    rftov_df = fetch_rftov_data(start_year, end_year + 1, playoffs=0)
    if rftov_df is None:
        return None

    # Clean and prepare RFTOV data
    rftov_df = rftov_df.dropna(subset=['rFTOV_100'])
    rftov_df['nba_id'] = rftov_df['nba_id'].astype(int)
    rftov_df['year'] = rftov_df['year'].astype(int)

    # Load DTOV results for each year
    dtov_data = []
    for year in range(start_year, end_year + 2):  # +2 to include year+1 for predictions
        year_2digit = year % 100

        # Try to find single-year TOV file (handle single digit years)
        tov_file = RESULTS_DIR / f"tov_{year_2digit}_rs_results.csv"
        if not tov_file.exists():
            tov_file = RESULTS_DIR / f"tov_{year_2digit:02d}_rs_results.csv"
        if not tov_file.exists():
            tov_file = RESULTS_DIR / f"tov_{year_2digit}_all_results.csv"
        if not tov_file.exists():
            tov_file = RESULTS_DIR / f"tov_{year_2digit:02d}_all_results.csv"

        if tov_file.exists():
            df = pd.read_csv(tov_file)
            df['year'] = year
            df = df.rename(columns={'player_id': 'nba_id', 'def': 'dtov_rapm'})
            dtov_data.append(df[['nba_id', 'year', 'dtov_rapm', 'def_poss']])
            logging.info(f"Loaded DTOV for {year}: {len(df)} players from {tov_file.name}")
        else:
            logging.warning(f"DTOV file not found for year {year}")

    if not dtov_data:
        logging.warning("No DTOV files found. Running single-year TOV analyses...")
        # We need to run the TOV analysis for each year
        return analyze_with_generated_dtov(rftov_df, start_year, end_year)

    dtov_df = pd.concat(dtov_data, ignore_index=True)
    dtov_df['nba_id'] = dtov_df['nba_id'].astype(int)

    # Create year-over-year analysis
    results = []

    for year in range(start_year, end_year):
        next_year = year + 1

        # Get RFTOV for current year
        rftov_current = rftov_df[rftov_df['year'] == year][['nba_id', 'rFTOV_100', 'Pos2']]
        rftov_current = rftov_current.rename(columns={'rFTOV_100': 'rftov_current'})

        # Get DTOV for next year
        dtov_next = dtov_df[dtov_df['year'] == next_year][['nba_id', 'dtov_rapm', 'def_poss']]
        dtov_next = dtov_next.rename(columns={'dtov_rapm': 'dtov_next', 'def_poss': 'poss_next'})

        # Merge
        merged = rftov_current.merge(dtov_next, on='nba_id', how='inner')

        if len(merged) < 50:
            logging.warning(f"Year {year}->{next_year}: Only {len(merged)} players matched, skipping")
            continue

        # Calculate correlation
        corr = merged['rftov_current'].corr(merged['dtov_next'])

        # Weighted correlation (by possessions)
        if 'poss_next' in merged.columns:
            weighted_mean_x = np.average(merged['rftov_current'], weights=merged['poss_next'])
            weighted_mean_y = np.average(merged['dtov_next'], weights=merged['poss_next'])
            weighted_cov = np.average(
                (merged['rftov_current'] - weighted_mean_x) * (merged['dtov_next'] - weighted_mean_y),
                weights=merged['poss_next']
            )
            weighted_std_x = np.sqrt(np.average((merged['rftov_current'] - weighted_mean_x)**2, weights=merged['poss_next']))
            weighted_std_y = np.sqrt(np.average((merged['dtov_next'] - weighted_mean_y)**2, weights=merged['poss_next']))
            weighted_corr = weighted_cov / (weighted_std_x * weighted_std_y)
        else:
            weighted_corr = corr

        results.append({
            'year_from': year,
            'year_to': next_year,
            'n_players': len(merged),
            'correlation': round(corr, 4),
            'weighted_correlation': round(weighted_corr, 4)
        })

        logging.info(f"Year {year}->{next_year}: n={len(merged)}, corr={corr:.4f}, weighted_corr={weighted_corr:.4f}")

    results_df = pd.DataFrame(results)

    # Summary statistics
    logging.info("\n" + "=" * 60)
    logging.info("SUMMARY")
    logging.info("=" * 60)
    logging.info(f"Average correlation: {results_df['correlation'].mean():.4f}")
    logging.info(f"Average weighted correlation: {results_df['weighted_correlation'].mean():.4f}")
    logging.info(f"Min correlation: {results_df['correlation'].min():.4f}")
    logging.info(f"Max correlation: {results_df['correlation'].max():.4f}")

    # Save results
    output_file = RESULTS_DIR / "rftov_dtov_predictiveness.csv"
    results_df.to_csv(output_file, index=False)
    logging.info(f"\nSaved predictiveness analysis to {output_file}")

    return results_df


def analyze_with_generated_dtov(rftov_df, start_year, end_year):
    """
    Generate single-year DTOV and analyze predictiveness.
    This is called when pre-computed DTOV files don't exist.
    """
    logging.info("Generating single-year DTOV analyses...")

    from rapm import run_simplified_rapm

    dtov_data = []
    for year in range(start_year, end_year + 1):
        year_2digit = year % 100
        tov_file = f"TOV{year_2digit:02d}.parquet"

        if (PROCESSED_DIR / tov_file).exists():
            logging.info(f"Running TOV analysis for {year}...")
            results = run_simplified_rapm([tov_file])
            if results is not None:
                results['year'] = year
                results = results.rename(columns={'player_id': 'nba_id', 'def': 'dtov_rapm'})
                dtov_data.append(results[['nba_id', 'year', 'dtov_rapm', 'def_poss']])

    if dtov_data:
        dtov_df = pd.concat(dtov_data, ignore_index=True)
        # Continue with analysis...
        return analyze_rftov_predictiveness_with_data(rftov_df, dtov_df, start_year, end_year)

    return None


def analyze_rftov_predictiveness_with_data(rftov_df, dtov_df, start_year, end_year):
    """Analyze predictiveness with provided data."""
    # Similar logic to analyze_rftov_predictiveness but with provided data
    pass


###############################################################################
# IN-SAMPLE CORRELATION ANALYSIS
###############################################################################

def analyze_insample_correlation(start_year: int, end_year: int, pos_adjust: bool = True):
    """
    Compute in-sample correlation: multi-year RFTOV vs multi-year DTOV (same period).

    This tests how well box-score RFTOV correlates with lineup-based DTOV
    in-sample, validating that they measure the same underlying skill.

    Args:
        start_year: First year of window (4-digit, e.g., 2021)
        end_year: Last year of window (4-digit, e.g., 2025)
        pos_adjust: If True, recenter RFTOV by position group before analysis

    Returns:
        DataFrame with correlation results
    """
    logging.info("=" * 60)
    logging.info(f"IN-SAMPLE CORRELATION ANALYSIS ({start_year}-{end_year})")
    logging.info(f"Position adjustment: {pos_adjust}")
    logging.info("=" * 60)

    # Fetch RFTOV data
    rftov_df = fetch_rftov_data(start_year, end_year, playoffs=0)
    if rftov_df is None:
        return None

    # Clean and prepare RFTOV data
    rftov_df = rftov_df.dropna(subset=['rFTOV_100'])
    rftov_df['nba_id'] = rftov_df['nba_id'].astype(int)
    rftov_df['year'] = rftov_df['year'].astype(int)

    # Add position groups
    rftov_df['pos_group'] = rftov_df['Pos2'].apply(get_position_group)
    logging.info(f"Position distribution:")
    logging.info(f"  {rftov_df['pos_group'].value_counts().to_dict()}")

    # Positional adjustment: subtract position group mean per year
    if pos_adjust:
        # Calculate position means per year
        pos_means = rftov_df.groupby(['year', 'pos_group'])['rFTOV_100'].transform('mean')
        rftov_df['rFTOV_adj'] = rftov_df['rFTOV_100'] - pos_means
        logging.info("Applied positional adjustment (subtracted position mean per year)")
    else:
        rftov_df['rFTOV_adj'] = rftov_df['rFTOV_100']

    # Average RFTOV per player across the window
    rftov_avg = rftov_df.groupby('nba_id').agg({
        'rFTOV_adj': 'mean',
        'rFTOV_100': 'mean',  # Also keep raw for comparison
        'pos_group': 'first'
    }).reset_index()
    rftov_avg = rftov_avg.rename(columns={
        'rFTOV_adj': 'rftov_adj_avg',
        'rFTOV_100': 'rftov_raw_avg'
    })

    # Load multi-year DTOV for the same period
    year_start_2digit = start_year % 100
    year_end_2digit = end_year % 100

    # Try to find existing multi-year TOV file
    dtov_file = RESULTS_DIR / f"tov_{year_start_2digit}_{year_end_2digit}_all_results.csv"
    if not dtov_file.exists():
        dtov_file = RESULTS_DIR / f"tov_{year_start_2digit}_{year_end_2digit}_rs_results.csv"

    if dtov_file.exists():
        dtov_df = pd.read_csv(dtov_file)
        logging.info(f"Loaded DTOV from {dtov_file.name}: {len(dtov_df)} players")
    else:
        logging.warning(f"DTOV file not found: {dtov_file}")
        logging.info("Run: python rapm.py TOV {start} {end} ALL first")
        return None

    dtov_df = dtov_df.rename(columns={'player_id': 'nba_id', 'def': 'dtov_rapm'})
    dtov_df['nba_id'] = dtov_df['nba_id'].astype(int)

    # Merge RFTOV and DTOV
    merged = rftov_avg.merge(dtov_df[['nba_id', 'dtov_rapm', 'def_poss']], on='nba_id', how='inner')
    logging.info(f"Matched {len(merged)} players with both RFTOV and DTOV")

    # Filter to players with minimum possessions
    min_poss = 1000
    merged_filtered = merged[merged['def_poss'] >= min_poss].copy()
    logging.info(f"Players with {min_poss}+ possessions: {len(merged_filtered)}")

    # Compute correlations
    results = {}

    # Overall correlations
    corr_raw = merged_filtered['rftov_raw_avg'].corr(merged_filtered['dtov_rapm'])
    corr_adj = merged_filtered['rftov_adj_avg'].corr(merged_filtered['dtov_rapm'])

    # Weighted correlations (by possessions)
    weights = merged_filtered['def_poss'].values
    def weighted_corr(x, y, w):
        """Compute weighted Pearson correlation."""
        mx = np.average(x, weights=w)
        my = np.average(y, weights=w)
        cov = np.average((x - mx) * (y - my), weights=w)
        sx = np.sqrt(np.average((x - mx)**2, weights=w))
        sy = np.sqrt(np.average((y - my)**2, weights=w))
        return cov / (sx * sy)

    wcorr_raw = weighted_corr(
        merged_filtered['rftov_raw_avg'].values,
        merged_filtered['dtov_rapm'].values,
        weights
    )
    wcorr_adj = weighted_corr(
        merged_filtered['rftov_adj_avg'].values,
        merged_filtered['dtov_rapm'].values,
        weights
    )

    logging.info("\n" + "=" * 60)
    logging.info("OVERALL CORRELATIONS")
    logging.info("=" * 60)
    logging.info(f"  n = {len(merged_filtered)} players")
    logging.info(f"  Raw RFTOV:          corr = {corr_raw:.4f}, weighted = {wcorr_raw:.4f}")
    logging.info(f"  Pos-adjusted RFTOV: corr = {corr_adj:.4f}, weighted = {wcorr_adj:.4f}")

    results['overall'] = {
        'n': len(merged_filtered),
        'corr_raw': round(corr_raw, 4),
        'corr_adj': round(corr_adj, 4),
        'wcorr_raw': round(wcorr_raw, 4),
        'wcorr_adj': round(wcorr_adj, 4)
    }

    # By position group
    logging.info("\nBY POSITION GROUP:")
    for pos in ['Guard', 'Wing', 'Big']:
        pos_df = merged_filtered[merged_filtered['pos_group'] == pos]
        if len(pos_df) < 20:
            logging.info(f"  {pos}: n={len(pos_df)} (too few for reliable correlation)")
            continue

        pos_corr_raw = pos_df['rftov_raw_avg'].corr(pos_df['dtov_rapm'])
        pos_corr_adj = pos_df['rftov_adj_avg'].corr(pos_df['dtov_rapm'])

        if len(pos_df) >= 20:
            pos_wcorr_adj = weighted_corr(
                pos_df['rftov_adj_avg'].values,
                pos_df['dtov_rapm'].values,
                pos_df['def_poss'].values
            )
        else:
            pos_wcorr_adj = pos_corr_adj

        logging.info(f"  {pos:6s}: n={len(pos_df):3d}, raw={pos_corr_raw:+.4f}, adj={pos_corr_adj:+.4f}, w_adj={pos_wcorr_adj:+.4f}")

        results[pos] = {
            'n': len(pos_df),
            'corr_raw': round(pos_corr_raw, 4),
            'corr_adj': round(pos_corr_adj, 4),
            'wcorr_adj': round(pos_wcorr_adj, 4)
        }

    # Save detailed results
    output_file = RESULTS_DIR / f"rftov_dtov_insample_{year_start_2digit}_{year_end_2digit}.csv"
    merged_filtered.to_csv(output_file, index=False)
    logging.info(f"\nSaved detailed results to {output_file}")

    return results


###############################################################################
# RESIDUAL PERSISTENCE ANALYSIS
###############################################################################

def analyze_residual_persistence(
    train_start: int, train_end: int,
    test_start: int, test_end: int,
    min_poss: int = 5000
):
    """
    Test whether DTOV residual (beyond RFTOV) persists over time.

    If residual persists, there's "hidden skill" not captured by box score.
    If residual doesn't persist, RFTOV tells you everything.

    Args:
        train_start, train_end: Training period years (4-digit)
        test_start, test_end: Test period years (4-digit)
        min_poss: Minimum possessions required in EACH period

    Returns:
        dict with correlation results and player-level data
    """
    logging.info("=" * 60)
    logging.info(f"RESIDUAL PERSISTENCE TEST")
    logging.info(f"Train: {train_start}-{train_end} → Test: {test_start}-{test_end}")
    logging.info(f"Min possessions: {min_poss} in each period")
    logging.info("=" * 60)

    # Load DTOV results for both periods
    train_dtov_file = RESULTS_DIR / f"tov_{train_start % 100}_{train_end % 100}_all_results.csv"
    test_dtov_file = RESULTS_DIR / f"tov_{test_start % 100}_{test_end % 100}_all_results.csv"

    if not train_dtov_file.exists():
        logging.error(f"Train DTOV file not found: {train_dtov_file}")
        return None
    if not test_dtov_file.exists():
        logging.error(f"Test DTOV file not found: {test_dtov_file}")
        return None

    train_dtov = pd.read_csv(train_dtov_file)
    test_dtov = pd.read_csv(test_dtov_file)

    train_dtov = train_dtov.rename(columns={'player_id': 'nba_id', 'def': 'dtov'})
    test_dtov = test_dtov.rename(columns={'player_id': 'nba_id', 'def': 'dtov'})

    logging.info(f"Loaded train DTOV: {len(train_dtov)} players")
    logging.info(f"Loaded test DTOV: {len(test_dtov)} players")

    # Fetch RFTOV for both periods
    train_rftov = fetch_rftov_data(train_start, train_end, playoffs=0)
    test_rftov = fetch_rftov_data(test_start, test_end, playoffs=0)

    if train_rftov is None or test_rftov is None:
        logging.error("Could not fetch RFTOV data")
        return None

    # Average RFTOV per player
    train_rftov_avg = train_rftov.groupby('nba_id')['rFTOV_100'].mean().reset_index()
    test_rftov_avg = test_rftov.groupby('nba_id')['rFTOV_100'].mean().reset_index()

    train_rftov_avg = train_rftov_avg.rename(columns={'rFTOV_100': 'rftov'})
    test_rftov_avg = test_rftov_avg.rename(columns={'rFTOV_100': 'rftov'})

    # Merge DTOV and RFTOV for each period
    train_df = train_dtov[['nba_id', 'dtov', 'def_poss']].merge(
        train_rftov_avg, on='nba_id', how='inner'
    )
    test_df = test_dtov[['nba_id', 'dtov', 'def_poss']].merge(
        test_rftov_avg, on='nba_id', how='inner'
    )

    logging.info(f"Train period: {len(train_df)} players with both DTOV and RFTOV")
    logging.info(f"Test period: {len(test_df)} players with both DTOV and RFTOV")

    # Find scaling factor β: DTOV ~ β × RFTOV
    # Use train period to fit β
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(
        train_df['rftov'], train_df['dtov']
    )
    beta = slope
    logging.info(f"Scaling factor β = {beta:.4f} (DTOV = β × RFTOV)")
    logging.info(f"  In-sample R² = {r_value**2:.4f}")

    # Calculate residuals for both periods
    train_df['predicted_dtov'] = beta * train_df['rftov']
    train_df['residual'] = train_df['dtov'] - train_df['predicted_dtov']

    test_df['predicted_dtov'] = beta * test_df['rftov']
    test_df['residual'] = test_df['dtov'] - test_df['predicted_dtov']

    # Merge train and test residuals for same players
    train_df = train_df.rename(columns={
        'dtov': 'train_dtov', 'rftov': 'train_rftov',
        'def_poss': 'train_poss', 'residual': 'train_residual',
        'predicted_dtov': 'train_predicted'
    })
    test_df = test_df.rename(columns={
        'dtov': 'test_dtov', 'rftov': 'test_rftov',
        'def_poss': 'test_poss', 'residual': 'test_residual',
        'predicted_dtov': 'test_predicted'
    })

    merged = train_df.merge(test_df, on='nba_id', how='inner')
    logging.info(f"Players in both periods: {len(merged)}")

    # Filter by minimum possessions in BOTH periods
    merged_filtered = merged[
        (merged['train_poss'] >= min_poss) &
        (merged['test_poss'] >= min_poss)
    ].copy()
    logging.info(f"Players with {min_poss}+ poss in both: {len(merged_filtered)}")

    if len(merged_filtered) < 20:
        logging.warning("Too few players for reliable analysis")
        return None

    # Calculate residual persistence correlation
    residual_corr = merged_filtered['train_residual'].corr(merged_filtered['test_residual'])

    # Also calculate DTOV stability (for comparison)
    dtov_corr = merged_filtered['train_dtov'].corr(merged_filtered['test_dtov'])

    # And RFTOV stability
    rftov_corr = merged_filtered['train_rftov'].corr(merged_filtered['test_rftov'])

    logging.info("\n" + "=" * 60)
    logging.info("RESULTS")
    logging.info("=" * 60)
    logging.info(f"Players analyzed: {len(merged_filtered)}")
    logging.info(f"")
    logging.info(f"Stability correlations (train → test):")
    logging.info(f"  RFTOV:         r = {rftov_corr:.3f}")
    logging.info(f"  DTOV:          r = {dtov_corr:.3f}")
    logging.info(f"  DTOV residual: r = {residual_corr:.3f}  <-- KEY METRIC")
    logging.info(f"")

    if residual_corr > 0.3:
        logging.info("INTERPRETATION: Hidden skill EXISTS!")
        logging.info("  Some players consistently outperform their RFTOV.")
    elif residual_corr > 0.15:
        logging.info("INTERPRETATION: Weak evidence of hidden skill.")
        logging.info("  Some signal, but mostly noise.")
    else:
        logging.info("INTERPRETATION: No hidden skill detected.")
        logging.info("  RFTOV tells you everything about DTOV.")

    # Load name map for player identification
    name_map_file = PROJECT_ROOT / "autocomplete_map.csv"
    try:
        name_df = pd.read_csv(name_map_file)
        name_map = dict(zip(name_df['nba_id'].astype(int), name_df['player_name']))
    except:
        name_map = {}

    merged_filtered['player_name'] = merged_filtered['nba_id'].map(
        lambda x: name_map.get(int(x), f"ID_{x}")
    )

    # Show top and bottom residual players
    logging.info("\n" + "=" * 60)
    logging.info("TOP 10 POSITIVE RESIDUAL (outperform RFTOV)")
    logging.info("=" * 60)
    top_residual = merged_filtered.nlargest(10, 'train_residual')
    for _, row in top_residual.iterrows():
        logging.info(
            f"  {row['player_name']:25s} | "
            f"Train res: {row['train_residual']:+.2f} | "
            f"Test res: {row['test_residual']:+.2f}"
        )

    logging.info("\n" + "=" * 60)
    logging.info("TOP 10 NEGATIVE RESIDUAL (underperform RFTOV)")
    logging.info("=" * 60)
    bottom_residual = merged_filtered.nsmallest(10, 'train_residual')
    for _, row in bottom_residual.iterrows():
        logging.info(
            f"  {row['player_name']:25s} | "
            f"Train res: {row['train_residual']:+.2f} | "
            f"Test res: {row['test_residual']:+.2f}"
        )

    return {
        'n_players': len(merged_filtered),
        'beta': beta,
        'residual_corr': residual_corr,
        'dtov_corr': dtov_corr,
        'rftov_corr': rftov_corr,
        'player_data': merged_filtered
    }


def run_residual_persistence_all_windows(min_poss: int = 5000):
    """
    Run residual persistence test across all available windows.
    """
    windows = [
        (2016, 2018, 2019, 2021),  # Window 1
        (2017, 2019, 2020, 2022),  # Window 2
        (2020, 2022, 2023, 2025),  # Window 3
    ]

    all_results = []

    for train_start, train_end, test_start, test_end in windows:
        logging.info("\n" + "#" * 70)
        logging.info(f"# WINDOW: {train_start}-{train_end} → {test_start}-{test_end}")
        logging.info("#" * 70 + "\n")

        result = analyze_residual_persistence(
            train_start, train_end, test_start, test_end, min_poss
        )

        if result:
            all_results.append({
                'window': f"{train_start}-{train_end} → {test_start}-{test_end}",
                'n_players': result['n_players'],
                'residual_corr': result['residual_corr'],
                'dtov_corr': result['dtov_corr'],
                'rftov_corr': result['rftov_corr']
            })

    # Summary
    if all_results:
        logging.info("\n" + "=" * 70)
        logging.info("SUMMARY ACROSS ALL WINDOWS")
        logging.info("=" * 70)

        avg_residual_corr = np.mean([r['residual_corr'] for r in all_results])
        avg_dtov_corr = np.mean([r['dtov_corr'] for r in all_results])
        avg_rftov_corr = np.mean([r['rftov_corr'] for r in all_results])

        for r in all_results:
            logging.info(
                f"  {r['window']:25s} | n={r['n_players']:3d} | "
                f"residual r={r['residual_corr']:.3f}"
            )

        logging.info(f"")
        logging.info(f"Average residual correlation: {avg_residual_corr:.3f}")
        logging.info(f"Average DTOV correlation:     {avg_dtov_corr:.3f}")
        logging.info(f"Average RFTOV correlation:    {avg_rftov_corr:.3f}")

        if avg_residual_corr > 0.25:
            logging.info("\nCONCLUSION: Hidden non-box-score skill EXISTS!")
        elif avg_residual_corr > 0.1:
            logging.info("\nCONCLUSION: Weak evidence of hidden skill.")
        else:
            logging.info("\nCONCLUSION: No hidden skill. RFTOV captures everything.")

    return all_results


###############################################################################
# GENERATE PRIOR CSV FOR RAPM.PY
###############################################################################

def generate_rftov_prior(start_year: int, end_year: int, output_file: str):
    """
    Generate RFTOV prior CSV for use with rapm.py --def-prior.

    Creates a CSV with columns: player_id, prior_value
    where prior_value is negated RFTOV (since RFTOV positive = good,
    but DTOV negative = good for forcing turnovers).

    Args:
        start_year: Starting year (4-digit, e.g., 2023)
        end_year: Ending year (4-digit, e.g., 2026)
        output_file: Output CSV path
    """
    logging.info("=" * 60)
    logging.info(f"GENERATING RFTOV PRIOR FOR rapm.py")
    logging.info(f"Years: {start_year}-{end_year}")
    logging.info("=" * 60)

    # Fetch RFTOV data from Supabase
    rftov_df = fetch_rftov_data(start_year, end_year, playoffs=0)
    if rftov_df is None:
        logging.error("Could not fetch RFTOV data")
        return None

    # Clean and average across years
    rftov_df = rftov_df.dropna(subset=['rFTOV_100'])
    rftov_df['nba_id'] = rftov_df['nba_id'].astype(int)

    # Average RFTOV per player across the window
    rftov_avg = rftov_df.groupby('nba_id')['rFTOV_100'].mean()

    # Negate: RFTOV positive = good, DTOV negative = good
    # So prior_value = -RFTOV to align with DTOV sign convention
    prior_df = pd.DataFrame({
        'player_id': rftov_avg.index,
        'prior_value': -rftov_avg.values  # Negated
    })

    # Sort by player_id for consistency
    prior_df = prior_df.sort_values('player_id').reset_index(drop=True)

    # Save
    output_path = Path(output_file)
    if not output_path.is_absolute():
        output_path = RESULTS_DIR / output_file

    prior_df.to_csv(output_path, index=False)
    logging.info(f"Saved prior for {len(prior_df)} players to {output_path}")

    # Show summary stats
    logging.info(f"Prior value stats:")
    logging.info(f"  Mean: {prior_df['prior_value'].mean():.3f}")
    logging.info(f"  Std:  {prior_df['prior_value'].std():.3f}")
    logging.info(f"  Min:  {prior_df['prior_value'].min():.3f}")
    logging.info(f"  Max:  {prior_df['prior_value'].max():.3f}")

    return prior_df


###############################################################################
# PRIOR-INFORMED RAPM
###############################################################################

def load_tov_data(input_files):
    """Load and prepare TOV data from processed parquet files."""
    all_dfs = []

    for f in input_files:
        f_path = Path(f)
        if not f_path.is_absolute():
            f_path = PROCESSED_DIR / f

        if not f_path.exists():
            logging.warning(f"File not found: {f_path}")
            continue

        logging.info(f"Loading {f_path}...")
        df = pd.read_parquet(f_path)

        # Prepare Off_Diff and Def_Diff for TOV
        df['Off_Diff'] = -df['Is_Turnover']  # Negative for offense (bad)
        df['Def_Diff'] = df['Is_Turnover']   # Positive for defense (good)

        all_dfs.append(df)
        logging.info(f"  Loaded {len(df)} possessions")

    if not all_dfs:
        return None

    return pd.concat(all_dfs, ignore_index=True)


def transform_to_home_away(df):
    """Transform O/D format to home/away format."""
    logging.info("Transforming O/D format to home/away format...")

    transformed_rows = []

    for game_id in df['game_id'].unique():
        game_df = df[df['game_id'] == game_id].copy().reset_index(drop=True)

        away_players = set()
        home_players = set()

        # First pass: identify home vs away
        for i in range(len(game_df)):
            row = game_df.iloc[i]
            off_players = [row[f'O{j}'] for j in range(1, 6)]
            def_players = [row[f'D{j}'] for j in range(1, 6)]

            if i == 0:
                away_pts = row['away_score']
                home_pts = row['home_score']
            else:
                prev = game_df.iloc[i-1]
                away_pts = row['away_score'] - prev['away_score']
                home_pts = row['home_score'] - prev['home_score']

            if away_pts > 0:
                away_players.update(off_players)
                home_players.update(def_players)
            elif home_pts > 0:
                home_players.update(off_players)
                away_players.update(def_players)

        # Second pass: create transformed rows
        for i in range(len(game_df)):
            row = game_df.iloc[i]
            off_players = [row[f'O{j}'] for j in range(1, 6)]
            def_players = [row[f'D{j}'] for j in range(1, 6)]
            pts = float(row['Off_Diff'])

            off_is_home = len(set(off_players) & home_players) > len(set(off_players) & away_players)

            transformed_row = {
                'home_poss': 1 if off_is_home else 0,
                'pts': pts,
                'season': int(row['Season']),
                'date': row['game_date'],
                'period': int(row['period']),
                'gameid': str(row['game_id'])
            }

            if off_is_home:
                for j in range(5):
                    transformed_row[f'h{j+1}'] = int(off_players[j])
                    transformed_row[f'a{j+1}'] = int(def_players[j])
            else:
                for j in range(5):
                    transformed_row[f'a{j+1}'] = int(off_players[j])
                    transformed_row[f'h{j+1}'] = int(def_players[j])

            transformed_rows.append(transformed_row)

    result_df = pd.DataFrame(transformed_rows)
    logging.info(f"Transformed {len(result_df)} possessions")
    return result_df


def build_design_matrix_defense_only(df, player_to_col):
    """
    Build sparse design matrix for defense-only RAPM.
    Only includes defensive player indicators.
    """
    logging.info("Building defense-only design matrix...")

    n_samples = len(df)
    n_features = len(player_to_col)

    a_players = df[[f'a{k}' for k in range(1, 6)]].values.astype(int)
    h_players = df[[f'h{k}' for k in range(1, 6)]].values.astype(int)
    home_poss = df['home_poss'].values
    pts = df['pts'].values.astype(np.float64)

    # Defense is the opposite of offense
    defense_players = np.where(home_poss[:, None] == 1, a_players, h_players)

    row_indices = []
    col_indices = []

    for player_idx in range(5):
        player_ids = defense_players[:, player_idx]
        for i, pid in enumerate(player_ids):
            def_key = f"{pid}_def"
            if def_key in player_to_col:
                row_indices.append(i)
                col_indices.append(player_to_col[def_key])

    data = np.ones(len(row_indices), dtype=np.float64)
    X = coo_matrix((data, (row_indices, col_indices)),
                   shape=(n_samples, n_features), dtype=np.float64)

    logging.info(f"Defense design matrix shape: {X.shape}")
    return X.tocsr(), pts


def run_prior_informed_dtov(
    input_files,
    start_year: int,
    end_year: int,
    prior_weight: float = 0.3,
    alpha: float = 3000,
    pos_adjust: bool = False
):
    """
    Run prior-informed RAPM for defensive turnovers.

    Uses RFTOV as a Bayesian prior to inform the DTOV estimates.

    Args:
        input_files: List of TOV parquet files
        start_year: Starting year (4-digit)
        end_year: Ending year (4-digit)
        prior_weight: Weight for the prior (0-1). Higher = more prior influence
        alpha: Ridge regularization parameter
        pos_adjust: If True, apply positional adjustment to RFTOV priors
    """
    logging.info("=" * 60)
    logging.info("PRIOR-INFORMED DTOV RAPM")
    logging.info(f"Prior weight: {prior_weight}, Alpha: {alpha}, Pos adjust: {pos_adjust}")
    logging.info("=" * 60)

    # Load TOV data
    df = load_tov_data(input_files)
    if df is None:
        logging.error("No TOV data loaded")
        return None

    # Fetch RFTOV priors from Supabase
    logging.info("Fetching RFTOV priors from Supabase...")
    rftov_df = fetch_rftov_data(start_year, end_year, playoffs=0)

    if rftov_df is None:
        logging.warning("Could not fetch RFTOV data, running without priors")
        prior_weight = 0
        rftov_priors = {}
    else:
        # Apply positional adjustment if requested
        if pos_adjust:
            rftov_df['pos_group'] = rftov_df['Pos2'].apply(get_position_group)
            # Subtract position group mean per year
            pos_means = rftov_df.groupby(['year', 'pos_group'])['rFTOV_100'].transform('mean')
            rftov_df['rFTOV_adj'] = rftov_df['rFTOV_100'] - pos_means
            logging.info("Applied positional adjustment to RFTOV priors")
            # Average position-adjusted RFTOV across years
            rftov_avg = rftov_df.groupby('nba_id')['rFTOV_adj'].mean()
        else:
            # Average raw RFTOV across years
            rftov_avg = rftov_df.groupby('nba_id')['rFTOV_100'].mean()

        rftov_priors = rftov_avg.to_dict()
        logging.info(f"Loaded RFTOV priors for {len(rftov_priors)} players")

    # Load name mapping
    name_map_file = PROJECT_ROOT / "autocomplete_map.csv"
    try:
        name_df = pd.read_csv(name_map_file)
        name_map = dict(zip(name_df['nba_id'].astype(str), name_df['player_name']))
    except:
        name_map = {}

    # Transform to home/away format
    df_transformed = transform_to_home_away(df)

    # Get unique players
    all_players = set()
    for col in [f'a{i}' for i in range(1, 6)] + [f'h{i}' for i in range(1, 6)]:
        all_players.update(df_transformed[col].astype(int).astype(str).unique())
    all_players = sorted(all_players)
    logging.info(f"Found {len(all_players)} unique players")

    # Build player-to-column mapping (defense only)
    player_to_col = {}
    col_to_player = {}
    for idx, p in enumerate(all_players):
        player_to_col[f"{p}_def"] = idx
        col_to_player[idx] = p

    # Build design matrix
    X, y = build_design_matrix_defense_only(df_transformed, player_to_col)

    # Center outcome
    y_mean = np.mean(y)
    y_centered = y - y_mean
    logging.info(f"Centered outcome, mean={y_mean:.4f}")

    # Calculate prior means for each player
    n_features = len(all_players)
    prior_means = np.zeros(n_features)

    for idx, p in enumerate(all_players):
        player_id = int(p)
        if player_id in rftov_priors:
            # RFTOV is positive = good (forces turnovers)
            # DTOV RAPM: negative = good (forces turnovers)
            # So we negate the RFTOV to align with DTOV sign convention
            prior_means[idx] = -rftov_priors[player_id] / 100  # Scale to per-possession

    logging.info(f"Set priors for {np.sum(prior_means != 0)} players")

    # Run prior-informed Ridge regression
    # The prior acts as a regularization toward the prior mean
    # Ridge with prior: minimize ||y - X*beta||^2 + alpha * ||beta - prior||^2

    # This is equivalent to: minimize ||y_adj - X*beta||^2 + alpha * ||beta||^2
    # where y_adj = y + alpha * X * prior (pseudo-observations)

    if prior_weight > 0 and np.any(prior_means != 0):
        logging.info("Applying prior-informed Ridge regression...")

        # Standard Ridge regression first
        ridge = Ridge(alpha=alpha, fit_intercept=False, solver='lsqr')
        ridge.fit(X, y_centered)
        beta_ridge = ridge.coef_

        # Count possessions from design matrix column sums (efficient)
        player_poss_arr = np.zeros(n_features)
        for idx, p in enumerate(all_players):
            col_idx = player_to_col[f"{p}_def"]
            player_poss_arr[idx] = X[:, col_idx].sum()

        # Adaptive shrinkage: MORE prior for HIGH-sample, LESS for low-sample
        # Rationale: We trust the prior (RFTOV) only when we have enough RAPM data
        #            to validate it. Low-sample players shrink toward zero instead.
        # Formula: shrinkage = prior_weight * poss/(poss + k)
        # k controls the "half-life" - at k possessions, shrinkage is half the max
        k = 5000  # At 5000 poss, prior weight reaches half its max
        adaptive_weights = prior_weight * (player_poss_arr / (player_poss_arr + k))

        # Shrink each player according to their adaptive weight
        beta = (1 - adaptive_weights) * beta_ridge + adaptive_weights * prior_means

        logging.info(f"Applied adaptive prior shrinkage (base weight {prior_weight}, k={k})")
        logging.info(f"  Low-sample (~100 poss): {prior_weight * (100/(100+k)):.1%} prior")
        logging.info(f"  Medium-sample (~5000 poss): {prior_weight * (5000/(5000+k)):.1%} prior")
        logging.info(f"  High-sample (~15000 poss): {prior_weight * (15000/(15000+k)):.1%} prior")
    else:
        # Standard Ridge without prior
        ridge = Ridge(alpha=alpha, fit_intercept=False, solver='lsqr')
        ridge.fit(X, y_centered)
        beta = ridge.coef_

    # Calculate possessions
    player_def_poss = defaultdict(int)

    for _, row in df.iterrows():
        def_pl = [str(int(row[f'D{k}'])) for k in range(1, 6) if pd.notna(row.get(f'D{k}'))]
        for p in def_pl:
            player_def_poss[p] += 1

    # Re-center so weighted average = 0
    sum_val = 0.0
    sum_poss = 0.0

    for p in all_players:
        def_key = f"{p}_def"
        val = beta[player_to_col[def_key]]
        poss = player_def_poss[p]
        sum_val += val * poss
        sum_poss += poss

    if sum_poss > 0:
        offset = sum_val / sum_poss
        beta -= offset
        logging.info(f"Re-centered by {offset:.4f}")

    # Build results
    results = []
    for p in all_players:
        def_key = f"{p}_def"
        def_val = beta[player_to_col[def_key]]
        poss = player_def_poss[p]

        player_name = name_map.get(p, f"ID_{p}")

        # Get prior value if available
        player_id = int(p)
        prior_val = rftov_priors.get(player_id, 0)

        results.append({
            'player_id': p,
            'player_name': player_name,
            'dtov_prior': round(def_val * 100, 2),  # Multiply by 100 for per-100 poss
            'rftov_100': round(prior_val, 2),
            'def_poss': poss
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('dtov_prior', ascending=True).reset_index(drop=True)

    # Save results
    year_range = f"{start_year % 100:02d}_{end_year % 100:02d}"
    output_file = RESULTS_DIR / f"dtov_prior_{year_range}_all_results.csv"
    results_df.to_csv(output_file, index=False)
    logging.info(f"Saved results to {output_file}")

    # Display top players (most negative = best at forcing turnovers)
    logging.info("\n" + "=" * 60)
    logging.info("TOP 20 PLAYERS BY PRIOR-INFORMED DTOV")
    logging.info("(Negative = good at forcing turnovers)")
    logging.info("=" * 60)
    for i, row in results_df.head(20).iterrows():
        logging.info(
            f"{i+1:2d}. {row['player_name']:25s} | "
            f"DTOV: {row['dtov_prior']:+.2f} | "
            f"RFTOV: {row['rftov_100']:+.2f} | "
            f"Poss: {row['def_poss']:5d}"
        )

    return results_df


###############################################################################
# MAIN
###############################################################################

def main():
    parser = argparse.ArgumentParser(
        description='Prior-Informed RAPM for Defensive Turnovers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze RFTOV predictiveness of DTOV (2015-2025)
  python prior_informed_rapm.py analyze 15 25

  # In-sample correlation: 5-year RFTOV vs 5-year DTOV
  python prior_informed_rapm.py insample 21 25
  python prior_informed_rapm.py insample 21 25 --no-pos-adjust

  # Run prior-informed DTOV RAPM (2023-2026, all games)
  python prior_informed_rapm.py run 23 26 ALL --prior-weight 0.45

  # Run with position-adjusted priors
  python prior_informed_rapm.py run 23 26 ALL --prior-weight 0.45 --pos-adjust

  # Generate RFTOV prior CSV for rapm.py --def-prior
  python prior_informed_rapm.py generate-prior 23 26 rftov_prior.csv
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Analyze subcommand (year-over-year predictiveness)
    analyze_parser = subparsers.add_parser('analyze', help='Analyze year-over-year RFTOV predictiveness')
    analyze_parser.add_argument('start_year', type=int, help='Start year (2-digit, e.g., 15 for 2015)')
    analyze_parser.add_argument('end_year', type=int, help='End year (2-digit, e.g., 25 for 2025)')

    # Insample subcommand (in-sample correlation)
    insample_parser = subparsers.add_parser('insample', help='In-sample correlation: multi-year RFTOV vs DTOV')
    insample_parser.add_argument('start_year', type=int, help='Start year (2-digit, e.g., 21 for 2021)')
    insample_parser.add_argument('end_year', type=int, help='End year (2-digit, e.g., 25 for 2025)')
    insample_parser.add_argument('--no-pos-adjust', action='store_true',
                                 help='Disable positional adjustment')

    # Run subcommand
    run_parser = subparsers.add_parser('run', help='Run prior-informed DTOV RAPM')
    run_parser.add_argument('start_year', type=int, help='Start year (2-digit)')
    run_parser.add_argument('end_year', type=int, help='End year (2-digit)')
    run_parser.add_argument('season_type', choices=['RS', 'PS', 'ALL'], help='Season type')
    run_parser.add_argument('--prior-weight', type=float, default=0.3,
                           help='Weight for RFTOV prior (0-1, default 0.3)')
    run_parser.add_argument('--alpha', type=float, default=3000,
                           help='Ridge regularization parameter (default 3000)')
    run_parser.add_argument('--pos-adjust', action='store_true',
                           help='Apply positional adjustment to RFTOV priors')

    # Generate-prior subcommand
    genprior_parser = subparsers.add_parser('generate-prior',
                                             help='Generate RFTOV prior CSV for rapm.py --def-prior')
    genprior_parser.add_argument('start_year', type=int, help='Start year (2-digit, e.g., 23 for 2023)')
    genprior_parser.add_argument('end_year', type=int, help='End year (2-digit, e.g., 26 for 2026)')
    genprior_parser.add_argument('output_file', type=str, help='Output CSV filename')

    # Residual persistence subcommand
    residual_parser = subparsers.add_parser('residual',
                                             help='Test if DTOV residual (beyond RFTOV) persists')
    residual_parser.add_argument('--all-windows', action='store_true',
                                  help='Run across all 3 test windows')
    residual_parser.add_argument('--min-poss', type=int, default=5000,
                                  help='Minimum possessions in each period (default: 5000)')

    args = parser.parse_args()

    if args.command == 'analyze':
        # Convert 2-digit years to 4-digit
        start_year = 2000 + args.start_year
        end_year = 2000 + args.end_year
        analyze_rftov_predictiveness(start_year, end_year)

    elif args.command == 'insample':
        # Convert 2-digit years to 4-digit
        start_year = 2000 + args.start_year
        end_year = 2000 + args.end_year
        pos_adjust = not args.no_pos_adjust
        analyze_insample_correlation(start_year, end_year, pos_adjust=pos_adjust)

    elif args.command == 'run':
        # Build input file list
        input_files = []
        for year in range(args.start_year, args.end_year + 1):
            if args.season_type in ['RS', 'ALL']:
                input_files.append(f"TOV{year:02d}.parquet")
            if args.season_type in ['PS', 'ALL']:
                input_files.append(f"TOV{year:02d}_PS.parquet")

        start_year = 2000 + args.start_year
        end_year = 2000 + args.end_year

        run_prior_informed_dtov(
            input_files,
            start_year,
            end_year,
            prior_weight=args.prior_weight,
            alpha=args.alpha,
            pos_adjust=args.pos_adjust
        )

    elif args.command == 'generate-prior':
        # Convert 2-digit years to 4-digit
        start_year = 2000 + args.start_year
        end_year = 2000 + args.end_year
        generate_rftov_prior(start_year, end_year, args.output_file)

    elif args.command == 'residual':
        # Run residual persistence analysis
        run_residual_persistence_all_windows(min_poss=args.min_poss)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
