#!/usr/bin/env python3
"""
WNBA Master RAPM Analysis Script

Runs all 4 RAPM types (RAPM, TS, TOV, REB) and creates weighted 6-factor output.

Usage:
    python master_wnba_rapm.py 24 25 RS
    python master_wnba_rapm.py 24 25 ALL
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.linear_model import Ridge
import statsmodels.api as sm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ---------------------------------------------------------------------
# RAPM Engine (adapted from rapm_wnba.py)
# ---------------------------------------------------------------------
def _known_players(players):
    known = []
    for player in players:
        if pd.isna(player):
            continue
        try:
            player_id = int(player)
        except (TypeError, ValueError):
            continue
        if player_id != 0:
            known.append(player)
    return known


def detect_file_type_and_prepare(df, pure=False):
    """Detect file type and prepare Off_Diff and Def_Diff columns."""
    df = df.copy()
    
    if 'Off_Diff' in df.columns and 'Def_Diff' in df.columns:
        if pure and 'Net_Diff' in df.columns:
            df['Off_Diff'] = df['Net_Diff']
            df['Def_Diff'] = -df['Net_Diff']
    elif 'Is_Turnover' in df.columns:
        df['Off_Diff'] = -df['Is_Turnover']
        df['Def_Diff'] = df['Is_Turnover']
    elif 'Offensive_Rebound' in df.columns:
        df['Off_Diff'] = df['Offensive_Rebound']
        df['Def_Diff'] = -df['Offensive_Rebound']
    elif 'Net_Diff' in df.columns:
        df['Off_Diff'] = df['Net_Diff']
        df['Def_Diff'] = -df['Net_Diff']
    else:
        raise ValueError("Could not detect file type")
    
    return df

def transform_to_home_away_format(df):
    """Transform O/D format to home/away format."""
    transformed_rows = []
    
    for game_id in df['game_id'].unique():
        game_df = df[df['game_id'] == game_id].copy().reset_index(drop=True)
        
        away_players = set()
        home_players = set()
        
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
                away_players.update(_known_players(off_players))
                home_players.update(_known_players(def_players))
            elif home_pts > 0:
                home_players.update(_known_players(off_players))
                away_players.update(_known_players(def_players))
        
        for i in range(len(game_df)):
            row = game_df.iloc[i]
            off_players = [row[f'O{j}'] for j in range(1, 6)]
            def_players = [row[f'D{j}'] for j in range(1, 6)]
            pts = float(row['Off_Diff'])
            known_off_players = set(_known_players(off_players))
            off_is_home = len(known_off_players & home_players) > len(known_off_players & away_players)
            
            transformed_row = {
                'home_poss': 1 if off_is_home else 0,
                'pts': pts,
                'season': int(row['Season']),
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
    
    return pd.DataFrame(transformed_rows)

def build_design_matrix(df, player_to_col):
    """Build sparse design matrix."""
    n_samples = len(df)
    n_features = len(player_to_col)
    
    a_players = df[[f'a{k}' for k in range(1, 6)]].values.astype(int)
    h_players = df[[f'h{k}' for k in range(1, 6)]].values.astype(int)
    home_poss = df['home_poss'].values
    pts = df['pts'].values.astype(np.float64)
    
    offense_players = np.where(home_poss[:, None] == 1, h_players, a_players)
    defense_players = np.where(home_poss[:, None] == 1, a_players, h_players)
    
    row_indices = []
    col_indices = []
    
    for player_idx in range(5):
        player_ids = offense_players[:, player_idx]
        for i, pid in enumerate(player_ids):
            off_key = f"{pid}_off"
            if off_key in player_to_col:
                row_indices.append(i)
                col_indices.append(player_to_col[off_key])
    
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
    
    return X.tocsr(), pts

def alternating_rapm(X, y, player_to_col, all_players, alpha=3000, max_iter=200, tol=1e-4):
    """Alternating Ridge regression."""
    n_features = X.shape[1]
    beta = np.zeros(n_features, dtype=np.float64)
    
    offense_indices = np.array([v for k, v in player_to_col.items() if k.endswith('_off')], dtype=int)
    defense_indices = np.array([v for k, v in player_to_col.items() if k.endswith('_def')], dtype=int)
    
    X_off = X[:, offense_indices]
    X_def = X[:, defense_indices]
    
    ridge_off = Ridge(alpha=alpha, fit_intercept=False, solver='lsqr')
    ridge_def = Ridge(alpha=alpha, fit_intercept=False, solver='lsqr')
    
    residual = y.copy()
    
    for iteration in range(max_iter):
        beta_prev = beta.copy()
        
        residual += X_off @ beta[offense_indices]
        ridge_off.fit(X_off, residual)
        beta[offense_indices] = ridge_off.coef_
        residual -= X_off @ beta[offense_indices]
        
        residual += X_def @ beta[defense_indices]
        ridge_def.fit(X_def, residual)
        beta[defense_indices] = ridge_def.coef_
        residual -= X_def @ beta[defense_indices]
        
        delta = np.linalg.norm(beta - beta_prev)
        if delta < tol:
            break
    
    return beta

def run_rapm_on_files(input_files, name_map, pure=False):
    """Run RAPM on a list of input files."""
    all_dfs = []
    for f in input_files:
        if os.path.exists(f):
            df = pd.read_csv(f)
            df = detect_file_type_and_prepare(df, pure=pure)
            all_dfs.append(df)
            logging.info(f"  Loaded {len(df)} rows from {f}")
    
    if not all_dfs:
        return None
    
    df = pd.concat(all_dfs, ignore_index=True)
    logging.info(f"  Combined: {len(df)} total possessions")
    
    df_transformed = transform_to_home_away_format(df)
    
    all_players = set()
    for col in [f'a{i}' for i in range(1, 6)] + [f'h{i}' for i in range(1, 6)]:
        all_players.update(df_transformed[col].astype(int).astype(str).unique())
    all_players = sorted(player for player in all_players if str(player) != "0")
    
    player_to_col = {}
    idx = 0
    for p in all_players:
        player_to_col[f"{p}_off"] = idx
        idx += 1
        player_to_col[f"{p}_def"] = idx
        idx += 1
    
    X, y = build_design_matrix(df_transformed, player_to_col)
    y_centered = y - np.mean(y)
    
    beta = alternating_rapm(X, y_centered, player_to_col, all_players)
    
    # Calculate possessions - separate off and def
    player_off_poss = defaultdict(int)
    player_def_poss = defaultdict(int)
    for _, row in df_transformed.iterrows():
        away_pl = [int(row[f'a{k}']) for k in range(1, 6)]
        home_pl = [int(row[f'h{k}']) for k in range(1, 6)]
        home_poss = row['home_poss']
        
        if home_poss == 1:
            off_players, def_players = home_pl, away_pl
        else:
            off_players, def_players = away_pl, home_pl
        
        for p in off_players:
            if p == 0:
                continue
            player_off_poss[str(p)] += 1
        for p in def_players:
            if p == 0:
                continue
            player_def_poss[str(p)] += 1
    
    # Re-center
    sum_off, sum_def = 0.0, 0.0
    sum_off_poss, sum_def_poss = 0.0, 0.0
    
    for p in all_players:
        off_val = beta[player_to_col[f"{p}_off"]]
        def_val = beta[player_to_col[f"{p}_def"]]
        off_poss = player_off_poss[p]
        def_poss = player_def_poss[p]
        sum_off += off_val * off_poss
        sum_def += def_val * def_poss
        sum_off_poss += off_poss
        sum_def_poss += def_poss
    
    if sum_off_poss > 0 and sum_def_poss > 0:
        off_offset = sum_off / sum_off_poss
        def_offset = sum_def / sum_def_poss
        for p in all_players:
            beta[player_to_col[f"{p}_off"]] -= off_offset
            beta[player_to_col[f"{p}_def"]] -= def_offset
    
    # Build results
    results = []
    for p in all_players:
        off_val = beta[player_to_col[f"{p}_off"]]
        def_val = beta[player_to_col[f"{p}_def"]]
        net_rapm = off_val - def_val
        off_poss = player_off_poss[p]
        def_poss = player_def_poss[p]
        
        player_name = name_map.get(p, f"ID_{p}")
        
        results.append({
            'player_id': p,
            'player_name': player_name,
            'off': round(off_val * 100, 2),
            'def': round(def_val * 100, 2),
            'net_rapm': round(net_rapm * 100, 2),
            'off_poss': off_poss,
            'def_poss': def_poss
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('net_rapm', ascending=False).reset_index(drop=True)
    
    return results_df

# ---------------------------------------------------------------------
# Main Analysis
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='WNBA Master RAPM Analysis')
    parser.add_argument('start_year', type=int, help='Starting year (2-digit, e.g., 24)')
    parser.add_argument('end_year', type=int, help='Ending year (2-digit, e.g., 25)')
    parser.add_argument('season_type', choices=['RS', 'PS', 'ALL'], 
                       help='Season type: RS, PS, or ALL')
    args = parser.parse_args()
    
    logging.info("=" * 70)
    logging.info("WNBA MASTER RAPM ANALYSIS")
    logging.info("=" * 70)
    logging.info(f"Years: {args.start_year}-{args.end_year}")
    logging.info(f"Season Type: {args.season_type}")
    logging.info("=" * 70)
    
    # Build file lists
    years = list(range(args.start_year, args.end_year + 1))
    processed_dir = Path('wnba_test/Processed')
    
    rapm_files, ts_files, tov_files, reb_files = [], [], [], []
    
    for year in years:
        if args.season_type in ['RS', 'ALL']:
            rapm_files.append(processed_dir / f"RAPM{year}.csv")
            ts_files.append(processed_dir / f"TS{year}.csv")
            tov_files.append(processed_dir / f"TOV{year}.csv")
            reb_files.append(processed_dir / f"REB{year}.csv")
        if args.season_type in ['PS', 'ALL']:
            rapm_files.append(processed_dir / f"RAPM{year}_PS.csv")
            ts_files.append(processed_dir / f"TS{year}_PS.csv")
            tov_files.append(processed_dir / f"TOV{year}_PS.csv")
            reb_files.append(processed_dir / f"REB{year}_PS.csv")
    
    # Load name map
    name_map = {}
    wnba_map_path = "wnba_test/player_index_map.csv"
    if os.path.exists(wnba_map_path):
        wnba_df = pd.read_csv(wnba_map_path)
        name_map = dict(zip(wnba_df['EntityId'].astype(str), wnba_df['pbp_name']))
        logging.info(f"Loaded {len(name_map)} player names")
    
    # Run all 4 RAPM types
    logging.info("\n" + "=" * 70)
    logging.info("Running RAPM (Pure)...")
    rapm_df = run_rapm_on_files([str(f) for f in rapm_files], name_map, pure=True)
    
    logging.info("\n" + "=" * 70)
    logging.info("Running TS RAPM...")
    ts_df = run_rapm_on_files([str(f) for f in ts_files], name_map, pure=False)
    
    logging.info("\n" + "=" * 70)
    logging.info("Running TOV RAPM...")
    tov_df = run_rapm_on_files([str(f) for f in tov_files], name_map, pure=False)
    
    logging.info("\n" + "=" * 70)
    logging.info("Running REB RAPM...")
    reb_df = run_rapm_on_files([str(f) for f in reb_files], name_map, pure=False)
    
    if any(df is None for df in [rapm_df, ts_df, tov_df, reb_df]):
        logging.error("One or more RAPM analyses failed. Exiting.")
        return 1
    
    # Create output directory
    year_range = '_'.join([f"{y:02d}" for y in years])
    season_suffix = args.season_type.lower()
    output_dir = Path('wnba_test/Results') / f"rapm_{year_range}_{season_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual results
    rapm_df.to_csv(output_dir / f"rapm_{year_range}_{season_suffix}_pure_results.csv", index=False)
    ts_df.to_csv(output_dir / f"ts_{year_range}_{season_suffix}_results.csv", index=False)
    tov_df.to_csv(output_dir / f"tov_{year_range}_{season_suffix}_results.csv", index=False)
    reb_df.to_csv(output_dir / f"reb_{year_range}_{season_suffix}_results.csv", index=False)
    
    logging.info(f"\nSaved individual RAPM results to {output_dir}")
    
    # Merge for regression
    logging.info("\n" + "=" * 70)
    logging.info("Running Regression Analysis...")
    
    merged = rapm_df[['player_id', 'player_name', 'off', 'def', 'net_rapm', 'off_poss', 'def_poss']].copy()
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
    
    # Use total possessions for weights
    merged['possessions'] = merged['off_poss'] + merged['def_poss']
    
    # Run Net RAPM regression
    X = merged[feature_cols]
    y = merged['net_rapm']
    weights = merged['possessions']
    
    X_with_const = sm.add_constant(X)
    model = sm.WLS(y, X_with_const, weights=weights)
    results_net = model.fit()
    
    logging.info(f"Net RAPM R² = {results_net.rsquared:.4f}")
    
    # Get coefficients
    coef_off_ts = results_net.params['off_ts']
    coef_def_ts = results_net.params['def_ts']
    coef_off_tov = results_net.params['off_tov']
    coef_def_tov = results_net.params['def_tov']
    coef_off_reb = results_net.params['off_reb']
    coef_def_reb = results_net.params['def_reb']
    
    logging.info("\nRegression Coefficients:")
    logging.info(f"  off_ts:  {coef_off_ts:.4f}")
    logging.info(f"  def_ts:  {coef_def_ts:.4f}")
    logging.info(f"  off_tov: {coef_off_tov:.4f}")
    logging.info(f"  def_tov: {coef_def_tov:.4f}")
    logging.info(f"  off_reb: {coef_off_reb:.4f}")
    logging.info(f"  def_reb: {coef_def_reb:.4f}")
    
    # Create weighted factors DataFrame
    latest_year = 2000 + args.end_year
    
    weighted_df = merged[['player_id', 'player_name']].copy()
    weighted_df['Latest_Year'] = latest_year
    weighted_df['oTS'] = (merged['off_ts'] * coef_off_ts).round(2)
    weighted_df['oTOV'] = (merged['off_tov'] * coef_off_tov).round(2)
    weighted_df['oREB'] = (merged['off_reb'] * coef_off_reb).round(2)
    weighted_df['dTS'] = (merged['def_ts'] * coef_def_ts).round(2)
    weighted_df['dTOV'] = (merged['def_tov'] * coef_def_tov).round(2)
    weighted_df['dREB'] = (merged['def_reb'] * coef_def_reb).round(2)
    weighted_df['off'] = merged['off_rapm'].round(2)
    weighted_df['def'] = (-merged['def_rapm']).round(2)
    weighted_df['net_rapm'] = merged['net_rapm'].round(2)
    weighted_df['off_poss'] = merged['off_poss']
    weighted_df['def_poss'] = merged['def_poss']
    
    # Calculate predicted net_rapm and residual
    intercept = results_net.params['const']
    weighted_df['predicted'] = (
        weighted_df['oTS'] + weighted_df['oTOV'] + weighted_df['oREB'] +
        weighted_df['dTS'] + weighted_df['dTOV'] + weighted_df['dREB'] + intercept
    ).round(2)
    weighted_df['residual'] = (weighted_df['net_rapm'] - weighted_df['predicted']).round(2)
    
    # Calculate component residuals
    weighted_df['off_residual'] = (
        weighted_df['off'] - (weighted_df['oTS'] + weighted_df['oTOV'] + weighted_df['oREB'])
    ).round(2)
    
    weighted_df['def_residual'] = (
        weighted_df['def'] - (weighted_df['dTS'] + weighted_df['dTOV'] + weighted_df['dREB'])
    ).round(2)
    
    weighted_df = weighted_df.sort_values('net_rapm', ascending=False)
    
    # Save weighted factors
    output_path = output_dir / 'weighted_factors.csv'
    weighted_df.to_csv(output_path, index=False)
    logging.info(f"\nSaved weighted factors to {output_path}")
    
    # Save regression coefficients with full stats
    coef_data = []
    coef_data.append({
        'variable': 'R_squared',
        'coefficient': results_net.rsquared,
        'std_error': np.nan,
        'p_value': np.nan,
        'conf_int_lower': np.nan,
        'conf_int_upper': np.nan
    })
    coef_data.append({
        'variable': 'Adjusted_R_squared',
        'coefficient': results_net.rsquared_adj,
        'std_error': np.nan,
        'p_value': np.nan,
        'conf_int_lower': np.nan,
        'conf_int_upper': np.nan
    })
    coef_data.append({
        'variable': 'Intercept',
        'coefficient': results_net.params['const'],
        'std_error': results_net.bse['const'],
        'p_value': results_net.pvalues['const'],
        'conf_int_lower': results_net.conf_int().loc['const', 0],
        'conf_int_upper': results_net.conf_int().loc['const', 1]
    })
    for col in feature_cols:
        coef_data.append({
            'variable': col,
            'coefficient': results_net.params[col],
            'std_error': results_net.bse[col],
            'p_value': results_net.pvalues[col],
            'conf_int_lower': results_net.conf_int().loc[col, 0],
            'conf_int_upper': results_net.conf_int().loc[col, 1]
        })
    coef_df = pd.DataFrame(coef_data)
    coef_df.to_csv(output_dir / 'regression_coefficients.csv', index=False)
    
    logging.info(f"Adjusted R² = {results_net.rsquared_adj:.4f}")
    
    # Calculate and log RMSE
    rmse = np.sqrt((weighted_df['residual'] ** 2).mean())
    logging.info(f"RMSE = {rmse:.4f}")
    
    # Display top 20
    logging.info("\n" + "=" * 70)
    logging.info("TOP 20 WNBA PLAYERS BY NET RAPM")
    logging.info("=" * 70)
    for i, row in weighted_df.head(20).iterrows():
        logging.info(
            f"{weighted_df.index.get_loc(i)+1:2d}. {row['player_name']:25s} | "
            f"Net: {row['net_rapm']:+.2f} | "
            f"Off: {row['off']:+.2f} | "
            f"Def: {row['def']:+.2f} | "
            f"OffPoss: {row['off_poss']:5d} | "
            f"DefPoss: {row['def_poss']:5d}"
        )
    
    logging.info("\n" + "=" * 70)
    logging.info("ANALYSIS COMPLETE!")
    logging.info("=" * 70)
    logging.info(f"Results saved to: {output_dir}")
    logging.info("\nGenerated files:")
    logging.info(f"  - rapm_{year_range}_{season_suffix}_pure_results.csv")
    logging.info(f"  - ts_{year_range}_{season_suffix}_results.csv")
    logging.info(f"  - tov_{year_range}_{season_suffix}_results.csv")
    logging.info(f"  - reb_{year_range}_{season_suffix}_results.csv")
    logging.info(f"  - weighted_factors.csv")
    logging.info(f"  - regression_coefficients.csv")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
