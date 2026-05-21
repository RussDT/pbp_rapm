#!/usr/bin/env python3
"""
Weighted Linear Regression Analysis for RAPM Prediction

This script predicts net_rapm using offensive and defensive metrics from
True Shooting (TS), Turnover (TOV), and Rebound (REB) data.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path

def load_and_merge_data():
    """Load all CSV files and merge them on player_id."""
    results_dir = Path('Results')
    
    # Load all datasets
    print("Loading data files...")
    rapm_df = pd.read_csv(results_dir / 'rapm_23_24_25_all_pure_results.csv')
    ts_df = pd.read_csv(results_dir / 'ts_23_24_25_all_results.csv')
    tov_df = pd.read_csv(results_dir / 'tov_23_24_25_all_results.csv')
    reb_df = pd.read_csv(results_dir / 'reb_23_24_25_all_results.csv')
    
    print(f"  RAPM: {len(rapm_df)} players")
    print(f"  TS: {len(ts_df)} players")
    print(f"  TOV: {len(tov_df)} players")
    print(f"  REB: {len(reb_df)} players")
    
    # Merge all datasets on player_id
    # Start with RAPM as base, keep player_name, off, def, net_rapm and possessions from it
    merged = rapm_df[['player_id', 'player_name', 'off', 'def', 'net_rapm', 'possessions']].copy()
    merged.rename(columns={'off': 'off_rapm', 'def': 'def_rapm'}, inplace=True)
    
    # Merge TS data (rename off/def columns)
    ts_subset = ts_df[['player_id', 'off', 'def']].rename(
        columns={'off': 'off_ts', 'def': 'def_ts'}
    )
    merged = merged.merge(ts_subset, on='player_id', how='inner')
    
    # Merge TOV data
    tov_subset = tov_df[['player_id', 'off', 'def']].rename(
        columns={'off': 'off_tov', 'def': 'def_tov'}
    )
    merged = merged.merge(tov_subset, on='player_id', how='inner')
    
    # Merge REB data
    reb_subset = reb_df[['player_id', 'off', 'def']].rename(
        columns={'off': 'off_reb', 'def': 'def_reb'}
    )
    merged = merged.merge(reb_subset, on='player_id', how='inner')
    
    print(f"\nMerged dataset: {len(merged)} players (players in all 4 files)")
    
    return merged

def run_weighted_regression(df, target_col, target_name):
    """Run weighted least squares regression."""
    print("\n" + "="*80)
    print(f"WEIGHTED LINEAR REGRESSION ANALYSIS - {target_name}")
    print("="*80)
    
    # Define features and target
    feature_cols = ['off_ts', 'def_ts', 'off_tov', 'def_tov', 'off_reb', 'def_reb']
    X = df[feature_cols]
    y = df[target_col]
    weights = df['possessions']
    
    # Add constant term for intercept
    X_with_const = sm.add_constant(X)
    
    # Fit weighted least squares model
    print(f"\nFitting weighted least squares model...")
    print(f"  Features: {', '.join(feature_cols)}")
    print(f"  Target: {target_col}")
    print(f"  Weights: possessions")
    print(f"  Sample size: {len(df)} players")
    
    model = sm.WLS(y, X_with_const, weights=weights)
    results = model.fit()
    
    return results, feature_cols

def print_results(results, feature_cols):
    """Print regression results to console."""
    print("\n" + "-"*80)
    print("MODEL SUMMARY")
    print("-"*80)
    print(results.summary())
    
    print("\n" + "-"*80)
    print("KEY METRICS")
    print("-"*80)
    print(f"R-squared:           {results.rsquared:.4f}")
    print(f"Adjusted R-squared:  {results.rsquared_adj:.4f}")
    print(f"F-statistic:         {results.fvalue:.2f}")
    print(f"Prob (F-statistic):  {results.f_pvalue:.2e}")
    
    print("\n" + "-"*80)
    print("COEFFICIENTS")
    print("-"*80)
    print(f"{'Variable':<15} {'Coefficient':>12} {'Std Error':>12} {'p-value':>12} {'[95% Conf':>12} {'Interval]':>12}")
    print("-"*80)
    
    # Print intercept
    print(f"{'Intercept':<15} {results.params['const']:>12.4f} {results.bse['const']:>12.4f} "
          f"{results.pvalues['const']:>12.4f} {results.conf_int().loc['const', 0]:>12.4f} "
          f"{results.conf_int().loc['const', 1]:>12.4f}")
    
    # Print feature coefficients
    for col in feature_cols:
        print(f"{col:<15} {results.params[col]:>12.4f} {results.bse[col]:>12.4f} "
              f"{results.pvalues[col]:>12.4f} {results.conf_int().loc[col, 0]:>12.4f} "
              f"{results.conf_int().loc[col, 1]:>12.4f}")

def save_regression_results(results, feature_cols, output_path, model_name=""):
    """Save regression coefficients and metrics to CSV."""
    # Create DataFrame with coefficients
    coef_data = []
    
    # Add metrics as first rows
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
    
    # Add intercept
    coef_data.append({
        'variable': 'Intercept',
        'coefficient': results.params['const'],
        'std_error': results.bse['const'],
        'p_value': results.pvalues['const'],
        'conf_int_lower': results.conf_int().loc['const', 0],
        'conf_int_upper': results.conf_int().loc['const', 1]
    })
    
    # Add feature coefficients
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
    coef_df.to_csv(output_path, index=False)
    if model_name:
        print(f"\n{model_name} regression results saved to: {output_path}")
    else:
        print(f"\nRegression results saved to: {output_path}")

def create_all_predictions(df, results, feature_cols, output_path, actual_col, pred_col_name, model_name=""):
    """Create predictions vs actuals for all players."""
    print("\n" + "="*80)
    print(f"ALL PLAYERS - PREDICTIONS VS ACTUALS - {model_name}")
    print("="*80)
    
    # Prepare features for prediction
    X = df[feature_cols]
    X_with_const = sm.add_constant(X)
    
    # Generate predictions
    df_pred = df.copy()
    df_pred[pred_col_name] = results.predict(X_with_const)
    df_pred['residual'] = df_pred[actual_col] - df_pred[pred_col_name]
    
    # Sort by actual value (descending)
    df_pred = df_pred.sort_values(actual_col, ascending=False)
    
    # Select and round columns
    output_df = df_pred[['player_name', actual_col, pred_col_name, 'residual', 'possessions']].copy()
    output_df[[actual_col, pred_col_name, 'residual']] = output_df[[actual_col, pred_col_name, 'residual']].round(2)
    
    # Save to CSV
    output_df.to_csv(output_path, index=False)
    if model_name:
        print(f"\n{model_name}: All {len(output_df)} player predictions saved to: {output_path}")
    else:
        print(f"\nAll {len(output_df)} player predictions saved to: {output_path}")
    
    # Print summary statistics
    print("\n" + "-"*80)
    print("SUMMARY STATISTICS FOR ALL PLAYERS")
    print("-"*80)
    print(f"Number of Players:     {len(output_df)}")
    print(f"Mean Actual:           {output_df[actual_col].mean():>10.2f}")
    print(f"Mean Predicted:        {output_df[pred_col_name].mean():>10.2f}")
    print(f"Mean Absolute Error:   {output_df['residual'].abs().mean():>10.2f}")
    print(f"RMSE:                  {np.sqrt((output_df['residual']**2).mean()):>10.2f}")
    print(f"Max Positive Residual: {output_df['residual'].max():>10.2f} ({output_df.loc[output_df['residual'].idxmax(), 'player_name']})")
    print(f"Max Negative Residual: {output_df['residual'].min():>10.2f} ({output_df.loc[output_df['residual'].idxmin(), 'player_name']})")

def create_weighted_factors_csv(df, results, output_path):
    """Create CSV with weighted factor contributions for each player."""
    print("\n" + "="*80)
    print("CREATING WEIGHTED FACTORS CSV")
    print("="*80)
    
    # Get coefficients
    coef_off_ts = results.params['off_ts']
    coef_def_ts = results.params['def_ts']
    coef_off_tov = results.params['off_tov']
    coef_def_tov = results.params['def_tov']
    coef_off_reb = results.params['off_reb']
    coef_def_reb = results.params['def_reb']
    
    # Create output dataframe
    output_df = df[['player_id', 'player_name']].copy()
    
    # Calculate weighted contributions (player's value * coefficient)
    output_df['oTS'] = (df['off_ts'] * coef_off_ts).round(2)
    output_df['oTOV'] = (df['off_tov'] * coef_off_tov).round(2)
    output_df['oREB'] = (df['off_reb'] * coef_off_reb).round(2)
    output_df['dTS'] = (df['def_ts'] * coef_def_ts).round(2)
    output_df['dTOV'] = (df['def_tov'] * coef_def_tov).round(2)
    output_df['dREB'] = (df['def_reb'] * coef_def_reb).round(2)
    
    # Add actual RAPM values
    output_df['off'] = df['off_rapm'].round(2)
    output_df['def'] = df['def_rapm'].round(2)
    output_df['net_rapm'] = df['net_rapm'].round(2)
    
    # Sort by net_rapm descending
    output_df = output_df.sort_values('net_rapm', ascending=False)
    
    # Save to CSV
    output_df.to_csv(output_path, index=False)
    print(f"\nWeighted factors CSV saved to: {output_path}")
    print(f"  Columns: player_id, player_name, oTS, oTOV, oREB, dTS, dTOV, dREB, off, def, net_rapm")
    print(f"  Rows: {len(output_df)} players")
    print(f"\nNote: Factor columns show each player's metric value * coefficient")
    print(f"  oTS  = off_ts  * {coef_off_ts:.4f}")
    print(f"  oTOV = off_tov * {coef_off_tov:.4f}")
    print(f"  oREB = off_reb * {coef_off_reb:.4f}")
    print(f"  dTS  = def_ts  * {coef_def_ts:.4f}")
    print(f"  dTOV = def_tov * {coef_def_tov:.4f}")
    print(f"  dREB = def_reb * {coef_def_reb:.4f}")

def create_top50_predictions(df, results, feature_cols, output_path):
    """Create predictions vs actuals for top 50 players by actual net_rapm."""
    print("\n" + "="*80)
    print("TOP 50 PLAYERS - PREDICTIONS VS ACTUALS")
    print("="*80)
    
    # Prepare features for prediction
    X = df[feature_cols]
    X_with_const = sm.add_constant(X)
    
    # Generate predictions
    df = df.copy()
    df['predicted_net_rapm'] = results.predict(X_with_const)
    df['residual'] = df['net_rapm'] - df['predicted_net_rapm']
    
    # Get top 50 by actual net_rapm
    top50 = df.nlargest(50, 'net_rapm')[['player_name', 'net_rapm', 'predicted_net_rapm', 'residual']].copy()
    top50 = top50.round(2)
    
    # Save to CSV
    top50.to_csv(output_path, index=False)
    print(f"\nTop 50 predictions saved to: {output_path}")
    
    # Print to console
    print("\n" + "-"*80)
    print(f"{'Rank':<6} {'Player':<30} {'Actual':>10} {'Predicted':>10} {'Residual':>10}")
    print("-"*80)
    for idx, row in enumerate(top50.itertuples(), 1):
        print(f"{idx:<6} {row.player_name:<30} {row.net_rapm:>10.2f} {row.predicted_net_rapm:>10.2f} {row.residual:>10.2f}")
    
    # Print summary statistics
    print("\n" + "-"*80)
    print("SUMMARY STATISTICS FOR TOP 50")
    print("-"*80)
    print(f"Mean Actual:           {top50['net_rapm'].mean():>10.2f}")
    print(f"Mean Predicted:        {top50['predicted_net_rapm'].mean():>10.2f}")
    print(f"Mean Absolute Error:   {top50['residual'].abs().mean():>10.2f}")
    print(f"RMSE:                  {np.sqrt((top50['residual']**2).mean()):>10.2f}")

def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("RAPM PREDICTION USING TS, TOV, AND REB METRICS")
    print("="*80)
    
    # Load and merge data
    df = load_and_merge_data()
    
    # =========================================================================
    # 1. NET RAPM REGRESSION
    # =========================================================================
    results_net, feature_cols = run_weighted_regression(df, 'net_rapm', 'NET RAPM')
    print_results(results_net, feature_cols)
    
    # Save regression results
    results_path = Path('Results') / 'rapm_net_regression_analysis.csv'
    save_regression_results(results_net, feature_cols, results_path, "Net RAPM")
    
    # Create top 50 predictions
    top50_path = Path('Results') / 'rapm_net_top50_predictions.csv'
    create_top50_predictions(df, results_net, feature_cols, top50_path)
    
    # Create all predictions
    all_predictions_path = Path('Results') / 'rapm_net_all_predictions.csv'
    create_all_predictions(df, results_net, feature_cols, all_predictions_path, 
                          'net_rapm', 'predicted_net_rapm', "Net RAPM")
    
    # Create weighted factors CSV
    weighted_factors_path = Path('Results') / 'rapm_weighted_factors.csv'
    create_weighted_factors_csv(df, results_net, weighted_factors_path)
    
    # =========================================================================
    # 2. OFFENSIVE RAPM REGRESSION
    # =========================================================================
    results_off, feature_cols = run_weighted_regression(df, 'off_rapm', 'OFFENSIVE RAPM')
    print_results(results_off, feature_cols)
    
    # Save regression results
    results_path = Path('Results') / 'rapm_off_regression_analysis.csv'
    save_regression_results(results_off, feature_cols, results_path, "Offensive RAPM")
    
    # Create all predictions
    all_predictions_path = Path('Results') / 'rapm_off_all_predictions.csv'
    create_all_predictions(df, results_off, feature_cols, all_predictions_path,
                          'off_rapm', 'predicted_off_rapm', "Offensive RAPM")
    
    # =========================================================================
    # 3. DEFENSIVE RAPM REGRESSION
    # =========================================================================
    results_def, feature_cols = run_weighted_regression(df, 'def_rapm', 'DEFENSIVE RAPM')
    print_results(results_def, feature_cols)
    
    # Save regression results
    results_path = Path('Results') / 'rapm_def_regression_analysis.csv'
    save_regression_results(results_def, feature_cols, results_path, "Defensive RAPM")
    
    # Create all predictions
    all_predictions_path = Path('Results') / 'rapm_def_all_predictions.csv'
    create_all_predictions(df, results_def, feature_cols, all_predictions_path,
                          'def_rapm', 'predicted_def_rapm', "Defensive RAPM")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - rapm_net_regression_analysis.csv (Net RAPM coefficients)")
    print("  - rapm_net_all_predictions.csv (Net RAPM predictions for all players)")
    print("  - rapm_net_top50_predictions.csv (Net RAPM predictions for top 50)")
    print("  - rapm_weighted_factors.csv (Weighted factor contributions)")
    print("  - rapm_off_regression_analysis.csv (Offensive RAPM coefficients)")
    print("  - rapm_off_all_predictions.csv (Offensive RAPM predictions for all players)")
    print("  - rapm_def_regression_analysis.csv (Defensive RAPM coefficients)")
    print("  - rapm_def_all_predictions.csv (Defensive RAPM predictions for all players)")

if __name__ == '__main__':
    main()

