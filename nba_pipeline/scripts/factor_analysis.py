"""
Factor Analysis Script
Analyzes relationships between the 6 factors (oTS, oTOV, oREB, dTS, dTOV, dREB)
and overall RAPM metrics (off, def, net_rapm)

Includes:
- Correlation analysis
- VIF (Variance Inflation Factor) for multicollinearity
- Regression analysis (how factors predict off/def/net)
- Residual analysis
"""

import pandas as pd
import numpy as np
import argparse
import os
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Try to import statsmodels for VIF, fall back if not available
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.api import OLS, add_constant
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Note: statsmodels not installed. VIF analysis will use manual calculation.")


def load_data(filepath):
    """Load weighted factors data"""
    df = pd.read_csv(filepath)
    return df


def correlation_analysis(df, factors, targets):
    """Analyze correlations between factors and targets"""
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS")
    print("="*80)

    all_vars = factors + targets
    corr_matrix = df[all_vars].corr()

    print("\n--- Full Correlation Matrix ---")
    print(corr_matrix.round(3).to_string())

    print("\n--- Factor-to-Factor Correlations (checking for multicollinearity) ---")
    factor_corr = df[factors].corr()
    print(factor_corr.round(3).to_string())

    print("\n--- High Correlations (|r| > 0.3) between factors ---")
    for i, f1 in enumerate(factors):
        for j, f2 in enumerate(factors):
            if i < j:
                r = factor_corr.loc[f1, f2]
                if abs(r) > 0.3:
                    print(f"  {f1} <-> {f2}: r = {r:.3f}")

    print("\n--- Factor-to-Target Correlations ---")
    for target in targets:
        print(f"\n  {target}:")
        for factor in factors:
            r = corr_matrix.loc[factor, target]
            strength = "strong" if abs(r) > 0.5 else "moderate" if abs(r) > 0.3 else "weak"
            print(f"    {factor}: r = {r:+.3f} ({strength})")

    return corr_matrix


def calculate_vif_manual(X):
    """Calculate VIF manually using R² from regressing each variable on others"""
    vif_data = []
    for i, col in enumerate(X.columns):
        y = X[col]
        X_other = X.drop(columns=[col])

        if X_other.shape[1] > 0:
            reg = LinearRegression()
            reg.fit(X_other, y)
            r2 = reg.score(X_other, y)
            vif = 1 / (1 - r2) if r2 < 1 else float('inf')
        else:
            vif = 1.0

        vif_data.append({'variable': col, 'VIF': vif})

    return pd.DataFrame(vif_data)


def vif_analysis(df, factors):
    """Calculate Variance Inflation Factors"""
    print("\n" + "="*80)
    print("VARIANCE INFLATION FACTOR (VIF) ANALYSIS")
    print("="*80)
    print("\nVIF measures multicollinearity. Rules of thumb:")
    print("  VIF = 1: No correlation")
    print("  VIF < 5: Moderate correlation, usually acceptable")
    print("  VIF 5-10: High correlation, may be problematic")
    print("  VIF > 10: Severe multicollinearity, problematic")

    X = df[factors].dropna()

    if HAS_STATSMODELS:
        X_const = add_constant(X)
        vif_data = pd.DataFrame()
        vif_data['variable'] = X.columns
        vif_data['VIF'] = [variance_inflation_factor(X_const.values, i+1) for i in range(len(X.columns))]
    else:
        vif_data = calculate_vif_manual(X)

    vif_data = vif_data.sort_values('VIF', ascending=False)

    print("\n--- VIF Results ---")
    for _, row in vif_data.iterrows():
        status = "OK" if row['VIF'] < 5 else "CAUTION" if row['VIF'] < 10 else "PROBLEM"
        print(f"  {row['variable']:8s}: VIF = {row['VIF']:6.2f} [{status}]")

    return vif_data


def regression_analysis(df, factors, target, min_poss=1000):
    """Run regression to predict target from factors"""
    print(f"\n--- Predicting {target} from 6 factors ---")

    # Filter for minimum possessions
    df_filt = df[df['possessions'] >= min_poss].copy()
    print(f"  Players with {min_poss}+ possessions: {len(df_filt)}")

    X = df_filt[factors].values
    y = df_filt[target].values

    # Fit OLS regression
    if HAS_STATSMODELS:
        X_const = add_constant(X)
        model = OLS(y, X_const).fit()

        print(f"\n  R²:          {model.rsquared:.4f}")
        print(f"  Adjusted R²: {model.rsquared_adj:.4f}")
        print(f"  F-statistic: {model.fvalue:.2f} (p = {model.f_pvalue:.2e})")

        print(f"\n  Coefficients:")
        print(f"  {'Variable':12s} {'Coef':>10s} {'Std Err':>10s} {'t':>8s} {'P>|t|':>10s} {'95% CI':>20s}")
        print(f"  {'-'*72}")
        ci = model.conf_int()
        print(f"  {'Intercept':12s} {model.params[0]:10.4f} {model.bse[0]:10.4f} {model.tvalues[0]:8.2f} {model.pvalues[0]:10.4f} [{ci[0,0]:7.3f}, {ci[0,1]:7.3f}]")
        for i, factor in enumerate(factors):
            idx = i + 1
            sig = "***" if model.pvalues[idx] < 0.001 else "**" if model.pvalues[idx] < 0.01 else "*" if model.pvalues[idx] < 0.05 else ""
            print(f"  {factor:12s} {model.params[idx]:10.4f} {model.bse[idx]:10.4f} {model.tvalues[idx]:8.2f} {model.pvalues[idx]:10.4f} [{ci[idx,0]:7.3f}, {ci[idx,1]:7.3f}] {sig}")

        # Get predictions and residuals
        predictions = model.predict(X_const)
        residuals = y - predictions

    else:
        reg = LinearRegression()
        reg.fit(X, y)
        predictions = reg.predict(X)
        residuals = y - predictions
        r2 = r2_score(y, predictions)
        n, p = X.shape
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        print(f"\n  R²:          {r2:.4f}")
        print(f"  Adjusted R²: {adj_r2:.4f}")

        print(f"\n  Coefficients:")
        print(f"  {'Intercept':12s} {reg.intercept_:10.4f}")
        for i, factor in enumerate(factors):
            print(f"  {factor:12s} {reg.coef_[i]:10.4f}")

    # Residual analysis
    df_filt['predicted'] = predictions
    df_filt['residual'] = residuals

    print(f"\n  Residual Statistics:")
    print(f"    Mean:   {residuals.mean():+.4f}")
    print(f"    Std:    {residuals.std():.4f}")
    print(f"    Min:    {residuals.min():+.4f}")
    print(f"    Max:    {residuals.max():+.4f}")

    # Top positive and negative residuals
    print(f"\n  Top 10 Positive Residuals (under-predicted by factors):")
    top_pos = df_filt.nlargest(10, 'residual')[['player_name', target, 'predicted', 'residual', 'possessions']]
    for _, row in top_pos.iterrows():
        print(f"    {row['player_name']:25s} actual={row[target]:+.2f} pred={row['predicted']:+.2f} resid={row['residual']:+.2f} poss={row['possessions']:.0f}")

    print(f"\n  Top 10 Negative Residuals (over-predicted by factors):")
    top_neg = df_filt.nsmallest(10, 'residual')[['player_name', target, 'predicted', 'residual', 'possessions']]
    for _, row in top_neg.iterrows():
        print(f"    {row['player_name']:25s} actual={row[target]:+.2f} pred={row['predicted']:+.2f} resid={row['residual']:+.2f} poss={row['possessions']:.0f}")

    return df_filt[['player_name', target, 'predicted', 'residual', 'possessions']]


def cross_factor_prediction(df, factors, min_poss=1000):
    """Check if offensive factors predict defensive outcomes and vice versa"""
    print("\n" + "="*80)
    print("CROSS-FACTOR PREDICTION ANALYSIS")
    print("="*80)
    print("Do offensive factors predict defensive RAPM? (and vice versa)")
    print("This checks if there are 'two-way' player effects hidden in the data.\n")

    df_filt = df[df['possessions'] >= min_poss].copy()

    off_factors = ['oTS', 'oTOV', 'oREB']
    def_factors = ['dTS', 'dTOV', 'dREB']

    # Offensive factors predicting def
    print("--- Offensive factors predicting DEF RAPM ---")
    X = df_filt[off_factors].values
    y = df_filt['def'].values
    reg = LinearRegression()
    reg.fit(X, y)
    r2 = reg.score(X, y)
    print(f"  R² = {r2:.4f}")
    print(f"  Coefficients: {dict(zip(off_factors, reg.coef_.round(3)))}")

    # Defensive factors predicting off
    print("\n--- Defensive factors predicting OFF RAPM ---")
    X = df_filt[def_factors].values
    y = df_filt['off'].values
    reg = LinearRegression()
    reg.fit(X, y)
    r2 = reg.score(X, y)
    print(f"  R² = {r2:.4f}")
    print(f"  Coefficients: {dict(zip(def_factors, reg.coef_.round(3)))}")

    # All 6 factors predicting off
    print("\n--- All 6 factors predicting OFF RAPM ---")
    X = df_filt[factors].values
    y = df_filt['off'].values
    reg = LinearRegression()
    reg.fit(X, y)
    r2 = reg.score(X, y)
    print(f"  R² = {r2:.4f}")
    print(f"  Coefficients:")
    for f, c in zip(factors, reg.coef_):
        print(f"    {f}: {c:+.4f}")

    # All 6 factors predicting def
    print("\n--- All 6 factors predicting DEF RAPM ---")
    X = df_filt[factors].values
    y = df_filt['def'].values
    reg = LinearRegression()
    reg.fit(X, y)
    r2 = reg.score(X, y)
    print(f"  R² = {r2:.4f}")
    print(f"  Coefficients:")
    for f, c in zip(factors, reg.coef_):
        print(f"    {f}: {c:+.4f}")


def factor_importance_analysis(df, factors, min_poss=1000):
    """Analyze relative importance of each factor"""
    print("\n" + "="*80)
    print("FACTOR IMPORTANCE ANALYSIS")
    print("="*80)
    print("Standardized coefficients show relative importance of each factor.\n")

    df_filt = df[df['possessions'] >= min_poss].copy()

    for target in ['off', 'def', 'net_rapm']:
        print(f"\n--- Standardized Coefficients for {target} ---")

        X = df_filt[factors].values
        y = df_filt[target].values

        # Standardize
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

        reg = LinearRegression()
        reg.fit(X_scaled, y_scaled)

        importance = list(zip(factors, reg.coef_))
        importance.sort(key=lambda x: abs(x[1]), reverse=True)

        print(f"  {'Factor':8s} {'Std Coef':>10s} {'|Coef|':>10s} {'Rank':>6s}")
        print(f"  {'-'*36}")
        for rank, (factor, coef) in enumerate(importance, 1):
            bar = '█' * int(abs(coef) * 20)
            print(f"  {factor:8s} {coef:+10.4f} {abs(coef):10.4f} {rank:6d}  {bar}")


def simple_sum_analysis(df, min_poss=1000):
    """Analyze how well simple sums predict off/def"""
    print("\n" + "="*80)
    print("SIMPLE SUM ANALYSIS")
    print("="*80)
    print("How well does oTS + oTOV + oREB predict off RAPM? (coefficient = 1 assumption)\n")

    df_filt = df[df['possessions'] >= min_poss].copy()

    # Offensive
    df_filt['off_sum'] = df_filt['oTS'] + df_filt['oTOV'] + df_filt['oREB']
    df_filt['off_residual'] = df_filt['off'] - df_filt['off_sum']

    corr_off = df_filt['off'].corr(df_filt['off_sum'])
    mae_off = df_filt['off_residual'].abs().mean()

    print(f"OFF RAPM vs (oTS + oTOV + oREB):")
    print(f"  Correlation: {corr_off:.4f}")
    print(f"  Mean Absolute Error: {mae_off:.3f}")
    print(f"  Mean Residual: {df_filt['off_residual'].mean():+.4f}")
    print(f"  Std Residual: {df_filt['off_residual'].std():.4f}")

    # Defensive
    df_filt['def_sum'] = df_filt['dTS'] + df_filt['dTOV'] + df_filt['dREB']
    df_filt['def_residual'] = df_filt['def'] - df_filt['def_sum']

    corr_def = df_filt['def'].corr(df_filt['def_sum'])
    mae_def = df_filt['def_residual'].abs().mean()

    print(f"\nDEF RAPM vs (dTS + dTOV + dREB):")
    print(f"  Correlation: {corr_def:.4f}")
    print(f"  Mean Absolute Error: {mae_def:.3f}")
    print(f"  Mean Residual: {df_filt['def_residual'].mean():+.4f}")
    print(f"  Std Residual: {df_filt['def_residual'].std():.4f}")

    # Net
    df_filt['net_sum'] = df_filt['off_sum'] - df_filt['def_sum']
    df_filt['net_residual'] = df_filt['net_rapm'] - df_filt['net_sum']

    corr_net = df_filt['net_rapm'].corr(df_filt['net_sum'])
    mae_net = df_filt['net_residual'].abs().mean()

    print(f"\nNET RAPM vs (off_sum - def_sum):")
    print(f"  Correlation: {corr_net:.4f}")
    print(f"  Mean Absolute Error: {mae_net:.3f}")
    print(f"  Mean Residual: {df_filt['net_residual'].mean():+.4f}")
    print(f"  Std Residual: {df_filt['net_residual'].std():.4f}")

    # Biggest discrepancies
    print(f"\n--- Top 15 Players: OFF RAPM much higher than factor sum ---")
    top = df_filt.nlargest(15, 'off_residual')[['player_name', 'off', 'off_sum', 'off_residual', 'oTS', 'oTOV', 'oREB', 'possessions']]
    print(top.to_string(index=False))

    print(f"\n--- Top 15 Players: OFF RAPM much lower than factor sum ---")
    bottom = df_filt.nsmallest(15, 'off_residual')[['player_name', 'off', 'off_sum', 'off_residual', 'oTS', 'oTOV', 'oREB', 'possessions']]
    print(bottom.to_string(index=False))


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Factor Analysis for RAPM')
    parser.add_argument('--folder', type=str, default='rapm_24_26_all',
                        help='Results folder name (e.g., rapm_22_26_all, rapm_24_26_all)')
    parser.add_argument('--min-poss', type=int, default=1000,
                        help='Minimum possessions filter')
    args = parser.parse_args()

    # Configuration
    base_dir = '/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/results'
    folder_name = args.folder
    # Extract the suffix for the weighted_factors file (e.g., 24_26_all from rapm_24_26_all)
    suffix = folder_name.replace('rapm_', '')
    filepath = os.path.join(base_dir, folder_name, f'weighted_factors_{suffix}.csv')

    factors = ['oTS', 'oTOV', 'oREB', 'dTS', 'dTOV', 'dREB']
    targets = ['off', 'def', 'net_rapm']
    min_possessions = args.min_poss

    print("="*80)
    print("RAPM FACTOR ANALYSIS")
    print("="*80)
    print(f"Data: {filepath}")
    print(f"Factors: {factors}")
    print(f"Targets: {targets}")
    print(f"Min Possessions Filter: {min_possessions}")

    # Load data
    df = load_data(filepath)
    print(f"\nLoaded {len(df)} players")
    print(f"Players with {min_possessions}+ possessions: {len(df[df['possessions'] >= min_possessions])}")

    # Run analyses
    corr_matrix = correlation_analysis(df, factors, targets)
    vif_data = vif_analysis(df, factors)

    print("\n" + "="*80)
    print("REGRESSION ANALYSIS: Predicting RAPM from Factors")
    print("="*80)

    for target in targets:
        regression_analysis(df, factors, target, min_possessions)

    cross_factor_prediction(df, factors, min_possessions)
    factor_importance_analysis(df, factors, min_possessions)
    simple_sum_analysis(df, min_possessions)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
