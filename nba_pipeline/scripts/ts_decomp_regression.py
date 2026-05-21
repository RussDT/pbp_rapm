#!/usr/bin/env python3
"""
TS Decomposition Regression

Decomposes oTS / dTS (True Shooting RAPM) into three components:
  SQ      - Shot Quality   (FGA=initial_ev, FT=1.0)
  FT      - FT Premium     (FGA=1.0,        FT=ExpFT)
  CONTEST - Contest/Exec   (FGA=actual-EV,  FT=1.0)

Runs WLS regression separately for offense and defense:
  off_ts ~ off_sq + off_ft + off_contest  (weights = off_poss)
  def_ts ~ def_sq + def_ft + def_contest  (weights = def_poss)

Outputs weighted factor contributions and regression diagnostics.

Usage (from nba_pipeline/scripts/):
    python ts_decomp_regression.py 24 26 ALL
    python ts_decomp_regression.py 24 26 ALL --timedecay
    python ts_decomp_regression.py 24 26 ALL --min-poss 5000
"""

import argparse
import logging
import sys
import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PIPELINE_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PIPELINE_ROOT / "results"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def find_results_file(prefix, year_range, season_suffix, timedecay=False, half_life=700):
    """
    Locate a rapm.py results CSV by trying common locations.
    Returns Path or raises FileNotFoundError.
    """
    td_str = f"_td{int(half_life)}" if timedecay else ""
    filename = f"{prefix}_{year_range}_{season_suffix}{td_str}_results.csv"

    search_dirs = []
    ts_decomp_dir = RESULTS_DIR / f"ts_decomp_{year_range}_{season_suffix}"
    if timedecay:
        search_dirs.append(RESULTS_DIR / "td")
        search_dirs.append(RESULTS_DIR / "td" / f"rapm_{year_range}_{season_suffix}_td{int(half_life)}")
        search_dirs.append(ts_decomp_dir)
    else:
        search_dirs.append(RESULTS_DIR)
        search_dirs.append(ts_decomp_dir)
        search_dirs.append(RESULTS_DIR / f"rapm_{year_range}_{season_suffix}")
        search_dirs.append(RESULTS_DIR / f"rapm_{year_range}_{season_suffix}_all")

    for d in search_dirs:
        candidate = d / filename
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find {filename} in: {[str(d) for d in search_dirs]}"
    )


def run_regression(y, X, weights, label):
    """Fit WLS regression, log summary, return results."""
    X_const = sm.add_constant(X)
    model = sm.WLS(y, X_const, weights=weights)
    results = model.fit()

    logging.info(f"\n--- {label} Regression ---")
    logging.info(f"  R²:          {results.rsquared:.4f}")
    logging.info(f"  Adj R²:      {results.rsquared_adj:.4f}")
    logging.info(f"  N:           {len(y)}")
    logging.info(f"  Coefficients:")
    for name, coef, se, pval in zip(
        results.params.index, results.params, results.bse, results.pvalues
    ):
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        logging.info(f"    {name:12s}  {coef:+.4f}  (se={se:.4f}, p={pval:.3f}) {sig}")
    return results


def main():
    parser = argparse.ArgumentParser(description='TS Decomposition Regression')
    parser.add_argument('start_year', type=int, help='Start year (2-digit, e.g., 24)')
    parser.add_argument('end_year', type=int, help='End year (2-digit, e.g., 26)')
    parser.add_argument('season_type', choices=['RS', 'PS', 'ALL'], help='Season type')
    parser.add_argument('--timedecay', '-td', action='store_true', help='Use time-decay variants')
    parser.add_argument('--half-life', type=int, default=700, help='Time decay half-life (default 700)')
    parser.add_argument('--min-poss', type=int, default=3000, help='Min possessions filter (default 3000)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: auto-named folder in results/)')
    args = parser.parse_args()

    season_suffix = args.season_type.lower()
    yr = f"{args.start_year:02d}_{args.end_year:02d}" if args.start_year != args.end_year else f"{args.start_year:02d}"
    td = args.timedecay
    hl = args.half_life
    td_str = f"_td{hl}" if td else ""

    logging.info("=" * 70)
    logging.info("TS DECOMPOSITION REGRESSION")
    logging.info("=" * 70)
    logging.info(f"  Years:        {yr}, Season: {season_suffix}")
    logging.info(f"  Time decay:   {'YES (half-life=' + str(hl) + ')' if td else 'NO'}")
    logging.info(f"  Min poss:     {args.min_poss}")

    # --- Load files ---
    try:
        ts_path = find_results_file("ts", yr, season_suffix, timedecay=td, half_life=hl)
        sq_path = find_results_file("sqposs", yr, season_suffix, timedecay=td, half_life=hl)
        ft_path = find_results_file("ftpremium", yr, season_suffix, timedecay=td, half_life=hl)
        ct_path = find_results_file("contest", yr, season_suffix, timedecay=td, half_life=hl)
    except FileNotFoundError as e:
        logging.error(str(e))
        sys.exit(1)

    logging.info(f"\nLoading files:")
    for label, path in [("TS", ts_path), ("SQ_POSS", sq_path), ("FT_PREMIUM", ft_path), ("CONTEST", ct_path)]:
        logging.info(f"  {label}: {path.relative_to(PIPELINE_ROOT)}")

    ts_df = pd.read_csv(ts_path)
    sq_df = pd.read_csv(sq_path)
    ft_df = pd.read_csv(ft_path)
    ct_df = pd.read_csv(ct_path)

    logging.info(f"\nRows: TS={len(ts_df)}, SQ={len(sq_df)}, FT={len(ft_df)}, CT={len(ct_df)}")

    # --- Merge ---
    merged = ts_df[['player_id', 'player_name', 'off', 'def', 'net_rapm', 'possessions', 'off_poss', 'def_poss']].copy()
    merged.rename(columns={'off': 'off_ts', 'def': 'def_ts', 'net_rapm': 'net_ts'}, inplace=True)

    for df, prefix in [(sq_df, 'sq'), (ft_df, 'ft'), (ct_df, 'contest')]:
        sub = df[['player_id', 'off', 'def']].rename(
            columns={'off': f'off_{prefix}', 'def': f'def_{prefix}'}
        )
        merged = merged.merge(sub, on='player_id', how='inner')

    logging.info(f"Merged: {len(merged)} players")

    # --- Filter ---
    filt = merged[merged['possessions'] >= args.min_poss].copy()
    logging.info(f"After {args.min_poss}+ poss filter: {len(filt)} players")

    # --- Regressions ---
    off_features = ['off_sq', 'off_ft', 'off_contest']
    def_features = ['def_sq', 'def_ft', 'def_contest']

    off_results = run_regression(
        filt['off_ts'], filt[off_features], filt['off_poss'],
        "OFFENSIVE: off_ts ~ off_sq + off_ft + off_contest"
    )

    def_results = run_regression(
        filt['def_ts'], filt[def_features], filt['def_poss'],
        "DEFENSIVE: def_ts ~ def_sq + def_ft + def_contest"
    )

    # --- Weighted factors on FULL merged set ---
    def apply_weights(df, results, features, prefix_map):
        """Apply regression coefficients to get weighted component contributions."""
        out = {}
        for feat, out_col in prefix_map.items():
            coef = results.params.get(feat, 0.0)
            out[out_col] = (df[feat] * coef).round(2)
        return out

    off_map = {'off_sq': 'oSQ', 'off_ft': 'oFT', 'off_contest': 'oCONTEST'}
    def_map = {'def_sq': 'dSQ', 'def_ft': 'dFT', 'def_contest': 'dCONTEST'}

    output = merged[['player_id', 'player_name', 'possessions', 'off_poss', 'def_poss']].copy()
    output['oTS'] = merged['off_ts'].round(2)
    output['dTS'] = merged['def_ts'].round(2)
    output['net_ts'] = merged['net_ts'].round(2)

    for col, val in apply_weights(merged, off_results, list(off_map.keys()), off_map).items():
        output[col] = val
    for col, val in apply_weights(merged, def_results, list(def_map.keys()), def_map).items():
        output[col] = val

    output['oRESID'] = (output['oTS'] - (output['oSQ'] + output['oFT'] + output['oCONTEST'])).round(2)
    output['dRESID'] = (output['dTS'] - (output['dSQ'] + output['dFT'] + output['dCONTEST'])).round(2)

    output = output.sort_values('net_ts', ascending=False)

    # --- Output directory ---
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        folder_name = f"ts_decomp_{yr}_{season_suffix}{td_str}"
        out_dir = RESULTS_DIR / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save weighted factors
    wf_path = out_dir / f"ts_decomp_factors_{yr}_{season_suffix}{td_str}.csv"
    output.to_csv(wf_path, index=False)
    logging.info(f"\nSaved weighted factors → {wf_path.relative_to(PIPELINE_ROOT)}")

    # Save regression coefficients
    for label, results, features in [
        ('off', off_results, off_features),
        ('def', def_results, def_features)
    ]:
        coef_rows = []
        for k in ['const'] + features:
            coef_rows.append({
                'variable': k,
                'coefficient': results.params[k],
                'std_error': results.bse[k],
                'p_value': results.pvalues[k],
                'ci_lower': results.conf_int().loc[k, 0],
                'ci_upper': results.conf_int().loc[k, 1],
            })
        coef_df = pd.DataFrame(coef_rows)
        coef_df.loc[len(coef_df)] = {'variable': 'R_squared', 'coefficient': results.rsquared}
        coef_path = out_dir / f"ts_decomp_{label}_coefficients_{yr}_{season_suffix}{td_str}.csv"
        coef_df.to_csv(coef_path, index=False)
        logging.info(f"Saved {label} coefficients → {coef_path.name}")

    # --- Print top 20 by dTS ---
    logging.info("\n--- Top 20 Defenders (dTS) ---")
    logging.info(f"{'Player':28s} {'dTS':>6} {'dSQ':>6} {'dFT':>6} {'dCONT':>7} {'dRES':>6}  poss")
    top_def = output.nsmallest(20, 'dTS')
    for _, row in top_def.iterrows():
        logging.info(
            f"  {row['player_name']:26s} {row['dTS']:+6.2f} {row['dSQ']:+6.2f} "
            f"{row['dFT']:+6.2f} {row['dCONTEST']:+7.2f} {row['dRESID']:+6.2f}  {int(row['possessions']):>5}"
        )

    logging.info("\nDone!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
