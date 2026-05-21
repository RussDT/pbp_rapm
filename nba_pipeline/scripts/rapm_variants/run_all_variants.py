#!/usr/bin/env python3
"""
RAPM Variants Orchestrator

Runs all RAPM variants with forward-looking backtesting:
1. Baseline (7 configs)
2. Time-decay (21 configs)
3. Possession-weighted ridge (105 configs)
4. Time-decay + Possession-weighted ridge (315 configs)

Total: 448 configurations across 9 folds = 4,032 evaluations

Outputs:
- Per-variant results in nba_pipeline/results/rapm_variants/[variant]_results/
- Master comparison in nba_pipeline/results/rapm_variants/master_results.csv
"""

import logging
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from .base_rapm import (
    BACKTEST_FOLDS,
    RESULTS_DIR,
    load_rapm_data,
    prepare_data,
    get_all_players,
    build_player_mapping,
    get_player_possessions_fast,
)
from .rapm_baseline import (
    run_baseline_backtest,
    train_and_save_final_model as train_baseline_final,
    generate_configs as baseline_configs,
    train_baseline,
)
from .rapm_timedecay import (
    run_timedecay_backtest,
    train_and_save_final_model as train_timedecay_final,
    generate_configs as timedecay_configs,
    train_timedecay,
)
from .rapm_possridge import (
    run_possridge_backtest,
    train_and_save_final_model as train_possridge_final,
    generate_configs as possridge_configs,
    train_possridge,
)
from .rapm_timedecay_possridge import (
    run_timedecay_possridge_backtest,
    train_and_save_final_model as train_combined_final,
    generate_configs as combined_configs,
    train_timedecay_possridge,
)
from .backtest import (
    compute_stint_predictions_fast,
    compute_rmse,
    compute_rmse_by_possession_bin,
    train_final_model,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


###############################################################################
# ORCHESTRATION
###############################################################################

def run_all_variants(
    quick_mode: bool = False,
    variants: Optional[List[str]] = None,
    skip_backtest: bool = False,
    final_train_years: List[int] = [22, 23, 24, 25]
):
    """
    Run all RAPM variants.

    Args:
        quick_mode: If True, use only 2 folds for faster testing
        variants: List of variants to run (default: all)
        skip_backtest: Skip backtest and use default configs
        final_train_years: Years to use for final model training
    """
    all_variants = ['baseline', 'timedecay', 'possridge', 'timedecay_possridge']

    if variants is None:
        variants = all_variants

    folds = BACKTEST_FOLDS[:2] if quick_mode else None

    results = {}

    logging.info("=" * 80)
    logging.info("RAPM VARIANTS ORCHESTRATOR")
    logging.info("=" * 80)
    logging.info(f"Variants to run: {variants}")
    logging.info(f"Quick mode: {quick_mode}")
    logging.info(f"Folds: {len(folds) if folds else len(BACKTEST_FOLDS)}")

    # Count total configs
    total_configs = 0
    for v in variants:
        if v == 'baseline':
            total_configs += len(baseline_configs())
        elif v == 'timedecay':
            total_configs += len(timedecay_configs())
        elif v == 'possridge':
            total_configs += len(possridge_configs())
        elif v == 'timedecay_possridge':
            total_configs += len(combined_configs())

    n_folds = len(folds) if folds else len(BACKTEST_FOLDS)
    total_evals = total_configs * n_folds

    logging.info(f"Total configs: {total_configs}")
    logging.info(f"Total evaluations: {total_evals}")
    logging.info("=" * 80)

    # Run each variant
    if 'baseline' in variants:
        logging.info("\n" + "=" * 60)
        logging.info("RUNNING BASELINE VARIANT")
        logging.info("=" * 60)

        if skip_backtest:
            best_config = {'lambda_off': 3000, 'lambda_def': 4500}
        else:
            best_config = run_baseline_backtest(folds=folds)

        train_baseline_final(best_config, train_years=final_train_years)
        results['baseline'] = best_config

    if 'timedecay' in variants:
        logging.info("\n" + "=" * 60)
        logging.info("RUNNING TIME-DECAY VARIANT")
        logging.info("=" * 60)

        if skip_backtest:
            best_config = {'lambda_off': 3000, 'lambda_def': 4500, 'decay_base': 0.999}
        else:
            best_config = run_timedecay_backtest(folds=folds)

        train_timedecay_final(best_config, train_years=final_train_years)
        results['timedecay'] = best_config

    if 'possridge' in variants:
        logging.info("\n" + "=" * 60)
        logging.info("RUNNING POSSESSION-WEIGHTED RIDGE VARIANT")
        logging.info("=" * 60)

        if skip_backtest:
            best_config = {'lambda_off': 3000, 'lambda_def': 4500, 'p': 0.5, 'm_max': 3.0}
        else:
            best_config = run_possridge_backtest(folds=folds)

        train_possridge_final(best_config, train_years=final_train_years)
        results['possridge'] = best_config

    if 'timedecay_possridge' in variants:
        logging.info("\n" + "=" * 60)
        logging.info("RUNNING TIME-DECAY + POSSESSION-WEIGHTED RIDGE VARIANT")
        logging.info("=" * 60)

        if skip_backtest:
            best_config = {
                'lambda_off': 3000,
                'lambda_def': 4500,
                'decay_base': 0.999,
                'p': 0.5,
                'm_max': 3.0
            }
        else:
            best_config = run_timedecay_possridge_backtest(folds=folds)

        train_combined_final(best_config, train_years=final_train_years)
        results['timedecay_possridge'] = best_config

    # Generate master results
    generate_master_results(results, variants)

    return results


###############################################################################
# MASTER RESULTS
###############################################################################

def generate_master_results(
    best_configs: Dict[str, Dict[str, Any]],
    variants: List[str]
):
    """
    Generate master_results.csv comparing all variants.
    """
    logging.info("\n" + "=" * 60)
    logging.info("GENERATING MASTER RESULTS")
    logging.info("=" * 60)

    rows = []

    for variant in variants:
        if variant not in best_configs:
            continue

        config = best_configs[variant]

        # Read the config_summary.csv from each variant
        summary_file = RESULTS_DIR / f"{variant}_results" / "config_summary.csv"

        if summary_file.exists():
            summary_df = pd.read_csv(summary_file)
            # Get best row (already sorted by avg_rmse)
            best_row = summary_df.iloc[0]

            row = {
                'variant': variant,
                'best_lambda_def': config.get('lambda_def'),
                'best_decay_base': config.get('decay_base'),
                'best_p': config.get('p'),
                'best_m_max': config.get('m_max'),
                'avg_rmse': best_row['avg_rmse'],
                'std_rmse': best_row['std_rmse'],
                'avg_coverage': best_row.get('avg_coverage', None)
            }
            rows.append(row)

    if rows:
        master_df = pd.DataFrame(rows)
        master_df = master_df.sort_values('avg_rmse')

        master_file = RESULTS_DIR / 'master_results.csv'
        master_df.to_csv(master_file, index=False)
        logging.info(f"Saved master results to {master_file}")

        # Print summary
        logging.info("\n" + "=" * 60)
        logging.info("VARIANT COMPARISON")
        logging.info("=" * 60)

        for _, row in master_df.iterrows():
            logging.info(
                f"{row['variant']:25s} | "
                f"RMSE: {row['avg_rmse']:.4f} +/- {row['std_rmse']:.4f}"
            )

        # Best overall
        best = master_df.iloc[0]
        logging.info("\n" + "-" * 60)
        logging.info(f"BEST VARIANT: {best['variant']}")
        logging.info(f"  Avg RMSE: {best['avg_rmse']:.4f}")
        logging.info("-" * 60)


###############################################################################
# DIAGNOSTIC ANALYSIS
###############################################################################

def run_diagnostic_analysis(
    variant: str,
    best_config: Dict[str, Any],
    train_years: List[int] = [22, 23, 24, 25],
    eval_year: int = 25
):
    """
    Run diagnostic analysis for a variant:
    - RMSE by possession bins
    - Year-to-year correlation (if applicable)
    """
    logging.info(f"\nRunning diagnostics for {variant}...")

    # Get training function
    train_funcs = {
        'baseline': train_baseline,
        'timedecay': train_timedecay,
        'possridge': train_possridge,
        'timedecay_possridge': train_timedecay_possridge,
    }
    train_func = train_funcs[variant]

    # Train model
    beta, players, player_to_col, df_train = train_final_model(
        train_func=train_func,
        best_config=best_config,
        train_years=train_years
    )

    total_poss, _, _ = get_player_possessions_fast(df_train, players)

    # Load eval data
    df_eval_raw = load_rapm_data([eval_year])
    df_eval = prepare_data(df_eval_raw)

    # RMSE by possession bins
    poss_bins = [0, 500, 1000, 2000, 5000, float('inf')]
    bin_df = compute_rmse_by_possession_bin(
        df_eval, beta, player_to_col, total_poss, bins=poss_bins
    )

    logging.info("\nRMSE by Possession Bins:")
    for _, row in bin_df.iterrows():
        logging.info(f"  {row['bin']:15s}: RMSE={row['rmse']:.4f} (n={row['n_stints']})")

    # Save diagnostic
    output_dir = RESULTS_DIR / f"{variant}_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    diag_file = output_dir / 'rmse_by_possession_bin.csv'
    bin_df.to_csv(diag_file, index=False)

    return bin_df


###############################################################################
# CLI
###############################################################################

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Run all RAPM variants with backtesting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all variants with full backtesting
  python -m rapm_variants.run_all_variants

  # Quick test (2 folds only)
  python -m rapm_variants.run_all_variants --quick

  # Run specific variants
  python -m rapm_variants.run_all_variants --variants baseline timedecay

  # Skip backtest, use default configs
  python -m rapm_variants.run_all_variants --skip-backtest
        """
    )

    parser.add_argument('--quick', action='store_true',
                        help='Quick test with 2 folds only')
    parser.add_argument('--variants', nargs='+', default=None,
                        choices=['baseline', 'timedecay', 'possridge', 'timedecay_possridge'],
                        help='Specific variants to run')
    parser.add_argument('--skip-backtest', action='store_true',
                        help='Skip backtest, use default configs')
    parser.add_argument('--train-years', nargs='+', type=int, default=[22, 23, 24, 25],
                        help='Years for final model training')
    parser.add_argument('--diagnostics', action='store_true',
                        help='Run diagnostic analysis after backtesting')

    args = parser.parse_args()

    results = run_all_variants(
        quick_mode=args.quick,
        variants=args.variants,
        skip_backtest=args.skip_backtest,
        final_train_years=args.train_years
    )

    if args.diagnostics:
        for variant, config in results.items():
            run_diagnostic_analysis(
                variant=variant,
                best_config=config,
                train_years=args.train_years
            )
