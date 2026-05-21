#!/usr/bin/env python3
"""
Run player-level RAPM prediction evaluation across all variants.

This is a better evaluation metric than stint-level RMSE because:
1. It directly tests player impact estimation accuracy
2. It shows meaningful differences between hyperparameters
3. It correlates with the actual use case (estimating player value)

Usage:
    python -m rapm_variants.run_player_prediction_eval [--quick]
"""

import argparse
import logging
import pandas as pd
from pathlib import Path

from .base_rapm import RESULTS_DIR, BACKTEST_FOLDS
from .backtest import run_player_prediction_backtest

# Variant imports
from .rapm_baseline import train_baseline, generate_configs as baseline_generate_configs
from .rapm_timedecay import train_timedecay, generate_configs as timedecay_generate_configs
from .rapm_possridge import train_possridge, generate_configs as possridge_generate_configs
from .rapm_timedecay_possridge import train_timedecay_possridge, generate_configs as combined_generate_configs

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def run_all_player_prediction_evals(quick: bool = False):
    """Run player prediction evaluation for all variants."""

    # Use subset of folds for quick testing
    folds = BACKTEST_FOLDS[:2] if quick else BACKTEST_FOLDS

    logging.info("=" * 70)
    logging.info("PLAYER PREDICTION EVALUATION")
    logging.info(f"Folds: {len(folds)}")
    logging.info(f"Quick mode: {quick}")
    logging.info("=" * 70)

    results = {}

    # Generate config lists
    baseline_configs = baseline_generate_configs()
    timedecay_configs = timedecay_generate_configs()
    possridge_configs = possridge_generate_configs()
    combined_configs = combined_generate_configs()

    # 1. Baseline
    logging.info("\n" + "=" * 70)
    logging.info("VARIANT 1: BASELINE")
    logging.info("=" * 70)
    best_config, all_results = run_player_prediction_backtest(
        train_func=train_baseline,
        configs=baseline_configs,
        variant_name="baseline",
        folds=folds,
        min_poss=500,
        include_playoffs=True
    )
    results['baseline'] = {
        'best_config': best_config,
        'avg_pearson': max(r.avg_pearson_net for r in all_results),
        'avg_spearman': max(r.avg_spearman_net for r in all_results),
        'avg_rmse': min(r.avg_rmse_net for r in all_results)
    }

    # 2. Time Decay
    logging.info("\n" + "=" * 70)
    logging.info("VARIANT 2: TIME DECAY")
    logging.info("=" * 70)
    best_config, all_results = run_player_prediction_backtest(
        train_func=train_timedecay,
        configs=timedecay_configs,
        variant_name="timedecay",
        folds=folds,
        min_poss=500,
        include_playoffs=True
    )
    results['timedecay'] = {
        'best_config': best_config,
        'avg_pearson': max(r.avg_pearson_net for r in all_results),
        'avg_spearman': max(r.avg_spearman_net for r in all_results),
        'avg_rmse': min(r.avg_rmse_net for r in all_results)
    }

    # 3. Possession Ridge
    logging.info("\n" + "=" * 70)
    logging.info("VARIANT 3: POSSESSION RIDGE")
    logging.info("=" * 70)
    # Use subset of configs for quick mode
    pr_configs = possridge_configs[:15] if quick else possridge_configs
    best_config, all_results = run_player_prediction_backtest(
        train_func=train_possridge,
        configs=pr_configs,
        variant_name="possridge",
        folds=folds,
        min_poss=500,
        include_playoffs=True
    )
    results['possridge'] = {
        'best_config': best_config,
        'avg_pearson': max(r.avg_pearson_net for r in all_results),
        'avg_spearman': max(r.avg_spearman_net for r in all_results),
        'avg_rmse': min(r.avg_rmse_net for r in all_results)
    }

    # 4. Time Decay + Possession Ridge
    logging.info("\n" + "=" * 70)
    logging.info("VARIANT 4: TIME DECAY + POSSESSION RIDGE")
    logging.info("=" * 70)
    # Use subset of configs for quick mode
    cb_configs = combined_configs[:20] if quick else combined_configs
    best_config, all_results = run_player_prediction_backtest(
        train_func=train_timedecay_possridge,
        configs=cb_configs,
        variant_name="timedecay_possridge",
        folds=folds,
        min_poss=500,
        include_playoffs=True
    )
    results['timedecay_possridge'] = {
        'best_config': best_config,
        'avg_pearson': max(r.avg_pearson_net for r in all_results),
        'avg_spearman': max(r.avg_spearman_net for r in all_results),
        'avg_rmse': min(r.avg_rmse_net for r in all_results)
    }

    # Create master results
    logging.info("\n" + "=" * 70)
    logging.info("MASTER RESULTS (PLAYER PREDICTION)")
    logging.info("=" * 70)

    master_rows = []
    for variant, data in results.items():
        row = {
            'variant': variant,
            'best_lambda_def': data['best_config'].get('lambda_def'),
            'best_decay_base': data['best_config'].get('decay_base'),
            'best_p': data['best_config'].get('p'),
            'best_m_max': data['best_config'].get('m_max'),
            'avg_pearson': data['avg_pearson'],
            'avg_spearman': data['avg_spearman'],
            'avg_rmse': data['avg_rmse']
        }
        master_rows.append(row)
        logging.info(f"{variant}: Pearson={data['avg_pearson']:.4f}, config={data['best_config']}")

    master_df = pd.DataFrame(master_rows)
    master_df = master_df.sort_values('avg_pearson', ascending=False)

    master_file = RESULTS_DIR / 'master_player_prediction_results.csv'
    master_df.to_csv(master_file, index=False)

    logging.info(f"\nMaster results saved to {master_file}")
    print("\n" + master_df.to_string(index=False))

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run player prediction evaluation for RAPM variants"
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Quick mode: use 2 folds and fewer configs'
    )
    args = parser.parse_args()

    run_all_player_prediction_evals(quick=args.quick)


if __name__ == '__main__':
    main()
