#!/usr/bin/env python3
"""
RAPM Baseline Variant

Standard alternating ridge regression:
- Offense lambda: FIXED at 3000
- Defense lambda: Tuned over grid [1500, 2500, 3000, 4500, 6000, 8000, 10000]
- No time decay
- No possession weighting

Grid size: 7 configs
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any

from .base_rapm import (
    LAMBDA_OFF_BASE,
    LAMBDA_DEF_GRID,
    load_rapm_data,
    prepare_data,
    get_all_players,
    build_player_mapping,
    build_design_matrix,
    get_player_possessions_fast,
    recenter_coefficients,
    extract_coefficients,
    format_results,
    load_name_map,
    RESULTS_DIR,
)
from .possession_ridge import alternating_ridge
from .backtest import run_backtest, train_final_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

VARIANT_NAME = "baseline"


###############################################################################
# TRAINING FUNCTION
###############################################################################

def train_baseline(
    df_train: pd.DataFrame,
    players: List[str],
    player_to_col: Dict[str, int],
    config: Dict[str, Any]
) -> np.ndarray:
    """
    Train baseline RAPM model.

    Args:
        df_train: Prepared training DataFrame
        players: List of player IDs
        player_to_col: Column mapping
        config: Must contain 'lambda_def'

    Returns:
        Coefficient vector (beta)
    """
    lambda_off = LAMBDA_OFF_BASE  # Fixed at 3000
    lambda_def = config['lambda_def']

    # Build design matrix
    X, y = build_design_matrix(df_train, player_to_col)

    # Center y
    y_mean = np.mean(y)
    y_centered = y - y_mean

    # Run alternating ridge
    beta = alternating_ridge(
        X=X,
        y=y_centered,
        player_to_col=player_to_col,
        alpha_offense=float(lambda_off),
        alpha_defense=float(lambda_def),
        max_iter=200,
        tol=1e-4
    )

    # Re-center coefficients
    total_poss, _, _ = get_player_possessions_fast(df_train, players)
    beta = recenter_coefficients(beta, players, player_to_col, total_poss)

    return beta


###############################################################################
# CONFIG GENERATION
###############################################################################

def generate_configs() -> List[Dict[str, Any]]:
    """Generate all hyperparameter configurations for baseline."""
    configs = []
    for lambda_def in LAMBDA_DEF_GRID:
        configs.append({
            'lambda_off': LAMBDA_OFF_BASE,
            'lambda_def': lambda_def
        })
    return configs


###############################################################################
# MAIN
###############################################################################

def run_baseline_backtest(
    folds=None,
    save_results: bool = True
) -> Dict[str, Any]:
    """
    Run full backtest for baseline variant.

    Returns:
        best_config with lowest average RMSE
    """
    configs = generate_configs()
    logging.info(f"Baseline configs: {len(configs)}")

    best_config, all_results = run_backtest(
        train_func=train_baseline,
        configs=configs,
        variant_name=VARIANT_NAME,
        folds=folds,
        save_results=save_results
    )

    return best_config


def train_and_save_final_model(
    best_config: Dict[str, Any],
    train_years: List[int] = [22, 23, 24, 25]
):
    """
    Train final model with best config and save player impacts.
    """
    logging.info(f"Training final baseline model with config: {best_config}")
    logging.info(f"Training years: {train_years}")

    beta, players, player_to_col, df_train = train_final_model(
        train_func=train_baseline,
        best_config=best_config,
        train_years=train_years
    )

    # Get possessions
    total_poss, off_poss, def_poss = get_player_possessions_fast(df_train, players)

    # Extract and format coefficients
    coef_df = extract_coefficients(beta, players, player_to_col)
    name_map = load_name_map()
    results_df = format_results(coef_df, total_poss, off_poss, def_poss, name_map)

    # Save
    output_dir = RESULTS_DIR / f"{VARIANT_NAME}_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / 'best_model_player_impacts.csv'
    results_df.to_csv(output_file, index=False)
    logging.info(f"Saved player impacts to {output_file}")

    # Print top 20
    logging.info("\nTop 20 by net RAPM:")
    for i, row in results_df.head(20).iterrows():
        logging.info(
            f"{i+1:2d}. {row['player_name']:25s} | "
            f"Net: {row['net_rapm']:+6.2f} | "
            f"Off: {row['off']:+6.2f} | "
            f"Def: {row['def']:+6.2f} | "
            f"Poss: {row['possessions']:6d}"
        )

    return results_df


###############################################################################
# CLI
###############################################################################

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run baseline RAPM variant')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test with 2 folds only')
    parser.add_argument('--skip-backtest', action='store_true',
                        help='Skip backtest, use default config')
    parser.add_argument('--lambda-def', type=float, default=None,
                        help='Override defense lambda')

    args = parser.parse_args()

    if args.skip_backtest:
        # Use default or specified config
        best_config = {
            'lambda_off': LAMBDA_OFF_BASE,
            'lambda_def': args.lambda_def if args.lambda_def else 4500
        }
    else:
        # Run backtest
        if args.quick:
            from .base_rapm import BACKTEST_FOLDS
            folds = BACKTEST_FOLDS[:2]
        else:
            folds = None

        best_config = run_baseline_backtest(folds=folds)

    # Train and save final model
    train_and_save_final_model(best_config)
