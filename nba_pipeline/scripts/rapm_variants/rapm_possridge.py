#!/usr/bin/env python3
"""
RAPM Possession-Weighted Ridge Variant

Per-coefficient ridge penalties based on possession counts:
- Multiplier: m_j = clamp((N_ref / (N_j + N0))^p, m_min, m_max)
- Per-player lambda: lambda_j = lambda_base * m_j
- Applied to BOTH offense and defense coefficients
- Offense BASE lambda: FIXED at 3000
- No time decay

Tune: (lambda_def_base, p, m_max)
Grid size: 7 x 5 x 3 = 105 configs
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any

from .base_rapm import (
    LAMBDA_OFF_BASE,
    LAMBDA_DEF_GRID,
    P_GRID,
    M_MAX_GRID,
    N_REF,
    N0,
    M_MIN,
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
from .possession_ridge import alternating_possession_ridge
from .backtest import run_backtest, train_final_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

VARIANT_NAME = "possridge"


###############################################################################
# TRAINING FUNCTION
###############################################################################

def train_possridge(
    df_train: pd.DataFrame,
    players: List[str],
    player_to_col: Dict[str, int],
    config: Dict[str, Any]
) -> np.ndarray:
    """
    Train possession-weighted ridge RAPM model.

    Args:
        df_train: Prepared training DataFrame
        players: List of player IDs
        player_to_col: Column mapping
        config: Must contain 'lambda_def', 'p', 'm_max'

    Returns:
        Coefficient vector (beta)
    """
    lambda_off_base = LAMBDA_OFF_BASE  # Fixed at 3000
    lambda_def_base = config['lambda_def']
    p = config['p']
    m_max = config['m_max']

    # Build design matrix
    X, y = build_design_matrix(df_train, player_to_col)

    # Get possession counts (needed for per-player penalties)
    total_poss, off_poss, def_poss = get_player_possessions_fast(df_train, players)

    # Center y
    y_mean = np.mean(y)
    y_centered = y - y_mean

    # Run alternating possession-weighted ridge
    beta = alternating_possession_ridge(
        X=X,
        y=y_centered,
        players=players,
        player_to_col=player_to_col,
        off_poss=off_poss,
        def_poss=def_poss,
        lambda_off_base=float(lambda_off_base),
        lambda_def_base=float(lambda_def_base),
        p=float(p),
        m_max=float(m_max),
        max_iter=200,
        tol=1e-4
    )

    # Re-center coefficients
    beta = recenter_coefficients(beta, players, player_to_col, total_poss)

    return beta


###############################################################################
# CONFIG GENERATION
###############################################################################

def generate_configs() -> List[Dict[str, Any]]:
    """Generate all hyperparameter configurations for possession-weighted ridge."""
    configs = []
    for lambda_def in LAMBDA_DEF_GRID:
        for p in P_GRID:
            for m_max in M_MAX_GRID:
                configs.append({
                    'lambda_off': LAMBDA_OFF_BASE,
                    'lambda_def': lambda_def,
                    'p': p,
                    'm_max': m_max
                })
    return configs


###############################################################################
# MAIN
###############################################################################

def run_possridge_backtest(
    folds=None,
    save_results: bool = True
) -> Dict[str, Any]:
    """
    Run full backtest for possession-weighted ridge variant.

    Returns:
        best_config with lowest average RMSE
    """
    configs = generate_configs()
    logging.info(f"Possession-weighted ridge configs: {len(configs)}")

    best_config, all_results = run_backtest(
        train_func=train_possridge,
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
    logging.info(f"Training final possession-weighted ridge model with config: {best_config}")
    logging.info(f"Training years: {train_years}")

    beta, players, player_to_col, df_train = train_final_model(
        train_func=train_possridge,
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

    parser = argparse.ArgumentParser(description='Run possession-weighted ridge RAPM variant')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test with 2 folds only')
    parser.add_argument('--skip-backtest', action='store_true',
                        help='Skip backtest, use default config')
    parser.add_argument('--lambda-def', type=float, default=None,
                        help='Override defense lambda base')
    parser.add_argument('--p', type=float, default=None,
                        help='Override p (multiplier power)')
    parser.add_argument('--m-max', type=float, default=None,
                        help='Override m_max (max multiplier)')

    args = parser.parse_args()

    if args.skip_backtest:
        # Use default or specified config
        best_config = {
            'lambda_off': LAMBDA_OFF_BASE,
            'lambda_def': args.lambda_def if args.lambda_def else 4500,
            'p': args.p if args.p else 0.5,
            'm_max': args.m_max if args.m_max else 3.0
        }
    else:
        # Run backtest
        if args.quick:
            from .base_rapm import BACKTEST_FOLDS
            folds = BACKTEST_FOLDS[:2]
        else:
            folds = None

        best_config = run_possridge_backtest(folds=folds)

    # Train and save final model
    train_and_save_final_model(best_config)
