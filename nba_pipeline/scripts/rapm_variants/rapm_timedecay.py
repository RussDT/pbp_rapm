#!/usr/bin/env python3
"""
RAPM Time-Decay Variant

Time-decay weighted ridge regression:
- Time-decay row weights: w_i = decay_base ^ age_days
- Row-scaling: X * sqrt(w), y * sqrt(w)
- Offense lambda: FIXED at 3000
- Defense lambda: Tuned over grid

Tune: (decay_base, lambda_def)
Grid size: 3 x 7 = 21 configs
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from scipy.sparse import csr_matrix

from .base_rapm import (
    LAMBDA_OFF_BASE,
    LAMBDA_DEF_GRID,
    DECAY_GRID,
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
    compute_time_decay_weights,
    RESULTS_DIR,
)
from .possession_ridge import alternating_ridge
from .backtest import run_backtest, train_final_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

VARIANT_NAME = "timedecay"


###############################################################################
# TRAINING FUNCTION
###############################################################################

def train_timedecay(
    df_train: pd.DataFrame,
    players: List[str],
    player_to_col: Dict[str, int],
    config: Dict[str, Any]
) -> np.ndarray:
    """
    Train time-decay RAPM model.

    Args:
        df_train: Prepared training DataFrame
        players: List of player IDs
        player_to_col: Column mapping
        config: Must contain 'lambda_def', 'decay_base'

    Returns:
        Coefficient vector (beta)
    """
    lambda_off = config.get('lambda_off', LAMBDA_OFF_BASE)
    lambda_def = config['lambda_def']
    decay_base = config['decay_base']

    # Build design matrix
    X, y = build_design_matrix(df_train, player_to_col)

    # Compute time-decay weights
    weights = compute_time_decay_weights(df_train, decay_base=decay_base)

    # Weighted mean for centering
    y_mean = np.average(y, weights=weights)
    y_centered = y - y_mean

    # Apply row weights: sqrt(w) scaling
    sqrt_w = np.sqrt(weights).astype(np.float64)

    # Scale X rows
    X_weighted = X.multiply(sqrt_w[:, np.newaxis]).tocsr()

    # Scale y
    y_weighted = y_centered * sqrt_w

    # Run alternating ridge on weighted data
    beta = alternating_ridge(
        X=X_weighted,
        y=y_weighted,
        player_to_col=player_to_col,
        alpha_offense=float(lambda_off),
        alpha_defense=float(lambda_def),
        max_iter=200,
        tol=1e-4
    )

    # Re-center coefficients (use unweighted possessions for interpretability)
    total_poss, _, _ = get_player_possessions_fast(df_train, players)
    beta = recenter_coefficients(beta, players, player_to_col, total_poss)

    return beta


###############################################################################
# CONFIG GENERATION
###############################################################################

def generate_configs() -> List[Dict[str, Any]]:
    """Generate all hyperparameter configurations for time-decay variant."""
    configs = []
    for decay_base in DECAY_GRID:
        for lambda_def in LAMBDA_DEF_GRID:
            configs.append({
                'lambda_off': LAMBDA_OFF_BASE,
                'lambda_def': lambda_def,
                'decay_base': decay_base
            })
    return configs


###############################################################################
# MAIN
###############################################################################

def run_timedecay_backtest(
    folds=None,
    save_results: bool = True
) -> Dict[str, Any]:
    """
    Run full backtest for time-decay variant.

    Returns:
        best_config with lowest average RMSE
    """
    configs = generate_configs()
    logging.info(f"Time-decay configs: {len(configs)}")

    best_config, all_results = run_backtest(
        train_func=train_timedecay,
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
    logging.info(f"Training final time-decay model with config: {best_config}")
    logging.info(f"Training years: {train_years}")

    beta, players, player_to_col, df_train = train_final_model(
        train_func=train_timedecay,
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

def half_life_to_decay_base(half_life_days: float) -> float:
    """Convert half-life in days to decay_base."""
    return 0.5 ** (1.0 / half_life_days)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run time-decay RAPM variant')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test with 2 folds only')
    parser.add_argument('--skip-backtest', action='store_true',
                        help='Skip backtest, use default config')
    parser.add_argument('--lambda-off', type=float, default=None,
                        help='Override offense lambda (default: 3000)')
    parser.add_argument('--lambda-def', type=float, default=None,
                        help='Override defense lambda (default: 4500)')
    parser.add_argument('--decay-base', type=float, default=None,
                        help='Override decay base (daily decay rate)')
    parser.add_argument('--half-life', type=float, default=None,
                        help='Half-life in days (alternative to --decay-base)')
    parser.add_argument('--years', type=int, nargs='+', default=None,
                        help='Training years (2-digit, e.g., --years 21 22 23 24 25)')

    args = parser.parse_args()

    # Determine decay_base from either --decay-base or --half-life
    if args.half_life:
        decay_base = half_life_to_decay_base(args.half_life)
        logging.info(f"Half-life {args.half_life} days -> decay_base {decay_base:.6f}")
    elif args.decay_base:
        decay_base = args.decay_base
    else:
        decay_base = 0.999  # Default ~700 day half-life

    # Determine training years
    train_years = args.years if args.years else [22, 23, 24, 25]

    if args.skip_backtest:
        # Use default or specified config
        best_config = {
            'lambda_off': args.lambda_off if args.lambda_off else LAMBDA_OFF_BASE,
            'lambda_def': args.lambda_def if args.lambda_def else 4500,
            'decay_base': decay_base
        }
    else:
        # Run backtest
        if args.quick:
            from .base_rapm import BACKTEST_FOLDS
            folds = BACKTEST_FOLDS[:2]
        else:
            folds = None

        best_config = run_timedecay_backtest(folds=folds)
        # Override decay_base if specified
        if args.half_life or args.decay_base:
            best_config['decay_base'] = decay_base

    # Train and save final model
    train_and_save_final_model(best_config, train_years=train_years)
