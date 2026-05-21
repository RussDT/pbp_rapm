#!/usr/bin/env python3
"""
RAPM Time-Decay + Possession-Weighted Ridge Variant

Combines both approaches:
- Time-decay row weights: w_i = decay_base ^ age_days
- Per-coefficient penalties: lambda_j = lambda_base * m_j
- Offense BASE lambda: FIXED at 3000

Tune: (decay_base, lambda_def_base, p, m_max)
Grid size: 3 x 7 x 5 x 3 = 315 configs
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
    compute_time_decay_weights,
    RESULTS_DIR,
)
from .possession_ridge import (
    compute_multiplier,
    possession_weighted_ridge,
)
from .backtest import run_backtest, train_final_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

VARIANT_NAME = "timedecay_possridge"


###############################################################################
# CUSTOM ALTERNATING SOLVER FOR COMBINED APPROACH
###############################################################################

def alternating_timedecay_possridge(
    X: csr_matrix,
    y: np.ndarray,
    weights: np.ndarray,
    players: List[str],
    player_to_col: Dict[str, int],
    off_poss: Dict[str, int],
    def_poss: Dict[str, int],
    lambda_off_base: float,
    lambda_def_base: float,
    p: float,
    m_max: float,
    max_iter: int = 200,
    tol: float = 1e-4
) -> np.ndarray:
    """
    Alternating minimization with time-decay weights AND possession-weighted penalties.

    Args:
        X: Design matrix (NOT yet weighted)
        y: Target (centered, NOT yet weighted)
        weights: Per-row time-decay weights
        players, player_to_col: Column mapping
        off_poss, def_poss: Possession counts
        lambda_off_base, lambda_def_base: Base penalties
        p, m_max: Possession weighting params
        max_iter, tol: Convergence params

    Returns:
        Coefficient vector (beta)
    """
    logging.info(
        f"Running alternating time-decay + possession-weighted ridge: "
        f"lambda_off={lambda_off_base}, lambda_def={lambda_def_base}, "
        f"p={p}, m_max={m_max}"
    )

    n_features = len(player_to_col)
    beta = np.zeros(n_features, dtype=np.float64)

    # Apply time-decay weights to X and y
    sqrt_w = np.sqrt(weights).astype(np.float64)
    X_weighted = X.multiply(sqrt_w[:, np.newaxis]).tocsr()
    y_weighted = y * sqrt_w

    # Separate offense and defense indices
    offense_indices = []
    defense_indices = []

    for key, idx in player_to_col.items():
        if key.endswith('_off'):
            offense_indices.append(idx)
        elif key.endswith('_def'):
            defense_indices.append(idx)

    offense_indices = np.array(offense_indices, dtype=int)
    defense_indices = np.array(defense_indices, dtype=int)

    X_off = X_weighted[:, offense_indices]
    X_def = X_weighted[:, defense_indices]

    # Build per-player lambda vectors
    n_off = len(offense_indices)
    n_def = len(defense_indices)

    lambda_off_vec = np.zeros(n_off, dtype=np.float64)
    lambda_def_vec = np.zeros(n_def, dtype=np.float64)

    # Map: full index -> subset index
    off_full_to_sub = {full_idx: sub_idx for sub_idx, full_idx in enumerate(offense_indices)}
    def_full_to_sub = {full_idx: sub_idx for sub_idx, full_idx in enumerate(defense_indices)}

    for player in players:
        off_key = f"{player}_off"
        def_key = f"{player}_def"

        if off_key in player_to_col:
            full_idx = player_to_col[off_key]
            sub_idx = off_full_to_sub[full_idx]
            n_poss = off_poss.get(player, 0)
            m = compute_multiplier(n_poss, N_REF, N0, p, M_MIN, m_max)
            lambda_off_vec[sub_idx] = lambda_off_base * m

        if def_key in player_to_col:
            full_idx = player_to_col[def_key]
            sub_idx = def_full_to_sub[full_idx]
            n_poss = def_poss.get(player, 0)
            m = compute_multiplier(n_poss, N_REF, N0, p, M_MIN, m_max)
            lambda_def_vec[sub_idx] = lambda_def_base * m

    residual = y_weighted.copy()

    for iteration in range(max_iter):
        beta_prev = beta.copy()

        # Update offense coefficients
        residual += X_off @ beta[offense_indices]
        beta_off = possession_weighted_ridge(X_off, residual, lambda_off_vec)
        beta[offense_indices] = beta_off
        residual -= X_off @ beta_off

        # Update defense coefficients
        residual += X_def @ beta[defense_indices]
        beta_def = possession_weighted_ridge(X_def, residual, lambda_def_vec)
        beta[defense_indices] = beta_def
        residual -= X_def @ beta_def

        # Check convergence
        delta_beta = np.linalg.norm(beta - beta_prev)
        if (iteration + 1) % 20 == 0:
            logging.debug(f"Iteration {iteration+1}, delta_beta={delta_beta:.6f}")
        if delta_beta < tol:
            logging.info(f"Converged after {iteration+1} iterations")
            break
    else:
        logging.info("Max iterations reached")

    return beta


###############################################################################
# TRAINING FUNCTION
###############################################################################

def train_timedecay_possridge(
    df_train: pd.DataFrame,
    players: List[str],
    player_to_col: Dict[str, int],
    config: Dict[str, Any]
) -> np.ndarray:
    """
    Train combined time-decay + possession-weighted ridge RAPM model.

    Args:
        df_train: Prepared training DataFrame
        players: List of player IDs
        player_to_col: Column mapping
        config: Must contain 'lambda_def', 'decay_base', 'p', 'm_max'

    Returns:
        Coefficient vector (beta)
    """
    lambda_off_base = LAMBDA_OFF_BASE  # Fixed at 3000
    lambda_def_base = config['lambda_def']
    decay_base = config['decay_base']
    p = config['p']
    m_max = config['m_max']

    # Build design matrix
    X, y = build_design_matrix(df_train, player_to_col)

    # Compute time-decay weights
    weights = compute_time_decay_weights(df_train, decay_base=decay_base)

    # Get possession counts
    total_poss, off_poss, def_poss = get_player_possessions_fast(df_train, players)

    # Weighted mean for centering
    y_mean = np.average(y, weights=weights)
    y_centered = y - y_mean

    # Run combined solver
    beta = alternating_timedecay_possridge(
        X=X,
        y=y_centered,
        weights=weights,
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
    """Generate all hyperparameter configurations for combined variant."""
    configs = []
    for decay_base in DECAY_GRID:
        for lambda_def in LAMBDA_DEF_GRID:
            for p in P_GRID:
                for m_max in M_MAX_GRID:
                    configs.append({
                        'lambda_off': LAMBDA_OFF_BASE,
                        'lambda_def': lambda_def,
                        'decay_base': decay_base,
                        'p': p,
                        'm_max': m_max
                    })
    return configs


###############################################################################
# MAIN
###############################################################################

def run_timedecay_possridge_backtest(
    folds=None,
    save_results: bool = True
) -> Dict[str, Any]:
    """
    Run full backtest for combined time-decay + possession-weighted ridge variant.

    Returns:
        best_config with lowest average RMSE
    """
    configs = generate_configs()
    logging.info(f"Combined time-decay + possession-weighted configs: {len(configs)}")

    best_config, all_results = run_backtest(
        train_func=train_timedecay_possridge,
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
    logging.info(f"Training final combined model with config: {best_config}")
    logging.info(f"Training years: {train_years}")

    beta, players, player_to_col, df_train = train_final_model(
        train_func=train_timedecay_possridge,
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

    parser = argparse.ArgumentParser(
        description='Run combined time-decay + possession-weighted ridge RAPM variant'
    )
    parser.add_argument('--quick', action='store_true',
                        help='Quick test with 2 folds only')
    parser.add_argument('--skip-backtest', action='store_true',
                        help='Skip backtest, use default config')
    parser.add_argument('--lambda-def', type=float, default=None,
                        help='Override defense lambda base')
    parser.add_argument('--decay-base', type=float, default=None,
                        help='Override decay base')
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
            'decay_base': args.decay_base if args.decay_base else 0.999,
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

        best_config = run_timedecay_possridge_backtest(folds=folds)

    # Train and save final model
    train_and_save_final_model(best_config)
