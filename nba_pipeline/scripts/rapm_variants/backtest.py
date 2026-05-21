"""
Forward-looking backtest framework for RAPM variants.

Key design:
- Train on K=3 years, evaluate on next year (9 folds total)
- RMSE computed at stint/possession level, NOT player level
- Same folds used for all variants for fair comparison
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable, Any, Optional
from pathlib import Path
from dataclasses import dataclass
import json
from datetime import datetime

from .base_rapm import (
    BACKTEST_FOLDS,
    load_rapm_data,
    prepare_data,
    get_all_players,
    build_player_mapping,
    build_design_matrix,
    get_player_possessions_fast,
    RESULTS_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


###############################################################################
# DATA CLASSES
###############################################################################

@dataclass
class FoldResult:
    """Results from a single fold evaluation."""
    fold_idx: int
    train_years: List[int]
    eval_year: int
    config: Dict[str, Any]
    rmse: float
    coverage: float
    n_eval_stints: int
    n_covered_stints: int


@dataclass
class BacktestResult:
    """Aggregated results across all folds for a single config."""
    config: Dict[str, Any]
    avg_rmse: float
    std_rmse: float
    avg_coverage: float
    fold_results: List[FoldResult]


###############################################################################
# RMSE CALCULATION AT STINT LEVEL
###############################################################################

def compute_stint_predictions(
    df_eval: pd.DataFrame,
    beta: np.ndarray,
    player_to_col: Dict[str, int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute predictions for evaluation stints.

    For each stint:
        predicted_pts = sum(offense_betas) + sum(defense_betas)

    100% COVERAGE: Unknown players are treated as league average (0 contribution).
    This ensures all stints are evaluated, not just veteran-heavy lineups.

    Returns:
        actual: Actual point values
        predicted: Predicted point values
        covered: Boolean mask (always True for 100% coverage)
    """
    n_stints = len(df_eval)
    actual = df_eval['Off_Diff'].values.astype(np.float64)
    predicted = np.zeros(n_stints, dtype=np.float64)
    covered = np.ones(n_stints, dtype=bool)  # Always True - 100% coverage

    # Get player columns
    off_cols = ['O1', 'O2', 'O3', 'O4', 'O5']
    def_cols = ['D1', 'D2', 'D3', 'D4', 'D5']

    for i, row in df_eval.iterrows():
        idx = df_eval.index.get_loc(i)
        pred = 0.0

        # Sum offense betas (unknown players contribute 0)
        for col in off_cols:
            pid = str(int(row[col]))
            key = f"{pid}_off"
            if key in player_to_col:
                pred += beta[player_to_col[key]]
            # else: unknown player contributes 0 (league average)

        # Sum defense betas (unknown players contribute 0)
        for col in def_cols:
            pid = str(int(row[col]))
            key = f"{pid}_def"
            if key in player_to_col:
                pred += beta[player_to_col[key]]
            # else: unknown player contributes 0 (league average)

        predicted[idx] = pred
        # covered[idx] is already True - 100% coverage

    return actual, predicted, covered


def compute_stint_predictions_fast(
    df_eval: pd.DataFrame,
    beta: np.ndarray,
    player_to_col: Dict[str, int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fast vectorized stint prediction computation.

    100% COVERAGE: Unknown players are treated as league average (0 contribution).
    This ensures all stints are evaluated, not just veteran-heavy lineups.

    Returns:
        actual, predicted, covered arrays
    """
    n_stints = len(df_eval)
    actual = df_eval['Off_Diff'].values.astype(np.float64)
    predicted = np.zeros(n_stints, dtype=np.float64)
    covered = np.ones(n_stints, dtype=bool)  # Always True - 100% coverage

    # Process offense columns
    # Unknown players contribute 0 (league average) to prediction
    for col in ['O1', 'O2', 'O3', 'O4', 'O5']:
        pids = df_eval[col].astype(int).astype(str)
        for i, pid in enumerate(pids):
            key = f"{pid}_off"
            if key in player_to_col:
                predicted[i] += beta[player_to_col[key]]
            # else: player unknown, contributes 0 (implicit - already initialized)

    # Process defense columns
    # Unknown players contribute 0 (league average) to prediction
    for col in ['D1', 'D2', 'D3', 'D4', 'D5']:
        pids = df_eval[col].astype(int).astype(str)
        for i, pid in enumerate(pids):
            key = f"{pid}_def"
            if key in player_to_col:
                predicted[i] += beta[player_to_col[key]]
            # else: player unknown, contributes 0 (implicit - already initialized)

    return actual, predicted, covered


def compute_rmse(
    actual: np.ndarray,
    predicted: np.ndarray,
    covered: np.ndarray
) -> Tuple[float, float]:
    """
    Compute RMSE on covered stints.

    Returns:
        rmse: Root mean squared error
        coverage: Fraction of stints with all known players
    """
    n_total = len(actual)
    n_covered = covered.sum()

    if n_covered == 0:
        return float('inf'), 0.0

    coverage = n_covered / n_total

    # RMSE on covered stints only
    errors = actual[covered] - predicted[covered]
    rmse = np.sqrt(np.mean(errors ** 2))

    return rmse, coverage


###############################################################################
# SINGLE FOLD EVALUATION
###############################################################################

def evaluate_fold(
    fold_idx: int,
    fold_def: Dict,
    train_func: Callable,
    config: Dict[str, Any],
    include_playoffs: bool = True
) -> FoldResult:
    """
    Evaluate a single fold.

    Args:
        fold_idx: Index of this fold
        fold_def: Dict with 'train' and 'eval' keys (eval can be int or list)
        train_func: Function that takes (train_df, config) and returns beta
        config: Hyperparameter configuration
        include_playoffs: Whether to include playoff data

    Returns:
        FoldResult
    """
    train_years = fold_def['train']
    eval_years = fold_def['eval']

    # Handle eval being single year or list
    if isinstance(eval_years, int):
        eval_years = [eval_years]

    logging.info(f"Fold {fold_idx+1}: Train={train_years}, Eval={eval_years}")

    # Load training data
    df_train_raw = load_rapm_data(train_years, include_playoffs=include_playoffs)
    df_train = prepare_data(df_train_raw, use_pure=True)

    # Load evaluation data (now handles multiple years)
    df_eval_raw = load_rapm_data(eval_years, include_playoffs=include_playoffs)
    df_eval = prepare_data(df_eval_raw, use_pure=True)

    # Get players from training data only
    train_players = get_all_players(df_train)
    player_to_col = build_player_mapping(train_players)

    # Train model
    beta = train_func(df_train, train_players, player_to_col, config)

    # Evaluate on eval data
    actual, predicted, covered = compute_stint_predictions_fast(
        df_eval, beta, player_to_col
    )
    rmse, coverage = compute_rmse(actual, predicted, covered)

    return FoldResult(
        fold_idx=fold_idx,
        train_years=train_years,
        eval_year=eval_years,  # Now stores list
        config=config.copy(),
        rmse=rmse,
        coverage=coverage,
        n_eval_stints=len(df_eval),
        n_covered_stints=int(covered.sum())
    )


###############################################################################
# GRID SEARCH WITH BACKTEST
###############################################################################

def run_backtest(
    train_func: Callable,
    configs: List[Dict[str, Any]],
    variant_name: str,
    folds: Optional[List[Dict]] = None,
    include_playoffs: bool = True,
    save_results: bool = True
) -> Tuple[Dict[str, Any], List[BacktestResult]]:
    """
    Run forward-looking backtest across all folds and configs.

    Args:
        train_func: Function(df_train, players, player_to_col, config) -> beta
        configs: List of hyperparameter configurations to try
        variant_name: Name for logging and output files
        folds: Override default folds (for testing)
        include_playoffs: Include playoff data
        save_results: Whether to save results to disk

    Returns:
        best_config: Configuration with lowest average RMSE
        all_results: List of BacktestResult for all configs
    """
    if folds is None:
        folds = BACKTEST_FOLDS

    logging.info(f"=" * 60)
    logging.info(f"Running backtest for {variant_name}")
    logging.info(f"Configs to test: {len(configs)}")
    logging.info(f"Folds: {len(folds)}")
    logging.info(f"=" * 60)

    all_results = []

    for config_idx, config in enumerate(configs):
        logging.info(f"\nConfig {config_idx+1}/{len(configs)}: {config}")

        fold_results = []
        for fold_idx, fold_def in enumerate(folds):
            result = evaluate_fold(
                fold_idx=fold_idx,
                fold_def=fold_def,
                train_func=train_func,
                config=config,
                include_playoffs=include_playoffs
            )
            fold_results.append(result)
            logging.info(
                f"  Fold {fold_idx+1}: RMSE={result.rmse:.4f}, "
                f"Coverage={result.coverage:.1%}"
            )

        # Aggregate results
        rmses = [r.rmse for r in fold_results]
        coverages = [r.coverage for r in fold_results]

        backtest_result = BacktestResult(
            config=config.copy(),
            avg_rmse=np.mean(rmses),
            std_rmse=np.std(rmses),
            avg_coverage=np.mean(coverages),
            fold_results=fold_results
        )
        all_results.append(backtest_result)

        logging.info(
            f"  Avg RMSE: {backtest_result.avg_rmse:.4f} +/- {backtest_result.std_rmse:.4f}"
        )

    # Find best config
    best_result = min(all_results, key=lambda r: r.avg_rmse)
    best_config = best_result.config

    logging.info(f"\n" + "=" * 60)
    logging.info(f"Best config for {variant_name}:")
    logging.info(f"  {best_config}")
    logging.info(f"  Avg RMSE: {best_result.avg_rmse:.4f}")
    logging.info(f"=" * 60)

    # Save results
    if save_results:
        save_backtest_results(variant_name, all_results, best_config)

    return best_config, all_results


###############################################################################
# RESULTS SAVING
###############################################################################

def save_backtest_results(
    variant_name: str,
    all_results: List[BacktestResult],
    best_config: Dict[str, Any]
):
    """Save backtest results to disk."""
    variant_dir = RESULTS_DIR / f"{variant_name}_results"
    variant_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_data = {
        'variant': variant_name,
        'timestamp': datetime.now().isoformat(),
        'best_config': best_config,
        'n_configs_tested': len(all_results),
        'n_folds': len(all_results[0].fold_results) if all_results else 0
    }

    config_file = variant_dir / 'config.json'
    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2)

    # Save per-fold metrics
    rows = []
    for result in all_results:
        for fold in result.fold_results:
            row = fold.config.copy()
            row['fold_idx'] = fold.fold_idx
            row['eval_year'] = fold.eval_year
            row['train_years'] = str(fold.train_years)
            row['rmse'] = fold.rmse
            row['coverage'] = fold.coverage
            row['n_eval_stints'] = fold.n_eval_stints
            row['n_covered_stints'] = fold.n_covered_stints
            rows.append(row)

    metrics_df = pd.DataFrame(rows)
    metrics_file = variant_dir / 'per_fold_metrics.csv'
    metrics_df.to_csv(metrics_file, index=False)

    # Save best params
    best_df = pd.DataFrame([best_config])
    best_file = variant_dir / 'best_params.csv'
    best_df.to_csv(best_file, index=False)

    # Save summary
    summary_rows = []
    for result in all_results:
        row = result.config.copy()
        row['avg_rmse'] = result.avg_rmse
        row['std_rmse'] = result.std_rmse
        row['avg_coverage'] = result.avg_coverage
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values('avg_rmse')
    summary_file = variant_dir / 'config_summary.csv'
    summary_df.to_csv(summary_file, index=False)

    logging.info(f"Results saved to {variant_dir}")


###############################################################################
# FINAL MODEL TRAINING
###############################################################################

def train_final_model(
    train_func: Callable,
    best_config: Dict[str, Any],
    train_years: List[int],
    include_playoffs: bool = True
) -> Tuple[np.ndarray, List[str], Dict[str, int], pd.DataFrame]:
    """
    Train final model with best config on specified years.

    Returns:
        beta: Coefficient vector
        players: List of player IDs
        player_to_col: Column mapping
        df_train: Training DataFrame
    """
    df_train_raw = load_rapm_data(train_years, include_playoffs=include_playoffs)
    df_train = prepare_data(df_train_raw, use_pure=True)

    players = get_all_players(df_train)
    player_to_col = build_player_mapping(players)

    beta = train_func(df_train, players, player_to_col, best_config)

    return beta, players, player_to_col, df_train


###############################################################################
# DIAGNOSTIC UTILITIES
###############################################################################

def compute_rmse_by_possession_bin(
    df_eval: pd.DataFrame,
    beta: np.ndarray,
    player_to_col: Dict[str, int],
    total_poss: Dict[str, int],
    bins: List[int] = [0, 500, 1000, 2000, 5000, float('inf')]
) -> pd.DataFrame:
    """
    Compute RMSE broken down by player possession bins.

    For each stint, categorize by the minimum possessions among the 10 players.
    """
    actual, predicted, covered = compute_stint_predictions_fast(
        df_eval, beta, player_to_col
    )

    # Get min possessions for each stint
    min_poss = np.zeros(len(df_eval), dtype=int)

    for i, row in df_eval.iterrows():
        idx = df_eval.index.get_loc(i)
        poss_list = []
        for col in ['O1', 'O2', 'O3', 'O4', 'O5', 'D1', 'D2', 'D3', 'D4', 'D5']:
            pid = str(int(row[col]))
            poss_list.append(total_poss.get(pid, 0))
        min_poss[idx] = min(poss_list) if poss_list else 0

    # Compute RMSE by bin
    results = []
    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i+1]
        mask = (min_poss >= low) & (min_poss < high) & covered

        if mask.sum() > 0:
            bin_rmse = np.sqrt(np.mean((actual[mask] - predicted[mask]) ** 2))
            results.append({
                'bin': f"{low}-{high if high != float('inf') else '+'}",
                'n_stints': int(mask.sum()),
                'rmse': bin_rmse
            })

    return pd.DataFrame(results)


def compute_year_to_year_correlation(
    beta_year1: np.ndarray,
    beta_year2: np.ndarray,
    players_common: List[str],
    player_to_col_1: Dict[str, int],
    player_to_col_2: Dict[str, int]
) -> Tuple[float, float]:
    """
    Compute Spearman correlation of player impacts between two years.

    Returns:
        off_corr: Offense correlation
        def_corr: Defense correlation
    """
    from scipy.stats import spearmanr

    off_1 = []
    off_2 = []
    def_1 = []
    def_2 = []

    for p in players_common:
        off_key = f"{p}_off"
        def_key = f"{p}_def"

        if off_key in player_to_col_1 and off_key in player_to_col_2:
            off_1.append(beta_year1[player_to_col_1[off_key]])
            off_2.append(beta_year2[player_to_col_2[off_key]])

        if def_key in player_to_col_1 and def_key in player_to_col_2:
            def_1.append(beta_year1[player_to_col_1[def_key]])
            def_2.append(beta_year2[player_to_col_2[def_key]])

    off_corr = spearmanr(off_1, off_2).correlation if len(off_1) > 2 else 0.0
    def_corr = spearmanr(def_1, def_2).correlation if len(def_1) > 2 else 0.0

    return off_corr, def_corr


###############################################################################
# PLAYER-LEVEL RAPM PREDICTION EVALUATION
###############################################################################

@dataclass
class PlayerPredictionResult:
    """Results from player-level RAPM prediction evaluation."""
    fold_idx: int
    train_years: List[int]
    eval_years: List[int]
    config: Dict[str, Any]
    pearson_net: float
    spearman_net: float
    rmse_net: float
    pearson_off: float
    spearman_off: float
    rmse_off: float
    pearson_def: float
    spearman_def: float
    rmse_def: float
    n_players: int


@dataclass
class PlayerPredictionBacktestResult:
    """Aggregated player prediction results across all folds."""
    config: Dict[str, Any]
    avg_pearson_net: float
    avg_spearman_net: float
    avg_rmse_net: float
    std_rmse_net: float
    fold_results: List[PlayerPredictionResult]


def evaluate_fold_player_prediction(
    fold_idx: int,
    fold_def: Dict,
    train_func: Callable,
    config: Dict[str, Any],
    min_poss: int = 500,
    include_playoffs: bool = True
) -> PlayerPredictionResult:
    """
    Evaluate a fold by predicting future player RAPM.

    This is a better evaluation metric than stint-level RMSE because:
    1. It directly tests what we care about (player impact estimation)
    2. It shows meaningful differences between hyperparameters
    3. It's less noisy than stint-level predictions

    The approach:
    - Train RAPM on training years
    - Train separate RAPM on eval years (as "ground truth")
    - Compare predictions to ground truth for common players

    Args:
        fold_idx: Index of this fold
        fold_def: Dict with 'train' and 'eval' keys
        train_func: Function that takes (train_df, config) and returns beta
        config: Hyperparameter configuration
        min_poss: Minimum possessions in BOTH periods to include player
        include_playoffs: Whether to include playoff data

    Returns:
        PlayerPredictionResult
    """
    from scipy.stats import pearsonr, spearmanr

    train_years = fold_def['train']
    eval_years = fold_def['eval']

    if isinstance(eval_years, int):
        eval_years = [eval_years]

    logging.info(f"Fold {fold_idx+1}: Train={train_years}, Eval={eval_years}")

    # Load and prepare training data
    df_train_raw = load_rapm_data(train_years, include_playoffs=include_playoffs)
    df_train = prepare_data(df_train_raw, use_pure=True)
    train_players = get_all_players(df_train)
    train_player_to_col = build_player_mapping(train_players)

    # Load and prepare eval data (to compute "ground truth" RAPM)
    df_eval_raw = load_rapm_data(eval_years, include_playoffs=include_playoffs)
    df_eval = prepare_data(df_eval_raw, use_pure=True)
    eval_players = get_all_players(df_eval)
    eval_player_to_col = build_player_mapping(eval_players)

    # Train model on training data
    beta_train = train_func(df_train, train_players, train_player_to_col, config)

    # Train model on eval data (ground truth)
    beta_eval = train_func(df_eval, eval_players, eval_player_to_col, config)

    # Get possessions for filtering
    train_poss, _, _ = get_player_possessions_fast(df_train, train_players)
    eval_poss, _, _ = get_player_possessions_fast(df_eval, eval_players)

    # Find common players with sufficient possessions in both periods
    common_players = []
    for p in set(train_players) & set(eval_players):
        train_p = train_poss.get(p, 0)
        eval_p = eval_poss.get(p, 0)
        if train_p >= min_poss and eval_p >= min_poss:
            common_players.append(p)

    if len(common_players) < 10:
        logging.warning(f"Only {len(common_players)} common players with min_poss={min_poss}")
        return PlayerPredictionResult(
            fold_idx=fold_idx,
            train_years=train_years,
            eval_years=eval_years,
            config=config.copy(),
            pearson_net=0.0, spearman_net=0.0, rmse_net=float('inf'),
            pearson_off=0.0, spearman_off=0.0, rmse_off=float('inf'),
            pearson_def=0.0, spearman_def=0.0, rmse_def=float('inf'),
            n_players=len(common_players)
        )

    # Extract coefficients for common players
    pred_off, pred_def, actual_off, actual_def = [], [], [], []

    for p in common_players:
        off_key = f"{p}_off"
        def_key = f"{p}_def"

        if off_key in train_player_to_col and off_key in eval_player_to_col:
            pred_off.append(beta_train[train_player_to_col[off_key]])
            actual_off.append(beta_eval[eval_player_to_col[off_key]])

        if def_key in train_player_to_col and def_key in eval_player_to_col:
            pred_def.append(beta_train[train_player_to_col[def_key]])
            actual_def.append(beta_eval[eval_player_to_col[def_key]])

    pred_off = np.array(pred_off)
    pred_def = np.array(pred_def)
    actual_off = np.array(actual_off)
    actual_def = np.array(actual_def)

    # Net RAPM
    pred_net = pred_off + pred_def
    actual_net = actual_off + actual_def

    # Compute metrics
    pearson_net = pearsonr(pred_net, actual_net)[0] if len(pred_net) > 2 else 0.0
    spearman_net = spearmanr(pred_net, actual_net)[0] if len(pred_net) > 2 else 0.0
    rmse_net = np.sqrt(np.mean((pred_net - actual_net) ** 2)) if len(pred_net) > 0 else float('inf')

    pearson_off = pearsonr(pred_off, actual_off)[0] if len(pred_off) > 2 else 0.0
    spearman_off = spearmanr(pred_off, actual_off)[0] if len(pred_off) > 2 else 0.0
    rmse_off = np.sqrt(np.mean((pred_off - actual_off) ** 2)) if len(pred_off) > 0 else float('inf')

    pearson_def = pearsonr(pred_def, actual_def)[0] if len(pred_def) > 2 else 0.0
    spearman_def = spearmanr(pred_def, actual_def)[0] if len(pred_def) > 2 else 0.0
    rmse_def = np.sqrt(np.mean((pred_def - actual_def) ** 2)) if len(pred_def) > 0 else float('inf')

    return PlayerPredictionResult(
        fold_idx=fold_idx,
        train_years=train_years,
        eval_years=eval_years,
        config=config.copy(),
        pearson_net=pearson_net,
        spearman_net=spearman_net,
        rmse_net=rmse_net,
        pearson_off=pearson_off,
        spearman_off=spearman_off,
        rmse_off=rmse_off,
        pearson_def=pearson_def,
        spearman_def=spearman_def,
        rmse_def=rmse_def,
        n_players=len(common_players)
    )


def run_player_prediction_backtest(
    train_func: Callable,
    configs: List[Dict[str, Any]],
    variant_name: str,
    folds: Optional[List[Dict]] = None,
    min_poss: int = 500,
    include_playoffs: bool = True,
    save_results: bool = True
) -> Tuple[Dict[str, Any], List[PlayerPredictionBacktestResult]]:
    """
    Run player-level RAPM prediction backtest.

    This evaluates hyperparameters by how well predicted RAPM correlates
    with future actual RAPM for the same players.

    Args:
        train_func: Function(df_train, players, player_to_col, config) -> beta
        configs: List of hyperparameter configurations
        variant_name: Name for logging and output files
        folds: Override default folds
        min_poss: Minimum possessions to include player
        include_playoffs: Include playoff data
        save_results: Whether to save results to disk

    Returns:
        best_config: Configuration with highest average correlation
        all_results: List of PlayerPredictionBacktestResult
    """
    if folds is None:
        folds = BACKTEST_FOLDS

    logging.info(f"=" * 60)
    logging.info(f"Running PLAYER PREDICTION backtest for {variant_name}")
    logging.info(f"Configs to test: {len(configs)}")
    logging.info(f"Folds: {len(folds)}")
    logging.info(f"Min possessions: {min_poss}")
    logging.info(f"=" * 60)

    all_results = []

    for config_idx, config in enumerate(configs):
        logging.info(f"\nConfig {config_idx+1}/{len(configs)}: {config}")

        fold_results = []
        for fold_idx, fold_def in enumerate(folds):
            result = evaluate_fold_player_prediction(
                fold_idx=fold_idx,
                fold_def=fold_def,
                train_func=train_func,
                config=config,
                min_poss=min_poss,
                include_playoffs=include_playoffs
            )
            fold_results.append(result)
            logging.info(
                f"  Fold {fold_idx+1}: Pearson={result.pearson_net:.4f}, "
                f"Spearman={result.spearman_net:.4f}, RMSE={result.rmse_net:.4f}, "
                f"N={result.n_players}"
            )

        # Aggregate results
        pearson_vals = [r.pearson_net for r in fold_results]
        spearman_vals = [r.spearman_net for r in fold_results]
        rmse_vals = [r.rmse_net for r in fold_results]

        backtest_result = PlayerPredictionBacktestResult(
            config=config.copy(),
            avg_pearson_net=np.mean(pearson_vals),
            avg_spearman_net=np.mean(spearman_vals),
            avg_rmse_net=np.mean(rmse_vals),
            std_rmse_net=np.std(rmse_vals),
            fold_results=fold_results
        )
        all_results.append(backtest_result)

        logging.info(
            f"  Avg Pearson: {backtest_result.avg_pearson_net:.4f}, "
            f"Avg Spearman: {backtest_result.avg_spearman_net:.4f}, "
            f"Avg RMSE: {backtest_result.avg_rmse_net:.4f}"
        )

    # Find best config (maximize correlation, not minimize RMSE)
    best_result = max(all_results, key=lambda r: r.avg_pearson_net)
    best_config = best_result.config

    logging.info(f"\n" + "=" * 60)
    logging.info(f"Best config for {variant_name} (by Pearson correlation):")
    logging.info(f"  {best_config}")
    logging.info(f"  Avg Pearson: {best_result.avg_pearson_net:.4f}")
    logging.info(f"  Avg Spearman: {best_result.avg_spearman_net:.4f}")
    logging.info(f"  Avg RMSE: {best_result.avg_rmse_net:.4f}")
    logging.info(f"=" * 60)

    # Save results
    if save_results:
        save_player_prediction_results(variant_name, all_results, best_config)

    return best_config, all_results


def save_player_prediction_results(
    variant_name: str,
    all_results: List[PlayerPredictionBacktestResult],
    best_config: Dict[str, Any]
):
    """Save player prediction backtest results to disk."""
    variant_dir = RESULTS_DIR / f"{variant_name}_player_pred_results"
    variant_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_data = {
        'variant': variant_name,
        'evaluation_type': 'player_prediction',
        'timestamp': datetime.now().isoformat(),
        'best_config': best_config,
        'n_configs_tested': len(all_results),
        'n_folds': len(all_results[0].fold_results) if all_results else 0
    }

    config_file = variant_dir / 'config.json'
    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2)

    # Save per-fold metrics
    rows = []
    for result in all_results:
        for fold in result.fold_results:
            row = fold.config.copy()
            row['fold_idx'] = fold.fold_idx
            row['eval_years'] = str(fold.eval_years)
            row['train_years'] = str(fold.train_years)
            row['pearson_net'] = fold.pearson_net
            row['spearman_net'] = fold.spearman_net
            row['rmse_net'] = fold.rmse_net
            row['pearson_off'] = fold.pearson_off
            row['spearman_off'] = fold.spearman_off
            row['rmse_off'] = fold.rmse_off
            row['pearson_def'] = fold.pearson_def
            row['spearman_def'] = fold.spearman_def
            row['rmse_def'] = fold.rmse_def
            row['n_players'] = fold.n_players
            rows.append(row)

    metrics_df = pd.DataFrame(rows)
    metrics_file = variant_dir / 'per_fold_metrics.csv'
    metrics_df.to_csv(metrics_file, index=False)

    # Save summary
    summary_rows = []
    for result in all_results:
        row = result.config.copy()
        row['avg_pearson_net'] = result.avg_pearson_net
        row['avg_spearman_net'] = result.avg_spearman_net
        row['avg_rmse_net'] = result.avg_rmse_net
        row['std_rmse_net'] = result.std_rmse_net
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values('avg_pearson_net', ascending=False)
    summary_file = variant_dir / 'config_summary.csv'
    summary_df.to_csv(summary_file, index=False)

    # Save best params
    best_df = pd.DataFrame([best_config])
    best_file = variant_dir / 'best_params.csv'
    best_df.to_csv(best_file, index=False)

    logging.info(f"Player prediction results saved to {variant_dir}")
