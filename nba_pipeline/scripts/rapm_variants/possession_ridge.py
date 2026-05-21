"""
Possession-weighted ridge regression solver.

sklearn's Ridge doesn't support per-coefficient penalties.
This module provides a custom solver using the normal equations:

    min ||y - Xb||^2 + sum(lambda_j * b_j^2)

Solution: b = (X'X + diag(lambda))^-1 X'y
"""

import logging
import numpy as np
from scipy.sparse import diags, csr_matrix, issparse
from scipy.sparse.linalg import cg
from sklearn.linear_model import Ridge
from typing import Optional, Tuple, Dict, List

from .base_rapm import N_REF, N0, M_MIN

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


###############################################################################
# MULTIPLIER COMPUTATION
###############################################################################

def compute_multiplier(
    N_j: int,
    N_ref: int = N_REF,
    N0: int = N0,
    p: float = 0.5,
    m_min: float = M_MIN,
    m_max: float = 3.0
) -> float:
    """
    Compute possession-weighted penalty multiplier for a player.

    m_j = clamp((N_ref / (N_j + N0))^p, m_min, m_max)

    Args:
        N_j: Player's possession count
        N_ref: Reference possessions (default 2000)
        N0: Smoothing constant (default 200)
        p: Power exponent (0.0 = no scaling, 1.0 = linear)
        m_min: Minimum multiplier
        m_max: Maximum multiplier

    Returns:
        Multiplier value
    """
    if p == 0.0:
        return 1.0

    ratio = N_ref / (N_j + N0)
    m_j = ratio ** p
    return max(m_min, min(m_max, m_j))


def build_lambda_vector(
    players: List[str],
    player_to_col: Dict[str, int],
    off_poss: Dict[str, int],
    def_poss: Dict[str, int],
    lambda_off_base: float,
    lambda_def_base: float,
    p: float = 0.5,
    m_max: float = 3.0,
    N_ref: int = N_REF,
    N0: int = N0,
    m_min: float = M_MIN
) -> np.ndarray:
    """
    Build per-coefficient lambda vector for possession-weighted ridge.

    Args:
        players: List of player IDs
        player_to_col: Mapping from player keys to column indices
        off_poss: Offensive possessions per player
        def_poss: Defensive possessions per player
        lambda_off_base: Base lambda for offense
        lambda_def_base: Base lambda for defense
        p: Power exponent for multiplier
        m_max: Maximum multiplier

    Returns:
        Array of lambda values, shape (n_features,)
    """
    n_features = len(player_to_col)
    lambda_vec = np.zeros(n_features, dtype=np.float64)

    for player in players:
        # Offense coefficient
        off_key = f"{player}_off"
        if off_key in player_to_col:
            n_off = off_poss.get(player, 0)
            m_off = compute_multiplier(n_off, N_ref, N0, p, m_min, m_max)
            lambda_vec[player_to_col[off_key]] = lambda_off_base * m_off

        # Defense coefficient
        def_key = f"{player}_def"
        if def_key in player_to_col:
            n_def = def_poss.get(player, 0)
            m_def = compute_multiplier(n_def, N_ref, N0, p, m_min, m_max)
            lambda_vec[player_to_col[def_key]] = lambda_def_base * m_def

    return lambda_vec


###############################################################################
# CUSTOM RIDGE SOLVER
###############################################################################

def possession_weighted_ridge(
    X: csr_matrix,
    y: np.ndarray,
    lambda_vec: np.ndarray,
    tol: float = 1e-6,
    maxiter: int = 2000
) -> np.ndarray:
    """
    Solve ridge regression with per-coefficient penalties.

    min ||y - Xb||^2 + sum(lambda_j * b_j^2)

    Solution: b = (X'X + diag(lambda))^-1 X'y

    Uses conjugate gradient for efficiency with sparse matrices.

    Args:
        X: Design matrix (sparse CSR)
        y: Target vector
        lambda_vec: Per-coefficient penalty values
        tol: Convergence tolerance
        maxiter: Maximum iterations

    Returns:
        Coefficient vector
    """
    # Ensure X is CSR
    if not isinstance(X, csr_matrix):
        X = csr_matrix(X)

    # Build (X'X + diag(lambda))
    XtX = X.T @ X
    penalty = diags(lambda_vec, format='csr')
    A = XtX + penalty

    # Build X'y
    Xty = X.T @ y

    # Solve using conjugate gradient
    beta, info = cg(A.tocsr(), Xty, rtol=tol, maxiter=maxiter)

    if info > 0:
        logging.warning(f"CG did not converge after {maxiter} iterations")
    elif info < 0:
        logging.error(f"CG illegal input or breakdown")

    return beta


###############################################################################
# ALTERNATING RIDGE WITH PER-COEFFICIENT PENALTIES
###############################################################################

def alternating_possession_ridge(
    X: csr_matrix,
    y: np.ndarray,
    players: List[str],
    player_to_col: Dict[str, int],
    off_poss: Dict[str, int],
    def_poss: Dict[str, int],
    lambda_off_base: float,
    lambda_def_base: float,
    p: float = 0.5,
    m_max: float = 3.0,
    max_iter: int = 200,
    tol: float = 1e-4
) -> np.ndarray:
    """
    Alternating minimization RAPM with possession-weighted penalties.

    Alternates between:
    1. Fix defense, optimize offense with per-player penalties
    2. Fix offense, optimize defense with per-player penalties

    Args:
        X: Full design matrix
        y: Target (centered)
        players: List of player IDs
        player_to_col: Column mapping
        off_poss, def_poss: Possession counts
        lambda_off_base, lambda_def_base: Base penalties
        p: Multiplier power
        m_max: Max multiplier
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        Coefficient vector
    """
    logging.info(
        f"Running alternating possession-weighted ridge: "
        f"lambda_off={lambda_off_base}, lambda_def={lambda_def_base}, "
        f"p={p}, m_max={m_max}"
    )

    n_features = len(player_to_col)
    beta = np.zeros(n_features, dtype=np.float64)

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

    X_off = X[:, offense_indices]
    X_def = X[:, defense_indices]

    # Build per-player lambda vectors (for offense and defense subsets)
    # Need to map from full indices to subset indices
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

    residual = y.copy()

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
# STANDARD ALTERNATING RIDGE (for baseline/timedecay without possession weighting)
###############################################################################

def alternating_ridge(
    X: csr_matrix,
    y: np.ndarray,
    player_to_col: Dict[str, int],
    alpha_offense: float = 3000.0,
    alpha_defense: float = 3000.0,
    max_iter: int = 200,
    tol: float = 1e-4
) -> np.ndarray:
    """
    Standard alternating minimization RAPM with uniform penalties.

    Uses sklearn Ridge for speed.

    Args:
        X: Design matrix
        y: Target (centered, optionally weighted)
        player_to_col: Column mapping
        alpha_offense: Ridge penalty for offense
        alpha_defense: Ridge penalty for defense
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        Coefficient vector
    """
    logging.info(
        f"Running standard alternating ridge: "
        f"alpha_off={alpha_offense}, alpha_def={alpha_defense}"
    )

    n_features = len(player_to_col)
    beta = np.zeros(n_features, dtype=np.float64)

    # Separate indices
    offense_indices = []
    defense_indices = []

    for key, idx in player_to_col.items():
        if key.endswith('_off'):
            offense_indices.append(idx)
        elif key.endswith('_def'):
            defense_indices.append(idx)

    offense_indices = np.array(offense_indices, dtype=int)
    defense_indices = np.array(defense_indices, dtype=int)

    X_off = X[:, offense_indices]
    X_def = X[:, defense_indices]

    ridge_off = Ridge(alpha=float(alpha_offense), fit_intercept=False, solver='lsqr')
    ridge_def = Ridge(alpha=float(alpha_defense), fit_intercept=False, solver='lsqr')

    residual = y.copy()

    for iteration in range(max_iter):
        beta_prev = beta.copy()

        # Update offense
        residual += X_off @ beta[offense_indices]
        ridge_off.fit(X_off, residual)
        beta[offense_indices] = ridge_off.coef_
        residual -= X_off @ ridge_off.coef_

        # Update defense
        residual += X_def @ beta[defense_indices]
        ridge_def.fit(X_def, residual)
        beta[defense_indices] = ridge_def.coef_
        residual -= X_def @ ridge_def.coef_

        delta_beta = np.linalg.norm(beta - beta_prev)
        if (iteration + 1) % 20 == 0:
            logging.debug(f"Iteration {iteration+1}, delta_beta={delta_beta:.6f}")
        if delta_beta < tol:
            logging.info(f"Converged after {iteration+1} iterations")
            break
    else:
        logging.info("Max iterations reached")

    return beta
