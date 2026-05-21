#!/usr/bin/env python3
"""
RAPM Analysis Script (Pipeline Version)

Runs RAPM analysis on processed NBA data.
Reads from nba_pipeline/processed/ and outputs to nba_pipeline/results/

Usage:
    python rapm.py <prefix> <start_year> <end_year> <season_type> [--pure]

Examples:
    python rapm.py RAPM 25 25 RS
    python rapm.py RAPM 23 26 ALL --pure
    python rapm.py TS 24 26 RS
"""

import logging
import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix, csr_matrix, coo_matrix
from sklearn.linear_model import Ridge
from collections import defaultdict
import multiprocessing as mp
from functools import partial
from pathlib import Path

# Get the pipeline root directory (parent of scripts/)
SCRIPT_DIR = Path(__file__).parent
PIPELINE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = PIPELINE_ROOT.parent
PROCESSED_DIR = PIPELINE_ROOT / "processed"
RESULTS_DIR = PIPELINE_ROOT / "results"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Determine number of cores to use (leave 1 core free for system)
N_CORES = max(1, mp.cpu_count() - 1)

###############################################################################
# 0) TIME DECAY UTILITIES
###############################################################################
def half_life_to_decay_base(half_life_days: float) -> float:
    """Convert half-life in days to decay_base."""
    return 0.5 ** (1.0 / half_life_days)


def compute_time_decay_weights(df: pd.DataFrame, decay_base: float = 0.999) -> np.ndarray:
    """
    Compute per-row weights using exponential time decay.
    weight_i = decay_base ^ age_days

    Args:
        df: DataFrame with 'date' column (from transform_to_home_away_format)
        decay_base: Daily decay rate (e.g., 0.999 for ~700 day half-life)

    Returns:
        Array of weights, shape (n_samples,)
    """
    dates = pd.to_datetime(df['date'], errors='coerce')
    reference_date = dates.max()
    age_days = (reference_date - dates).dt.total_seconds() / 86400.0
    age_days = age_days.fillna(0)
    weights = np.power(decay_base, age_days.values)
    weights = np.clip(weights, 0.0, 1.0)
    return weights.astype(np.float64)


###############################################################################
# 1) DETECT FILE TYPE AND PREPARE COLUMNS
###############################################################################
def detect_file_type_and_prepare(df, pure=False):
    """
    Detect the type of CSV file and prepare Off_Diff and Def_Diff columns.

    File types:
    - RAPM: has Net_Diff, Off_Diff, Def_Diff already
    - SPECIAL_RAPM: has Net_Diff, Off_Diff, Def_Diff (same as RAPM)
    - TOV: has Is_Turnover -> create Off_Diff, Def_Diff
    - REB: has Offensive_Rebound -> create Off_Diff, Def_Diff
    - RIM_FREQ: has Is_Rim_Attempt -> create Off_Diff, Def_Diff
    - RIM_FG_PCT: has Is_Rim_Make -> create Off_Diff, Def_Diff
    - MIDRANGE_FREQ: has Is_Midrange_Attempt -> create Off_Diff, Def_Diff
    - TRANSITION_FREQ: has Is_Transition -> create Off_Diff, Def_Diff
    - TRANSITION_RIM: has Is_Transition_Rim -> create Off_Diff, Def_Diff
    - INITIAL_EV: has Initial_EV -> create Off_Diff, Def_Diff
    - TS: has Net_Diff -> create Off_Diff, Def_Diff

    Args:
        df: Input dataframe
        pure: If True, use Net_Diff for RAPM files (removes 3pt/FT luck adjustments)
    """
    df = df.copy()
    
    if 'Off_Diff' in df.columns and 'Def_Diff' in df.columns:
        if pure and 'Net_Diff' in df.columns:
            logging.info("Detected RAPM file - PURE mode (using Net_Diff, ignoring luck adjustments)")
            # Use Net_Diff instead of adjusted Off_Diff/Def_Diff
            df['Off_Diff'] = df['Net_Diff']
            df['Def_Diff'] = -df['Net_Diff']
        else:
            logging.info("Detected RAPM file (has Off_Diff and Def_Diff)")
            # Use existing Off_Diff/Def_Diff with luck adjustments
            pass
    elif 'Is_Turnover' in df.columns:
        logging.info("Detected TOV file (has Is_Turnover)")
        # For turnovers: offense commits turnover (bad), defense forces turnover (good)
        df['Off_Diff'] = -df['Is_Turnover']  # Negative for offense (bad)
        df['Def_Diff'] = df['Is_Turnover']   # Positive for defense (good)
    elif 'Offensive_Rebound' in df.columns:
        logging.info("Detected REB file (has Offensive_Rebound)")
        # For offensive rebounds: offense gets rebound (good), defense allows rebound (bad)
        df['Off_Diff'] = df['Offensive_Rebound']   # Positive for offense (good)
        df['Def_Diff'] = -df['Offensive_Rebound']  # Negative for defense (bad)
    elif 'Is_Rim_Attempt' in df.columns:
        logging.info("Detected RIM_FREQ file (has Is_Rim_Attempt)")
        # For rim frequency: offense attempts rim (good), defense allows rim attempt (bad)
        df['Off_Diff'] = df['Is_Rim_Attempt']    # Positive for offense (good for attacking rim)
        df['Def_Diff'] = -df['Is_Rim_Attempt']   # Negative for defense (bad to allow rim attempts)
    elif 'Is_Rim_Make' in df.columns:
        logging.info("Detected RIM_FG_PCT file (has Is_Rim_Make)")
        # For rim FG%: offense makes rim (good), defense allows rim make (bad)
        df['Off_Diff'] = df['Is_Rim_Make']    # Positive for offense (good to make rims)
        df['Def_Diff'] = -df['Is_Rim_Make']   # Negative for defense (bad to allow rim makes)
    elif 'Is_Midrange_Attempt' in df.columns:
        logging.info("Detected MIDRANGE_FREQ file (has Is_Midrange_Attempt)")
        # For midrange frequency: offense takes midrange, defense allows midrange
        df['Off_Diff'] = df['Is_Midrange_Attempt']
        df['Def_Diff'] = -df['Is_Midrange_Attempt']
    elif 'Is_Transition' in df.columns:
        logging.info("Detected TRANSITION_FREQ file (has Is_Transition)")
        # For transition frequency: offense plays in transition (good), defense allows transition (bad)
        df['Off_Diff'] = df['Is_Transition']
        df['Def_Diff'] = -df['Is_Transition']
    elif 'Is_Transition_Rim' in df.columns:
        logging.info("Detected TRANSITION_RIM file (has Is_Transition_Rim)")
        # For transition rim: offense gets transition rim (good), defense allows transition rim (bad)
        df['Off_Diff'] = df['Is_Transition_Rim']
        df['Def_Diff'] = -df['Is_Transition_Rim']
    elif 'Initial_EV' in df.columns:
        logging.info("Detected INITIAL_EV file (has Initial_EV)")
        # For initial expected value: offense shot quality, defense allows shot quality
        df['Off_Diff'] = df['Initial_EV']
        df['Def_Diff'] = -df['Initial_EV']
    elif 'Net_Diff' in df.columns:
        logging.info("Detected TS file (has Net_Diff)")
        # For true shooting: use Net_Diff as points
        df['Off_Diff'] = df['Net_Diff']   # Points scored by offense
        df['Def_Diff'] = -df['Net_Diff']  # Points allowed by defense (negative)
    else:
        logging.error("Unknown file type - no recognized metric column found")
        raise ValueError("Could not detect file type")
    
    return df

###############################################################################
# 2) TRANSFORM O/D FORMAT TO HOME/AWAY FORMAT
###############################################################################
def transform_to_home_away_format(df):
    """
    Transform O/D format (O1-O5, D1-D5) to home/away format (a1-a5, h1-h5).
    Uses Off_Diff to determine points scored per possession.
    """
    logging.info("Transforming O/D format to home/away format...")
    
    # Create output DataFrame
    transformed_rows = []
    
    # Process each game separately
    for game_id in df['game_id'].unique():
        game_df = df[df['game_id'] == game_id].copy().reset_index(drop=True)
        
        # Determine team assignments by tracking score changes
        away_players = set()
        home_players = set()
        
        # First pass: identify home vs away players
        for i in range(len(game_df)):
            row = game_df.iloc[i]
            
            # Get offensive and defensive players
            off_players = [row[f'O{j}'] for j in range(1, 6)]
            def_players = [row[f'D{j}'] for j in range(1, 6)]
            
            # Calculate score changes
            if i == 0:
                away_pts_scored = row['away_score']
                home_pts_scored = row['home_score']
            else:
                prev_row = game_df.iloc[i-1]
                away_pts_scored = row['away_score'] - prev_row['away_score']
                home_pts_scored = row['home_score'] - prev_row['home_score']
            
            # Determine who scored
            if away_pts_scored > 0:
                # Away team scored, so offense = away, defense = home
                away_players.update(off_players)
                home_players.update(def_players)
            elif home_pts_scored > 0:
                # Home team scored, so offense = home, defense = away
                home_players.update(off_players)
                away_players.update(def_players)
        
        # Second pass: create transformed rows
        for i in range(len(game_df)):
            row = game_df.iloc[i]
            
            # Get players
            off_players = [row[f'O{j}'] for j in range(1, 6)]
            def_players = [row[f'D{j}'] for j in range(1, 6)]
            
            # Use Off_Diff for points (already prepared by detect_file_type_and_prepare)
            pts = float(row['Off_Diff'])
            
            # Determine if offense is home or away based on player overlap
            off_is_home = len(set(off_players) & home_players) > len(set(off_players) & away_players)
            
            transformed_row = {
                'home_poss': 1 if off_is_home else 0,
                'pts': pts,
                'season': int(row['Season']),
                'date': row['game_date'],
                'period': int(row['period']),
                'gameid': str(row['game_id'])
            }
            
            # Assign players to home/away
            if off_is_home:
                # Offense = home, Defense = away
                for j in range(5):
                    transformed_row[f'h{j+1}'] = int(off_players[j])
                    transformed_row[f'a{j+1}'] = int(def_players[j])
            else:
                # Offense = away, Defense = home
                for j in range(5):
                    transformed_row[f'a{j+1}'] = int(off_players[j])
                    transformed_row[f'h{j+1}'] = int(def_players[j])
            
            transformed_rows.append(transformed_row)
    
    result_df = pd.DataFrame(transformed_rows)
    logging.info(f"Transformed {len(result_df)} possessions")
    return result_df

###############################################################################
# 3) BUILD SIMPLIFIED DESIGN MATRIX (OPTIMIZED)
###############################################################################
def build_simple_design_matrix(df, player_to_col):
    """
    Build sparse design matrix with only offense/defense player indicators.
    Vectorized version for speed - processes all rows at once.
    """
    logging.info("Building design matrix (vectorized)...")
    
    n_samples = len(df)
    n_features = len(player_to_col)
    
    # Extract player columns as numpy arrays for speed
    a_players = df[[f'a{k}' for k in range(1, 6)]].values.astype(int)
    h_players = df[[f'h{k}' for k in range(1, 6)]].values.astype(int)
    home_poss = df['home_poss'].values
    pts = df['pts'].values.astype(np.float64)
    
    # Determine offense/defense based on home_poss
    # Shape: (n_samples, 5)
    offense_players = np.where(home_poss[:, None] == 1, h_players, a_players)
    defense_players = np.where(home_poss[:, None] == 1, a_players, h_players)
    
    # Build sparse matrix using COO format (faster for construction)
    row_indices = []
    col_indices = []
    
    # Process offensive players
    for player_idx in range(5):
        player_ids = offense_players[:, player_idx]
        for i, pid in enumerate(player_ids):
            off_key = f"{pid}_off"
            if off_key in player_to_col:
                row_indices.append(i)
                col_indices.append(player_to_col[off_key])
    
    # Process defensive players
    for player_idx in range(5):
        player_ids = defense_players[:, player_idx]
        for i, pid in enumerate(player_ids):
            def_key = f"{pid}_def"
            if def_key in player_to_col:
                row_indices.append(i)
                col_indices.append(player_to_col[def_key])
    
    # Create sparse matrix
    data = np.ones(len(row_indices), dtype=np.float64)
    X = coo_matrix((data, (row_indices, col_indices)), 
                   shape=(n_samples, n_features), dtype=np.float64)
    
    logging.info(f"Design matrix shape: {X.shape}")
    return X.tocsr(), pts

###############################################################################
# 4) SIMPLIFIED ALTERNATING RIDGE REGRESSION
###############################################################################
def simplified_alternating_rapm(X, y, player_to_col, all_players,
                                 alpha_offense=3000, alpha_defense=3000,
                                 max_iter=200, tol=1e-4,
                                 def_prior=None, weights=None):
    """
    Simplified alternating minimization for RAPM.
    Only offense and defense coefficients, no other features.
    Optimized with multi-core support.

    Args:
        def_prior: Optional dict mapping player_id (str) to prior value.
                   If provided, defense coefficients shrink toward prior instead of zero.
        weights: Optional array of row weights for time decay.
                 If provided, applies sqrt(w) scaling to X and y.
    """
    prior_msg = " with defense prior" if def_prior is not None else ""
    weight_msg = " with time decay" if weights is not None else ""
    logging.info(f"Running simplified alternating Ridge regression{prior_msg}{weight_msg} (using {N_CORES} cores)...")
    
    n_samples, n_features = X.shape
    beta = np.zeros(n_features, dtype=np.float64)

    # Apply time decay weights if provided
    if weights is not None:
        sqrt_w = np.sqrt(weights).astype(np.float64)
        # Scale X rows by sqrt(w)
        X = X.multiply(sqrt_w[:, np.newaxis]).tocsr()
        # Scale y by sqrt(w)
        y = y * sqrt_w
        logging.info(f"Applied time decay weights (min={weights.min():.4f}, max={weights.max():.4f})")

    # Separate offense and defense indices
    offense_indices = []
    defense_indices = []
    
    for k, v in player_to_col.items():
        if k.endswith('_off'):
            offense_indices.append(v)
        elif k.endswith('_def'):
            defense_indices.append(v)
    
    offense_indices = np.array(offense_indices, dtype=int)
    defense_indices = np.array(defense_indices, dtype=int)
    
    X_off = X[:, offense_indices]
    X_def = X[:, defense_indices]
    
    # Use solver='lsqr' for sparse matrices - faster than default 'auto'
    ridge_off = Ridge(alpha=alpha_offense, fit_intercept=False, solver='lsqr')
    ridge_def = Ridge(alpha=alpha_defense, fit_intercept=False, solver='lsqr')
    
    residual = y.copy()
    
    for iteration in range(max_iter):
        beta_prev = beta.copy()
        
        # Update offense coefficients
        residual += X_off @ beta[offense_indices]
        ridge_off.fit(X_off, residual)
        beta_off = ridge_off.coef_
        beta[offense_indices] = beta_off
        residual -= X_off @ beta_off
        
        # Update defense coefficients
        residual += X_def @ beta[defense_indices]

        if def_prior is not None:
            # Build prior vector for defense indices (in player order)
            prior_vec = np.array([def_prior.get(p, 0.0) for p in all_players], dtype=np.float64)
            # Adjust target: subtract X_def @ prior from residual
            # This transforms: min ||residual - X_def @ β||² + α||β - prior||²
            # Into: min ||adjusted_residual - X_def @ β*||² + α||β*||²
            # Where β = β* + prior
            adjusted_residual = residual - X_def @ prior_vec
            ridge_def.fit(X_def, adjusted_residual)
            # Add prior back to get final coefficients
            beta_def = ridge_def.coef_ + prior_vec
        else:
            ridge_def.fit(X_def, residual)
            beta_def = ridge_def.coef_

        beta[defense_indices] = beta_def
        residual -= X_def @ beta_def
        
        # Check convergence
        delta_beta = np.linalg.norm(beta - beta_prev)
        if (iteration + 1) % 20 == 0:
            logging.info(f"Iteration {iteration+1}, delta_beta={delta_beta:.6f}")
        if delta_beta < tol:
            logging.info(f"Converged after {iteration+1} iterations")
            break
    else:
        logging.info("Max iterations reached")
    
    return beta

###############################################################################
# HELPER: FILE LOADING FUNCTIONS
###############################################################################
def _load_single_file(input_file, pure):
    """Load and prepare a single file. Returns None if file can't be loaded."""
    if not Path(input_file).exists():
        return None

    try:
        df = pd.read_parquet(input_file)
        df = detect_file_type_and_prepare(df, pure=pure)
        return (input_file, df, len(df))
    except Exception as e:
        return None

def _load_files_parallel(input_files, pure):
    """Load multiple files in parallel using multiprocessing."""
    from multiprocessing import Pool
    
    with Pool(processes=N_CORES) as pool:
        load_func = partial(_load_single_file, pure=pure)
        results = pool.map(load_func, input_files)
    
    # Filter out None results and log
    all_dfs = []
    for i, result in enumerate(results):
        if result is None:
            logging.warning(f"File not found or error loading: {input_files[i]}")
        else:
            filename, df, n_rows = result
            logging.info(f"  Loaded {n_rows} possessions from {filename}")
            all_dfs.append(df)
    
    return all_dfs

def _load_files_sequential(input_files, pure):
    """Load files sequentially (for single file or fallback)."""
    all_dfs = []
    for input_file in input_files:
        if not Path(input_file).exists():
            logging.warning(f"File not found, skipping: {input_file}")
            continue

        logging.info(f"Loading {input_file}...")
        try:
            df = pd.read_parquet(input_file)
            logging.info(f"  Loaded {len(df)} possessions from {input_file}")
            df = detect_file_type_and_prepare(df, pure=pure)
            all_dfs.append(df)
        except Exception as e:
            logging.warning(f"Error loading {input_file}: {e}. Skipping.")
            continue

    return all_dfs

###############################################################################
# 5) MAIN EXECUTION
###############################################################################
def run_simplified_rapm(input_files, name_map_file=None, pure=False,
                        def_prior_file=None, def_alpha=None, off_alpha=None,
                        timedecay=False, half_life=None):
    """
    Main function to run simplified RAPM analysis.

    Args:
        input_files: Single file path (str) or list of file paths for multi-year analysis
        name_map_file: Path to player name mapping CSV (defaults to project root)
        pure: If True, use Net_Diff for RAPM files (removes 3pt/FT luck adjustments)
        def_prior_file: Path to CSV with defense prior values (player_id, prior_value)
        def_alpha: Override defense alpha (default: 3000)
        off_alpha: Override offense alpha (default: 3000)
        timedecay: If True, apply time decay weighting
        half_life: Half-life in days for time decay (default: 700)
    """
    # Default name map file location
    if name_map_file is None:
        name_map_file = PROJECT_ROOT / "autocomplete_map.csv"
    
    # Ensure input_files is a list
    if isinstance(input_files, str):
        input_files = [input_files]
    
    # Prepend processed directory to files that don't have a directory path
    processed_files = []
    for f in input_files:
        f_path = Path(f)
        if not f_path.is_absolute() and f_path.parent == Path('.'):
            # No directory in path, prepend processed dir
            processed_files.append(str(PROCESSED_DIR / f))
        else:
            # Already has a directory path, use as-is
            processed_files.append(str(f))
    input_files = processed_files
    
    logging.info("=" * 60)
    if len(input_files) > 1:
        mode_str = f"MULTI-YEAR RAPM ANALYSIS ({len(input_files)} files)"
    else:
        mode_str = "SIMPLIFIED RAPM ANALYSIS"
    
    if pure:
        mode_str += " - PURE MODE"
    
    logging.info(mode_str)
    logging.info("=" * 60)
    
    # Load and combine data from all files
    # Use parallel loading if multiple files
    if len(input_files) > 1:
        logging.info(f"Using {N_CORES} cores for parallel file loading...")
        all_dfs = _load_files_parallel(input_files, pure)
    else:
        all_dfs = _load_files_sequential(input_files, pure)
    
    # Check if we loaded any files
    if not all_dfs:
        logging.error("No files were successfully loaded. Exiting.")
        return None
    
    # Combine all dataframes
    df = pd.concat(all_dfs, ignore_index=True)
    logging.info(f"Combined total: {len(df)} possessions across {len(all_dfs)} file(s)")
    
    # Load name mapping
    logging.info(f"Loading {name_map_file}...")
    try:
        name_df = pd.read_csv(name_map_file)
        name_map = dict(zip(name_df['nba_id'].astype(str), name_df['player_name']))
        logging.info(f"Loaded {len(name_map)} player names")
    except Exception as e:
        logging.warning(f"Could not load name map: {e}")
        name_map = {}

    # Load defense prior if specified
    def_prior = None
    if def_prior_file:
        logging.info(f"Loading defense prior from {def_prior_file}...")
        try:
            prior_df = pd.read_csv(def_prior_file)
            # Convert to dict, scale from per-100 to per-possession (divide by 100)
            def_prior = dict(zip(prior_df['player_id'].astype(str),
                                 prior_df['prior_value'] / 100))
            logging.info(f"Loaded prior for {len(def_prior)} players")
        except Exception as e:
            logging.warning(f"Could not load defense prior: {e}")
            def_prior = None

    # Transform data to home/away format
    df_transformed = transform_to_home_away_format(df)
    
    # Get all unique players
    all_players = set()
    for col in [f'a{i}' for i in range(1, 6)] + [f'h{i}' for i in range(1, 6)]:
        all_players.update(df_transformed[col].astype(int).astype(str).unique())
    all_players = sorted(all_players)
    logging.info(f"Found {len(all_players)} unique players")
    
    # Build player-to-column mapping
    player_to_col = {}
    idx = 0
    for p in all_players:
        player_to_col[f"{p}_off"] = idx
        idx += 1
        player_to_col[f"{p}_def"] = idx
        idx += 1
    
    # Build design matrix
    X, y = build_simple_design_matrix(df_transformed, player_to_col)

    # Compute time decay weights if enabled
    weights = None
    if timedecay:
        decay_half_life = half_life if half_life else 700
        decay_base = half_life_to_decay_base(decay_half_life)
        weights = compute_time_decay_weights(df_transformed, decay_base=decay_base)
        logging.info(f"Time decay enabled: half-life={decay_half_life} days, decay_base={decay_base:.6f}")
        # Use weighted mean for centering
        y_mean = np.average(y, weights=weights)
    else:
        y_mean = np.mean(y)

    # Center the outcome
    y_centered = y - y_mean
    logging.info(f"Centered outcome, mean={y_mean:.3f}")
    
    # Run alternating Ridge regression
    actual_off_alpha = off_alpha if off_alpha is not None else 3000
    actual_def_alpha = def_alpha if def_alpha is not None else 3000
    beta = simplified_alternating_rapm(
        X, y_centered, player_to_col, all_players,
        alpha_offense=actual_off_alpha, alpha_defense=actual_def_alpha,
        max_iter=200, tol=1e-4,
        def_prior=def_prior,
        weights=weights
    )
    
    # Calculate player possessions (total, offensive, defensive)
    player_possessions = defaultdict(int)
    player_off_poss = defaultdict(int)
    player_def_poss = defaultdict(int)
    
    # Count from transformed data for total possessions
    for _, row in df_transformed.iterrows():
        away_pl = [int(row[f'a{k}']) for k in range(1, 6)]
        home_pl = [int(row[f'h{k}']) for k in range(1, 6)]
        for p in away_pl + home_pl:
            player_possessions[str(p)] += 1
    
    # Count from original data for offensive/defensive possessions
    for _, row in df.iterrows():
        off_pl = [str(int(row[f'O{k}'])) for k in range(1, 6) if pd.notna(row.get(f'O{k}'))]
        def_pl = [str(int(row[f'D{k}'])) for k in range(1, 6) if pd.notna(row.get(f'D{k}'))]
        for p in off_pl:
            player_off_poss[p] += 1
        for p in def_pl:
            player_def_poss[p] += 1
    
    # Re-center offense and defense separately so weighted averages = 0
    sum_off = 0.0
    sum_def = 0.0
    sum_poss = 0.0
    
    for p in all_players:
        off_key = f"{p}_off"
        def_key = f"{p}_def"
        off_val = beta[player_to_col[off_key]]
        def_val = beta[player_to_col[def_key]]
        poss = player_possessions[p]
        sum_off += off_val * poss
        sum_def += def_val * poss
        sum_poss += poss
    
    if sum_poss > 0:
        off_offset = sum_off / sum_poss
        def_offset = sum_def / sum_poss
        logging.info(f"Re-centering offense by {off_offset:.4f}, defense by {def_offset:.4f}")
        
        for p in all_players:
            off_key = f"{p}_off"
            def_key = f"{p}_def"
            beta[player_to_col[off_key]] -= off_offset
            beta[player_to_col[def_key]] -= def_offset
    
    # Build results DataFrame
    results = []
    for p in all_players:
        off_key = f"{p}_off"
        def_key = f"{p}_def"
        off_val = beta[player_to_col[off_key]]
        def_val = beta[player_to_col[def_key]]
        net_rapm = off_val - def_val
        poss = player_possessions[p]
        
        player_name = name_map.get(p, f"ID_{p}")
        
        # Multiply by 100 and round to 2 decimal places
        results.append({
            'player_id': p,
            'player_name': player_name,
            'off': round(off_val * 100, 2),
            'def': round(def_val * 100, 2),
            'net_rapm': round(net_rapm * 100, 2),
            'possessions': poss,
            'off_poss': player_off_poss[p],
            'def_poss': player_def_poss[p]
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('net_rapm', ascending=False).reset_index(drop=True)
    
    # Ensure results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Extract metadata from input files for naming
    years = []
    file_type = None
    has_ps = False
    has_rs = False
    
    for f in input_files:
        base = Path(f).stem
        
        # Extract file type (RAPM, TOV, REB, TS)
        if file_type is None:
            file_type = ''.join([c for c in base if c.isalpha()]).lower()
            # Remove 'ps' suffix from type if present
            if file_type.endswith('ps'):
                file_type = file_type[:-2]
        
        # Check if this is a playoff file
        if base.endswith('_PS'):
            has_ps = True
        else:
            has_rs = True
        
        # Extract year digits
        year_str = ''.join([c for c in base if c.isdigit()])
        if year_str:
            years.append(year_str)
    
    # Determine season type suffix
    if has_ps and has_rs:
        season_suffix = "all"
    elif has_ps:
        season_suffix = "ps"
    else:
        season_suffix = "rs"
    
    # Build filename with start_end year format
    if years:
        years_sorted = sorted(set(years))
        start_year = years_sorted[0]
        end_year = years_sorted[-1]

        if start_year == end_year:
            year_range = start_year
        else:
            year_range = f"{start_year}_{end_year}"

        # Build suffix based on options
        suffix_parts = [season_suffix]
        if pure:
            suffix_parts.append("pure")
        if actual_off_alpha != 3000 or actual_def_alpha != 3000:
            suffix_parts.append(f"a{int(actual_off_alpha)}_{int(actual_def_alpha)}")
        if def_prior is not None:
            suffix_parts.append("prior")
        if timedecay:
            td_half = half_life if half_life else 700
            suffix_parts.append(f"td{int(td_half)}")

        suffix = "_".join(suffix_parts)
        # Use _td subfolder for time decay results
        if timedecay:
            output_dir = RESULTS_DIR / "td"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"{file_type}_{year_range}_{suffix}_results.csv"
        else:
            output_file = RESULTS_DIR / f"{file_type}_{year_range}_{suffix}_results.csv"
    else:
        # Fallback
        suffix_parts = [season_suffix]
        if pure:
            suffix_parts.append("pure")
        if actual_off_alpha != 3000 or actual_def_alpha != 3000:
            suffix_parts.append(f"a{int(actual_off_alpha)}_{int(actual_def_alpha)}")
        if def_prior is not None:
            suffix_parts.append("prior")
        if timedecay:
            td_half = half_life if half_life else 700
            suffix_parts.append(f"td{int(td_half)}")

        suffix = "_".join(suffix_parts)
        if timedecay:
            output_dir = RESULTS_DIR / "td"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"{file_type}_{suffix}_results.csv"
        else:
            output_file = RESULTS_DIR / f"{file_type}_{suffix}_results.csv"
    
    results_df.to_csv(output_file, index=False)
    logging.info(f"Saved results to {output_file}")
    
    # Display top 20
    logging.info("\n" + "=" * 60)
    logging.info("TOP 20 PLAYERS BY NET RAPM")
    logging.info("=" * 60)
    for i, row in results_df.head(20).iterrows():
        logging.info(
            f"{i+1:2d}. {row['player_name']:25s} | "
            f"Net: {row['net_rapm']:+.2f} | "
            f"Off: {row['off']:+.2f} | "
            f"Def: {row['def']:+.2f} | "
            f"Poss: {row['possessions']:5d}"
        )
    
    return results_df

###############################################################################
# RUN IF MAIN
###############################################################################
if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run RAPM analysis on processed NBA data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single year regular season
  python rapm.py RAPM 25 25 RS
  
  # Multi-year regular season (2024-25 through 2025-26)
  python rapm.py RAPM 25 26 RS
  
  # Playoffs only
  python rapm.py RAPM 23 25 PS
  
  # Both regular season and playoffs combined
  python rapm.py TOV 24 26 ALL
  
  # Add --pure flag for pure RAPM (no luck adjustments)
  python rapm.py RAPM 24 26 RS --pure
  
  # Use all CPU cores for maximum speed
  python rapm.py RAPM 23 26 ALL --cores 16

  # Time decay with 700-day half-life (outputs to results/td/)
  python rapm.py RAPM 21 26 ALL --timedecay --half-life 700
        """
    )
    
    parser.add_argument('prefix',
                       choices=['RAPM', 'TOV', 'REB', 'TS', 'RIM_FREQ', 'RIM_FG_PCT', 'LA_RAPM', 'MIDRANGE_FREQ',
                                'TRANSITION_FREQ', 'TRANSITION_RIM', 'INITIAL_EV', 'SPECIAL_RAPM', 'EV_RAPM'],
                       help='Data type to analyze')
    parser.add_argument('start_year', 
                       type=int,
                       help='Starting year (2-digit, e.g., 24 for 2024-25)')
    parser.add_argument('end_year', 
                       type=int,
                       help='Ending year (2-digit, inclusive, e.g., 26 for 2025-26)')
    parser.add_argument('season_type',
                       choices=['RS', 'PS', 'ALL'],
                       help='Season type: RS (regular season), PS (playoffs), or ALL (both)')
    parser.add_argument('--pure',
                       action='store_true',
                       help='Use pure RAPM mode (removes 3pt/FT luck adjustments)')
    
    # Determine available cores for help text
    default_cores = max(1, mp.cpu_count() - 1)
    max_cores = mp.cpu_count()
    
    parser.add_argument('--cores', '-j',
                       type=int,
                       default=None,
                       help=f'Number of CPU cores to use (default: {default_cores}, max: {max_cores})')

    parser.add_argument('--def-prior',
                       type=str,
                       default=None,
                       help='CSV file with defense prior values (player_id, prior_value). Shrinks defense coefficients toward prior instead of zero.')

    parser.add_argument('--def-alpha',
                       type=float,
                       default=None,
                       help='Override defense alpha (higher = stronger pull toward prior/zero). Default: 3000')

    parser.add_argument('--off-alpha',
                       type=float,
                       default=None,
                       help='Override offense alpha (higher = more regularization). Default: 3000')

    parser.add_argument('--timedecay', '-td',
                       action='store_true',
                       help='Apply time decay weighting (recent games weighted more)')

    parser.add_argument('--half-life',
                       type=float,
                       default=None,
                       help='Half-life in days for time decay (default: 700). Implies --timedecay.')

    args = parser.parse_args()

    # --half-life implies --timedecay
    if args.half_life is not None:
        args.timedecay = True
    
    # Determine how many cores to use and update global
    if args.cores is not None:
        cores_to_use = max(1, min(args.cores, mp.cpu_count()))
        logging.info(f"Using {cores_to_use} CPU cores (user specified)")
        # Update the global for use in functions
        globals()['N_CORES'] = cores_to_use
    else:
        cores_to_use = N_CORES
        logging.info(f"Using {cores_to_use} CPU cores (auto-detected)")
    
    # Build list of files to process
    input_files = []
    years = range(args.start_year, args.end_year + 1)
    
    for year in years:
        if args.season_type in ['RS', 'ALL']:
            # Regular season file
            filename = f"{args.prefix}{year:02d}.parquet"
            input_files.append(filename)

        if args.season_type in ['PS', 'ALL']:
            # Playoff file
            filename = f"{args.prefix}{year:02d}_PS.parquet"
            input_files.append(filename)
    
    # Log what we're processing
    logging.info(f"Processing {args.prefix} data for years {args.start_year}-{args.end_year} ({args.season_type})")
    logging.info(f"Files to process: {', '.join(input_files)}")
    
    # Run the analysis
    results = run_simplified_rapm(input_files, pure=args.pure,
                                  def_prior_file=args.def_prior,
                                  def_alpha=args.def_alpha,
                                  off_alpha=args.off_alpha,
                                  timedecay=args.timedecay,
                                  half_life=args.half_life)
    
    if results is not None:
        logging.info("\nDone!")
    else:
        logging.error("\nAnalysis failed - no data was processed.")
        sys.exit(1)

