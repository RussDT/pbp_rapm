import logging
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.linear_model import Ridge
from collections import defaultdict
import multiprocessing as mp
from functools import partial

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Determine number of cores to use (leave 1 core free for system)
N_CORES = max(1, mp.cpu_count() - 1)

###############################################################################
# 1) DETECT FILE TYPE AND PREPARE COLUMNS
###############################################################################
def detect_file_type_and_prepare(df, pure=False):
    """
    Detect the type of CSV file and prepare Off_Diff and Def_Diff columns.

    File types:
    - RAPM: has Net_Diff, Off_Diff, Def_Diff already
    - TOV: has Is_Turnover -> create Off_Diff, Def_Diff
    - REB: has Offensive_Rebound -> create Off_Diff, Def_Diff
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

    NOTE: This uses game score deltas to infer which lineup is home vs away.
    """
    logging.info("Transforming O/D format to home/away format...")

    transformed_rows = []

    for game_id in df['game_id'].unique():
        game_df = df[df['game_id'] == game_id].copy().reset_index(drop=True)

        away_players = set()
        home_players = set()

        # First pass: infer home vs away rosters
        for i in range(len(game_df)):
            row = game_df.iloc[i]

            off_players = [row[f'O{j}'] for j in range(1, 6)]
            def_players = [row[f'D{j}'] for j in range(1, 6)]

            if i == 0:
                away_pts_scored = row['away_score']
                home_pts_scored = row['home_score']
            else:
                prev_row = game_df.iloc[i-1]
                away_pts_scored = row['away_score'] - prev_row['away_score']
                home_pts_scored = row['home_score'] - prev_row['home_score']

            if away_pts_scored > 0:
                away_players.update(off_players)
                home_players.update(def_players)
            elif home_pts_scored > 0:
                home_players.update(off_players)
                away_players.update(def_players)

        # Second pass: build rows
        for i in range(len(game_df)):
            row = game_df.iloc[i]

            off_players = [row[f'O{j}'] for j in range(1, 6)]
            def_players = [row[f'D{j}'] for j in range(1, 6)]

            pts = float(row['Off_Diff'])

            off_is_home = len(set(off_players) & home_players) > len(set(off_players) & away_players)

            transformed_row = {
                'home_poss': 1 if off_is_home else 0,
                'pts': pts,
                'season': int(row['Season']) if 'Season' in row else -1,
                'date': row['game_date'] if 'game_date' in row else None,
                'period': int(row['period']) if 'period' in row else -1,
                'gameid': str(row['game_id'])
            }

            if off_is_home:
                for j in range(5):
                    transformed_row[f'h{j+1}'] = int(off_players[j])
                    transformed_row[f'a{j+1}'] = int(def_players[j])
            else:
                for j in range(5):
                    transformed_row[f'a{j+1}'] = int(off_players[j])
                    transformed_row[f'h{j+1}'] = int(def_players[j])

            transformed_rows.append(transformed_row)

    result_df = pd.DataFrame(transformed_rows)
    logging.info(f"Transformed {len(result_df)} possessions")
    return result_df

###############################################################################
# 2.5) TIME-DECAY WEIGHTS
###############################################################################
def compute_time_decay_weights(df_transformed, half_life_days=None):
    """
    Compute per-possession weights using an exponential half-life in DAYS.

    weight = 0.5 ** (age_days / half_life_days)

    If half_life_days is None, returns all-ones.

    If dates are missing/unparseable, those rows get weight=1.0.
    """
    n = len(df_transformed)
    if half_life_days is None:
        return np.ones(n, dtype=np.float64)

    dates = pd.to_datetime(df_transformed.get('date', None), errors='coerce')
    if dates is None or dates.isna().all():
        logging.warning("half_life provided but no valid dates found; using uniform weights.")
        return np.ones(n, dtype=np.float64)

    max_date = dates.max()
    age_days = (max_date - dates).dt.total_seconds() / 86400.0

    w = np.ones(n, dtype=np.float64)
    mask = ~age_days.isna()
    w[mask.values] = np.power(0.5, (age_days[mask].values / float(half_life_days)))

    # guard for any weird negatives (future dates)
    w = np.clip(w, 0.0, 1.0)

    logging.info(
        f"Time-decay weights: half_life_days={half_life_days}, "
        f"min={w.min():.6f}, median={np.median(w):.6f}, max={w.max():.6f}, sum={w.sum():.1f}"
    )
    return w

###############################################################################
# 3) BUILD SIMPLIFIED DESIGN MATRIX (OPTIMIZED)
###############################################################################
def build_simple_design_matrix(df, player_to_col):
    """
    Build sparse design matrix with only offense/defense player indicators.

    NOTE: We model y (pts) as:
        y ≈ sum(offense_betas) + sum(defense_betas)
    where defense_betas represent "points allowed" (lower is better).

    Returns:
        X (csr), y (np array)
    """
    logging.info("Building design matrix (vectorized)...")

    n_samples = len(df)
    n_features = len(player_to_col)

    a_players = df[[f'a{k}' for k in range(1, 6)]].values.astype(int)
    h_players = df[[f'h{k}' for k in range(1, 6)]].values.astype(int)
    home_poss = df['home_poss'].values
    pts = df['pts'].values.astype(np.float64)

    offense_players = np.where(home_poss[:, None] == 1, h_players, a_players)
    defense_players = np.where(home_poss[:, None] == 1, a_players, h_players)

    row_indices = []
    col_indices = []

    # offense
    for player_idx in range(5):
        player_ids = offense_players[:, player_idx]
        for i, pid in enumerate(player_ids):
            off_key = f"{pid}_off"
            if off_key in player_to_col:
                row_indices.append(i)
                col_indices.append(player_to_col[off_key])

    # defense
    for player_idx in range(5):
        player_ids = defense_players[:, player_idx]
        for i, pid in enumerate(player_ids):
            def_key = f"{pid}_def"
            if def_key in player_to_col:
                row_indices.append(i)
                col_indices.append(player_to_col[def_key])

    data = np.ones(len(row_indices), dtype=np.float64)
    X = coo_matrix((data, (row_indices, col_indices)),
                   shape=(n_samples, n_features), dtype=np.float64)

    logging.info(f"Design matrix shape: {X.shape}")
    return X.tocsr(), pts

###############################################################################
# 4) SIMPLIFIED ALTERNATING RIDGE REGRESSION
###############################################################################
def simplified_alternating_rapm(
    X, y, player_to_col,
    alpha_offense=3000.0, alpha_defense=3000.0,
    max_iter=200, tol=1e-4
):
    """
    Alternating minimization for RAPM with separate offense/defense ridge penalties.

    IMPORTANT:
    In this codebase, the ridge penalty knob is sklearn Ridge(alpha=...).
    So if someone calls it "lambda", it maps here as alpha_* after any scaling choice.

    Args:
        X: (n_samples, n_features) sparse matrix
        y: (n_samples,) outcome vector (already centered; may already be weighted via sqrt(w))
        player_to_col: mapping to indices
        alpha_offense: ridge alpha for offense coefficients
        alpha_defense: ridge alpha for defense coefficients
    """
    logging.info(f"Running simplified alternating Ridge regression (using {N_CORES} cores)...")
    logging.info(f"Ridge alphas: offense={alpha_offense:.6f}, defense={alpha_defense:.6f}")

    n_samples, n_features = X.shape
    beta = np.zeros(n_features, dtype=np.float64)

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

    ridge_off = Ridge(alpha=float(alpha_offense), fit_intercept=False, solver='lsqr')
    ridge_def = Ridge(alpha=float(alpha_defense), fit_intercept=False, solver='lsqr')

    residual = y.copy()

    for iteration in range(max_iter):
        beta_prev = beta.copy()

        # offense update
        residual += X_off @ beta[offense_indices]
        ridge_off.fit(X_off, residual)
        beta_off = ridge_off.coef_
        beta[offense_indices] = beta_off
        residual -= X_off @ beta_off

        # defense update
        residual += X_def @ beta[defense_indices]
        ridge_def.fit(X_def, residual)
        beta_def = ridge_def.coef_
        beta[defense_indices] = beta_def
        residual -= X_def @ beta_def

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
    import os
    if not os.path.exists(input_file):
        return None

    try:
        df = pd.read_csv(input_file)
        df = detect_file_type_and_prepare(df, pure=pure)
        return (input_file, df, len(df))
    except Exception:
        return None

def _load_files_parallel(input_files, pure):
    from multiprocessing import Pool

    with Pool(processes=N_CORES) as pool:
        load_func = partial(_load_single_file, pure=pure)
        results = pool.map(load_func, input_files)

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
    import os
    all_dfs = []
    for input_file in input_files:
        if not os.path.exists(input_file):
            logging.warning(f"File not found, skipping: {input_file}")
            continue

        logging.info(f"Loading {input_file}...")
        try:
            df = pd.read_csv(input_file)
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
def run_simplified_rapm(
    input_files,
    name_map_file="autocomplete_map.csv",
    pure=False,
    half_life_days=None,
    alpha_offense=3000.0,
    alpha_defense=3000.0,
    lambda_offense=None,
    lambda_defense=None,
    lambda_to_alpha='sumw'
):
    """
    Main function to run simplified RAPM analysis.

    Args:
        input_files: single file path (str) or list of files
        name_map_file: mapping CSV
        pure: pure RAPM toggle
        half_life_days: time-decay half life in days (None disables)
        alpha_offense/defense: Ridge alphas used in sklearn Ridge(alpha=...)
        lambda_offense/defense: Lambda values to convert to alpha (optional)
        lambda_to_alpha: How to convert lambda to alpha ('sumw', 'n', 'none')
    """
    if isinstance(input_files, str):
        input_files = [input_files]

    import os
    processed_files = []
    for f in input_files:
        if os.path.dirname(f) == '':
            processed_files.append(os.path.join('nba_pipeline', 'processed', f))
        else:
            processed_files.append(f)
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

    if len(input_files) > 1:
        logging.info(f"Using {N_CORES} cores for parallel file loading...")
        all_dfs = _load_files_parallel(input_files, pure)
    else:
        all_dfs = _load_files_sequential(input_files, pure)

    if not all_dfs:
        logging.error("No files were successfully loaded. Exiting.")
        return None

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

    # Transform
    df_transformed = transform_to_home_away_format(df)

    # All players
    all_players = set()
    for col in [f'a{i}' for i in range(1, 6)] + [f'h{i}' for i in range(1, 6)]:
        all_players.update(df_transformed[col].astype(int).astype(str).unique())
    all_players = sorted(all_players)
    logging.info(f"Found {len(all_players)} unique players")

    # Map
    player_to_col = {}
    idx = 0
    for p in all_players:
        player_to_col[f"{p}_off"] = idx
        idx += 1
        player_to_col[f"{p}_def"] = idx
        idx += 1

    # Build X, y
    X, y = build_simple_design_matrix(df_transformed, player_to_col)

    # Weights (time-decay)
    w = compute_time_decay_weights(df_transformed, half_life_days=half_life_days)

    # Lambda-to-alpha conversion (if lambda values provided)
    if lambda_offense is not None or lambda_defense is not None:
        n_samples = len(df_transformed)
        sum_weights = float(np.sum(w))
        
        if lambda_to_alpha == 'sumw':
            scale_factor = sum_weights
        elif lambda_to_alpha == 'n':
            scale_factor = n_samples
        else:  # 'none'
            scale_factor = 1.0
        
        if lambda_offense is not None:
            alpha_offense = float(lambda_offense) * scale_factor
            logging.info(f"Converted lambda_offense={lambda_offense} to alpha_offense={alpha_offense:.2f} (scale={scale_factor:.2f})")
        if lambda_defense is not None:
            alpha_defense = float(lambda_defense) * scale_factor
            logging.info(f"Converted lambda_defense={lambda_defense} to alpha_defense={alpha_defense:.2f} (scale={scale_factor:.2f})")

    # Weighted centering (since we fit_intercept=False)
    mu = np.average(y, weights=w) if w is not None else float(np.mean(y))
    y_centered = y - mu
    logging.info(f"Centered outcome, weighted_mean={mu:.6f}")

    # Apply weights by row-scaling: minimize sum w_i (y_i - x_i b)^2 + alpha ||b||^2
    sqrtw = np.sqrt(w).astype(np.float64)
    Xw = X.multiply(sqrtw[:, None]).tocsr()  # Ensure CSR format after multiply
    yw = y_centered * sqrtw

    # Fit
    beta = simplified_alternating_rapm(
        Xw, yw, player_to_col,
        alpha_offense=alpha_offense, alpha_defense=alpha_defense,
        max_iter=200, tol=1e-4
    )

    # Possessions (unweighted counts; you can switch to weights if you want)
    player_possessions = defaultdict(int)
    player_off_poss = defaultdict(int)
    player_def_poss = defaultdict(int)

    for _, row in df_transformed.iterrows():
        away_pl = [int(row[f'a{k}']) for k in range(1, 6)]
        home_pl = [int(row[f'h{k}']) for k in range(1, 6)]
        for p in away_pl + home_pl:
            player_possessions[str(p)] += 1

    for _, row in df.iterrows():
        off_pl = [str(int(row[f'O{k}'])) for k in range(1, 6) if pd.notna(row.get(f'O{k}'))]
        def_pl = [str(int(row[f'D{k}'])) for k in range(1, 6) if pd.notna(row.get(f'D{k}'))]
        for p in off_pl:
            player_off_poss[p] += 1
        for p in def_pl:
            player_def_poss[p] += 1

    # Re-center offense/defense so possession-weighted averages ~ 0 (using unweighted poss here)
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
        logging.info(f"Re-centering offense by {off_offset:.6f}, defense by {def_offset:.6f}")

        for p in all_players:
            beta[player_to_col[f"{p}_off"]] -= off_offset
            beta[player_to_col[f"{p}_def"]] -= def_offset

    # Results DF
    results = []
    for p in all_players:
        off_key = f"{p}_off"
        def_key = f"{p}_def"
        off_val = beta[player_to_col[off_key]]
        def_val = beta[player_to_col[def_key]]
        net_rapm = off_val - def_val
        poss = player_possessions[p]

        player_name = name_map.get(p, f"ID_{p}")

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

    results_df = pd.DataFrame(results).sort_values('net_rapm', ascending=False).reset_index(drop=True)

    # Output filename logic - outputs to nba_pipeline/results/ with _td suffix
    import os
    
    results_base_dir = os.path.join("nba_pipeline", "results")
    os.makedirs(results_base_dir, exist_ok=True)

    years = []
    file_type = None
    has_ps = False
    has_rs = False

    for f in input_files:
        base = os.path.splitext(os.path.basename(f))[0]

        if file_type is None:
            file_type = ''.join([c for c in base if c.isalpha()]).lower()
            if file_type.endswith('ps'):
                file_type = file_type[:-2]

        if base.endswith('_PS'):
            has_ps = True
        else:
            has_rs = True

        year_str = ''.join([c for c in base if c.isdigit()])
        if year_str:
            years.append(year_str)

    if has_ps and has_rs:
        season_suffix = "all"
    elif has_ps:
        season_suffix = "ps"
    else:
        season_suffix = "rs"

    if years:
        years_sorted = sorted(set(years))
        # Use start_end format (e.g., 23_26 instead of 23_24_25_26)
        start_year = years_sorted[0]
        end_year = years_sorted[-1]
        year_range = start_year if start_year == end_year else f"{start_year}_{end_year}"
        
        # Create folder with _td suffix
        folder_name = f"{file_type}_{year_range}_{season_suffix}_td"
        folder_path = os.path.join(results_base_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        
        if pure:
            output_file = os.path.join(folder_path, f"{file_type}_{year_range}_{season_suffix}_td_pure_results.csv")
        else:
            output_file = os.path.join(folder_path, f"{file_type}_{year_range}_{season_suffix}_td_results.csv")
    else:
        folder_name = f"{file_type}_{season_suffix}_td"
        folder_path = os.path.join(results_base_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        
        if pure:
            output_file = os.path.join(folder_path, f"{file_type}_{season_suffix}_td_pure_results.csv")
        else:
            output_file = os.path.join(folder_path, f"{file_type}_{season_suffix}_td_results.csv")

    results_df.to_csv(output_file, index=False)
    logging.info(f"Saved results to {output_file}")

    # Top 20
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
  python rapm.py RAPM 25 25 RS
  python rapm.py RAPM 25 26 RS
  python rapm.py RAPM 23 25 PS
  python rapm.py TOV 24 26 ALL
  python rapm.py RAPM 24 26 RS --pure
  python rapm.py RAPM 23 26 ALL --cores 16

  # Time decay + "lambda" style params (lambda -> alpha conversion controlled by --lambda-to-alpha):
  python rapm.py RAPM 23 25 ALL --pure --half-life 712.357 --lambda-off 0.009542 --lambda-def 0.022591
        """
    )

    parser.add_argument('prefix',
                        choices=['RAPM', 'TOV', 'REB', 'TS'],
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

    default_cores = max(1, mp.cpu_count() - 1)
    max_cores = mp.cpu_count()

    parser.add_argument('--cores', '-j',
                        type=int,
                        default=None,
                        help=f'Number of CPU cores to use (default: {default_cores}, max: {max_cores})')

    # Time decay
    parser.add_argument('--half-life',
                        type=float,
                        default=None,
                        help='Half-life in DAYS for time-decay weighting (disables if omitted).')

    # Direct alphas (sklearn Ridge alpha)
    parser.add_argument('--alpha-off',
                        type=float,
                        default=None,
                        help='Ridge alpha for offense (sklearn Ridge(alpha=...)). Overrides lambda conversion.')
    parser.add_argument('--alpha-def',
                        type=float,
                        default=None,
                        help='Ridge alpha for defense (sklearn Ridge(alpha=...)). Overrides lambda conversion.')

    # Lambda-style inputs (converted to alpha unless you set --lambda-to-alpha none)
    parser.add_argument('--lambda-off',
                        type=float,
                        default=None,
                        help='Lambda for offense (optionally converted to alpha via --lambda-to-alpha).')
    parser.add_argument('--lambda-def',
                        type=float,
                        default=None,
                        help='Lambda for defense (optionally converted to alpha via --lambda-to-alpha).')

    parser.add_argument('--lambda-to-alpha',
                        choices=['sumw', 'n', 'none'],
                        default='sumw',
                        help=(
                            "How to convert lambda -> alpha. "
                            "'sumw': alpha=lambda*sum(weights) (default). "
                            "'n': alpha=lambda*n_samples. "
                            "'none': alpha=lambda (no scaling)."
                        )
    )

    args = parser.parse_args()

    # cores
    if args.cores is not None:
        cores_to_use = max(1, min(args.cores, mp.cpu_count()))
        logging.info(f"Using {cores_to_use} CPU cores (user specified)")
        globals()['N_CORES'] = cores_to_use
    else:
        cores_to_use = N_CORES
        logging.info(f"Using {cores_to_use} CPU cores (auto-detected)")

    # build files
    input_files = []
    years = range(args.start_year, args.end_year + 1)

    for year in years:
        if args.season_type in ['RS', 'ALL']:
            input_files.append(f"{args.prefix}{year:02d}.csv")
        if args.season_type in ['PS', 'ALL']:
            input_files.append(f"{args.prefix}{year:02d}_PS.csv")

    logging.info(f"Processing {args.prefix} data for years {args.start_year}-{args.end_year} ({args.season_type})")
    logging.info(f"Files to process: {', '.join(input_files)}")

    name_map_file = "autocomplete_map.csv"

    # Default alphas
    alpha_off = 3000.0
    alpha_def = 3000.0

    # If user specified alphas explicitly, use them
    if args.alpha_off is not None:
        alpha_off = float(args.alpha_off)
    if args.alpha_def is not None:
        alpha_def = float(args.alpha_def)

    # Run the analysis - lambda conversion happens inside run_simplified_rapm
    results = run_simplified_rapm(
        input_files=input_files,
        name_map_file=name_map_file,
        pure=args.pure,
        half_life_days=args.half_life,
        alpha_offense=alpha_off,
        alpha_defense=alpha_def,
        lambda_offense=args.lambda_off,
        lambda_defense=args.lambda_def,
        lambda_to_alpha=args.lambda_to_alpha
    )

    if results is not None:
        logging.info("\nDone!")
    else:
        logging.error("\nAnalysis failed - no data was processed.")
        sys.exit(1)