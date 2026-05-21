import logging
import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix, csr_matrix, coo_matrix
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
    Also tracks running score differential for rubberband effect.
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
        
        # Second pass: create transformed rows with score tracking
        home_score = 0
        away_score = 0
        
        for i in range(len(game_df)):
            row = game_df.iloc[i]
            
            # Get players
            off_players = [row[f'O{j}'] for j in range(1, 6)]
            def_players = [row[f'D{j}'] for j in range(1, 6)]
            
            # Use Off_Diff for points (already prepared by detect_file_type_and_prepare)
            pts = float(row['Off_Diff'])
            
            # Determine if offense is home or away based on player overlap
            off_is_home = len(set(off_players) & home_players) > len(set(off_players) & away_players)
            
            # Calculate score_diff BEFORE updating score (offensive team's current margin)
            if off_is_home:
                score_diff = home_score - away_score
            else:
                score_diff = away_score - home_score
            
            transformed_row = {
                'home_poss': 1 if off_is_home else 0,
                'pts': pts,
                'season': int(row['Season']),
                'date': row['game_date'],
                'period': int(row['period']),
                'gameid': str(row['game_id']),
                'score_diff': score_diff
            }
            
            # Assign players to home/away
            if off_is_home:
                # Offense = home, Defense = away
                for j in range(5):
                    transformed_row[f'h{j+1}'] = int(off_players[j])
                    transformed_row[f'a{j+1}'] = int(def_players[j])
                # Update score
                home_score += pts
            else:
                # Offense = away, Defense = home
                for j in range(5):
                    transformed_row[f'a{j+1}'] = int(off_players[j])
                    transformed_row[f'h{j+1}'] = int(def_players[j])
                # Update score
                away_score += pts
            
            transformed_rows.append(transformed_row)
    
    result_df = pd.DataFrame(transformed_rows)
    logging.info(f"Transformed {len(result_df)} possessions")
    return result_df

###############################################################################
# 3) BUILD DESIGN MATRIX WITH RUBBERBAND EFFECT (OPTIMIZED)
###############################################################################
def build_design_matrix_with_rubberband(df, player_to_col, max_poly=3):
    """
    Build sparse design matrix with:
    - Offense/defense player indicators
    - Score differential polynomial features (rubberband effect)
    Vectorized version for speed - processes all rows at once.
    """
    logging.info("Building design matrix with rubberband features (vectorized)...")
    
    n_samples = len(df)
    n_features = len(player_to_col)
    
    # Extract player columns as numpy arrays for speed
    a_players = df[[f'a{k}' for k in range(1, 6)]].values.astype(int)
    h_players = df[[f'h{k}' for k in range(1, 6)]].values.astype(int)
    home_poss = df['home_poss'].values
    pts = df['pts'].values.astype(np.float64)
    score_diff = df['score_diff'].values.astype(np.float64)
    period = df['period'].values.astype(int)
    
    # Determine offense/defense based on home_poss
    # Shape: (n_samples, 5)
    offense_players = np.where(home_poss[:, None] == 1, h_players, a_players)
    defense_players = np.where(home_poss[:, None] == 1, a_players, h_players)
    
    # Build sparse matrix using COO format (faster for construction)
    row_indices = []
    col_indices = []
    data_values = []
    
    # Process offensive players
    for player_idx in range(5):
        player_ids = offense_players[:, player_idx]
        for i, pid in enumerate(player_ids):
            off_key = f"{pid}_off"
            if off_key in player_to_col:
                row_indices.append(i)
                col_indices.append(player_to_col[off_key])
                data_values.append(1.0)
    
    # Process defensive players
    for player_idx in range(5):
        player_ids = defense_players[:, player_idx]
        for i, pid in enumerate(player_ids):
            def_key = f"{pid}_def"
            if def_key in player_to_col:
                row_indices.append(i)
                col_indices.append(player_to_col[def_key])
                data_values.append(1.0)
    
    # Add score differential polynomial features
    for i in range(n_samples):
        sd = score_diff[i]
        p = period[i]
        
        for poly_degree in range(1, max_poly + 1):
            col_name = f"sd_poly{poly_degree}_period{p}"
            if col_name in player_to_col:
                row_indices.append(i)
                col_indices.append(player_to_col[col_name])
                data_values.append(sd ** poly_degree)
    
    # Create sparse matrix
    data_array = np.array(data_values, dtype=np.float64)
    X = coo_matrix((data_array, (row_indices, col_indices)), 
                   shape=(n_samples, n_features), dtype=np.float64)
    
    logging.info(f"Design matrix shape: {X.shape}")
    return X.tocsr(), pts

###############################################################################
# 4) ALTERNATING RIDGE REGRESSION WITH RUBBERBAND
###############################################################################
def alternating_rapm_with_rubberband(X, y, player_to_col, all_players,
                                      alpha_offense=3000, alpha_defense=3000, alpha_poly=50,
                                      max_iter=200, tol=1e-4):
    """
    Alternating minimization for RAPM with rubberband effect.
    Separates offense, defense, and score differential polynomial coefficients.
    Optimized with multi-core support.
    """
    logging.info(f"Running alternating Ridge regression with rubberband (using {N_CORES} cores)...")
    
    n_samples, n_features = X.shape
    beta = np.zeros(n_features, dtype=np.float64)
    
    # Separate offense, defense, and polynomial indices
    offense_indices = []
    defense_indices = []
    poly_indices = []
    
    for k, v in player_to_col.items():
        if k.endswith('_off'):
            offense_indices.append(v)
        elif k.endswith('_def'):
            defense_indices.append(v)
        elif k.startswith('sd_poly'):
            poly_indices.append(v)
    
    offense_indices = np.array(offense_indices, dtype=int)
    defense_indices = np.array(defense_indices, dtype=int)
    poly_indices = np.array(poly_indices, dtype=int)
    
    X_off = X[:, offense_indices]
    X_def = X[:, defense_indices]
    X_poly = X[:, poly_indices] if len(poly_indices) > 0 else None
    
    # Use solver='lsqr' for sparse matrices - faster than default 'auto'
    ridge_off = Ridge(alpha=alpha_offense, fit_intercept=False, solver='lsqr')
    ridge_def = Ridge(alpha=alpha_defense, fit_intercept=False, solver='lsqr')
    ridge_poly = Ridge(alpha=alpha_poly, fit_intercept=False, solver='lsqr')
    
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
        ridge_def.fit(X_def, residual)
        beta_def = ridge_def.coef_
        beta[defense_indices] = beta_def
        residual -= X_def @ beta_def
        
        # Update polynomial coefficients (rubberband effect)
        if X_poly is not None:
            residual += X_poly @ beta[poly_indices]
            ridge_poly.fit(X_poly, residual)
            beta_poly = ridge_poly.coef_
            beta[poly_indices] = beta_poly
            residual -= X_poly @ beta_poly
        
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
    import os
    if not os.path.exists(input_file):
        return None
    
    try:
        df = pd.read_csv(input_file)
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
# HELPER: SAVE RUBBERBAND ANALYSIS
###############################################################################
def save_rubberband_analysis(beta, player_to_col, output_prefix):
    """
    Extract and save rubberband effect analysis:
    1. Coefficients CSV - the learned polynomial weights
    2. Effects CSV - predicted impact at various score margins
    """
    import os
    
    # Extract polynomial coefficients
    poly_coeffs = []
    for feature_name, idx in player_to_col.items():
        if feature_name.startswith('sd_poly'):
            poly_coeffs.append({
                'feature': feature_name,
                'coefficient': beta[idx]
            })
    
    # Save coefficients
    coeffs_df = pd.DataFrame(poly_coeffs)
    coeffs_file = f"{output_prefix}_rubberband_coefficients.csv"
    coeffs_df.to_csv(coeffs_file, index=False)
    logging.info(f"Saved rubberband coefficients to {coeffs_file}")
    
    # Calculate effects at different score margins
    # For each period, calculate the total effect at margins: -20, -15, -10, -5, 0, +5, +10, +15, +20
    score_margins = [-20, -15, -10, -5, 0, 5, 10, 15, 20]
    effects = []
    
    for period in [1, 2, 3, 4]:
        for margin in score_margins:
            # Sum up all polynomial terms for this period and margin
            total_effect = 0.0
            for poly_degree in [1, 2, 3]:
                feature_name = f"sd_poly{poly_degree}_period{period}"
                if feature_name in player_to_col:
                    coeff = beta[player_to_col[feature_name]]
                    total_effect += coeff * (margin ** poly_degree)
            
            effects.append({
                'period': period,
                'score_margin': margin,
                'predicted_effect': total_effect,
                'predicted_effect_per100': total_effect * 100  # Scale to per 100 possessions
            })
    
    # Save effects
    effects_df = pd.DataFrame(effects)
    effects_file = f"{output_prefix}_rubberband_effects.csv"
    effects_df.to_csv(effects_file, index=False)
    logging.info(f"Saved rubberband effects to {effects_file}")
    
    # Log summary
    logging.info("\n" + "=" * 60)
    logging.info("RUBBERBAND EFFECT SUMMARY")
    logging.info("=" * 60)
    logging.info("\nPolynomial Coefficients:")
    for _, row in coeffs_df.iterrows():
        logging.info(f"  {row['feature']:20s}: {row['coefficient']:+.6f}")
    
    logging.info("\nSample Effects (Period 4):")
    period4_effects = effects_df[effects_df['period'] == 4]
    for _, row in period4_effects.iterrows():
        margin = int(row['score_margin'])
        effect = row['predicted_effect_per100']
        logging.info(f"  Margin {margin:+3d}: {effect:+.3f} pts/100")

###############################################################################
# 5) MAIN EXECUTION
###############################################################################
def run_simplified_rapm(input_files, name_map_file="autocomplete_map.csv", pure=False):
    """
    Main function to run simplified RAPM analysis.
    
    Args:
        input_files: Single file path (str) or list of file paths for multi-year analysis
        name_map_file: Path to player name mapping CSV
        pure: If True, use Net_Diff for RAPM files (removes 3pt/FT luck adjustments)
    """
    # Ensure input_files is a list
    if isinstance(input_files, str):
        input_files = [input_files]
    
    # Prepend 'Processed/' to files that don't have a directory path
    import os
    processed_files = []
    for f in input_files:
        if os.path.dirname(f) == '':
            # No directory in path, prepend Processed/
            processed_files.append(os.path.join('Processed', f))
        else:
            # Already has a directory path, use as-is
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
    
    # Player offense/defense features
    for p in all_players:
        player_to_col[f"{p}_off"] = idx
        idx += 1
        player_to_col[f"{p}_def"] = idx
        idx += 1
    
    # Score differential polynomial features (rubberband effect)
    # Add sd_poly1, sd_poly2, sd_poly3 for each period (1-4)
    for poly_degree in range(1, 4):  # 1, 2, 3
        for period in range(1, 5):  # 1, 2, 3, 4
            feature_name = f"sd_poly{poly_degree}_period{period}"
            player_to_col[feature_name] = idx
            idx += 1
    
    logging.info(f"Created {len(player_to_col)} features ({len(all_players)*2} player features + 12 rubberband features)")
    
    # Build design matrix with rubberband features
    X, y = build_design_matrix_with_rubberband(df_transformed, player_to_col, max_poly=3)
    
    # Center the outcome
    y_centered = y - np.mean(y)
    logging.info(f"Centered outcome, mean={np.mean(y):.3f}")
    
    # Run alternating Ridge regression with rubberband
    beta = alternating_rapm_with_rubberband(
        X, y_centered, player_to_col, all_players,
        alpha_offense=3000, alpha_defense=3000, alpha_poly=50,
        max_iter=200, tol=1e-4
    )
    
    # Calculate player possessions
    player_possessions = defaultdict(int)
    for _, row in df_transformed.iterrows():
        away_pl = [int(row[f'a{k}']) for k in range(1, 6)]
        home_pl = [int(row[f'h{k}']) for k in range(1, 6)]
        for p in away_pl + home_pl:
            player_possessions[str(p)] += 1
    
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
            'possessions': poss
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('net_rapm', ascending=False).reset_index(drop=True)
    
    # Generate dynamic output filename based on input
    import os
    
    # Ensure Results and Rubberband_Results directories exist
    os.makedirs("Results", exist_ok=True)
    os.makedirs("Rubberband_Results", exist_ok=True)
    
    # Extract metadata from input files for naming
    years = []
    file_type = None
    has_ps = False
    has_rs = False
    
    for f in input_files:
        base = os.path.splitext(os.path.basename(f))[0]
        
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
    
    # Build filename base
    if years:
        years_sorted = sorted(set(years))
        if len(years_sorted) == 1:
            year_range = years_sorted[0]
        else:
            year_range = f"{'_'.join(years_sorted)}"
        
        if pure:
            filename_base = f"{file_type}_{year_range}_{season_suffix}_pure"
        else:
            filename_base = f"{file_type}_{year_range}_{season_suffix}"
    else:
        # Fallback
        if pure:
            filename_base = f"{file_type}_{season_suffix}_pure"
        else:
            filename_base = f"{file_type}_{season_suffix}"
    
    # Save player results to both directories
    output_file = f"Results/{filename_base}_results.csv"
    results_df.to_csv(output_file, index=False)
    logging.info(f"Saved results to {output_file}")
    
    rubberband_output_file = f"Rubberband_Results/{filename_base}_results.csv"
    results_df.to_csv(rubberband_output_file, index=False)
    logging.info(f"Saved results to {rubberband_output_file}")
    
    # Save rubberband analysis (coefficients and effects)
    rubberband_prefix = f"Rubberband_Results/{filename_base}"
    save_rubberband_analysis(beta, player_to_col, rubberband_prefix)
    
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
    
    # Determine available cores for help text
    default_cores = max(1, mp.cpu_count() - 1)
    max_cores = mp.cpu_count()
    
    parser.add_argument('--cores', '-j',
                       type=int,
                       default=None,
                       help=f'Number of CPU cores to use (default: {default_cores}, max: {max_cores})')
    
    args = parser.parse_args()
    
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
            filename = f"{args.prefix}{year:02d}.csv"
            input_files.append(filename)
        
        if args.season_type in ['PS', 'ALL']:
            # Playoff file
            filename = f"{args.prefix}{year:02d}_PS.csv"
            input_files.append(filename)
    
    # Log what we're processing
    logging.info(f"Processing {args.prefix} data for years {args.start_year}-{args.end_year} ({args.season_type})")
    logging.info(f"Files to process: {', '.join(input_files)}")
    
    # Name map file is always the same
    name_map_file = "autocomplete_map.csv"
    
    # Run the analysis
    results = run_simplified_rapm(input_files, name_map_file, pure=args.pure)
    
    if results is not None:
        logging.info("\nDone!")
    else:
        logging.error("\nAnalysis failed - no data was processed.")
        sys.exit(1)

