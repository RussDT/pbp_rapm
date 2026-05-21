"""
Base RAPM utilities shared across all variants.

Provides:
- Data loading from parquet files
- Design matrix construction
- Player possession counting
- Common constants and grids
"""

import logging
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Paths
SCRIPT_DIR = Path(__file__).parent
PIPELINE_ROOT = SCRIPT_DIR.parent.parent
PROJECT_ROOT = PIPELINE_ROOT.parent
PROCESSED_DIR = PIPELINE_ROOT / "processed"
RESULTS_DIR = PIPELINE_ROOT / "results" / "rapm_variants"

# Ensure results dir exists
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

###############################################################################
# CONSTANTS - HYPERPARAMETER GRIDS
###############################################################################

# Fixed offense lambda (NEVER tuned)
LAMBDA_OFF_BASE = 3000

# Defense lambda grid (must include 4500)
LAMBDA_DEF_GRID = [1500, 2500, 3000, 4500, 6000, 8000, 10000]

# Possession-weighted ridge grids
P_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]  # p=0.0 means no possession scaling
M_MAX_GRID = [2.0, 3.0, 4.0]  # Max multiplier for low-poss players

# Time decay grid (daily decay rate)
DECAY_GRID = [0.997, 0.999, 0.9995]

# Fixed possession-weighted ridge parameters
N_REF = 2000   # Reference possessions
N0 = 200       # Smoothing constant
M_MIN = 0.7    # Minimum multiplier

# Fold definitions for forward-looking backtest
# Train on 3 years, evaluate on next 2 years
# Using years 14-25 (2013-14 through 2024-25)
BACKTEST_FOLDS = [
    {"train": [14, 15, 16], "eval": [17, 18]},
    {"train": [15, 16, 17], "eval": [18, 19]},
    {"train": [16, 17, 18], "eval": [19, 20]},
    {"train": [17, 18, 19], "eval": [20, 21]},
    {"train": [18, 19, 20], "eval": [21, 22]},
    {"train": [19, 20, 21], "eval": [22, 23]},
    {"train": [20, 21, 22], "eval": [23, 24]},
    {"train": [21, 22, 23], "eval": [24, 25]},
]


###############################################################################
# DATA LOADING
###############################################################################

def load_rapm_data(years: List[int], include_playoffs: bool = True) -> pd.DataFrame:
    """
    Load RAPM parquet files for specified years.

    Args:
        years: List of 2-digit years (e.g., [14, 15, 16])
        include_playoffs: Whether to include playoff data

    Returns:
        Combined DataFrame with all possessions
    """
    all_dfs = []

    for year in years:
        # Regular season
        rs_file = PROCESSED_DIR / f"RAPM{year:02d}.parquet"
        if rs_file.exists():
            df = pd.read_parquet(rs_file)
            df['is_playoff'] = False
            all_dfs.append(df)
            logging.debug(f"Loaded {len(df)} possessions from {rs_file.name}")
        else:
            logging.warning(f"File not found: {rs_file}")

        # Playoffs
        if include_playoffs:
            ps_file = PROCESSED_DIR / f"RAPM{year:02d}_PS.parquet"
            if ps_file.exists():
                df = pd.read_parquet(ps_file)
                df['is_playoff'] = True
                all_dfs.append(df)
                logging.debug(f"Loaded {len(df)} possessions from {ps_file.name}")

    if not all_dfs:
        raise ValueError(f"No data files found for years {years}")

    combined = pd.concat(all_dfs, ignore_index=True)
    logging.info(f"Loaded {len(combined)} total possessions for years {years}")

    return combined


def prepare_data(df: pd.DataFrame, use_pure: bool = True) -> pd.DataFrame:
    """
    Prepare data for RAPM analysis.

    Args:
        df: Raw DataFrame from load_rapm_data
        use_pure: If True, use Net_Diff (ignores luck adjustments)

    Returns:
        DataFrame with Off_Diff, Def_Diff prepared
    """
    df = df.copy()

    if use_pure and 'Net_Diff' in df.columns:
        # Pure RAPM: use Net_Diff directly
        df['Off_Diff'] = df['Net_Diff']
        df['Def_Diff'] = -df['Net_Diff']
    elif 'Off_Diff' not in df.columns or 'Def_Diff' not in df.columns:
        # Fallback to Net_Diff
        df['Off_Diff'] = df['Net_Diff']
        df['Def_Diff'] = -df['Net_Diff']

    # Parse game_date
    df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce')

    return df


###############################################################################
# DESIGN MATRIX CONSTRUCTION
###############################################################################

def get_all_players(df: pd.DataFrame) -> List[str]:
    """Extract sorted list of unique player IDs from DataFrame."""
    all_players = set()
    for col in ['O1', 'O2', 'O3', 'O4', 'O5', 'D1', 'D2', 'D3', 'D4', 'D5']:
        all_players.update(df[col].astype(int).astype(str).unique())
    return sorted(all_players)


def build_player_mapping(players: List[str]) -> Dict[str, int]:
    """
    Build mapping from player keys to column indices.

    Returns dict with keys like "12345_off" -> 0, "12345_def" -> 1, etc.
    """
    player_to_col = {}
    idx = 0
    for p in players:
        player_to_col[f"{p}_off"] = idx
        idx += 1
        player_to_col[f"{p}_def"] = idx
        idx += 1
    return player_to_col


def build_design_matrix(
    df: pd.DataFrame,
    player_to_col: Dict[str, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build sparse design matrix for RAPM.

    Model: y_i = sum(offense_betas) + sum(defense_betas) + epsilon

    Each row has +1 for 5 offense player columns and +1 for 5 defense player columns.

    Args:
        df: DataFrame with O1-O5, D1-D5, Off_Diff columns
        player_to_col: Mapping from player keys to column indices

    Returns:
        X (csr_matrix), y (ndarray)
    """
    n_samples = len(df)
    n_features = len(player_to_col)

    # Extract player columns
    offense_players = df[['O1', 'O2', 'O3', 'O4', 'O5']].values.astype(int)
    defense_players = df[['D1', 'D2', 'D3', 'D4', 'D5']].values.astype(int)
    y = df['Off_Diff'].values.astype(np.float64)

    # Build COO sparse matrix
    row_indices = []
    col_indices = []

    # Offense players
    for player_idx in range(5):
        for i, pid in enumerate(offense_players[:, player_idx]):
            off_key = f"{pid}_off"
            if off_key in player_to_col:
                row_indices.append(i)
                col_indices.append(player_to_col[off_key])

    # Defense players
    for player_idx in range(5):
        for i, pid in enumerate(defense_players[:, player_idx]):
            def_key = f"{pid}_def"
            if def_key in player_to_col:
                row_indices.append(i)
                col_indices.append(player_to_col[def_key])

    data = np.ones(len(row_indices), dtype=np.float64)
    X = coo_matrix(
        (data, (row_indices, col_indices)),
        shape=(n_samples, n_features),
        dtype=np.float64
    ).tocsr()

    return X, y


###############################################################################
# POSSESSION COUNTING
###############################################################################

def get_player_possessions(
    df: pd.DataFrame,
    players: List[str]
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    """
    Count possessions for each player.

    Returns:
        total_poss: Total possessions (offense + defense)
        off_poss: Offensive possessions only
        def_poss: Defensive possessions only
    """
    total_poss = defaultdict(int)
    off_poss = defaultdict(int)
    def_poss = defaultdict(int)

    # Count from O/D columns
    for _, row in df.iterrows():
        for col in ['O1', 'O2', 'O3', 'O4', 'O5']:
            pid = str(int(row[col]))
            total_poss[pid] += 1
            off_poss[pid] += 1

        for col in ['D1', 'D2', 'D3', 'D4', 'D5']:
            pid = str(int(row[col]))
            total_poss[pid] += 1
            def_poss[pid] += 1

    return dict(total_poss), dict(off_poss), dict(def_poss)


def get_player_possessions_fast(
    df: pd.DataFrame,
    players: List[str]
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    """
    Fast vectorized possession counting.

    Returns:
        total_poss, off_poss, def_poss dictionaries
    """
    # Initialize with zeros for all players
    off_poss = {p: 0 for p in players}
    def_poss = {p: 0 for p in players}

    # Count offense
    for col in ['O1', 'O2', 'O3', 'O4', 'O5']:
        counts = df[col].astype(int).astype(str).value_counts()
        for pid, count in counts.items():
            if pid in off_poss:
                off_poss[pid] += count

    # Count defense
    for col in ['D1', 'D2', 'D3', 'D4', 'D5']:
        counts = df[col].astype(int).astype(str).value_counts()
        for pid, count in counts.items():
            if pid in def_poss:
                def_poss[pid] += count

    # Total
    total_poss = {p: off_poss.get(p, 0) + def_poss.get(p, 0) for p in players}

    return total_poss, off_poss, def_poss


###############################################################################
# TIME DECAY UTILITIES
###############################################################################

def compute_time_decay_weights(
    df: pd.DataFrame,
    decay_base: float = 0.999,
    reference_date: Optional[pd.Timestamp] = None
) -> np.ndarray:
    """
    Compute per-row weights using exponential time decay.

    weight_i = decay_base ^ age_days

    Args:
        df: DataFrame with 'game_date' column
        decay_base: Daily decay rate (e.g., 0.999)
        reference_date: Date to measure age from (defaults to max date in data)

    Returns:
        Array of weights, shape (n_samples,)
    """
    dates = pd.to_datetime(df['game_date'], errors='coerce')

    if reference_date is None:
        reference_date = dates.max()

    age_days = (reference_date - dates).dt.total_seconds() / 86400.0

    # Handle missing dates
    age_days = age_days.fillna(0)

    weights = np.power(decay_base, age_days.values)
    weights = np.clip(weights, 0.0, 1.0)

    return weights.astype(np.float64)


###############################################################################
# COEFFICIENT EXTRACTION
###############################################################################

def extract_coefficients(
    beta: np.ndarray,
    players: List[str],
    player_to_col: Dict[str, int]
) -> pd.DataFrame:
    """
    Extract offense/defense coefficients from beta vector.

    Returns DataFrame with columns: player_id, off, def, net
    (Values are per-possession, NOT multiplied by 100 yet)
    """
    results = []
    for p in players:
        off_val = beta[player_to_col[f"{p}_off"]]
        def_val = beta[player_to_col[f"{p}_def"]]
        net_val = off_val - def_val

        results.append({
            'player_id': p,
            'off': off_val,
            'def': def_val,
            'net': net_val
        })

    return pd.DataFrame(results)


def recenter_coefficients(
    beta: np.ndarray,
    players: List[str],
    player_to_col: Dict[str, int],
    total_poss: Dict[str, int]
) -> np.ndarray:
    """
    Re-center offense/defense coefficients so possession-weighted means are zero.

    Returns modified beta (in-place modification)
    """
    # Compute possession-weighted means
    sum_off = 0.0
    sum_def = 0.0
    sum_poss = 0.0

    for p in players:
        poss = total_poss.get(p, 0)
        if poss > 0:
            sum_off += beta[player_to_col[f"{p}_off"]] * poss
            sum_def += beta[player_to_col[f"{p}_def"]] * poss
            sum_poss += poss

    if sum_poss > 0:
        off_offset = sum_off / sum_poss
        def_offset = sum_def / sum_poss

        for p in players:
            beta[player_to_col[f"{p}_off"]] -= off_offset
            beta[player_to_col[f"{p}_def"]] -= def_offset

    return beta


###############################################################################
# NAME MAPPING
###############################################################################

def load_name_map() -> Dict[str, str]:
    """Load player ID to name mapping."""
    name_map_file = PROJECT_ROOT / "autocomplete_map.csv"

    try:
        name_df = pd.read_csv(name_map_file)
        return dict(zip(name_df['nba_id'].astype(str), name_df['player_name']))
    except Exception as e:
        logging.warning(f"Could not load name map: {e}")
        return {}


###############################################################################
# RESULTS FORMATTING
###############################################################################

def format_results(
    coef_df: pd.DataFrame,
    total_poss: Dict[str, int],
    off_poss: Dict[str, int],
    def_poss: Dict[str, int],
    name_map: Dict[str, str]
) -> pd.DataFrame:
    """
    Format coefficients into final results DataFrame.

    Multiplies by 100 for per-100 possessions scale.
    """
    results = []

    for _, row in coef_df.iterrows():
        pid = row['player_id']
        results.append({
            'player_id': pid,
            'player_name': name_map.get(pid, f"ID_{pid}"),
            'off': round(row['off'] * 100, 2),
            'def': round(row['def'] * 100, 2),
            'net_rapm': round(row['net'] * 100, 2),
            'possessions': total_poss.get(pid, 0),
            'off_poss': off_poss.get(pid, 0),
            'def_poss': def_poss.get(pid, 0)
        })

    df = pd.DataFrame(results)
    df = df.sort_values('net_rapm', ascending=False).reset_index(drop=True)

    return df
