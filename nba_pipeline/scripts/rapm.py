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
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold, KFold
from collections import defaultdict
import multiprocessing as mp
from functools import partial
from pathlib import Path
import os
import shutil
import subprocess

from dotenv import load_dotenv
from supabase import create_client, Client

# Get the pipeline root directory (parent of scripts/)
SCRIPT_DIR = Path(__file__).parent
PIPELINE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = PIPELINE_ROOT.parent
PROCESSED_DIR = PIPELINE_ROOT / "processed"
RESULTS_DIR = PIPELINE_ROOT / "results"
MASTER_RESULTS_DIR = PIPELINE_ROOT / "master_results"
RAPMS_DIR = PROJECT_ROOT.parent / "rapms"
RAPMS_MASTER_RESULTS_DIR = RAPMS_DIR / "master_results"
DEFAULT_DARKO_HISTORY_PATH = PROJECT_ROOT.parent / "2026_NBA_PIPELINE" / "databallr" / "darko" / "dpm_history.csv"

load_dotenv(PROJECT_ROOT / ".env")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Determine number of cores to use (cap at 8 to avoid overloading the machine)
N_CORES = min(8, max(1, mp.cpu_count() - 1))
AGE_CURVE_PRIOR_K = 1000.0
AGE_DUMMY_MIN = 18
AGE_DUMMY_MAX = 38
AGE_DUMMY_REFERENCE = 27
STRENGTH_BUCKETS = ("not_strong", "strong")
ALT_BASELINE_PREFIXES = {
    'ALT_TS',
    'ALT_EFG',
    'ALT_SQ',
    'ALT_MAKE',
    'ALT_FT',
    'ALT_FT_FREQ',
    'ALT_FT_SEVERITY',
    'ALT_EFG_VALUE',
    'ALT_EFG_BASELINE',
    'ALT_EFG_RIM',
    'ALT_EFG_MID',
    'ALT_EFG_THREE',
    'ALT_EFG_RIM_FREQ',
    'ALT_EFG_RIM_FG',
    'ALT_EFG_MID_FREQ',
    'ALT_EFG_MID_FG',
    'ALT_EFG_THREE_FREQ',
    'ALT_EFG_THREE_FG',
    'ALT_TOV_VALUE',
    'ALT_BADPASS_TOV_VALUE',
    'ALT_SCORING_TOV_VALUE',
    'ALT_FC_COMPLETION',
}
ALT_FIRST_CHANCE_TOV_PREFIXES = {'ALT_TOV', 'ALT_BADPASS_TOV', 'ALT_SCORING_TOV'}
ALT_PREFIXES = ALT_BASELINE_PREFIXES | ALT_FIRST_CHANCE_TOV_PREFIXES
FIRST_CHANCE_MODE_PREFIXES = {
    'FC_TRANSITION_SCORING',
    'FC_HALFCOURT_SCORING',
    'FC_MODE_MIX',
    'FC_TRANSITION_VALUE',
    'FC_HALFCOURT_VALUE',
}
FIRST_CHANCE_ALIAS_PREFIXES = ALT_PREFIXES | FIRST_CHANCE_MODE_PREFIXES
ALT_COMPONENT_COLUMNS = {
    'ALT_TS': 'Net_Diff',
    'ALT_EFG': 'FC_EFG_Diff',
    'ALT_SQ': 'FC_SQ_Diff',
    'ALT_MAKE': 'FC_MAKE_Diff',
    'ALT_FT': 'FC_FT_Diff',
    'ALT_FT_FREQ': 'FC_FT_FREQ_Diff',
    'ALT_FT_SEVERITY': 'FC_FT_SEVERITY_Diff',
    'ALT_EFG_VALUE': 'FC_EFG_VALUE_Diff',
    'ALT_EFG_BASELINE': 'FC_EFG_BASELINE_Diff',
    'ALT_EFG_RIM': 'FC_RIM_EFG_AA_Diff',
    'ALT_EFG_MID': 'FC_MID_EFG_AA_Diff',
    'ALT_EFG_THREE': 'FC_THREE_EFG_AA_Diff',
    'ALT_EFG_RIM_FREQ': 'FC_RIM_EFG_FREQ_Diff',
    'ALT_EFG_RIM_FG': 'FC_RIM_EFG_FG_Diff',
    'ALT_EFG_MID_FREQ': 'FC_MID_EFG_FREQ_Diff',
    'ALT_EFG_MID_FG': 'FC_MID_EFG_FG_Diff',
    'ALT_EFG_THREE_FREQ': 'FC_THREE_EFG_FREQ_Diff',
    'ALT_EFG_THREE_FG': 'FC_THREE_EFG_FG_Diff',
}
DEFAULT_ALPHA_GRID = [300, 500, 1000, 2000, 3000, 5000, 7000, 10000]
AGE_CURVE_TARGETS = {
    'off': {
        'poss_col': 'OffPoss',
        'target_cols': ['five_year_orapm', 'four_year_orapm', 'three_year_orapm', 'two_year_orapm', 'off_rapm'],
        'horizon_weights': [5.0, 4.0, 3.0, 2.0, 1.0],
        'label': 'Offense',
    },
    'def': {
        'poss_col': 'DefPoss',
        'target_cols': ['five_year_drapm', 'four_year_drapm', 'three_year_drapm', 'two_year_drapm', 'def_rapm'],
        'horizon_weights': [5.0, 4.0, 3.0, 2.0, 1.0],
        'label': 'Defense',
    },
}

###############################################################################
# 0) TIME DECAY UTILITIES
###############################################################################
def half_life_to_decay_base(half_life_days: float) -> float:
    """Convert half-life in days to decay_base."""
    return 0.5 ** (1.0 / half_life_days)


def normalize_season_end_year(year_value: int) -> int:
    """Convert two-digit season end year to four digits."""
    year_value = int(year_value)
    if year_value < 100:
        return 2000 + year_value if year_value <= 50 else 1900 + year_value
    return year_value


def expand_two_digit_year_window(start_year: int, end_year: int) -> list[int]:
    """Expand a two-digit season-end window, allowing 97-00 century rollover."""
    start_year = int(start_year)
    end_year = int(end_year)
    start_full = normalize_season_end_year(start_year)
    end_full = normalize_season_end_year(end_year)
    if end_full < start_full:
        end_full += 100
    return [full_year % 100 for full_year in range(start_full, end_full + 1)]


def build_output_suffix(
    season_suffix: str,
    pure: bool = False,
    rubberband: bool = False,
    season_effects: bool = False,
    fixed_season_effects: bool = False,
    age_dummies: bool = False,
    age_poly_degree: int | None = None,
    alpha_cv: bool = False,
    off_alpha: float = 3000,
    def_alpha: float = 3000,
    age_curve: bool = False,
    off_prior=None,
    def_prior=None,
    timedecay: bool = False,
    half_life=None,
    strength_splits: bool = False,
    strength_legacy_scale: bool = False,
    strength_sensitivity: bool = False,
) -> str:
    """Build a result suffix that stays aligned across output artifacts."""
    parts = [season_suffix]
    if pure:
        parts.append("pure")
    if rubberband:
        parts.append("rb")
    if season_effects:
        parts.append("se")
    if fixed_season_effects:
        parts.append("fse")
    if age_dummies:
        parts.append("agefe")
    if age_poly_degree is not None:
        parts.append(f"agepoly{int(age_poly_degree)}")
    if alpha_cv:
        parts.append("cv")
    if off_alpha != 3000 or def_alpha != 3000:
        parts.append(f"a{int(off_alpha)}_{int(def_alpha)}")
    if age_curve:
        parts.append("age")
    if off_prior is not None and def_prior is not None and not age_curve:
        parts.append("offdefprior")
    elif off_prior is not None and not age_curve:
        parts.append("offprior")
    elif def_prior is not None and not age_curve:
        parts.append("prior")
    if timedecay:
        td_half = half_life if half_life else 700
        parts.append(f"td{int(td_half)}")
    if strength_splits:
        parts.append("strength")
    if strength_legacy_scale:
        parts.append("scaled")
    if strength_sensitivity:
        parts.append("strengthsens")
    return "_".join(parts)


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


def compute_time_decay_weights_from_dates(dates, decay_base: float = 0.999) -> np.ndarray:
    """Compute time-decay weights directly from a date-like series."""
    parsed_dates = pd.to_datetime(pd.Series(dates), errors='coerce')
    reference_date = parsed_dates.max()
    age_days = (reference_date - parsed_dates).dt.total_seconds() / 86400.0
    age_days = age_days.fillna(0)
    weights = np.power(decay_base, age_days.values)
    weights = np.clip(weights, 0.0, 1.0)
    return weights.astype(np.float64)


def compute_alt_first_chance_baselines(
    df: pd.DataFrame,
    timedecay: bool = False,
    half_life=None,
    requested_prefixes: list[str] | None = None,
) -> dict[str, float]:
    """Compute same-sample first-chance non-turnover baselines for the requested ALT scoring aliases."""
    if 'Net_Diff' not in df.columns or 'Is_Turnover' not in df.columns:
        raise ValueError("ALT aliases require FIRST_CHANCE rows with Net_Diff and Is_Turnover.")

    non_tov_mask = pd.to_numeric(df['Is_Turnover'], errors='coerce').fillna(0).astype(int) == 0
    if not non_tov_mask.any():
        raise ValueError("Could not compute ALT baseline: no non-turnover FIRST_CHANCE rows found.")

    if timedecay:
        decay_half_life = half_life if half_life else 700
        decay_base = half_life_to_decay_base(decay_half_life)
        date_col = 'game_date' if 'game_date' in df.columns else 'date'
        weights = compute_time_decay_weights_from_dates(df.loc[non_tov_mask, date_col], decay_base=decay_base)
    else:
        decay_half_life = None
        weights = None

    prefixes_to_compute = requested_prefixes if requested_prefixes is not None else list(ALT_COMPONENT_COLUMNS.keys())
    component_prefixes_to_compute: list[str] = []
    for prefix in prefixes_to_compute:
        if prefix in {'ALT_TOV_VALUE', 'ALT_BADPASS_TOV_VALUE', 'ALT_SCORING_TOV_VALUE'}:
            component_prefixes_to_compute.append('ALT_TS')
        elif prefix == 'ALT_FC_COMPLETION':
            component_prefixes_to_compute.extend(['ALT_EFG_VALUE', 'ALT_FT'])
        else:
            component_prefixes_to_compute.append(prefix)
    component_prefixes_to_compute = list(dict.fromkeys(component_prefixes_to_compute))

    baselines: dict[str, float] = {}
    for prefix in component_prefixes_to_compute:
        if prefix not in ALT_COMPONENT_COLUMNS:
            raise ValueError(f"Unsupported ALT baseline prefix: {prefix}")
        column = ALT_COMPONENT_COLUMNS[prefix]
        if column not in df.columns:
            if prefix == 'ALT_TS':
                raise ValueError("ALT_TS requires FIRST_CHANCE rows with Net_Diff.")
            continue

        non_tov_values = pd.to_numeric(df.loc[non_tov_mask, column], errors='coerce')
        if non_tov_values.isna().any():
            missing_count = int(non_tov_values.isna().sum())
            raise ValueError(
                f"{prefix} requires {column} on every non-turnover FIRST_CHANCE row; "
                f"found {missing_count} missing values in the loaded sample."
            )
        values = non_tov_values.fillna(0.0).values
        if weights is not None:
            baselines[prefix] = float(np.average(values, weights=weights))
        else:
            baselines[prefix] = float(values.mean())

    if {'ALT_TS', 'ALT_EFG'}.issubset(baselines) and 'ALT_FT' in prefixes_to_compute:
        baselines['ALT_FT'] = baselines['ALT_TS'] - baselines['ALT_EFG']
    if {'ALT_TS', 'ALT_SQ', 'ALT_MAKE'}.issubset(baselines) and 'ALT_FT' in prefixes_to_compute:
        baselines['ALT_FT'] = baselines['ALT_TS'] - baselines['ALT_SQ'] - baselines['ALT_MAKE']
    if 'ALT_TOV_VALUE' in prefixes_to_compute and 'ALT_TS' in baselines:
        baselines['ALT_TOV_VALUE'] = baselines['ALT_TS']
    if 'ALT_BADPASS_TOV_VALUE' in prefixes_to_compute and 'ALT_TS' in baselines:
        baselines['ALT_BADPASS_TOV_VALUE'] = baselines['ALT_TS']
    if 'ALT_SCORING_TOV_VALUE' in prefixes_to_compute and 'ALT_TS' in baselines:
        baselines['ALT_SCORING_TOV_VALUE'] = baselines['ALT_TS']
    if (
        'ALT_FC_COMPLETION' in prefixes_to_compute and
        {'ALT_EFG_VALUE', 'ALT_FT'}.issubset(baselines)
    ):
        baselines['ALT_FC_COMPLETION'] = baselines['ALT_EFG_VALUE'] + baselines['ALT_FT']

    if timedecay:
        logging.info(
            "Computed ALT first-chance baselines %s for %s with time decay (half-life=%s)",
            {key: round(value, 6) for key, value in baselines.items()},
            prefixes_to_compute,
            decay_half_life,
        )
    else:
        logging.info(
            "Computed ALT first-chance baselines %s for %s on the exact loaded sample",
            {key: round(value, 6) for key, value in baselines.items()},
            prefixes_to_compute,
        )

    return baselines


def compute_first_chance_mode_baselines(
    df: pd.DataFrame,
    timedecay: bool = False,
    half_life=None,
) -> dict[str, float]:
    """Compute same-sample first-chance transition/half-court PPP baselines."""
    required = {'Net_Diff', 'Is_FC_Transition_Possession'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "First-chance mode aliases require FIRST_CHANCE rows with "
            f"{sorted(missing)}."
        )

    y = pd.to_numeric(df['Net_Diff'], errors='coerce')
    if y.isna().any():
        raise ValueError("First-chance mode aliases found missing Net_Diff values.")

    is_transition = (
        pd.to_numeric(df['Is_FC_Transition_Possession'], errors='coerce')
        .fillna(0)
        .astype(int)
        .eq(1)
    )
    if not is_transition.any():
        raise ValueError("Could not compute first-chance transition baseline: no transition rows found.")
    if is_transition.all():
        raise ValueError("Could not compute first-chance half-court baseline: no non-transition rows found.")

    if timedecay:
        decay_half_life = half_life if half_life else 700
        decay_base = half_life_to_decay_base(decay_half_life)
        date_col = 'game_date' if 'game_date' in df.columns else 'date'
        weights = compute_time_decay_weights_from_dates(df[date_col], decay_base=decay_base)
    else:
        decay_half_life = None
        weights = None

    values = y.values
    transition_mask = is_transition.values
    if weights is None:
        transition_ppp = float(values[transition_mask].mean())
        halfcourt_ppp = float(values[~transition_mask].mean())
    else:
        transition_ppp = float(np.average(values[transition_mask], weights=weights[transition_mask]))
        halfcourt_ppp = float(np.average(values[~transition_mask], weights=weights[~transition_mask]))

    baselines = {
        'transition_ppp': transition_ppp,
        'halfcourt_ppp': halfcourt_ppp,
    }
    if timedecay:
        logging.info(
            "Computed first-chance mode baselines %s with time decay (half-life=%s)",
            {key: round(value, 6) for key, value in baselines.items()},
            decay_half_life,
        )
    else:
        logging.info(
            "Computed first-chance mode baselines %s on the exact loaded sample",
            {key: round(value, 6) for key, value in baselines.items()},
        )
    return baselines


def parse_alpha_grid(alpha_grid):
    """Normalize an alpha grid into a sorted list of positive floats."""
    if alpha_grid is None:
        return [float(alpha) for alpha in DEFAULT_ALPHA_GRID]
    if isinstance(alpha_grid, str):
        values = [part.strip() for part in alpha_grid.split(',') if part.strip()]
    else:
        values = list(alpha_grid)

    parsed = []
    for value in values:
        alpha = float(value)
        if alpha <= 0:
            raise ValueError(f"Alpha values must be positive; received {value}")
        parsed.append(alpha)

    if not parsed:
        raise ValueError("Alpha grid must contain at least one positive value.")

    return sorted(set(parsed))


def weighted_mse(y_true, y_pred, weights=None) -> float:
    """Compute mean squared error with optional weights."""
    residual = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
    if weights is None:
        return float(np.mean(residual ** 2))

    weights = np.asarray(weights, dtype=np.float64)
    return float(np.average(residual ** 2, weights=weights))


def build_cv_splits(groups, n_splits):
    """Create train/validation index pairs, preferring grouped folds by game."""
    groups_series = pd.Series(groups)
    n_samples = len(groups_series)
    unique_groups = groups_series.nunique(dropna=True)

    if n_samples < 2:
        raise ValueError("Need at least 2 samples to run alpha cross-validation.")

    actual_splits = max(2, min(int(n_splits), n_samples))
    if unique_groups >= actual_splits:
        splitter = GroupKFold(n_splits=actual_splits)
        split_iter = splitter.split(np.zeros(n_samples), groups=groups_series.values)
        split_type = "group"
    else:
        splitter = KFold(n_splits=actual_splits, shuffle=False)
        split_iter = splitter.split(np.zeros(n_samples))
        split_type = "row"

    splits = [(np.asarray(train_idx), np.asarray(val_idx)) for train_idx, val_idx in split_iter]
    logging.info(
        "Alpha CV using %d %s-based folds (%d unique groups, %d rows)",
        len(splits),
        split_type,
        int(unique_groups),
        n_samples,
    )
    return splits, split_type


def publish_result_to_rapms(output_file: Path, publish_name: str = None) -> tuple[Path, Path]:
    """
    Copy a result CSV into local master_results and the downstream rapms repo, then push.

    Returns:
        (local_master_path, rapms_master_path)
    """
    source_path = Path(output_file)
    if not source_path.exists():
        raise FileNotFoundError(f"Cannot publish missing output file: {source_path}")

    publish_basename = publish_name if publish_name else source_path.name
    local_master_path = MASTER_RESULTS_DIR / publish_basename
    rapms_master_path = RAPMS_MASTER_RESULTS_DIR / publish_basename

    MASTER_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RAPMS_MASTER_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logging.info("Publishing %s to %s and %s", source_path.name, local_master_path, rapms_master_path)
    shutil.copy2(source_path, local_master_path)
    shutil.copy2(source_path, rapms_master_path)

    subprocess.run(
        ['git', '-C', str(RAPMS_DIR), 'add', str(rapms_master_path.relative_to(RAPMS_DIR))],
        check=True,
        capture_output=True,
        text=True,
    )

    diff_result = subprocess.run(
        ['git', '-C', str(RAPMS_DIR), 'diff', '--cached', '--quiet'],
        capture_output=True,
        text=True,
    )
    if diff_result.returncode == 0:
        logging.info("No downstream rapms changes detected after publish; skipping commit/push")
        return local_master_path, rapms_master_path
    if diff_result.returncode != 1:
        raise RuntimeError(f"git diff --cached --quiet failed: {diff_result.stderr}")

    commit_message = f"Publish {publish_basename}"
    subprocess.run(
        ['git', '-C', str(RAPMS_DIR), 'commit', '-m', commit_message],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ['git', '-C', str(RAPMS_DIR), 'push', 'origin', 'main'],
        check=True,
        capture_output=True,
        text=True,
    )
    logging.info("Published %s to rapms and pushed origin/main", publish_basename)
    return local_master_path, rapms_master_path


def normalize_season_series(values) -> pd.Series:
    """Normalize raw season values to 4-digit season end years."""
    numeric = pd.to_numeric(pd.Series(values), errors='coerce')
    normalized = numeric.map(lambda x: normalize_season_end_year(int(x)) if pd.notna(x) else pd.NA)
    return normalized.astype('Int64')


def parse_processed_season_end_year(path) -> int | None:
    """Parse a 4-digit season end year from a processed parquet filename."""
    stem = Path(path).stem.replace("_PS", "")
    digits = "".join(ch for ch in stem if ch.isdigit())
    if not digits:
        return None
    yy = int(digits[-2:])
    return 2000 + yy if yy <= 50 else 1900 + yy


###############################################################################
# 0b) AGE CURVE UTILITIES
###############################################################################
def fetch_age_source_darko(max_year: int, darko_history_path) -> pd.DataFrame:
    """
    Load player-season ages from DARKO history.

    This is an age lookup only. It does not use DARKO DPM values as RAPM targets.
    """
    path = Path(darko_history_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing DARKO history file for age source: {path}")

    logging.info("Loading age source from DARKO history through season end year %s: %s", max_year, path)
    df = pd.read_csv(path, usecols=["nba_id", "season", "date", "age"])
    df["nba_id"] = pd.to_numeric(df["nba_id"], errors="coerce")
    df["year"] = pd.to_numeric(df["season"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["Age"] = pd.to_numeric(df["age"], errors="coerce")

    df = df.dropna(subset=["nba_id", "year", "Age"]).copy()
    df["nba_id"] = df["nba_id"].astype(int)
    df["year"] = df["year"].astype(int)
    df = df[df["year"] <= int(max_year)].copy()
    if df.empty:
        raise ValueError(f"No DARKO age rows found through season end year {max_year}.")

    df = df.sort_values(["nba_id", "year", "date"]).drop_duplicates(["nba_id", "year"], keep="last")
    df = df[["nba_id", "year", "Age"]].copy()
    df["year"] = df["year"].astype("Int64")
    df["nba_id"] = df["nba_id"].astype("Int64")
    logging.info(
        "Loaded %d DARKO age rows covering seasons %d-%d",
        len(df),
        int(df["year"].min()),
        int(df["year"].max()),
    )
    return df


def build_age_lookup_from_source(age_source_df: pd.DataFrame) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Build sorted player -> (years, ages) lookup arrays for fast fallback age access."""
    lookup = {}
    for player_id, group in age_source_df.groupby("nba_id"):
        ordered = group.sort_values("year")
        years = ordered["year"].astype(int).to_numpy()
        ages = ordered["Age"].astype(float).to_numpy()
        lookup[int(player_id)] = (years, ages)
    return lookup


def lookup_age_from_arrays(
    age_lookup: dict[int, tuple[np.ndarray, np.ndarray]],
    player_id: int,
    season_end_year: int,
) -> float | None:
    """Return exact player-season age or latest available age at/before the season."""
    entry = age_lookup.get(int(player_id))
    if entry is None:
        return None
    years, ages = entry
    idx = int(np.searchsorted(years, int(season_end_year), side="right") - 1)
    if idx < 0:
        return None
    return float(ages[idx])


def load_age_poly_curves(age_poly_coefficients, prefix: str) -> tuple[dict[str, np.poly1d], int]:
    """Load offense/defense fixed age-polynomial curves for one metric prefix."""
    path = Path(age_poly_coefficients)
    coeffs = pd.read_csv(path)
    required = {"metric", "side", "degree"}
    missing = required - set(coeffs.columns)
    if missing:
        raise ValueError(f"Age polynomial coefficient file missing columns: {sorted(missing)}")

    metric = str(prefix).lower()
    curves: dict[str, np.poly1d] = {}
    degrees = set()
    for _, row in coeffs.iterrows():
        row_metric = str(row["metric"]).lower()
        side = str(row["side"]).lower()
        if row_metric != metric or side not in {"off", "def"}:
            continue
        degree = int(row["degree"])
        degrees.add(degree)
        values = [float(row[f"coef_age_pow_{power}"]) for power in range(degree, -1, -1)]
        curves[side] = np.poly1d(values)

    missing_sides = {"off", "def"} - set(curves)
    if missing_sides:
        raise ValueError(f"Age polynomial coefficient file missing {prefix} sides: {sorted(missing_sides)}")
    if len(degrees) != 1:
        raise ValueError(f"Expected one polynomial degree for {prefix}, found {sorted(degrees)}")
    return curves, degrees.pop()


def compute_fixed_age_poly_offsets(
    df: pd.DataFrame,
    curves: dict[str, np.poly1d],
    age_lookup: dict[int, tuple[np.ndarray, np.ndarray]],
    *,
    reference_age: float = AGE_DUMMY_REFERENCE,
    min_age: float = AGE_DUMMY_MIN,
    max_age: float = AGE_DUMMY_MAX,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Compute known row offsets from fitted age curves.

    Curves are stored in per-100 units. For a player at age a, the fixed
    row contribution is curve(a) - curve(reference_age). The solver subtracts
    this known contribution from y, estimating reference-age player effects.
    """
    n_rows = len(df)
    offsets_per_100 = np.zeros(n_rows, dtype=np.float64)
    season_values = pd.to_numeric(df["season"], errors="coerce").fillna(0).astype(int).to_numpy()
    home_poss = df["home_poss"].to_numpy(dtype=int)
    h_players = df[[f"h{k}" for k in range(1, 6)]].to_numpy(dtype=int)
    a_players = df[[f"a{k}" for k in range(1, 6)]].to_numpy(dtype=int)
    offense_players = np.where(home_poss[:, None] == 1, h_players, a_players)
    defense_players = np.where(home_poss[:, None] == 1, a_players, h_players)

    mapped_slots = 0
    missing_slots = 0
    clipped_slots = 0
    cache: dict[tuple[str, int, int], float | None] = {}

    def cached_delta(side: str, player_id: int, season_end_year: int) -> float | None:
        nonlocal clipped_slots
        key = (side, int(player_id), int(season_end_year))
        if key in cache:
            return cache[key]
        age = lookup_age_from_arrays(age_lookup, int(player_id), int(season_end_year))
        if age is None:
            cache[key] = None
            return None
        clipped_age = min(max(float(age), float(min_age)), float(max_age))
        value = float(curves[side](clipped_age) - curves[side](float(reference_age)))
        cache[key] = value
        return value

    for side, players in [("off", offense_players), ("def", defense_players)]:
        for slot_idx in range(players.shape[1]):
            player_ids = players[:, slot_idx]
            valid_mask = (player_ids > 0) & (season_values > 0)
            valid_indices = np.where(valid_mask)[0]
            if len(valid_indices) == 0:
                continue
            pairs = np.column_stack((player_ids[valid_indices], season_values[valid_indices]))
            unique_pairs, inverse = np.unique(pairs, axis=0, return_inverse=True)
            pair_values = np.zeros(len(unique_pairs), dtype=np.float64)
            pair_mapped = np.zeros(len(unique_pairs), dtype=bool)
            for i, (player_id, season_end_year) in enumerate(unique_pairs):
                age = lookup_age_from_arrays(age_lookup, int(player_id), int(season_end_year))
                if age is None:
                    continue
                clipped_age = min(max(float(age), float(min_age)), float(max_age))
                if clipped_age != float(age):
                    clipped_slots += 1
                pair_values[i] = float(curves[side](clipped_age) - curves[side](float(reference_age)))
                pair_mapped[i] = True
                cache[(side, int(player_id), int(season_end_year))] = pair_values[i]
            offsets_per_100[valid_indices] += pair_values[inverse]
            mapped_slots += int(pair_mapped[inverse].sum())
            missing_slots += int((~pair_mapped[inverse]).sum())

    offsets = offsets_per_100 / 100.0
    summary = {
        "rows": float(n_rows),
        "mapped_slots": float(mapped_slots),
        "missing_slots": float(missing_slots),
        "clipped_slots": float(clipped_slots),
        "mean_offset_per_100": float(offsets_per_100.mean()) if n_rows else 0.0,
        "min_offset_per_100": float(offsets_per_100.min()) if n_rows else 0.0,
        "max_offset_per_100": float(offsets_per_100.max()) if n_rows else 0.0,
        "reference_age": float(reference_age),
    }
    return offsets, summary


def fetch_age_curve_source_supabase(max_year: int) -> pd.DataFrame:
    """
    Load regular-season player age + rolling RAPM history from player_stats_with_metrics.
    """
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")

    if not url or not key:
        raise ValueError("SUPABASE_URL or SUPABASE_KEY missing; age-curve mode requires Supabase access.")

    supabase: Client = create_client(url, key)
    select_cols = ', '.join([
        'nba_id',
        'year',
        'playoffs',
        '"Age"',
        '"OffPoss"',
        '"DefPoss"',
        'five_year_orapm',
        'four_year_orapm',
        'three_year_orapm',
        'two_year_orapm',
        'off_rapm',
        'five_year_drapm',
        'four_year_drapm',
        'three_year_drapm',
        'two_year_drapm',
        'def_rapm',
    ])

    logging.info(f"Loading RAPM age-curve source rows through season end year {max_year}...")
    page_size = 1000
    start = 0
    all_rows = []

    while True:
        response = (
            supabase.table("player_stats_with_metrics")
            .select(select_cols)
            .eq("playoffs", 0)
            .lte("year", max_year)
            .order("year")
            .order("nba_id")
            .range(start, start + page_size - 1)
            .execute()
        )
        batch = response.data or []
        if not batch:
            break
        all_rows.extend(batch)
        if len(batch) < page_size:
            break
        start += page_size

    if not all_rows:
        raise ValueError("No player_stats_with_metrics rows found for age-curve mode.")

    df = pd.DataFrame(all_rows)
    numeric_cols = ['Age', 'OffPoss', 'DefPoss']
    for side_config in AGE_CURVE_TARGETS.values():
        numeric_cols.extend(side_config['target_cols'])

    for col in set(numeric_cols):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
    df['nba_id'] = pd.to_numeric(df['nba_id'], errors='coerce').astype('Int64')
    logging.info(f"Loaded {len(df)} regular-season rows for RAPM age-curve priors")
    return df


def build_age_curve(age_source_df: pd.DataFrame, side: str) -> dict:
    """
    Build a smoothed empirical-Bayes age curve for offense or defense.
    """
    config = AGE_CURVE_TARGETS[side]
    poss_col = config['poss_col']
    target_cols = config['target_cols']
    horizon_weights = config['horizon_weights']

    working = age_source_df[['Age', poss_col] + target_cols].copy()
    target = working[target_cols].bfill(axis=1).iloc[:, 0]
    horizon = np.select([working[col].notna() for col in target_cols], horizon_weights, default=np.nan)
    poss = pd.to_numeric(working[poss_col], errors='coerce').fillna(0.0).clip(lower=0.0)
    ages = pd.to_numeric(working['Age'], errors='coerce')
    weights = np.sqrt(poss) * horizon

    valid = target.notna() & ages.notna() & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        raise ValueError(f"No valid rows available to build the {side} age curve.")

    train = pd.DataFrame({
        'age': ages[valid].astype(float),
        'age_floor': np.floor(ages[valid]).astype(int),
        'target': target[valid].astype(float),
        'weight': weights[valid].astype(float),
    })
    train['weighted_target'] = train['target'] * train['weight']

    global_mean = float(np.average(train['target'], weights=train['weight']))
    grouped = train.groupby('age_floor', as_index=True).agg(
        total_weight=('weight', 'sum'),
        weighted_target=('weighted_target', 'sum'),
        rows=('target', 'size'),
    )
    grouped['age_mean'] = grouped['weighted_target'] / grouped['total_weight']
    grouped['shrunk_mean'] = (
        (grouped['total_weight'] * grouped['age_mean']) + (AGE_CURVE_PRIOR_K * global_mean)
    ) / (grouped['total_weight'] + AGE_CURVE_PRIOR_K)

    curve = grouped['shrunk_mean'].reindex(range(int(grouped.index.min()), int(grouped.index.max()) + 1))
    curve = curve.interpolate(limit_direction='both')

    smoothed = curve.copy()
    for age in curve.index:
        values = []
        local_weights = []
        for delta, neighbor_weight in [(-1, 1.0), (0, 2.0), (1, 1.0)]:
            neighbor_age = age + delta
            if neighbor_age in curve.index and pd.notna(curve.loc[neighbor_age]):
                values.append(float(curve.loc[neighbor_age]))
                local_weights.append(neighbor_weight)
        smoothed.loc[age] = np.average(values, weights=local_weights)

    table = " ".join([f"{int(age)}:{value:+.2f}" for age, value in smoothed.items()])
    logging.info(f"{config['label']} age curve (pts/100): {table}")

    return {
        'side': side,
        'curve': smoothed.astype(float),
        'global_mean': global_mean,
        'min_age': float(smoothed.index.min()),
        'max_age': float(smoothed.index.max()),
    }


def evaluate_age_curve(curve_info: dict, age_value: float):
    """Evaluate a smoothed age curve with clipping at the observed endpoints."""
    if age_value is None or pd.isna(age_value):
        return None

    curve = curve_info['curve']
    clipped_age = min(max(float(age_value), curve_info['min_age']), curve_info['max_age'])
    lower_age = int(np.floor(clipped_age))
    upper_age = int(np.ceil(clipped_age))

    if lower_age == upper_age:
        return float(curve.loc[lower_age])

    lower_val = float(curve.loc[lower_age])
    upper_val = float(curve.loc[upper_age])
    frac = clipped_age - lower_age
    return lower_val + frac * (upper_val - lower_val)


def build_age_curve_priors(age_source_df: pd.DataFrame, start_year: int, end_year: int, all_players) -> tuple:
    """
    Build player prior dictionaries from smoothed age curves.
    """
    full_start_year = normalize_season_end_year(start_year)
    full_end_year = normalize_season_end_year(end_year)

    full_source_df = age_source_df.copy()
    full_source_df['player_id'] = full_source_df['nba_id'].astype('Int64').astype(str)
    target_players = set(all_players)
    age_source_df = full_source_df[full_source_df['player_id'].isin(target_players)].copy()

    if age_source_df.empty:
        raise ValueError("No overlapping player_stats_with_metrics rows for players in this RAPM run.")

    curves = {
        'off': build_age_curve(full_source_df[full_source_df['year'] <= full_end_year], 'off'),
        'def': build_age_curve(full_source_df[full_source_df['year'] <= full_end_year], 'def'),
    }

    priors = {'off': {}, 'def': {}}

    for side, config in AGE_CURVE_TARGETS.items():
        poss_col = config['poss_col']
        side_rows = age_source_df[['player_id', 'year', 'Age', poss_col]].copy()
        side_rows[poss_col] = pd.to_numeric(side_rows[poss_col], errors='coerce').fillna(0.0).clip(lower=0.0)
        side_rows = side_rows[side_rows['Age'].notna()]

        if side_rows.empty:
            continue

        for player_id in target_players:
            player_rows = side_rows[side_rows['player_id'] == player_id]
            if player_rows.empty:
                continue

            player_window = player_rows[
                player_rows['year'].between(full_start_year, full_end_year, inclusive='both')
            ]
            player_window = player_window[player_window[poss_col] > 0]

            if not player_window.empty:
                effective_age = float(np.average(player_window['Age'], weights=player_window[poss_col]))
            else:
                fallback_rows = player_rows[player_rows['year'] <= full_end_year].sort_values('year', ascending=False)
                if fallback_rows.empty:
                    continue
                effective_age = float(fallback_rows.iloc[0]['Age'])

            prior_pts_per_100 = evaluate_age_curve(curves[side], effective_age)
            if prior_pts_per_100 is None or not np.isfinite(prior_pts_per_100):
                continue

            priors[side][player_id] = prior_pts_per_100 / 100.0

    logging.info(
        "Age-curve priors built for %d/%d offensive players and %d/%d defensive players",
        len(priors['off']), len(target_players), len(priors['def']), len(target_players)
    )
    return priors['off'], priors['def'], curves


def clamp_age_bin(age_value: float):
    """Collapse ages onto the configured integer dummy buckets."""
    if age_value is None or pd.isna(age_value):
        return None
    age_bin = int(np.floor(float(age_value)))
    return int(min(max(age_bin, AGE_DUMMY_MIN), AGE_DUMMY_MAX))


def register_age_dummy_features(player_to_col, start_idx):
    """Register separate offense and defense age fixed-effect features."""
    ages = list(range(AGE_DUMMY_MIN, AGE_DUMMY_MAX + 1))
    metadata = {
        'enabled': True,
        'ages': ages,
        'reference_age': AGE_DUMMY_REFERENCE,
        'off_feature_names': [],
        'def_feature_names': [],
        'feature_names': [],
        'off_slot_counts': {age: 0 for age in ages},
        'def_slot_counts': {age: 0 for age in ages},
        'total_slot_counts': {age: 0 for age in ages},
        'player_coverage': {},
    }

    idx = start_idx
    for age in ages:
        if age == AGE_DUMMY_REFERENCE:
            continue
        off_feat_name = f"age_off_fe_{age}"
        player_to_col[off_feat_name] = idx
        metadata['off_feature_names'].append(off_feat_name)
        metadata['feature_names'].append(off_feat_name)
        idx += 1
        def_feat_name = f"age_def_fe_{age}"
        player_to_col[def_feat_name] = idx
        metadata['def_feature_names'].append(def_feat_name)
        metadata['feature_names'].append(def_feat_name)
        idx += 1

    logging.info(
        "Added %d age fixed-effect columns (%d offense + %d defense, ages %d-%d, reference=%d)",
        len(metadata['feature_names']),
        len(metadata['off_feature_names']),
        len(metadata['def_feature_names']),
        AGE_DUMMY_MIN,
        AGE_DUMMY_MAX,
        AGE_DUMMY_REFERENCE,
    )
    return metadata, idx


def prepare_age_dummy_payload(age_source_df: pd.DataFrame, df_transformed: pd.DataFrame, all_players) -> dict:
    """
    Build season-aware age-bin arrays aligned to the transformed home/away sample.

    Missing season rows fall back to the latest regular-season age at or before the
    possession season. Players with no age row at all remain unassigned.
    """
    source = age_source_df[['nba_id', 'year', 'Age']].copy()
    source['player_id'] = source['nba_id'].astype('Int64').astype(str)
    source['year'] = pd.to_numeric(source['year'], errors='coerce').astype('Int64')
    source['Age'] = pd.to_numeric(source['Age'], errors='coerce')
    source = source[source['player_id'].isin(set(all_players))]
    source = source[source['year'].notna() & source['Age'].notna()].copy()
    source['age_bin'] = source['Age'].map(clamp_age_bin)
    source = source[source['age_bin'].notna()].copy()

    if source.empty:
        raise ValueError("No overlapping player_stats_with_metrics ages found for age-dummy mode.")

    source['year'] = source['year'].astype(int)
    source['age_bin'] = source['age_bin'].astype(int)
    source = source.sort_values(['player_id', 'year']).drop_duplicates(['player_id', 'year'], keep='last')

    target_seasons = sorted(pd.Series(df_transformed['season']).dropna().astype(int).unique().tolist())
    lookup_by_season = {}
    for season in target_seasons:
        season_rows = source[source['year'] <= season].drop_duplicates('player_id', keep='last')
        lookup_by_season[season] = dict(zip(season_rows['player_id'], season_rows['age_bin']))

    h_players = df_transformed[[f'h{k}' for k in range(1, 6)]].values.astype(int)
    a_players = df_transformed[[f'a{k}' for k in range(1, 6)]].values.astype(int)
    season_values = pd.to_numeric(df_transformed['season'], errors='coerce').fillna(-1).astype(int).values
    h_age_bins = np.full(h_players.shape, -1, dtype=np.int16)
    a_age_bins = np.full(a_players.shape, -1, dtype=np.int16)

    for season in target_seasons:
        season_lookup = lookup_by_season.get(season, {})
        if not season_lookup:
            continue
        row_idx = np.where(season_values == season)[0]
        if len(row_idx) == 0:
            continue

        for player_array, target_array in ((h_players, h_age_bins), (a_players, a_age_bins)):
            mapped = (
                pd.Series(player_array[row_idx].ravel())
                .astype(str)
                .map(season_lookup)
                .fillna(-1)
                .astype(np.int16)
                .to_numpy()
                .reshape(-1, 5)
            )
            target_array[row_idx] = mapped

    covered_players = set()
    for season_lookup in lookup_by_season.values():
        covered_players.update(season_lookup.keys())
    missing_players = sorted(set(all_players) - covered_players)

    total_slots = int(h_age_bins.size + a_age_bins.size)
    mapped_slots = int((h_age_bins >= AGE_DUMMY_MIN).sum() + (a_age_bins >= AGE_DUMMY_MIN).sum())
    logging.info(
        "Age-dummy coverage: %d/%d players, %d/%d lineup slots (%.2f%%)",
        len(covered_players),
        len(all_players),
        mapped_slots,
        total_slots,
        100.0 * mapped_slots / total_slots if total_slots else 0.0,
    )
    if missing_players:
        logging.info("Players without age rows in this run: %d", len(missing_players))

    return {
        'enabled': True,
        'h_age_bins': h_age_bins,
        'a_age_bins': a_age_bins,
        'covered_players': len(covered_players),
        'missing_players': missing_players,
        'mapped_slots': mapped_slots,
        'total_slots': total_slots,
    }


###############################################################################
# 0c) DARKO STRENGTH-SPLIT UTILITIES
###############################################################################
def load_darko_strength_lookup(darko_history_path) -> dict:
    """Load DARKO history values keyed by (nba_id, season end year)."""
    path = Path(darko_history_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing DARKO history file: {path}")

    darko_df = pd.read_csv(path, usecols=["nba_id", "season", "date", "o_dpm", "d_dpm"])
    darko_df["nba_id"] = pd.to_numeric(darko_df["nba_id"], errors="coerce")
    darko_df["season"] = pd.to_numeric(darko_df["season"], errors="coerce")
    darko_df["date"] = pd.to_datetime(darko_df["date"], errors="coerce")
    for col in ["o_dpm", "d_dpm"]:
        darko_df[col] = pd.to_numeric(darko_df[col], errors="coerce").fillna(0.0)

    darko_df = darko_df.dropna(subset=["nba_id", "season"]).copy()
    darko_df["nba_id"] = darko_df["nba_id"].astype(int)
    darko_df["season"] = darko_df["season"].astype(int)
    darko_df = darko_df.sort_values(["nba_id", "season", "date"]).drop_duplicates(
        ["nba_id", "season"],
        keep="last",
    )

    lookup = {
        "off": {
            (int(row.nba_id), int(row.season)): float(row.o_dpm)
            for row in darko_df.itertuples(index=False)
        },
        "def": {
            (int(row.nba_id), int(row.season)): float(row.d_dpm)
            for row in darko_df.itertuples(index=False)
        },
        "path": str(path),
        "rows": int(len(darko_df)),
        "min_season": int(darko_df["season"].min()) if not darko_df.empty else None,
        "max_season": int(darko_df["season"].max()) if not darko_df.empty else None,
    }
    logging.info(
        "Loaded %d DARKO player-season rows from %s (seasons %s-%s)",
        lookup["rows"],
        path,
        lookup["min_season"],
        lookup["max_season"],
    )
    return lookup


def classify_darko_strength(lineup_total: float, side: str) -> str:
    """
    Bucket a five-player lineup's DARKO strength.

    Mirrors the legacy thresholds:
    - opposing defense: strong > 2.0, else not_strong
    - opposing offense: strong > 2.5, else not_strong
    """
    strong_cut = 2.0 if side == "def" else 2.5
    return "strong" if lineup_total > strong_cut else "not_strong"


def prepare_strength_split_payload(df_transformed: pd.DataFrame, darko_history_path) -> dict:
    """Build per-row opponent-strength buckets and per-player exposure counts."""
    darko_lookup = load_darko_strength_lookup(darko_history_path)

    h_players = df_transformed[[f"h{k}" for k in range(1, 6)]].values.astype(int)
    a_players = df_transformed[[f"a{k}" for k in range(1, 6)]].values.astype(int)
    home_poss = df_transformed["home_poss"].values
    seasons = pd.to_numeric(df_transformed["season"], errors="coerce").fillna(0).astype(int).values

    offense_players = np.where(home_poss[:, None] == 1, h_players, a_players)
    defense_players = np.where(home_poss[:, None] == 1, a_players, h_players)

    off_lookup = darko_lookup["off"]
    def_lookup = darko_lookup["def"]

    off_buckets = np.empty(len(df_transformed), dtype=object)
    def_buckets = np.empty(len(df_transformed), dtype=object)
    missing_off_slots = 0
    missing_def_slots = 0

    off_counts = defaultdict(lambda: {bucket: 0 for bucket in STRENGTH_BUCKETS})
    def_counts = defaultdict(lambda: {bucket: 0 for bucket in STRENGTH_BUCKETS})
    bucket_row_counts = {
        "off": {bucket: 0 for bucket in STRENGTH_BUCKETS},
        "def": {bucket: 0 for bucket in STRENGTH_BUCKETS},
    }

    for row_idx, season in enumerate(seasons):
        off_total = 0.0
        def_total = 0.0

        for pid in offense_players[row_idx]:
            if pid <= 0:
                continue
            key = (int(pid), int(season))
            if key not in off_lookup:
                missing_off_slots += 1
            off_total += off_lookup.get(key, 0.0)

        for pid in defense_players[row_idx]:
            if pid <= 0:
                continue
            key = (int(pid), int(season))
            if key not in def_lookup:
                missing_def_slots += 1
            def_total += def_lookup.get(key, 0.0)

        opp_def_bucket = classify_darko_strength(def_total, "def")
        opp_off_bucket = classify_darko_strength(off_total, "off")
        off_buckets[row_idx] = opp_def_bucket
        def_buckets[row_idx] = opp_off_bucket
        bucket_row_counts["off"][opp_def_bucket] += 1
        bucket_row_counts["def"][opp_off_bucket] += 1

        for pid in offense_players[row_idx]:
            if pid > 0:
                off_counts[str(int(pid))][opp_def_bucket] += 1
        for pid in defense_players[row_idx]:
            if pid > 0:
                def_counts[str(int(pid))][opp_off_bucket] += 1

    total_slots = int(len(df_transformed) * 5)
    logging.info(
        "DARKO strength rows, offense vs defense buckets: %s; defense vs offense buckets: %s",
        bucket_row_counts["off"],
        bucket_row_counts["def"],
    )
    if missing_off_slots or missing_def_slots:
        logging.info(
            "DARKO strength missing lineup slots defaulted to 0: offense=%d/%d, defense=%d/%d",
            missing_off_slots,
            total_slots,
            missing_def_slots,
            total_slots,
        )

    return {
        "enabled": True,
        "off_buckets": off_buckets,
        "def_buckets": def_buckets,
        "off_counts": off_counts,
        "def_counts": def_counts,
        "bucket_row_counts": bucket_row_counts,
        "darko_path": str(darko_history_path),
        "missing_off_slots": missing_off_slots,
        "missing_def_slots": missing_def_slots,
        "total_slots": total_slots,
    }


def register_strength_split_features(player_to_col, all_players, start_idx):
    """Register per-player strong/not-strong interaction features."""
    metadata = {
        "enabled": True,
        "feature_names": [],
        "off_feature_names": [],
        "def_feature_names": [],
        "buckets": list(STRENGTH_BUCKETS),
    }

    idx = start_idx
    for player_id in all_players:
        for bucket in STRENGTH_BUCKETS:
            off_feat = f"{player_id}_facing_{bucket}_off"
            player_to_col[off_feat] = idx
            metadata["off_feature_names"].append(off_feat)
            metadata["feature_names"].append(off_feat)
            idx += 1

        for bucket in STRENGTH_BUCKETS:
            def_feat = f"{player_id}_facing_{bucket}_def"
            player_to_col[def_feat] = idx
            metadata["def_feature_names"].append(def_feat)
            metadata["feature_names"].append(def_feat)
            idx += 1

    logging.info(
        "Added %d DARKO strength-split player interaction columns (%d players x 4)",
        len(metadata["feature_names"]),
        len(all_players),
    )
    return metadata, idx


def append_strength_split_entries(
    row_indices,
    col_indices,
    offense_players,
    defense_players,
    player_to_col,
    strength_split_payload,
):
    """Append row entries for per-player opponent-strength interaction features."""
    if not strength_split_payload or not strength_split_payload.get("enabled"):
        return

    off_buckets = strength_split_payload["off_buckets"]
    def_buckets = strength_split_payload["def_buckets"]

    for player_idx in range(5):
        player_ids = offense_players[:, player_idx]
        for row_idx, pid in enumerate(player_ids):
            if pid <= 0:
                continue
            feat_name = f"{int(pid)}_facing_{off_buckets[row_idx]}_off"
            if feat_name in player_to_col:
                row_indices.append(row_idx)
                col_indices.append(player_to_col[feat_name])

    for player_idx in range(5):
        player_ids = defense_players[:, player_idx]
        for row_idx, pid in enumerate(player_ids):
            if pid <= 0:
                continue
            feat_name = f"{int(pid)}_facing_{def_buckets[row_idx]}_def"
            if feat_name in player_to_col:
                row_indices.append(row_idx)
                col_indices.append(player_to_col[feat_name])


def prepare_strength_sensitivity_payload(df_transformed: pd.DataFrame, darko_history_path) -> dict:
    """
    Build continuous opponent-strength z-scores from DARKO lineup sums.

    Offensive slope rows use opposing defensive DARKO (`d_dpm`) strength.
    Defensive slope rows use opposing offensive DARKO (`o_dpm`) strength.
    Both are standardized within season so a one-unit slope is a one-season-SD
    harder opponent lineup.
    """
    darko_lookup = load_darko_strength_lookup(darko_history_path)

    h_players = df_transformed[[f"h{k}" for k in range(1, 6)]].values.astype(int)
    a_players = df_transformed[[f"a{k}" for k in range(1, 6)]].values.astype(int)
    home_poss = df_transformed["home_poss"].values
    seasons = pd.to_numeric(df_transformed["season"], errors="coerce").fillna(0).astype(int).values

    offense_players = np.where(home_poss[:, None] == 1, h_players, a_players)
    defense_players = np.where(home_poss[:, None] == 1, a_players, h_players)

    off_lookup = darko_lookup["off"]
    def_lookup = darko_lookup["def"]

    opp_def_strength = np.zeros(len(df_transformed), dtype=np.float64)
    opp_off_strength = np.zeros(len(df_transformed), dtype=np.float64)
    missing_off_slots = 0
    missing_def_slots = 0

    for row_idx, season in enumerate(seasons):
        off_total = 0.0
        def_total = 0.0

        for pid in offense_players[row_idx]:
            if pid <= 0:
                continue
            key = (int(pid), int(season))
            if key not in off_lookup:
                missing_off_slots += 1
            off_total += off_lookup.get(key, 0.0)

        for pid in defense_players[row_idx]:
            if pid <= 0:
                continue
            key = (int(pid), int(season))
            if key not in def_lookup:
                missing_def_slots += 1
            def_total += def_lookup.get(key, 0.0)

        opp_def_strength[row_idx] = def_total
        opp_off_strength[row_idx] = off_total

    strength_df = pd.DataFrame({
        "season": seasons,
        "opp_def_strength": opp_def_strength,
        "opp_off_strength": opp_off_strength,
    })
    season_stats = (
        strength_df
        .groupby("season", as_index=True)
        .agg(
            opp_def_mean=("opp_def_strength", "mean"),
            opp_def_std=("opp_def_strength", "std"),
            opp_off_mean=("opp_off_strength", "mean"),
            opp_off_std=("opp_off_strength", "std"),
        )
    )
    season_stats[["opp_def_std", "opp_off_std"]] = season_stats[["opp_def_std", "opp_off_std"]].replace(0, np.nan).fillna(1.0)

    stat_lookup = season_stats.to_dict("index")
    opp_def_z = np.zeros(len(df_transformed), dtype=np.float64)
    opp_off_z = np.zeros(len(df_transformed), dtype=np.float64)
    for row_idx, season in enumerate(seasons):
        stats = stat_lookup[int(season)]
        opp_def_z[row_idx] = (
            opp_def_strength[row_idx] - stats["opp_def_mean"]
        ) / stats["opp_def_std"]
        opp_off_z[row_idx] = (
            opp_off_strength[row_idx] - stats["opp_off_mean"]
        ) / stats["opp_off_std"]

    total_slots = int(len(df_transformed) * 5)
    logging.info(
        "DARKO sensitivity z-score ranges: opp_def_z %.2f..%.2f, opp_off_z %.2f..%.2f",
        float(np.nanmin(opp_def_z)),
        float(np.nanmax(opp_def_z)),
        float(np.nanmin(opp_off_z)),
        float(np.nanmax(opp_off_z)),
    )
    if missing_off_slots or missing_def_slots:
        logging.info(
            "DARKO sensitivity missing lineup slots defaulted to 0: offense=%d/%d, defense=%d/%d",
            missing_off_slots,
            total_slots,
            missing_def_slots,
            total_slots,
        )

    return {
        "enabled": True,
        "opp_def_z": opp_def_z,
        "opp_off_z": opp_off_z,
        "off_exposure_sum": defaultdict(float),
        "def_exposure_sum": defaultdict(float),
        "darko_path": str(darko_history_path),
        "missing_off_slots": missing_off_slots,
        "missing_def_slots": missing_def_slots,
        "total_slots": total_slots,
        "season_stats": season_stats.reset_index().to_dict("records"),
    }


def register_strength_sensitivity_features(player_to_col, all_players, start_idx):
    """Register per-player continuous opponent-strength slope features."""
    metadata = {
        "enabled": True,
        "feature_names": [],
        "off_feature_names": [],
        "def_feature_names": [],
    }

    idx = start_idx
    for player_id in all_players:
        off_feat = f"{player_id}_strength_slope_off"
        player_to_col[off_feat] = idx
        metadata["off_feature_names"].append(off_feat)
        metadata["feature_names"].append(off_feat)
        idx += 1

        def_feat = f"{player_id}_strength_slope_def"
        player_to_col[def_feat] = idx
        metadata["def_feature_names"].append(def_feat)
        metadata["feature_names"].append(def_feat)
        idx += 1

    logging.info(
        "Added %d DARKO strength-sensitivity slope columns (%d players x 2)",
        len(metadata["feature_names"]),
        len(all_players),
    )
    return metadata, idx


def append_strength_sensitivity_entries(
    row_indices,
    col_indices,
    values,
    offense_players,
    defense_players,
    player_to_col,
    strength_sensitivity_payload,
    feature_scale,
):
    """Append row entries for per-player continuous opponent-strength slopes."""
    if not strength_sensitivity_payload or not strength_sensitivity_payload.get("enabled"):
        return

    opp_def_z = strength_sensitivity_payload["opp_def_z"]
    opp_off_z = strength_sensitivity_payload["opp_off_z"]
    off_exposure_sum = strength_sensitivity_payload["off_exposure_sum"]
    def_exposure_sum = strength_sensitivity_payload["def_exposure_sum"]

    for player_idx in range(5):
        player_ids = offense_players[:, player_idx]
        for row_idx, pid in enumerate(player_ids):
            if pid <= 0:
                continue
            player_id = str(int(pid))
            feat_name = f"{player_id}_strength_slope_off"
            if feat_name in player_to_col:
                row_indices.append(row_idx)
                col_indices.append(player_to_col[feat_name])
                values.append(float(opp_def_z[row_idx]) * feature_scale)
                off_exposure_sum[player_id] += float(opp_def_z[row_idx])

    for player_idx in range(5):
        player_ids = defense_players[:, player_idx]
        for row_idx, pid in enumerate(player_ids):
            if pid <= 0:
                continue
            player_id = str(int(pid))
            feat_name = f"{player_id}_strength_slope_def"
            if feat_name in player_to_col:
                row_indices.append(row_idx)
                col_indices.append(player_to_col[feat_name])
                values.append(float(opp_off_z[row_idx]) * feature_scale)
                def_exposure_sum[player_id] += float(opp_off_z[row_idx])

###############################################################################
# 1) DETECT FILE TYPE AND PREPARE COLUMNS
###############################################################################
def detect_file_type_and_prepare(df, pure=False, prefix=None, alt_baselines=None,
                                 fc_mode_baselines=None):
    """
    Detect the type of CSV file and prepare Off_Diff and Def_Diff columns.

    File types:
    - RAPM: has Net_Diff, Off_Diff, Def_Diff already
    - SPECIAL_RAPM: has Net_Diff, Off_Diff, Def_Diff (same as RAPM)
    - TOV: has Is_Turnover -> create Off_Diff, Def_Diff
    - BLOCK_RECOVERY: has Block_Recovered_By_Defense -> create Off_Diff, Def_Diff
    - BADPASS_TOV: uses Is_BadPass_TOV from TOV file
    - SCORING_TOV: derives Is_Turnover - Is_BadPass_TOV from TOV file
    - ALT_TS: uses FIRST_CHANCE Net_Diff plus same-sample non-turnover baseline on turnover rows
    - ALT_EFG / ALT_SQ / ALT_MAKE / ALT_FT: use FIRST_CHANCE scoring diffs plus same-sample turnover-row baselines
    - ALT_TOV / ALT_BADPASS_TOV / ALT_SCORING_TOV: use true first-chance turnover flags
    - FC_TRANSITION_SCORING / FC_HALFCOURT_SCORING: same-denominator conditional targets using same-sample mode PPP placeholders
    - FC_MODE_MIX / FC_TRANSITION_VALUE / FC_HALFCOURT_VALUE: additive first-chance mode decomposition targets
    - REB: has Offensive_Rebound -> create Off_Diff, Def_Diff
    - ASSIST_POINTS: has Assist_Points -> create Off_Diff, Def_Diff
    - RIM_ASSIST: has Is_Rim_Assist -> create Off_Diff, Def_Diff
    - DUNK: has Is_Dunk -> create Off_Diff, Def_Diff
    - DUNK_ASSIST: has Is_Dunk_Assist -> create Off_Diff, Def_Diff
    - RIM_FREQ: has Is_Rim_Attempt -> create Off_Diff, Def_Diff
    - RIM_FG_PCT: has Is_Rim_Make -> create Off_Diff, Def_Diff
    - THREE_FREQ: has Is_Three_Attempt -> create Off_Diff, Def_Diff
    - THREE_FG_PCT: has Is_Three_Make -> create Off_Diff, Def_Diff
    - MIDRANGE_FREQ: has Is_Midrange_Attempt -> create Off_Diff, Def_Diff
    - MIDRANGE_FG_PCT: has Is_Midrange_Make -> create Off_Diff, Def_Diff
    - PLAYTYPE_TS_MIX: has Playtype_Exp_PTS -> create Off_Diff, Def_Diff
    - TRANSITION_FREQ: has Is_Transition -> create Off_Diff, Def_Diff
    - TRANSITION_RIM: has Is_Transition_Rim -> create Off_Diff, Def_Diff
    - INITIAL_EV: has Initial_EV -> create Off_Diff, Def_Diff
    - RUSSELL_SHOTQUALITY / CONTEXT_SHOTQUALITY: has Off_Diff, Def_Diff from processed possession parquet
    - TS: has Net_Diff -> create Off_Diff, Def_Diff

    Args:
        df: Input dataframe
        pure: If True, use Net_Diff for RAPM files (removes 3pt/FT luck adjustments)
        prefix: Requested metric prefix (e.g., 'BADPASS_TOV', 'SCORING_TOV') for aliased metrics
    """
    df = df.copy()

    # Handle aliased prefixes that share a parquet file
    if prefix == 'BADPASS_TOV' and 'Is_BadPass_TOV' in df.columns:
        logging.info("Detected BADPASS_TOV (from TOV file)")
        df['Off_Diff'] = -df['Is_BadPass_TOV']
        df['Def_Diff'] = df['Is_BadPass_TOV']
        return df
    elif prefix == 'SCORING_TOV' and 'Is_Turnover' in df.columns:
        logging.info("Detected SCORING_TOV (derived from TOV file)")
        scoring_tov = df['Is_Turnover'] - df.get('Is_BadPass_TOV', 0)
        df['Off_Diff'] = -scoring_tov
        df['Def_Diff'] = scoring_tov
        return df
    elif prefix == 'ALT_TOV_VALUE' and 'Is_Turnover' in df.columns:
        if alt_baselines is None or prefix not in alt_baselines:
            raise ValueError("ALT_TOV_VALUE requires a same-sample ALT_TS baseline.")
        logging.info("Detected ALT_TOV_VALUE (point-valued first-chance turnover complement)")
        is_turnover = pd.to_numeric(df['Is_Turnover'], errors='coerce').fillna(0).astype(int)
        df['Off_Diff'] = np.where(
            is_turnover == 1,
            -alt_baselines[prefix],
            0.0,
        )
        df['Def_Diff'] = -df['Off_Diff']
        return df
    elif prefix in {'ALT_BADPASS_TOV_VALUE', 'ALT_SCORING_TOV_VALUE'} and 'Is_Turnover' in df.columns:
        if alt_baselines is None or prefix not in alt_baselines:
            raise ValueError(f"{prefix} requires a same-sample ALT_TS baseline.")
        if 'Is_BadPass_TOV' not in df.columns:
            raise ValueError(f"{prefix} requires FIRST_CHANCE rows with Is_BadPass_TOV.")
        logging.info("Detected %s (point-valued first-chance turnover child complement)", prefix)
        is_turnover = pd.to_numeric(df['Is_Turnover'], errors='coerce').fillna(0).astype(int)
        badpass_tov = pd.to_numeric(df['Is_BadPass_TOV'], errors='coerce').fillna(0).astype(int)
        if prefix == 'ALT_BADPASS_TOV_VALUE':
            child_mask = (is_turnover == 1) & (badpass_tov == 1)
        else:
            child_mask = (is_turnover == 1) & (badpass_tov == 0)
        df['Off_Diff'] = np.where(
            child_mask,
            -alt_baselines[prefix],
            0.0,
        )
        df['Def_Diff'] = -df['Off_Diff']
        return df
    elif prefix == 'ALT_FC_COMPLETION' and 'Is_Turnover' in df.columns:
        required = {'ALT_EFG_VALUE', 'ALT_FT'}
        if alt_baselines is None or not required.issubset(alt_baselines):
            raise ValueError("ALT_FC_COMPLETION requires same-sample ALT_EFG_VALUE and ALT_FT baselines.")
        for source_col in ['FC_EFG_VALUE_Diff', 'FC_FT_Diff', 'Net_Diff']:
            if source_col not in df.columns:
                raise ValueError(f"ALT_FC_COMPLETION requires FIRST_CHANCE rows with {source_col}.")
        logging.info("Detected ALT_FC_COMPLETION (point-valued first-chance complement to EFG value + FT)")
        is_turnover = pd.to_numeric(df['Is_Turnover'], errors='coerce').fillna(0).astype(int)
        first_chance = pd.to_numeric(df['Net_Diff'], errors='coerce')
        efg_value = pd.to_numeric(df['FC_EFG_VALUE_Diff'], errors='coerce')
        ft_value = pd.to_numeric(df['FC_FT_Diff'], errors='coerce')
        if first_chance.isna().any() or efg_value.isna().any() or ft_value.isna().any():
            raise ValueError("ALT_FC_COMPLETION found missing source values in the loaded sample.")
        efg_value_target = np.where(is_turnover == 1, alt_baselines['ALT_EFG_VALUE'], efg_value)
        ft_value_target = np.where(is_turnover == 1, alt_baselines['ALT_FT'], ft_value)
        df['Off_Diff'] = first_chance - efg_value_target - ft_value_target
        df['Def_Diff'] = -df['Off_Diff']
        return df
    elif prefix in ALT_COMPONENT_COLUMNS and 'Is_Turnover' in df.columns:
        if alt_baselines is None or prefix not in alt_baselines:
            raise ValueError(f"{prefix} requires a same-sample baseline computed from the exact loaded sample.")
        source_col = ALT_COMPONENT_COLUMNS[prefix]
        if source_col not in df.columns:
            raise ValueError(f"{prefix} requires FIRST_CHANCE rows with {source_col}.")
        logging.info("Detected %s (from FIRST_CHANCE file)", prefix)
        is_turnover = pd.to_numeric(df['Is_Turnover'], errors='coerce').fillna(0).astype(int)
        component_diff = pd.to_numeric(df[source_col], errors='coerce')
        if component_diff.isna().any():
            raise ValueError(f"{prefix} found missing {source_col} values in the loaded sample.")
        df['Off_Diff'] = np.where(is_turnover == 1, alt_baselines[prefix], component_diff)
        df['Def_Diff'] = -df['Off_Diff']
        return df
    elif prefix == 'ALT_TOV' and 'Is_Turnover' in df.columns:
        logging.info("Detected ALT_TOV (true first-chance turnover surface)")
        is_turnover = pd.to_numeric(df['Is_Turnover'], errors='coerce').fillna(0).astype(int)
        df['Off_Diff'] = -is_turnover
        df['Def_Diff'] = is_turnover
        return df
    elif prefix == 'ALT_BADPASS_TOV' and 'Is_BadPass_TOV' in df.columns:
        logging.info("Detected ALT_BADPASS_TOV (true first-chance bad-pass turnover surface)")
        badpass_tov = pd.to_numeric(df['Is_BadPass_TOV'], errors='coerce').fillna(0).astype(int)
        df['Off_Diff'] = -badpass_tov
        df['Def_Diff'] = badpass_tov
        return df
    elif prefix == 'ALT_SCORING_TOV' and 'Is_Turnover' in df.columns:
        logging.info("Detected ALT_SCORING_TOV (true first-chance scoring-turnover surface)")
        is_turnover = pd.to_numeric(df['Is_Turnover'], errors='coerce').fillna(0).astype(int)
        badpass_tov = pd.to_numeric(df.get('Is_BadPass_TOV', 0), errors='coerce').fillna(0).astype(int)
        scoring_tov = is_turnover - badpass_tov
        df['Off_Diff'] = -scoring_tov
        df['Def_Diff'] = scoring_tov
        return df
    elif prefix == 'FIRST_CHANCE' and 'Net_Diff' in df.columns:
        logging.info("Detected FIRST_CHANCE (point-value surface)")
        net_diff = pd.to_numeric(df['Net_Diff'], errors='coerce').fillna(0.0)
        df['Off_Diff'] = net_diff
        df['Def_Diff'] = -net_diff
        return df
    elif prefix in FIRST_CHANCE_MODE_PREFIXES and 'Net_Diff' in df.columns:
        if fc_mode_baselines is None:
            raise ValueError(f"{prefix} requires same-sample first-chance mode baselines.")
        if 'Is_FC_Transition_Possession' not in df.columns:
            raise ValueError(f"{prefix} requires FIRST_CHANCE rows with Is_FC_Transition_Possession.")
        logging.info("Detected %s (same-denominator first-chance mode target)", prefix)
        first_chance = pd.to_numeric(df['Net_Diff'], errors='coerce')
        if first_chance.isna().any():
            raise ValueError(f"{prefix} found missing Net_Diff values in the loaded sample.")
        is_transition = (
            pd.to_numeric(df['Is_FC_Transition_Possession'], errors='coerce')
            .fillna(0)
            .astype(int)
            .eq(1)
        )
        transition_ppp = fc_mode_baselines['transition_ppp']
        halfcourt_ppp = fc_mode_baselines['halfcourt_ppp']
        if prefix == 'FC_TRANSITION_SCORING':
            target = np.where(is_transition, first_chance, transition_ppp)
        elif prefix == 'FC_HALFCOURT_SCORING':
            target = np.where(is_transition, halfcourt_ppp, first_chance)
        elif prefix == 'FC_MODE_MIX':
            target = np.where(is_transition, transition_ppp, halfcourt_ppp)
        elif prefix == 'FC_TRANSITION_VALUE':
            target = np.where(is_transition, first_chance - transition_ppp, 0.0)
        elif prefix == 'FC_HALFCOURT_VALUE':
            target = np.where(is_transition, 0.0, first_chance - halfcourt_ppp)
        else:
            raise ValueError(f"Unsupported first-chance mode prefix: {prefix}")
        df['Off_Diff'] = target
        df['Def_Diff'] = -df['Off_Diff']
        return df

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
    elif 'Block_Recovered_By_Defense' in df.columns:
        logging.info("Detected BLOCK_RECOVERY file (has Block_Recovered_By_Defense)")
        df['Off_Diff'] = -df['Block_Recovered_By_Defense']
        df['Def_Diff'] = df['Block_Recovered_By_Defense']
    elif 'Assist_Points' in df.columns:
        logging.info("Detected ASSIST_POINTS file (has Assist_Points)")
        df['Off_Diff'] = df['Assist_Points']
        df['Def_Diff'] = -df['Assist_Points']
    elif 'Is_Rim_Assist' in df.columns:
        logging.info("Detected RIM_ASSIST file (has Is_Rim_Assist)")
        df['Off_Diff'] = df['Is_Rim_Assist']
        df['Def_Diff'] = -df['Is_Rim_Assist']
    elif 'Is_Dunk_Assist' in df.columns:
        logging.info("Detected DUNK_ASSIST file (has Is_Dunk_Assist)")
        df['Off_Diff'] = df['Is_Dunk_Assist']
        df['Def_Diff'] = -df['Is_Dunk_Assist']
    elif 'Is_Dunk' in df.columns:
        logging.info("Detected DUNK file (has Is_Dunk)")
        df['Off_Diff'] = df['Is_Dunk']
        df['Def_Diff'] = -df['Is_Dunk']
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
    elif 'Is_Midrange_Make' in df.columns:
        logging.info("Detected MIDRANGE_FG_PCT file (has Is_Midrange_Make)")
        # For midrange FG%: offense makes midrange (good), defense allows make (bad)
        df['Off_Diff'] = df['Is_Midrange_Make']
        df['Def_Diff'] = -df['Is_Midrange_Make']
    elif 'Is_Three_Attempt' in df.columns:
        logging.info("Detected THREE_FREQ file (has Is_Three_Attempt)")
        df['Off_Diff'] = df['Is_Three_Attempt']
        df['Def_Diff'] = -df['Is_Three_Attempt']
    elif 'Is_Three_Make' in df.columns:
        logging.info("Detected THREE_FG_PCT file (has Is_Three_Make)")
        df['Off_Diff'] = df['Is_Three_Make']
        df['Def_Diff'] = -df['Is_Three_Make']
    elif 'Playtype_Exp_PTS' in df.columns:
        logging.info("Detected PLAYTYPE_TS_MIX file (has Playtype_Exp_PTS)")
        df['Off_Diff'] = df['Playtype_Exp_PTS']
        df['Def_Diff'] = -df['Playtype_Exp_PTS']
    elif 'Playtype_Proxy_PTS' in df.columns:
        logging.info("Detected PLAYTYPE_PROXY_PTS file (has Playtype_Proxy_PTS)")
        df['Off_Diff'] = df['Playtype_Proxy_PTS']
        df['Def_Diff'] = -df['Playtype_Proxy_PTS']
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

    Vectorized implementation — builds home/away player sets per game, then
    determines offense direction for all rows at once using numpy operations.
    """
    logging.info("Transforming O/D format to home/away format (vectorized)...")

    df = df.copy().reset_index(drop=True)

    # Extract player arrays
    o_cols = [f'O{j}' for j in range(1, 6)]
    d_cols = [f'D{j}' for j in range(1, 6)]
    O = df[o_cols].values.astype(np.float64)
    D = df[d_cols].values.astype(np.float64)

    # Compute per-row score deltas
    away_score = df['away_score'].values.astype(np.float64)
    home_score = df['home_score'].values.astype(np.float64)
    game_ids = df['game_id'].values

    # Detect game boundaries
    game_starts = np.empty(len(df), dtype=bool)
    game_starts[0] = True
    game_starts[1:] = game_ids[1:] != game_ids[:-1]

    # Score deltas: for first row of each game, delta = score itself; otherwise diff
    away_delta = np.empty(len(df), dtype=np.float64)
    home_delta = np.empty(len(df), dtype=np.float64)
    away_delta[0] = away_score[0]
    home_delta[0] = home_score[0]
    away_delta[1:] = away_score[1:] - away_score[:-1]
    home_delta[1:] = home_score[1:] - home_score[:-1]
    # Reset at game boundaries
    away_delta[game_starts] = away_score[game_starts]
    home_delta[game_starts] = home_score[game_starts]

    # First pass: build home AND away player sets per game
    # When away scores (away_delta > 0): O players are away, D players are home
    # When home scores (home_delta > 0): O players are home, D players are away
    logging.info("  Building home/away player sets per game...")
    game_home_players = {}
    game_away_players = {}

    away_scored = away_delta > 0
    home_scored = (home_delta > 0) & ~away_scored  # prioritize away if both

    # Rows where away scored -> D players are home, O players are away
    for idx in np.where(away_scored)[0]:
        gid = game_ids[idx]
        if gid not in game_home_players:
            game_home_players[gid] = set()
            game_away_players[gid] = set()
        game_home_players[gid].update(pid for pid in D[idx] if pid > 0)
        game_away_players[gid].update(pid for pid in O[idx] if pid > 0)

    # Rows where home scored -> O players are home, D players are away
    for idx in np.where(home_scored)[0]:
        gid = game_ids[idx]
        if gid not in game_home_players:
            game_home_players[gid] = set()
            game_away_players[gid] = set()
        game_home_players[gid].update(pid for pid in O[idx] if pid > 0)
        game_away_players[gid].update(pid for pid in D[idx] if pid > 0)

    # Second pass: determine off_is_home using 5-player majority vote
    # For each row, count how many of O1-O5 overlap with home vs away sets
    logging.info("  Determining offense direction per row (majority vote)...")
    home_lookup = set()
    away_lookup = set()
    for gid, home_set in game_home_players.items():
        for pid in home_set:
            home_lookup.add((gid, pid))
    for gid, away_set in game_away_players.items():
        for pid in away_set:
            away_lookup.add((gid, pid))

    # Count O1-O5 overlap with home and away sets per row
    home_count = np.zeros(len(df), dtype=int)
    away_count = np.zeros(len(df), dtype=int)
    for j in range(5):
        o_vals = O[:, j]
        home_count += np.array(
            [(game_ids[i], o_vals[i]) in home_lookup for i in range(len(df))],
            dtype=int
        )
        away_count += np.array(
            [(game_ids[i], o_vals[i]) in away_lookup for i in range(len(df))],
            dtype=int
        )
    # Majority vote: off_is_home when more O players match home than away
    # Ties (e.g. 5 vs 5) resolve to False (same as old code's > operator)
    off_is_home = home_count > away_count

    # Pre-possession score margin (use previous row's margin, 0 at game start)
    logging.info("  Computing pre-possession score margins...")
    sm_series = df['score_margin'].astype(str).str.upper().replace('TIE', '0')
    home_margin = pd.to_numeric(sm_series, errors='coerce').fillna(0).values.astype(np.float64)

    # Shift margin by 1 within each game (pre-possession = previous row's margin)
    prev_margin = np.zeros(len(df), dtype=np.float64)
    prev_margin[1:] = home_margin[:-1]
    prev_margin[game_starts] = 0  # Game starts at 0

    # Convert to offensive-team-relative score_diff
    score_diff = np.where(off_is_home, prev_margin, -prev_margin)

    # Build output arrays
    logging.info("  Building output dataframe...")
    n = len(df)
    home_poss = off_is_home.astype(int)
    pts = df['Off_Diff'].values.astype(np.float64)

    # Player assignment: when off_is_home, h=O and a=D; otherwise h=D and a=O
    h = np.where(off_is_home[:, None], O, D).astype(int)
    a = np.where(off_is_home[:, None], D, O).astype(int)

    season_source = df['source_season_end_year'].values if 'source_season_end_year' in df.columns else df['Season'].values

    result_df = pd.DataFrame({
        'home_poss': home_poss,
        'pts': pts,
        'season': normalize_season_series(season_source),
        'season_phase': df['season_phase'].astype(str).values if 'season_phase' in df.columns else np.repeat('RS', len(df)),
        'date': df['game_date'].values,
        'period': df['period'].values.astype(int),
        'gameid': df['game_id'].astype(str).values,
        'score_diff': score_diff,
        'h1': h[:, 0], 'h2': h[:, 1], 'h3': h[:, 2], 'h4': h[:, 3], 'h5': h[:, 4],
        'a1': a[:, 0], 'a2': a[:, 1], 'a3': a[:, 2], 'a4': a[:, 3], 'a5': a[:, 4],
    })

    logging.info(f"Transformed {len(result_df)} possessions")
    return result_df

###############################################################################
# 3) BUILD SIMPLIFIED DESIGN MATRIX (OPTIMIZED)
###############################################################################
def register_season_effect_features(df, player_to_col, start_idx):
    """
    Add one season fixed-effect column per non-reference season-phase combo.

    Uses the earliest observed season-phase combo as the omitted reference category,
    preferring RS over PS within the same season.
    """
    season_phase_df = (
        df[['season', 'season_phase']]
        .dropna()
        .copy()
    )
    if season_phase_df.empty:
        metadata = {
            'enabled': False,
            'season_keys': [],
            'reference_key': None,
            'reference_season': None,
            'reference_phase': None,
            'feature_names': [],
        }
        return metadata, start_idx

    season_phase_df['season'] = season_phase_df['season'].astype(int)
    season_phase_df['season_phase'] = season_phase_df['season_phase'].astype(str).str.upper()
    phase_order = {'RS': 0, 'PS': 1}
    season_keys = sorted(
        {(row.season, row.season_phase) for row in season_phase_df.itertuples(index=False)},
        key=lambda item: (item[0], phase_order.get(item[1], 99), item[1]),
    )
    metadata = {
        'enabled': False,
        'season_keys': season_keys,
        'reference_key': None,
        'reference_season': None,
        'reference_phase': None,
        'feature_names': [],
    }

    if len(season_keys) <= 1:
        return metadata, start_idx

    reference_season, reference_phase = season_keys[0]
    metadata['enabled'] = True
    metadata['reference_key'] = (reference_season, reference_phase)
    metadata['reference_season'] = reference_season
    metadata['reference_phase'] = reference_phase

    idx = start_idx
    for season, phase in season_keys[1:]:
        feat_name = f"season_fe_{season}_{phase.lower()}"
        player_to_col[feat_name] = idx
        metadata['feature_names'].append(feat_name)
        idx += 1

    logging.info(
        "Added %d season fixed-effect columns (reference=%s_%s)",
        len(metadata['feature_names']),
        reference_season,
        reference_phase,
    )
    return metadata, idx


def append_age_dummy_entries(
    signed_row_indices,
    signed_col_indices,
    signed_values,
    offense_age_bins,
    defense_age_bins,
    player_to_col,
    age_effect_meta,
):
    """Append separate offense and defense age-bin fixed effects."""
    if not age_effect_meta.get('enabled'):
        return

    reference_age = age_effect_meta['reference_age']
    for age in age_effect_meta['ages']:
        off_counts = (offense_age_bins == age).sum(axis=1).astype(np.int16)
        def_counts = (defense_age_bins == age).sum(axis=1).astype(np.int16)
        total_slots = int(off_counts.sum() + def_counts.sum())
        age_effect_meta['off_slot_counts'][age] = int(off_counts.sum())
        age_effect_meta['def_slot_counts'][age] = int(def_counts.sum())
        age_effect_meta['total_slot_counts'][age] = total_slots

        if age == reference_age:
            continue

        off_rows = np.where(off_counts != 0)[0]
        if len(off_rows) > 0:
            feat_name = f"age_off_fe_{age}"
            signed_row_indices.extend(off_rows.tolist())
            signed_col_indices.extend([player_to_col[feat_name]] * len(off_rows))
            signed_values.extend(off_counts[off_rows].astype(np.float64).tolist())

        def_rows = np.where(def_counts != 0)[0]
        if len(def_rows) > 0:
            feat_name = f"age_def_fe_{age}"
            signed_row_indices.extend(def_rows.tolist())
            signed_col_indices.extend([player_to_col[feat_name]] * len(def_rows))
            signed_values.extend(def_counts[def_rows].astype(np.float64).tolist())


def build_simple_design_matrix(
    df,
    player_to_col,
    season_effects=False,
    age_dummy_payload=None,
    strength_split_payload=None,
    strength_sensitivity_payload=None,
    strength_sensitivity_feature_scale=1.0,
):
    """
    Build sparse design matrix with only offense/defense player indicators.
    Vectorized version for speed - processes all rows at once.
    """
    logging.info("Building design matrix (vectorized)...")
    
    n_samples = len(df)
    n_player_features = max(v for v in player_to_col.values()) + 1
    idx = n_player_features
    season_effect_meta = {
        'enabled': False,
        'season_keys': [],
        'reference_key': None,
        'reference_season': None,
        'reference_phase': None,
        'feature_names': [],
    }

    if season_effects:
        season_effect_meta, idx = register_season_effect_features(df, player_to_col, idx)

    age_effect_meta = {'enabled': False}
    if age_dummy_payload and age_dummy_payload.get('enabled'):
        age_effect_meta, idx = register_age_dummy_features(player_to_col, idx)
        age_effect_meta['player_coverage'] = {
            'covered_players': age_dummy_payload['covered_players'],
            'missing_players': len(age_dummy_payload['missing_players']),
            'mapped_slots': age_dummy_payload['mapped_slots'],
            'total_slots': age_dummy_payload['total_slots'],
        }

    strength_effect_meta = {'enabled': False}
    if strength_split_payload and strength_split_payload.get('enabled'):
        strength_effect_meta, idx = register_strength_split_features(player_to_col, sorted(strength_split_payload['all_players']), idx)

    strength_sensitivity_meta = {'enabled': False}
    if strength_sensitivity_payload and strength_sensitivity_payload.get('enabled'):
        strength_sensitivity_meta, idx = register_strength_sensitivity_features(
            player_to_col,
            sorted(strength_sensitivity_payload['all_players']),
            idx,
        )
        strength_sensitivity_meta['feature_scale'] = strength_sensitivity_feature_scale

    n_features = idx
    
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
    signed_row_indices = []
    signed_col_indices = []
    signed_values = []
    
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

    append_strength_split_entries(
        row_indices,
        col_indices,
        offense_players,
        defense_players,
        player_to_col,
        strength_split_payload,
    )

    append_strength_sensitivity_entries(
        signed_row_indices,
        signed_col_indices,
        signed_values,
        offense_players,
        defense_players,
        player_to_col,
        strength_sensitivity_payload,
        strength_sensitivity_feature_scale,
    )

    # Process season fixed effects
    if season_effect_meta['enabled']:
        season_values = df['season'].values.astype(int)
        phase_values = df['season_phase'].astype(str).values
        for feat_name in season_effect_meta['feature_names']:
            season_str, phase_str = feat_name.replace("season_fe_", "", 1).rsplit("_", 1)
            season = int(season_str)
            phase = phase_str.upper()
            season_rows = np.where((season_values == season) & (phase_values == phase))[0]
            if len(season_rows) == 0:
                continue
            row_indices.extend(season_rows.tolist())
            col_indices.extend([player_to_col[feat_name]] * len(season_rows))

    if age_effect_meta['enabled']:
        h_age_bins = age_dummy_payload['h_age_bins']
        a_age_bins = age_dummy_payload['a_age_bins']
        offense_age_bins = np.where(home_poss[:, None] == 1, h_age_bins, a_age_bins)
        defense_age_bins = np.where(home_poss[:, None] == 1, a_age_bins, h_age_bins)
        append_age_dummy_entries(
            signed_row_indices,
            signed_col_indices,
            signed_values,
            offense_age_bins,
            defense_age_bins,
            player_to_col,
            age_effect_meta,
        )
    
    # Create sparse matrix
    rows = np.array(row_indices + signed_row_indices, dtype=np.int32)
    cols = np.array(col_indices + signed_col_indices, dtype=np.int32)
    data = np.concatenate([
        np.ones(len(row_indices), dtype=np.float64),
        np.array(signed_values, dtype=np.float64),
    ])
    X = coo_matrix((data, (rows, cols)),
                   shape=(n_samples, n_features), dtype=np.float64)
    
    n_season_features = len(season_effect_meta['feature_names'])
    n_age_features = len(age_effect_meta.get('feature_names', []))
    n_strength_features = len(strength_effect_meta.get('feature_names', []))
    n_strength_sensitivity_features = len(strength_sensitivity_meta.get('feature_names', []))
    if n_season_features > 0:
        logging.info(
            "Design matrix shape: %s (%d player + %d season + %d age + %d strength + %d sensitivity)",
            X.shape,
            n_player_features,
            n_season_features,
            n_age_features,
            n_strength_features,
            n_strength_sensitivity_features,
        )
    elif n_age_features > 0 or n_strength_features > 0 or n_strength_sensitivity_features > 0:
        logging.info(
            "Design matrix shape: %s (%d player + %d age + %d strength + %d sensitivity)",
            X.shape,
            n_player_features,
            n_age_features,
            n_strength_features,
            n_strength_sensitivity_features,
        )
    else:
        logging.info(f"Design matrix shape: {X.shape}")
    return X.tocsr(), pts, {
        'season_effects': season_effect_meta,
        'age_dummies': age_effect_meta,
        'strength_splits': strength_effect_meta,
        'strength_sensitivity': strength_sensitivity_meta,
    }

###############################################################################
# 3b) BUILD DESIGN MATRIX WITH RUBBERBAND FEATURES
###############################################################################
# Rubberband bins: offensive-team-relative score differential
RUBBERBAND_BINS = [
    ('rb_neg3', None, -15),    # Far behind: sd <= -15
    ('rb_neg2', -15, -10),     # Behind: -15 < sd <= -10
    ('rb_neg1', -10, -5),      # Slightly behind: -10 < sd <= -5
    # Close game: -5 < sd <= 5 (REFERENCE — omitted)
    ('rb_pos1', 5, 10),        # Slightly ahead: 5 < sd <= 10
    ('rb_pos2', 10, 15),       # Ahead: 10 < sd <= 15
    ('rb_pos3', 15, None),     # Far ahead: sd > 15
]

def build_design_matrix_with_rubberband(
    df,
    player_to_col,
    season_effects=False,
    age_dummy_payload=None,
    strength_split_payload=None,
    strength_sensitivity_payload=None,
    strength_sensitivity_feature_scale=1.0,
):
    """
    Build sparse design matrix with player indicators + rubberband bin x period features.

    Rubberband features are binary indicators for 6 score-margin bins x 4 periods = 24 features.
    Close game (-5 to +5) is the reference category (omitted).
    Periods 5+ are capped to period 4 (overtime treated as Q4).
    """
    logging.info("Building design matrix with rubberband features (vectorized)...")

    n_samples = len(df)
    n_player_features = max(v for v in player_to_col.values()) + 1

    # Add rubberband feature columns to player_to_col
    rb_feature_names = []
    rb_start_idx = n_player_features
    idx = rb_start_idx
    for bin_name, _, _ in RUBBERBAND_BINS:
        for period in range(1, 5):  # periods 1-4
            feat_name = f"{bin_name}_period{period}"
            player_to_col[feat_name] = idx
            rb_feature_names.append(feat_name)
            idx += 1

    n_features = idx
    n_rb_features = len(rb_feature_names)
    logging.info(f"Added {n_rb_features} rubberband features (6 bins x 4 periods)")

    season_effect_meta = {
        'enabled': False,
        'season_keys': [],
        'reference_key': None,
        'reference_season': None,
        'reference_phase': None,
        'feature_names': [],
    }
    if season_effects:
        season_effect_meta, idx = register_season_effect_features(df, player_to_col, idx)
    age_effect_meta = {'enabled': False}
    if age_dummy_payload and age_dummy_payload.get('enabled'):
        age_effect_meta, idx = register_age_dummy_features(player_to_col, idx)
        age_effect_meta['player_coverage'] = {
            'covered_players': age_dummy_payload['covered_players'],
            'missing_players': len(age_dummy_payload['missing_players']),
            'mapped_slots': age_dummy_payload['mapped_slots'],
            'total_slots': age_dummy_payload['total_slots'],
        }
    strength_effect_meta = {'enabled': False}
    if strength_split_payload and strength_split_payload.get('enabled'):
        strength_effect_meta, idx = register_strength_split_features(player_to_col, sorted(strength_split_payload['all_players']), idx)
    strength_sensitivity_meta = {'enabled': False}
    if strength_sensitivity_payload and strength_sensitivity_payload.get('enabled'):
        strength_sensitivity_meta, idx = register_strength_sensitivity_features(
            player_to_col,
            sorted(strength_sensitivity_payload['all_players']),
            idx,
        )
        strength_sensitivity_meta['feature_scale'] = strength_sensitivity_feature_scale
    n_features = idx

    # Extract player columns as numpy arrays for speed
    a_players = df[[f'a{k}' for k in range(1, 6)]].values.astype(int)
    h_players = df[[f'h{k}' for k in range(1, 6)]].values.astype(int)
    home_poss = df['home_poss'].values
    pts = df['pts'].values.astype(np.float64)
    score_diff = df['score_diff'].values.astype(np.float64)
    periods = df['period'].values.astype(int)

    # Cap periods at 4 (overtime = Q4)
    periods = np.clip(periods, 1, 4)

    # Determine offense/defense
    offense_players = np.where(home_poss[:, None] == 1, h_players, a_players)
    defense_players = np.where(home_poss[:, None] == 1, a_players, h_players)

    # Build sparse matrix using COO format
    row_indices = []
    col_indices = []
    signed_row_indices = []
    signed_col_indices = []
    signed_values = []

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

    append_strength_split_entries(
        row_indices,
        col_indices,
        offense_players,
        defense_players,
        player_to_col,
        strength_split_payload,
    )

    append_strength_sensitivity_entries(
        signed_row_indices,
        signed_col_indices,
        signed_values,
        offense_players,
        defense_players,
        player_to_col,
        strength_sensitivity_payload,
        strength_sensitivity_feature_scale,
    )

    # Process rubberband features (vectorized per bin)
    for bin_name, low, high in RUBBERBAND_BINS:
        # Determine which rows fall in this bin
        if low is None:
            mask = score_diff <= high
        elif high is None:
            mask = score_diff > low
        else:
            mask = (score_diff > low) & (score_diff <= high)

        bin_rows = np.where(mask)[0]
        if len(bin_rows) == 0:
            continue

        bin_periods = periods[bin_rows]
        for period in range(1, 5):
            feat_name = f"{bin_name}_period{period}"
            col_idx = player_to_col[feat_name]
            period_mask = bin_periods == period
            period_rows = bin_rows[period_mask]
            for r in period_rows:
                row_indices.append(r)
                col_indices.append(col_idx)

    # Process season fixed effects
    if season_effect_meta['enabled']:
        season_values = df['season'].values.astype(int)
        phase_values = df['season_phase'].astype(str).values
        for feat_name in season_effect_meta['feature_names']:
            season_str, phase_str = feat_name.replace("season_fe_", "", 1).rsplit("_", 1)
            season = int(season_str)
            phase = phase_str.upper()
            season_rows = np.where((season_values == season) & (phase_values == phase))[0]
            if len(season_rows) == 0:
                continue
            row_indices.extend(season_rows.tolist())
            col_indices.extend([player_to_col[feat_name]] * len(season_rows))

    if age_effect_meta['enabled']:
        h_age_bins = age_dummy_payload['h_age_bins']
        a_age_bins = age_dummy_payload['a_age_bins']
        offense_age_bins = np.where(home_poss[:, None] == 1, h_age_bins, a_age_bins)
        defense_age_bins = np.where(home_poss[:, None] == 1, a_age_bins, h_age_bins)
        append_age_dummy_entries(
            signed_row_indices,
            signed_col_indices,
            signed_values,
            offense_age_bins,
            defense_age_bins,
            player_to_col,
            age_effect_meta,
        )

    # Create sparse matrix
    rows = np.array(row_indices + signed_row_indices, dtype=np.int32)
    cols = np.array(col_indices + signed_col_indices, dtype=np.int32)
    data = np.concatenate([
        np.ones(len(row_indices), dtype=np.float64),
        np.array(signed_values, dtype=np.float64),
    ])
    X = coo_matrix((data, (rows, cols)),
                   shape=(n_samples, n_features), dtype=np.float64)

    n_season_features = len(season_effect_meta['feature_names'])
    n_age_features = len(age_effect_meta.get('feature_names', []))
    n_strength_features = len(strength_effect_meta.get('feature_names', []))
    n_strength_sensitivity_features = len(strength_sensitivity_meta.get('feature_names', []))
    logging.info(
        "Design matrix shape: %s (%d player + %d rubberband + %d season + %d age + %d strength + %d sensitivity)",
        X.shape,
        n_player_features,
        n_rb_features,
        n_season_features,
        n_age_features,
        n_strength_features,
        n_strength_sensitivity_features,
    )
    return X.tocsr(), pts, {
        'season_effects': season_effect_meta,
        'age_dummies': age_effect_meta,
        'strength_splits': strength_effect_meta,
        'strength_sensitivity': strength_sensitivity_meta,
    }

###############################################################################
# 4) SIMPLIFIED ALTERNATING RIDGE REGRESSION
###############################################################################
def simplified_alternating_rapm(X, y, player_to_col, all_players,
                                 alpha_offense=3000, alpha_defense=3000,
                                 max_iter=200, tol=1e-4,
                                 off_prior=None, def_prior=None, weights=None,
                                 rubberband=False, alpha_rb=None,
                                 season_effects=False,
                                 age_dummies=False):
    """
    Alternating minimization for RAPM with optional rubberband block.

    Args:
        off_prior: Optional dict mapping player_id (str) to offensive prior value.
        def_prior: Optional dict mapping player_id (str) to defensive prior value.
        weights: Optional array of row weights for time decay.
        rubberband: If True, add third alternating block for rubberband features.
        alpha_rb: Regularization for rubberband features (default: 10).
        season_effects: If True, estimate separate season dummy coefficients.
        age_dummies: If True, estimate signed age-bin nuisance coefficients.
    """
    prior_parts = []
    if off_prior is not None:
        prior_parts.append("offense prior")
    if def_prior is not None:
        prior_parts.append("defense prior")
    prior_msg = f" with {' and '.join(prior_parts)}" if prior_parts else ""
    weight_msg = " with time decay" if weights is not None else ""
    rb_msg = " with rubberband" if rubberband else ""
    season_msg = " with season effects" if season_effects else ""
    age_msg = " with age fixed effects" if age_dummies else ""
    logging.info(f"Running alternating Ridge regression{prior_msg}{weight_msg}{rb_msg}{season_msg}{age_msg} (using {N_CORES} cores)...")
    
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

    # Identify rubberband indices if enabled
    rb_indices = np.array([], dtype=int)
    if rubberband:
        rb_indices = np.array([v for k, v in player_to_col.items() if k.startswith('rb_')], dtype=int)
        actual_alpha_rb = alpha_rb if alpha_rb is not None else 10
        logging.info(f"Rubberband: {len(rb_indices)} features, alpha_rb={actual_alpha_rb}")

    season_indices = np.array([], dtype=int)
    if season_effects:
        season_indices = np.array([v for k, v in player_to_col.items() if k.startswith('season_fe_')], dtype=int)
        if len(season_indices) > 0:
            logging.info(f"Season effects: {len(season_indices)} features")

    age_indices = np.array([], dtype=int)
    if age_dummies:
        age_indices = np.array(
            [
                v for k, v in player_to_col.items()
                if k.startswith('age_off_fe_') or k.startswith('age_def_fe_')
            ],
            dtype=int,
        )
        if len(age_indices) > 0:
            logging.info(f"Age fixed effects: {len(age_indices)} features")

    X_off = X[:, offense_indices]
    X_def = X[:, defense_indices]
    X_season = X[:, season_indices] if len(season_indices) > 0 else None
    X_age = X[:, age_indices] if len(age_indices) > 0 else None
    col_to_feature = {v: k for k, v in player_to_col.items()}

    def build_prior_vector(prior, indices, side):
        if prior is None:
            return None
        suffix = f"_{side}"
        values = []
        for col_idx in indices:
            feature_name = col_to_feature[col_idx]
            if feature_name.endswith(suffix) and "_facing_" not in feature_name:
                player_id = feature_name[:-len(suffix)]
                values.append(prior.get(player_id, 0.0))
            else:
                values.append(0.0)
        return np.array(values, dtype=np.float64)

    off_prior_vec = None
    def_prior_vec = None
    if off_prior is not None:
        off_prior_vec = build_prior_vector(off_prior, offense_indices, "off")
    if def_prior is not None:
        def_prior_vec = build_prior_vector(def_prior, defense_indices, "def")

    # Use solver='lsqr' for sparse matrices - faster than default 'auto'
    ridge_off = Ridge(alpha=alpha_offense, fit_intercept=False, solver='lsqr')
    ridge_def = Ridge(alpha=alpha_defense, fit_intercept=False, solver='lsqr')

    if rubberband and len(rb_indices) > 0:
        X_rb = X[:, rb_indices]
        ridge_rb = Ridge(alpha=actual_alpha_rb, fit_intercept=False, solver='lsqr')
    season_model = LinearRegression(fit_intercept=False) if len(season_indices) > 0 else None
    age_model = LinearRegression(fit_intercept=False) if len(age_indices) > 0 else None
    
    residual = y.copy()
    
    for iteration in range(max_iter):
        beta_prev = beta.copy()
        
        # Update offense coefficients
        residual += X_off @ beta[offense_indices]
        if off_prior_vec is not None:
            adjusted_residual = residual - X_off @ off_prior_vec
            ridge_off.fit(X_off, adjusted_residual)
            beta_off = ridge_off.coef_ + off_prior_vec
        else:
            ridge_off.fit(X_off, residual)
            beta_off = ridge_off.coef_
        beta[offense_indices] = beta_off
        residual -= X_off @ beta_off
        
        # Update defense coefficients
        residual += X_def @ beta[defense_indices]

        if def_prior_vec is not None:
            # Adjust target: subtract X_def @ prior from residual
            # This transforms: min ||residual - X_def @ β||² + α||β - prior||²
            # Into: min ||adjusted_residual - X_def @ β*||² + α||β*||²
            # Where β = β* + prior
            adjusted_residual = residual - X_def @ def_prior_vec
            ridge_def.fit(X_def, adjusted_residual)
            # Add prior back to get final coefficients
            beta_def = ridge_def.coef_ + def_prior_vec
        else:
            ridge_def.fit(X_def, residual)
            beta_def = ridge_def.coef_

        beta[defense_indices] = beta_def
        residual -= X_def @ beta_def

        # Update season fixed effects (unpenalized nuisance block)
        if len(season_indices) > 0:
            residual += X_season @ beta[season_indices]
            season_model.fit(X_season, residual)
            beta_season = season_model.coef_
            beta[season_indices] = beta_season
            residual -= X_season @ beta_season

        # Update age fixed effects (unpenalized nuisance block)
        if len(age_indices) > 0:
            residual += X_age @ beta[age_indices]
            age_model.fit(X_age, residual)
            beta_age = age_model.coef_
            beta[age_indices] = beta_age
            residual -= X_age @ beta_age

        # Update rubberband coefficients (third block)
        if rubberband and len(rb_indices) > 0:
            residual += X_rb @ beta[rb_indices]
            ridge_rb.fit(X_rb, residual)
            beta_rb = ridge_rb.coef_
            beta[rb_indices] = beta_rb
            residual -= X_rb @ beta_rb

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


def select_cv_alphas(X, y_centered, groups, player_to_col, all_players,
                     off_prior=None, def_prior=None, weights=None,
                     rubberband=False, alpha_rb=None,
                     season_effects=False,
                     age_dummies=False,
                     off_alpha_seed=3000, def_alpha_seed=3000,
                     alpha_grid=None, cv_folds=5, cv_passes=2,
                     max_iter=100, tol=1e-4):
    """
    Coordinate-search offense and defense alphas with cross-validation.

    Returns:
        (best_off_alpha, best_def_alpha, summary_df)
    """
    alpha_candidates = parse_alpha_grid(alpha_grid)
    splits, split_type = build_cv_splits(groups, cv_folds)

    summary_rows = []
    current_off = float(off_alpha_seed)
    current_def = float(def_alpha_seed)

    def evaluate_axis(axis, candidate_alpha, fixed_off, fixed_def):
        fold_losses = []
        for fold_idx, (train_idx, val_idx) in enumerate(splits, start=1):
            train_weights = weights[train_idx] if weights is not None else None
            beta = simplified_alternating_rapm(
                X[train_idx],
                y_centered[train_idx],
                player_to_col,
                all_players,
                alpha_offense=candidate_alpha if axis == 'off' else fixed_off,
                alpha_defense=candidate_alpha if axis == 'def' else fixed_def,
                max_iter=max_iter,
                tol=tol,
                off_prior=off_prior,
                def_prior=def_prior,
                weights=train_weights,
                rubberband=rubberband,
                alpha_rb=alpha_rb,
                season_effects=season_effects,
                age_dummies=age_dummies,
            )
            y_pred = X[val_idx] @ beta
            val_weights = weights[val_idx] if weights is not None else None
            loss = weighted_mse(y_centered[val_idx], y_pred, weights=val_weights)
            fold_losses.append(loss)
            summary_rows.append({
                'axis': axis,
                'pass': pass_idx,
                'candidate_alpha': float(candidate_alpha),
                'fixed_off_alpha': float(fixed_off),
                'fixed_def_alpha': float(fixed_def),
                'fold': fold_idx,
                'split_type': split_type,
                'loss': float(loss),
            })
        return float(np.mean(fold_losses))

    logging.info(
        "Starting alpha CV over grid %s with seeds off=%s def=%s",
        alpha_candidates,
        off_alpha_seed,
        def_alpha_seed,
    )

    for pass_idx in range(1, max(1, int(cv_passes)) + 1):
        logging.info(
            "Alpha CV pass %d/%d (current off=%s, def=%s)",
            pass_idx,
            max(1, int(cv_passes)),
            current_off,
            current_def,
        )

        off_scores = {
            candidate: evaluate_axis('off', candidate, fixed_off=current_off, fixed_def=current_def)
            for candidate in alpha_candidates
        }
        best_off = min(off_scores, key=off_scores.get)
        logging.info("Best offensive alpha on pass %d: %s (loss=%.8f)", pass_idx, best_off, off_scores[best_off])

        def_scores = {
            candidate: evaluate_axis('def', candidate, fixed_off=best_off, fixed_def=current_def)
            for candidate in alpha_candidates
        }
        best_def = min(def_scores, key=def_scores.get)
        logging.info("Best defensive alpha on pass %d: %s (loss=%.8f)", pass_idx, best_def, def_scores[best_def])

        if best_off == current_off and best_def == current_def:
            logging.info("Alpha CV converged after pass %d", pass_idx)
            current_off = best_off
            current_def = best_def
            break

        current_off = best_off
        current_def = best_def

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        aggregate = (
            summary_df
            .groupby(['axis', 'pass', 'candidate_alpha', 'fixed_off_alpha', 'fixed_def_alpha'], as_index=False)
            .agg(mean_loss=('loss', 'mean'))
        )
        summary_df = summary_df.merge(
            aggregate,
            on=['axis', 'pass', 'candidate_alpha', 'fixed_off_alpha', 'fixed_def_alpha'],
            how='left',
        )

    return float(current_off), float(current_def), summary_df

###############################################################################
# HELPER: FILE LOADING FUNCTIONS
###############################################################################
def _load_single_file(input_file, pure, prefix=None):
    """Load and prepare a single file. Returns None if file can't be loaded."""
    if not Path(input_file).exists():
        return None

    try:
        df = pd.read_parquet(input_file)
        season_phase = 'PS' if Path(input_file).stem.endswith('_PS') else 'RS'
        df['season_phase'] = season_phase
        season_end_year = parse_processed_season_end_year(input_file)
        if season_end_year is not None:
            df['source_season_end_year'] = season_end_year
        if prefix not in FIRST_CHANCE_ALIAS_PREFIXES:
            df = detect_file_type_and_prepare(df, pure=pure, prefix=prefix)
        return (input_file, df, len(df))
    except Exception as e:
        return None

def _load_files_parallel(input_files, pure, prefix=None):
    """Load multiple files in parallel using multiprocessing."""
    from multiprocessing import Pool

    with Pool(processes=N_CORES) as pool:
        load_func = partial(_load_single_file, pure=pure, prefix=prefix)
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

def _load_files_sequential(input_files, pure, prefix=None):
    """Load files sequentially (for single file or fallback)."""
    all_dfs = []
    for input_file in input_files:
        if not Path(input_file).exists():
            logging.warning(f"File not found, skipping: {input_file}")
            continue

        logging.info(f"Loading {input_file}...")
        try:
            df = pd.read_parquet(input_file)
            season_phase = 'PS' if Path(input_file).stem.endswith('_PS') else 'RS'
            df['season_phase'] = season_phase
            season_end_year = parse_processed_season_end_year(input_file)
            if season_end_year is not None:
                df['source_season_end_year'] = season_end_year
            logging.info(f"  Loaded {len(df)} possessions from {input_file}")
            if prefix not in FIRST_CHANCE_ALIAS_PREFIXES:
                df = detect_file_type_and_prepare(df, pure=pure, prefix=prefix)
            all_dfs.append(df)
        except Exception as e:
            logging.warning(f"Error loading {input_file}: {e}. Skipping.")
            continue

    return all_dfs

###############################################################################
# HELPER: RUBBERBAND ANALYSIS OUTPUT
###############################################################################
def save_rubberband_analysis(beta, player_to_col, output_dir, prefix):
    """
    Save rubberband coefficients and effects analysis.

    Args:
        beta: Full coefficient vector
        player_to_col: Mapping of feature names to column indices
        output_dir: Directory to save results
        prefix: Filename prefix (e.g., 'rapm_26_all_pure_rb')
    """
    # Extract rubberband coefficients
    rb_data = []
    for k, v in player_to_col.items():
        if k.startswith('rb_'):
            # Parse bin and period from feature name like "rb_neg3_period1"
            parts = k.rsplit('_period', 1)
            bin_name = parts[0]
            period = int(parts[1])
            coef = beta[v] * 100  # Scale to per-100-possessions
            rb_data.append({
                'feature': k,
                'bin': bin_name,
                'period': period,
                'coefficient': round(coef, 4)
            })

    rb_df = pd.DataFrame(rb_data)
    rb_df = rb_df.sort_values(['period', 'bin']).reset_index(drop=True)

    # Save coefficients
    coef_file = output_dir / f"{prefix}_rubberband_coefficients.csv"
    rb_df.to_csv(coef_file, index=False)
    logging.info(f"Saved rubberband coefficients to {coef_file}")

    # Build effects summary: predicted effect at representative margins for each period
    margins = [-25, -20, -15, -12, -8, -3, 0, 3, 8, 12, 15, 20, 25]
    effects_rows = []
    for margin in margins:
        for period in range(1, 5):
            # Determine which bin this margin falls into
            effect = 0.0
            for bin_name, low, high in RUBBERBAND_BINS:
                in_bin = False
                if low is None:
                    in_bin = margin <= high
                elif high is None:
                    in_bin = margin > low
                else:
                    in_bin = margin > low and margin <= high
                if in_bin:
                    feat_name = f"{bin_name}_period{period}"
                    if feat_name in player_to_col:
                        effect = beta[player_to_col[feat_name]] * 100
                    break
            effects_rows.append({
                'score_margin': margin,
                'period': period,
                'effect_per_100': round(effect, 4)
            })

    effects_df = pd.DataFrame(effects_rows)
    effects_file = output_dir / f"{prefix}_rubberband_effects.csv"
    effects_df.to_csv(effects_file, index=False)
    logging.info(f"Saved rubberband effects to {effects_file}")

    # Log summary table
    logging.info("\nRubberband Effects (per 100 possessions, relative to close game):")
    logging.info(f"{'Margin':>8s}  {'Q1':>8s}  {'Q2':>8s}  {'Q3':>8s}  {'Q4':>8s}")
    logging.info("-" * 44)
    for margin in margins:
        effects = []
        for period in range(1, 5):
            row = effects_df[(effects_df['score_margin'] == margin) & (effects_df['period'] == period)]
            effects.append(f"{row['effect_per_100'].values[0]:+8.3f}")
        logging.info(f"{margin:>+8d}  {'  '.join(effects)}")

    return rb_df, effects_df


def save_season_effect_analysis(beta, player_to_col, season_effect_meta, output_dir, prefix):
    """
    Save season fixed-effect coefficients, relative to the omitted reference season-phase.
    """
    if not season_effect_meta.get('enabled'):
        return None

    rows = [{
        'season': int(season_effect_meta['reference_season']),
        'season_phase': str(season_effect_meta['reference_phase']),
        'effect_per_100': 0.0,
        'is_reference': True,
    }]

    for feat_name in season_effect_meta['feature_names']:
        season_str, phase_str = feat_name.replace("season_fe_", "", 1).rsplit("_", 1)
        season = int(season_str)
        rows.append({
            'season': season,
            'season_phase': phase_str.upper(),
            'effect_per_100': round(beta[player_to_col[feat_name]] * 100, 4),
            'is_reference': False,
        })

    phase_order = {'RS': 0, 'PS': 1}
    season_df = pd.DataFrame(rows)
    season_df['_phase_order'] = season_df['season_phase'].map(phase_order).fillna(99)
    season_df = season_df.sort_values(['season', '_phase_order']).drop(columns=['_phase_order']).reset_index(drop=True)
    output_path = output_dir / f"{prefix}_season_effects.csv"
    season_df.to_csv(output_path, index=False)
    logging.info(f"Saved season fixed effects to {output_path}")
    return season_df


def save_age_effect_analysis(beta, player_to_col, age_effect_meta, output_dir, prefix):
    """Save offense, defense, and net age fixed effects."""
    if not age_effect_meta.get('enabled'):
        return None

    reference_age = int(age_effect_meta['reference_age'])
    off_raw_effects = {}
    def_raw_effects = {}
    for age in age_effect_meta['ages']:
        if age == reference_age:
            off_raw_effects[age] = 0.0
            def_raw_effects[age] = 0.0
        else:
            off_raw_effects[age] = float(beta[player_to_col[f"age_off_fe_{age}"]] * 100.0)
            def_raw_effects[age] = float(beta[player_to_col[f"age_def_fe_{age}"]] * 100.0)

    off_weights = np.array([age_effect_meta['off_slot_counts'].get(age, 0) for age in age_effect_meta['ages']], dtype=np.float64)
    off_values = np.array([off_raw_effects[age] for age in age_effect_meta['ages']], dtype=np.float64)
    off_center = float(np.average(off_values, weights=off_weights)) if off_weights.sum() > 0 else float(off_values.mean())

    def_weights = np.array([age_effect_meta['def_slot_counts'].get(age, 0) for age in age_effect_meta['ages']], dtype=np.float64)
    def_values = np.array([def_raw_effects[age] for age in age_effect_meta['ages']], dtype=np.float64)
    def_center = float(np.average(def_values, weights=def_weights)) if def_weights.sum() > 0 else float(def_values.mean())

    rows = []
    for age in age_effect_meta['ages']:
        off_slots = int(age_effect_meta['off_slot_counts'].get(age, 0))
        def_slots = int(age_effect_meta['def_slot_counts'].get(age, 0))
        total_slots = int(age_effect_meta['total_slot_counts'].get(age, 0))
        off_raw = off_raw_effects[age]
        def_raw = def_raw_effects[age]
        off_centered = off_raw_effects[age] - off_center
        def_centered = def_raw_effects[age] - def_center
        net_centered = off_centered - def_centered
        if off_slots == 0:
            off_raw = np.nan
            off_centered = np.nan
        if def_slots == 0:
            def_raw = np.nan
            def_centered = np.nan
        if total_slots == 0 or pd.isna(off_centered) or pd.isna(def_centered):
            net_centered = np.nan
        rows.append({
            'age': int(age),
            'off_raw_effect_per_100': round(off_raw, 4) if pd.notna(off_raw) else np.nan,
            'off_centered_effect_per_100': round(off_centered, 4) if pd.notna(off_centered) else np.nan,
            'def_raw_effect_per_100': round(def_raw, 4) if pd.notna(def_raw) else np.nan,
            'def_centered_effect_per_100': round(def_centered, 4) if pd.notna(def_centered) else np.nan,
            'net_centered_effect_per_100': round(net_centered, 4) if pd.notna(net_centered) else np.nan,
            'off_slots': off_slots,
            'def_slots': def_slots,
            'total_slots': total_slots,
            'is_reference': age == reference_age,
        })

    age_df = pd.DataFrame(rows).sort_values('age').reset_index(drop=True)
    output_path = output_dir / f"{prefix}_age_effects.csv"
    age_df.to_csv(output_path, index=False)
    logging.info(f"Saved age fixed effects to {output_path}")
    return age_df


def _clean_player_name(name):
    """Normalize player-name candidates and reject placeholder labels."""
    if pd.isna(name):
        return None

    cleaned = " ".join(str(name).strip().split())
    if not cleaned or cleaned.lower() in {"nan", "none"}:
        return None
    if cleaned.startswith("ID_"):
        return None
    return cleaned


def _normalize_player_id(player_id):
    """Normalize player IDs to the string format used by RAPM outputs."""
    if pd.isna(player_id):
        return None
    try:
        return str(int(float(player_id)))
    except (TypeError, ValueError):
        return None


def _player_name_quality(name):
    """Prefer full multi-token names over truncated labels."""
    cleaned = _clean_player_name(name)
    if cleaned is None:
        return (-1, -1)
    return (len(cleaned.split()), len(cleaned))


def _store_player_name(name_map, player_id, player_name):
    """Keep the best known name for each player ID."""
    normalized_id = _normalize_player_id(player_id)
    cleaned_name = _clean_player_name(player_name)
    if normalized_id is None or cleaned_name is None:
        return

    current_name = name_map.get(normalized_id)
    if current_name is None or _player_name_quality(cleaned_name) > _player_name_quality(current_name):
        name_map[normalized_id] = cleaned_name


def _ingest_name_frame(name_map, frame, id_col, name_col, allowed_ids=None):
    """Merge player-name candidates from a dataframe into the working map."""
    if frame is None or frame.empty or id_col not in frame.columns or name_col not in frame.columns:
        return

    subset = frame[[id_col, name_col]].dropna(subset=[id_col, name_col]).copy()
    if allowed_ids is not None:
        normalized_ids = subset[id_col].map(_normalize_player_id)
        subset = subset[normalized_ids.isin(allowed_ids)].copy()

    for player_id, player_name in subset.itertuples(index=False, name=None):
        _store_player_name(name_map, player_id, player_name)


def _load_neighbor_identity_names(allowed_ids=None):
    """Load supplemental names from the main pipeline repo when available."""
    neighbor_root = PROJECT_ROOT.parent / "2026_NBA_PIPELINE"
    candidate_paths = [
        neighbor_root / "nba_rapm" / "merged_rs_identity_cols.csv",
        neighbor_root / "csvs" / "merged_rs_full.csv.gz",
        neighbor_root / "csvs" / "merged_rs_full.csv",
    ]

    name_map = {}
    for path in candidate_paths:
        if not path.exists():
            continue

        try:
            frame = pd.read_csv(
                path,
                usecols=lambda column: column in {"nba_id", "player_name", "ShortName"},
                low_memory=False,
            )
        except Exception as exc:
            logging.warning(f"Could not read supplemental player names from {path}: {exc}")
            continue

        name_col = "player_name" if "player_name" in frame.columns else "ShortName"
        _ingest_name_frame(name_map, frame, "nba_id", name_col, allowed_ids=allowed_ids)

    return name_map


def _load_raw_pbp_names(raw_data_dir, allowed_ids=None):
    """Recover missing names directly from raw PBP parquet event columns."""
    name_map = {}
    raw_files = sorted(raw_data_dir.glob("NBA*.parquet"))
    player_columns = [
        ("player1_id", "player1_name"),
        ("player2_id", "player2_name"),
        ("player3_id", "player3_name"),
    ]

    for path in raw_files:
        try:
            frame = pd.read_parquet(path, columns=[column for pair in player_columns for column in pair])
        except Exception as exc:
            logging.warning(f"Could not read raw player names from {path}: {exc}")
            continue

        for id_col, name_col in player_columns:
            _ingest_name_frame(name_map, frame, id_col, name_col, allowed_ids=allowed_ids)

    return name_map


def build_player_name_map(player_ids, name_map_file):
    """Build the best available player-name map for the current RAPM run."""
    allowed_ids = {_normalize_player_id(player_id) for player_id in player_ids}
    allowed_ids.discard(None)

    name_map = {}
    name_map_path = Path(name_map_file)

    logging.info(f"Loading primary player-name map from {name_map_path}...")
    try:
        primary = pd.read_csv(name_map_path)
        _ingest_name_frame(name_map, primary, "nba_id", "player_name", allowed_ids=allowed_ids)
        logging.info(f"Loaded {len(name_map)} player names from the primary autocomplete map")
    except Exception as exc:
        logging.warning(f"Could not load primary player-name map: {exc}")

    unresolved_ids = sorted(allowed_ids - set(name_map))
    if unresolved_ids:
        supplemental = _load_neighbor_identity_names(allowed_ids=set(unresolved_ids))
        for player_id, player_name in supplemental.items():
            _store_player_name(name_map, player_id, player_name)
        logging.info(
            "Supplemented %d player names from the 2026_NBA_PIPELINE identity sources",
            len(supplemental),
        )

    unresolved_ids = sorted(allowed_ids - set(name_map))
    if unresolved_ids:
        raw_names = _load_raw_pbp_names(PIPELINE_ROOT / "raw_data", allowed_ids=set(unresolved_ids))
        for player_id, player_name in raw_names.items():
            _store_player_name(name_map, player_id, player_name)
        logging.info("Recovered %d player names from raw PBP fallback sources", len(raw_names))

    unresolved_ids = sorted(allowed_ids - set(name_map))
    if unresolved_ids:
        logging.warning(
            "Leaving %d player names unresolved; placeholders remain for IDs such as %s",
            len(unresolved_ids),
            ", ".join(unresolved_ids[:10]),
        )

    return name_map

###############################################################################
# 5) MAIN EXECUTION
###############################################################################
def run_simplified_rapm(input_files, name_map_file=None, pure=False,
                        off_prior_file=None, def_prior_file=None,
                        def_alpha=None, off_alpha=None,
                        cv_alpha=False, cv_folds=5, cv_passes=2, cv_alpha_grid=None,
                        publish_to_rapms=False, publish_name=None,
                        timedecay=False, half_life=None,
                        rubberband=False, alpha_rb=None,
                        season_effects=False,
                        fixed_season_baselines=None,
                        age_dummies=False,
                        age_poly_coefficients=None,
                        age_poly_reference_age=AGE_DUMMY_REFERENCE,
                        age_curve=False,
                        strength_splits=False,
                        strength_legacy_scale=False,
                        strength_sensitivity=False,
                        strength_sensitivity_alpha_mult=4.0,
                        darko_history_path=None,
                        start_year=None, end_year=None,
                        prefix=None):
    """
    Main function to run simplified RAPM analysis.

    Args:
        input_files: Single file path (str) or list of file paths for multi-year analysis
        name_map_file: Path to player name mapping CSV (defaults to project root)
        pure: If True, use Net_Diff for RAPM files (removes 3pt/FT luck adjustments)
        off_prior_file: Path to CSV with offense prior values (player_id, prior_value)
        def_prior_file: Path to CSV with defense prior values (player_id, prior_value)
        def_alpha: Override defense alpha (default: 3000)
        off_alpha: Override offense alpha (default: 3000)
        cv_alpha: If True, cross-validate separate offense/defense alphas
        cv_folds: Number of CV folds for alpha selection
        cv_passes: Number of coordinate-search passes between offense and defense alpha CV
        cv_alpha_grid: Optional alpha grid (iterable or comma-separated string)
        publish_to_rapms: If True, copy the result CSV into master_results and push the downstream rapms repo
        publish_name: Optional published filename override
        timedecay: If True, apply time decay weighting
        half_life: Half-life in days for time decay (default: 700)
        season_effects: If True, add season fixed effects for multi-season runs
        fixed_season_baselines: Optional dict of season end year -> fixed row baseline to subtract from y
        age_dummies: If True, add signed age fixed effects for the requested metric
        age_poly_coefficients: Optional polynomial age curve CSV. Values are applied as fixed row offsets.
        age_poly_reference_age: Reference age for fixed age-poly offsets
        age_curve: If True, build RAPM priors by shrinking toward age mean
        strength_splits: If True, add DARKO opponent strong/not-strong player interactions
        strength_legacy_scale: If True, target-scale strength-split overall offense/defense like the legacy script
        strength_sensitivity: If True, add continuous DARKO opponent-strength slope interactions
        strength_sensitivity_alpha_mult: Slope-feature penalty multiplier implemented by column scaling
        darko_history_path: Path to DARKO DPM history for strength_splits
        prefix: Requested metric prefix for aliased metrics (e.g., 'BADPASS_TOV')
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
    if rubberband:
        mode_str += " - RUBBERBAND"
    if season_effects:
        mode_str += " - SEASON FE"
    if fixed_season_baselines is not None:
        mode_str += " - FIXED SEASON OFFSETS"
    if age_dummies:
        mode_str += " - AGE FE"
    if age_poly_coefficients is not None:
        mode_str += " - FIXED AGE POLY"
    if age_curve:
        mode_str += " - AGE CURVE"
    if strength_splits:
        mode_str += " - DARKO STRENGTH SPLITS"
    if strength_legacy_scale:
        mode_str += " - LEGACY STRENGTH SCALE"
    if strength_sensitivity:
        mode_str += " - DARKO STRENGTH SENSITIVITY"
    if cv_alpha:
        mode_str += " - ALPHA CV"

    logging.info(mode_str)
    logging.info("=" * 60)
    
    # Load and combine data from all files
    # Use parallel loading if multiple files
    if len(input_files) > 1:
        logging.info(f"Using {N_CORES} cores for parallel file loading...")
        all_dfs = _load_files_parallel(input_files, pure, prefix=prefix)
    else:
        all_dfs = _load_files_sequential(input_files, pure, prefix=prefix)
    
    # Check if we loaded any files
    if not all_dfs:
        logging.error("No files were successfully loaded. Exiting.")
        return None
    
    # Combine all dataframes
    df = pd.concat(all_dfs, ignore_index=True)
    logging.info(f"Combined total: {len(df)} possessions across {len(all_dfs)} file(s)")

    if prefix in ALT_BASELINE_PREFIXES:
        alt_baselines = compute_alt_first_chance_baselines(
            df,
            timedecay=timedecay,
            half_life=half_life,
            requested_prefixes=[prefix],
        )
        df = detect_file_type_and_prepare(df, pure=pure, prefix=prefix, alt_baselines=alt_baselines)
    elif prefix in ALT_FIRST_CHANCE_TOV_PREFIXES:
        df = detect_file_type_and_prepare(df, pure=pure, prefix=prefix)
    elif prefix in FIRST_CHANCE_MODE_PREFIXES:
        fc_mode_baselines = compute_first_chance_mode_baselines(
            df,
            timedecay=timedecay,
            half_life=half_life,
        )
        df = detect_file_type_and_prepare(
            df,
            pure=pure,
            prefix=prefix,
            fc_mode_baselines=fc_mode_baselines,
        )
    
    def load_prior(prior_file, label):
        if not prior_file:
            return None
        logging.info(f"Loading {label} prior from {prior_file}...")
        try:
            prior_df = pd.read_csv(prior_file)
            prior = dict(zip(prior_df['player_id'].astype(str),
                             prior_df['prior_value'] / 100))
            logging.info(f"Loaded {label} prior for {len(prior)} players")
            return prior
        except Exception as e:
            logging.warning(f"Could not load {label} prior: {e}")
            return None

    # Load priors if specified
    off_prior = load_prior(off_prior_file, "offense")
    def_prior = load_prior(def_prior_file, "defense")

    # Transform data to home/away format
    df_transformed = transform_to_home_away_format(df)
    
    # Get all unique players
    all_players = set()
    for col in [f'a{i}' for i in range(1, 6)] + [f'h{i}' for i in range(1, 6)]:
        player_ids = pd.to_numeric(df_transformed[col], errors='coerce').fillna(0).astype(int)
        all_players.update(str(pid) for pid in player_ids.unique() if pid > 0)
    all_players = sorted(all_players)
    logging.info(f"Found {len(all_players)} unique players")
    name_map = build_player_name_map(all_players, name_map_file)

    age_dummy_payload = None
    if age_dummies:
        if end_year is None:
            raise ValueError("age_dummies mode requires an explicit end_year.")
        age_source_df = fetch_age_source_darko(
            normalize_season_end_year(end_year),
            darko_history_path if darko_history_path else DEFAULT_DARKO_HISTORY_PATH,
        )
        age_dummy_payload = prepare_age_dummy_payload(age_source_df, df_transformed, all_players)

    if age_curve:
        if start_year is None or end_year is None:
            raise ValueError("age_curve mode requires explicit start_year and end_year.")
        age_source_df = fetch_age_curve_source_supabase(normalize_season_end_year(end_year))
        off_prior, def_prior, _ = build_age_curve_priors(age_source_df, start_year, end_year, all_players)

    strength_split_payload = None
    if strength_splits:
        strength_path = darko_history_path if darko_history_path else DEFAULT_DARKO_HISTORY_PATH
        strength_split_payload = prepare_strength_split_payload(df_transformed, strength_path)
        strength_split_payload['all_players'] = all_players

    strength_sensitivity_payload = None
    strength_sensitivity_feature_scale = 1.0
    if strength_sensitivity:
        strength_path = darko_history_path if darko_history_path else DEFAULT_DARKO_HISTORY_PATH
        strength_sensitivity_payload = prepare_strength_sensitivity_payload(df_transformed, strength_path)
        strength_sensitivity_payload['all_players'] = all_players
        if strength_sensitivity_alpha_mult <= 0:
            raise ValueError("strength_sensitivity_alpha_mult must be positive.")
        strength_sensitivity_feature_scale = float(1.0 / np.sqrt(strength_sensitivity_alpha_mult))
        logging.info(
            "Strength sensitivity slope feature scale %.4f implements %.2fx base ridge penalty",
            strength_sensitivity_feature_scale,
            strength_sensitivity_alpha_mult,
        )
    
    # Build player-to-column mapping
    player_to_col = {}
    idx = 0
    for p in all_players:
        player_to_col[f"{p}_off"] = idx
        idx += 1
        player_to_col[f"{p}_def"] = idx
        idx += 1
    
    # Build design matrix
    if rubberband:
        X, y, feature_meta = build_design_matrix_with_rubberband(
            df_transformed,
            player_to_col,
            season_effects=season_effects,
            age_dummy_payload=age_dummy_payload,
            strength_split_payload=strength_split_payload,
            strength_sensitivity_payload=strength_sensitivity_payload,
            strength_sensitivity_feature_scale=strength_sensitivity_feature_scale,
        )
    else:
        X, y, feature_meta = build_simple_design_matrix(
            df_transformed,
            player_to_col,
            season_effects=season_effects,
            age_dummy_payload=age_dummy_payload,
            strength_split_payload=strength_split_payload,
            strength_sensitivity_payload=strength_sensitivity_payload,
            strength_sensitivity_feature_scale=strength_sensitivity_feature_scale,
        )

    fixed_offset_rows = []
    total_fixed_offset = np.zeros(len(y), dtype=np.float64)

    if fixed_season_baselines is not None:
        season_baselines = {int(k): float(v) for k, v in dict(fixed_season_baselines).items()}
        season_values = pd.to_numeric(df_transformed["season"], errors="coerce")
        season_offsets = season_values.map(season_baselines).fillna(0.0).to_numpy(dtype=np.float64)
        total_fixed_offset += season_offsets
        missing_rows = int((~season_values.astype("Int64").isin(season_baselines.keys())).sum())
        fixed_offset_rows.append({
            "offset_type": "fixed_rs_season_baseline",
            "rows": int(len(y)),
            "missing_rows": missing_rows,
            "mean_offset_per_100": float(season_offsets.mean() * 100.0) if len(season_offsets) else 0.0,
            "min_offset_per_100": float(season_offsets.min() * 100.0) if len(season_offsets) else 0.0,
            "max_offset_per_100": float(season_offsets.max() * 100.0) if len(season_offsets) else 0.0,
            "reference": "RS raw mean by season; same season value used for RS and PS",
        })
        logging.info(
            "Applied fixed RS season baselines to y: mean=%.4f pts/100 min=%.4f max=%.4f missing_rows=%d",
            fixed_offset_rows[-1]["mean_offset_per_100"],
            fixed_offset_rows[-1]["min_offset_per_100"],
            fixed_offset_rows[-1]["max_offset_per_100"],
            missing_rows,
        )

    age_poly_degree = None
    if age_poly_coefficients is not None:
        if prefix is None:
            raise ValueError("age_poly_coefficients requires an explicit metric prefix.")
        if end_year is None:
            raise ValueError("age_poly_coefficients requires an explicit end_year.")
        curves, age_poly_degree = load_age_poly_curves(age_poly_coefficients, prefix)
        age_source_df = fetch_age_source_darko(
            normalize_season_end_year(end_year),
            darko_history_path if darko_history_path else DEFAULT_DARKO_HISTORY_PATH,
        )
        age_lookup = build_age_lookup_from_source(age_source_df)
        age_offsets, age_summary = compute_fixed_age_poly_offsets(
            df_transformed,
            curves,
            age_lookup,
            reference_age=float(age_poly_reference_age),
        )
        total_fixed_offset += age_offsets
        fixed_offset_rows.append({
            "offset_type": f"fixed_age_poly{age_poly_degree}",
            "rows": int(age_summary["rows"]),
            "missing_rows": int(age_summary["missing_slots"]),
            "mean_offset_per_100": float(age_summary["mean_offset_per_100"]),
            "min_offset_per_100": float(age_summary["min_offset_per_100"]),
            "max_offset_per_100": float(age_summary["max_offset_per_100"]),
            "reference": f"curve(age)-curve({float(age_poly_reference_age):g}) summed across O/D slots",
            "mapped_slots": int(age_summary["mapped_slots"]),
            "missing_slots": int(age_summary["missing_slots"]),
            "clipped_slots": int(age_summary["clipped_slots"]),
            "reference_age": float(age_summary["reference_age"]),
        })
        logging.info(
            "Applied fixed age-poly offsets to y: degree=%s ref_age=%.1f mean=%.4f pts/100 min=%.4f max=%.4f mapped_slots=%d missing_slots=%d",
            age_poly_degree,
            float(age_poly_reference_age),
            age_summary["mean_offset_per_100"],
            age_summary["min_offset_per_100"],
            age_summary["max_offset_per_100"],
            int(age_summary["mapped_slots"]),
            int(age_summary["missing_slots"]),
        )

    if fixed_offset_rows:
        y = y - total_fixed_offset
        fixed_offset_rows.append({
            "offset_type": "total_fixed_offset",
            "rows": int(len(y)),
            "missing_rows": np.nan,
            "mean_offset_per_100": float(total_fixed_offset.mean() * 100.0) if len(total_fixed_offset) else 0.0,
            "min_offset_per_100": float(total_fixed_offset.min() * 100.0) if len(total_fixed_offset) else 0.0,
            "max_offset_per_100": float(total_fixed_offset.max() * 100.0) if len(total_fixed_offset) else 0.0,
            "reference": "subtracted from y before centering and solving",
        })

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
    
    cv_summary = None
    # Run alternating Ridge regression
    actual_off_alpha = off_alpha if off_alpha is not None else 3000
    actual_def_alpha = def_alpha if def_alpha is not None else 3000
    if cv_alpha:
        actual_off_alpha, actual_def_alpha, cv_summary = select_cv_alphas(
            X,
            y_centered,
            df_transformed['gameid'].values,
            player_to_col,
            all_players,
            off_prior=off_prior,
            def_prior=def_prior,
            weights=weights,
            rubberband=rubberband,
            alpha_rb=alpha_rb,
            season_effects=season_effects,
            age_dummies=age_dummies,
            off_alpha_seed=actual_off_alpha,
            def_alpha_seed=actual_def_alpha,
            alpha_grid=cv_alpha_grid,
            cv_folds=cv_folds,
            cv_passes=cv_passes,
            max_iter=100,
            tol=1e-4,
        )
        logging.info(
            "Alpha CV selected off_alpha=%s and def_alpha=%s",
            actual_off_alpha,
            actual_def_alpha,
        )
    beta = simplified_alternating_rapm(
        X, y_centered, player_to_col, all_players,
        alpha_offense=actual_off_alpha, alpha_defense=actual_def_alpha,
        max_iter=200, tol=1e-4,
        off_prior=off_prior,
        def_prior=def_prior,
        weights=weights,
        rubberband=rubberband,
        alpha_rb=alpha_rb,
        season_effects=season_effects,
        age_dummies=age_dummies,
    )
    
    # Calculate player possessions (total, offensive, defensive)
    def count_player_slots(frame, columns):
        values = pd.to_numeric(
            pd.Series(frame[columns].to_numpy().ravel()),
            errors='coerce',
        ).dropna().astype(int)
        values = values[values > 0]
        return defaultdict(int, values.astype(str).value_counts().to_dict())

    player_possessions = count_player_slots(
        df_transformed,
        [f'a{k}' for k in range(1, 6)] + [f'h{k}' for k in range(1, 6)],
    )
    player_off_poss = count_player_slots(df, [f'O{k}' for k in range(1, 6)])
    player_def_poss = count_player_slots(df, [f'D{k}' for k in range(1, 6)])
    
    # Re-center offense and defense separately so weighted averages = 0.
    # Component additivity is preserved because every alias uses the same
    # linear centering operator on the same fixed possession universe.
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
    
    strength_offsets = {}
    if strength_split_payload and strength_split_payload.get('enabled'):
        for side in ['off', 'def']:
            counts_lookup = strength_split_payload[f'{side}_counts']
            for bucket in STRENGTH_BUCKETS:
                values = []
                weights_for_bucket = []
                for player_id in all_players:
                    feat_name = f"{player_id}_facing_{bucket}_{side}"
                    count = counts_lookup[player_id][bucket]
                    if feat_name in player_to_col and count > 0:
                        values.append(beta[player_to_col[feat_name]] * 100.0)
                        weights_for_bucket.append(count)
                if weights_for_bucket:
                    strength_offsets[(side, bucket)] = float(np.average(values, weights=weights_for_bucket))
                else:
                    strength_offsets[(side, bucket)] = 0.0
        logging.info(
            "Strength split report offsets: %s",
            {f"{side}_{bucket}": round(value, 4) for (side, bucket), value in strength_offsets.items()},
        )

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

        # Multiply by 100 and keep three decimals so additive decompositions
        # remain visibly tight in exported result CSVs.
        row_result = {
            'player_id': p,
            'player_name': player_name,
            'off': round(off_val * 100, 3),
            'def': round(def_val * 100, 3),
            'net_rapm': round(net_rapm * 100, 3),
            'possessions': poss,
            'off_poss': player_off_poss[p],
            'def_poss': player_def_poss[p]
        }

        if strength_split_payload and strength_split_payload.get('enabled'):
            off_counts = strength_split_payload['off_counts'][p]
            def_counts = strength_split_payload['def_counts'][p]
            overall_values = {}

            for side, base_value, counts_lookup in [
                ('off', off_val * 100.0, off_counts),
                ('def', def_val * 100.0, def_counts),
            ]:
                side_poss = player_off_poss[p] if side == 'off' else player_def_poss[p]
                weighted_overall = 0.0
                for bucket in STRENGTH_BUCKETS:
                    feat_name = f"{p}_facing_{bucket}_{side}"
                    raw_coeff = beta[player_to_col[feat_name]] * 100.0 if feat_name in player_to_col else 0.0
                    centered_coeff = raw_coeff - strength_offsets.get((side, bucket), 0.0)
                    faced_count = counts_lookup[bucket]
                    bucket_pct = faced_count / side_poss if side_poss else 0.0
                    vs_value = base_value + centered_coeff

                    row_result[f'facing_{bucket}_{side}'] = round(centered_coeff, 2)
                    row_result[f'faced_{bucket}_{side}'] = int(faced_count)
                    row_result[f'{bucket}_perc_{side}'] = round(bucket_pct, 4)
                    row_result[f'{side}_vs_{bucket}'] = round(vs_value, 2)
                    weighted_overall += bucket_pct * vs_value

                row_result[f'overall_{side}'] = round(weighted_overall, 2)
                overall_values[side] = weighted_overall

            row_result['base_net_rapm'] = row_result['net_rapm']
            row_result['overall_net_rapm'] = round(overall_values['off'] - overall_values['def'], 2)
            row_result['net_rapm'] = row_result['overall_net_rapm']

        if strength_sensitivity_payload and strength_sensitivity_payload.get('enabled'):
            off_slope_key = f"{p}_strength_slope_off"
            def_slope_key = f"{p}_strength_slope_def"
            off_slope = (
                beta[player_to_col[off_slope_key]] * strength_sensitivity_feature_scale * 100.0
                if off_slope_key in player_to_col
                else 0.0
            )
            def_slope = (
                beta[player_to_col[def_slope_key]] * strength_sensitivity_feature_scale * 100.0
                if def_slope_key in player_to_col
                else 0.0
            )
            base_off = off_val * 100.0
            base_def = def_val * 100.0
            base_net = base_off - base_def
            net_slope = off_slope - def_slope

            off_poss = player_off_poss[p]
            def_poss = player_def_poss[p]
            observed_off_z = (
                strength_sensitivity_payload['off_exposure_sum'][p] / off_poss
                if off_poss
                else 0.0
            )
            observed_def_z = (
                strength_sensitivity_payload['def_exposure_sum'][p] / def_poss
                if def_poss
                else 0.0
            )

            row_result.update({
                'off_strength_slope': round(off_slope, 2),
                'def_strength_slope': round(def_slope, 2),
                'net_strength_slope': round(net_slope, 2),
                'net_vs_weak_1sd': round(base_net - net_slope, 2),
                'net_vs_avg_strength': round(base_net, 2),
                'net_vs_strong_1sd': round(base_net + net_slope, 2),
                'strong_minus_weak': round(2.0 * net_slope, 2),
                'weak_minus_strong_drop': round(-2.0 * net_slope, 2),
                'observed_opp_def_strength_z': round(observed_off_z, 4),
                'observed_opp_off_strength_z': round(observed_def_z, 4),
                'strength_sensitivity_alpha_mult': round(float(strength_sensitivity_alpha_mult), 4),
            })

        results.append(row_result)
    
    results_df = pd.DataFrame(results)

    if strength_legacy_scale and strength_split_payload and strength_split_payload.get('enabled'):
        target_stdevs = {'off': 1.48, 'def': 1.36}
        for side in ['off', 'def']:
            poss_col = f'{side}_poss'
            overall_col = f'overall_{side}'
            weights_for_side = pd.to_numeric(results_df[poss_col], errors='coerce').fillna(0.0).clip(lower=0.0)
            overall_values = pd.to_numeric(results_df[overall_col], errors='coerce').fillna(0.0)
            valid = weights_for_side > 0

            if valid.any():
                weighted_avg = float(np.average(overall_values[valid], weights=weights_for_side[valid]))
            else:
                weighted_avg = 0.0

            centered_overall = overall_values - weighted_avg
            actual_stdev = float(centered_overall.std())
            scaling_factor = target_stdevs[side] / actual_stdev if actual_stdev > 0 else 1.0
            scaled_overall = centered_overall * scaling_factor
            side_delta = scaled_overall - overall_values

            results_df[f'{side}_legacy_scale_factor'] = round(scaling_factor, 6)
            results_df[f'{side}_legacy_center'] = round(weighted_avg, 6)
            results_df[f'{side}_legacy_diff'] = side_delta.round(2)
            results_df[f'scaled_overall_{side}'] = scaled_overall.round(2)

            for bucket in STRENGTH_BUCKETS:
                source_col = f'{side}_vs_{bucket}'
                if source_col in results_df.columns:
                    results_df[f'scaled_{side}_vs_{bucket}'] = (
                        pd.to_numeric(results_df[source_col], errors='coerce').fillna(0.0) + side_delta
                    ).round(2)

        results_df['scaled_overall_net_rapm'] = (
            results_df['scaled_overall_off'] - results_df['scaled_overall_def']
        ).round(2)
        results_df['net_rapm'] = results_df['scaled_overall_net_rapm']

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
            yy = int(year_str[-2:])
            years.append((normalize_season_end_year(yy), f"{yy:02d}"))
    
    # Override file_type for aliased prefixes (e.g., BADPASS_TOV loads TOV files)
    if prefix and prefix.lower() != file_type:
        file_type = prefix.lower()

    # Determine season type suffix
    if has_ps and has_rs:
        season_suffix = "all"
    elif has_ps:
        season_suffix = "ps"
    else:
        season_suffix = "rs"
    
    # Build filename with start_end year format
    if years:
        years_sorted = sorted(set(years), key=lambda item: item[0])
        start_year = years_sorted[0][1]
        end_year = years_sorted[-1][1]

        if start_year == end_year:
            year_range = start_year
        else:
            year_range = f"{start_year}_{end_year}"

        # Build suffix based on options
        suffix = build_output_suffix(
            season_suffix,
            pure=pure,
            rubberband=rubberband,
            season_effects=season_effects,
            fixed_season_effects=fixed_season_baselines is not None,
            age_dummies=age_dummies,
            age_poly_degree=age_poly_degree,
            alpha_cv=cv_alpha,
            off_alpha=actual_off_alpha,
            def_alpha=actual_def_alpha,
            age_curve=age_curve,
            off_prior=off_prior,
            def_prior=def_prior,
            timedecay=timedecay,
            half_life=half_life,
            strength_splits=strength_splits,
            strength_legacy_scale=strength_legacy_scale,
            strength_sensitivity=strength_sensitivity,
        )
        # Use _td subfolder for time decay results
        if timedecay:
            output_dir = RESULTS_DIR / "td"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"{file_type}_{year_range}_{suffix}_results.csv"
        else:
            output_file = RESULTS_DIR / f"{file_type}_{year_range}_{suffix}_results.csv"
    else:
        # Fallback
        suffix = build_output_suffix(
            season_suffix,
            pure=pure,
            rubberband=rubberband,
            season_effects=season_effects,
            fixed_season_effects=fixed_season_baselines is not None,
            age_dummies=age_dummies,
            age_poly_degree=age_poly_degree,
            alpha_cv=cv_alpha,
            off_alpha=actual_off_alpha,
            def_alpha=actual_def_alpha,
            age_curve=age_curve,
            off_prior=off_prior,
            def_prior=def_prior,
            timedecay=timedecay,
            half_life=half_life,
            strength_splits=strength_splits,
            strength_legacy_scale=strength_legacy_scale,
            strength_sensitivity=strength_sensitivity,
        )
        if timedecay:
            output_dir = RESULTS_DIR / "td"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"{file_type}_{suffix}_results.csv"
        else:
            output_file = RESULTS_DIR / f"{file_type}_{suffix}_results.csv"

    results_df.to_csv(output_file, index=False)
    logging.info(f"Saved results to {output_file}")
    if fixed_offset_rows:
        fixed_offsets_file = output_file.with_name(output_file.stem.replace("_results", "_fixed_offsets") + ".csv")
        pd.DataFrame(fixed_offset_rows).to_csv(fixed_offsets_file, index=False)
        logging.info(f"Saved fixed offset summary to {fixed_offsets_file}")
    if cv_summary is not None and not cv_summary.empty:
        alpha_cv_file = output_file.with_name(output_file.stem.replace("_results", "_alpha_cv") + ".csv")
        cv_summary.to_csv(alpha_cv_file, index=False)
        logging.info(f"Saved alpha CV summary to {alpha_cv_file}")
    if publish_to_rapms:
        local_publish_path, rapms_publish_path = publish_result_to_rapms(output_file, publish_name=publish_name)
        logging.info("Published result to %s and %s", local_publish_path, rapms_publish_path)

    # Save rubberband analysis if enabled
    if rubberband:
        rb_output_dir = output_file.parent
        rb_prefix = output_file.stem.replace("_results", "")
        save_rubberband_analysis(beta, player_to_col, rb_output_dir, rb_prefix)
    if season_effects and feature_meta['season_effects'].get('enabled'):
        se_output_dir = output_file.parent
        se_prefix = output_file.stem.replace("_results", "")
        save_season_effect_analysis(beta, player_to_col, feature_meta['season_effects'], se_output_dir, se_prefix)
    if age_dummies and feature_meta['age_dummies'].get('enabled'):
        age_output_dir = output_file.parent
        age_prefix = output_file.stem.replace("_results", "")
        save_age_effect_analysis(beta, player_to_col, feature_meta['age_dummies'], age_output_dir, age_prefix)
    
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

  # Rubberband effect (score-margin x period adjustment)
  python rapm.py RAPM 26 26 ALL --rubberband --pure

  # Add season fixed effects across a long multi-year window
  python rapm.py MIDRANGE_FREQ 14 26 ALL --season-effects

  # Add signed age fixed effects inside a metric solve
  python rapm.py RAPM 14 26 ALL --rubberband --age-dummies --pure
  python rapm.py TS 14 26 ALL --rubberband --age-dummies

  # Age-curve prior mode (RAPM only)
  python rapm.py RAPM 14 26 ALL --rubberband --age-curve --pure

  # DARKO strong/not-strong opponent split interactions
  python rapm.py RAPM 14 26 ALL --pure --strength-splits

  # DARKO split with legacy target standard-deviation scaling
  python rapm.py RAPM 14 26 ALL --strength-splits --strength-legacy-scale

  # Continuous DARKO opponent-strength sensitivity slopes
  python rapm.py RAPM 14 26 ALL --strength-sensitivity

  # Cross-validate separate offense/defense alphas
  python rapm.py RAPM 26 26 RS --cv-alpha
        """
    )
    
    parser.add_argument('prefix',
                       choices=['RAPM', 'TOV', 'REB', 'TS', 'RIM_FREQ', 'RIM_FG_PCT', 'LA_RAPM', 'MIDRANGE_FREQ',
                                'MIDRANGE_FG_PCT',
                                'ASSIST_POINTS', 'RIM_ASSIST', 'DUNK', 'DUNK_ASSIST',
                                'BLOCK_RECOVERY',
                                'THREE_FREQ', 'THREE_FG_PCT', 'PLAYTYPE_TS_MIX',
                                'PLAYTYPE_PROXY_PTS',
                                'TRANSITION_FREQ', 'TRANSITION_RIM', 'INITIAL_EV', 'INITIAL_EV_BETA', 'SPECIAL_RAPM', 'EV_RAPM',
                                'RUSSELL_SHOTQUALITY', 'CONTEXT_SHOTQUALITY',
                               'SQ_POSS', 'FT_PREMIUM', 'CONTEST', 'SECOND_CHANCE', 'FIRST_CHANCE',
                               'FIRST_CHANCE_CLEAN', 'SECOND_CHANCE_CLEAN', 'BADPASS_TOV', 'SCORING_TOV',
                              'ALT_TS', 'ALT_EFG', 'ALT_SQ', 'ALT_MAKE', 'ALT_FT',
                              'ALT_FT_FREQ', 'ALT_FT_SEVERITY',
                              'ALT_EFG_VALUE',
                              'ALT_EFG_BASELINE', 'ALT_EFG_RIM', 'ALT_EFG_MID', 'ALT_EFG_THREE',
                              'ALT_EFG_RIM_FREQ', 'ALT_EFG_RIM_FG',
                              'ALT_EFG_MID_FREQ', 'ALT_EFG_MID_FG',
                              'ALT_EFG_THREE_FREQ', 'ALT_EFG_THREE_FG',
                              'ALT_TOV_VALUE', 'ALT_BADPASS_TOV_VALUE', 'ALT_SCORING_TOV_VALUE',
                              'ALT_FC_COMPLETION',
                              'FC_TRANSITION_SCORING', 'FC_HALFCOURT_SCORING',
                              'FC_MODE_MIX', 'FC_TRANSITION_VALUE', 'FC_HALFCOURT_VALUE',
                              'ALT_TOV', 'ALT_BADPASS_TOV', 'ALT_SCORING_TOV'],
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

    parser.add_argument('--off-prior',
                       type=str,
                       default=None,
                       help='CSV file with offense prior values (player_id, prior_value). Shrinks offense coefficients toward prior instead of zero.')

    parser.add_argument('--def-alpha',
                       type=float,
                       default=None,
                       help='Override defense alpha (higher = stronger pull toward prior/zero). Default: 3000')

    parser.add_argument('--off-alpha',
                       type=float,
                       default=None,
                       help='Override offense alpha (higher = more regularization). Default: 3000')

    parser.add_argument('--cv-alpha',
                       action='store_true',
                       help='Cross-validate offense and defense alphas separately instead of fixing them at 3000/overrides.')

    parser.add_argument('--cv-folds',
                       type=int,
                       default=5,
                       help='Number of folds for alpha cross-validation (default: 5)')

    parser.add_argument('--cv-passes',
                       type=int,
                       default=2,
                       help='Coordinate-search passes between offense and defense alpha CV (default: 2)')

    parser.add_argument('--cv-alpha-grid',
                       type=str,
                       default=None,
                       help='Comma-separated alpha grid for cross-validation (default: 300,500,1000,2000,3000,5000,7000,10000)')

    parser.add_argument('--publish-to-rapms',
                       action='store_true',
                       help='Copy the finished result CSV into local master_results and /Users/russellthomas/Docs/rapms/master_results, then commit/push rapms if changed.')

    parser.add_argument('--publish-name',
                       type=str,
                       default=None,
                       help='Optional published filename override when using --publish-to-rapms.')

    parser.add_argument('--timedecay', '-td',
                       action='store_true',
                       help='Apply time decay weighting (recent games weighted more)')

    parser.add_argument('--half-life',
                       type=float,
                       default=None,
                       help='Half-life in days for time decay (default: 700). Implies --timedecay.')

    parser.add_argument('--rubberband', '-rb',
                       action='store_true',
                       help='Enable rubberband effect (score-margin x period adjustment)')

    parser.add_argument('--alpha-rb',
                       type=float,
                       default=None,
                       help='Rubberband regularization alpha (default: 10). Lower = less regularization.')

    parser.add_argument('--season-effects', '-se',
                       action='store_true',
                       help='Add season fixed effects (one omitted-reference dummy per season) to absorb leaguewide year-to-year baseline shifts.')

    parser.add_argument('--fixed-season-baselines',
                       type=str,
                       default=None,
                       help='CSV with columns season and baseline_per_possession to subtract fixed season constants from y before solving.')

    parser.add_argument('--age-dummies',
                       action='store_true',
                       help='Add signed age-bin fixed effects for ages 18-38 (38+ collapsed into 38), using dpm_history age lookups.')

    parser.add_argument('--age-poly-coefficients',
                       type=str,
                       default=None,
                       help='Polynomial age-curve coefficient CSV. Applies curve(age)-curve(reference) as fixed row offsets before solving.')

    parser.add_argument('--age-poly-reference-age',
                       type=float,
                       default=float(AGE_DUMMY_REFERENCE),
                       help='Reference age for --age-poly-coefficients fixed offsets (default: 27).')

    parser.add_argument('--age-curve',
                       action='store_true',
                       help='RAPM-only mode: shrink offense/defense toward a learned age-mean prior from player_stats_with_metrics.')

    parser.add_argument('--strength-splits',
                       action='store_true',
                       help='RAPM-only mode: add DARKO strong/not-strong opponent interaction columns for each player.')

    parser.add_argument('--strength-legacy-scale',
                       action='store_true',
                       help='Use with --strength-splits: target-scale overall offense/defense to the legacy strong/not-strong script stdevs.')

    parser.add_argument('--strength-sensitivity',
                       action='store_true',
                       help='RAPM-only mode: add continuous season-standardized DARKO opponent-strength slope columns for each player.')

    parser.add_argument('--strength-sensitivity-alpha-mult',
                       type=float,
                       default=4.0,
                       help='Penalty multiplier for strength-sensitivity slope columns relative to base player columns (default: 4.0).')

    parser.add_argument('--darko-history',
                       type=str,
                       default=str(DEFAULT_DARKO_HISTORY_PATH),
                       help=f'Path to DARKO DPM history CSV for --age-dummies, --strength-splits, and --strength-sensitivity (default: {DEFAULT_DARKO_HISTORY_PATH}).')

    args = parser.parse_args()

    # --half-life implies --timedecay
    if args.half_life is not None:
        args.timedecay = True

    if args.age_dummies and args.age_curve:
        parser.error("--age-dummies cannot be combined with --age-curve.")
    if args.age_dummies and args.age_poly_coefficients:
        parser.error("--age-dummies cannot be combined with --age-poly-coefficients.")
    if args.age_curve and args.prefix != 'RAPM':
        parser.error("--age-curve is only supported for RAPM.")
    if args.age_curve and (args.off_prior or args.def_prior):
        parser.error("--age-curve cannot be combined with --off-prior or --def-prior.")
    if args.strength_splits and args.prefix != 'RAPM':
        parser.error("--strength-splits is only supported for RAPM.")
    if args.strength_legacy_scale and not args.strength_splits:
        parser.error("--strength-legacy-scale requires --strength-splits.")
    if args.strength_sensitivity and args.prefix != 'RAPM':
        parser.error("--strength-sensitivity is only supported for RAPM.")
    if args.strength_sensitivity and args.strength_splits:
        parser.error("--strength-sensitivity should be run separately from --strength-splits.")
    if args.strength_sensitivity_alpha_mult <= 0:
        parser.error("--strength-sensitivity-alpha-mult must be positive.")
    if args.cv_folds < 2:
        parser.error("--cv-folds must be at least 2.")
    if args.cv_passes < 1:
        parser.error("--cv-passes must be at least 1.")
    if args.cv_alpha:
        try:
            parse_alpha_grid(args.cv_alpha_grid)
        except ValueError as exc:
            parser.error(str(exc))
    if args.publish_name and not args.publish_to_rapms:
        parser.error("--publish-name requires --publish-to-rapms.")
    
    # Determine how many cores to use and update global
    if args.cores is not None:
        cores_to_use = max(1, min(args.cores, mp.cpu_count()))
        logging.info(f"Using {cores_to_use} CPU cores (user specified)")
        # Update the global for use in functions
        globals()['N_CORES'] = cores_to_use
    else:
        cores_to_use = N_CORES
        logging.info(f"Using {cores_to_use} CPU cores (auto-detected)")
    
    # Prefix aliasing: some metrics share a parquet file
    PREFIX_FILE_MAP = {
        'BADPASS_TOV': 'TOV',
        'SCORING_TOV': 'TOV',
        'ALT_TS': 'FIRST_CHANCE',
        'ALT_EFG': 'FIRST_CHANCE',
        'ALT_SQ': 'FIRST_CHANCE',
        'ALT_MAKE': 'FIRST_CHANCE',
        'ALT_FT': 'FIRST_CHANCE',
        'ALT_FT_FREQ': 'FIRST_CHANCE',
        'ALT_FT_SEVERITY': 'FIRST_CHANCE',
        'ALT_EFG_VALUE': 'FIRST_CHANCE',
        'ALT_EFG_BASELINE': 'FIRST_CHANCE',
        'ALT_EFG_RIM': 'FIRST_CHANCE',
        'ALT_EFG_MID': 'FIRST_CHANCE',
        'ALT_EFG_THREE': 'FIRST_CHANCE',
        'ALT_EFG_RIM_FREQ': 'FIRST_CHANCE',
        'ALT_EFG_RIM_FG': 'FIRST_CHANCE',
        'ALT_EFG_MID_FREQ': 'FIRST_CHANCE',
        'ALT_EFG_MID_FG': 'FIRST_CHANCE',
        'ALT_EFG_THREE_FREQ': 'FIRST_CHANCE',
        'ALT_EFG_THREE_FG': 'FIRST_CHANCE',
        'ALT_TOV_VALUE': 'FIRST_CHANCE',
        'ALT_BADPASS_TOV_VALUE': 'FIRST_CHANCE',
        'ALT_SCORING_TOV_VALUE': 'FIRST_CHANCE',
        'ALT_FC_COMPLETION': 'FIRST_CHANCE',
        'FC_TRANSITION_SCORING': 'FIRST_CHANCE',
        'FC_HALFCOURT_SCORING': 'FIRST_CHANCE',
        'FC_MODE_MIX': 'FIRST_CHANCE',
        'FC_TRANSITION_VALUE': 'FIRST_CHANCE',
        'FC_HALFCOURT_VALUE': 'FIRST_CHANCE',
        'ALT_TOV': 'FIRST_CHANCE',
        'ALT_BADPASS_TOV': 'FIRST_CHANCE',
        'ALT_SCORING_TOV': 'FIRST_CHANCE',
    }
    file_prefix = PREFIX_FILE_MAP.get(args.prefix, args.prefix)

    # Build list of files to process
    input_files = []
    years = expand_two_digit_year_window(args.start_year, args.end_year)

    for year in years:
        if args.season_type in ['RS', 'ALL']:
            # Regular season file
            filename = f"{file_prefix}{year:02d}.parquet"
            input_files.append(filename)

        if args.season_type in ['PS', 'ALL']:
            # Playoff file
            filename = f"{file_prefix}{year:02d}_PS.parquet"
            input_files.append(filename)
    
    # Log what we're processing
    logging.info(f"Processing {args.prefix} data for years {args.start_year}-{args.end_year} ({args.season_type})")
    logging.info(f"Files to process: {', '.join(input_files)}")

    fixed_season_baselines = None
    if args.fixed_season_baselines:
        baseline_df = pd.read_csv(args.fixed_season_baselines)
        required = {"season", "baseline_per_possession"}
        missing = required - set(baseline_df.columns)
        if missing:
            parser.error(f"--fixed-season-baselines missing columns: {sorted(missing)}")
        if "metric" in baseline_df.columns:
            metric_df = baseline_df[baseline_df["metric"].astype(str).str.upper() == args.prefix.upper()].copy()
        else:
            metric_df = baseline_df.copy()
        metric_df = metric_df.copy()
        metric_df["season"] = pd.to_numeric(metric_df["season"], errors="coerce")
        metric_df["baseline_per_possession"] = pd.to_numeric(metric_df["baseline_per_possession"], errors="coerce")
        metric_df = metric_df.dropna(subset=["season", "baseline_per_possession"])
        fixed_season_baselines = dict(zip(metric_df["season"].astype(int), metric_df["baseline_per_possession"].astype(float)))
        if not fixed_season_baselines:
            parser.error(f"No fixed season baselines found for prefix {args.prefix}.")
    
    # Run the analysis
    results = run_simplified_rapm(input_files, pure=args.pure,
                                  off_prior_file=args.off_prior,
                                  def_prior_file=args.def_prior,
                                  def_alpha=args.def_alpha,
                                  off_alpha=args.off_alpha,
                                  cv_alpha=args.cv_alpha,
                                  cv_folds=args.cv_folds,
                                  cv_passes=args.cv_passes,
                                  cv_alpha_grid=args.cv_alpha_grid,
                                  publish_to_rapms=args.publish_to_rapms,
                                  publish_name=args.publish_name,
                                  timedecay=args.timedecay,
                                  half_life=args.half_life,
                                  rubberband=args.rubberband,
                                  alpha_rb=args.alpha_rb,
                                  season_effects=args.season_effects,
                                  fixed_season_baselines=fixed_season_baselines,
                                  age_dummies=args.age_dummies,
                                  age_poly_coefficients=args.age_poly_coefficients,
                                  age_poly_reference_age=args.age_poly_reference_age,
                                  age_curve=args.age_curve,
                                  strength_splits=args.strength_splits,
                                  strength_legacy_scale=args.strength_legacy_scale,
                                  strength_sensitivity=args.strength_sensitivity,
                                  strength_sensitivity_alpha_mult=args.strength_sensitivity_alpha_mult,
                                  darko_history_path=args.darko_history,
                                  start_year=args.start_year,
                                  end_year=args.end_year,
                                  prefix=args.prefix)
    
    if results is not None:
        logging.info("\nDone!")
    else:
        logging.error("\nAnalysis failed - no data was processed.")
        sys.exit(1)
