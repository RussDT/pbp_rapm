"""
Update the current-season slice of SCALEDOUTPUT_SMALLER.csv.

Same transformation as scposs but only for the 3Y and 5Y windows. Despite the
historical `2026` in this filename, the script is season parameterized: pass
`--year 2027` (or set `NBA_RAPM_SEASON_YEAR=2027`). Season year N means season
(N-1)-N; two-digit result tokens are derived from it.

Paths:
  NBA_RAPM_RESULTS_DIR  RAPM results root (default <repo>/nba_pipeline/results)
  NBA_RAPM_CSVS_DIR     csvs repo checkout (default <repo>/../csvs)
"""

import argparse
import os
from pathlib import Path

import pandas as pd

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_LATEST_YEAR = int(os.environ.get("NBA_RAPM_SEASON_YEAR", "2027"))

RESULTS_DIR = Path(
    os.environ.get("NBA_RAPM_RESULTS_DIR", PROJECT_ROOT / "nba_pipeline" / "results")
)
CSVS_DIR = Path(os.environ.get("NBA_RAPM_CSVS_DIR", PROJECT_ROOT.parent / "csvs"))
TARGET_FILE = CSVS_DIR / "SCALEDOUTPUT_SMALLER.csv"

# Window lengths published here: 3Y and 5Y only
WINDOW_LENGTHS = (3, 5)


def two_digit(season_year: int) -> str:
    """Two-digit season-end form used in result folder and file names."""
    return f"{int(season_year) % 100:02d}"


def range_token(latest_year: int, length: int) -> str:
    """`26` for a single season, `24_26` for a 3-season window, etc."""
    end_yy = two_digit(latest_year)
    if length <= 1:
        return end_yy
    return f"{two_digit(latest_year - length + 1)}_{end_yy}"


def build_source_files(latest_year: int):
    """Build (folder, weighted_factors_file, year_interval) tuples."""
    source_files = []
    for length in WINDOW_LENGTHS:
        token = range_token(latest_year, length)
        source_files.append(
            (
                f"rapm_{token}_all_se",
                f"weighted_factors_{token}_all_se.csv",
                f"{length}Y",
            )
        )
    return source_files


# Target column order
TARGET_COLUMNS = [
    "nba_id", "player_name", "Latest_Year", "Year_Interval",
    "Off_RAPM", "Def_RAPM", "Off_Poss",
    "sc_OFF_TS", "sc_OFF_TOV", "sc_OFF_REB",
    "sc_DEF_TS", "sc_DEF_TOV", "sc_DEF_REB",
    "OVR_RAPM", "sc_POSS", "off_diff", "def_diff",
    "Off_RAPM_rank", "Def_RAPM_rank",
    "sc_OFF_TS_rank", "sc_OFF_TOV_rank", "sc_OFF_REB_rank",
    "sc_DEF_TS_rank", "sc_DEF_TOV_rank", "sc_DEF_REB_rank",
    "OVR_RAPM_rank", "sc_POSS_rank"
]


def load_and_transform(folder: str, wf_file: str, year_interval: str) -> pd.DataFrame:
    """Load weighted_factors and transform to scposs format."""
    wf_path = os.path.join(RESULTS_DIR, folder, wf_file)
    
    print(f"Loading {folder} ({year_interval})...")
    
    # Load weighted_factors
    df = pd.read_csv(wf_path)
    
    # Rename columns
    df = df.rename(columns={
        "player_id": "nba_id",
        "off": "Off_RAPM",
        "def": "Def_RAPM",
        "off_poss": "Off_Poss",
        "oTS": "sc_OFF_TS",
        "oTOV": "sc_OFF_TOV",
        "oREB": "sc_OFF_REB",
        "dTS": "sc_DEF_TS",
        "dTOV": "sc_DEF_TOV",
        "dREB": "sc_DEF_REB",
        "net_rapm": "OVR_RAPM"
    })
    
    # Def_RAPM is already correctly signed in source
    
    # Add Year_Interval
    df["Year_Interval"] = year_interval
    
    # Calculate sc_POSS = oTOV + oREB + dTOV + dREB
    df["sc_POSS"] = df["sc_OFF_TOV"] + df["sc_OFF_REB"] + df["sc_DEF_TOV"] + df["sc_DEF_REB"]
    
    # Set placeholders to 0
    df["off_diff"] = 0
    df["def_diff"] = 0
    
    return df


def calculate_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate ranks for each Year_Interval group."""
    
    rank_mappings = {
        "Off_RAPM": "Off_RAPM_rank",
        "Def_RAPM": "Def_RAPM_rank",
        "sc_OFF_TS": "sc_OFF_TS_rank",
        "sc_OFF_TOV": "sc_OFF_TOV_rank",
        "sc_OFF_REB": "sc_OFF_REB_rank",
        "sc_DEF_TS": "sc_DEF_TS_rank",
        "sc_DEF_TOV": "sc_DEF_TOV_rank",
        "sc_DEF_REB": "sc_DEF_REB_rank",
        "OVR_RAPM": "OVR_RAPM_rank",
        "sc_POSS": "sc_POSS_rank"
    }
    
    for source_col, rank_col in rank_mappings.items():
        df[rank_col] = df.groupby("Year_Interval")[source_col].rank(
            ascending=False, method="min"
        ).astype(int)
    
    return df


def main(latest_year: int = DEFAULT_LATEST_YEAR):
    # Load and transform source files (3Y and 5Y only)
    all_new_data = []
    for folder, wf_file, year_interval in build_source_files(latest_year):
        df = load_and_transform(folder, wf_file, year_interval)
        all_new_data.append(df)
        print(f"  -> {len(df)} rows")
    
    # Combine new rows for the target season
    new_season_df = pd.concat(all_new_data, ignore_index=True)
    print(f"\nTotal new {latest_year} rows before ranks: {len(new_season_df)}")
    
    # Calculate ranks
    new_season_df = calculate_ranks(new_season_df)
    
    # Select and order columns
    new_season_df = new_season_df[TARGET_COLUMNS]
    
    print(f"Total new {latest_year} rows: {len(new_season_df)}")
    
    # Load existing data
    print(f"\nLoading existing {TARGET_FILE}...")
    existing_df = pd.read_csv(TARGET_FILE)
    print(f"  -> {len(existing_df)} total rows")
    
    # Remove existing rows for the target season
    existing_season_count = (existing_df["Latest_Year"] == latest_year).sum()
    print(f"  -> {existing_season_count} existing {latest_year} rows to replace")
    
    non_season_df = existing_df[existing_df["Latest_Year"] != latest_year]
    print(f"  -> {len(non_season_df)} non-{latest_year} rows to keep")
    
    # Combine
    final_df = pd.concat([non_season_df, new_season_df], ignore_index=True)
    
    # Sort
    final_df = final_df.sort_values(["nba_id", "Latest_Year", "Year_Interval"])
    
    # Save
    print(f"\nSaving to {TARGET_FILE}...")
    final_df.to_csv(TARGET_FILE, index=False)
    print(f"  -> {len(final_df)} total rows written")
    
    # Sample
    print(f"\nSample of new {latest_year} data:")
    sample = new_season_df.head(5)
    print(sample[["nba_id", "player_name", "Year_Interval", "Off_RAPM", "Def_RAPM", "OVR_RAPM"]].to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--year",
        type=int,
        default=DEFAULT_LATEST_YEAR,
        help="Season end year to rebuild (default %(default)s; 2027 = 2026-27)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args().year)

