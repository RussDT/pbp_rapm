"""
Update the current-season scposs rows in the per-player CSV files.

Reads `weighted_factors_<window>_all.csv` from the per-window
`nba_pipeline/results/rapm_<window>_all` folders, transforms them, and updates
`<csvs>/scposs/{nba_id}.csv`. Despite the historical `2026` in this filename,
the script is season parameterized: pass `--year 2027` (or set
`NBA_RAPM_SEASON_YEAR=2027`). Season year N means season (N-1)-N; two-digit
result tokens are derived from it.

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
TARGET_DIR = CSVS_DIR / "scposs"

# Window lengths published to scposs
WINDOW_LENGTHS = (1, 2, 3, 4, 5, 6)


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
            (f"rapm_{token}_all", f"weighted_factors_{token}_all.csv", f"{length}Y")
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
    
    # Calculate sc_POSS = oTOV + oREB + dTOV + dREB (using original column names before rename)
    df["sc_POSS"] = df["sc_OFF_TOV"] + df["sc_OFF_REB"] + df["sc_DEF_TOV"] + df["sc_DEF_REB"]
    
    # Set placeholders to 0
    df["off_diff"] = 0
    df["def_diff"] = 0
    
    return df


def calculate_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate ranks for each Year_Interval group."""
    
    # Columns to rank (higher is better for all)
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
    
    # Calculate ranks within each Year_Interval group
    for source_col, rank_col in rank_mappings.items():
        df[rank_col] = df.groupby("Year_Interval")[source_col].rank(
            ascending=False, method="min"
        ).astype(int)
    
    return df


def main(latest_year: int = DEFAULT_LATEST_YEAR):
    # Load and transform all source files
    all_new_data = []
    for folder, wf_file, year_interval in build_source_files(latest_year):
        df = load_and_transform(folder, wf_file, year_interval)
        all_new_data.append(df)
        print(f"  -> {len(df)} rows")
    
    # Combine all new rows for the target season
    new_season_df = pd.concat(all_new_data, ignore_index=True)
    print(f"\nTotal new {latest_year} rows before ranks: {len(new_season_df)}")
    
    # Calculate ranks
    new_season_df = calculate_ranks(new_season_df)
    
    # Select and order columns
    new_season_df = new_season_df[TARGET_COLUMNS]
    
    print(f"Total new {latest_year} rows: {len(new_season_df)}")
    
    # Get unique player IDs
    unique_players = new_season_df["nba_id"].unique()
    print(f"Unique players to update: {len(unique_players)}")
    
    # Ensure target directory exists
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    # Track stats
    updated_files = 0
    created_files = 0
    
    # Process each player
    for nba_id in unique_players:
        player_file = os.path.join(TARGET_DIR, f"{nba_id}.csv")
        player_new_data = new_season_df[new_season_df["nba_id"] == nba_id]
        
        if os.path.exists(player_file):
            # Load existing data
            existing_df = pd.read_csv(player_file)
            
            # Remove existing rows for the target season
            existing_df = existing_df[existing_df["Latest_Year"] != latest_year]
            
            # Append the rebuilt season rows
            combined_df = pd.concat([existing_df, player_new_data], ignore_index=True)
            
            # Sort by Latest_Year and Year_Interval
            combined_df = combined_df.sort_values(["Latest_Year", "Year_Interval"])
            
            updated_files += 1
        else:
            # Create new file with just the target-season rows
            combined_df = player_new_data.sort_values(["Latest_Year", "Year_Interval"])
            created_files += 1
        
        # Save to file
        combined_df.to_csv(player_file, index=False)
    
    print(f"\nUpdated {updated_files} existing player files")
    print(f"Created {created_files} new player files")
    
    # Print sample of new data
    print(f"\nSample of new {latest_year} data (first 5 rows from 2Y):")
    sample = new_season_df[new_season_df["Year_Interval"] == "2Y"].head(5)
    print(sample[["nba_id", "player_name", "Year_Interval", "Off_RAPM", "Def_RAPM", "OVR_RAPM", "sc_POSS"]].to_string(index=False))


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

