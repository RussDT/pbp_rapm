"""
Update the current-season RAPM slice of PureRAPM.csv.

Despite the historical `2026` in this filename, the script is season
parameterized: pass `--year 2027` (or set `NBA_RAPM_SEASON_YEAR=2027`). Season
year N means season (N-1)-N; two-digit result tokens are derived from it.

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

DEFAULT_ENDING_SEASON = int(os.environ.get("NBA_RAPM_SEASON_YEAR", "2027"))

RESULTS_DIR = Path(
    os.environ.get("NBA_RAPM_RESULTS_DIR", PROJECT_ROOT / "nba_pipeline" / "results")
)
CSVS_DIR = Path(os.environ.get("NBA_RAPM_CSVS_DIR", PROJECT_ROOT.parent / "csvs"))
TARGET_FILE = CSVS_DIR / "PureRAPM.csv"

# Window lengths published to PureRAPM.csv
RAPM_LENGTHS = (1, 2, 3, 4, 5, 6)


def two_digit(season_year: int) -> str:
    """Two-digit season-end form used in result folder and file names."""
    return f"{int(season_year) % 100:02d}"


def range_token(ending_season: int, length_of_rapm: int) -> str:
    """`26` for a single season, `25_26` for a 2-season window, etc."""
    end_yy = two_digit(ending_season)
    if length_of_rapm <= 1:
        return end_yy
    return f"{two_digit(ending_season - length_of_rapm + 1)}_{end_yy}"


def build_source_files(ending_season: int):
    """Build (folder, filename, length_of_rapm) tuples for every window."""
    source_files = []
    for length in RAPM_LENGTHS:
        token = range_token(ending_season, length)
        source_files.append(
            (f"rapm_{token}_all_se", f"weighted_factors_{token}_all_se.csv", length)
        )
    return source_files


def load_and_transform(folder: str, filename: str, length_of_rapm: int,
                       ending_season: int) -> pd.DataFrame:
    """Load a source file and transform it to match PureRAPM format."""
    filepath = os.path.join(RESULTS_DIR, folder, filename)
    print(f"Loading {filepath}...")
    
    df = pd.read_csv(filepath)
    
    # Rename columns
    df = df.rename(columns={
        "player_id": "nba_id",
        "player_name": "player",
        "net_rapm": "rapm"
    })
    
    # Def is already correctly signed in weighted_factors
    
    # Round to 1 decimal place
    df["off"] = df["off"].round(1)
    df["def"] = df["def"].round(1)
    df["rapm"] = df["rapm"].round(1)
    
    # Add length_of_rapm and ending_season
    df["length_of_rapm"] = length_of_rapm
    df["ending_season"] = ending_season
    
    # Calculate integer ranks (1 = best/highest)
    # For off: higher is better, so rank descending
    df["off rk"] = df["off"].rank(ascending=False, method="min").astype(int)
    # For def: higher is better (positive = good defense)
    df["def rk"] = df["def"].rank(ascending=False, method="min").astype(int)
    # For tot (rapm): higher is better
    df["tot rk"] = df["rapm"].rank(ascending=False, method="min").astype(int)
    
    # Add placeholder columns
    df["nuanced_per_game"] = 0.0
    df["per_game_rank"] = 0.0
    
    # Select and order columns to match target schema
    columns = [
        "nba_id", "player", "length_of_rapm", "ending_season",
        "off", "def", "rapm", "off rk", "def rk", "tot rk",
        "nuanced_per_game", "per_game_rank"
    ]
    
    return df[columns]


def main(ending_season: int = DEFAULT_ENDING_SEASON):
    # Load and transform all source files
    all_new_data = []
    for folder, filename, length_of_rapm in build_source_files(ending_season):
        df = load_and_transform(folder, filename, length_of_rapm, ending_season)
        all_new_data.append(df)
        print(f"  -> {len(df)} rows with length_of_rapm={length_of_rapm}")
    
    # Combine all new rows for the target season
    new_season_df = pd.concat(all_new_data, ignore_index=True)
    print(f"\nTotal new {ending_season} rows: {len(new_season_df)}")
    
    # Load existing PureRAPM.csv
    print(f"\nLoading existing {TARGET_FILE}...")
    existing_df = pd.read_csv(TARGET_FILE)
    print(f"  -> {len(existing_df)} total rows")
    
    # Remove existing rows for the target season
    existing_season_count = (existing_df["ending_season"] == ending_season).sum()
    print(f"  -> {existing_season_count} existing {ending_season} rows to replace")
    
    non_season_df = existing_df[existing_df["ending_season"] != ending_season]
    print(f"  -> {len(non_season_df)} non-{ending_season} rows to keep")
    
    # Combine: keep other seasons + add the rebuilt season
    final_df = pd.concat([non_season_df, new_season_df], ignore_index=True)
    
    # Sort by nba_id, length_of_rapm, ending_season for consistency
    final_df = final_df.sort_values(["nba_id", "length_of_rapm", "ending_season"])
    
    # Save back to PureRAPM.csv
    print(f"\nSaving to {TARGET_FILE}...")
    final_df.to_csv(TARGET_FILE, index=False)
    print(f"  -> {len(final_df)} total rows written")
    
    # Print sample of new data
    print(f"\nSample of new {ending_season} data (first 5 rows for each interval):")
    for length in RAPM_LENGTHS:
        print(f"\nLength: {length}Y")
        sample = new_season_df[new_season_df["length_of_rapm"] == length].head(5)
        print(sample.to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--year",
        type=int,
        default=DEFAULT_ENDING_SEASON,
        help="Season end year to rebuild (default %(default)s; 2027 = 2026-27)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args().year)

