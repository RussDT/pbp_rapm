"""
Script to update 2026 RAPM data in PureRAPM.csv
Reads from Results folder, transforms data, and replaces 2026 entries.
"""

import pandas as pd
import os

# Configuration
RESULTS_DIR = "nba_pipeline/results"
TARGET_FILE = "../csvs/PureRAPM.csv"
ENDING_SEASON = 2026

# Source files mapping: (folder, filename, length_of_rapm)
SOURCE_FILES = [
    ("rapm_26_all", "weighted_factors_26_all.csv", 1),
    ("rapm_25_26_all", "weighted_factors_25_26_all.csv", 2),
    ("rapm_24_26_all", "weighted_factors_24_26_all.csv", 3),
    ("rapm_23_26_all", "weighted_factors_23_26_all.csv", 4),
    ("rapm_22_26_all", "weighted_factors_22_26_all.csv", 5),
    ("rapm_21_26_all", "weighted_factors_21_26_all.csv", 6),
]


def load_and_transform(folder: str, filename: str, length_of_rapm: int) -> pd.DataFrame:
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
    df["ending_season"] = ENDING_SEASON
    
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


def main():
    # Load and transform all source files
    all_new_data = []
    for folder, filename, length_of_rapm in SOURCE_FILES:
        df = load_and_transform(folder, filename, length_of_rapm)
        all_new_data.append(df)
        print(f"  -> {len(df)} rows with length_of_rapm={length_of_rapm}")
    
    # Combine all new 2026 data
    new_2026_df = pd.concat(all_new_data, ignore_index=True)
    print(f"\nTotal new 2026 rows: {len(new_2026_df)}")
    
    # Load existing PureRAPM.csv
    print(f"\nLoading existing {TARGET_FILE}...")
    existing_df = pd.read_csv(TARGET_FILE)
    print(f"  -> {len(existing_df)} total rows")
    
    # Remove existing 2026 data
    existing_2026_count = (existing_df["ending_season"] == ENDING_SEASON).sum()
    print(f"  -> {existing_2026_count} existing 2026 rows to replace")
    
    non_2026_df = existing_df[existing_df["ending_season"] != ENDING_SEASON]
    print(f"  -> {len(non_2026_df)} non-2026 rows to keep")
    
    # Combine: keep non-2026 data + add new 2026 data
    final_df = pd.concat([non_2026_df, new_2026_df], ignore_index=True)
    
    # Sort by nba_id, length_of_rapm, ending_season for consistency
    final_df = final_df.sort_values(["nba_id", "length_of_rapm", "ending_season"])
    
    # Save back to PureRAPM.csv
    print(f"\nSaving to {TARGET_FILE}...")
    final_df.to_csv(TARGET_FILE, index=False)
    print(f"  -> {len(final_df)} total rows written")
    
    # Print sample of new data
    print("\nSample of new 2026 data (first 5 rows for each interval):")
    for length in range(1, 7):
        print(f"\nLength: {length}Y")
        sample = new_2026_df[new_2026_df["length_of_rapm"] == length].head(5)
        print(sample.to_string(index=False))


if __name__ == "__main__":
    main()

