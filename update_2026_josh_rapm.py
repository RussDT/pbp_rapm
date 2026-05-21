"""
Script to update 2026 data in josh_rapm CSV files.
Reads pure_results.csv from Results folders, divides values by 100,
rounds to 4 decimals, and updates josh_rapm files.
"""

import pandas as pd
import os

# Configuration
RESULTS_DIR = "nba_pipeline/results"
TARGET_DIR = "josh_rapm"
SECOND_TARGET_DIR = "../csvs/josh_rapm"
END_SEASON = 2026

# Source files mapping: (target_file, source_folder, source_file, start_season)
SOURCE_FILES = [
    ("5y_josh_rapm.csv", "rapm_22_26_all", "weighted_factors_22_26_all.csv", 2022),
    ("4y_josh_rapm.csv", "rapm_23_26_all", "weighted_factors_23_26_all.csv", 2023),
    ("3y_josh_rapm.csv", "rapm_24_26_all", "weighted_factors_24_26_all.csv", 2024),
    ("2y_josh_rapm.csv", "rapm_25_26_all", "weighted_factors_25_26_all.csv", 2025),
]

# Target column order
TARGET_COLUMNS = [
    "start_season", "end_season", "player_id", "player_name",
    "off", "def", "total_coefficient",
    "fraction_player", "f_period", "vanilla_per_game", "nuanced_per_game"
]


def load_and_transform(source_folder: str, source_file: str, start_season: int) -> pd.DataFrame:
    """Load weighted_factors and transform to josh_rapm format."""
    source_path = os.path.join(RESULTS_DIR, source_folder, source_file)
    print(f"  Loading {source_path}...")
    
    # Load source data
    df = pd.read_csv(source_path)
    
    # Transform: divide by 100 and round to 4 decimals
    df["off"] = (df["off"] / 100).round(4)
    # Def needs to be negative for good defense in josh_rapm format
    df["def"] = (df["def"] / 100 * -1).round(4)
    df["total_coefficient"] = (df["net_rapm"] / 100).round(4)
    
    # Add fixed values
    df["start_season"] = start_season
    df["end_season"] = END_SEASON
    df["fraction_player"] = 0
    df["f_period"] = 0
    df["vanilla_per_game"] = 0
    df["nuanced_per_game"] = 0
    
    # Select and order columns
    df = df[TARGET_COLUMNS]
    
    return df


def main():
    # Ensure second target directory exists
    os.makedirs(SECOND_TARGET_DIR, exist_ok=True)
    
    for target_file, source_folder, source_file, start_season in SOURCE_FILES:
        target_path = os.path.join(TARGET_DIR, target_file)
        second_target_path = os.path.join(SECOND_TARGET_DIR, target_file)
        print(f"\nProcessing {target_file} (Start: {start_season}, End: {END_SEASON})...")
        
        # Load existing data from main target (or second target if main doesn't exist)
        if os.path.exists(target_path):
            existing_df = pd.read_csv(target_path)
        elif os.path.exists(second_target_path):
            existing_df = pd.read_csv(second_target_path)
        else:
            print(f"  Warning: {target_file} not found in either {TARGET_DIR} or {SECOND_TARGET_DIR}")
            continue
            
        original_count = len(existing_df)
        
        # Remove existing 2026 rows
        existing_df = existing_df[existing_df["end_season"] != END_SEASON]
        removed_count = original_count - len(existing_df)
        print(f"  Removed {removed_count} existing 2026 rows")
        
        # Load and transform new 2026 data
        new_df = load_and_transform(source_folder, source_file, start_season)
        print(f"  Adding {len(new_df)} new 2026 rows")
        
        # Combine
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        
        # Sort by start_season, end_season, player_id
        combined_df = combined_df.sort_values(["start_season", "end_season", "player_id"])
        
        # Save to both locations
        combined_df.to_csv(target_path, index=False)
        print(f"  Saved {len(combined_df)} total rows to {target_path}")
        
        combined_df.to_csv(second_target_path, index=False)
        print(f"  Saved {len(combined_df)} total rows to {second_target_path}")
    
    print("\nDone! All josh_rapm files updated.")
    
    # Print sample
    print("\nSample of new 2026 data from 2y_josh_rapm.csv:")
    sample_df = pd.read_csv(os.path.join(TARGET_DIR, "2y_josh_rapm.csv"))
    sample = sample_df[sample_df["end_season"] == END_SEASON].head(10)
    print(sample[["player_id", "player_name", "off", "def", "total_coefficient"]].to_string(index=False))


if __name__ == "__main__":
    main()

