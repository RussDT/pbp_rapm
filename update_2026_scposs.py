"""
Script to update 2026 scposs data in individual player CSV files.
Reads weighted_factors.csv and pure_results.csv from Results folders,
transforms data, and updates ../csvs/scposs/{nba_id}.csv files.
"""

import pandas as pd
import os
from pathlib import Path

# Configuration
RESULTS_DIR = "nba_pipeline/results"
TARGET_DIR = "../csvs/scposs"
LATEST_YEAR = 2026

# Source files mapping: (folder, weighted_factors, year_interval)
SOURCE_FILES = [
    ("rapm_26_all", "weighted_factors_26_all.csv", "1Y"),
    ("rapm_25_26_all", "weighted_factors_25_26_all.csv", "2Y"),
    ("rapm_24_26_all", "weighted_factors_24_26_all.csv", "3Y"),
    ("rapm_23_26_all", "weighted_factors_23_26_all.csv", "4Y"),
    ("rapm_22_26_all", "weighted_factors_22_26_all.csv", "5Y"),
    ("rapm_21_26_all", "weighted_factors_21_26_all.csv", "6Y"),
]

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


def main():
    # Load and transform all source files
    all_new_data = []
    for folder, wf_file, year_interval in SOURCE_FILES:
        df = load_and_transform(folder, wf_file, year_interval)
        all_new_data.append(df)
        print(f"  -> {len(df)} rows")
    
    # Combine all new 2026 data
    new_2026_df = pd.concat(all_new_data, ignore_index=True)
    print(f"\nTotal new 2026 rows before ranks: {len(new_2026_df)}")
    
    # Calculate ranks
    new_2026_df = calculate_ranks(new_2026_df)
    
    # Select and order columns
    new_2026_df = new_2026_df[TARGET_COLUMNS]
    
    print(f"Total new 2026 rows: {len(new_2026_df)}")
    
    # Get unique player IDs
    unique_players = new_2026_df["nba_id"].unique()
    print(f"Unique players to update: {len(unique_players)}")
    
    # Ensure target directory exists
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    # Track stats
    updated_files = 0
    created_files = 0
    
    # Process each player
    for nba_id in unique_players:
        player_file = os.path.join(TARGET_DIR, f"{nba_id}.csv")
        player_new_data = new_2026_df[new_2026_df["nba_id"] == nba_id]
        
        if os.path.exists(player_file):
            # Load existing data
            existing_df = pd.read_csv(player_file)
            
            # Remove existing 2026 rows
            existing_df = existing_df[existing_df["Latest_Year"] != LATEST_YEAR]
            
            # Append new 2026 data
            combined_df = pd.concat([existing_df, player_new_data], ignore_index=True)
            
            # Sort by Latest_Year and Year_Interval
            combined_df = combined_df.sort_values(["Latest_Year", "Year_Interval"])
            
            updated_files += 1
        else:
            # Create new file with just the 2026 data
            combined_df = player_new_data.sort_values(["Latest_Year", "Year_Interval"])
            created_files += 1
        
        # Save to file
        combined_df.to_csv(player_file, index=False)
    
    print(f"\nUpdated {updated_files} existing player files")
    print(f"Created {created_files} new player files")
    
    # Print sample of new data
    print("\nSample of new 2026 data (first 5 rows from 2Y):")
    sample = new_2026_df[new_2026_df["Year_Interval"] == "2Y"].head(5)
    print(sample[["nba_id", "player_name", "Year_Interval", "Off_RAPM", "Def_RAPM", "OVR_RAPM", "sc_POSS"]].to_string(index=False))


if __name__ == "__main__":
    main()

