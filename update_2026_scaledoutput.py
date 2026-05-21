"""
Script to update 2026 data in SCALEDOUTPUT_SMALLER.csv
Uses the same transformation as scposs but only for 3Y and 5Y data.
"""

import pandas as pd
import os

# Configuration
RESULTS_DIR = "nba_pipeline/results"
TARGET_FILE = "../csvs/SCALEDOUTPUT_SMALLER.csv"
LATEST_YEAR = 2026

# Source files mapping: only 3Y and 5Y (folder, weighted_factors, year_interval)
SOURCE_FILES = [
    ("rapm_24_26_all", "weighted_factors_24_26_all.csv", "3Y"),
    ("rapm_22_26_all", "weighted_factors_22_26_all.csv", "5Y"),
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


def main():
    # Load and transform source files (3Y and 5Y only)
    all_new_data = []
    for folder, wf_file, year_interval in SOURCE_FILES:
        df = load_and_transform(folder, wf_file, year_interval)
        all_new_data.append(df)
        print(f"  -> {len(df)} rows")
    
    # Combine new 2026 data
    new_2026_df = pd.concat(all_new_data, ignore_index=True)
    print(f"\nTotal new 2026 rows before ranks: {len(new_2026_df)}")
    
    # Calculate ranks
    new_2026_df = calculate_ranks(new_2026_df)
    
    # Select and order columns
    new_2026_df = new_2026_df[TARGET_COLUMNS]
    
    print(f"Total new 2026 rows: {len(new_2026_df)}")
    
    # Load existing data
    print(f"\nLoading existing {TARGET_FILE}...")
    existing_df = pd.read_csv(TARGET_FILE)
    print(f"  -> {len(existing_df)} total rows")
    
    # Remove existing 2026 rows
    existing_2026_count = (existing_df["Latest_Year"] == LATEST_YEAR).sum()
    print(f"  -> {existing_2026_count} existing 2026 rows to replace")
    
    non_2026_df = existing_df[existing_df["Latest_Year"] != LATEST_YEAR]
    print(f"  -> {len(non_2026_df)} non-2026 rows to keep")
    
    # Combine
    final_df = pd.concat([non_2026_df, new_2026_df], ignore_index=True)
    
    # Sort
    final_df = final_df.sort_values(["nba_id", "Latest_Year", "Year_Interval"])
    
    # Save
    print(f"\nSaving to {TARGET_FILE}...")
    final_df.to_csv(TARGET_FILE, index=False)
    print(f"  -> {len(final_df)} total rows written")
    
    # Sample
    print("\nSample of new 2026 data:")
    sample = new_2026_df.head(5)
    print(sample[["nba_id", "player_name", "Year_Interval", "Off_RAPM", "Def_RAPM", "OVR_RAPM"]].to_string(index=False))


if __name__ == "__main__":
    main()

