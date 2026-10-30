"""
Update the current-season slice of the josh_rapm CSV exports.

Despite the historical `2026` in this filename, the script is season
parameterized: pass `--year 2027` (or set `NBA_RAPM_SEASON_YEAR=2027`) to
rebuild the 2026-27 season. The season-year convention is that year N means
season (N-1)-N, and the two-digit forms used in result folder/file names are
derived from it.

Reads `weighted_factors_<start>_<end>_all_se.csv` from the per-window
`nba_pipeline/results/rapm_<start>_<end>_all_se` folders, divides values by
100, rounds to 4 decimals, flips `def` so negative is good defense, and
replaces the rows for the target `end_season` in each josh_rapm window file.

Paths:
  NBA_RAPM_RESULTS_DIR  RAPM results root (default <repo>/nba_pipeline/results)
  NBA_RAPM_JOSH_DIR     in-repo josh_rapm output dir (default <repo>/josh_rapm)
  NBA_RAPM_CSVS_DIR     csvs repo checkout (default <repo>/../csvs); the
                        josh_rapm subfolder inside it is the published target
"""

import argparse
import os
from pathlib import Path

import pandas as pd

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_END_SEASON = int(os.environ.get("NBA_RAPM_SEASON_YEAR", "2027"))

RESULTS_DIR = Path(
    os.environ.get("NBA_RAPM_RESULTS_DIR", PROJECT_ROOT / "nba_pipeline" / "results")
)
TARGET_DIR = Path(os.environ.get("NBA_RAPM_JOSH_DIR", PROJECT_ROOT / "josh_rapm"))
CSVS_DIR = Path(os.environ.get("NBA_RAPM_CSVS_DIR", PROJECT_ROOT.parent / "csvs"))
SECOND_TARGET_DIR = CSVS_DIR / "josh_rapm"

# Window lengths published as josh_rapm files, longest first (matches the
# historical SOURCE_FILES ordering).
WINDOW_LENGTHS = (5, 4, 3, 2)

# Target column order
TARGET_COLUMNS = [
    "start_season", "end_season", "player_id", "player_name",
    "off", "def", "total_coefficient",
    "fraction_player", "f_period", "vanilla_per_game", "nuanced_per_game"
]


def two_digit(season_year: int) -> str:
    """Two-digit season-end form used in result folder and file names."""
    return f"{int(season_year) % 100:02d}"


def build_source_files(end_season: int):
    """Build (target_file, source_folder, source_file, start_season) tuples.

    A window of length L ending in `end_season` starts at
    `end_season - L + 1`; e.g. the 5y window for 2026 is 2022-2026 and maps to
    `rapm_22_26_all_se/weighted_factors_22_26_all_se.csv`.
    """
    end_yy = two_digit(end_season)
    source_files = []
    for length in WINDOW_LENGTHS:
        start_season = end_season - length + 1
        range_token = f"{two_digit(start_season)}_{end_yy}"
        source_files.append(
            (
                f"{length}y_josh_rapm.csv",
                f"rapm_{range_token}_all_se",
                f"weighted_factors_{range_token}_all_se.csv",
                start_season,
            )
        )
    return source_files


def load_and_transform(
    source_folder: str, source_file: str, start_season: int, end_season: int
) -> pd.DataFrame:
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
    df["end_season"] = end_season
    df["fraction_player"] = 0
    df["f_period"] = 0
    df["vanilla_per_game"] = 0
    df["nuanced_per_game"] = 0

    # Select and order columns
    df = df[TARGET_COLUMNS]

    return df


def main(end_season: int = DEFAULT_END_SEASON):
    # Ensure second target directory exists
    os.makedirs(SECOND_TARGET_DIR, exist_ok=True)

    for target_file, source_folder, source_file, start_season in build_source_files(end_season):
        target_path = os.path.join(TARGET_DIR, target_file)
        second_target_path = os.path.join(SECOND_TARGET_DIR, target_file)
        print(f"\nProcessing {target_file} (Start: {start_season}, End: {end_season})...")

        # Load existing data from main target (or second target if main doesn't exist)
        if os.path.exists(target_path):
            existing_df = pd.read_csv(target_path)
        elif os.path.exists(second_target_path):
            existing_df = pd.read_csv(second_target_path)
        else:
            print(f"  Warning: {target_file} not found in either {TARGET_DIR} or {SECOND_TARGET_DIR}")
            continue

        original_count = len(existing_df)

        # Remove existing rows for the target season
        existing_df = existing_df[existing_df["end_season"] != end_season]
        removed_count = original_count - len(existing_df)
        print(f"  Removed {removed_count} existing {end_season} rows")

        # Load and transform new data for the target season
        new_df = load_and_transform(source_folder, source_file, start_season, end_season)
        print(f"  Adding {len(new_df)} new {end_season} rows")

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
    print(f"\nSample of new {end_season} data from 2y_josh_rapm.csv:")
    sample_df = pd.read_csv(os.path.join(TARGET_DIR, "2y_josh_rapm.csv"))
    sample = sample_df[sample_df["end_season"] == end_season].head(10)
    print(sample[["player_id", "player_name", "off", "def", "total_coefficient"]].to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--year",
        type=int,
        default=DEFAULT_END_SEASON,
        help="Season end year to rebuild (default %(default)s; 2027 = 2026-27)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args().year)
