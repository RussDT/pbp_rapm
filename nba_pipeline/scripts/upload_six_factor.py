#!/usr/bin/env python3
"""
Upload Six-Factor RAPM to Supabase

Reads multiple weighted_factors CSVs from master_results (different year windows),
calculates ranks and percentiles per window, then upserts to the six_factor table.

Uploads:
  - 21_26 through 26_all (no td, no rb) → rb=false, timedecay=false
  - 14_26_all_rb → rb=true, timedecay=false

Usage:
    python upload_six_factor.py
"""

import logging
import math
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client

# Paths
SCRIPT_DIR = Path(__file__).parent
PIPELINE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = PIPELINE_ROOT.parent
MASTER_RESULTS_DIR = PIPELINE_ROOT / "master_results"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

load_dotenv(PROJECT_ROOT / ".env")

# Files to upload: (filename, year_interval, rb)
SOURCE_FILES = [
    ("weighted_factors_26_all.csv", "1Y", False),
    ("weighted_factors_25_26_all.csv", "2Y", False),
    ("weighted_factors_24_26_all.csv", "3Y", False),
    ("weighted_factors_23_26_all.csv", "4Y", False),
    ("weighted_factors_22_26_all.csv", "5Y", False),
    ("weighted_factors_21_26_all.csv", "6Y", False),
    ("weighted_factors_14_26_all_rb.csv", "13Y", True),
]

# Column rename: CSV → table
COLUMN_MAP = {
    'player_id': 'nba_id',
    'Latest_Year': 'year',
    'oTS': 'off_ts',
    'oTOV': 'off_tov',
    'oREB': 'off_reb',
    'dTS': 'def_ts',
    'dTOV': 'def_tov',
    'dREB': 'def_reb',
    'off': 'off_rapm',
    'def': 'def_rapm',
    'net_rapm': 'ovr_rapm',
}

# Metrics that get ranks/percentiles (higher = better for all)
RANKED_METRICS = [
    'off_ts', 'off_tov', 'off_reb', 'def_ts', 'def_tov', 'def_reb',
    'off_rapm', 'def_rapm', 'ovr_rapm', 'poss_val'
]

# Table columns in order
TABLE_COLUMNS = [
    'nba_id', 'player_name', 'year', 'year_interval',
    'off_rapm', 'def_rapm', 'ovr_rapm', 'off_poss',
    'off_ts', 'off_tov', 'off_reb', 'def_ts', 'def_tov', 'def_reb',
    'poss_val', 'off_diff', 'def_diff',
    'rb', 'timedecay',
    'off_rapm_rk', 'off_rapm_pct',
    'def_rapm_rk', 'def_rapm_pct',
    'off_ts_rk', 'off_ts_pct',
    'off_tov_rk', 'off_tov_pct',
    'off_reb_rk', 'off_reb_pct',
    'def_ts_rk', 'def_ts_pct',
    'def_tov_rk', 'def_tov_pct',
    'def_reb_rk', 'def_reb_pct',
    'ovr_rapm_rk', 'ovr_rapm_pct',
    'poss_val_rk', 'poss_val_pct',
]


def compute_ranks_and_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate rank (1=best) and percentile (0-1) for each metric."""
    n = len(df)
    for metric in RANKED_METRICS:
        if metric not in df.columns:
            continue

        rank_col = f"{metric}_rk"
        pct_col = f"{metric}_pct"

        df[rank_col] = df[metric].rank(ascending=False, method='min').astype(int)

        # Percentile: 0-1 scale (matching existing table convention)
        if n > 1:
            df[pct_col] = ((n - df[rank_col]) / (n - 1)).round(2)
        else:
            df[pct_col] = 1.0

    return df


def clean_records(records_list: list) -> list:
    """Convert to clean JSON-serializable records."""
    int_cols = {'nba_id', 'year'} | {f'{m}_rk' for m in RANKED_METRICS}
    bool_cols = {'rb', 'timedecay'}

    clean = []
    for rec in records_list:
        row = {}
        for k, v in rec.items():
            if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                row[k] = None
            elif k in int_cols and v is not None:
                row[k] = int(v)
            elif k in bool_cols:
                row[k] = bool(v)
            else:
                row[k] = v
        clean.append(row)
    return clean


def main():
    # Connect to Supabase
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        logging.error("SUPABASE_URL or SUPABASE_KEY missing in .env")
        sys.exit(1)

    supabase: Client = create_client(url, key)

    # Get current year from the first available file
    year = None
    all_records = []

    for filename, year_interval, is_rb in SOURCE_FILES:
        filepath = MASTER_RESULTS_DIR / filename
        if not filepath.exists():
            logging.warning(f"Source file not found, skipping: {filename}")
            continue

        logging.info(f"\nProcessing {filename} (interval={year_interval}, rb={is_rb})...")
        df = pd.read_csv(filepath)
        logging.info(f"  Loaded {len(df)} players")

        # Rename columns
        df = df.rename(columns=COLUMN_MAP)

        # Get year from data
        if year is None:
            year = int(df['year'].iloc[0])
            logging.info(f"  Target year: {year}")

        # Add metadata
        df['year_interval'] = year_interval
        df['rb'] = is_rb
        df['timedecay'] = False

        # Calculate poss_val (sum of secondary factors)
        df['poss_val'] = (df['off_tov'] + df['off_reb'] + df['def_reb'] + df['def_tov']).round(2)

        # Calculate off_diff and def_diff (residuals)
        df['off_diff'] = (df['off_rapm'] - (df['off_ts'] + df['off_tov'] + df['off_reb'])).round(2)
        df['def_diff'] = (df['def_rapm'] - (df['def_ts'] + df['def_tov'] + df['def_reb'])).round(2)

        # Compute ranks/percentiles within this interval
        df = compute_ranks_and_percentiles(df)

        # Select table columns
        df = df[TABLE_COLUMNS]

        records = clean_records(df.to_dict(orient='records'))
        all_records.extend(records)
        logging.info(f"  Prepared {len(records)} records")

    if not all_records:
        logging.error("No records to upload")
        sys.exit(1)

    # Delete existing rows for this year, then insert
    table_name = "six_factor"
    logging.info(f"\nClearing existing {table_name} data for year={year}...")
    supabase.table(table_name).delete().eq("year", year).execute()

    # Insert in batches
    batch_size = 500
    total = len(all_records)
    for i in range(0, total, batch_size):
        batch = all_records[i:i + batch_size]
        supabase.table(table_name).insert(batch).execute()
        logging.info(f"  Inserted batch {i // batch_size + 1} ({min(i + batch_size, total)}/{total})")

    logging.info(f"\nSuccessfully uploaded {total} rows to {table_name} for year={year}")

    # Summary
    logging.info("")
    logging.info("=" * 60)
    logging.info("SIX FACTOR UPLOAD SUMMARY")
    logging.info("=" * 60)
    logging.info(f"  Year: {year}")
    logging.info(f"  Total rows uploaded: {total}")

    from collections import Counter
    interval_counts = Counter(r['year_interval'] for r in all_records)
    for interval, cnt in sorted(interval_counts.items(), key=lambda x: int(x[0].replace('Y', ''))):
        rb_flag = " (rb)" if any(r['rb'] for r in all_records if r['year_interval'] == interval) else ""
        logging.info(f"    {interval}: {cnt} players{rb_flag}")

    # Top 5 by ovr_rapm from 6Y window
    six_year = [r for r in all_records if r.get('year_interval') == '6Y']
    if six_year:
        top5 = sorted(six_year, key=lambda r: r.get('ovr_rapm') or -999, reverse=True)[:5]
        logging.info(f"\n  Top 5 by ovr_rapm (6Y):")
        for row in top5:
            logging.info(f"    {row['player_name']:25s} {row['ovr_rapm']:+.2f} (rk {row['ovr_rapm_rk']})")


if __name__ == '__main__':
    main()
