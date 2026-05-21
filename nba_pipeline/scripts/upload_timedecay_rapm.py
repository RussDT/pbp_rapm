#!/usr/bin/env python3
"""
Upload Time-Decay RAPM Weighted Factors to Supabase

Reads a weighted_factors CSV from master_results, joins with player_stats
for team/position info, calculates ranks and percentiles, then upserts
to the timedecay_rapm table.

Usage:
    python upload_timedecay_rapm.py                          # default: 21-26 file
    python upload_timedecay_rapm.py --source-file ../master_results/weighted_factors_14_18_all_rb_td700.csv
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client

# Get the pipeline root directory
SCRIPT_DIR = Path(__file__).parent
PIPELINE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = PIPELINE_ROOT.parent
MASTER_RESULTS_DIR = PIPELINE_ROOT / "master_results"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

# Default source file
DEFAULT_SOURCE_FILE = MASTER_RESULTS_DIR / "weighted_factors_21_26_all_rb_td700.csv"

# Metrics that get ranks/percentiles (higher = better for all)
RANKED_METRICS = [
    'oTS', 'oTOV', 'oREB', 'dTS', 'dTOV', 'dREB',
    'off', 'def', 'net_rapm', 'pval'
]


def fetch_player_info(supabase: Client, year: int) -> pd.DataFrame:
    """Fetch team and position info from player_stats for the given year."""
    logging.info(f"Fetching player_stats for year {year}...")

    response = supabase.table("player_stats") \
        .select('nba_id, "TeamId", "TeamAbbreviation", "Pos2"') \
        .eq("year", year) \
        .eq("playoffs", 0) \
        .execute()

    if not response.data:
        logging.warning(f"No player_stats data for year {year}")
        return pd.DataFrame()

    df = pd.DataFrame(response.data)
    # Deduplicate: keep first row per nba_id (some players traded mid-season)
    df = df.drop_duplicates(subset='nba_id', keep='first')
    logging.info(f"  Fetched {len(df)} unique players from player_stats")
    return df


def compute_ranks_and_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate rank (1=best) and percentile (0-100) for each metric."""
    n = len(df)
    for metric in RANKED_METRICS:
        if metric not in df.columns:
            logging.warning(f"Metric {metric} not found in data, skipping")
            continue

        # Rank: 1 = highest value (best), ascending=False
        rank_col = f"{metric}_rank"
        pct_col = f"{metric}_pct"

        df[rank_col] = df[metric].rank(ascending=False, method='min').astype(int)

        # Percentile: (N - rank) / (N - 1) * 100
        if n > 1:
            df[pct_col] = ((n - df[rank_col]) / (n - 1) * 100).round(1)
        else:
            df[pct_col] = 100.0

    return df


def upload_to_supabase_records(supabase: Client, records: list, year: int) -> bool:
    """Delete existing rows for this year, then insert records in batches."""
    table_name = "timedecay_rapm"

    # Delete only rows for this specific year (preserves other years)
    logging.info(f"Clearing existing {table_name} data for year={year}...")
    supabase.table(table_name).delete().eq("year", year).execute()

    # Insert in batches of 500
    batch_size = 500
    total = len(records)

    for i in range(0, total, batch_size):
        batch = records[i:i + batch_size]
        supabase.table(table_name).insert(batch).execute()
        logging.info(f"  Inserted batch {i // batch_size + 1} ({min(i + batch_size, total)}/{total})")

    logging.info(f"Successfully uploaded {total} rows to {table_name} for year={year}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Upload time-decay RAPM to Supabase")
    parser.add_argument('--source-file', type=str, default=None,
                        help='Path to weighted_factors CSV (default: weighted_factors_21_26_all_rb_td700.csv)')
    args = parser.parse_args()

    source_file = Path(args.source_file) if args.source_file else DEFAULT_SOURCE_FILE

    # Validate source file
    if not source_file.exists():
        logging.error(f"Source file not found: {source_file}")
        sys.exit(1)

    # Connect to Supabase
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        logging.error("SUPABASE_URL or SUPABASE_KEY missing in .env")
        sys.exit(1)

    supabase: Client = create_client(url, key)

    # Load weighted factors CSV
    logging.info(f"Loading {source_file.name}...")
    df = pd.read_csv(source_file)
    logging.info(f"  Loaded {len(df)} players")

    # Get year from data
    year = int(df['Latest_Year'].iloc[0])
    logging.info(f"  Latest year: {year}")

    # Fetch player info from Supabase
    player_info = fetch_player_info(supabase, year)

    # Rename CSV columns to match table schema
    df = df.rename(columns={
        'player_id': 'nba_id',
        'Latest_Year': 'year',
    })

    # Left-join with player_stats for team/position
    if not player_info.empty:
        player_info = player_info.rename(columns={
            'TeamId': 'team_id',
            'TeamAbbreviation': 'team_abbreviation',
            'Pos2': 'pos2'
        })
        df = df.merge(player_info[['nba_id', 'team_id', 'team_abbreviation', 'pos2']],
                       on='nba_id', how='left')
        matched = df['team_id'].notna().sum()
        logging.info(f"  Matched {matched}/{len(df)} players with team/position info")
    else:
        df['team_id'] = None
        df['team_abbreviation'] = None
        df['pos2'] = None

    # Calculate pval
    df['pval'] = df['oTOV'] + df['oREB'] + df['dREB'] + df['dTOV']
    logging.info(f"  Calculated pval (mean={df['pval'].mean():.3f})")

    # Calculate ranks and percentiles
    df = compute_ranks_and_percentiles(df)

    # Select only the columns that match the table schema
    table_columns = [
        'nba_id', 'year', 'player_name', 'team_id', 'team_abbreviation', 'pos2',
        'oTS', 'oTOV', 'oREB', 'dTS', 'dTOV', 'dREB',
        'off', 'def', 'net_rapm', 'RESID', 'off_poss', 'pval',
        'oTS_rank', 'oTS_pct', 'oTOV_rank', 'oTOV_pct',
        'oREB_rank', 'oREB_pct', 'dTS_rank', 'dTS_pct',
        'dTOV_rank', 'dTOV_pct', 'dREB_rank', 'dREB_pct',
        'off_rank', 'off_pct', 'def_rank', 'def_pct',
        'net_rapm_rank', 'net_rapm_pct', 'pval_rank', 'pval_pct'
    ]
    df = df[table_columns]

    # Convert to records and clean for JSON serialization
    import numpy as np
    import math

    int_cols = {'nba_id', 'year', 'team_id'} | {f'{m}_rank' for m in RANKED_METRICS}
    records = df.to_dict(orient='records')
    clean_records = []
    for rec in records:
        clean = {}
        for k, v in rec.items():
            if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                clean[k] = None
            elif k in int_cols and v is not None:
                clean[k] = int(v)
            else:
                clean[k] = v
        clean_records.append(clean)

    # Upload to Supabase
    upload_to_supabase_records(supabase, clean_records, year)

    # Summary
    logging.info("")
    logging.info("=" * 50)
    logging.info("UPLOAD SUMMARY")
    logging.info("=" * 50)
    logging.info(f"  Players uploaded: {len(clean_records)}")
    top5 = sorted([r for r in clean_records if r.get('net_rapm_rank')],
                  key=lambda r: r['net_rapm_rank'])[:5]
    logging.info(f"  Top 5 by net_rapm:")
    for row in top5:
        logging.info(f"    {row['player_name']:25s} {row['net_rapm']:+.2f} (rank {row['net_rapm_rank']})")


if __name__ == '__main__':
    main()
