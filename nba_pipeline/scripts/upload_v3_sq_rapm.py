#!/usr/bin/env python3
"""
Upload TS Decomposition (v3 SQ RAPM) to Supabase

Reads ts_decomp_factors_24_26_all_td700.csv, joins with player_stats
for team info, then upserts to the v3_sq_rapm table.

Usage:
    python upload_v3_sq_rapm.py
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
RESULTS_DIR = PIPELINE_ROOT / "results"

# Source file
SOURCE_FILE = RESULTS_DIR / "ts_decomp_24_26_all_td700" / "ts_decomp_factors_24_26_all_td700.csv"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

load_dotenv(PROJECT_ROOT / ".env")


def fetch_team_abbreviations(supabase: Client, nba_ids: list, year: int) -> dict:
    """Fetch TeamAbbreviation from player_stats_with_metrics_mat for given year."""
    logging.info(f"Fetching team abbreviations for year {year}...")

    # Query in batches (Supabase has URL length limits)
    id_to_team = {}
    batch_size = 200
    for i in range(0, len(nba_ids), batch_size):
        batch = nba_ids[i:i + batch_size]
        response = supabase.table("player_stats_with_metrics_mat") \
            .select('nba_id, "TeamAbbreviation"') \
            .eq("year", year) \
            .in_("nba_id", batch) \
            .execute()

        for row in response.data:
            if row['nba_id'] not in id_to_team and row.get('TeamAbbreviation'):
                id_to_team[row['nba_id']] = row['TeamAbbreviation']

    logging.info(f"  Found {len(id_to_team)} players for year {year}")
    return id_to_team


def main():
    # Validate source file
    if not SOURCE_FILE.exists():
        logging.error(f"Source file not found: {SOURCE_FILE}")
        logging.error("Run the td700 decomposition first:")
        logging.error("  python rapm.py TS 24 26 ALL --timedecay --half-life 700")
        logging.error("  python rapm.py SQ_POSS 24 26 ALL --timedecay --half-life 700")
        logging.error("  python rapm.py FT_PREMIUM 24 26 ALL --timedecay --half-life 700")
        logging.error("  python rapm.py CONTEST 24 26 ALL --timedecay --half-life 700")
        logging.error("  python ts_decomp_regression.py 24 26 ALL --timedecay")
        sys.exit(1)

    # Connect to Supabase
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        logging.error("SUPABASE_URL or SUPABASE_KEY missing in .env")
        sys.exit(1)

    supabase: Client = create_client(url, key)

    # Load decomposition CSV
    logging.info(f"Loading {SOURCE_FILE.name}...")
    df = pd.read_csv(SOURCE_FILE)
    logging.info(f"  Loaded {len(df)} players")

    # Rename player_id -> nba_id
    df = df.rename(columns={'player_id': 'nba_id'})

    # Fetch team abbreviations: try 2026, fallback to 2025
    nba_ids = df['nba_id'].tolist()
    team_map = fetch_team_abbreviations(supabase, nba_ids, 2026)

    missing_ids = [nid for nid in nba_ids if nid not in team_map]
    if missing_ids:
        fallback = fetch_team_abbreviations(supabase, missing_ids, 2025)
        team_map.update(fallback)
        logging.info(f"  After 2025 fallback: {len(team_map)} total players matched")

    df['team_abbreviation'] = df['nba_id'].map(team_map)
    df['year'] = 2026

    # Select columns matching table schema
    table_columns = [
        'nba_id', 'year', 'player_name', 'team_abbreviation',
        'oTS', 'dTS', 'net_ts',
        'oSQ', 'oFT', 'oCONTEST',
        'dSQ', 'dFT', 'dCONTEST',
        'oRESID', 'dRESID',
        'possessions', 'off_poss', 'def_poss'
    ]
    df = df[table_columns]

    # Convert to clean records
    records = df.to_dict(orient='records')
    clean_records = []
    for rec in records:
        clean = {}
        for k, v in rec.items():
            if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                clean[k] = None
            elif k in ('nba_id', 'year'):
                clean[k] = int(v)
            else:
                clean[k] = v
        clean_records.append(clean)

    # Delete all existing rows, then insert in batches
    table_name = "v3_sq_rapm"
    logging.info(f"Clearing existing {table_name} data...")
    supabase.table(table_name).delete().neq("nba_id", 0).execute()

    batch_size = 500
    total = len(clean_records)
    for i in range(0, total, batch_size):
        batch = clean_records[i:i + batch_size]
        supabase.table(table_name).insert(batch).execute()
        logging.info(f"  Inserted batch {i // batch_size + 1} ({min(i + batch_size, total)}/{total})")

    logging.info(f"Successfully uploaded {total} rows to {table_name}")

    # Summary
    logging.info("")
    logging.info("=" * 50)
    logging.info("UPLOAD SUMMARY")
    logging.info("=" * 50)
    logging.info(f"  Players uploaded: {total}")
    top5 = sorted(clean_records, key=lambda r: r.get('net_ts') or -999, reverse=True)[:5]
    logging.info("  Top 5 by net_ts:")
    for row in top5:
        logging.info(f"    {row['player_name']:25s} net={row['net_ts']:+.2f}  oSQ={row.get('oSQ', 0):+.2f}  oFT={row.get('oFT', 0):+.2f}  oCONT={row.get('oCONTEST', 0):+.2f}")


if __name__ == '__main__':
    main()
