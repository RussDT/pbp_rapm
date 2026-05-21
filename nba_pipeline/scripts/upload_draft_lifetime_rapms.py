#!/usr/bin/env python3
"""
Upload Draft Lifetime RAPM to Supabase

Combines two data sources into a single lifetime RAPM estimate per player:
  1. Modern: weighted_factors_14_26_all_rb.csv (2014-2026, rubberband-adjusted)
  2. Historical: SCALEDOUTPUT_SMALLER.csv filtered to Latest_Year=2013, Year_Interval=5Y (~2009-2013)

For players in both sources, values are possession-weighted averages.
For players in only one source, values are used directly.
RESID is recalculated as net_rapm - sum(6 factors).

Usage:
    python upload_draft_lifetime_rapms.py
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

# Data source paths
MODERN_CSV = MASTER_RESULTS_DIR / "weighted_factors_14_26_all_rb.csv"
HISTORICAL_CSV = PROJECT_ROOT.parent / "csvs" / "SCALEDOUTPUT_SMALLER.csv"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

# Columns shared between both sources (after renaming)
FACTOR_COLS = ['oTS', 'oTOV', 'oREB', 'dTS', 'dTOV', 'dREB']
RAPM_COLS = ['off', 'def', 'net_rapm']
WEIGHTED_COLS = FACTOR_COLS + RAPM_COLS

# Historical column rename mapping
HIST_RENAME = {
    'nba_id': 'nba_id',
    'player_name': 'player_name',
    'Off_RAPM': 'off',
    'Def_RAPM': 'def',
    'OVR_RAPM': 'net_rapm',
    'sc_OFF_TS': 'oTS',
    'sc_OFF_TOV': 'oTOV',
    'sc_OFF_REB': 'oREB',
    'sc_DEF_TS': 'dTS',
    'sc_DEF_TOV': 'dTOV',
    'sc_DEF_REB': 'dREB',
    'Off_Poss': 'off_poss',
}


def load_modern() -> pd.DataFrame:
    """Load the modern (2014-2026) weighted factors CSV."""
    logging.info(f"Loading modern data: {MODERN_CSV.name}")
    df = pd.read_csv(MODERN_CSV)
    df = df.rename(columns={'player_id': 'nba_id'})
    df = df[['nba_id', 'player_name'] + WEIGHTED_COLS + ['off_poss']]
    logging.info(f"  Loaded {len(df)} players (2014-2026)")
    return df


def load_historical() -> pd.DataFrame:
    """Load the historical (~2009-2013) data from SCALEDOUTPUT."""
    logging.info(f"Loading historical data: {HISTORICAL_CSV.name}")
    df = pd.read_csv(HISTORICAL_CSV)
    df = df[(df['Latest_Year'] == 2013) & (df['Year_Interval'] == '5Y')]
    logging.info(f"  Filtered to {len(df)} players (2013/5Y)")

    # Rename columns to match modern schema
    df = df.rename(columns=HIST_RENAME)
    df = df[['nba_id', 'player_name'] + WEIGHTED_COLS + ['off_poss']]
    return df


def combine_sources(modern: pd.DataFrame, historical: pd.DataFrame) -> pd.DataFrame:
    """Possession-weight combine the two sources."""
    # Merge on nba_id with indicator
    merged = modern.merge(
        historical,
        on='nba_id',
        how='outer',
        suffixes=('_m', '_h'),
        indicator=True,
    )

    rows = []
    for _, r in merged.iterrows():
        row = {'nba_id': int(r['nba_id'])}

        if r['_merge'] == 'left_only':
            # Modern only
            row['player_name'] = r['player_name_m']
            for col in WEIGHTED_COLS:
                row[col] = r[f'{col}_m']
            row['off_poss'] = r['off_poss_m']

        elif r['_merge'] == 'right_only':
            # Historical only
            row['player_name'] = r['player_name_h']
            for col in WEIGHTED_COLS:
                row[col] = r[f'{col}_h']
            row['off_poss'] = r['off_poss_h']

        else:
            # Both sources — possession-weighted average
            # Use modern name (more recent)
            row['player_name'] = r['player_name_m']
            poss_m = r['off_poss_m']
            poss_h = r['off_poss_h']
            total_poss = poss_m + poss_h

            for col in WEIGHTED_COLS:
                val_m = r[f'{col}_m']
                val_h = r[f'{col}_h']
                row[col] = (val_m * poss_m + val_h * poss_h) / total_poss

            row['off_poss'] = total_poss

        rows.append(row)

    result = pd.DataFrame(rows)

    # Recalculate RESID = net_rapm - sum(6 factors)
    result['RESID'] = result['net_rapm'] - result[FACTOR_COLS].sum(axis=1)

    # Count sources
    n_modern = (merged['_merge'] == 'left_only').sum()
    n_hist = (merged['_merge'] == 'right_only').sum()
    n_both = (merged['_merge'] == 'both').sum()
    logging.info(f"  Combined: {n_modern} modern-only, {n_hist} historical-only, {n_both} both → {len(result)} total")

    return result


def upload_to_supabase(supabase: Client, records: list) -> bool:
    """Clear table and insert records in batches."""
    table_name = "draft_lifetime_rapms"

    logging.info(f"Clearing existing {table_name} data...")
    supabase.table(table_name).delete().neq("nba_id", -1).execute()

    batch_size = 500
    total = len(records)
    for i in range(0, total, batch_size):
        batch = records[i:i + batch_size]
        supabase.table(table_name).insert(batch).execute()
        logging.info(f"  Inserted batch {i // batch_size + 1} ({min(i + batch_size, total)}/{total})")

    logging.info(f"Successfully uploaded {total} rows to {table_name}")
    return True


def main():
    # Validate source files
    for path in [MODERN_CSV, HISTORICAL_CSV]:
        if not path.exists():
            logging.error(f"Source file not found: {path}")
            sys.exit(1)

    # Connect to Supabase
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        logging.error("SUPABASE_URL or SUPABASE_KEY missing in .env")
        sys.exit(1)

    supabase: Client = create_client(url, key)

    # Load and combine
    modern = load_modern()
    historical = load_historical()
    combined = combine_sources(modern, historical)

    # Sort by net_rapm descending
    combined = combined.sort_values('net_rapm', ascending=False).reset_index(drop=True)

    # Clean for JSON serialization
    int_cols = {'nba_id', 'off_poss'}
    records = combined.to_dict(orient='records')
    clean_records = []
    for rec in records:
        clean = {}
        for k, v in rec.items():
            if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                clean[k] = None
            elif k in int_cols and v is not None:
                clean[k] = int(v)
            else:
                clean[k] = round(v, 4) if isinstance(v, float) else v
        clean_records.append(clean)

    # Upload
    upload_to_supabase(supabase, clean_records)

    # Summary
    logging.info("")
    logging.info("=" * 50)
    logging.info("UPLOAD SUMMARY")
    logging.info("=" * 50)
    logging.info(f"  Players uploaded: {len(clean_records)}")
    logging.info(f"  Top 10 by net_rapm:")
    for row in clean_records[:10]:
        logging.info(f"    {row['player_name']:25s} {row['net_rapm']:+.2f}  (off_poss={row['off_poss']})")


if __name__ == '__main__':
    main()
