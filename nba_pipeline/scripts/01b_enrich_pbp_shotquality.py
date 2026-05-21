#!/usr/bin/env python3
"""
Enrich raw PBP parquets with ShotQuality data.

Adds columns to raw_data/NBA*.parquet files:
- is_transition: bool
- is_transition_dunk_or_layup: bool
- initial_ev: float (shot expected value at release)
- dynamic_ev: float (shot expected value with game context)
- sq_shooter_id: int (shooter NBA player ID from SQ)
- sq_defender_id: int (closest defender NBA player ID from SQ)
- sq_play_descriptors: raw ShotQuality descriptor list string
- sq_descriptor_bundle: normalized descriptor bundle string
- sq_desc_*: boolean descriptor columns for stable downstream modeling

Usage:
    python 01b_enrich_pbp_shotquality.py           # All years (24, 25, 26) + PS
    python 01b_enrich_pbp_shotquality.py 24        # Single year (RS + PS)
    python 01b_enrich_pbp_shotquality.py 24 25     # Specific years
"""

import pandas as pd
import ast
import sys
from pathlib import Path
from glob import glob

# Paths
SQ_BASE = Path('/Users/russellthomas/Docs/2026_shotquality/nba_games')
RAW_BASE = Path(__file__).parent.parent / 'raw_data'
SQ_DESCRIPTOR_TAGS = [
    'alley_oop',
    'catch_and_shoot',
    'dunk',
    'floater',
    'hook',
    'illegal_defense',
    'in_paint',
    'jumpshot',
    'lay_up',
    'off_cut',
    'off_drive',
    'pick_and_roll',
    'post_up',
    'pull_up',
    'reverse',
    'step_back',
    'tip_in',
    'transition',
    'turnaround',
    'unsportsmanlike',
]


def parse_descriptors(desc):
    """Parse play_descriptors string into list of tags"""
    if pd.isna(desc):
        return []
    try:
        return ast.literal_eval(desc)
    except:
        return []


def descriptor_col_name(tag: str) -> str:
    """Convert a ShotQuality descriptor tag into a stable boolean column name."""
    return f"sq_desc_{tag}"


def normalize_descriptor_bundle(tags) -> str:
    """Convert descriptor tags into a stable, order-independent bundle string."""
    cleaned = sorted({str(tag).strip() for tag in tags if str(tag).strip()})
    return '|'.join(cleaned)


def load_shotquality_year(year: int) -> pd.DataFrame:
    """Load all shotquality CSVs for a year into single DataFrame"""
    sq_folder = SQ_BASE / str(year)
    if not sq_folder.exists():
        raise FileNotFoundError(f'No shotquality folder: {sq_folder}')

    sq_files = list(sq_folder.glob('*.csv'))
    print(f'  Loading {len(sq_files)} shotquality files...')

    required_cols = ['game_id', 'GAME_EVENT_ID', 'play_descriptors',
                     'ev.initial_ev.value', 'personId']
    optional_cols = ['ev.dynamic_ev.value', 'opp_personId']

    dfs = []
    for f in sq_files:
        try:
            # Read header to check available columns
            header = pd.read_csv(f, nrows=0).columns.tolist()
            use_cols = [c for c in required_cols if c in header]
            if len(use_cols) < len(required_cols):
                missing = set(required_cols) - set(header)
                print(f'    Warning: {f.name} missing required cols: {missing}')
                continue
            use_cols += [c for c in optional_cols if c in header]
            df = pd.read_csv(f, usecols=use_cols)
            # Add missing optional columns as NaN
            for c in optional_cols:
                if c not in df.columns:
                    df[c] = pd.NA
            dfs.append(df)
        except Exception as e:
            print(f'    Warning: Error reading {f.name}: {e}')

    if not dfs:
        raise ValueError(f'No valid shotquality data for {year}')

    sq = pd.concat(dfs, ignore_index=True)

    # Parse play_descriptors and create derived columns
    sq['tags'] = sq['play_descriptors'].apply(parse_descriptors)
    sq['is_transition'] = sq['tags'].apply(lambda x: 'transition' in x)
    sq['is_transition_dunk_or_layup'] = sq['tags'].apply(
        lambda x: 'transition' in x and ('dunk' in x or 'lay_up' in x)
    )
    sq['sq_descriptor_bundle'] = sq['tags'].apply(normalize_descriptor_bundle)
    for tag in SQ_DESCRIPTOR_TAGS:
        sq[descriptor_col_name(tag)] = sq['tags'].apply(lambda x, tag=tag: tag in x)

    # Select and rename columns for merge
    descriptor_cols = [descriptor_col_name(tag) for tag in SQ_DESCRIPTOR_TAGS]
    sq_merge = sq[['game_id', 'GAME_EVENT_ID', 'play_descriptors', 'sq_descriptor_bundle',
                   'is_transition', 'is_transition_dunk_or_layup', 'ev.initial_ev.value',
                   'ev.dynamic_ev.value', 'personId', 'opp_personId', *descriptor_cols]].copy()
    sq_merge = sq_merge.rename(columns={
        'GAME_EVENT_ID': 'event_num',
        'play_descriptors': 'sq_play_descriptors',
        'ev.initial_ev.value': 'initial_ev',
        'ev.dynamic_ev.value': 'dynamic_ev',
        'personId': 'sq_shooter_id',
        'opp_personId': 'sq_defender_id'
    })
    sq_merge['event_num'] = sq_merge['event_num'].astype('Int64')

    return sq_merge


def enrich_parquet(raw_file: Path, sq_data: pd.DataFrame) -> bool:
    """Enrich a single raw PBP parquet with shotquality data"""
    if not raw_file.exists():
        print(f'  Skipping {raw_file.name}: file not found')
        return False

    print(f'  Processing {raw_file.name}...')

    # Load raw PBP
    raw = pd.read_parquet(raw_file)
    initial_cols = len(raw.columns)

    # Drop existing SQ columns if re-running
    sq_cols = ['is_transition', 'is_transition_dunk_or_layup', 'initial_ev',
               'dynamic_ev', 'sq_shooter_id', 'sq_defender_id',
               'sq_play_descriptors', 'sq_descriptor_bundle'] + [
                   descriptor_col_name(tag) for tag in SQ_DESCRIPTOR_TAGS
               ]
    existing_sq_cols = [c for c in sq_cols if c in raw.columns]
    if existing_sq_cols:
        raw = raw.drop(columns=existing_sq_cols)
        print(f'    Dropped existing SQ columns: {existing_sq_cols}')

    # Normalize game_id types (raw has '0022500014', SQ has 22500014)
    raw_game_id_orig = raw['game_id'].copy()  # preserve original for saving
    raw['game_id'] = raw['game_id'].astype(str).str.lstrip('0')
    sq_data = sq_data.copy()
    sq_data['game_id'] = sq_data['game_id'].astype(str).str.lstrip('0')

    # Filter SQ data to only games in this parquet
    game_ids = set(raw['game_id'].unique())
    sq_filtered = sq_data[sq_data['game_id'].isin(game_ids)].copy()

    # Normalize event_num types and dedupe SQ (multiple SQ rows per event possible)
    sq_filtered['event_num'] = sq_filtered['event_num'].astype('Int64')
    raw['event_num'] = raw['event_num'].astype('Int64')
    sq_filtered = sq_filtered.drop_duplicates(subset=['game_id', 'event_num'], keep='first')

    # Merge
    enriched = raw.merge(sq_filtered, on=['game_id', 'event_num'], how='left')

    # Stats
    has_sq = enriched['is_transition'].notna().sum()
    trans = enriched['is_transition'].sum() if 'is_transition' in enriched else 0
    trans_dl = enriched['is_transition_dunk_or_layup'].sum() if 'is_transition_dunk_or_layup' in enriched else 0

    print(f'    Rows: {len(enriched):,} | With SQ: {has_sq:,} ({has_sq/len(enriched)*100:.1f}%)')
    print(f'    Transition: {trans:,} shots, {trans_dl:,} dunks/layups')
    print(f'    Columns: {initial_cols} -> {len(enriched.columns)}')

    # Restore original game_id format before saving
    enriched['game_id'] = raw_game_id_orig.values

    # Save back
    enriched.to_parquet(raw_file, index=False)
    print(f'    Saved: {raw_file.name}')

    return True


def process_year(year: int):
    """Process regular season and playoffs for a year"""
    suffix = str(year)[2:]

    print(f'\n=== Processing {year} ===')

    # Load shotquality data for this year
    try:
        sq_data = load_shotquality_year(year)
        print(f'  Loaded {len(sq_data):,} shots from shotquality')
    except FileNotFoundError as e:
        print(f'  {e}')
        return

    # Process regular season
    rs_file = RAW_BASE / f'NBA{suffix}.parquet'
    enrich_parquet(rs_file, sq_data)

    # Process playoffs
    ps_file = RAW_BASE / f'NBA{suffix}_PS.parquet'
    enrich_parquet(ps_file, sq_data)


def main():
    # Determine years to process
    if len(sys.argv) > 1:
        years = [2000 + int(y) if int(y) < 100 else int(y) for y in sys.argv[1:]]
    else:
        years = [2024, 2025, 2026]

    print(f'Enriching raw PBP with ShotQuality data')
    print(f'Years: {years}')

    for year in years:
        process_year(year)

    print('\nDone!')


if __name__ == '__main__':
    main()
