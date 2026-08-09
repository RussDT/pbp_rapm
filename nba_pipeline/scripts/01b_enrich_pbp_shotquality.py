#!/usr/bin/env python3
"""
Enrich raw PBP parquets with ShotQuality data.

Adds columns to raw_data/NBA*.parquet files:
- is_transition: bool
- is_transition_dunk_or_layup: bool
- initial_ev: float (shot expected value at release)
- dynamic_ev: float (shot expected value with game context)
- sq_shooter_id: int (shooter NBA player ID from SQ)
- sq_primary_defender_id: int (ShotQuality closest-defender NBA player ID)
- sq_match_status: exact-key, provider-ambiguous, raw-key-ambiguous, or unmatched
- sq_match_candidate_count: distinct ShotQuality candidates for the event key
- sq_source_row_count: raw ShotQuality rows for the event key
- sq_provider: provider provenance (`shotquality`)
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
import numpy as np
from pathlib import Path

# Paths
SQ_BASE = Path('/Users/russellthomas/Docs/2026_shotquality/nba_games')
RAW_BASE = Path(__file__).parent.parent / 'raw_data'
SQ_JOIN_KEYS = ['game_id', 'event_num']
SQ_MATCH_DIAGNOSTIC_COLUMNS = [
    'sq_provider',
    'sq_match_status',
    'sq_match_candidate_count',
    'sq_source_row_count',
]
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
        'opp_personId': 'sq_primary_defender_id'
    })
    sq_merge['event_num'] = sq_merge['event_num'].astype('Int64')

    return sq_merge


def _normalized_game_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lstrip('0')


def resolve_shotquality_candidates(sq_data: pd.DataFrame) -> pd.DataFrame:
    """Return one explicit ShotQuality assignment or ambiguity row per event key.

    Exact duplicate provider rows are harmless and collapse after all payload
    columns agree. Distinct rows on the same key are not resolved arbitrarily:
    their payload is nulled and the key is retained as provider-ambiguous.
    """
    missing = set(SQ_JOIN_KEYS) - set(sq_data.columns)
    if missing:
        raise ValueError(f'ShotQuality data missing join columns: {sorted(missing)}')

    work = sq_data.copy()
    work['game_id'] = _normalized_game_id(work['game_id'])
    work['event_num'] = pd.to_numeric(work['event_num'], errors='coerce').astype('Int64')
    work = work.dropna(subset=SQ_JOIN_KEYS)

    payload_columns = [
        col for col in work.columns
        if col not in {*SQ_JOIN_KEYS, *SQ_MATCH_DIAGNOSTIC_COLUMNS}
    ]
    output_columns = [
        *SQ_JOIN_KEYS,
        *payload_columns,
        *SQ_MATCH_DIAGNOSTIC_COLUMNS,
    ]
    if work.empty:
        return pd.DataFrame(columns=output_columns)

    source_counts = (
        work.groupby(SQ_JOIN_KEYS, dropna=False)
        .size()
        .rename('sq_source_row_count')
        .reset_index()
    )

    distinct = work.drop_duplicates(subset=[*SQ_JOIN_KEYS, *payload_columns])
    candidate_counts = (
        distinct.groupby(SQ_JOIN_KEYS, dropna=False)
        .size()
        .rename('sq_match_candidate_count')
        .reset_index()
    )
    counts = source_counts.merge(
        candidate_counts,
        on=SQ_JOIN_KEYS,
        how='inner',
        validate='one_to_one',
    )

    unique = distinct.merge(
        counts,
        on=SQ_JOIN_KEYS,
        how='inner',
        validate='many_to_one',
    )
    unique = unique[unique['sq_match_candidate_count'].eq(1)].copy()
    unique['sq_provider'] = 'shotquality'
    unique['sq_match_status'] = 'exact_event_key'

    ambiguous = counts[counts['sq_match_candidate_count'].gt(1)].copy()
    for col in payload_columns:
        ambiguous[col] = pd.NA
    ambiguous['sq_provider'] = 'shotquality'
    ambiguous['sq_match_status'] = 'ambiguous_provider_event_key'

    frames = [frame for frame in (unique, ambiguous) if not frame.empty]
    resolved = pd.concat(frames, ignore_index=True, sort=False)
    if resolved.duplicated(SQ_JOIN_KEYS).any():
        raise AssertionError('Resolved ShotQuality assignments are not one row per key')
    return resolved[output_columns]


def attach_shotquality_rows(raw: pd.DataFrame, sq_data: pd.DataFrame) -> pd.DataFrame:
    """Attach ShotQuality rows without assigning one provider shot twice.

    `(game_id, event_num)` is a transitional crosswalk, not a canonical event
    key. If either side has multiple distinct candidates, the enrichment
    payload stays null and the ambiguity is recorded for later reconciliation.
    """
    missing = set(SQ_JOIN_KEYS) - set(raw.columns)
    if missing:
        raise ValueError(f'Raw PBP missing join columns: {sorted(missing)}')

    raw_work = raw.copy()
    raw_work['_sq_original_order'] = np.arange(len(raw_work), dtype='int64')
    raw_work['game_id'] = _normalized_game_id(raw_work['game_id'])
    raw_work['event_num'] = pd.to_numeric(
        raw_work['event_num'], errors='coerce'
    ).astype('Int64')

    game_ids = set(raw_work['game_id'].unique())
    sq_filtered = sq_data.copy()
    sq_filtered['game_id'] = _normalized_game_id(sq_filtered['game_id'])
    sq_filtered = sq_filtered[sq_filtered['game_id'].isin(game_ids)].copy()
    resolved = resolve_shotquality_candidates(sq_filtered)

    payload_columns = [
        col for col in resolved.columns
        if col not in {*SQ_JOIN_KEYS, *SQ_MATCH_DIAGNOSTIC_COLUMNS}
    ]
    enriched = raw_work.merge(
        resolved,
        on=SQ_JOIN_KEYS,
        how='left',
        validate='many_to_one',
        sort=False,
    )

    ambiguous_rows = enriched.duplicated(SQ_JOIN_KEYS, keep=False).to_numpy()
    if ambiguous_rows.any():
        for col in payload_columns:
            if pd.api.types.is_bool_dtype(enriched[col].dtype):
                enriched[col] = enriched[col].astype('boolean')
        enriched.loc[ambiguous_rows, payload_columns] = pd.NA
        enriched.loc[ambiguous_rows, 'sq_match_status'] = 'ambiguous_raw_event_key'

    enriched['sq_match_status'] = enriched['sq_match_status'].fillna('unmatched')
    enriched = enriched.sort_values('_sq_original_order', kind='stable')
    return enriched.drop(columns=['_sq_original_order']).reset_index(drop=True)


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
               'dynamic_ev', 'sq_shooter_id', 'sq_primary_defender_id',
               'sq_defender_id',  # deprecated legacy field removed on rerun
               'sq_play_descriptors', 'sq_descriptor_bundle',
               *SQ_MATCH_DIAGNOSTIC_COLUMNS] + [
                   descriptor_col_name(tag) for tag in SQ_DESCRIPTOR_TAGS
               ]
    existing_sq_cols = [c for c in sq_cols if c in raw.columns]
    if existing_sq_cols:
        raw = raw.drop(columns=existing_sq_cols)
        print(f'    Dropped existing SQ columns: {existing_sq_cols}')

    # Preserve the original ID representation for the written raw surface.
    raw_game_id_orig = raw['game_id'].copy()  # preserve original for saving
    enriched = attach_shotquality_rows(raw, sq_data)

    # Stats
    has_sq = enriched['is_transition'].notna().sum()
    trans = enriched['is_transition'].sum() if 'is_transition' in enriched else 0
    trans_dl = enriched['is_transition_dunk_or_layup'].sum() if 'is_transition_dunk_or_layup' in enriched else 0
    provider_ambiguous = enriched['sq_match_status'].eq(
        'ambiguous_provider_event_key'
    ).sum()
    raw_ambiguous = enriched['sq_match_status'].eq(
        'ambiguous_raw_event_key'
    ).sum()

    print(f'    Rows: {len(enriched):,} | With SQ: {has_sq:,} ({has_sq/len(enriched)*100:.1f}%)')
    print(f'    Transition: {trans:,} shots, {trans_dl:,} dunks/layups')
    print(
        f'    Ambiguous matches: provider={provider_ambiguous:,}, '
        f'raw_event_key={raw_ambiguous:,}'
    )
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
