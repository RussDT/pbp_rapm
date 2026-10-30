"""Build the ShotQuality game_id -> NBA game_id bridge.

`merged_game_ids.csv` in the ShotQuality repo covers 1,273 of 1,387 season-2025
games and **zero** of the 1,355 season-2026 games, so re-enriching raw PBP for the
current season had no usable key. This rebuilds the bridge from scratch for every
season by matching on (Eastern-local game date, home final score, away final score),
which is effectively unique within a date.

ShotQuality timestamps are UTC and must be converted to US/Eastern before matching:
a 02:30 UTC tip is the previous calendar day in NBA terms.

Preseason is dropped -- the repo's raw PBP does not carry it.

Validated against the surviving `merged_game_ids.csv` rows; the script fails loudly
if it disagrees with that known-good mapping.
"""

import sys
from pathlib import Path

import pandas as pd

SQ = Path('/Users/russ/Documents/Codex/basketball/nba/research/shotquality')
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'sq_nba_game_bridge.csv'
SEASONS = (25, 26)


def nba_games() -> pd.DataFrame:
    """Final score and Eastern game date for every NBA game in raw PBP."""
    frames = []
    for y in SEASONS:
        for suf in ('', '_PS'):
            raw = ROOT / 'raw_data' / f'NBA{y}{suf}.parquet'
            proc = ROOT / 'processed' / f'RAPM{y}{suf}.parquet'
            if not raw.exists() or not proc.exists():
                print(f'  skip NBA{y}{suf} (missing raw or processed)')
                continue
            r = pd.read_parquet(raw, columns=['game_id', 'away_score', 'home_score'])
            p = pd.read_parquet(proc, columns=['game_id', 'game_date'])
            for d in (r, p):
                d['game_id'] = d['game_id'].astype(str).str.zfill(10)
            fin = r.groupby('game_id')[['away_score', 'home_score']].max().reset_index()
            dt = p.groupby('game_id')['game_date'].first().reset_index()
            m = fin.merge(dt, on='game_id')
            m['season'] = 2000 + y
            m['season_type'] = 'PS' if suf else 'RS'
            frames.append(m)
            print(f'  NBA{y}{suf}: {len(m)} games')
    if not frames:
        sys.exit('no NBA raw/processed parquets found')
    d = pd.concat(frames, ignore_index=True)
    d['game_date'] = pd.to_datetime(d['game_date']).dt.strftime('%Y-%m-%d')
    return d


def sq_games() -> pd.DataFrame:
    p = SQ / 'store' / 'games.parquet'
    if not p.exists():
        sys.exit(f'{p} missing; run sync_shotquality.py first')
    d = pd.read_parquet(p)
    d = d[d['game_status'] == 'Final'].copy()
    et = pd.to_datetime(d['game_datetime_utc'], errors='coerce', utc=True)
    d['game_date'] = et.dt.tz_convert('US/Eastern').dt.strftime('%Y-%m-%d')
    pre = d['game_descriptors.event_type_id'] == 'Preseason'
    print(f'  SQ: {len(d)} final games ({pre.sum()} preseason dropped)')
    return d[~pre]


def main():
    print('loading NBA side...')
    nba = nba_games()
    print('loading ShotQuality side...')
    sq = sq_games()

    key = ['game_date', 'home_score', 'away_score']
    for d in (sq, nba):
        for c in ('home_score', 'away_score'):
            d[c] = pd.to_numeric(d[c], errors='coerce')

    m = sq.merge(nba, on=key, how='left', suffixes=('', '_nba'))
    # A date+score collision would produce more rows than SQ games; drop those
    # rather than guess, and report them.
    counts = m.groupby('game_id').size()
    ambig = set(counts[counts > 1].index)
    m = m[~m['game_id'].isin(ambig)]
    matched = m[m['game_id_nba'].notna()].copy()

    print(f'\nmatched {len(matched)} of {len(sq)} SQ games '
          f'({len(ambig)} ambiguous on date+score, dropped)')
    print(matched.groupby(['season_nba', 'season_type']).size().to_string())

    unmatched = m[m['game_id_nba'].isna()]
    if len(unmatched):
        print(f'\n{len(unmatched)} unmatched, by event type:')
        print(unmatched['game_descriptors.event_type_id'].value_counts().to_string())

    # ShotQuality can carry two game ids for the same NBA game (observed on the
    # 2026 Finals: one tagged with an event type, one with a null type). Resolve
    # on the labelled row rather than keep-first, and report what was dropped.
    dup = matched[matched.duplicated('game_id_nba', keep=False)]
    if len(dup):
        matched['_lab'] = matched['game_descriptors.event_type_id'].notna().astype(int)
        matched = (matched.sort_values(['game_id_nba', '_lab', 'game_id'],
                                       ascending=[True, False, True])
                   .drop_duplicates('game_id_nba', keep='first')
                   .drop(columns='_lab'))
        print(f'\n{len(dup)} rows collided on {dup["game_id_nba"].nunique()} NBA game ids '
              f'(provider duplicates); kept the labelled row for each:')
        print(dup.sort_values('game_id_nba')[
            ['game_id', 'game_id_nba', 'game_date',
             'game_descriptors.event_type_id']].to_string(index=False))

    # --- validate against the surviving known-good mapping ---
    ref = SQ / 'merged_game_ids.csv'
    if ref.exists():
        r = pd.read_csv(ref).dropna()
        r['nba_game_id'] = r['nba_game_id'].astype('int64').astype(str).str.zfill(10)
        chk = matched.merge(r, on='game_id', how='inner', suffixes=('', '_ref'))
        agree = (chk['game_id_nba'] == chk['nba_game_id']).sum()
        print(f'\nvalidation vs merged_game_ids.csv: {agree}/{len(chk)} agree')
        if len(chk) and agree < len(chk):
            bad = chk[chk['game_id_nba'] != chk['nba_game_id']]
            print(bad[['game_id', 'game_id_nba', 'nba_game_id', 'game_date']].head(10).to_string())
            sys.exit('bridge disagrees with the known-good mapping; not writing')

    out = matched[['game_id', 'game_id_nba', 'season_nba', 'season_type', 'game_date',
                   'game_descriptors.event_type_id']].copy()
    out.columns = ['sq_game_id', 'nba_game_id', 'season', 'season_type', 'game_date', 'event_type']
    out.to_csv(OUT, index=False)
    print(f'\n-> {OUT} ({len(out)} rows)')


if __name__ == '__main__':
    main()
