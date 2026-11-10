#!/usr/bin/env python3
"""Publish one finished dated VPM board to Supabase; dry-run unless --publish.

Each date is one insert-only row, so readers never see a partial board and a
refused or failed run leaves the current board live.
"""
from pathlib import Path
import argparse
import json
import os
import sys

TABLE = 'nba_vpm_preview_boards'
REPO = Path(__file__).resolve().parents[3]


def client():
    from dotenv import load_dotenv
    from supabase import create_client
    load_dotenv(REPO/'.env')
    url, key = os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_SERVICE_ROLE_KEY')
    if not url or not key:
        raise RuntimeError('SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required')
    return create_client(url, key)


def check(board, latest):
    """Return why this board should not go live, or None when it should."""
    meta, players = board['metadata'], board['players']
    if board['player_count'] != len(players) or not players:
        raise ValueError('Player count differs')
    if meta['estimate_date'] != board['snapshot_date'] or meta['schema'] != 'full_feature_vpm_preview_v1':
        raise ValueError('Board identity differs')
    if len({p['nba_id'] for p in players}) != len(players):
        raise ValueError('Duplicate players')
    if any(p.get(k) is None for p in players for k in ['ovpm', 'dvpm', 'vpm']):
        raise ValueError('Missing rating')
    if not latest:
        return None
    if board['snapshot_date'] <= latest['snapshot_date']:
        return 'not_newer_than_live_board'
    if meta['current_season_games'] <= latest['metadata']['current_season_games']:
        return 'no_new_games'
    if len(players) < .95*latest['player_count']:
        raise ValueError('Player count fell more than 5% from the live board')
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('preview', type=Path)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    board = json.loads(args.preview.read_bytes())
    db = client()
    rows = db.table(TABLE).select('snapshot_date,player_count,metadata').order('snapshot_date', desc=True).limit(1).execute().data
    latest = rows[0] if rows else None
    skip = check(board, latest)
    summary = dict(snapshot_date=board['snapshot_date'], players=board['player_count'],
                   games=board['metadata']['current_season_games'], live=latest and latest['snapshot_date'], skip=skip)
    if skip or not args.publish:
        print(json.dumps(dict(summary, status='not_published')))
        return 0
    db.table(TABLE).insert(dict(snapshot_date=board['snapshot_date'], player_count=board['player_count'],
                                metadata=board['metadata'], players=board['players'])).execute()
    live = db.table(TABLE).select('*').eq('snapshot_date', board['snapshot_date']).execute().data[0]
    if live['player_count'] != board['player_count'] or live['metadata'] != board['metadata'] or live['players'] != board['players']:
        raise RuntimeError('Read-back differs from what was published')
    receipt = dict(summary, status='published_and_read_back', published_at=live['published_at'],
                   source_manifest_sha256=board['metadata']['source_manifest_sha256'])
    (args.preview.parent/'preview_publication.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))
    return 0


if __name__ == '__main__':
    sys.exit(main())
