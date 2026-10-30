"""Read current-season Databallr logs into a separate dated local snapshot.

No model fitting, publication, historical-source edits, or missing-value fills.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import time

import pandas as pd
from databallr_r2_read import R2Reader


def save_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def refresh(out, cut, season):
    out = Path(out).resolve()
    cut = pd.Timestamp(cut)
    if cut != cut.normalize() or cut.tz is not None or season != cut.year+1:
        raise ValueError('Explicit current-season autumn midnight cut required')
    out.mkdir(parents=True, exist_ok=True)
    if (out/'refresh.json').exists():
        raise FileExistsError('Completed source snapshot already exists')
    reader = R2Reader()
    objects = []
    for kind in ['player', 'team']:
        rows, _ = reader.objects('databallr-datasets', f'nba/{kind}-logs/v1/{season}/rs/')
        objects.extend((kind, row) for row in rows
                       if row['key'].endswith('.json') and Path(row['key']).name.split('.')[0].isdigit())
    if not objects:
        raise ValueError('Current-season cloud logs absent')
    for name in ['player_game', 'player_game_report']:
        if not (out/(name+'_catalog.json')).exists():
            save_json(out/(name+'_catalog.json'), reader.catalog_table(name))
    if not (out/'game_manifest.json').exists():
        reader.download('databallr-datasets', f'nba/pbp-analytics/v1/{season}/rs/manifest.json', out/'game_manifest.json')
    if not (out/'source_inventory.json').exists():
        save_json(out/'source_inventory.json', dict(cut=str(cut.date()), season=season,
            retrieved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), objects=objects,
            raw_report_absence_is_missing=True))
    inventory = json.loads((out/'source_inventory.json').read_bytes())
    if inventory['cut'] != str(cut.date()) or inventory['season'] != season:
        raise ValueError('Source snapshot date/season differs')
    objects = [(kind,row) for kind,row in inventory['objects']
               if Path(row['key']).name.split('.')[0].isdigit()]
    records = []
    def get(item):
        kind, row = item
        path = out/'objects'/kind/Path(row['key']).name
        if path.exists():
            raw = path.read_bytes()
            if len(raw) != row['size']:
                raise ValueError('Retained download byte count differs')
            proof = dict(bucket='databallr-datasets', key=row['key'], bytes=len(raw),
                         sha256=hashlib.sha256(raw).hexdigest())
        else:
            proof = reader.download('databallr-datasets', row['key'], path)
        return dict(kind=kind, local_file=str(path), **proof)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(get, item) for item in objects]
        for future in as_completed(futures):
            records.append(future.result())
            if len(records)%50 == 0 or len(records)==len(objects):
                print(json.dumps(dict(stage='cloud_logs_downloaded', complete=len(records), total=len(objects))), flush=True)
    summaries = {}
    for kind, id_column in [('player','person_id'), ('team','team_id')]:
        for part in ['core', 'report']:
            frames = []
            for record in records:
                is_report = record['key'].endswith('.report.json')
                if record['kind'] != kind or is_report != (part=='report'):
                    continue
                payload = json.loads(Path(record['local_file']).read_bytes())
                frame = pd.DataFrame(payload['columns'])
                if len(frame) != payload['rows']:
                    raise ValueError('Columnar log row count differs')
                entity_id = int(Path(record['key']).name.split('.')[0])
                if id_column in frame and not pd.to_numeric(frame[id_column]).eq(entity_id).all():
                    raise ValueError('Entity/file identity differs')
                frame[id_column] = entity_id
                frame['season'] = season
                frame['playoffs'] = False
                frame['game_date'] = pd.to_datetime(frame['date'].astype(str), format='%Y%m%d', errors='raise')
                frame = frame.loc[frame.game_date.lt(cut)].copy()
                frames.append(frame)
            if not frames:
                summaries[kind+'_'+part] = dict(rows=0, status='not_published_missing')
                continue
            frame = pd.concat(frames, ignore_index=True)
            if frame.duplicated(['game_id', id_column]).any():
                raise ValueError('Duplicate entity/game rows')
            destination = out/(kind+'_'+part+'.parquet')
            frame.to_parquet(destination, index=False)
            summaries[kind+'_'+part] = dict(rows=len(frame), entities=int(frame[id_column].nunique()),
                games=int(frame.game_id.nunique()), first_date=str(frame.game_date.min().date()),
                last_date=str(frame.game_date.max().date()), path=str(destination),
                sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
    games = json.loads((out/'game_manifest.json').read_bytes())['games']
    expected = {int(g['id']) for g in games if pd.to_datetime(str(g['date']),format='%Y%m%d') < cut}
    player_games = set(pd.read_parquet(out/'player_core.parquet',columns=['game_id']).game_id.astype(int))
    team_games = set(pd.read_parquet(out/'team_core.parquet',columns=['game_id']).game_id.astype(int))
    summary = dict(status='downloaded_current_cloud_inputs_not_a_ratings_board', cut=str(cut.date()),
        season=season, summaries=summaries, manifest_games=len(expected),
        player_missing_game_ids=sorted(expected-player_games), team_missing_game_ids=sorted(expected-team_games),
        model_fits=0, ratings_produced=False, object_reads=records,
        retrieved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save_json(out/'refresh.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='object_reads'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--cut',required=True)
    parser.add_argument('--season',type=int,required=True)
    args=parser.parse_args()
    refresh(args.out,args.cut,args.season)
