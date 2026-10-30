"""Fetch current cloud plays and translate explicit lineups to RAPM raw columns."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from databallr_r2_read import R2Reader


def main(root):
    root=Path(root).resolve()
    refresh=json.loads((root/'refresh.json').read_bytes())
    if refresh['cut']!='2026-10-30' or refresh['season']!=2027:
        raise ValueError('Exact owner-requested current cut required')
    dest=root/'native_source'
    dest.mkdir(exist_ok=True)
    manifest=json.loads((root/'game_manifest.json').read_bytes())
    games=[g for g in manifest['games'] if int(g['date'])<20261030]
    reader=R2Reader()
    def fetch(g):
        path=dest/'games'/(str(g['id'])+'.json')
        if not path.exists():reader.download('databallr-datasets',g['key'],path)
        payload=json.loads(path.read_bytes())
        if payload['version']!=1 or payload['game_id']!=g['id']:
            raise ValueError('Cloud game identity differs')
        return g,path,payload
    data=[]
    with ThreadPoolExecutor(max_workers=4)as pool:
        for future in as_completed([pool.submit(fetch,g)for g in games]):
            data.append(future.result())
    core=pd.read_parquet(root/'player_core.parquet')
    teams=pd.read_parquet(root/'team_core.parquet')
    records=[];proof=[]
    for game,path,payload in sorted(data,key=lambda x:(x[0]['date'],x[0]['id'])):
        gid=int(game['id']);tc=teams.loc[teams.game_id.eq(gid)]
        if len(tc)!=2 or set(tc.is_home.astype(bool))!={False,True}:
            raise ValueError('Exactly one home and away team required')
        home=tc.loc[tc.is_home.astype(bool)].iloc[0]
        pc=core.loc[core.game_id.eq(gid)]
        hids=set(pc.loc[pc.team_id.eq(home.team_id),'person_id'].astype(int))
        aids=set(pc.loc[pc.team_id.ne(home.team_id),'person_id'].astype(int))
        if hids&aids:raise ValueError('Ambiguous player/team identity')
        for play in payload['plays']:
            raw=play['raw'];kind=raw['actionType'].lower();desc=raw.get('description')or''
            if int(raw['date'])!=int(game['date']) or raw['season']!=2027:
                raise ValueError('Current play date/season differs')
            if kind in ['2pt','3pt'] and raw.get('isFieldGoal'):
                event=1 if raw.get('shotResult')=='Made' else 2
            else:
                event={'freethrow':3,'rebound':4,'turnover':5,'foul':6,'violation':7,
                       'substitution':8,'timeout':9,'jumpball':10,'ejection':11,
                       'period':12}.get(kind,0)
            original_description=desc
            if event==1 and 'PTS' not in desc.upper():
                # Legacy arithmetic recognizes the PTS token on made shots.
                # Its value is sourced from the cloud play, never inferred from a rating.
                points=int(play['points'])
                if points not in [2,3]:raise ValueError('Made field goal requires exact 2/3 observed points')
                desc+=f' ({points} PTS)'
            clock=raw['clock'];m,s=clock[2:-1].split('M');remaining=60*int(m)+float(s)
            period=int(raw['period']);elapsed=720*min(period-1,4)+300*max(period-5,0)+(720 if period<=4 else 300)-remaining
            players=set(int(v)for v in play['players'])
            hp=sorted(players&hids);ap=sorted(players&aids)
            valid=play['lineupValid'] and len(hp)==5 and len(ap)==5 and len(players)==10
            if not valid:
                # Keep unresolved evidence; the canonical source gate must see it.
                hp=ap=[np.nan]*5
            acting_home=raw.get('teamId')==int(home.team_id)
            row=dict(game_id=str(gid).zfill(10),event_num=int(raw['event_order']),
                event_type=event,event_action_type=0,period=period,minute_game=elapsed/60,
                time_remaining=max(48,48+5*max(period-4,0))-elapsed/60,
                time_quarter=f'{int(remaining//60):02d}:{int(remaining%60):02d}',
                minute_remaining_quarter=int(remaining//60),seconds_remaining_quarter=remaining%60,
                home_description=desc if acting_home else None,
                visitor_description=None if acting_home else desc,neutral_description=None,
                home_score=raw['scoreHome'],away_score=raw['scoreAway'],
                score=f"{raw['scoreHome']} - {raw['scoreAway']}",
                score_margin=raw['scoreHome']-raw['scoreAway'],
                player1_id=raw.get('person_id'),player1_name=raw.get('playerName'),
                player1_team_id=raw.get('teamId'),player1_team_abbreviation=raw.get('team'),
                player2_id=np.nan,player3_id=np.nan,loc_x=raw.get('xLegacy'),loc_y=raw.get('yLegacy'),
                game_date=pd.to_datetime(str(game['date']),format='%Y%m%d'),
                cloud_event_order=raw['event_order'],cloud_action_number=raw['actionNumber'],cloud_event_id=raw['event_id'],
                cloud_lineup_valid=bool(valid),cloud_original_description=original_description)
            for i in range(5):row[f'home_player{i+1}']=hp[i];row[f'away_player{i+1}']=ap[i]
            records.append(row)
        proof.append(dict(game_id=gid,key=game['key'],path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    frame=pd.DataFrame(records)
    if frame.duplicated(['game_id','cloud_event_id']).any():
        raise ValueError('Duplicate source event identity')
    out=dest/'NBA27_current.parquet'
    if out.exists():raise FileExistsError('Current raw snapshot already exists')
    frame.to_parquet(out,index=False)
    result=dict(cut='2026-10-30',season=2027,raw_path=str(out),
        sha256=hashlib.sha256(out.read_bytes()).hexdigest(),games=frame.game_id.nunique(),rows=len(frame),
        unresolved_lineup_rows=int((~frame.cloud_lineup_valid).sum()),source_objects=proof,
        lineage='Cloud full plays, explicit per-play ten-player identity, same-game team identity from core logs',
        native_admitted=False,ratings_produced=False)
    result['adapter_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (dest/'source_current.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items()if k!='source_objects'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    main(parser.parse_args().root)
