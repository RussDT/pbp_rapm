"""Cache completed-season official Synergy inputs for the fresh BAM rebuild."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import pandas as pd
import requests
from fetch_matchup_game_archive import HEADERS

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'cache/matchup_bam_fresh_inputs_v1'
TYPES=['Isolation','Postup','PRBallHandler']

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def validate(frame,year,play_type):
    required=['SEASON_ID','PLAYER_ID','TEAM_ID','PLAY_TYPE','TYPE_GROUPING','PPP','POSS','GP']
    if frame.empty or not set(required).issubset(frame):raise ValueError('Missing source schema/cohort')
    if not pd.to_numeric(frame.SEASON_ID,errors='raise').eq(20000+year-1).all():raise ValueError('Wrong completed RS source year')
    if not frame.PLAY_TYPE.eq(play_type).all() or not frame.TYPE_GROUPING.str.lower().eq('offensive').all():raise ValueError('Wrong playtype/side')
    if frame.duplicated(['PLAYER_ID','TEAM_ID','PLAY_TYPE']).any():raise ValueError('Duplicate source player/team/playtype')
    if not frame.PLAYER_ID.gt(0).all() or not frame.TEAM_ID.gt(0).all() or not frame.GP.gt(0).all():raise ValueError('Invalid source identity/exposure')
    if frame[['PPP','POSS']].isna().any().any() or frame[['PPP','POSS']].lt(0).any().any():raise ValueError('Missing/negative source production')

def fetch():
    OUT.mkdir(parents=True,exist_ok=True);receipts=[];frames=[]
    with requests.Session() as session:
        session.headers.update(HEADERS)
        for year in range(2018,2026):
            for play_type in TYPES:
                params=dict(LeagueID='00',PerMode='PerGame',PlayerOrTeam='P',SeasonYear=f'{year-1}-{year%100:02d}',SeasonType='Regular Season',PlayType=play_type,TypeGrouping='offensive')
                raw=OUT/f'{year}_{play_type}.json';meta=raw.with_suffix('.meta.json')
                if raw.exists():
                    previous=json.loads(meta.read_text())
                    if previous['sha256']!=sha(raw) or previous['parameters']!=params:raise ValueError('Cached source changed')
                else:
                    response=session.get('https://stats.nba.com/stats/synergyplaytypes',params=params,timeout=(10,30))
                    if response.status_code in [403,429]:raise PermissionError(f'NBA HTTP {response.status_code}; stop for cooldown')
                    response.raise_for_status();payload=response.json()
                    if 'resultSets' not in payload:raise ValueError('Missing official resultSets')
                    raw.write_bytes(response.content)
                    meta.write_text(json.dumps(dict(fetched_at=datetime.now(timezone.utc).isoformat(),endpoint='SynergyPlayTypes',parameters=params,sha256=sha(raw),publication_vintage_certified=False),indent=2)+'\n')
                    time.sleep(1)
                sets=json.loads(raw.read_text())['resultSets']
                if len(sets)!=1:raise ValueError('Ambiguous source result set')
                frame=pd.DataFrame(sets[0]['rowSet'],columns=sets[0]['headers']);validate(frame,year,play_type)
                frame['year']=year;frames.append(frame)
                receipts.append(dict(year=year,play_type=play_type,rows=len(frame),raw=str(raw.resolve()),metadata=str(meta.resolve()),sha256=sha(raw),metadata_sha256=sha(meta)))
                print(json.dumps(receipts[-1]),flush=True)
    data=pd.concat(frames,ignore_index=True);path=OUT/'synergy_player_playtypes.parquet';data.to_parquet(path,index=False)
    sources={str(Path(__file__).resolve()):sha(__file__)}
    for row in receipts:sources[row['raw']]=row['sha256'];sources[row['metadata']]=row['metadata_sha256']
    manifest=dict(status='completed_official_RS_inputs',years=list(range(2018,2026)),play_types=TYPES,per_mode='PerGame',rows=len(data),receipts=receipts,source_hashes=sources,artifact_hashes={path.name:sha(path)},protected_outcomes=0,publication_vintage_certified=False,legacy_annual_counts_or_metadata_used=False)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest

if __name__=='__main__':fetch()
