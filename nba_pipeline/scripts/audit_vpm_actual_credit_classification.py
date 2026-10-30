"""Additional independent raw shot-zone and FT-reference audit for actuals.

No producer/source/kernel imports. Bind the receipt to the finished output and
this verifier without editing any frozen producer or earlier audit receipt.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def audit(root):
    root=Path(root);receipt={};count=0
    # Frozen official layup/dunk/tip action taxonomy, independently enumerated.
    rim_codes={5,6,41,43,75,98,99,100,7,9,50,52,87,107,108,97}
    for folder in sorted((root/'ledgers').iterdir()):
        src=pd.read_parquet(folder/'source_events.parquet')
        text=src.home_description.fillna('')+' '+src.visitor_description.fillna('')
        three=text.str.contains('3PT',case=False,regex=False)
        rim=pd.to_numeric(src.event_action_type).isin(rim_codes)|text.str.contains(r'\blayup\b|\bdunk\b|\btip\b',case=False,regex=True)
        zone=np.where(three,'three',np.where(rim,'rim','mid'))
        shots=src.event_type.isin([1,2])
        if not np.array_equal(zone[shots],src.loc[shots,'zone']):raise ValueError('Raw shot zone mismatch')
        ft=src.event_type.eq(3)
        retained=text.str.contains('Technical|Flagrant|Clear Path|Transition',case=False,regex=True)
        one=text.str.contains(r'\b1 of 1\b',regex=True)
        two=text.str.contains(r'\b[12] of 2\b',regex=True)
        treble=text.str.contains(r'\b[123] of 3\b',regex=True)
        ref=np.where(ft&~retained&~one,np.where(treble,1/3,np.where(two,.5,0.)),0.)
        native=pd.read_parquet(folder/'native.parquet')
        admin=native.possession_exposure.eq(0)
        # Retained penalties identified through their foul context can have
        # ordinary FT text. Verify that context against the untouched raw file.
        admin_actions=src.row.isin(native.index[admin])
        ref[admin_actions]=0.
        pipeline=Path(__file__).resolve().parents[1]
        raw=pd.read_parquet(pipeline/f'raw_data/NBA{int(folder.name)%100:02d}.parquet',columns=[
            'game_id','event_num','event_type','period','minute_remaining_quarter','seconds_remaining_quarter',
            'player1_id','home_description','visitor_description'])
        raw.game_id=pd.to_numeric(raw.game_id).astype('int64')
        raw=raw.drop_duplicates().sort_values(['game_id','period','minute_remaining_quarter','seconds_remaining_quarter','event_num'],ascending=[True,True,False,False,True])
        groups={int(g):p.reset_index(drop=True) for g,p in raw.loc[raw.game_id.isin(src.game_id.unique())].groupby('game_id',sort=False)}
        proved_admin=set()
        for e in src.loc[admin_actions].itertuples():
            game=groups[int(e.game_id)];position=np.flatnonzero(game.event_num.eq(e.event_num))
            if len(position)!=1:raise ValueError('Admin FT lacks unique original event')
            before=game.iloc[:position[0]]
            foul=before.loc[before.event_type.isin([1,2,5,6])].tail(1)
            foul_text='' if foul.empty else str(foul.home_description.iloc[0] or '')+' '+str(foul.visitor_description.iloc[0] or '')
            recent=before.loc[before.event_type.isin([1,2,3,5])]
            recent=recent.loc[~recent.event_num.map(lambda n:(int(e.game_id),int(n)) in proved_admin)]
            recent_live=recent.tail(1)
            associated=not recent_live.empty and int(recent_live.event_type.iloc[0])==1 and int(recent_live.player1_id.iloc[0])==int(e.player1_id)
            line=str(e.home_description or '')+' '+str(e.visitor_description or '')
            is_one=bool(re.search(r'\b1 of 1\b',line))
            explicit=bool(re.search(r'Technical|Defensive\s*3\s*Sec|Flagrant|Clear Path|Transition',line,re.I))
            explicit|=bool(re.search(r'AWAY.FROM.PLAY|T\.FOUL|FLAGRANT|CLEAR.PATH|TRANSITION',foul_text,re.I))
            if not (explicit or (is_one and not associated)):
                raise ValueError(f'Unproved retained penalty FT {e.game_id}/{e.event_num}')
            proved_admin.add((int(e.game_id),int(e.event_num)))
        gap=float(np.max(abs(ref-src.ft_reference.to_numpy())))
        if gap>1e-14:raise ValueError('Raw FT opportunity reference mismatch')
        # Actual possession and admin rows are disjoint; no extra possession
        # can be created by a retained-penalty FT.
        if not native.loc[admin,'terminal_reason'].eq('administrative_free_throw').all():raise ValueError('Unexplained zero exposure')
        if not native.possession_exposure.isin([0.,1.]).all():raise ValueError('Invalid possession exposure')
        receipt[folder.name]=dict(actions=len(src),field_goal_attempts=int(shots.sum()),
            free_throw_attempts=int(ft.sum()),administrative_FT_rows=int(admin.sum()),
            maximum_FT_reference_gap=gap,source_sha256=sha(folder/'source_events.parquet'),native_sha256=sha(folder/'native.parquet'))
        count+=len(src)
    result=dict(status='passed_independent_raw_zones_FT_references_and_admin_exposure',
                seasons=receipt,actions=count,verifier_sha256=sha(__file__),
                registration_sha256=sha(root/'registration.json'))
    path=root/'source_classification_audit.json'
    if path.exists() and json.loads(path.read_text())!=result:raise ValueError('Existing classification audit differs')
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);audit(p.parse_args().root)
