"""Independent event-to-possession accounting for v50 FT replacement."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from build_vpm_dated_ft_context import ROOT,OUT,ATOMS,normal,sha,write_json


def audit_file(year,phase,probabilities):
    suffix='_PS' if phase=='PS' else ''
    rawpath=ROOT/f'raw_data/NBA{year%100:02d}{suffix}.parquet'
    raw=pd.read_parquet(rawpath,columns=['game_id','event_num','period','time_quarter',
        'minute_remaining_quarter','seconds_remaining_quarter',
        'player1_id','home_description','visitor_description']).drop_duplicates()
    raw.game_id=normal(raw.game_id)
    # Verify the frozen canonical possession contract, whose event builder
    # sorts by these original fields. A few cached MM:SS strings disagree by
    # one second; changing that ordering here would redefine the control.
    raw['clock']=pd.to_numeric(raw.minute_remaining_quarter)*60+pd.to_numeric(raw.seconds_remaining_quarter)
    if raw.clock.isna().any():raise ValueError('Unparseable event clock')
    raw=raw.sort_values(['game_id','period','clock','event_num'],ascending=[True,True,False,True],kind='stable')
    raw['rank']=raw.groupby('game_id').cumcount()
    atoms=pd.read_parquet(ATOMS/f'atoms_{year}_{phase}.parquet')
    atoms.game_id=normal(atoms.game_id)
    duplicate=raw.loc[raw.duplicated(['game_id','event_num'],keep=False)]
    if len(duplicate):
        unique=duplicate.groupby(['game_id','event_num'])[['period','clock']].nunique()
        text=duplicate.home_description.fillna('')+' '+duplicate.visitor_description.fillna('')
        if unique.gt(1).any().any() or text.str.contains('Free Throw',case=False).any():
            raise ValueError('Conflicting raw terminal event/FT identity')
    # Modern team-heave/block companion rows can share a non-FT event ID and
    # clock. They contain no intervening FT; the final source rank is exact
    # for cumulative FT accounting at that terminal key.
    ranks=raw.groupby(['game_id','event_num'],as_index=False)['rank'].max()
    eop=atoms[['game_id','event_num']].merge(ranks,
        on=['game_id','event_num'],validate='one_to_one',how='left')
    if eop['rank'].isna().any():raise ValueError('Canonical terminal event absent in raw source')
    text=raw.home_description.fillna('')+' '+raw.visitor_description.fillna('')
    ft=raw.loc[text.str.contains('Free Throw',case=False)].copy()
    if ft.duplicated(['game_id','event_num']).any():raise ValueError('Conflicting FT event identity')
    idx=pd.MultiIndex.from_arrays([ft.game_id,pd.to_numeric(ft.player1_id).astype('int64')])
    ft['p']=probabilities.reindex(idx).to_numpy()
    if ft.p.isna().any():raise ValueError('FT expectation absent')
    desc=ft.home_description.fillna('')+' '+ft.visitor_description.fillna('')
    # A substring test would classify made FTs by Yves Missi as misses.
    ft['delta']=ft.p-(~desc.str.contains(r'\bMISS\b',case=False,regex=True)).astype(float)
    keys=[];tail=[]
    eopgroups={gid:g.sort_values('rank') for gid,g in eop.groupby('game_id')}
    for gid,g in ft.groupby('game_id'):
        end=eopgroups[gid]
        ix=np.searchsorted(end['rank'].to_numpy(),g['rank'].to_numpy(),side='left')
        valid=ix<len(end)
        part=g.loc[valid,['game_id','delta']].copy()
        part['event_num']=end.event_num.to_numpy()[ix[valid]]
        keys.append(part)
        if (~valid).any():tail.append(g.loc[~valid,['game_id','event_num','delta']])
    delta=pd.concat(keys).groupby(['game_id','event_num']).delta.sum()
    newpath=OUT/f'ft_{year}_{phase}.parquet';new=pd.read_parquet(newpath)
    expected=delta.reindex(pd.MultiIndex.from_frame(new[['game_id','event_num']]),fill_value=0).to_numpy()
    gap=float(np.max(abs(expected-new.ft_delta.to_numpy())))
    if gap>1e-9:
        bad=np.flatnonzero(abs(expected-new.ft_delta.to_numpy())>1e-9)
        print(new.iloc[bad[:10]].assign(independent_delta=expected[bad[:10]]).to_string(index=False),flush=True)
        raise ValueError(f'FT event/possession accounting gap {year} {phase}: {gap}')
    return dict(season=year,phase=phase,rows=len(new),ft_events=len(ft),
        unassigned_tail_events=sum(len(x) for x in tail),
        unassigned_tail_expected_delta=float(sum(x.delta.sum() for x in tail)),
        duplicate_non_ft_companion_rows=len(duplicate),
        max_possession_delta_gap=gap,source_hashes={str(p):sha(p) for p in [rawpath,newpath,Path(__file__)]})


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--years',default=','.join(map(str,range(2016,2027))))
    args=ap.parse_args();p=pd.read_parquet(OUT/'ft_game_predictions.parquet').set_index(['game_id','player_id']).ft_expectation
    records=[]
    for year in map(int,args.years.split(',')):
        for phase in ['RS','PS']:
            record=audit_file(year,phase,p);records.append(record)
            print(f'Independent FT event accounting {year} {phase}: gap {record["max_possession_delta_gap"]:.2g}, tail events {record["unassigned_tail_events"]}',flush=True)
    dest=OUT/('independent_event_audit.json' if len(records)==22 else 'partial_event_audit.json')
    write_json(dest,dict(status='passed',files=records,max_gap=max(x['max_possession_delta_gap'] for x in records),
        tail_contract='Events after the last retained canonical possession are unassigned, not invented lineup credit',
        ft_predictions_sha256=sha(OUT/'ft_game_predictions.parquet')))


if __name__=='__main__':main()
