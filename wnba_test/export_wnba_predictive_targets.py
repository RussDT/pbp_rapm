#!/usr/bin/env python3
"""Isolated actual-FT WNBA possession/skill-count adapter for sibling VPM.

Exports counts and outcome-safe shot context, never fits player ratings. Whole
ambiguous or score-incomplete games are quarantined rather than silently used.
"""
from pathlib import Path
import sys, json, hashlib
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wnba_test.clean_decomp_wnba import build_clean_chance_base
from wnba_test.build_wnba_pbp_shot_quality import build_shot_rows


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(out_dir, years=range(2022, 2027)):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    receipt=out_dir/'manifest.json'
    if receipt.exists():
        for source,expected in json.loads(receipt.read_text())['sources'].items():
            if fingerprint(source)!=expected:raise ValueError('Frozen source changed; choose a new output directory')
    logs_path = ROOT / 'wnba_test/Results/gameday/player_game_logs.csv'
    logs = pd.read_csv(logs_path)
    logs['game_id'] = logs.game_id.astype('int64')
    game_dates = logs.groupby('game_id').game_date.first()
    scores = logs.groupby(['game_id','home']).team_score.first()
    sources = {str(logs_path): fingerprint(logs_path)}
    audits=[]; frames=[]; shot_frames=[]
    for year in years:
        if year >= 2027: raise ValueError('2027 protected')
        for suffix in ('','_PS'):
            path=ROOT/f'wnba_test/GabrielNormalizedClean/WNBA{year%100:02d}{suffix}.csv'
            if not path.exists(): continue
            sources[str(path)] = fingerprint(path)
            cache=out_dir/f'base_{year}{suffix}.parquet'
            lock=cache.with_suffix('.json')
            key={'raw':sources[str(path)],'adapter':fingerprint(__file__), 'clean':fingerprint(ROOT/'wnba_test/clean_decomp_wnba.py')}
            raw=pd.read_csv(path,low_memory=False)
            if cache.exists() and lock.exists() and json.loads(lock.read_text())==key:
                base=pd.read_parquet(cache)
            else:
                base=build_clean_chance_base(path,year,str(year),'PREDICTIVE_ACTUAL_FT',bool(suffix),actual_fts=True)
                if base is None: continue
                cols=['game_id','event_num','Actual_Diff','first_fga_event','poss_offense']+[f'{s}{i}' for s in 'OD' for i in range(1,6)]
                base=base[cols].copy()
                base.to_parquet(cache,index=False);lock.write_text(json.dumps(key))
            base['game_id']=pd.to_numeric(base.game_id).astype('int64')
            base['game_date']=base.game_id.map(game_dates)
            base['home']=base.poss_offense.eq('Home').astype(int)
            base['season']=year; base['season_type']='ps' if suffix else 'rs'
            lineup=[f'{s}{i}' for s in 'OD' for i in range(1,6)]
            base[lineup]=base[lineup].apply(pd.to_numeric,errors='coerce')
            bad=base[lineup].isna().any(axis=1)|base[lineup].le(0).any(axis=1)|base[lineup].nunique(axis=1).ne(10)
            invalid=set(base.loc[bad,'game_id'])
            totals=base.groupby(['game_id','home']).Actual_Diff.sum()
            expected=scores.reindex(totals.index)
            gaps=totals-expected
            for gid in base.game_id.unique():
                g=gaps.loc[gid]; valid=gid not in invalid and g.notna().all() and len(g)==2 and g.abs().max()<1e-7
                audits.append(dict(game_id=int(gid),season=year,valid=bool(valid),lineup_valid=gid not in invalid,max_score_gap=float(g.abs().max()) if g.notna().all() else None,date_known=bool(pd.notna(game_dates.get(gid)))))
            good={r['game_id'] for r in audits if r['valid'] and r['date_known']}
            base=base[base.game_id.isin(good)].copy()
            if base.empty: continue
            base[lineup]=base[lineup].astype('int64')
            if not np.allclose(base.Actual_Diff,np.round(base.Actual_Diff),atol=1e-8): raise ValueError('non-integer actual scoring')
            shots=build_shot_rows(raw,year,'ps' if suffix else 'rs')
            shots['game_id']=shots.game_id.astype('int64')
            shots=shots[shots.game_id.isin(good)].copy()
            shots['game_date']=shots.game_id.map(game_dates)
            frames.append(base); shot_frames.append(shots)
            print(year,suffix,len(base),'rows',base.game_id.nunique(),'complete games',flush=True)
    rows=pd.concat(frames,ignore_index=True).sort_values(['game_date','game_id','event_num']).reset_index(drop=True)
    shots=pd.concat(shot_frames,ignore_index=True)
    if rows.duplicated(['game_id','event_num']).any(): raise ValueError('duplicate terminals')
    if shots.duplicated(['game_id','event_num']).any(): raise ValueError('duplicate shots')
    rows.to_parquet(out_dir/'possessions.parquet',index=False)
    shots.to_parquet(out_dir/'shots.parquet',index=False)
    good_games=set(rows.game_id)
    played=logs[logs.game_id.isin(good_games)&logs.minutes.gt(0)].copy()
    pieces=[]
    for side in 'OD':
        melt=rows.melt(id_vars=['game_id'],value_vars=[f'{side}{i}' for i in range(1,6)],value_name='person_id')
        pieces.append(melt.groupby(['game_id','person_id']).size().rename(side))
    exposure=pd.concat(pieces,axis=1).fillna(0)
    played=played.merge(exposure.reset_index(),on=['game_id','person_id'],validate='one_to_one')
    played['possessions']=(played.O+played.D)/2
    # Raw shot parser's stable NBA numeric identity; no fuzzy name crosswalk.
    sid='shooter_id'
    zone=shots.copy(); zone['person_id']=pd.to_numeric(zone[sid]).astype('int64')
    zone['zone']=np.where(zone.shot_zone.eq('rim'),'rim',np.where(zone.shot_value.eq(3),'three','mid'))
    zone['make']=zone.actual_points.gt(0).astype(int)
    cnt=zone.groupby(['game_id','person_id','zone']).agg(fga=('make','size'),fgm=('make','sum')).unstack('zone').fillna(0)
    cnt.columns=[f'{z}_{stat}' for stat,z in cnt.columns]
    played=played.merge(cnt.reset_index(),on=['game_id','person_id'],how='left',validate='one_to_one')
    zcols=[f'{z}_{s}' for z in ('rim','mid','three') for s in ('fga','fgm')]
    played[zcols]=played[zcols].fillna(0)
    if not np.array_equal(played[['rim_fga','mid_fga','three_fga']].sum(axis=1),played.fga): raise ValueError('box/shot FGA disagreement')
    if not np.array_equal(played[['rim_fgm','mid_fgm','three_fgm']].sum(axis=1),played.fgm): raise ValueError('box/shot FGM disagreement')
    played.to_parquet(out_dir/'game_counts.parquet',index=False)
    pd.DataFrame(audits).to_csv(out_dir/'game_audit.csv',index=False)
    report=dict(status='research',point_target='actual FG and FT',rows=len(rows),games=rows.game_id.nunique(),latest_date=str(rows.game_date.max()),excluded_games=sum(not r['valid'] or not r['date_known'] for r in audits),sources=sources)
    (out_dir/'manifest.json').write_text(json.dumps(report,indent=2))
    return report

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=ROOT/'wnba_test/Results/predictive_parent_targets_v1')
    a=p.parse_args();print(json.dumps(export(a.output_dir),indent=2))
