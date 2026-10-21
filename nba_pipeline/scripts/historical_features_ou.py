"""Quarterly replay of existing legally dated annual-OU skill vintages.

Never fills pre-2015 cuts using later fitted parameters. The fixed-EB history
is a distinct package; only its identically defined tracking/MPG fields are
reused here, while all own-skill estimates are emitted by original OU code.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

import numpy as np
import pandas as pd

import build_monthly_own_skills as C
import build_monthly_defensive_skills as D
from build_nba_estimated_skills import SKILLS
from export_vpm_own_skills import snapshot_rows
from monthly_ou_filter import FastOU
from historical_features import ROOT, OUT as BACKBONE, cuts, write_json, sha

OUT = ROOT/'results/estimated_skills/vpm_historical_features_ou_v1'


def legal_vintage(cut, vintages=range(2015,2026)):
    legal = [y for y in vintages if C.vintage_cut(y)<=pd.Timestamp(cut)]
    return max(legal) if legal else None


def core_task(task):
    out,year,name=task;out=Path(out)
    dest=out/'snapshots'/str(year)/(name+'.parquet')
    if dest.exists():
        return dict(year=year,skill=name,status='cached',rows=len(pd.read_parquet(dest)))
    selected=[c for c in cuts() if legal_vintage(c)==year]
    model=json.loads((C.OUT/'models'/str(year)/(name+'.json')).read_text())
    if any(pd.Timestamp(model['fit_end'])>=c for c in selected):
        raise ValueError('OU model fit not strictly before emission')
    rows=C.export_one(C.OUT,model,selected)
    dest.parent.mkdir(parents=True,exist_ok=True);rows.to_parquet(dest,index=False)
    return dict(year=year,skill=name,status='exported',rows=len(rows))


def defense_year(year):
    selected=[c for c in cuts() if legal_vintage(c)==year]
    model=json.loads((D.OUT/'models'/f'{year}.json').read_text());parts=[]
    d=pd.read_parquet(D.OUT/'dfga_counts.parquet');d=d.loc[d.game_date.lt(max(selected))].copy()
    sub=D.T.subpanel(d,D.T.STATS['dfga']);m=model['dfga'];engine=FastOU(sub,C.OUT/'kernel')
    _,_,post,var=engine.run(sub.obs_rate.to_numpy(),m['params'],m['mean'])
    states=sub[['player_id','game_id','game_date']].copy();states['value']=100*np.maximum(post,0)
    states['state_variance']=10000*var;states['observed_exposure']=sub.groupby('player_id').exposure.cumsum()
    rows=snapshot_rows(states,selected);rows['feature']='own_dfga100'
    rows['parameter_fit_end']=pd.Timestamp(m['fit_end']);parts.append(rows)
    d=pd.read_parquet(D.OUT/'rim_counts.parquet');d=d.loc[d.game_date.lt(max(selected))].copy()
    d['league_pct']=D.lagged_league(d);m=model['rim'];s=m['selected']
    values=D.RimEngine(d).run(m['volume_half'],s['half_life'],s['prior_attempts'],m['mean'])
    for name,col,exposure in [('own_rim_saved100',5,'rim_dfga'),
            ('own_rim_contests100',3,'def_poss'),('own_rim_suppression',4,'rim_dfga')]:
        states=d[['player_id','game_id','game_date']].copy();states['value']=values[:,col]
        states['state_variance']=np.nan;states['observed_exposure']=d.groupby('player_id')[exposure].cumsum()
        rows=snapshot_rows(states,selected);rows['feature']=name
        rows['parameter_fit_end']=pd.Timestamp(m['fit_end']);parts.append(rows)
    rows=pd.concat(parts,ignore_index=True);rows['parameter_fit_cut']=pd.Timestamp(model['fit_cut'])
    rows['vintage']=year;rows['model_version']='original_annual_vintage_defensive_ou_and_rim'
    rows['model_sha256']=sha(D.OUT/'models'/f'{year}.json')
    if not rows.last_observation_date.lt(rows.estimate_date).all() or not rows.parameter_fit_end.lt(rows.estimate_date).all():
        raise ValueError('Defensive OU fit/observation fence')
    return rows


def build(out=OUT,workers=2):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    tasks=[(str(out),y,s.name) for y in range(2015,2026) for s in SKILLS]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(core_task,t) for t in tasks]):
            print(json.dumps(f.result()),flush=True)
    for year in range(2015,2026):
        dest=out/'snapshots'/str(year)/'defensive.parquet'
        if not dest.exists():
            defense_year(year).to_parquet(dest,index=False)
        print('OU_defense',year,flush=True)
    paths=sorted((out/'snapshots').glob('*/*.parquet'))
    long=pd.concat([pd.read_parquet(p) for p in paths],ignore_index=True)
    if long.duplicated(['player_id','estimate_date','feature']).any():
        raise ValueError('Duplicate OU feature')
    if not long.last_observation_date.lt(long.estimate_date).all() or not long.parameter_fit_end.lt(long.estimate_date).all():
        raise ValueError('OU feature date fence')
    wide=long.pivot(index=['player_id','estimate_date'],columns='feature',values='value').reset_index()
    for zone,attempt in [('rim','rim_fga_rate'),('mid','mid_fga_rate'),('three','three_fga_rate'),('ft','fta_rate')]:
        wide[f'own_{zone}_makes100']=wide[f'own_{attempt}']*wide[f'own_{zone}_pct']
    backbone=pd.read_parquet(BACKBONE/'features.parquet')
    copy=[c for c in backbone if not c.startswith('own_') and c!='feature_recipe']
    backdrop=backbone[copy]
    backdrop=backdrop.loc[backdrop.estimate_date.isin(wide.estimate_date.unique())]
    wide=backdrop.merge(wide,on=['player_id','estimate_date'],how='left',validate='one_to_one')
    extras=pd.read_parquet(BACKBONE/'provenance.parquet')
    extras=extras.loc[(extras.feature.str.startswith('def_')|extras.feature.eq('ctx_mpg'))&
        extras.estimate_date.isin(wide.estimate_date.unique())].copy()
    extras['model_version']=extras.estimator
    long=pd.concat([long,extras],ignore_index=True)
    if long.duplicated(['player_id','estimate_date','feature']).any():
        raise ValueError('Duplicate full OU input provenance')
    if not long.last_observation_date.lt(long.estimate_date).all():
        raise ValueError('Extra defensive/MPG provenance date fence')
    for c in list(wide):
        if c.startswith('own_'):
            wide[c+'_missing']=wide[c].isna().astype(int)
    wide['feature_recipe']='original_annual_ou_legal_vintage_with_tracking_mpg'
    wide.to_parquet(out/'features.parquet',index=False);long.to_parquet(out/'provenance.parquet',index=False)
    models=[C.OUT/'models'/str(y)/(s.name+'.json') for y in range(2015,2026) for s in SKILLS]
    models += [D.OUT/'models'/f'{y}.json' for y in range(2015,2026)]
    source=[C.SOURCE/'player_game_lines.parquet',C.OUT/'game_context.parquet',C.AGE,
        D.OUT/'rim_counts.parquet',D.OUT/'dfga_counts.parquet',BACKBONE/'features.parquet',
        BACKBONE/'provenance.parquet',BACKBONE/'manifest.json',
        Path(__file__),Path(C.__file__),Path(D.__file__)]
    manifest=dict(status='passed_legal_annual_ou_replay_input_candidate',rows=len(wide),
        cuts=sorted(str(pd.Timestamp(c).date()) for c in wide.estimate_date.unique()),
        unavailable_cuts=[str(c.date()) for c in cuts() if legal_vintage(c) is None],
        observation_rule='strictly before cut, original max-state-age730days',
        parameter_rule='latest existing annual model with fit_cut<=cut and fit_end<cut',
        latest_model='2025 vintage carried through2026October; no newly fitted2026model',
        extra_fields='same canonical fixedTD550/K1000 defensive tracking summaries and exact365dayMPG as current selected method',
        models={str(p):sha(p) for p in models},sources={str(p):sha(p) for p in source},
        artifacts={n:sha(out/n) for n in ['features.parquet','provenance.parquet']},protected_outcomes=0,
        limitations=['Research architecture retrospective, publication vintages uncertified',
            'Unavailable before2015October; no later OU parameters applied to earlyhistory',
            'Current oldOU underlying source includes historical quarantines in raw own-skill observations as original selected estimator did; historical fixedEB excludes them',
            'No predictive replacement is selected by merely emitting theseinputs'])
    write_json(out/'manifest.json',manifest)
    return manifest


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,default=OUT)
    ap.add_argument('--workers',type=int,default=2);args=ap.parse_args();build(args.out,args.workers)
