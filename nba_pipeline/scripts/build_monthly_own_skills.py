"""Annual-vintage own-skill fits and strict pre-cut monthly neutral states.

All parameter selection is restricted to a vintage's training prefix. Context
and age/debut corrections use preceding seasons. Fixed architecture and search
bounds are research choices made today, not a claim the model existed in 2019.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

import build_nba_estimated_skills as B
from build_nba_estimated_skills_v2 import LayoutV2, bounds_v2, run_filter_v2
from build_nba_estimated_skills_v2_stageb import build_corrections
from build_estimated_skills_game_context import one_file
import estimated_skills_context as X
from monthly_ou_filter import FastOU, SOURCE as KERNEL_SOURCE
from export_vpm_own_skills import snapshot_rows

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results/estimated_skills/monthly_source_2014_2026'
OUT=ROOT/'results/estimated_skills/monthly_vintage_v1'
C_GRID=(2.,5.,10.,20.,40.)
AGE=ROOT/'results/estimated_skills/age_lookup.parquet'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def vintage_cut(year):return pd.Timestamp(f'{year}-12-01' if year==2020 else f'{year}-10-01')


def prepare(out):
    out.mkdir(parents=True,exist_ok=True)
    target=out/'game_context.parquet'
    if not target.exists():
        frames=[pd.read_parquet(ROOT/'results/estimated_skills/game_context.parquet')]
        for year in range(2014,2019):
            for phase in ['', '_PS']:
                frames.append(one_file(ROOT/'raw_data'/f'NBA{year%100:02d}{phase}.parquet'))
        context=pd.concat(frames,ignore_index=True)
        context.game_id=pd.to_numeric(context.game_id).astype(np.int64).astype(str).str.zfill(10)
        if context.duplicated(['game_id','player_id']).any():raise ValueError('Context overlap')
        context.to_parquet(target,index=False)
    inputs=[SOURCE/'player_game_lines.parquet',target,AGE,Path(__file__),KERNEL_SOURCE,
            Path(__file__).with_name('monthly_ou_filter.py')]
    inputs += [Path(__file__).with_name(n) for n in ['build_nba_estimated_skills.py',
        'build_nba_estimated_skills_v2.py','build_nba_estimated_skills_v2_stageb.py',
        'estimated_skills_context.py','estimated_skills_corrections.py']]
    signature={str(p):sha(p) for p in inputs}
    lock=out/'source_lock.json'
    if lock.exists() and json.loads(lock.read_text())!=signature:raise ValueError('Monthly skill source lock changed; use a new package')
    write_json(lock,signature)
    return signature


def load_panel(out,end):
    d=pd.read_parquet(SOURCE/'player_game_lines.parquet')
    d.game_date=pd.to_datetime(d.game_date)
    d=d.loc[d.game_date.lt(end)].copy()
    d['tot_poss']=d.off_poss+d.def_poss;d['usg_events']=d.fga+.44*d.fta+d.tov
    d=d.loc[d.tot_poss.gt(0)|d.fga.gt(0)|d.fta.gt(0)].copy()
    ctx=pd.read_parquet(out/'game_context.parquet')
    d=d.merge(ctx,on=['game_id','player_id'],how='left',validate='one_to_one')
    return d.sort_values(['player_id','game_date','game_id'],kind='stable').reset_index(drop=True),ctx


def context_columns(sub,spec,base,residual,centers,absent,c,reference_year):
    opponent_as_team=sub.drop(columns='opp_team_id').assign(opp_team_id=sub.team_id.values)
    cols=X.schedule_features(sub)+[
        X.running_group_residual(sub,residual,sub.exposure.to_numpy(),c,reference_end=reference_year),
        X.previous_season_group_residual(sub,residual,sub.exposure.to_numpy(),c,reference_end=reference_year),
        X.running_group_residual(opponent_as_team,residual,sub.exposure.to_numpy(),c,reference_end=reference_year)]
    if spec.kind=='rate':cols+=X.teammate_absence_features(sub,spec,sub.exposure.to_numpy(),centers,base,absent,reference_end=reference_year)
    return np.column_stack(cols)


def fit_one(task):
    out,year,name,maxiter,popsize=task;out=Path(out);cut=vintage_cut(year)
    dest=out/'models'/str(year)/f'{name}.json'
    spec=next(s for s in B.SKILLS if s.name==name)
    if dest.exists():return dict(year=year,skill=name,status='cached')
    started=time.time();df,ctx=load_panel(out,cut)
    sub=B.skill_subpanel(df,spec);obs=sub.obs_rate.to_numpy();w=sub.exposure.to_numpy()
    seasons=sub.season.to_numpy();hp=sub.step_k.to_numpy()>0
    fit=hp&(seasons>seasons.min())
    if fit.sum()<100:raise ValueError('Insufficient chronological skill training')
    B.WARMUP_SEASON=int(seasons.min())
    league,_=B.league_center(sub,spec)
    mean=float(np.average(obs,weights=w));engine=FastOU(sub,out/'kernel')
    def objective(pred):return float(np.average((pred[fit]-obs[fit])**2,weights=w[fit]))
    choices=[]
    for centered in ([False,True] if spec.kind=='pct' else [False]):
        centers=league if centered else np.zeros(len(sub));mr=0. if centered else mean
        response=obs-centers;variance=float(np.average((response-mr)**2,weights=w))
        bounds=bounds_v2(variance,float(w.mean()))
        def loss(p):return objective(engine.run(response,p,mr)[0]+centers)
        result=differential_evolution(loss,bounds,seed=20260928,maxiter=maxiter,popsize=popsize,tol=1e-7,polish=False)
        choices.append(dict(centered=centered,params=result.x.tolist(),loss=float(result.fun),converged=bool(result.success)))
    stage_a=min(choices,key=lambda c:c['loss']);centered=stage_a['centered']
    centers=league if centered else np.zeros(len(sub));mr=0. if centered else mean;response=obs-centers
    variance=float(np.average((response-mr)**2,weights=w))
    # Independent algebra check for this skill/vintage before accepting the fast fit.
    slow=run_filter_v2(LayoutV2(sub),response,stage_a['params'],mr)
    fast=engine.run(response,stage_a['params'],mr)
    gap=max(float(np.max(abs(a-b))) for a,b in zip(slow,fast))
    if gap>1e-10:raise ValueError(f'Fast OU reconstruction failed: {gap}')
    base=fast[0]+centers;residual=np.where(hp,obs-base,np.nan)
    absent=X.absent_teammates(df,ctx) if spec.kind=='rate' else None
    reference_year=int(seasons.max());target_seasons=sorted(set(seasons))[1:];best=None
    for c in C_GRID:
        cols=context_columns(sub,spec,base,residual,league,absent,c,reference_year)
        context,betas=X.expanding_linear_context(cols,residual,w,seasons,hp,target_seasons)
        value=objective(engine.run(response,stage_a['params'],mr,trans=context)[0]+centers)
        if best is None or value<best[0]:best=(value,c,context)
    _,c,context=best
    age=pd.read_parquet(AGE)
    init,shift,trans,meta=build_corrections(sub,spec,age,mean,sorted(set(seasons)))
    init=init-centers
    def final_loss(p):return objective(engine.run(response,p[:6],mr,init,shift,trans,context,p[6])[0]+centers)
    bounds=bounds_v2(variance,float(w.mean()))+[(0.,2.)]
    result=differential_evolution(final_loss,bounds,seed=20260928,maxiter=maxiter,popsize=popsize,tol=1e-7,polish=False)
    payload=dict(skill=name,vintage=year,fit_cut=str(cut.date()),fit_end=str(sub.game_date.max().date()),
        training_rows=len(sub),training_seasons=sorted(map(int,set(seasons))),objective_rows=int(fit.sum()),
        mean_rate=mean,centered=centered,stage_a=stage_a,centering_candidates=choices,
        context_c=c,params=result.x.tolist(),objective=float(result.fun),converged=bool(result.success),
        iterations=int(result.nit),maxiter=maxiter,popsize=popsize,seed=20260928,
        reference_year=reference_year,fast_slow_max_gap=gap,seconds=round(time.time()-started,2),
        source_lock_sha256=sha(out/'source_lock.json'),status='chronological_fit_research')
    write_json(dest,payload)
    return dict(year=year,skill=name,seconds=payload['seconds'],objective=payload['objective'])


def export_one(out,model,cuts):
    """The same fitted parameters replay only observations before needed cuts."""
    spec=next(s for s in B.SKILLS if s.name==model['skill'])
    df,ctx=load_panel(out,max(cuts));sub=B.skill_subpanel(df,spec)
    B.WARMUP_SEASON=int(sub.season.min())
    league,_=B.league_center(sub,spec);centers=league if model['centered'] else np.zeros(len(sub))
    mr=0. if model['centered'] else model['mean_rate'];obs=sub.obs_rate.to_numpy();response=obs-centers
    engine=FastOU(sub,out/'kernel');w=sub.exposure.to_numpy();hp=sub.step_k.to_numpy()>0
    base=engine.run(response,model['stage_a']['params'],mr)[0]+centers
    residual=np.where(hp,obs-base,np.nan)
    absent=X.absent_teammates(df,ctx) if spec.kind=='rate' else None
    cols=context_columns(sub,spec,base,residual,league,absent,model['context_c'],model['reference_year'])
    context,_=X.expanding_linear_context(cols,residual,w,sub.season.to_numpy(),hp,sorted(sub.season.unique())[1:])
    init,shift,trans,_=build_corrections(sub,spec,pd.read_parquet(AGE),model['mean_rate'],sorted(sub.season.unique()))
    p=model['params'];_,_,post,var=engine.run(response,p[:6],mr,init-centers,shift,trans,context,p[6])
    values=post+centers
    # Gaussian rate/state outputs must honor the physical count contract.
    bounded=np.clip(values,0,1) if spec.kind=='pct' else np.maximum(values,0)
    scale=1 if spec.kind=='pct' else 100
    states=sub[['player_id','game_id','game_date']].copy()
    states['value']=bounded*scale;states['state_variance']=var*scale**2
    states['observed_exposure']=sub.groupby('player_id').exposure.cumsum()
    rows=snapshot_rows(states,cuts)
    rows['feature']='own_'+spec.name;rows['parameter_fit_end']=pd.Timestamp(model['fit_end'])
    rows['parameter_fit_cut']=pd.Timestamp(model['fit_cut']);rows['vintage']=model['vintage']
    rows['model_version']='monthly_annual_vintage_ou_context_neutral'
    rows['model_sha256']=sha(out/'models'/str(model['vintage'])/(spec.name+'.json'))
    if not rows.last_observation_date.lt(rows.estimate_date).all() or not rows.parameter_fit_end.lt(rows.estimate_date).all():
        raise ValueError('Monthly skill date fence failed')
    return rows


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,default=OUT)
    ap.add_argument('--stage',choices=['prepare','train','export'],required=True)
    ap.add_argument('--years',default=','.join(map(str,range(2015,2026))))
    ap.add_argument('--skills',default=','.join(s.name for s in B.SKILLS))
    ap.add_argument('--workers',type=int,default=3);ap.add_argument('--maxiter',type=int,default=60);ap.add_argument('--popsize',type=int,default=15)
    args=ap.parse_args();prepare(args.out)
    years=list(map(int,args.years.split(',')));skills=args.skills.split(',')
    if any(y<2015 or y>2025 for y in years):raise ValueError('Only registered 2015–2025 model vintages')
    if args.stage=='train':
        tasks=[(str(args.out),year,skill,args.maxiter,args.popsize) for year in years for skill in skills]
        # Compile once before workers can race to build the shared library.
        from monthly_ou_filter import load_kernel
        load_kernel(args.out/'kernel')
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(fit_one,t) for t in tasks]):print(json.dumps(future.result()),flush=True)
    elif args.stage=='export':
        cuts=sorted(set(pd.date_range('2019-01-01','2026-09-01',freq='MS'))|{pd.Timestamp(f'{y}-10-01') for y in range(2015,2026)})
        for year in years:
            vc=vintage_cut(year);next_vc=vintage_cut(year+1)
            selected=[c for c in cuts if vc<=c<next_vc]
            if not selected:continue
            for skill in skills:
                dest=args.out/'snapshots'/str(year)/(skill+'.parquet')
                if dest.exists():continue
                model=json.loads((args.out/'models'/str(year)/(skill+'.json')).read_text())
                rows=export_one(args.out,model,selected);dest.parent.mkdir(parents=True,exist_ok=True)
                rows.to_parquet(dest,index=False);print(year,skill,len(rows),flush=True)


if __name__=='__main__':main()
