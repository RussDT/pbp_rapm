"""Chronological DFGA and recovered six-foot states for monthly own-skill VPM.

Counts remain measurements, never repaired/model-filled observations. The
2014–2016 unprefixed rim files are cumulative; recover only consecutive PBP
appearances before any team change, using the same gate as recovered_v3.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

import build_tracking_next_game as T
from build_nba_estimated_skills_v2 import bounds_v2, LayoutV2, run_filter_v2
from build_rim_defense_matchup_skills import recover_cumulative_counts
from build_predictive_rim_saved import lagged_league, rolling
from build_deflections_next_game import compare
from build_monthly_own_skills import ROOT, SOURCE, OUT as CORE, sha, write_json, vintage_cut
from monthly_ou_filter import FastOU, load_kernel
from export_vpm_own_skills import snapshot_rows

OUT=ROOT/'results/estimated_skills/monthly_defense_v1'
MODERN=ROOT/'results/estimated_skills/rim_defense_matchups_recovered_v3/player_games.parquet'


def prepare(out):
    out.mkdir(parents=True,exist_ok=True)
    sources=[SOURCE/'player_game_lines.parquet', CORE/'game_context.parquet', T.TRACKING, MODERN,
             Path(__file__),Path(T.__file__),Path(__file__).with_name('build_rim_defense_matchup_skills.py'),
             Path(__file__).with_name('build_predictive_rim_saved.py')]
    sources += [ROOT.parent/f'player_sheets/game_report/year_files/{y}{s}_games.csv'
                for y in range(2014,2019) for s in ['', 'ps']]
    signature={str(p):sha(p) for p in sources}
    lock=out/'source_lock.json'
    if lock.exists() and json.loads(lock.read_text())!=signature:raise ValueError('Defense source lock changed')
    if (out/'rim_counts.parquet').exists() and (out/'dfga_counts.parquet').exists():return
    lines=pd.read_parquet(SOURCE/'player_game_lines.parquet')
    ctx=pd.read_parquet(CORE/'game_context.parquet')[['game_id','player_id','team_id']]
    appearances=lines.merge(ctx,on=['game_id','player_id'],how='left',validate='one_to_one')
    keys=['season','season_type','player_id','game_date']
    ambiguous=appearances.duplicated(keys,keep=False)
    appearances=appearances.loc[~ambiguous].copy()
    T.WARMUP=2014
    tracking,_=T.build_panel(pd.read_parquet(T.TRACKING),lines)
    tracking.loc[tracking.dfga_valid].to_parquet(out/'dfga_counts.parquet',index=False)
    rows=[];audit=[]
    for y in range(2014,2019):
        for suffix in ['', 'ps']:
            path=ROOT.parent/f'player_sheets/game_report/year_files/{y}{suffix}_games.csv'
            keep=['PLAYER_ID','PLAYER_NAME','date','MIN','AST','FGA','TEAM_ID']
            d=pd.read_csv(path,usecols=lambda c:c in keep or 'LT_06' in c,low_memory=False)
            prefix=next(p for p in ['lt6ft_totals_','less_6ft_def_','']
                        if all(p+c in d for c in ['FGA_LT_06','FGM_LT_06','NS_LT_06_PCT']))
            d['season']=y;d['season_type']='ps' if suffix else 'rs'
            d['game_date']=pd.to_datetime(d.date.astype(str),format='%Y%m%d')
            d=d.rename(columns={'PLAYER_ID':'player_id','PLAYER_NAME':'tracking_name'})
            for target,c in [('rim_dfga','FGA_LT_06'),('rim_dfgm','FGM_LT_06'),('provider_normal_pct','NS_LT_06_PCT')]:
                d[target]=pd.to_numeric(d[prefix+c],errors='coerce')
            d['field_family']=prefix or 'cumulative_pending_validation'
            if prefix=='':
                if suffix or y not in [2014,2015,2016]:raise ValueError('Unvalidated early rim field family')
                ordered=d.sort_values(['player_id','game_date'])
                decreases=int(ordered.groupby('player_id').rim_dfga.diff().lt(0).sum())
                if decreases:raise ValueError('Expected monotone archived cumulative count')
                d=recover_cumulative_counts(d,appearances)
            p=appearances.loc[appearances.season.eq(y)&appearances.season_type.eq('ps' if suffix else 'rs')]
            g=p.merge(d[keys+['MIN','AST','FGA','rim_dfga','rim_dfgm','field_family']+
                       [c for c in ['recovery_status'] if c in d]],on=keys,how='left',validate='one_to_one')
            valid=(g.MIN.sub(g.minutes).abs().lt(.5)&g.AST.eq(g.ast)&g.FGA.eq(g.fga)&
                   g.def_poss.gt(0)&g.rim_dfga.ge(0)&g.rim_dfgm.ge(0)&g.rim_dfgm.le(g.rim_dfga))
            valid &= g.groupby(['game_date','team_id']).rim_dfga.transform('sum').gt(0)
            g['rim_counts_valid']=valid
            audit.append(dict(season=y,phase=suffix or 'rs',rows=len(g),valid=int(valid.sum()),
                              field_family=prefix or 'cumulative_differenced',
                              reasons=g.recovery_status.value_counts().to_dict() if 'recovery_status' in g else {}))
            rows.append(g)
    modern=pd.read_parquet(MODERN)
    d=pd.concat([*rows,modern],ignore_index=True)
    d=d.loc[d.rim_counts_valid].copy().sort_values(['player_id','game_date','game_id']).reset_index(drop=True)
    if d.duplicated(['player_id','game_id']).any():raise ValueError('Rim count overlap')
    ratio=d.groupby('season').rim_dfga.sum()/d.groupby('season').def_poss.sum()
    if ratio.gt(.5).any():raise ValueError('Cumulative rim counts leaked into daily observations')
    keep=['player_id','game_id','game_date','season','season_type','def_poss','rim_dfga','rim_dfgm','field_family']
    d[keep].to_parquet(out/'rim_counts.parquet',index=False)
    write_json(out/'source_audit.json',dict(early_rim=audit,rim_valid_by_season=d.groupby('season').size().to_dict(),
        dfga_valid_by_season=tracking.loc[tracking.dfga_valid].groupby('season').size().to_dict(),
        ambiguous_appearance_rows=int(ambiguous.sum()),rim_volume_ratio=ratio.to_dict()))
    write_json(lock,signature)


class RimEngine:
    """Vectorize independent players at each appearance, preserving rolling()."""
    def __init__(self,d):
        self.ids=pd.factorize(d.player_id)[0];self.np=int(self.ids.max())+1
        step=d.groupby('player_id').cumcount().to_numpy()
        order=np.argsort(step,kind='stable');splits=np.flatnonzero(np.diff(step[order]))+1
        self.steps=np.split(order,splits)
        self.gap=d.groupby('player_id').game_date.diff().dt.days.fillna(0).to_numpy()
        self.n=d.rim_dfga.to_numpy();self.m=d.rim_dfgm.to_numpy();self.e=d.def_poss.to_numpy()
        self.q=d.league_pct.to_numpy()

    def run(self,vh,sh,k,mean):
        vn=np.zeros(self.np);ve=vn.copy();sn=vn.copy();sr=vn.copy();out=np.empty((len(self.ids),6))
        for idx in self.steps:
            p=self.ids[idx];vd=2**(-self.gap[idx]/vh);sd=2**(-self.gap[idx]/sh)
            vn[p]*=vd;ve[p]*=vd;sn[p]*=sd;sr[p]*=sd
            v=(vn[p]+1000*mean)/(ve[p]+1000);s=np.clip(sr[p]/(sn[p]+k),self.q[idx]-1,self.q[idx])
            out[idx,0]=100*v;out[idx,1]=s;out[idx,2]=200*v*s
            vn[p]+=self.n[idx];ve[p]+=self.e[idx];sn[p]+=self.n[idx];sr[p]+=self.q[idx]*self.n[idx]-self.m[idx]
            v=(vn[p]+1000*mean)/(ve[p]+1000);s=np.clip(sr[p]/(sn[p]+k),self.q[idx]-1,self.q[idx])
            out[idx,3]=100*v;out[idx,4]=s;out[idx,5]=200*v*s
        return out


def fit(task):
    out,year=task;out=Path(out);dest=out/'models'/f'{year}.json'
    if dest.exists():return dict(year=year,status='cached')
    cut=vintage_cut(year)
    d=pd.read_parquet(out/'dfga_counts.parquet');d=d.loc[d.game_date.lt(cut)].copy()
    sub=T.subpanel(d,T.STATS['dfga']);w=sub.exposure.to_numpy();obs=sub.obs_rate.to_numpy()
    mask=(sub.season.gt(sub.season.min())&sub.step_k.gt(0)).to_numpy()
    mean=float(np.average(obs,weights=w));var=float(np.average((obs-mean)**2,weights=w))
    engine=FastOU(sub,CORE/'kernel')
    def loss(p):return float(np.average((engine.run(obs,p,mean)[0][mask]-obs[mask])**2,weights=w[mask]))
    res=differential_evolution(loss,bounds_v2(var,float(w.mean())),seed=20260928,maxiter=60,popsize=15,tol=1e-7,polish=False)
    slow=run_filter_v2(LayoutV2(sub),obs,res.x,mean)
    gap=max(float(np.max(abs(a-b))) for a,b in zip(slow,engine.run(obs,res.x,mean)))
    if gap>1e-10:raise ValueError('DFGA OU verification failed')
    dfga=dict(params=res.x.tolist(),mean=mean,fit_end=str(sub.game_date.max().date()),objective=float(res.fun),fast_slow_gap=gap)
    d=pd.read_parquet(out/'rim_counts.parquet');d=d.loc[d.game_date.lt(cut)].copy()
    d['league_pct']=lagged_league(d);d['target100']=200*(d.league_pct*d.rim_dfga-d.rim_dfgm)/d.def_poss
    d['exposure']=d.def_poss
    warm=d.season.eq(d.season.min());mean=float(d.loc[warm,'rim_dfga'].sum()/d.loc[warm,'def_poss'].sum())
    engine=RimEngine(d);seasons=sorted(set(d.season))[1:]
    def mse(pred,target):return float(np.mean([np.average((pred[d.season.eq(s)]-target[d.season.eq(s)])**2,
                                                   weights=d.loc[d.season.eq(s),'def_poss']) for s in seasons]))
    volumes=[dict(half_life=h,mse=mse(engine.run(h,730,400,mean)[:,0],100*d.rim_dfga/d.def_poss)) for h in [90,365,730]]
    vh=min(volumes,key=lambda r:r['mse'])['half_life'];grid=[]
    for h in [365,730,1460]:
        for k in [100,400,1600]:
            name=f'h{h}_k{k}';d[name]=engine.run(vh,h,k,mean)[:,2]
            grid.append(dict(name=name,half_life=h,prior_attempts=k,mse=mse(d[name],d.target100)))
    best=min(grid,key=lambda r:r['mse']);fitrows=d.loc[d.season.isin(seasons)]
    eligible=[]
    for r in grid:
        c=compare(fitrows,best['name'],r['name'],draws=300)
        if c['delta_mse']<=c['delta_se']+1e-12:eligible.append(r)
    selected=max(eligible,key=lambda r:(r['prior_attempts'],r['half_life']))
    check=rolling(d,vh,selected['half_life'],selected['prior_attempts'],mean)
    gap=float(np.max(abs(check-engine.run(vh,selected['half_life'],selected['prior_attempts'],mean))))
    if gap>1e-10:raise ValueError('Rim rolling verification failed')
    rim=dict(selected=selected,volume_half=vh,mean=mean,fit_end=str(d.game_date.max().date()),
             volume_grid=volumes,suppression_grid=grid,fast_slow_gap=gap,training_seasons=list(map(int,seasons)))
    write_json(dest,dict(vintage=year,fit_cut=str(cut.date()),dfga=dfga,rim=rim,source_lock_sha256=sha(out/'source_lock.json')))
    return dict(year=year,dfga_objective=dfga['objective'],rim_selected=selected['name'])


def export(out,year):
    dest=out/'snapshots'/f'{year}.parquet'
    if dest.exists():return
    cuts=sorted(set(pd.date_range('2019-01-01','2026-09-01',freq='MS'))|{pd.Timestamp(f'{y}-10-01') for y in range(2015,2026)})
    cuts=[c for c in cuts if vintage_cut(year)<=c<vintage_cut(year+1)]
    model=json.loads((out/'models'/f'{year}.json').read_text());parts=[]
    d=pd.read_parquet(out/'dfga_counts.parquet');d=d.loc[d.game_date.lt(max(cuts))].copy()
    sub=T.subpanel(d,T.STATS['dfga']);m=model['dfga'];engine=FastOU(sub,CORE/'kernel')
    _,_,post,var=engine.run(sub.obs_rate.to_numpy(),m['params'],m['mean'])
    states=sub[['player_id','game_id','game_date']].copy();states['value']=100*np.maximum(post,0)
    states['state_variance']=10000*var;states['observed_exposure']=sub.groupby('player_id').exposure.cumsum()
    frame=snapshot_rows(states,cuts);frame['feature']='own_dfga100';frame['parameter_fit_end']=pd.Timestamp(m['fit_end']);parts.append(frame)
    d=pd.read_parquet(out/'rim_counts.parquet');d=d.loc[d.game_date.lt(max(cuts))].copy();d['league_pct']=lagged_league(d)
    m=model['rim'];s=m['selected'];a=RimEngine(d).run(m['volume_half'],s['half_life'],s['prior_attempts'],m['mean'])
    for name,col,exposure in [('own_rim_saved100',5,'rim_dfga'),('own_rim_contests100',3,'def_poss'),('own_rim_suppression',4,'rim_dfga')]:
        states=d[['player_id','game_id','game_date']].copy();states['value']=a[:,col];states['state_variance']=np.nan
        states['observed_exposure']=d.groupby('player_id')[exposure].cumsum()
        frame=snapshot_rows(states,cuts);frame['feature']=name;frame['parameter_fit_end']=pd.Timestamp(m['fit_end']);parts.append(frame)
    rows=pd.concat(parts,ignore_index=True);rows['parameter_fit_cut']=pd.Timestamp(model['fit_cut']);rows['vintage']=year
    rows['model_version']='monthly_annual_vintage_defensive';rows['model_sha256']=sha(out/'models'/f'{year}.json')
    if not rows.last_observation_date.lt(rows.estimate_date).all() or not rows.parameter_fit_end.lt(rows.estimate_date).all():
        raise ValueError('Defensive date fence failed')
    dest.parent.mkdir(parents=True,exist_ok=True);rows.to_parquet(dest,index=False)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,default=OUT)
    ap.add_argument('--stage',choices=['prepare','train','export'],required=True);ap.add_argument('--workers',type=int,default=4)
    args=ap.parse_args();prepare(args.out)
    if args.stage=='train':
        load_kernel(CORE/'kernel')
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for f in as_completed([pool.submit(fit,(str(args.out),y)) for y in range(2015,2026)]):print(json.dumps(f.result()),flush=True)
    elif args.stage=='export':
        for y in range(2015,2026):export(args.out,y);print('exported',y,flush=True)


if __name__=='__main__':main()
