"""Annual past-only OU rim defended-make counts, matching the DFGA skill filter."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,json,hashlib,time
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from build_nba_estimated_skills import SkillSpec,skill_subpanel
from build_nba_estimated_skills_v2 import bounds_v2,LayoutV2,run_filter_v2
from monthly_ou_filter import FastOU,load_kernel
from build_monthly_own_skills import vintage_cut
from export_vpm_own_skills import snapshot_rows

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/estimated_skills/vpm_rim_defended_makes_v1'
SPEC=SkillSpec('rim_dfgm','rim_dfgm','def_poss','rate')
DATES=sorted([f'{y}-10-01' for y in range(2015,2026) if y!=2020]+['2020-12-01','2026-09-01'])


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,data):Path(path).write_text(json.dumps(data,indent=2)+'\n')


def prepare(out):
    out.mkdir(parents=True,exist_ok=True);(out/'models').mkdir(exist_ok=True)
    source=ROOT/'results/estimated_skills/monthly_defense_v1/rim_counts.parquet'
    d=pd.read_parquet(source).sort_values(['player_id','game_date','game_id']).reset_index(drop=True)
    assert not d.duplicated(['player_id','game_id']).any() and d.season.max()==2026
    assert d.def_poss.gt(0).all() and d.rim_dfgm.ge(0).all() and d.rim_dfgm.le(d.rim_dfga).all()
    assert d.rim_dfgm.eq(d.rim_dfgm.round()).all() and d.rim_dfga.eq(d.rim_dfga.round()).all()
    assert d.game_date.lt('2026-09-01').all()
    # The recovered source's cumulative-era counts remain validated increments,
    # never raw snapshots. Pin its complete source receipt and audit.
    files=[source,source.parent/'source_lock.json',source.parent/'source_audit.json',Path(__file__),
           Path(__file__).with_name('monthly_ou_filter.py'),Path(__file__).with_name('monthly_ou_kernel.cpp'),
           Path(__file__).with_name('build_nba_estimated_skills.py'),Path(__file__).with_name('build_nba_estimated_skills_v2.py'),
           Path(__file__).with_name('export_vpm_own_skills.py')]
    sig={str(f):sha(f) for f in files};dest=out/'source_lock.json'
    if dest.exists():assert json.loads(dest.read_text())==sig
    else:save(dest,sig)
    d.to_parquet(out/'measurements.parquet',index=False)
    load_kernel(out/'kernel')


def fit(task):
    out,year=Path(task[0]),task[1];dest=out/'models'/f'{year}.json'
    if dest.exists():return json.loads(dest.read_text())
    started=time.time();cut=vintage_cut(year)
    d=pd.read_parquet(out/'measurements.parquet');d=d.loc[d.game_date.lt(cut)].copy()
    sub=skill_subpanel(d,SPEC);w=sub.exposure.to_numpy();obs=sub.obs_rate.to_numpy()
    mask=(sub.season.gt(sub.season.min())&sub.step_k.gt(0)).to_numpy()
    assert mask.sum()>100
    mean=float(np.average(obs,weights=w));var=float(np.average((obs-mean)**2,weights=w));engine=FastOU(sub,out/'kernel')
    def loss(p):return float(np.average((engine.run(obs,p,mean)[0][mask]-obs[mask])**2,weights=w[mask]))
    result=differential_evolution(loss,bounds_v2(var,float(w.mean())),seed=20260928,maxiter=60,popsize=15,tol=1e-7,polish=False)
    # Same DFGA optimizer/search/seed and units; independent slow replay validates algebra.
    slow=run_filter_v2(LayoutV2(sub),obs,result.x,mean);fast=engine.run(obs,result.x,mean)
    gap=max(float(np.max(abs(a-b))) for a,b in zip(slow,fast));assert gap<1e-10
    meta=dict(vintage=year,fit_cut=str(cut.date()),fit_end=str(sub.game_date.max().date()),params=result.x.tolist(),mean=mean,objective=float(result.fun),rows=len(sub),fast_slow_gap=gap,seconds=time.time()-started)
    save(dest,meta);return meta


def export(out):
    d=pd.read_parquet(out/'measurements.parquet');frames=[];parts=[];diagnostics=[]
    for year in range(2015,2026):
        cut=vintage_cut(year);cuts=[pd.Timestamp(x) for x in DATES if cut<=pd.Timestamp(x)<vintage_cut(year+1)]
        if not cuts:continue
        meta=json.loads((out/'models'/f'{year}.json').read_text());assert pd.Timestamp(meta['fit_end'])<cut
        sub=skill_subpanel(d.loc[d.game_date.lt(max(cuts))].copy(),SPEC);obs=sub.obs_rate.to_numpy();engine=FastOU(sub,out/'kernel')
        pred,_,post,var=engine.run(obs,meta['params'],meta['mean'])
        states=sub[['player_id','game_id','game_date']].copy();states['value']=100*np.maximum(post,0);states['state_variance']=10000*var
        states['observed_exposure']=sub.groupby('player_id').exposure.cumsum();p=snapshot_rows(states,cuts)
        p['feature']='own_rim_dfgm100';p['parameter_fit_end']=pd.Timestamp(meta['fit_end']);p['parameter_fit_cut']=cut;p['vintage']=year
        assert p.last_observation_date.lt(p.estimate_date).all();parts.append(p)
        for c in cuts:
            x=p.loc[p.estimate_date.eq(c),['player_id','value']].rename(columns={'value':'own_rim_dfgm100'}).copy();x['estimate_date']=c;frames.append(x)
        # Year-forward diagnostic at naturally pregame predictions using frozen annual fits.
        mask=sub.game_date.ge(cut)&sub.game_date.lt(vintage_cut(year+1))&sub.step_k.gt(0)
        q=sub.loc[mask,['player_id','game_id','game_date','season','exposure','obs_rate']].copy()
        q['prediction']=100*np.maximum(pred[mask],0);q['population']=100*meta['mean'];q['target']=100*q.obs_rate
        if len(q):diagnostics.append(q)
    f=pd.concat(frames,ignore_index=True);p=pd.concat(parts,ignore_index=True)
    assert not f.duplicated(['player_id','estimate_date']).any()
    f.to_parquet(out/'features.parquet',index=False);p.to_parquet(out/'provenance.parquet',index=False)
    q=pd.concat(diagnostics,ignore_index=True);q.to_parquet(out/'pregame_diagnostic.parquet',index=False)
    names=['features.parquet','provenance.parquet','pregame_diagnostic.parquet','measurements.parquet']
    meta=dict(status='passed',feature='own_rim_dfgm100',method='same annual past-only two-state OU count-rate filter as own_dfga100',source='validated rim_counts, including recovered cumulative-era increments',max_state_age_days=730,dates=DATES,source_lock_sha256=sha(out/'source_lock.json'),models={str(y):sha(out/'models'/f'{y}.json') for y in range(2015,2026)},artifact_hashes={n:sha(out/n) for n in names},measured_player_games=len(d),protected_outcomes=0,limits=['Rim DFGM is predicted independently of rim DFGA; coherent bounds are audited, not silently clipped to another separately fitted state.','Historical source/publication availability remains uncertified.'])
    save(out/'manifest.json',meta);print(json.dumps(dict(status='passed',rows=len(f),measurements=len(d))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=OUT);p.add_argument('--stage',choices=['prepare','fit','export','all'],default='all');p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    if a.stage in ['prepare','all']:prepare(a.out)
    if a.stage in ['fit','all']:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            for f in as_completed([pool.submit(fit,(a.out,y)) for y in range(2015,2026)]):print(json.dumps(f.result()),flush=True)
    if a.stage in ['export','all']:export(a.out)
