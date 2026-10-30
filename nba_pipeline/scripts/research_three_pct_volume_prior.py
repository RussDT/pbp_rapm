"""Single-feature, prior-season shooting prior challenger; historical research."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from build_nba_estimated_skills import skill_subpanel, SKILLS,FilterLayout,run_filter
from build_deflections_next_game import compare
ROOT=Path(__file__).resolve().parents[1]/'results/estimated_skills'
OUT=ROOT/'three_pct_volume_prior'

def volume_features(d,mean):
    g=d.groupby('player_id',sort=False)
    a=g.three_fga.cumsum()-d.three_fga
    e=g.off_poss.cumsum()-d.off_poss
    return 100*(a+200*mean)/(e+200)

def main():
    OUT.mkdir(exist_ok=True,parents=True)
    full=pd.read_parquet(ROOT/'player_game_lines.parquet').sort_values(['player_id','game_date','game_id']).reset_index(drop=True)
    full=full[full.season.le(2025)&full.off_poss.gt(0)].copy()
    warm=full[full.season.eq(2019)]
    volume_mean=warm.three_fga.sum()/warm.off_poss.sum()
    full['prior_3pa100']=volume_features(full,volume_mean)
    spec=next(x for x in SKILLS if x.name=='three_pct')
    sub=skill_subpanel(full,spec)
    cfg=json.loads((ROOT/'hyperparams.json').read_text())['three_pct']
    assert cfg['selected_variant']=='uncentered'
    params=list(cfg['variants']['uncentered']['params'].values())
    mean=cfg['league_mean_rate_pre_holdout']
    layout=FilterLayout(sub);obs=sub.obs_rate.to_numpy()
    base,_,_,_=run_filter(layout,obs,params,mean)
    prior=np.full(len(sub),mean);curves=[]
    for year in range(2020,2026):
        train=sub.season.lt(year);test=sub.season.eq(year)
        model=make_pipeline(PolynomialFeatures(2,include_bias=False),StandardScaler(),Ridge(alpha=10))
        model.fit(np.log1p(sub.loc[train,['prior_3pa100']]),obs[train],ridge__sample_weight=sub.loc[train,'exposure'])
        prior[test]=np.clip(model.predict(np.log1p(sub.loc[test,['prior_3pa100']])),.01,.99)
        for x in [0,.25,.5,1,2,5,10,15]:curves.append({'season':year,'prior_3pa100':x,'prior_pct':100*np.clip(model.predict([[np.log1p(x)]])[0],.01,.99)})
    pred={'control':base}
    for strength in [.25,.5,1.]:
        anchor=mean+strength*(prior-mean)
        residual,_,_,_=run_filter(layout,obs-anchor,params,0.)
        pred[str(strength)]=np.clip(anchor+residual,0,1)
    d=sub[['player_id','game_id','season','exposure','prior_3pa100']].copy();d['target100']=100*obs
    for k,v in pred.items():d[k]=100*v
    d=d[sub.step_k.gt(0)]
    dev=d[d.season.between(2020,2022)]
    results={k:compare(dev,k,'control',draws=1000) for k in pred if k!='control'}
    best=min(results,key=lambda k:results[k]['mse_per100'])
    selected=best if results[best]['delta_mse']>results[best]['delta_se'] else 'control'
    # One-SE simpler candidate relative to empirical best.
    if selected!='control':
        for k in ['0.25','0.5','1.0']:
            c=compare(dev,k,best,draws=1000)
            if c['mse_per100']<=c['baseline_mse_per100']+c['delta_se']:
                selected=k;break
    report={'selected_strength':selected,'development':results,'notes':'Existing v1 parameters reused; historical development only, not as-of-trained replay. 2026 unscored; 2027 untouched.'}
    (OUT/'development_gate.json').write_text(json.dumps(report,indent=2))
    stress=d[d.season.between(2023,2025)]
    report['stress']={}
    for label,x in [('all',stress),('low_volume_under_1',stress[stress.prior_3pa100.lt(1)]),('under_3',stress[stress.prior_3pa100.lt(3)])]:
        report['stress'][label]=compare(x,selected,'control',draws=2000)
    report['stress_by_season']={str(y):compare(x,selected,'control',draws=1000) for y,x in stress.groupby('season')}
    (OUT/'report.json').write_text(json.dumps(report,indent=2));pd.DataFrame(curves).to_csv(OUT/'prior_curves.csv',index=False)
    d.to_parquet(OUT/'comparison_rows.parquet',index=False)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
