"""Prior blocked-FGA share incremental test for rim shooting."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from build_nba_estimated_skills import SkillSpec,skill_subpanel,FilterLayout,run_filter
from research_shooting_fusion import comparison
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/estimated_skills/rim_blocked_prior'

def prior_ratio(d,numer,denom,pseudo,mean):
    g=d.groupby('player_id',sort=False)
    return (g[numer].cumsum()-d[numer]+pseudo*mean)/(g[denom].cumsum()-d[denom]+pseudo)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    d=pd.read_parquet(ROOT/'results/estimated_skills/player_game_lines.parquet')
    d=d[d.season.between(2019,2025)&d.off_poss.gt(0)].copy()
    keys=['player_id','game_date','season','season_type'];d=d[~d.duplicated(keys,keep=False)]
    chunks=[]
    for y in range(2019,2026):
        for suffix,phase in [('', 'rs'),('ps','ps')]:
            p=ROOT.parent/f'player_sheets/game_report/year_files/{y}{suffix}_games.csv'
            t=pd.read_csv(p,usecols=['PLAYER_ID','date','BLKA','FGA','MIN','AST']);t['season']=y;t['season_type']=phase
            t['game_date']=pd.to_datetime(t.date.astype(str));t=t.rename(columns={'PLAYER_ID':'player_id'})
            chunks.append(t)
    t=pd.concat(chunks);t=t[~t.duplicated(keys,keep=False)]
    d=d.merge(t[keys+['BLKA','FGA','MIN','AST']],on=keys,how='left',validate='one_to_one').sort_values(['player_id','game_date','game_id']).reset_index(drop=True)
    valid=d.fga.eq(d.FGA)&d.ast.eq(d.AST)&d.minutes.sub(d.MIN).abs().le(.5)&d.BLKA.between(0,d.FGA)
    d['measured_fga']=d.fga.where(valid,0);d['blocked']=d.BLKA.where(valid,0)
    # Missing rows do not imply zero blocked attempts; advance neither ratio sum.
    w=d[d.season.eq(2019)]
    d['blocked_share']=prior_ratio(d,'blocked','measured_fga',100,w.blocked.sum()/w.measured_fga.sum())
    d['rim_volume']=100*prior_ratio(d,'rim_fga','off_poss',200,w.rim_fga.sum()/w.off_poss.sum())
    d['ft_prior']=prior_ratio(d,'ftm','fta',20,w.ftm.sum()/w.fta.sum())
    g=d.groupby('player_id');d['prior_rim_attempts']=g.rim_fga.cumsum()-d.rim_fga
    d['prior_measured_fga']=g.measured_fga.cumsum()-d.measured_fga
    sub=skill_subpanel(d,SkillSpec('rim_pct','rim_fgm','rim_fga','pct'))
    mean=w.rim_fgm.sum()/w.rim_fga.sum();warm=sub[sub.season.eq(2019)].reset_index(drop=True)
    choices=[]
    for q in [1e-7,1e-6,1e-5]:
        for r in [.25,1.]:
            params=[0,q,0,r,.01];p=run_filter(FilterLayout(warm),warm.obs_rate.to_numpy(),params,mean)[0];p=np.clip(p,.001,.999)
            loss=-(warm.rim_fgm*np.log(p)+(warm.rim_fga-warm.rim_fgm)*np.log1p(-p)).sum()/warm.rim_fga.sum()
            choices.append((loss,params))
    params=min(choices,key=lambda x:x[0])[1]
    # All-game filter: skip shooting update for zero-attempt games, retain prediction.
    k=np.zeros(len(d));var=np.zeros(len(d))
    for _,ids in d.groupby('player_id',sort=False).indices.items():
        theta=mean;v=.01;last=None
        for i in ids:
            row=d.iloc[i];v+=params[1]*(0 if last is None else (row.game_date-last).days);k[i]=theta;var[i]=v
            if row.rim_fga>0:
                gain=v/(v+params[3]/row.rim_fga);theta+=gain*(row.rim_fgm/row.rim_fga-theta);v*=1-gain
            last=row.game_date
    d['kalman']=k
    x=pd.DataFrame({'kalman':k,'variance':var,'logvolume':np.log1p(d.rim_volume),'ft':d.ft_prior,'logattempts':np.log1p(d.prior_rim_attempts),'blocked_share':d.blocked_share})
    obs=np.divide(d.rim_fgm,d.rim_fga,out=np.zeros(len(d)),where=d.rim_fga.gt(0))
    coefs=[]
    for name in ['control','with_blocked']:d[name]=k
    for year in range(2021,2026):
        tr=d.season.between(2020,year-1)&d.rim_fga.gt(0);te=d.season.eq(year)
        for name,cols in [('control',list(x.columns[:-1])),('with_blocked',list(x.columns))]:
            model=make_pipeline(StandardScaler(),Ridge(alpha=1000))
            model.fit(x.loc[tr,cols],obs[tr]-k[tr],ridge__sample_weight=d.loc[tr,'rim_fga'])
            d.loc[te,name]=np.clip(k[te]+model.predict(x.loc[te,cols]),.001,.999)
            if name=='with_blocked':coefs.append({'season':year,'blocked_coefficient_fraction':model[-1].coef_[-1]/model[0].scale_[-1]})
    e=d.copy();e['three_fgm']=e.rim_fgm;e['three_fga']=e.rim_fga;e['volume']=e.rim_volume
    dev=e[e.season.between(2021,2022)]
    report={'feature':'prior BLKA / measured FGA, all-shot blocked share','base_params':params,'coverage':float(valid.mean()),'coefficients':coefs,'development':{m:comparison(dev,'with_blocked','control',m) for m in ['logloss','pct_mse','makes_mse']}}
    c=report['development']['logloss'];j=report['development']['makes_mse'];report['passes_development']=c['delta']>c['se'] and j['delta']>=-j['se']
    (OUT/'development_gate.json').write_text(json.dumps(report,indent=2))
    stress=e[e.season.between(2023,2025)]
    report['stress']={label:{m:comparison(z,'with_blocked','control',m) for m in ['logloss','pct_mse','makes_mse']} for label,z in [('all',stress),('established',stress[stress.prior_rim_attempts.ge(200)]),('low_history',stress[stress.prior_rim_attempts.lt(200)])]}
    report['stress_by_season']={str(y):comparison(z,'with_blocked','control') for y,z in stress.groupby('season')}
    (OUT/'report.json').write_text(json.dumps(report,indent=2));d.to_parquet(OUT/'predictions.parquet',index=False)
    # Separate derived 2P% display diagnostic from existing saved states.
    cur=pd.read_csv(ROOT/'results/estimated_skills/current_states.csv');cur=cur[cur.pts_rate_last_season.eq(2026)].copy()
    denom=cur.p_rim_fga_rate+cur.p_mid_fga_rate
    cur['derived_two_pct']=(cur.p_rim_pct*cur.p_rim_fga_rate+cur.p_mid_pct*cur.p_mid_fga_rate)/denom.replace(0,np.nan)
    cur[['player_id','player_name','derived_two_pct','p_rim_pct','p_mid_pct','rim_pct_last_date','mid_pct_last_date']].to_csv(OUT/'derived_current_two_pct.csv',index=False)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
