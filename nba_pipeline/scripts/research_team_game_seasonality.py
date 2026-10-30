"""Known team-game-number context, integrated into shooting-state updates."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import Ridge
from research_shooting_fusion import ROOT,comparison
OUT=ROOT/'results/estimated_skills/team_game_seasonality'

def schedule(lines,ctx):
    d=lines.merge(ctx,on=['player_id','game_id'],validate='one_to_one')
    games=d[['game_id','game_date','season','season_type','team_id']].dropna(subset=['team_id']).drop_duplicates().sort_values(['game_date','game_id'])
    assert not games.duplicated(['team_id','game_id']).any()
    games['team_game']=games.groupby(['team_id','season','season_type']).cumcount()+1
    games['team_gap']=games.groupby(['team_id','season']).game_date.diff().dt.days.fillna(7)
    return d.merge(games,on=['game_id','game_date','season','season_type','team_id'],validate='many_to_one')

def filtered(d,anchor,offset,q,r):
    pred=np.zeros(len(d));neutral=np.zeros(len(d))
    for _,ids in d.groupby('player_id',sort=False).indices.items():
        theta=0.;v=.01;last=None
        for i in ids:
            row=d.iloc[i];v+=q*(0 if last is None else (row.game_date-last).days)
            neutral[i]=anchor[i]+theta;pred[i]=neutral[i]+offset[i]
            if row.three_fga>0:
                gain=v/(v+r/row.three_fga);theta+=gain*(row.three_fgm/row.three_fga-anchor[i]-offset[i]-theta);v*=1-gain
            last=row.game_date
    return np.clip(pred,.001,.999),neutral

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source=ROOT/'results/estimated_skills/shooting_fusion'
    d=pd.read_parquet(source/'predictions.parquet')
    # Build team schedule on full underlying panel, not just eligible shooters.
    lines=pd.read_parquet(ROOT/'results/estimated_skills/player_game_lines.parquet');lines=lines[lines.season.between(2019,2025)]
    ctx=pd.read_parquet(ROOT/'results/estimated_skills/game_context.parquet')
    full=schedule(lines,ctx)
    meta=full[['player_id','game_id','team_game','team_gap','is_home','team_id']].drop_duplicates(['player_id','game_id'],keep=False)
    d=d.drop(columns=['team_id','is_home'],errors='ignore').merge(meta,on=['player_id','game_id'],validate='one_to_one').sort_values(['player_id','game_date','game_id']).reset_index(drop=True)
    assert d.team_game.notna().all()
    cfg=json.loads((source/'report.json').read_text());q=cfg['base_params']['q_day'];r=cfg['base_params']['r']
    obs=np.divide(d.three_fgm,d.three_fga,out=np.zeros(len(d)),where=d.three_fga.gt(0))
    anchor=np.full(len(d),.35);xprior=pd.DataFrame({'volume':np.log1p(d.volume),'ft':d.ft})
    for year in range(2020,2026):
        train=d.season.lt(year)&d.three_fga.gt(0);test=d.season.eq(year)
        model=make_pipeline(PolynomialFeatures(2,include_bias=False),StandardScaler(),Ridge(alpha=10))
        model.fit(xprior[train],obs[train],ridge__sample_weight=d.loc[train,'three_fga'])
        anchor[test]=.35+.5*(np.clip(model.predict(xprior[test]),.01,.99)-.35)
    baseline,_=filtered(d,anchor,np.zeros(len(d)),q,r)
    assert np.max(np.abs(baseline-d.volume_ft))<1e-8,'Base reconstruction differs'
    rs=d.season_type.eq('rs').astype(float);g=(d.team_game.clip(1,82)-41)/41
    x=pd.DataFrame({'home':d.is_home-.5,'b2b':d.team_gap.eq(1).astype(float),'rest3':d.team_gap.ge(3).astype(float),'ps':1-rs,'g':g*rs,'g2':g*g*rs,'g3':g*g*g*rs})
    offsets={k:np.zeros(len(d)) for k in ['control','seasonal']};curves=[]
    for year in range(2021,2026):
        train=d.season.between(2020,year-1)&d.three_fga.gt(0);test=d.season.eq(year)
        for name,cols in [('control',['home','b2b','rest3','ps']),('seasonal',list(x.columns))]:
            model=make_pipeline(StandardScaler(),Ridge(alpha=10))
            model.fit(x.loc[train,cols],obs[train]-baseline[train],ridge__sample_weight=d.loc[train,'three_fga'])
            offsets[name][test]=model.predict(x.loc[test,cols])
            if name=='seasonal':
                for game in [1,10,20,30,41,50,60,70,82]:
                    z=(game-41)/41;row=pd.DataFrame([{'home':0,'b2b':0,'rest3':0,'ps':0,'g':z,'g2':z*z,'g3':z*z*z}])
                    curves.append({'season':year,'team_game':game,'context_pp':100*model.predict(row[cols])[0]})
    d['no_context']=baseline
    for strength in [0.,.5,1.]:
        offset=offsets['control']+strength*(offsets['seasonal']-offsets['control'])
        d[str(strength)],d['neutral_'+str(strength)]=filtered(d,anchor,offset,q,r)
        d['context_'+str(strength)]=offset
    dev=d[d.season.between(2021,2022)]
    scores={k:comparison(dev,k,'0.0') for k in ['0.0','0.5','1.0']};best=min(scores,key=lambda k:scores[k]['loss'])
    selected=best
    for k in ['0.0','0.5','1.0']:
        c=comparison(dev,k,best)
        if c['loss']<=c['control_loss']+c['se']:selected=k;break
    primary=comparison(dev,selected,'0.0');joint=comparison(dev,selected,'0.0','makes_mse')
    passed=primary['delta']>primary['se'] and joint['delta']>=-joint['se']
    report={'selected':selected,'passes_development':passed,'development':scores,'joint_gate':joint,'interpretation':'seasonal cubic increment beyond home/rest/playoff/intercept control; previous-season training; 2026 absent'}
    (OUT/'development_gate.json').write_text(json.dumps(report,indent=2))
    stress=d[d.season.between(2023,2025)]
    report['stress']={k:comparison(stress,selected,'0.0',k) for k in ['logloss','pct_mse','makes_mse']}
    report['stress_vs_no_context']={k:comparison(stress,selected,'no_context',k) for k in ['logloss','pct_mse','makes_mse']}
    report['by_season']={str(y):comparison(z,selected,'0.0') for y,z in stress.groupby('season')}
    (OUT/'report.json').write_text(json.dumps(report,indent=2));pd.DataFrame(curves).to_csv(OUT/'seasonal_curves.csv',index=False)
    d.to_parquet(OUT/'predictions.parquet',index=False)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
