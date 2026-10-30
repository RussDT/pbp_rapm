"""Chronological 3P fusion with binomial and joint makes gates."""
import json,hashlib,argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/estimated_skills/shooting_fusion'
GRID=[(q,r) for q in [1e-7,1e-6,1e-5] for r in [.25,1.]]
ORDER=['kalman','ew30','ew180','ew730','volume','volume_ft','linear10','linear1000','boost']

def states(d,q,r):
    out=np.zeros((len(d),11))
    for _,ids in d.groupby('player_id',sort=False).indices.items():
        theta=.35;v=.01;last=None;tot3=totft=games=0;team=None
        makes=np.array([7.,7.,7.]);att=np.array([20.,20.,20.]);fm=15.;fa=20.;va=12.;ve=200.
        for i in ids:
            row=d.iloc[i];date=row.game_date
            dt=0 if last is None else (date-last).days
            decay=np.exp(-np.log(2)*dt/np.array([30.,180.,730.]));makes*=decay;att*=decay
            dec=np.exp(-np.log(2)*dt/180);fm*=dec;fa*=dec;va*=dec;ve*=dec
            v+=q*dt
            out[i]=[theta,v,*list(makes/np.maximum(att,1e-12)),fm/max(fa,1e-12),100*va/max(ve,1e-12),tot3,totft,games,float(team is not None and pd.notna(team) and pd.notna(row.team) and team!=row.team)]
            if row.three_fga>0:
                gain=v/(v+r/row.three_fga);theta+=gain*(row.three_fgm/row.three_fga-theta);v*=1-gain
            makes+=row.three_fgm;att+=row.three_fga;fm+=row.ftm;fa+=row.fta;va+=row.three_fga;ve+=row.off_poss
            tot3+=row.three_fga;totft+=row.fta;games+=1;last=date;team=row.team
    return out

def residual_filter(d,anchor,q,r):
    pred=np.empty(len(d));post=np.empty(len(d))
    for _,ids in d.groupby('player_id',sort=False).indices.items():
        theta=0.;v=.01;last=None
        for i in ids:
            row=d.iloc[i];dt=0 if last is None else (row.game_date-last).days;v+=q*dt
            pred[i]=anchor[i]+theta
            if row.three_fga>0:
                gain=v/(v+r/row.three_fga);theta+=gain*(row.three_fgm/row.three_fga-anchor[i]-theta);v*=1-gain
            post[i]=anchor[i]+theta;last=row.game_date
    return np.clip(pred,.001,.999),np.clip(post,.001,.999)

def losses(d,p,kind):
    p=np.clip(p,.001,.999);a=d.three_fga.to_numpy();m=d.three_fgm.to_numpy()
    if kind=='logloss':return -(m*np.log(p)+(a-m)*np.log1p(-p)),a
    if kind=='pct_mse':return a*(p-np.divide(m,a,out=np.zeros(len(a)),where=a>0))**2,a
    w=d.off_poss.to_numpy();return (100*d.volume.to_numpy()/100*p-100*m/w)**2*w,w

def comparison(d,a,b,kind='logloss',draws=1000):
    la,w=losses(d,d[a].to_numpy(),kind);lb,_=losses(d,d[b].to_numpy(),kind)
    rng=np.random.default_rng(20260914);boot=np.zeros(draws);ma=[];mb=[]
    for year in sorted(d.season.unique()):
        take=d.season.eq(year).to_numpy();ids=pd.factorize(d.loc[take,'game_id'])[0];n=ids.max()+1
        aa=np.bincount(ids,weights=la[take]);bb=np.bincount(ids,weights=lb[take]);ww=np.bincount(ids,weights=w[take])
        ma.append(aa.sum()/ww.sum());mb.append(bb.sum()/ww.sum())
        for k in range(0,draws,100):
            c=rng.multinomial(n,np.full(n,1/n),size=min(100,draws-k));den=c@ww
            boot[k:k+len(c)]+=(c@(bb-aa))/np.maximum(den,1e-12)/d.season.nunique()
    delta=float(np.mean(mb)-np.mean(ma))
    return dict(candidate=a,control=b,metric=kind,loss=float(np.mean(ma)),control_loss=float(np.mean(mb)),gain_pct=100*delta/np.mean(mb),delta=delta,se=float(boot.std(ddof=1)),ci95=np.percentile(boot,[2.5,97.5]).tolist(),rows=len(d),attempts=float(d.three_fga.sum()))

def main():
    global OUT
    ap=argparse.ArgumentParser()
    ap.add_argument('--physical-features',action='store_true')
    ap.add_argument('--allow-undated-measurements',action='store_true')
    args=ap.parse_args()
    if args.physical_features:
        if not args.allow_undated_measurements:ap.error('Physical snapshot is undated: explicitly allow retrospective measurements or omit physical features')
        OUT=OUT.parent/'shooting_fusion_physical_retrospective'
    OUT.mkdir(parents=True,exist_ok=True)
    source=ROOT/'results/estimated_skills/player_game_lines.parquet'
    d=pd.read_parquet(source);d=d[d.season.between(2019,2025)&d.off_poss.gt(0)].copy()
    # Team is for diagnostic only, reconstructed from exact tracking IDs/dates.
    t=pd.read_parquet(ROOT/'results/estimated_skills/tracking_backfill/tracking_backfill.parquet')
    t['game_date']=pd.to_datetime(t.date.astype(str));t=t.rename(columns={'PLAYER_ID':'player_id','TEAM_ID':'team'})
    keys=['player_id','game_date','season'];t=t.drop_duplicates(keys,keep=False)
    d=d.merge(t[keys+['team']],on=keys,how='left',validate='many_to_one').sort_values(['player_id','game_date','game_id']).reset_index(drop=True)
    duplicates=d.duplicated(['player_id','game_date'],keep=False)
    (OUT/'data_audit.json').write_text(json.dumps({'ambiguous_player_date_rows_excluded':int(duplicates.sum())}))
    d=d[~duplicates].reset_index(drop=True)
    assert d[['three_fga','three_fgm','fta','ftm','off_poss']].notna().all().all()
    assert ((d.three_fgm<=d.three_fga)&(d.ftm<=d.fta)&d.three_fgm.ge(0)&d.ftm.ge(0)).all()
    # Fit base hyperparameters exclusively on warm-up season, then replay.
    warm=d[d.season.eq(2019)].reset_index(drop=True);scores=[]
    for q,r in GRID:
        x=states(warm,q,r);ll,w=losses(warm,x[:,0],'logloss');scores.append(float(ll.sum()/w.sum()))
    q,r=GRID[int(np.argmin(scores))]
    print('base selected',q,r,flush=True)
    x=states(d,q,r)
    for k,v in zip(['kalman','variance','ew30','ew180','ew730','ft','volume','prior3pa','priorfta','games','team_change'],x.T):d[k]=v
    obs=np.divide(d.three_fgm,d.three_fga,out=np.zeros(len(d)),where=d.three_fga.gt(0))
    anchors={k:np.full(len(d),.35) for k in ['volume','volume_ft']}
    for year in range(2020,2026):
        train=d.season.lt(year)&d.three_fga.gt(0);test=d.season.eq(year)
        for name,cols in [('volume',['volume']),('volume_ft',['volume','ft'])]:
            z=d[cols].copy();z['volume']=np.log1p(z.volume)
            model=make_pipeline(PolynomialFeatures(2,include_bias=False),StandardScaler(),Ridge(alpha=10))
            model.fit(z[train],obs[train],ridge__sample_weight=d.loc[train,'three_fga'])
            anchors[name][test]=.35+.5*(np.clip(model.predict(z[test]),.01,.99)-.35)
    volume=d.volume.copy()
    for name,anchor in anchors.items():d[name+'_pred'],_=residual_filter(d,anchor,q,r)
    # Keep volume rate as its own feature; prediction label volume lives in evaluation frame.
    feats=pd.DataFrame({'base':d.kalman,'fast_delta':d.ew30-d.kalman,'medium_delta':d.ew180-d.kalman,'slow_delta':d.ew730-d.kalman,'ft':d.ft,'logvolume':np.log1p(volume),'variance':d.variance,'log3pa':np.log1p(d.prior3pa),'logfta':np.log1p(d.priorfta)})
    if args.physical_features:
        from estimated_skills_physical import attach_physical,FEATURES
        physical=attach_physical(d[['player_id']],allow_undated=True)
        for col in FEATURES:
            feats[col]=physical[col].to_numpy()
            feats[col+'_missing']=physical[col].isna().astype(float).to_numpy()
        (OUT/'physical_coverage.json').write_text(json.dumps({'rows':len(d),'covered_rows':int(physical.wingspan_in.notna().sum()),'historical_availability_verified':False}))
    for name in ['linear10','linear1000','boost']:d[name]=d.kalman
    for year in range(2021,2026):
        train=d.season.between(2020,year-1)&d.three_fga.gt(0);test=d.season.eq(year)
        for name in ['linear10','linear1000','boost']:
            if name=='boost':model=make_pipeline(SimpleImputer(strategy='median'),HistGradientBoostingRegressor(max_iter=100,learning_rate=.05,max_leaf_nodes=4,min_samples_leaf=200,l2_regularization=100,early_stopping=False,random_state=20260914))
            else:model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=float(name[6:])))
            kwargs={'histgradientboostingregressor__sample_weight':d.loc[train,'three_fga']} if name=='boost' else {'ridge__sample_weight':d.loc[train,'three_fga']}
            model.fit(feats[train],obs[train]-d.loc[train,'kalman'],**kwargs)
            d.loc[test,name]=np.clip(d.loc[test,'kalman']+model.predict(feats[test]),.001,.999)
        print('fusion year',year,flush=True)
    # comparisons need separate model and exposure-prediction names.
    d['volume_rate']=volume
    evals=d.rename(columns={'volume':'unused_volume','volume_pred':'volume_model','volume_ft_pred':'volume_ft'})
    evals['volume']=volume
    order=[('volume_model' if k=='volume' else k) for k in ORDER]
    dev=evals[evals.season.between(2021,2022)]
    measures={k:comparison(dev,k,'kalman') for k in order}
    best=min(order,key=lambda k:measures[k]['loss'])
    selected=best
    for k in order:
        c=comparison(dev,k,best)
        if c['loss']<=c['control_loss']+c['se']:selected=k;break
    controls=order[:6];control=min(controls,key=lambda k:measures[k]['loss'])
    gate=comparison(dev,selected,control);joint=comparison(dev,selected,control,'makes_mse')
    if selected not in controls and not(gate['delta']>gate['se'] and joint['delta']>=-joint['se']):selected=control
    report=dict(physical_features=args.physical_features,physical_history_verified=False if args.physical_features else None,base_params={'q_day':q,'r':r},development=measures,selected=selected,strongest_simple=control,development_gate=gate,joint_gate=joint)
    (OUT/'development_gate.json').write_text(json.dumps(report,indent=2))
    stress=evals[evals.season.between(2023,2025)]
    report['stress']={}
    for label,part in [('all',stress),('under1',stress[stress.volume.lt(1)]),('under3',stress[stress.volume.lt(3)]),('volume5plus',stress[stress.volume.ge(5)]),('first20',stress[stress.games.lt(20)]),('team_change',stress[stress.team_change.eq(1)])]:
        report['stress'][label]={kind:comparison(part,selected,control,kind) for kind in ['logloss','pct_mse','makes_mse']}
    report['stress_vs_kalman']={kind:comparison(stress,selected,'kalman',kind) for kind in ['logloss','pct_mse','makes_mse']}
    report['selected_vs_volume']={label:{kind:comparison(part,selected,'volume_model',kind) for kind in ['logloss','pct_mse','makes_mse']} for label,part in [('all',stress),('under1',stress[stress.volume.lt(1)]),('under3',stress[stress.volume.lt(3)])]}
    snapshot=evals[evals.season.ge(2021)][['player_id','game_id','game_date','season','volume',selected]].copy()
    snapshot['model']=selected; snapshot['fit_through_season']=snapshot.season-1
    snapshot['expected_3pm100']=snapshot.volume*snapshot[selected]
    snapshot.rename(columns={selected:'pregame_three_pct','volume':'pregame_3pa100'}).to_parquet(OUT/'selected_dated_predictions.parquet',index=False)
    report['by_season']={str(y):comparison(part,selected,control) for y,part in stress.groupby('season')}
    calibration=stress.assign(pred=stress[selected],bucket=pd.cut(stress[selected],[0,.2,.3,.35,.4,.45,1]))
    records=[]
    for b,p in calibration.groupby('bucket',observed=True):
        records.append({'bucket':str(b),'attempts':p.three_fga.sum(),'predicted':np.average(p.pred,weights=p.three_fga),'observed':p.three_fgm.sum()/p.three_fga.sum()})
    pd.DataFrame(records).to_csv(OUT/'calibration.csv',index=False)
    report['dependencies']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),source,ROOT/'docs/SHOOTING_FUSION.md']}
    (OUT/'report.json').write_text(json.dumps(report,indent=2))
    evals.to_parquet(OUT/'predictions.parquet',index=False)
    print('selected',selected,'simple',control,flush=True)
    print(json.dumps(report['stress']['all'],indent=2),flush=True)
if __name__=='__main__':main()
