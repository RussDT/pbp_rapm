"""Historical pregame deflection increment for steals forecasts."""
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from build_deflections_next_game import compare
ROOT=Path(__file__).resolve().parents[1]/'results/estimated_skills';OUT=ROOT/'deflections_steals_increment'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    s=pd.read_parquet(ROOT/'predictions.parquet');s=s[s.skill.eq('stl_rate') & s.season.between(2019,2025)].copy()
    f=pd.read_parquet(ROOT/'deflections_next_game/predictions.parquet')
    keys=['player_id','game_id','game_date']
    assert not s.duplicated(keys).any() and not f.duplicated(keys).any()
    d=s.merge(f[keys+['next_ou_pregame100','step_k']],on=keys,validate='one_to_one')
    d=d[d.step_k.gt(0)].copy();d['target100']=100*d.obs_rate;d['kalman']=100*d.pred_rate
    d['deflections']=d.next_ou_pregame100
    for c in ['calibrated','plus_deflections']:d[c]=np.nan
    coefs=[]
    for year in range(2021,2026):
        train=d.season.between(2020,year-1);test=d.season.eq(year)
        for name,cols in [('calibrated',['kalman']),('plus_deflections',['kalman','deflections'])]:
            model=make_pipeline(StandardScaler(),Ridge(alpha=10))
            model.fit(d.loc[train,cols],d.loc[train,'target100']-d.loc[train,'kalman'],ridge__sample_weight=d.loc[train,'exposure'])
            d.loc[test,name]=np.maximum(0,d.loc[test,'kalman']+model.predict(d.loc[test,cols]))
            if name=='plus_deflections':coefs.append({'season':year,'deflection_coefficient':model[-1].coef_[-1]/model[0].scale_[-1]})
    dev=d[d.season.between(2021,2022)]
    report={'method':'Prior-season residual ridge alpha10; pregame estimates only; historical research because existing base hyperparameters used broader development windows','development':compare(dev,'plus_deflections','calibrated'),'coefficients':coefs}
    c=report['development'];report['passes_development']=c['delta_mse']>c['delta_se']
    (OUT/'development_gate.json').write_text(json.dumps(report,indent=2))
    stress=d[d.season.between(2023,2025)]
    report['stress_vs_calibrated']=compare(stress,'plus_deflections','calibrated')
    report['stress_vs_original']=compare(stress,'plus_deflections','kalman')
    report['by_season']={str(y):compare(x,'plus_deflections','calibrated') for y,x in stress.groupby('season')}
    report['coverage']={'available_steals_rows':len(s),'matched_prior_deflection_rows':len(d),'unmatched_or_no_prior':len(s)-len(d)}
    (OUT/'report.json').write_text(json.dumps(report,indent=2));d.to_parquet(OUT/'comparison_rows.parquet',index=False)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
