"""Same-day cached EPM vs original and deflection-enhanced steals."""
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from compare_nba_estimated_skills_to_epm import SNAPSHOT_DATES
from build_deflections_next_game import compare
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/estimated_skills';OUT=BASE/'enhanced_steals_epm'
OUT.mkdir(parents=True,exist_ok=True)
train=pd.read_parquet(BASE/'deflections_steals_increment/comparison_rows.parquet');train=train[train.season.between(2020,2025)]
model=make_pipeline(StandardScaler(),Ridge(alpha=10))
model.fit(train[['kalman','deflections']],train.target100-train.kalman,ridge__sample_weight=train.exposure)
s=pd.read_parquet(BASE/'predictions.parquet');s=s[s.skill.eq('stl_rate')&s.season.eq(2026)].copy()
f=pd.read_parquet(BASE/'deflections_next_game/predictions.parquet')
keys=['player_id','game_id','game_date'];s=s.merge(f[keys+['next_ou_pregame100','step_k']],on=keys,validate='one_to_one');s=s[s.step_k.gt(0)]
s['kalman']=100*s.pred_rate;s['deflections']=s.next_ou_pregame100;s['target100']=100*s.obs_rate
s['enhanced']=np.maximum(0,s.kalman+model.predict(s[['kalman','deflections']]))
rows=[]
for day in SNAPSHOT_DATES:
    e=pd.read_csv(ROOT/f'cache/epm/epm_all_{day}_go0.csv');assert e.player_id.is_unique
    assert e.requested_date.eq(day).all() and pd.to_datetime(e.date).le(pd.Timestamp(day)).all()
    d=s[s.game_date.eq(pd.Timestamp(day))].merge(e[['player_id','p_stl_100','date']],on='player_id',validate='one_to_one')
    d['epm']=d.p_stl_100;d['snapshot']=day;rows.append(d)
d=pd.concat(rows,ignore_index=True).dropna(subset=['epm','enhanced','kalman'])
assert not d.duplicated(['player_id','game_id']).any()
r={'enhanced_vs_epm':compare(d,'enhanced','epm'),'original_vs_epm':compare(d,'kalman','epm'),'enhanced_vs_original':compare(d,'enhanced','kalman'),'rows':len(d),'games':int(d.game_id.nunique()),'snapshots_with_games':int(d.snapshot.nunique()),'fit_seasons':[2020,2025]}
r['bias100']={k:float(np.average(d[k]-d.target100,weights=d.exposure)) for k in ['enhanced','kalman','epm']}
(OUT/'report.json').write_text(json.dumps(r,indent=2));d.to_parquet(OUT/'private_rows.parquet',index=False)
print(json.dumps(r,indent=2))
