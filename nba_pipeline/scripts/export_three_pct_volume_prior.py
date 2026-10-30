"""Export the frozen volume-prior challenger without scoring 2026 outcomes."""
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.linear_model import Ridge
from research_three_pct_volume_prior import ROOT,OUT,volume_features
from build_nba_estimated_skills import skill_subpanel,SKILLS,FilterLayout,run_filter
full=pd.read_parquet(ROOT/'player_game_lines.parquet').sort_values(['player_id','game_date','game_id']).reset_index(drop=True)
full=full[full.season.le(2026)&full.off_poss.gt(0)].copy()
warm=full[full.season.eq(2019)]
full['prior_3pa100']=volume_features(full,warm.three_fga.sum()/warm.off_poss.sum())
sub=skill_subpanel(full,next(s for s in SKILLS if s.name=='three_pct'))
cfg=json.loads((ROOT/'hyperparams.json').read_text())['three_pct']
strength=float(json.loads((OUT/'development_gate.json').read_text())['selected_strength'])
mean=cfg['league_mean_rate_pre_holdout'];prior=np.full(len(sub),mean)
for year in range(2020,2027):
    train=sub.season.lt(year);test=sub.season.eq(year)
    model=make_pipeline(PolynomialFeatures(2,include_bias=False),StandardScaler(),Ridge(alpha=10))
    model.fit(np.log1p(sub.loc[train,['prior_3pa100']]),sub.loc[train,'obs_rate'],ridge__sample_weight=sub.loc[train,'exposure'])
    prior[test]=np.clip(model.predict(np.log1p(sub.loc[test,['prior_3pa100']])),.01,.99)
anchor=mean+strength*(prior-mean)
_,_,post,var=run_filter(FilterLayout(sub),sub.obs_rate.to_numpy()-anchor,list(cfg['variants']['uncentered']['params'].values()),0.)
sub['new_pct']=np.clip(anchor+post,0,1);sub['new_var']=var
last=sub.groupby('player_id').tail(1).set_index('player_id')
folder=ROOT/'current_player_table'
p=folder/'current_players_provenance.csv'
s=pd.read_csv(p)
if 'p_three_pct_v1' not in s:s['p_three_pct_v1']=s.p_three_pct
s['p_three_pct']=s.player_id.map(last.new_pct)
s['p_three_pct_var']=s.player_id.map(last.new_var)
s['three_pct_last_date']=s.player_id.map(last.game_date)
s['three_pct_last_season']=s.player_id.map(last.season)
s['three_pct_model']='volume_prior_strength0.5_research'
s.to_csv(p,index=False)
d=pd.read_csv(folder/'current_players.csv');d['3P%']=d.player_id.map(s.set_index('player_id').p_three_pct)*100
d.to_csv(folder/'current_players.csv',index=False)
page=(folder/'index.html').read_text()
start=page.index('const rows=')+len('const rows=');end=page.index(';let sort=',start)
page=page[:start]+d.to_json(orient='records').replace('</','<\\/')+page[end:]
page=page.replace('PBP: v1 export; deflections: next-game OU.','PBP: v1 except 3P%: volume-informed prior (strength 0.5, research); deflections: next-game OU. 3P% uses its latest shooting observation; it is not advanced through later zero-attempt games. Model variance excludes fitted-prior uncertainty.')
(folder/'index.html').write_text(page)
assert d.player_id.is_unique and len(d)==len(s)
print(s.loc[s.player_id.eq(203991),['Player','p_three_pct_v1','p_three_pct','three_pct_last_date']].to_string(index=False))
