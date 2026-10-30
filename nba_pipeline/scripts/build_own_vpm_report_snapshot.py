"""Prepare reviewed local data for the own-skill O/D HTML research report. Does not rerun models."""
import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
root=Path(__file__).resolve().parents[2]
b=root.parent/'vpm/vpm/results/harness/vpm_own_skills_parent_integration_v1'
f=root/'nba_pipeline/results/estimated_skills/vpm_neutral_v1'
out=root/'nba_pipeline/results/estimated_skills/own_vpm_report_snapshot.json'
current=b/'2026-09-14'
board=pd.read_csv(current/'current_parent_board.csv')
features=pd.read_parquet(f/'features.parquet'); features=features[features.estimate_date.eq('2026-09-14')].drop(columns='estimate_date')
board=board.merge(features,on='player_id',how='left',validate='one_to_one')
models=json.loads((current/'prior_models.json').read_text())
coefs=pd.read_csv(current/'prior_coefficients.csv')
meta=json.loads((f/'manifest.json').read_text())
labels={'rim_pct':'Rim make probability','mid_pct':'Midrange make probability','three_pct':'Three-point make probability','ft_pct':'Free-throw make probability','rim_fga_rate':'Rim attempts','mid_fga_rate':'Midrange attempts','three_fga_rate':'Three-point attempts','fta_rate':'Free-throw attempts','pts_rate':'Points','ast_rate':'Assists','tov_rate':'Turnovers','oreb_rate':'Offensive rebounds','usg_pct':'Usage events','dreb_rate':'Defensive rebounds','stl_rate':'Steals','blk_rate':'Blocks','pf_committed_rate':'Personal fouls','rim_saved100':'Rim points saved','rim_contests100':'Rim contests','rim_suppression':'Rim suppression','dfga100':'Defended field-goal attempts','rim_makes100':'Rim makes','mid_makes100':'Midrange makes','three_makes100':'Three-point makes','ft_makes100':'Free-throw makes'}
feature_rows=[]
for name,d in meta['feature_definitions'].items():
    used=[]
    for side in ['off','def']:
        if name in models['own_defense'][side]['imputation_means']: used.append('Offense' if side=='off' else 'Defense')
    feature_rows.append({'feature':name,'label':labels[name[4:]],'units':d['units'].replace('off_poss','offensive possessions').replace('def_poss','defensive possessions').replace('tot_poss','combined possessions'),'used':', '.join(used) or 'Exported; not directly mapped','missing_current':int(board[name].isna().sum()),'definition':d.get('formula') or (d.get('numerator','')+' / '+d.get('denominator','') if 'numerator' in d else d['source_field'])})
provenance=pd.read_parquet(f/'feature_provenance.parquet'); provenance=provenance[provenance.estimate_date.eq('2026-09-14') & provenance.player_id.isin(board.player_id)]
provenance=provenance[['player_id','feature','last_observation_date','age_days','is_missing','model_version']]
checks=[]
for arm in models:
 for side in ['off','def']:
  c=coefs[(coefs.arm==arm)&(coefs.side==side)]
  pred=pd.Series(0.,index=board.index)
  for row in c.itertuples():
   if row.feature=='intercept': x=1.
   elif row.feature.endswith('_missing'): x=board[row.feature[:-8]].isna().astype(float)
   else: x=board[row.feature].fillna(models[arm][side]['imputation_means'][row.feature])
   pred+=x*row.coefficient_points100
  gap=(pred-board[f'{arm}_prior_{side}']).abs().max(); checks.append(float(gap))
assert max(checks)<1e-10, checks
comparisons=[]
refnames={'rapm_reference':'Plain RAPM · 2000 / 4000','matched_no_prior':'Same solve, zero prior','own_core':'Own core · blocks, steals, DREB','epm_skill_parent':'Corrected EPM-skill parent','epm_benchmark':'Complete EPM'}
for ref,label in refnames.items():
 d=json.loads((b/'2024-10-01'/f'bootstrap_vs_{ref}.json').read_text()); r=next(r for r in d['results'] if r['candidate']=='own_defense')
 comparisons.append({'reference':label,'gain':r['gain_vs_reference_x1e6'],'low':r['ci_low_x1e6'],'high':r['ci_high_x1e6']})
def rows(df):return json.loads(df.to_json(orient='records',date_format='iso'))
def query(data,files,grain,note):
 return {'rows':data,'source':{'type':'local research artifacts','files':files,'description':note,'grain':grain,'period':'Current snapshot 2026-09-14; replay cut 2024-10-01 for 2024–25','evidenceFlow':[{'title':'Saved research output','detail':'Read the named frozen artifacts; no solver or feature selection rerun.'}],'caveats':[note]},'methods':[{'language':'text','code':'Read saved files; preserve player IDs and numeric precision. Current board joins own feature snapshot by player_id. Comparison rows select own_defense in each reference bootstrap receipt.'}]}
queries={
 'board':query(rows(board),['current_parent_board.csv','features.parquet'],'one player in latest-season PBP participation universe','Research O/D parent. Positive defense is good; net = offense + defense. This is not a verified current roster.'),
 'coefficients':query(rows(coefs),['prior_coefficients.csv'],'arm × side × coefficient','Positive-good points per 100 per feature unit; includes 1.5 prior scale and intercept. Coefficients are joint associations, not causal values or importance.'),
 'features':query(feature_rows,['manifest.json','features.parquet'],'one exported feature','25 exported features; extended model directly maps 12 offense and 7 defense inputs. Missingness counts cover 582 players.'),
 'comparisons':query(comparisons,[f'bootstrap_vs_{r}.json' for r in refnames],'one model comparison on the same 93,034 possessions','Positive MSE gain favors extended own. Paired game-clustered 95% intervals; 2,000 replicates. Development only; expected-FT scoring target.'),
 'provenance':query(rows(provenance),['feature_provenance.parquet'],'player × feature','Latest observed neutral states strictly before the cut; held across idle days, missing after 730 days.'),
 'method':query([{'as_of':'2026-09-14','players':582,'training_possessions':2781191,'latest_training_date':'2026-06-13','off_alpha':4000,'def_alpha':8000,'prior_scale':1.5,'half_life_days':550,'mapper_alpha':.1,'replay_possessions':93034,'replay_games':1054,'all_replay_possessions':243859,'tests_passed':61,'prior_reconstruction_max_error':max(checks)}],['manifest.json','verification.json','OWN_SKILL_PARENT_INTEGRATION.md'],'one saved build','Historical skill parameters were selected through 2025. No untouched EPM victory or component admission.'),
}
# Exact training-only imputation values needed for transparent player calculations.
queries['imputation']=query([{'arm':a,'side':s,'feature':k,'mean':v} for a,ss in models.items() for s,m in ss.items() for k,v in m['imputation_means'].items()],['prior_models.json'],'arm × side × feature','Weighted training-only means; missing values are not measured zeroes.')
snapshot={'title':'Building our predictive VPM','surface':'report','status':'research','generatedAt':datetime.now(timezone.utc).isoformat(),'report':{'asOf':'2026-09-14'},'filters':[],'queries':queries}
out.write_text(json.dumps(snapshot,ensure_ascii=False,allow_nan=False))
print(out);print('players',len(board),'max reconstruction gap',max(checks),'snapshot bytes',out.stat().st_size)
