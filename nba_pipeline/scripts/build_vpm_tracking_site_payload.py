"""Export owner-requested expanded own-skill SQ board without private benchmarks."""
from pathlib import Path
import json
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
SITE=ROOT.parent/'vpm-sq-board-site'
SKILLS=ROOT/'nba_pipeline/results/estimated_skills/vpm_tracking_v1'
BOARD=ROOT.parent/'vpm/vpm/results/harness/vpm_own_tracking_sq_v1/2026-09-14/current_parent_board.csv'
def build():
 d=json.loads((SITE/'dist/data.json').read_text());board=pd.read_csv(BOARD).set_index('player_id');features=pd.read_parquet(SKILLS/'features.parquet');features=features.loc[features.estimate_date.eq('2026-09-14')].set_index('player_id');prov=pd.read_parquet(SKILLS/'feature_provenance.parquet');prov=prov.loc[prov.estimate_date.eq('2026-09-14')];dates={(int(x.player_id),x.feature):str(x.last_observation_date.date()) for x in prov.itertuples()}
 added=[('own_deflections100','Deflections','defense','events'),('own_contested_dreb100','Contested defensive rebounds','defense','events'),('own_potential_assists100','Potential assists','offense','events'),('own_drives100','Drives','offense','events'),('own_possession_minutes100','Possession time','offense','minutes')]
 for key,label,side,unit in added:d['features'].append(dict(key=key,label=label,side=side,units=unit+' per 100 '+('defensive' if side=='defense' else 'offensive')+' possessions',used_in_prior=True,format='rate'))
 for p in d['players']:
  pid=p['id'];r=board.loc[pid]
  for dest,src in {'off':'sq_own_defense_off','def':'sq_own_defense_def','net':'sq_own_defense_net','prior_off':'own_defense_prior_off','prior_def':'own_defense_prior_def','ev_off':'ev_own_defense_off','ev_def':'ev_own_defense_def','mid_off':'mid_residual_off','mid_def':'mid_residual_def','three_off':'three_residual_off','three_def':'three_residual_def','off_possessions':'off_possessions','def_possessions':'def_possessions'}.items():p[dest]=float(r[src])
  for f in d['features']:
   k=f['key'];v=features.loc[pid,k];p['skills'][k]=None if pd.isna(v) else float(v)
   if (pid,k) in dates:p['skill_dates'][k]=dates[pid,k]
  assert abs(p['net']-p['off']-p['def'])<1e-10
  for side in ['off','def']:assert abs(p[side]-p['ev_'+side]-p['mid_'+side]-p['three_'+side])<1e-10
 d['players'].sort(key=lambda p:p['net'],reverse=True)
 d['meta'].update(title='SQ VPM · Expanded skills research',model_id='vpm_own_tracking_sq_v1',rim_repaired=True,model_note='Own-skill SQ O/D parent with 15 offense and 9 defense inputs. Adds deflections, contested defensive rebounds, potential assists, drives and possession time, and uses repaired rim history. Positive O/D is good.',skill_note='Model-matched held skill states from vpm_tracking_v1. All observations precede the September14 cutoff; fixed historical tuning means this remains development research.')
 d['meta']['caveats'][0]='Expanded inputs are fitted into the priors and lineup model, not just displayed. Original board remains available. This is a development challenger, not an untouched-confirmed improvement.'
 (SITE/'dist/expanded.json').write_text(json.dumps(d,ensure_ascii=False,allow_nan=False,separators=(',',':')))
 print(json.dumps({'players':len(d['players']),'features':len(d['features']),'top':[(x['name'],x['net']) for x in d['players'][:5]]},indent=2))
if __name__=='__main__':build()
