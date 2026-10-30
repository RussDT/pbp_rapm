"""Export saved component VPM outputs only; no raw provider skills."""
from pathlib import Path
import json
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT.parent/'vpm/vpm/results/harness/vpm_offseason_alpha_incumbent_board_v1_2026-08-21/player_estimates_incumbent_v4_x2_2026-08-21.csv'
TERMS={'DECOMP_NET_RIM_FREQ':'Rim frequency','DECOMP_NET_RIM_FG':'Rim conversion','DECOMP_NET_MID_FG':'Midrange SQ','DECOMP_NET_THREE_FREQ':'Three-point frequency','DECOMP_NET_THREE_FG':'Three-point SQ','ALT_FT':'Free throws','ALT_TOV_NET_VALUE':'Turnovers','DECOMP_NET_MID_VALUE':'Midrange residual','DECOMP_NET_EFG':'Three-point residual','SECOND_CHANCE_CLEAN':'Second chance'}
def build(source=SOURCE,output="components.json",meta=None):
 df=pd.read_csv(source);df=df.loc[df.current_player.eq(True)]
 features=[dict(key=prefix+'_'+key,label=label,side=side,units='points per 100 possessions',format='impact',used_in_prior=False) for side,prefix in [('offense','o'),('defense','d')] for key,label in TERMS.items()]
 names={int(p['id']):p['name'] for p in json.loads((ROOT.parent/'vpm-sq-board-site/dist/fouls.json').read_text())['players']}
 players=[]
 for row in df.to_dict('records'):
  p=dict(id=int(row['player_id']),name=names.get(int(row['player_id']),row['player_name']),off=row['vpm_off'],def_=row['vpm_def'],net=row['vpm_net'],off_possessions=row['off_poss'],def_possessions=row['def_poss'],skills={f['key']:row[f['key']] for f in features},skill_dates={})
  p['def']=p.pop('def_');p.update({k:row[k] for k in ['oTS','oTOV','oSC','dTS','dTOV','dSC']})
  assert abs(p['net']-p['off']-p['def'])<1e-10
  for side,prefix in [('off','o'),('def','d')]:assert abs(p[side]-sum(row[prefix+'_'+k] for k in TERMS))<1e-10
  players.append(p)
 d=dict(meta=dict(architecture='components',as_of='2026-08-21',model_id='incumbent_v4_x2',model_note='Saved component VPM incumbent: EPM predictive-skill priors ×2, component-specific penalties, TD550. Games through June 13, 2026. No full-EPM blend. Research snapshot; not a refit with our new own-skill inputs.'),features=features,players=players)
 if meta:d['meta'].update(meta)
 (ROOT.parent/'vpm-sq-board-site/dist'/output).write_text(json.dumps(d,ensure_ascii=False,allow_nan=False,separators=(',',':')))
 print(f'{len(players)} players; O/D/component closure passed')
if __name__=='__main__':build()
