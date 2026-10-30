"""Append measured-only held tracking states to a versioned own-skill package.

Development-only: frozen tracking filters were selected on 2020–2022.
No next-game opponent or schedule adjustment is exported.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
from export_vpm_own_skills import snapshot_rows
ROOT=Path(__file__).resolve().parents[1]
LANES={
 'deflections':('deflections_next_game','own_deflections100','valid','reason','defensive'),
 'contested_dreb':('tracking_next_game/contested_dreb','own_contested_dreb100','contested_dreb_valid','contested_dreb_reason','defensive'),
 'potential_assists':('tracking_next_game/potential_assists','own_potential_assists100','potential_assists_valid','potential_assists_reason','offensive'),
 'drives':('tracking_next_game/drives','own_drives100','drives_valid','drives_reason','offensive'),
 'possession_minutes':('tracking_next_game/possession_minutes','own_possession_minutes100','possession_minutes_valid','possession_minutes_reason','offensive'),
}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def build(base,out,lanes=None):
 lanes=LANES if lanes is None else lanes
 if out.exists():raise FileExistsError(out)
 manifest=json.loads((base/'manifest.json').read_text());wide=pd.read_parquet(base/'features.parquet');long=pd.read_parquet(base/'feature_provenance.parquet');rows=[];coverage=[]
 for lane,(folder,feature,valid,reason,side) in lanes.items():
  directory=ROOT/'results/estimated_skills'/folder;path=directory/'predictions.parquet';frozen=directory/'frozen.json';d=pd.read_parquet(path);cfg=json.loads(frozen.read_text())
  if not d[valid].all() or not d[reason].eq('observed').all():raise ValueError(f'{lane}: unmeasured inputs')
  d=d.sort_values(['player_id','game_date','game_id'],kind='stable').copy()
  d['value']=d[cfg['selected']+'_postgame100'];d['state_variance']=np.nan;d['observed_exposure']=d.groupby('player_id').exposure.cumsum()
  p=snapshot_rows(d[['player_id','game_id','game_date','value','state_variance','observed_exposure']],manifest['cuts']);p['feature']=feature;p['model_version']=lane+'_'+cfg['selected']+'_held_post';p['parameter_fit_end']=pd.Timestamp('2022-06-30');p['model_sha256']=sha(frozen);rows.append(p)
  w=p[['player_id','estimate_date','value']].rename(columns={'value':feature});wide=wide.merge(w,on=['player_id','estimate_date'],how='left',validate='one_to_one');wide[feature+'_missing']=wide[feature].isna().astype(int)
  manifest['feature_definitions'][feature]={'units':('minutes' if lane=='possession_minutes' else 'events')+f' per 100 {side} possessions','state':'selected measured-only postgame state, held at latest covered observation; no opponent/schedule forecast','fit_end':'2022-06-30','first_observation':str(d.game_date.min().date()),'source_field':cfg['selected']+'_postgame100'}
  manifest['sources'][str(path.resolve())]=sha(path);manifest['sources'][str(frozen.resolve())]=sha(frozen)
  for cut,g in wide.groupby('estimate_date'):coverage.append(dict(feature=feature,cut=str(cut.date()),exported_players=len(g),available=int(g[feature].notna().sum())))
 long=pd.concat([long,*rows],ignore_index=True)
 if long.duplicated(['player_id','estimate_date','feature']).any():raise ValueError('Duplicate provenance')
 if not long.last_observation_date.lt(long.estimate_date).all():raise ValueError('Future observation')
 out.mkdir(parents=True);wide.to_parquet(out/'features.parquet',index=False);long.to_parquet(out/'feature_provenance.parquet',index=False);pd.DataFrame(coverage).to_csv(out/'tracking_coverage.csv',index=False)
 manifest.update(status='owner_requested_tracking_prior_development',rows=len(wide),base_manifest_sha256=sha(base/'manifest.json'),added_tracking_features=[x[1] for x in lanes.values()])
 manifest['sources'][str(Path(__file__).resolve())]=sha(Path(__file__))
 manifest['artifacts']={n:sha(out/n) for n in ['features.parquet','feature_provenance.parquet']}
 manifest['limitations'].append('Added tracking parameters selected on 2020–2022; historical dates are development replay. Postgame filters do not remove all role/context effects. No college/G-League prior.')
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({'rows':len(wide),'features':len(manifest['feature_definitions']),'added':manifest['added_tracking_features']},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,default=ROOT/'results/estimated_skills/vpm_neutral_rim_recovered_v3');p.add_argument('--out',type=Path,default=ROOT/'results/estimated_skills/vpm_tracking_v1');a=p.parse_args();build(a.base,a.out)
