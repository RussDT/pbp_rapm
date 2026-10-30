"""Append individual lagged TD550 snapshot components to own skills."""
from pathlib import Path
import json,hashlib
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def select_snapshot(snapshots,cut):
 eligible=[x for x in snapshots if pd.Timestamp(x[0])<=pd.Timestamp(cut)]
 if not eligible:raise ValueError('No pre-cut factor snapshot')
 return max(eligible,key=lambda x:x[0])
def build():
 base=ROOT/'results/estimated_skills/vpm_fouls_v1';out=ROOT/'results/estimated_skills/vpm_factors_v1'
 if out.exists():raise FileExistsError(out)
 m=json.loads((base/'manifest.json').read_text());wide=pd.read_parquet(base/'features.parquet');long=pd.read_parquet(base/'feature_provenance.parquet');snapshots=[]
 for p in (ROOT/'results/vpm_netcont_asof_td550').glob('*/*/manifest.json'):
  receipt=json.loads(p.read_text());snapshots.append((receipt['cutoff_date'],p,receipt))
 records=[];add=[]
 for cut,g in wide.groupby('estimate_date',sort=True):
  date,path,receipt=select_snapshot(snapshots,cut);source=path.parent/'player_estimates.csv';d=pd.read_csv(source).set_index('player_id');assert receipt['prior_scale']==0 and receipt['half_life_days']==550
  assert d.as_of_date.eq(date).all() and not d.index.duplicated().any()
  cols=[c for c in d.columns if c.startswith(('o_','d_'))];assert len(cols)==20
  source_hash=sha(source)
  for c in cols:
   key='td550_'+c;values=g.player_id.map(d[c]);wide.loc[g.index,key]=values.to_numpy();wide.loc[g.index,key+'_missing']=values.isna().astype(int).to_numpy()
   for pid,value in zip(g.player_id,values):
    add.append(dict(player_id=pid,game_id=None,value=value,state_variance=float('nan'),observed_exposure=float('nan'),estimate_date=cut,last_observation_date=pd.Timestamp(date)-pd.Timedelta(days=1),age_days=(cut-pd.Timestamp(date)).days+1,is_missing=pd.isna(value),feature=key,model_version='legacy_td550_no_prior_repaired_snapshot',parameter_fit_end=pd.Timestamp('2026-09-06'),model_sha256=source_hash,source_snapshot_date=pd.Timestamp(date),data_end_exclusive=pd.Timestamp(date),observation_date_is_upper_bound=True))
   m['feature_definitions'][key]=dict(units='points per 100 possessions, positive good',state='held individual no-prior TD550 component',source_field=c,date_rule='latest snapshot <= cut; snapshot excludes its date; provenance last_observation_date is conservative data upper bound, not player last game')
  m['sources'][str(source)]=sha(source);m['sources'][str(path)]=sha(path);records.append(dict(cut=str(cut.date()),source_snapshot=date,matched=int(g.player_id.isin(d.index).sum()),rows=len(g)))
 long=pd.concat([long,pd.DataFrame(add)],ignore_index=True);assert long.last_observation_date.lt(long.estimate_date).all()
 out.mkdir();wide.to_parquet(out/'features.parquet',index=False);long.to_parquet(out/'feature_provenance.parquet',index=False)
 m.update(base_manifest_sha256=sha(base/'manifest.json'),status='development_factor_feature_challenger',factor_snapshot_alignment=records);m['artifacts']={x:sha(out/x) for x in ['features.parquet','feature_provenance.parquet']};m['limitations'].append('Retrospectively built and repaired dated TD550 factors; fixed model choices not pristine as-of validation. Prior and likelihood share past PBP; require incremental forward comparison. No total O/D/net or factor sums enter.')
 (out/'manifest.json').write_text(json.dumps(m,indent=2));print(json.dumps(records,indent=2))
if __name__=='__main__':build()
