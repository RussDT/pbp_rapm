"""Exact-ID anthropometric feature contract; no invented historical dates."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'cache/measurements_master.json'
FEATURES=['height_wo_shoes_in','wingspan_in','wingspan_minus_height_in']

def attach_physical(d,allow_undated=False):
    """Historical callers must explicitly acknowledge undated measurement history."""
    if not allow_undated:
        raise ValueError('Measurements have no historical availability dates; pass allow_undated=True only for current use or labeled retrospective research.')
    m=pd.read_json(SOURCE).rename(columns={'nba_id':'player_id','player':'measurement_player_name'})
    m=m[m.player_id.gt(0)].copy()
    if m.player_id.duplicated().any():raise ValueError('Duplicate measurement player IDs')
    for col in FEATURES[:2]:
        m[col]=pd.to_numeric(m[col],errors='coerce')
        m[col]=m[col].where(m[col].between(55,110))
    m['wingspan_minus_height_in']=m.wingspan_in-m.height_wo_shoes_in
    m['wingspan_minus_height_in']=m.wingspan_minus_height_in.where(m.wingspan_minus_height_in.between(-10,25))
    z=d.merge(m[['player_id','measurement_player_name',*FEATURES]],on='player_id',how='left',validate='many_to_one')
    z['physical_measurement_date']=pd.NaT
    z['physical_historical_availability_verified']=False
    for col in FEATURES:z[col+'_missing']=z[col].isna()
    return z

def export():
    out=ROOT/'results/estimated_skills/physical_features';out.mkdir(parents=True,exist_ok=True)
    current=pd.read_csv(ROOT/'results/estimated_skills/current_player_table/current_players.csv')[['player_id','Player']]
    d=attach_physical(current,allow_undated=True);d.to_csv(out/'current_players.csv',index=False)
    report={'players':len(d),'coverage':{k:int(d[k].notna().sum()) for k in FEATURES},'source':'Databallr measurements_master','date_status':'source has no measurement timestamps; historical availability unverified','units':'inches','missing_policy':'retain NaN; fit imputers only on training folds','role':'candidate predictors, not automatic credit or prior precision'}
    (out/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':export()
