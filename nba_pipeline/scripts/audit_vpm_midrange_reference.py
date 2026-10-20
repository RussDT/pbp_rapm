"""Independent standard five-shot Net audit of the VPM midrange reference rollup."""
from pathlib import Path
import json,hashlib,tempfile
import numpy as np
import pandas as pd
from export_vpm_net_lane_atoms import ROOT,OUT,SLOTS
from vpm_net_lane_targets import targets,compact
COLUMNS=['FC_DECOMP_NET_RIM_FREQ_Diff','FC_DECOMP_NET_RIM_FG_Diff','FC_DECOMP_NET_MID_FG_Diff','FC_DECOMP_NET_THREE_FREQ_Diff','FC_DECOMP_NET_THREE_FG_Diff','FC_FT_Diff']
def canonical(path):
    from process_rapm_blocks import clean_chance_utils as c
    from process_rapm_blocks.common import _finalize_df
    old=c.fetch_player_stats_supabase
    try:
        c.fetch_player_stats_supabase=lambda *a:None
        d=c.build_clean_chance_base_py(str(path),16,'2015-16','NET_MID_REFERENCE_CHECK')
    finally:c.fetch_player_stats_supabase=old
    return _finalize_df(d,['Actual_Diff','First_Chance_Diff','Is_Turnover','SC_NET_CONT_Diff','FC_CONT_KAPPA_Diff',*COLUMNS],16)
def compare(d,g,r,w):
    for col in ['game_id','event_num',*SLOTS]:np.testing.assert_array_equal(pd.to_numeric(d[col]),pd.to_numeric(r[col]))
    y,_=targets(d,g,w);a=compact(y);b=r[COLUMNS].to_numpy(float);t=r.Is_Turnover.eq(1).to_numpy()
    for j in range(6):b[t,j]=np.average(b[~t,j],weights=w[~t])
    price=np.average((r.First_Chance_Diff+r.FC_CONT_KAPPA_Diff)[~t],weights=w[~t]);b=np.column_stack([b,np.where(t,-price,0),r.SC_NET_CONT_Diff]);gap=float(np.max(abs(a-b)));assert gap<1e-8
    return dict(rows=len(d),maximum_canonical_five_shot_gap=gap)
def run():
    d=pd.read_parquet(OUT/'atoms_2016_RS.parquet');g=pd.read_parquet(OUT/'games_2016_RS.parquet');d.game_date=pd.to_datetime(d.game_date);g.game_date=pd.to_datetime(g.game_date);raw=ROOT/'raw_data/NBA16.parquet';full=compare(d,g,canonical(raw),np.ones(len(d)))
    cut=pd.Timestamp('2016-01-01');d=d.loc[d.game_date.lt(cut)].reset_index(drop=True);g=g.loc[g.game_date.lt(cut)];source=pd.read_parquet(raw);source=source.loc[pd.to_numeric(source.game_id).isin(set(pd.to_numeric(d.game_id)))]
    with tempfile.TemporaryDirectory(prefix='vpm-mid-reference-') as td:
        path=Path(td)/'NBA16.parquet';source.to_parquet(path,index=False);r=canonical(path);prefix=compare(d,g,r,np.ones(len(d)));w=np.exp2(-(d.game_date.max()-d.game_date).dt.total_seconds().to_numpy()/(86400*550));weighted=compare(d,g,r,w)
    result=dict(status='passed',full=full,fresh_prefix=prefix,fresh_prefix_TD550=weighted,no_model_refit=True,protected_outcomes=0,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());(OUT/'independent_midrange_reference_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':run()
