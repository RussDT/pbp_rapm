"""Compare new price-free repricing to canonical Net action/possession targets."""
from pathlib import Path
import argparse,json,tempfile,hashlib
import numpy as np
import pandas as pd
from export_vpm_net_lane_atoms import ROOT,OUT,SLOTS
from vpm_net_lane_targets import targets,LANES,REFERENCES
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n')
COLUMNS=['FC_RIM_EFG_FREQ_NET_Diff','FC_RIM_EFG_FG_NET_Diff','FC_MID_EFG_FREQ_NET_Diff','FC_MID_EFG_FG_NET_Diff','FC_THREE_EFG_FREQ_NET_Diff','FC_THREE_EFG_FG_NET_Diff','FC_FT_Diff']

def canonical(raw,year):
    from process_rapm_blocks import clean_chance_utils as C
    from process_rapm_blocks.common import _finalize_df
    old=C.fetch_player_stats_supabase
    try:
        C.fetch_player_stats_supabase=lambda *a:None;base=C.build_clean_chance_base_py(str(raw),year%100,f'{year-1}-{year%100:02d}','NET_LANE_INDEPENDENT')
    finally:C.fetch_player_stats_supabase=old
    return _finalize_df(base,['Actual_Diff','First_Chance_Diff','Second_Chance_Diff','Is_Turnover','SC_NET_CONT_Diff','FC_CONT_KAPPA_Diff',*COLUMNS],year%100)

def compare(d,g,reference,weights):
    for c in ['game_id','event_num',*SLOTS]:np.testing.assert_array_equal(pd.to_numeric(d[c]),pd.to_numeric(reference[c]))
    y,pricing=targets(d,g,weights);r=reference[COLUMNS].to_numpy(float);turn=reference.Is_Turnover.eq(1).to_numpy()
    for j in range(7):r[turn,j]=np.average(r[~turn,j],weights=weights[~turn])
    price=np.average((reference.First_Chance_Diff+reference.FC_CONT_KAPPA_Diff)[~turn],weights=weights[~turn]);r=np.column_stack([r,np.where(turn,-price,0.),reference.SC_NET_CONT_Diff]);gap=float(np.max(abs(y[:,:9]-r)));assert gap<1e-8,(gap,np.unravel_index(np.argmax(abs(y[:,:9]-r)),r.shape));return dict(rows=len(d),maximum_canonical_component_gap=gap,net_turnover_price=price)

def pilot(out):
    raw=ROOT/'raw_data/NBA16.parquet';d=pd.read_parquet(out/'atoms_2016_RS.parquet');g=pd.read_parquet(out/'games_2016_RS.parquet');d.game_date=pd.to_datetime(d.game_date);g.game_date=pd.to_datetime(g.game_date);ref=canonical(raw,2016);full=compare(d,g,ref,np.ones(len(d)));cut=pd.Timestamp('2016-01-01');subset=d.loc[d.game_date.lt(cut)].reset_index(drop=True);pg=g.loc[g.game_date.lt(cut)];source=pd.read_parquet(raw);ids=set(pd.to_numeric(subset.game_id));keep=pd.to_numeric(source.game_id).isin(ids)
    with tempfile.TemporaryDirectory(prefix='net-lane-prefix-') as td:
        path=Path(td)/'NBA16.parquet';source.loc[keep].to_parquet(path,index=False);reference=canonical(path,2016);partial=compare(subset,pg,reference,np.ones(len(subset)));w=np.exp2(-(subset.game_date.max()-subset.game_date).dt.total_seconds().to_numpy()/(86400*550));decayed=compare(subset,pg,reference,w)
    result=dict(status='passed',full_sample=full,raw_truncated_sample=partial,raw_truncated_TD550=decayed,cut=str(cut.date()),target_code_sha256=sha(Path(__file__).with_name('vpm_net_lane_targets.py')),audit_code_sha256=sha(Path(__file__)),no_component_model_fit=True);save(out/'independent_pilot_audit.json',result);print('NET CANONICAL/PREFIX PILOT PASSED',flush=True)

def package(out):
    pilot=read(out/'independent_pilot_audit.json');assert pilot['status']=='passed' and pilot['target_code_sha256']==sha(Path(__file__).with_name('vpm_net_lane_targets.py'));inputs={};artifacts={};maxgap=0.;rows=0;files=0
    for year,phase in [(2015,'RS')]+[(y,p) for y in range(2016,2027) for p in ['RS','PS']]:
        f=out/f'atoms_{year}_{phase}.parquet';gp=out/f'games_{year}_{phase}.parquet';rec=read(f.with_suffix('.json'))
        for p,h in rec['inputs'].items():assert sha(p)==h,p;inputs[p]=h
        for p,h in rec['artifacts'].items():assert sha(out/p)==h;artifacts[p]=h
        d=pd.read_parquet(f);g=pd.read_parquet(gp);y,pr=targets(d,g);gap=float(np.max(abs(y[:,:len(LANES)].sum(axis=1)-d.actual_points)));maxgap=max(maxgap,gap);assert set(pd.to_numeric(d.game_id))==set(pd.to_numeric(g.game_id));assert not g.game_id.duplicated().any()
        if year>=2016:
            old=pd.read_parquet(ROOT/f'results/monthly_vpm_targets_clean_sq_v2/atoms_{year}_{phase}.parquet',columns=['actual_points','game_id','event_num',*SLOTS]);np.testing.assert_allclose(d.actual_points,old.actual_points,atol=1e-12,rtol=0)
            for c in ['game_id','event_num',*SLOTS]:np.testing.assert_array_equal(pd.to_numeric(d[c]),pd.to_numeric(old[c]))
            oldg=pd.read_parquet(ROOT/f'results/monthly_vpm_kappa_v1/games_{year}_{phase}.parquet').set_index('game_id');newg=g.set_index('game_id').reindex(oldg.index)
            for z in ['rim','mid','three']:
                for c in ['misses','points']:np.testing.assert_allclose(newg[z+'_'+c],oldg[z+'_'+c],atol=1e-12,rtol=0)
            inputs[str(ROOT/f'results/monthly_vpm_targets_clean_sq_v2/atoms_{year}_{phase}.parquet')]=sha(ROOT/f'results/monthly_vpm_targets_clean_sq_v2/atoms_{year}_{phase}.parquet');inputs[str(ROOT/f'results/monthly_vpm_kappa_v1/games_{year}_{phase}.parquet')]=sha(ROOT/f'results/monthly_vpm_kappa_v1/games_{year}_{phase}.parquet')
        for z in ['rim','mid','three']:
            assert d[z+'_attempts'].ge(-1e-9).all();np.testing.assert_allclose(d[z+'_attempts'],np.rint(d[z+'_attempts']),atol=1e-9,rtol=0);assert d[z+'_points'].ge(-1e-9).all();assert d[z+'_points'].le((3 if z=='three' else 2)*d[z+'_attempts']+1e-9).all()
        rows+=len(d);files+=1
    inputs[str(Path(__file__).with_name('vpm_net_lane_targets.py'))]=sha(Path(__file__).with_name('vpm_net_lane_targets.py'));inputs[str(Path(__file__))]=sha(Path(__file__));save(out/'source_lock.json',inputs);result=dict(status='passed',files=files,rows=rows,maximum_row_sum_gap=maxgap,actual_point_and_possession_parity=True,raw_continuation_count_parity=True,canonical_and_truncated_pilot=pilot,source_files_verified=len(inputs),artifacts=artifacts,protected_outcomes=0,preflight='per-game shot-count game-ID join fixed before any component prior fit; initial source package archived');save(out/'independent_source_audit.json',result);print('NET SOURCE PACKAGE PASSED',files,rows,maxgap,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);p.add_argument('--stage',choices=['pilot','package','all'],default='all');a=p.parse_args()
    if a.stage in ['pilot','all']:pilot(a.out)
    if a.stage in ['package','all']:package(a.out)
