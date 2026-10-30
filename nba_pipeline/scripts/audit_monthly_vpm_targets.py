"""Compare cached count repricing with a full raw-PBP cutoff rebuild."""
import argparse
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from build_monthly_own_skills import ROOT, write_json, sha
from build_monthly_vpm_targets import OUT, reprice_before_cut
from build_monthly_vpm_kappa import OUT as KAPPA
from build_vpm_target_swap import attach_ev_targets
from process_rapm_blocks import clean_chance_utils as clean
from process_rapm_blocks.common import _finalize_df


def audit(year,cut):
    source=OUT/f'atoms_{year}_RS.parquet';atoms=pd.read_parquet(source)
    kappa_path=KAPPA/f'games_{year}_RS.parquet'
    expected,pricing=reprice_before_cut(atoms,cut,pd.read_parquet(kappa_path))
    ids=set(pd.to_numeric(expected.game_id))
    raw=pd.read_parquet(ROOT/f'raw_data/NBA{year%100:02d}.parquet')
    raw=raw.loc[pd.to_numeric(raw.game_id).isin(ids)].copy()
    old=clean.fetch_player_stats_supabase
    with tempfile.TemporaryDirectory(prefix='monthly_vpm_raw_audit_') as tmp:
        path=Path(tmp)/f'NBA{year%100:02d}.parquet';raw.to_parquet(path,index=False)
        try:
            clean.fetch_player_stats_supabase=lambda *args:None
            base=clean.build_clean_chance_base_py(str(path),year%100,f'{year-1}-{year%100:02d}',
                'MONTHLY_CUTOFF_AUDIT',include_continuation_atoms=True)
        finally:clean.fetch_player_stats_supabase=old
    base['Net_Diff']=base.First_Chance_Diff
    columns=[c for c in base if c.startswith('FC_') and c.endswith('_Diff')]
    columns+=['Net_Diff','Is_Turnover','Actual_Diff','SC_AFTER_MISS_POINTS',
              'FC_MISS_RIM','FC_MISS_MID','FC_MISS_THREE']
    fc=_finalize_df(base,columns,year%100)
    evpath=ROOT.parent.parent/'vpm/vpm/data'/('russell_sq_v2_ev_16_22.parquet' if year<=2020 else 'russell_sq_v2_ev_21_26.parquet')
    ev=pd.read_parquet(evpath);gold,stats=attach_ev_targets(fc,raw,ev,net_lane=True)
    if len(gold)!=len(expected):raise ValueError('Cutoff row count differs')
    for c in ['game_id','event_num']:
        if not np.array_equal(pd.to_numeric(gold[c]),pd.to_numeric(expected[c])):raise ValueError('Cutoff row order differs')
    gaps={}
    for a,b in [('Actual_Diff','actual_points'),('VPM_MID_RESID_NET_Diff','mid_residual'),
                ('VPM_THREE_RESID_NET_Diff','three_residual'),('SC_AFTER_MISS_POINTS','SC_AFTER_MISS_POINTS'),
                ('FC_MISS_MID','FC_MISS_MID'),('FC_MISS_THREE','FC_MISS_THREE')]:
        gaps[a]=float(np.max(abs(gold[a].to_numpy()-expected[b].to_numpy())))
    result=dict(year=year,cut=cut,rows=len(gold),games=len(ids),max_gaps=gaps,pricing=pricing,
                atoms_sha256=sha(source),kappa_sha256=sha(kappa_path),ev_sha256=sha(evpath),passed=max(gaps.values())<1e-9)
    write_json(OUT/f'audit_raw_rebuild_{cut}.json',result)
    if not result['passed']:raise ValueError(f'Raw cutoff reproduction failed: {gaps}')
    print(result,flush=True)
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--year',type=int,required=True)
    ap.add_argument('--cut',required=True);args=ap.parse_args();audit(args.year,args.cut)
