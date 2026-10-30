"""Replace only SQ residual atoms using a new dated EV source.

RAPM owns canonical PBP, possession mapping and target construction. This
adapter rebuilds the eligibility audit from actual-FT raw PBP, never from an
old SQ residual or full-season shooter FT conversion estimate.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    with Path(path).open('rb') as stream:
        h=hashlib.sha256()
        for part in iter(lambda:stream.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def rebuild_one(task):
    pipeline,old_root,out,ev_path,year,phase=map(str,task)
    year=int(year);pipeline=Path(pipeline);old_root=Path(old_root);out=Path(out);ev_path=Path(ev_path)
    suffix='_PS' if phase=='PS' else '';dest=out/f'atoms_{year}_{phase}.parquet';receipt=dest.with_suffix('.json')
    original=old_root/dest.name;raw_path=pipeline/f'raw_data/NBA{year%100:02d}{suffix}.parquet'
    from build_vpm_target_swap import attach_ev_targets,REQUIRED_FIRST_CHANCE_COLUMNS
    from process_rapm_blocks import clean_chance_utils as clean
    from process_rapm_blocks.common import _finalize_df
    from build_monthly_vpm_targets import SLOTS
    source_hashes={str(p):sha(p) for p in [original,raw_path,ev_path,Path(__file__),Path(clean.__file__)]}
    if dest.exists() and receipt.exists():
        meta=json.loads(receipt.read_text())
        if meta['inputs']!=source_hashes or meta['output_sha256']!=sha(dest):raise ValueError('Clean target source/artifact changed')
        return meta
    atoms=pd.read_parquet(original)
    # Rebuild actual-FT child targets only to preserve canonical assignment and
    # eligibility checks; no new full-season shooter percentage is fetched.
    fetch=clean.fetch_player_stats_supabase
    try:
        clean.fetch_player_stats_supabase=lambda *args:None
        base=clean.build_clean_chance_base_py(str(raw_path),year%100,f'{year-1}-{year%100:02d}',
                                               'CLEAN_SQ_ATOMS',include_continuation_atoms=True)
    finally:clean.fetch_player_stats_supabase=fetch
    numer=sorted(REQUIRED_FIRST_CHANCE_COLUMNS-{'game_id','event_num'})+['Actual_Diff','First_Chance_Diff','Second_Chance_Diff']
    fc=_finalize_df(base,numer,year%100)
    if len(fc)!=len(atoms):raise ValueError('Rebuilt actual FT row count differs')
    for column in ['game_id','event_num',*SLOTS]:
        np.testing.assert_array_equal(pd.to_numeric(fc[column]),pd.to_numeric(atoms[column]))
    np.testing.assert_allclose(fc.Actual_Diff,atoms.actual_points,atol=1e-12,rtol=0)
    np.testing.assert_allclose(fc.First_Chance_Diff,atoms.First_Chance_Diff,atol=1e-12,rtol=0)
    raw=pd.read_parquet(raw_path,columns=['game_id','event_num','event_type','event_action_type','period',
        'time_quarter','minute_remaining_quarter','seconds_remaining_quarter','home_description','visitor_description']).drop_duplicates()
    # Adapter's canonical chronological assignment uses minutes+seconds fields.
    # Normalize those from MM:SS rather than mixing historical encodings.
    clock=raw.time_quarter.astype('string').str.extract(r'^(\d+):(\d+(?:\.\d+)?)$')
    raw['minute_remaining_quarter']=0
    raw['seconds_remaining_quarter']=pd.to_numeric(clock[0])*60+pd.to_numeric(clock[1])
    ev=pd.read_parquet(ev_path,filters=[('season_end_year','==',year)])
    fitted,stats=attach_ev_targets(fc,raw,ev,net_lane=False)
    rebuilt=atoms.copy()
    rebuilt['mid_gross_residual']=fitted.VPM_MID_RESID_Diff.to_numpy()
    rebuilt['three_gross_residual']=fitted.VPM_THREE_RESID_Diff.to_numpy()
    unchanged=[c for c in atoms if c not in ['mid_gross_residual','three_gross_residual']]
    pd.testing.assert_frame_equal(atoms[unchanged],rebuilt[unchanged])
    closure=float(abs(rebuilt.actual_points-rebuilt.First_Chance_Diff-rebuilt.Second_Chance_Diff).max())
    if closure>1e-9:raise ValueError('Actual FT parent closure changed')
    out.mkdir(parents=True,exist_ok=True);rebuilt.to_parquet(dest,index=False)
    meta=dict(season=year,phase=phase,rows=len(atoms),point_target='actual FG and FT unchanged',
        actual_parent_max_change=0.,other_atom_fields_max_change=0.,closure=closure,assignment=stats,
        eligibility='fresh raw actual-FT child identity; no old SQ or shooter season FT lookup',
        inputs=source_hashes,output_sha256=sha(dest))
    receipt.write_text(json.dumps(meta,indent=2)+'\n')
    print(f'Clean SQ atoms {year} {phase}: {len(atoms):,} rows, {stats["ev_attached_rows"]:,} swaps',flush=True)
    return meta


def rebuild_all(pipeline,old_root,out,ev_path,workers=4):
    tasks=[(pipeline,old_root,out,ev_path,year,phase) for year in range(2016,2027) for phase in ['RS','PS']]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        records=[f.result() for f in as_completed([pool.submit(rebuild_one,t) for t in tasks])]
    manifest=dict(rows=sum(r['rows'] for r in records),actual_parent_max_change=0.,files=records,
                  ev_path=str(ev_path),ev_sha256=sha(ev_path))
    (Path(out)/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest
