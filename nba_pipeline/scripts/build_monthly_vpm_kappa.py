"""Capture canonical continuation sufficient statistics before EOP filtering."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_monthly_own_skills import ROOT,sha,write_json
from build_monthly_vpm_targets import OUT as ATOMS
from process_rapm_blocks import clean_chance_utils as clean

OUT=ROOT/'results/monthly_vpm_kappa_v1'


def build(task):
    year,phase=task;suffix='_PS' if phase=='PS' else ''
    dest=OUT/f'games_{year}_{phase}.parquet';receipt=dest.with_suffix('.json')
    if dest.exists() and receipt.exists():return json.loads(receipt.read_text())
    raw=ROOT/f'raw_data/NBA{year%100:02d}{suffix}.parquet';counts=[];old=clean.fetch_player_stats_supabase
    try:
        clean.fetch_player_stats_supabase=lambda *args:None
        clean.build_clean_chance_base_py(str(raw),year%100,f'{year-1}-{year%100:02d}',
            'MONTHLY_KAPPA_COUNTS',continuation_game_counts=counts)
    finally:clean.fetch_player_stats_supabase=old
    d=pd.DataFrame(counts);d.game_id=pd.to_numeric(d.game_id).astype(np.int64)
    atom_path=ATOMS/f'atoms_{year}_{phase}.parquet'
    dates=pd.read_parquet(atom_path,columns=['game_id','game_date']).drop_duplicates()
    dates.game_id=pd.to_numeric(dates.game_id).astype(np.int64)
    d=dates.merge(d,on='game_id',how='left',validate='one_to_one')
    if not np.isfinite(d[[f'{z}_{c}' for z in ['rim','mid','three'] for c in ['misses','points']]]).all().all():
        raise ValueError('Missing canonical continuation games')
    d['source_season_end_year']=year;d['season_phase']=phase
    OUT.mkdir(parents=True,exist_ok=True);d.to_parquet(dest,index=False)
    report=dict(year=year,phase=phase,games=len(d),population='raw first-chance misses before EOP/lineup filtering',
        actual_FT=True,inputs={str(p):sha(p) for p in [raw,atom_path,Path(__file__),Path(clean.__file__)]},
        output_sha256=sha(dest))
    write_json(receipt,report);return report


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--years',default=','.join(map(str,range(2016,2027))))
    ap.add_argument('--workers',type=int,default=6);args=ap.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(build,(y,p)) for y in map(int,args.years.split(',')) for p in ['RS','PS']]):
            print(json.dumps(f.result()),flush=True)
