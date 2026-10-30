"""Observed point/count atoms for date-safe monthly own-SQ VPM targets.

Actual FT outcomes replace the archived full-season shooter FT percentages.
Net SQ residuals are repriced from strictly pre-cut continuation counts, in
each season/phase. Frozen chronological shot EV and possession identities stay
unchanged. The canonical target files and original v18 remain untouched.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from build_monthly_own_skills import ROOT, sha, write_json

OUT=ROOT/'results/monthly_vpm_targets_v1'
ARCHIVE=ROOT/'results/ft_actuals_repair_20260906/old_sq'
SLOTS=[f'{side}{i}' for side in ['O','D'] for i in range(1,6)]


def reprice_before_cut(atoms,cut,kappa_games):
    """Full historical files may enter; all estimates follow the date fence."""
    d=atoms.loc[pd.to_datetime(atoms.game_date).lt(pd.Timestamp(cut))].copy()
    games=kappa_games.loc[pd.to_datetime(kappa_games.game_date).lt(pd.Timestamp(cut))].copy()
    if games.duplicated(['source_season_end_year','season_phase','game_id']).any():
        raise ValueError('Duplicate continuation game counts')
    records=[]
    for (year,phase),g in d.groupby(['source_season_end_year','season_phase'],sort=True):
        record=dict(season=int(year),phase=phase,last_game=str(pd.to_datetime(g.game_date).max().date()))
        counts=games.loc[games.source_season_end_year.eq(year)&games.season_phase.eq(phase)]
        if set(pd.to_numeric(counts.game_id))!=set(pd.to_numeric(g.game_id)):
            raise ValueError('Continuation game population differs from monthly target')
        for zone,points in [('mid',2.),('three',3.)]:
            n=float(counts[f'{zone}_misses'].sum());total=float(counts[f'{zone}_points'].sum())
            kappa=total/n if n else 0.
            d.loc[g.index,f'{zone}_residual']=g[f'{zone}_gross_residual']*(1-kappa/points)
            record[zone]=dict(misses=n,continuation_points=total,kappa=kappa)
        records.append(record)
    if not np.isfinite(d[['actual_points','mid_residual','three_residual']]).all().all():
        raise ValueError('Nonfinite monthly target')
    return d,records


def build(task):
    year,phase,out=task;out=Path(out);suffix='_PS' if phase=='PS' else ''
    dest=out/f'atoms_{year}_{phase}.parquet';receipt=dest.with_suffix('.json')
    if dest.exists() and receipt.exists():return json.loads(receipt.read_text())
    from process_rapm_blocks import clean_chance_utils as clean
    from process_rapm_blocks.common import _finalize_df
    rawpath=ROOT/f'raw_data/NBA{year%100:02d}{suffix}.parquet'
    canonical=ROOT/f'processed/RAPM{year%100:02d}{suffix}.parquet'
    sqpath=ARCHIVE/f'FIRST_CHANCE{year%100:02d}{suffix}.parquet'
    old=clean.fetch_player_stats_supabase
    try:
        # The common missing-FT path uses observed made/missed outcomes. No
        # retrospective shooter-season percentage is fetched or supplied.
        clean.fetch_player_stats_supabase=lambda *args:None
        base=clean.build_clean_chance_base_py(str(rawpath),year%100,f'{year-1}-{year%100:02d}',
                                              'MONTHLY_VPM_ATOMS',include_continuation_atoms=True)
    finally:clean.fetch_player_stats_supabase=old
    numer=['Actual_Diff','First_Chance_Diff','Second_Chance_Diff','Is_Turnover',
           'SC_AFTER_MISS_POINTS',*[f'FC_MISS_{z}' for z in ['RIM','MID','THREE']]]
    atoms=_finalize_df(base,numer,year%100)
    reference=pd.read_parquet(canonical)
    sq=pd.read_parquet(sqpath)
    for source in [reference,sq]:
        if len(source)!=len(atoms):raise ValueError('Monthly atom row count differs')
        for c in ['game_id','event_num',*SLOTS]:
            if not np.array_equal(pd.to_numeric(source[c]),pd.to_numeric(atoms[c]),equal_nan=True):
                raise ValueError(f'Monthly atom possession alignment differs: {c}')
        if not np.array_equal(pd.to_datetime(source.game_date),pd.to_datetime(atoms.game_date)):
            raise ValueError('Monthly atom dates differ')
    atoms['actual_points']=atoms.pop('Actual_Diff')
    closure=float((atoms.actual_points-atoms.First_Chance_Diff-atoms.Second_Chance_Diff).abs().max())
    if closure>1e-9:raise ValueError('Actual FC/SC parent does not close')
    if not np.allclose(atoms.actual_points,np.rint(atoms.actual_points),atol=1e-9):
        raise ValueError('Monthly actual FT parent is not integer points')
    for zone in ['mid','three']:
        atoms[f'{zone}_gross_residual']=sq[f'VPM_{zone.upper()}_RESID_Diff'].to_numpy()
    atoms['source_season_end_year']=year;atoms['season_phase']=phase
    # Preserve the original design/metadata, replacing only point targets.
    design=[c for c in reference if c not in atoms and c not in ['Off_Diff','Def_Diff','Net_Diff']]
    for c in design:atoms[c]=reference[c].to_numpy()
    out.mkdir(parents=True,exist_ok=True);atoms.to_parquet(dest,index=False)
    report=dict(season=year,phase=phase,rows=len(atoms),games=int(atoms.game_id.nunique()),
        first_game=str(pd.to_datetime(atoms.game_date).min().date()),last_game=str(pd.to_datetime(atoms.game_date).max().date()),
        actual_closure=closure,point_target='actual FG and FT on canonical possession boundaries',
        inputs={str(p):sha(p) for p in [rawpath,canonical,sqpath,Path(__file__),Path(clean.__file__)]},
        output_sha256=sha(dest),status='observed_atoms_not_monthly_ratings')
    write_json(receipt,report);return report


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,default=OUT)
    ap.add_argument('--years',default=','.join(map(str,range(2016,2027))))
    ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks=[(y,p,str(args.out)) for y in map(int,args.years.split(',')) for p in ['RS','PS']]
        for f in as_completed([pool.submit(build,t) for t in tasks]):print(json.dumps(f.result()),flush=True)


if __name__=='__main__':main()
