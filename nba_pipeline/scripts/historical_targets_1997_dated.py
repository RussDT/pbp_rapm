"""Local-calendar adapter for the frozen early actual-point target extension.

The first frozen extension failed its unchanged exact-date guard: some raw
event rows lack dates, and the canonical finalizer's external schedule does
not recover every early row. Every affected game has one unique observed raw
date, independently equal to its canonical date. This adapter injects that
pinned per-game calendar into the unchanged finalizer, without changing row
boundaries, scoring, identity, prices, counts or any frozen source file.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json

import pandas as pd

import historical_targets_1997 as M

OUT = M.ROOT/'results/vpm_historical_targets_1997_dated_v1'


def prove_calendar(raw, canonical, year, phase):
    raw = raw[['game_id','game_date']].copy()
    canonical = canonical[['game_id','game_date']].copy()
    for frame in [raw, canonical]:
        frame['game_id'] = pd.to_numeric(frame.game_id,errors='raise').astype('int64')
        frame['game_date'] = pd.to_datetime(frame.game_date,errors='raise')
    if canonical.game_date.isna().any():
        raise ValueError('Canonical target date is missing')
    a=raw.groupby('game_id').game_date.agg(['min','max'])
    b=canonical.groupby('game_id').game_date.agg(['min','max'])
    if (set(a.index)!=set(b.index) or a['min'].isna().any()
            or not a['min'].equals(a['max']) or not b['min'].equals(b['max'])
            or not a['min'].equals(b['min'])):
        raise ValueError('Raw/canonical unique per-game dates are not independently identical')
    dates=[dict(GAME_ID=int(game),season=f'{year-1}-{year%100:02d}',game_date=str(date.date()))
           for game,date in a['min'].items()]
    return dict(year=year,phase=phase,games=len(a),raw_row_dates_missing=int(raw.game_date.isna().sum()),
        game_dates_missing=0,game_date_conflicts=0,raw_canonical_game_date_gaps=0,calendar=dates)


def preflight(out=OUT):
    out=Path(out).resolve()
    if (out/'registration.json').exists():
        registered=M.read(out/'registration.json');M.verify_pins(registered['input_hashes']);return registered
    out.mkdir(parents=True,exist_ok=True)
    records=[];pins={str(Path(__file__).resolve()):M.H.sha(__file__)}
    for year in range(1997,2000):
        for phase,suffix in [('RS',''),('PS','_PS')]:
            raw=M.ROOT/f'raw_data/NBA{year%100:02d}{suffix}.parquet'
            canonical=M.ROOT/f'processed/RAPM{year%100:02d}{suffix}.parquet'
            records.append(prove_calendar(pd.read_parquet(raw,columns=['game_id','game_date']),
                pd.read_parquet(canonical,columns=['game_id','game_date']),year,phase))
            pins.update({str(p.resolve()):M.H.sha(p) for p in [raw,canonical]})
    datefile=out/'dated_calendar_admission.json'
    M.write_once(datefile,dict(status='passed_exact_raw_canonical_early_game_calendar',files=records,
        policy='unique nonmissing raw per-game date independently equal to every canonical row; row-level raw NaT filled only by this observed game calendar',
        external_schedule_loaded=False,target_math_changed=False,input_hashes=pins))
    recipe=out/'dated_recipe.json'
    M.write_once(recipe,dict(status='registered_local_calendar_adapter_before_build',
        original_failed_root=str(M.OUT.resolve()),source_and_target_rows_unchanged=True,
        original_target_extension_code=str(Path(M.__file__).resolve()),
        original_target_extension_code_sha256=M.H.sha(M.__file__),
        dated_calendar_admission=str(datefile),dated_calendar_sha256=M.H.sha(datefile),
        external_game_date_lookup_replaced=True,scope='isolated worker function binding only',
        input_hashes=pins,protected_2027_outcomes=0))
    original=M.source_files
    def files():
        return sorted(set(original())|{Path(__file__).resolve(),
            M.ROOT/'tests/test_historical_targets_1997_dated.py',datefile,recipe})
    try:
        M.source_files=files
        return M.preflight(out)
    finally:
        M.source_files=original


def early_build(args):
    year,phase,out=args;out=Path(out)
    packet=M.read(out/'dated_calendar_admission.json')
    registered=M.read(out/'registration.json')
    if M.H.sha(out/'dated_calendar_admission.json')!=registered['input_hashes'][str((out/'dated_calendar_admission.json').resolve())]:
        raise ValueError('Frozen observed game calendar changed')
    rec=[r for r in packet['files'] if r['year']==year and r['phase']==phase]
    if len(rec)!=1:raise ValueError('Exact early year/phase calendar required')
    calendar=pd.DataFrame(rec[0]['calendar'])
    from process_rapm_blocks import common
    original=common.load_game_dates
    try:
        # Pure I/O binding; the finalizer's merge and all exact boundary/date
        # assertions remain unchanged. No network schedule enters this worker.
        common.load_game_dates=lambda *args,**kwargs:calendar.copy()
        return M.early_build((year,phase,out))
    finally:
        common.load_game_dates=original


def build(out=OUT,workers=1):
    out=Path(out).resolve();registered=M.read(out/'registration.json');M.verify_pins(registered['input_hashes'])
    for year in range(2000,2027):
        for phase in ['RS','PS']:
            rp=M.OLD/f'historical_{year}_{phase}.json';meta=M.read(rp)
            for name,digest in meta['artifacts'].items():
                if M.H.sha(M.OLD/name)!=digest:raise ValueError('Old target artifact changed')
            for original in [rp,*[M.OLD/name for name in meta['artifacts']]]:
                destination=out/original.name
                if destination.exists():
                    if M.H.sha(destination)!=M.H.sha(original):raise ValueError('Reused target differs')
                else:destination.symlink_to(original.resolve())
    tasks=[(y,p,str(out)) for y in range(1997,2000) for p in ['RS','PS']]
    if workers==1:results=[early_build(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:results=list(pool.map(early_build,tasks))
    M.verify_pins(registered['input_hashes'])
    print(json.dumps(dict(status='six_local_calendar_actual_measurement_pairs_built',files=len(results))),flush=True)


audit=M.audit
load_before_cut=M.load_before_cut
target_matrix=M.target_matrix
validate_population=M.validate_population
SLOTS=M.SLOTS


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,default=OUT)
    ap.add_argument('--years',required=True);ap.add_argument('--stage',choices=['preflight','build','audit','all'],default='all')
    ap.add_argument('--workers',type=int,default=1);args=ap.parse_args()
    if tuple(map(int,args.years.split(',')))!=M.YEARS or not 1<=args.workers<=2:
        raise ValueError('Exact completed1997–2026 registry and1–2workers required')
    if args.stage in ['preflight','all']:preflight(args.root)
    if args.stage in ['build','all']:build(args.root,args.workers)
    if args.stage in ['audit','all']:audit(args.root)


if __name__=='__main__':main()
