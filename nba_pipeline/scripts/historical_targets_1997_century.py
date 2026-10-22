"""Exact early-century resolver around the frozen local-calendar target adapter.

The frozen canonical helper passes 97/98/99 to a modern finalizer that resolves
them as2097/2098/2099. Its exact date assertion rejected both prior roots. This
new adapter resolves only the worker's exact early two-digit year to its proved
full year. All scoring, row algebra, source, calendar and identity guards remain.
"""
from contextlib import contextmanager
from pathlib import Path
import argparse
import json

import pandas as pd

import historical_targets_1997_dated as D
from process_rapm_blocks import common

M=D.M
OUT=M.ROOT/'results/vpm_historical_targets_1997_dated_v2'
_DATED_EARLY_BUILD=D.early_build


@contextmanager
def early_century(year):
    if year not in [1997,1998,1999]:
        raise ValueError('Only exact completed early-year workers may use century resolution')
    original=common.normalize_season_end_year
    def normalize(value):
        if str(value)==str(year%100):return year
        return original(value)
    try:
        common.normalize_season_end_year=normalize
        yield
    finally:
        common.normalize_season_end_year=original


def preflight(out=OUT):
    out=Path(out).resolve()
    if (out/'registration.json').exists():
        registered=M.read(out/'registration.json');M.verify_pins(registered['input_hashes']);return registered
    out.mkdir(parents=True,exist_ok=True)
    cases=[]
    for year in [1997,1998,1999]:
        original=common.normalize_season_end_year(year%100)
        with early_century(year):
            corrected=common.normalize_season_end_year(year%100)
            label=common.build_season_label_from_end_year(year%100)
        if corrected!=year or label!=f'{year-1}-{year%100:02d}':
            raise ValueError('Exact early-century proof failed')
        cases.append(dict(argument=year%100,original_full_year=original,
            admitted_full_year=corrected,admitted_season_label=label))
    recipe=out/'century_recipe.json'
    M.write_once(recipe,dict(status='registered_exact_early_century_resolver_before_build',
        cases=cases,scope='exact97/98/99 in corresponding1997/1998/1999 isolatedworker only; restoreinfinally',
        previous_failed_roots=[str(M.OUT.resolve()),str(D.OUT.resolve())],
        date_adapter=str(Path(D.__file__).resolve()),date_adapter_sha256=M.H.sha(D.__file__),
        scoring_row_price_identity_math_unchanged=True,
        input_hashes={str(Path(__file__).resolve()):M.H.sha(__file__),
            str(Path(common.__file__).resolve()):M.H.sha(common.__file__)},protected_2027_outcomes=0))
    original=M.source_files
    def files():return sorted(set(original())|{Path(__file__).resolve(),recipe,
        M.ROOT/'tests/test_historical_targets_1997_century.py'})
    try:
        M.source_files=files
        result=D.preflight(out)
    finally:M.source_files=original
    # The running feature exporter pins the old preflight admissions. They are
    # independent of this calendar/century I/O adaptation and must remain exact.
    equality={}
    for name in ['alias_admission.json','identity_source_packet.json']:
        prior=D.OUT/name
        if M.H.sha(out/name)!=M.H.sha(prior):
            raise ValueError('Date adapter changed feature identity/alias admission')
        equality[name]=dict(prior=str(prior.resolve()),new=str((out/name).resolve()),sha256=M.H.sha(prior))
    M.write_once(out/'crossroot_admission_equality.json',dict(status='passed_byte_identical_feature_admissions',
        files=equality,feature_reexport_required=False))
    return result


def early_build(args):
    year,phase,out=args
    with early_century(year):result=_DATED_EARLY_BUILD(args)
    data=pd.read_parquet(Path(out)/f'atoms_{year}_{phase}.parquet',
        columns=['Season','source_season_end_year','game_date'])
    if (not pd.to_numeric(data.Season).eq(year).all()
            or not pd.to_numeric(data.source_season_end_year).eq(year).all()
            or pd.to_datetime(data.game_date).isna().any()):
        raise ValueError('Rebuilt early target has wrong full-year epoch or missing dates')
    return result


def build(out=OUT,workers=1):
    original=D.early_build
    try:
        D.early_build=early_build
        return D.build(out,workers)
    finally:D.early_build=original


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
