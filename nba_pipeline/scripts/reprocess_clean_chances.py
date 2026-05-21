"""
Reprocess clean first-/second-chance parquet families together.

This builds the shared clean-chance base once per raw NBA file, then writes:
- FIRST_CHANCE
- FIRST_CHANCE_CLEAN
- SECOND_CHANCE_CLEAN
"""

import argparse
import os
import sys
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

from process_rapm_blocks.clean_chance_utils import build_clean_chance_base_py
from process_rapm_blocks.common import _finalize_df


RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'raw_data')
PROC_DIR = os.path.join(os.path.dirname(__file__), '..', 'processed')


def normalize_end_year(year):
    year = int(year)
    if year < 100:
        return 1900 + year if year >= 97 else 2000 + year
    return year


def season_suffix(year):
    return f"{normalize_end_year(year) % 100:02d}"


def season_label(year):
    end_year = normalize_end_year(year)
    return f"{end_year - 1}-{end_year % 100:02d}"


def parse_season_types(value):
    lowered = value.lower()
    if lowered == 'all':
        return ['rs', 'ps']
    out = [part.strip().lower() for part in lowered.split(',') if part.strip()]
    bad = sorted(set(out) - {'rs', 'ps'})
    if bad:
        raise ValueError(f"Unsupported season types: {bad}. Use rs, ps, or all.")
    return out


def first_chance_output(nba_filt, year):
    df = nba_filt.copy()
    df['Net_Diff'] = df['First_Chance_Diff']

    numerator_cols = ['Net_Diff', 'Is_Turnover', 'Is_BadPass_TOV']
    for col in [
        'FC_SQ_Diff',
        'FC_MAKE_Diff',
        'FC_FT_Diff',
        'FC_FT_FREQ_Diff',
        'FC_FT_SEVERITY_Diff',
        'FC_EFG_Diff',
        'FC_EFG_BASELINE_Diff',
        'FC_EFG_VALUE_Diff',
        'FC_RIM_EFG_AA_Diff',
        'FC_MID_EFG_AA_Diff',
        'FC_THREE_EFG_AA_Diff',
        'FC_RIM_EFG_FREQ_Diff',
        'FC_RIM_EFG_FG_Diff',
        'FC_MID_EFG_FREQ_Diff',
        'FC_MID_EFG_FG_Diff',
        'FC_THREE_EFG_FREQ_Diff',
        'FC_THREE_EFG_FG_Diff',
    ]:
        if col in df.columns:
            numerator_cols.append(col)
    return _finalize_df(df, numerator_cols, year)


def first_chance_clean_output(nba_filt, year):
    df = nba_filt.copy()
    first_chance = pd.to_numeric(df['First_Chance_Diff'], errors='coerce').fillna(0.0)
    is_turnover = pd.to_numeric(df['Is_Turnover'], errors='coerce').fillna(0).astype(int)

    df['FC_Raw_Diff'] = first_chance
    df['Net_Diff'] = np.where(is_turnover == 1, 0.0, first_chance)
    df['Off_Diff'] = df['Net_Diff']
    df['Def_Diff'] = -df['Off_Diff']

    return _finalize_df(
        df,
        ['Net_Diff', 'Off_Diff', 'Def_Diff', 'FC_Raw_Diff', 'Is_Turnover', 'Is_BadPass_TOV'],
        year,
    )


def second_chance_clean_output(nba_filt, year):
    df = nba_filt.copy()
    df['Net_Diff'] = df['Second_Chance_Diff']
    df['Off_Diff'] = df['Second_Chance_Diff']
    df['Def_Diff'] = -df['Second_Chance_Diff']
    return _finalize_df(df, ['Net_Diff', 'Off_Diff', 'Def_Diff'], year)


def process_one(args):
    fname, yr, label = args
    input_path = os.path.join(RAW_DIR, fname)
    t0 = time.time()
    try:
        base = build_clean_chance_base_py(input_path, yr, label, "CLEAN_CHANCES")
        if base is None or base.empty:
            return (fname, 0, False, "processor returned no rows", time.time() - t0)

        outputs = {
            fname.replace('NBA', 'FIRST_CHANCE', 1): first_chance_output(base, yr),
            fname.replace('NBA', 'FIRST_CHANCE_CLEAN', 1): first_chance_clean_output(base, yr),
            fname.replace('NBA', 'SECOND_CHANCE_CLEAN', 1): second_chance_clean_output(base, yr),
        }
        for out_name, out_df in outputs.items():
            out_df.to_parquet(os.path.join(PROC_DIR, out_name), index=False)
        return (fname, len(base), True, None, time.time() - t0)
    except Exception as exc:
        return (fname, 0, False, str(exc), time.time() - t0)


def parse_args():
    parser = argparse.ArgumentParser(description="Reprocess clean chance parquet families together.")
    parser.add_argument('start_year', nargs='?', type=int, default=1997)
    parser.add_argument('end_year', nargs='?', type=int, default=2026)
    parser.add_argument('--season-types', default='all', help='rs, ps, or all. Default: all.')
    parser.add_argument('--workers', type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    start_yr = normalize_end_year(args.start_year)
    end_yr = normalize_end_year(args.end_year)
    season_types = parse_season_types(args.season_types)

    jobs = []
    for yr in range(start_yr, end_yr + 1):
        suffix = season_suffix(yr)
        if 'rs' in season_types:
            rs = f'NBA{suffix}.parquet'
            if os.path.exists(os.path.join(RAW_DIR, rs)):
                jobs.append((rs, yr, f'RS {season_label(yr)}'))
        if 'ps' in season_types:
            ps = f'NBA{suffix}_PS.parquet'
            if os.path.exists(os.path.join(RAW_DIR, ps)):
                jobs.append((ps, yr, f'PS {season_label(yr)}'))

    print(
        f"Reprocessing {len(jobs)} raw files into FIRST_CHANCE, "
        "FIRST_CHANCE_CLEAN, SECOND_CHANCE_CLEAN "
        f"(years {start_yr}-{end_yr}, season_types={','.join(season_types)})"
    )
    if not jobs:
        return
    n_workers = args.workers if args.workers is not None else min(len(jobs), max(1, cpu_count() - 1))
    n_workers = max(1, min(n_workers, len(jobs)))
    print(f"Using {n_workers} parallel workers\n")

    t0 = time.time()
    succeeded = 0
    total_rows = 0
    failures = []
    with Pool(processes=n_workers) as pool:
        for fname, rows, ok, err, elapsed in pool.imap_unordered(process_one, jobs):
            if ok:
                succeeded += 1
                total_rows += rows
                print(f"{fname:<18} {rows:>10,} rows OK {elapsed:>7.1f}s", flush=True)
            else:
                failures.append((fname, err))
                print(f"{fname:<18} FAILED {err} ({elapsed:.1f}s)", flush=True)

    print("-" * 70)
    print(f"{succeeded}/{len(jobs)} files, {total_rows:,} rows in {time.time() - t0:.1f}s")
    if failures:
        for fname, err in failures:
            print(f"FAILED {fname}: {err}")
        sys.exit(1)


if __name__ == '__main__':
    main()
