"""
Reprocess parquets for a given metric across multiple years.
Uses multiprocessing to parallelize across files.

Usage:
    python reprocess_metric.py TOV              # TOV years 14-25
    python reprocess_metric.py TOV 21 25        # TOV years 21-25
    python reprocess_metric.py SECOND_CHANCE    # SECOND_CHANCE years 14-25
    python reprocess_metric.py TS 23 26         # TS years 23-26
"""

import os
import sys
import time
import argparse
from multiprocessing import Pool, cpu_count

RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'raw_data')
PROC_DIR = os.path.join(os.path.dirname(__file__), '..', 'processed')

# Map metric name to (module, processor_function, output_prefix, kwargs, tuple_index)
METRIC_REGISTRY = {
    'RAPM':            ('process_rapm',            'process_rapm_py',            'RAPM',            {'o_luck': 0.0, 'd_luck': 0.0, 'missing_ft_fallback': 'actual'}, 0),
    'LA_RAPM':         ('process_rapm',            'process_rapm_py',            'LA_RAPM',         {'o_luck': 1.0, 'd_luck': 1.0, 'missing_ft_fallback': 'actual'}, 1),
    'TS':              ('process_ts',              'process_ts_py',              'TS',              {}, None),
    'TOV':             ('process_tov',             'process_tov_py',             'TOV',             {}, None),
    'REB':             ('process_reb',             'process_reb_py',             'REB',             {}, None),
    'SHOOTER_OREB':    ('process_shooter_oreb',    'process_shooter_oreb_py',    'SHOOTER_OREB',    {}, None),
    'BLOCK_RECOVERY':  ('process_block_recovery',  'process_block_recovery_py',  'BLOCK_RECOVERY',  {}, None),
    'RIM_FREQ':        ('process_rim_freq',        'process_rim_freq_py',        'RIM_FREQ',        {}, None),
    'RIM_FG_PCT':      ('process_rim_fg_pct',      'process_rim_fg_pct_py',      'RIM_FG_PCT',      {}, None),
    'ASSIST_POINTS':   ('process_assist_points',   'process_assist_points_py',   'ASSIST_POINTS',   {}, None),
    'RIM_ASSIST':      ('process_rim_assist',      'process_rim_assist_py',      'RIM_ASSIST',      {}, None),
    'DUNK':            ('process_dunk',            'process_dunk_py',            'DUNK',            {}, None),
    'DUNK_ASSIST':     ('process_dunk_assist',     'process_dunk_assist_py',     'DUNK_ASSIST',     {}, None),
    'THREE_FREQ':      ('process_three_freq',      'process_three_freq_py',      'THREE_FREQ',      {}, None),
    'THREE_FG_PCT':    ('process_three_fg_pct',    'process_three_fg_pct_py',    'THREE_FG_PCT',    {}, None),
    'PLAYTYPE_TS_MIX': ('process_playtype_ts_mix', 'process_playtype_ts_mix_py', 'PLAYTYPE_TS_MIX', {}, None),
    'PLAYTYPE_PROXY_PTS': ('process_playtype_proxy_pts', 'process_playtype_proxy_pts_py', 'PLAYTYPE_PROXY_PTS', {}, None),
    'MIDRANGE_FREQ':   ('process_midrange_freq',   'process_midrange_freq_py',   'MIDRANGE_FREQ',   {}, None),
    'MIDRANGE_FG_PCT': ('process_midrange_fg_pct', 'process_midrange_fg_pct_py', 'MIDRANGE_FG_PCT', {}, None),
    'TRANSITION_FREQ': ('process_transition_freq', 'process_transition_freq_py', 'TRANSITION_FREQ', {}, None),
    'TRANSITION_RIM':  ('process_transition_rim',  'process_transition_rim_py',  'TRANSITION_RIM',  {}, None),
    'INITIAL_EV':      ('process_initial_ev',      'process_initial_ev_py',      'INITIAL_EV',      {}, None),
    'SPECIAL_RAPM':    ('process_special_rapm',    'process_special_rapm_py',    'SPECIAL_RAPM',    {}, None),
    'RUSSELL_SHOTQUALITY': ('process_russell_shotquality', 'process_russell_shotquality_py', 'RUSSELL_SHOTQUALITY', {}, None),
    'CONTEXT_SHOTQUALITY': ('process_russell_shotquality', 'process_context_shotquality_py', 'CONTEXT_SHOTQUALITY', {}, None),
    'SQ_POSS':         ('process_sq_poss',         'process_sq_poss_py',         'SQ_POSS',         {}, None),
    'FT_PREMIUM':      ('process_ft_premium',      'process_ft_premium_py',      'FT_PREMIUM',      {}, None),
    'CONTEST':         ('process_contest',         'process_contest_py',         'CONTEST',         {}, None),
    'SECOND_CHANCE':   ('process_second_chance',   'process_second_chance_py',   'SECOND_CHANCE',   {}, None),
    'FIRST_CHANCE':    ('process_first_chance',    'process_first_chance_py',    'FIRST_CHANCE',    {}, None),
    'FIRST_CHANCE_CLEAN': ('process_first_chance_clean', 'process_first_chance_clean_py', 'FIRST_CHANCE_CLEAN', {}, None),
    'SECOND_CHANCE_CLEAN': ('process_second_chance_clean', 'process_second_chance_clean_py', 'SECOND_CHANCE_CLEAN', {}, None),
}

# Resolved at module level after parsing args
_process_func = None
_output_prefix = None
_process_kwargs = None
_tuple_index = None
_raw_dir = RAW_DIR
_processed_dir = PROC_DIR


def _init_worker(metric, raw_dir=None, processed_dir=None):
    """Initialize the processor function in each worker process."""
    global _process_func, _output_prefix, _process_kwargs, _tuple_index, _raw_dir, _processed_dir
    if raw_dir is not None:
        _raw_dir = raw_dir
    if processed_dir is not None:
        _processed_dir = processed_dir
    module_name, func_name, _output_prefix, _process_kwargs, _tuple_index = METRIC_REGISTRY[metric]
    mod = __import__(f'process_rapm_blocks.{module_name}', fromlist=[func_name])
    _process_func = getattr(mod, func_name)


def process_one(args):
    """Process a single raw parquet."""
    fname, yr, label = args
    input_path = os.path.join(_raw_dir, fname)
    out_name = fname.replace('NBA', _output_prefix)
    out_path = os.path.join(_processed_dir, out_name)

    try:
        result = _process_func(input_path, yr, label, **(_process_kwargs or {}))
        if result is None:
            return (out_name, 0, False, "processor returned None")
        # process_rapm returns a tuple (no_la, la).
        if isinstance(result, tuple):
            result = result[_tuple_index] if _tuple_index is not None else result[0]
        if result is not None:
            result.to_parquet(out_path, index=False)
            return (out_name, len(result), True, None)
        else:
            return (out_name, 0, False, "processor returned None")
    except Exception as e:
        return (out_name, 0, False, str(e))


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reprocess metric parquets across NBA raw season files."
    )
    parser.add_argument('metric', help='Metric name to reprocess.')
    parser.add_argument('start_year', nargs='?', type=int, default=2014)
    parser.add_argument('end_year', nargs='?', type=int, default=2025)
    parser.add_argument(
        '--season-types',
        default='all',
        help='rs, ps, or all. Default: all.',
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Worker count override. Default: min(job count, cpu_count - 1).',
    )
    parser.add_argument(
        '--raw-dir',
        default=RAW_DIR,
        help='Input raw parquet directory. Default: nba_pipeline/raw_data.',
    )
    parser.add_argument(
        '--processed-dir',
        default=PROC_DIR,
        help='Output processed parquet directory. Default: nba_pipeline/processed.',
    )
    return parser.parse_args()


def main():
    global _raw_dir, _processed_dir
    args = parse_args()
    _raw_dir = args.raw_dir
    _processed_dir = args.processed_dir
    os.makedirs(_processed_dir, exist_ok=True)
    metric = args.metric.upper()
    if metric not in METRIC_REGISTRY:
        print(f"Unknown metric: {metric}")
        print(f"Available: {', '.join(sorted(METRIC_REGISTRY.keys()))}")
        sys.exit(1)

    start_yr = normalize_end_year(args.start_year)
    end_yr = normalize_end_year(args.end_year)
    season_types = parse_season_types(args.season_types)

    # Build file list
    jobs = []
    for yr in range(start_yr, end_yr + 1):
        suffix = season_suffix(yr)
        if 'rs' in season_types:
            rs = f'NBA{suffix}.parquet'
            if os.path.exists(os.path.join(_raw_dir, rs)):
                jobs.append((rs, yr, f'RS {season_label(yr)}'))
        if 'ps' in season_types:
            ps = f'NBA{suffix}_PS.parquet'
            if os.path.exists(os.path.join(_raw_dir, ps)):
                jobs.append((ps, yr, f'PS {season_label(yr)}'))

    print(
        f"Reprocessing {len(jobs)} {metric} files "
        f"(years {start_yr}-{end_yr}, season_types={','.join(season_types)})"
    )
    if not jobs:
        return
    n_workers = args.workers if args.workers is not None else min(len(jobs), max(1, cpu_count() - 1))
    n_workers = max(1, min(n_workers, len(jobs)))
    print(f"Using {n_workers} parallel workers\n")

    t0 = time.time()
    with Pool(processes=n_workers, initializer=_init_worker, initargs=(metric, _raw_dir, _processed_dir)) as pool:
        results = pool.map(process_one, jobs)

    # Summary
    print("\n" + "=" * 60)
    print(f"{'File':<30} {'Rows':>10} {'Status'}")
    print("-" * 60)
    for r in results:
        name, rows, ok, err = r
        if ok:
            print(f"{name:<30} {rows:>10,} OK")
        else:
            print(f"{name:<30} {'FAILED':>10} {err}")

    elapsed = time.time() - t0
    total_rows = sum(r[1] for r in results if r[2])
    succeeded = sum(1 for r in results if r[2])
    print("-" * 60)
    print(f"{succeeded}/{len(jobs)} files, {total_rows:,} total rows in {elapsed:.1f}s")


if __name__ == '__main__':
    main()
