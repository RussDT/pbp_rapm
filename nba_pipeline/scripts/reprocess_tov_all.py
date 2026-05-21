"""
Reprocess TOV parquets for years 14-25 (RS + PS) with the new Is_BadPass_TOV column.
Uses multiprocessing to parallelize across files.

Usage:
    python reprocess_tov_all.py          # All years 14-25
    python reprocess_tov_all.py 21 25    # Years 21-25 only
"""

import os
import sys
import time
from multiprocessing import Pool, cpu_count
from process_rapm_blocks.process_tov import process_tov_py

RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'raw_data')
PROC_DIR = os.path.join(os.path.dirname(__file__), '..', 'processed')


def process_one(args):
    """Process a single raw parquet into a TOV parquet."""
    fname, yr, label = args
    input_path = os.path.join(RAW_DIR, fname)
    out_name = fname.replace('NBA', 'TOV')
    out_path = os.path.join(PROC_DIR, out_name)

    try:
        result = process_tov_py(input_path, yr, label)
        if result is not None:
            result.to_parquet(out_path, index=False)
            bp = result['Is_BadPass_TOV'].sum()
            tot = result['Is_Turnover'].sum()
            return (out_name, len(result), bp, tot - bp, True)
        else:
            return (out_name, 0, 0, 0, False)
    except Exception as e:
        return (out_name, 0, 0, 0, False, str(e))


def main():
    start_yr = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    end_yr = int(sys.argv[2]) if len(sys.argv) > 2 else 25

    # Build file list
    jobs = []
    for yr in range(start_yr, end_yr + 1):
        rs = f'NBA{yr}.parquet'
        ps = f'NBA{yr}_PS.parquet'
        if os.path.exists(os.path.join(RAW_DIR, rs)):
            jobs.append((rs, yr, f'RS 20{yr-1:02d}-{yr:02d}'))
        if os.path.exists(os.path.join(RAW_DIR, ps)):
            jobs.append((ps, yr, f'PS 20{yr-1:02d}-{yr:02d}'))

    print(f"Reprocessing {len(jobs)} TOV files (years {start_yr}-{end_yr})")
    n_workers = min(len(jobs), max(1, cpu_count() - 1))
    print(f"Using {n_workers} parallel workers\n")

    t0 = time.time()
    with Pool(processes=n_workers) as pool:
        results = pool.map(process_one, jobs)

    # Summary
    print("\n" + "=" * 70)
    print(f"{'File':<25} {'Rows':>8} {'BadPass':>8} {'Scoring':>8} {'BP%':>6} {'Status'}")
    print("-" * 70)
    for r in results:
        name, rows, bp, sc, ok = r[:5]
        if ok:
            pct = f"{bp/(bp+sc)*100:.1f}%" if (bp + sc) > 0 else "N/A"
            print(f"{name:<25} {rows:>8,} {bp:>8,} {sc:>8,} {pct:>6} OK")
        else:
            err = r[5] if len(r) > 5 else "unknown error"
            print(f"{name:<25} {'FAILED':>8} {err}")

    elapsed = time.time() - t0
    total_rows = sum(r[1] for r in results if r[4])
    print("-" * 70)
    print(f"Total: {total_rows:,} rows in {elapsed:.1f}s")


if __name__ == '__main__':
    main()
