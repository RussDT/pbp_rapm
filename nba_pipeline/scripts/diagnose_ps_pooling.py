"""
Diagnose the effect of RS/PS pooling on per-player RAPM results.

Compares a BASELINE player results CSV vs a CANDIDATE player results CSV
(both in the standard rapm.py results schema: player_id, player_name, off,
def, net_rapm, possessions, off_poss, def_poss). Measures per-player deltas
and relates them to playoff floor participation over a season window, so we
can see whether a change (e.g. --ps-prior) moves playoff participants more
than non-participants.

Usage:
    python diagnose_ps_pooling.py \
        --baseline ../results/rapm_22_26_all_pure_rb_se_a2000_4000_results.csv \
        --candidate ../results/rapm_22_26_all_rb_se_a2000_4000_results.csv \
        --start-year 22 --end-year 26
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PIPELINE_ROOT / 'processed'
DEFAULT_OUTPUT_DIR = PIPELINE_ROOT / 'results' / 'diagnostics'

LINEUP_COLS = ['O1', 'O2', 'O3', 'O4', 'O5', 'D1', 'D2', 'D3', 'D4', 'D5']
MIN_POSS_FOR_MOVERS = 4000


def parse_args():
    parser = argparse.ArgumentParser(
        description='Quantify per-player deltas between two RAPM results CSVs '
                    'and relate them to playoff floor participation.')
    parser.add_argument('--baseline', required=True, type=Path,
                        help='Baseline player results CSV (standard rapm.py schema)')
    parser.add_argument('--candidate', required=True, type=Path,
                        help='Candidate player results CSV (standard rapm.py schema)')
    parser.add_argument('--start-year', required=True, type=int,
                        help='Two-digit season end year for the start of the participation window')
    parser.add_argument('--end-year', required=True, type=int,
                        help='Two-digit season end year for the end of the participation window')
    parser.add_argument('--labels', default='BASELINE,CANDIDATE',
                        help='Comma-separated display labels, e.g. "PURE,PSPRIOR"')
    parser.add_argument('--output', type=Path, default=None,
                        help='Output CSV path (default: results/diagnostics/'
                             'ps_pooling_delta_{baseline}_vs_{candidate}.csv)')
    return parser.parse_args()


def count_floor_appearances(parquet_path):
    """Count per-player floor appearances: one per row in which the player id
    occurs in any lineup slot. Returns a Series indexed by string player_id."""
    df = pd.read_parquet(parquet_path, columns=LINEUP_COLS)
    ids = df[LINEUP_COLS].to_numpy().ravel()
    ids = ids[~pd.isna(ids)].astype(np.int64)
    ids = ids[ids > 0]  # partial-lineup unknown slots are 0; ignore
    counts = pd.Series(ids).value_counts()
    counts.index = counts.index.astype(str)  # match results-CSV player_id keys
    return counts


def expand_two_digit_year_window(start_year, end_year):
    """Expand a two-digit season-end window, allowing 97-00 century rollover
    (mirrors rapm.py's canonical expansion)."""
    def normalize(y):
        y = int(y)
        if y < 100:
            return 2000 + y if y <= 50 else 1900 + y
        return y

    start_full = normalize(start_year)
    end_full = normalize(end_year)
    if end_full < start_full:
        end_full += 100
    return [full_year % 100 for full_year in range(start_full, end_full + 1)]


def build_participation(start_year, end_year):
    """Sum RS and PS floor appearances per player over the season window."""
    rs_counts = pd.Series(dtype=np.int64)
    ps_counts = pd.Series(dtype=np.int64)
    for yy in expand_two_digit_year_window(start_year, end_year):
        rs_path = PROCESSED_DIR / f'RAPM{yy:02d}.parquet'
        ps_path = PROCESSED_DIR / f'RAPM{yy:02d}_PS.parquet'
        if rs_path.exists():
            season_rs = count_floor_appearances(rs_path)
            rs_counts = rs_counts.add(season_rs, fill_value=0)
            logger.info('Season %02d RS: %s rows -> %s players',
                        yy, f'{int(season_rs.sum()):,}', len(season_rs))
        else:
            logger.warning('Missing RS parquet, skipping: %s', rs_path)
        if ps_path.exists():
            season_ps = count_floor_appearances(ps_path)
            ps_counts = ps_counts.add(season_ps, fill_value=0)
            logger.info('Season %02d PS: %s rows -> %s players',
                        yy, f'{int(season_ps.sum()):,}', len(season_ps))
        else:
            logger.warning('Missing PS parquet, skipping: %s', ps_path)

    if rs_counts.empty and ps_counts.empty:
        raise SystemExit(
            f'No RS or PS parquets found for window {start_year:02d}-{end_year:02d}; '
            'a participation-free delta report would be vacuous.'
        )

    part = pd.DataFrame({
        'rs_poss_window': rs_counts,
        'ps_poss_window': ps_counts,
    }).fillna(0).astype(np.int64)
    part.index.name = 'player_id'
    total = part['rs_poss_window'] + part['ps_poss_window']
    part['ps_share'] = np.where(total > 0, part['ps_poss_window'] / total, np.nan)
    return part.reset_index()


def load_results(path):
    df = pd.read_csv(path)
    # Normalize player_id to the string of the int so joins are stable even if
    # a CSV round-trip produced float-formatted ids.
    df['player_id'] = df['player_id'].astype(float).astype(np.int64).astype(str)
    return df


def fmt(value, width=8, decimals=3):
    if pd.isna(value):
        return 'nan'.rjust(width)
    return f'{value:+.{decimals}f}'.rjust(width)


def print_group_stats(label, group, weight_col):
    n = len(group)
    if n == 0:
        print(f'  {label:<28} n=0')
        return
    w = group[weight_col].to_numpy(dtype=float)
    weighted = (np.average(group['delta_net'], weights=w)
                if w.sum() > 0 else np.nan)
    print(f'  {label:<28} n={n:<5} '
          f'delta_net mean={fmt(group["delta_net"].mean())} med={fmt(group["delta_net"].median())} '
          f'wmean={fmt(weighted)} | '
          f'delta_off mean={fmt(group["delta_off"].mean())} med={fmt(group["delta_off"].median())} | '
          f'delta_def mean={fmt(group["delta_def"].mean())} med={fmt(group["delta_def"].median())}')


def print_report(merged, base_label, cand_label):
    print()
    print('=' * 100)
    print(f'PS POOLING DELTA REPORT: {cand_label} minus {base_label}')
    print('=' * 100)

    n = len(merged)
    corr = merged['base_net'].corr(merged['cand_net'])
    abs_delta = merged['delta_net'].abs()
    print(f'\nOverall: {n} joined players | Pearson r(net) = {corr:.5f} | '
          f'mean |delta_net| = {abs_delta.mean():.4f} | median |delta_net| = {abs_delta.median():.4f}')

    participants = merged[merged['ps_poss_window'] > 0]
    non_participants = merged[merged['ps_poss_window'] == 0]
    print('\nPlayoff participation split (window floor appearances):')
    print_group_stats('Playoff participants', participants, 'window_poss')
    print_group_stats('Non-participants', non_participants, 'window_poss')

    print('\nPlayoff participants by ps_share quartile (key table -- do deltas scale with playoff exposure?):')
    if len(participants) >= 4:
        quartiles, bins = pd.qcut(participants['ps_share'], 4,
                                  labels=False, retbins=True, duplicates='drop')
        boundary_str = ', '.join(f'{b:.4f}' for b in bins)
        print(f'  ps_share quartile boundaries: [{boundary_str}]')
        for q in sorted(quartiles.dropna().unique()):
            grp = participants[quartiles == q]
            lo, hi = bins[int(q)], bins[int(q) + 1]
            print_group_stats(f'Q{int(q) + 1} share {lo:.3f}-{hi:.3f}', grp, 'window_poss')
    else:
        print('  Too few playoff participants for quartile split.')

    movers = merged[merged['window_poss'] >= MIN_POSS_FOR_MOVERS]
    print(f'\nTop 15 by delta_net (window_poss >= {MIN_POSS_FOR_MOVERS}, n eligible = {len(movers)}):')
    cols_header = (f'  {"player":<28} {base_label + " net":>12} {cand_label + " net":>12} '
                   f'{"delta":>8} {"ps_share":>9}')
    for title, table in [
        ('top', movers.nlargest(15, 'delta_net')),
        ('bottom', movers.nsmallest(15, 'delta_net').sort_values('delta_net')),
    ]:
        if title == 'bottom':
            print(f'\nBottom 15 by delta_net (window_poss >= {MIN_POSS_FOR_MOVERS}):')
        print(cols_header)
        for _, row in table.iterrows():
            share = f'{row["ps_share"]:.3f}' if pd.notna(row['ps_share']) else 'nan'
            print(f'  {str(row["player_name"]):<28} {row["base_net"]:>12.3f} {row["cand_net"]:>12.3f} '
                  f'{row["delta_net"]:>+8.3f} {share:>9}')
    print()


def main():
    args = parse_args()
    labels = [s.strip() for s in args.labels.split(',')]
    if len(labels) != 2 or not all(labels):
        raise SystemExit('--labels must be two comma-separated names, e.g. "BASE,CAND"')
    base_label, cand_label = labels

    logger.info('Baseline (%s): %s', base_label, args.baseline)
    logger.info('Candidate (%s): %s', cand_label, args.candidate)
    base = load_results(args.baseline)
    cand = load_results(args.candidate)

    merged = base.merge(cand, on='player_id', how='inner', suffixes=('_base', '_cand'))
    only_base = len(base) - len(merged)
    only_cand = len(cand) - len(merged)
    logger.info('Joined %s players (inner); %s only in baseline, %s only in candidate',
                len(merged), only_base, only_cand)

    merged = merged.rename(columns={
        'player_name_base': 'player_name',
        'off_base': 'base_off', 'def_base': 'base_def', 'net_rapm_base': 'base_net',
        'off_cand': 'cand_off', 'def_cand': 'cand_def', 'net_rapm_cand': 'cand_net',
    })
    merged['player_name'] = merged['player_name'].fillna(merged['player_name_cand'])
    merged['delta_net'] = merged['cand_net'] - merged['base_net']
    merged['delta_off'] = merged['cand_off'] - merged['base_off']
    merged['delta_def'] = merged['cand_def'] - merged['base_def']

    logger.info('Building playoff participation window %02d-%02d from %s',
                args.start_year, args.end_year, PROCESSED_DIR)
    participation = build_participation(args.start_year, args.end_year)
    merged = merged.merge(participation, on='player_id', how='left')
    merged[['rs_poss_window', 'ps_poss_window']] = (
        merged[['rs_poss_window', 'ps_poss_window']].fillna(0).astype(np.int64))
    merged['window_poss'] = merged['rs_poss_window'] + merged['ps_poss_window']
    no_window = int((merged['window_poss'] == 0).sum())
    if no_window:
        logger.warning('%s joined players have zero window floor appearances '
                       '(results files may cover a different window)', no_window)

    print_report(merged, base_label, cand_label)

    output = args.output
    if output is None:
        output = (DEFAULT_OUTPUT_DIR /
                  f'ps_pooling_delta_{args.baseline.stem}_vs_{args.candidate.stem}.csv')
    output.parent.mkdir(parents=True, exist_ok=True)
    out_cols = ['player_id', 'player_name',
                'base_off', 'base_def', 'base_net',
                'cand_off', 'cand_def', 'cand_net',
                'delta_off', 'delta_def', 'delta_net',
                'rs_poss_window', 'ps_poss_window', 'ps_share']
    merged[out_cols].sort_values('delta_net', ascending=False).to_csv(output, index=False)
    logger.info('Wrote per-player delta table: %s (%s rows)', output, len(merged))


if __name__ == '__main__':
    main()
