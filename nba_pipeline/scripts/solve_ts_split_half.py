"""Solve TS RAPM separately on each calendar half of a single season.

Step 1 of the predeclared gate in `docs/SQ_DEFENDER_DTS_PRIOR.md`. The compact
outcome spec (`rim_suppression` + `pts_allowed`) was selected *after* reading the
univariate correlation table on the only available forward fold, which makes it a
best-of-eleven result rather than a pre-registered one. With no third SQ season on
this API key, the cheapest way to convert post-hoc selection into
selection-then-confirmation is to split 2024-25 in half:

    features from H1 2024-25  ->  dTS solved on H2 2024-25     (select here)
    features from    2024-25  ->  dTS solved on    2025-26     (confirm here)

This script produces the H2 target. It deliberately reuses rapm.py's own loading,
transform, design-matrix, and solver functions rather than reimplementing them, so
the half-season target carries identical conventions to the full-season incumbent
(`results/ts_25_all_results.csv`) and the two are directly comparable.

Halves are split on the median game date across the loaded RS+PS sample, so each
half holds about the same number of possessions rather than the same number of
days. Playoff games all fall in H2 by construction.

Sign convention is left exactly as rapm.py emits it: the ``def`` column is
negative-is-better. Downstream consumers flip it.
"""

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rapm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / 'processed'
RES = ROOT / 'results'


def load_season(year: int) -> pd.DataFrame:
    """Load RS + PS TS parquets through rapm.py's own preparation path."""
    frames = []
    for suf in ('', '_PS'):
        p = PROC / f'TS{year}{suf}.parquet'
        if not p.exists():
            print(f'  skip {p.name} (missing)')
            continue
        loaded = rapm._load_single_file(str(p), pure=False, prefix='TS')
        if loaded is None:
            sys.exit(f'failed to load {p}')
        _, df, n = loaded
        print(f'  {p.name}: {n:,} rows')
        frames.append(df)
    if not frames:
        sys.exit(f'no TS parquets for year {year}')
    return pd.concat(frames, ignore_index=True)


def solve(df: pd.DataFrame, off_alpha: float, def_alpha: float) -> pd.DataFrame:
    """Run the standard alternating ridge and return a results frame."""
    t = rapm.transform_to_home_away_format(df)

    all_players = set()
    for col in [f'a{i}' for i in range(1, 6)] + [f'h{i}' for i in range(1, 6)]:
        pid = pd.to_numeric(t[col], errors='coerce').fillna(0).astype(int)
        all_players.update(str(p) for p in pid.unique() if p > 0)
    all_players = sorted(all_players)

    player_to_col = {}
    idx = 0
    for p in all_players:
        player_to_col[f'{p}_off'] = idx
        idx += 1
        player_to_col[f'{p}_def'] = idx
        idx += 1

    X, y, _ = rapm.build_simple_design_matrix(t, player_to_col)
    beta = rapm.simplified_alternating_rapm(
        X, y - np.mean(y), player_to_col, all_players,
        alpha_offense=off_alpha, alpha_defense=def_alpha,
        max_iter=200, tol=1e-4,
    )

    # Possessions per player per side, using the same offense/defense slot
    # resolution rapm.py uses for its own centering.
    hp = t['home_poss'].values
    h = t[[f'h{k}' for k in range(1, 6)]].values.astype(int)
    a = t[[f'a{k}' for k in range(1, 6)]].values.astype(int)
    off_slots = np.where(hp[:, None] == 1, h, a)
    def_slots = np.where(hp[:, None] == 1, a, h)

    def counts(slots):
        v = pd.Series(slots.ravel())
        v = v[v > 0]
        return defaultdict(int, v.astype(str).value_counts().to_dict())

    off_poss, def_poss = counts(off_slots), counts(def_slots)
    rapm.recenter_player_coefficients_by_side_possessions(
        beta, player_to_col, all_players, off_poss, def_poss)

    # rapm.py reports per-100 possessions; match it so these targets are directly
    # comparable to results/ts_{yy}_all_results.csv.
    rows = []
    for p in all_players:
        rows.append({
            'player_id': int(p),
            'off': round(float(beta[player_to_col[f'{p}_off']]) * 100, 3),
            'def': round(float(beta[player_to_col[f'{p}_def']]) * 100, 3),
            'off_poss': off_poss[p],
            'def_poss': def_poss[p],
        })
    out = pd.DataFrame(rows)
    out['net_rapm'] = (out['off'] - out['def']).round(3)
    out['possessions'] = out['off_poss'] + out['def_poss']
    return out[['player_id', 'off', 'def', 'net_rapm',
                'possessions', 'off_poss', 'def_poss']]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('year', type=int, help='two-digit season end year, e.g. 25')
    ap.add_argument('--off-alpha', type=float, default=3000)
    ap.add_argument('--def-alpha', type=float, default=3000)
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format='%(message)s')

    print(f'loading TS{args.year}...')
    df = load_season(args.year)

    dates = pd.to_datetime(df['game_date'], errors='coerce')
    if dates.isna().any():
        sys.exit(f'{int(dates.isna().sum())} rows have an unparseable game_date')
    cut = dates.median()
    halves = {'h1': df[dates <= cut], 'h2': df[dates > cut]}
    print(f'\nsplit at {cut:%Y-%m-%d}: '
          f'h1 {len(halves["h1"]):,} rows, h2 {len(halves["h2"]):,} rows')

    for name, part in halves.items():
        print(f'\n=== {name} ===')
        r = solve(part, args.off_alpha, args.def_alpha)
        p = RES / f'ts_{args.year}_{name}_results.csv'
        r.to_csv(p, index=False)
        n1000 = int((r['def_poss'] >= 1000).sum())
        print(f'-> {p.name}  {len(r)} players, {n1000} with >=1000 def poss')


if __name__ == '__main__':
    main()
