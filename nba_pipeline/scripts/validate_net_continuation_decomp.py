#!/usr/bin/env python3
"""Validate the net-continuation (kappa) DECOMP tree against the gross tree.

Reads the validation rebuild (FIRST_CHANCE / SECOND_CHANCE_CLEAN 2024-2026
RS+PS with FC_DECOMP_NET_* + FC_CONT_KAPPA_Diff + SC_NET_CONT_Diff), then:

1. Row audit per year: legacy columns match production parquets exactly;
   kappa closure (NET_EFG - EFG - KAPPA == 0, SC_NET - SC + KAPPA == 0)
   holds across the FC/SC file pair on matched possessions.
2. Matched-alpha solves of the five shot components + SC, gross vs net,
   pooled 2024-2026 uniform weights, and reports the largest player-level
   movers per component plus reference bigs.

Usage: validate_net_continuation_decomp.py [--processed-dir DIR]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parents[2] / 'vpm' / 'stabilized_rapm' / 'scripts')
                if (SCRIPTS.parents[2] / 'vpm').exists() else str(SCRIPTS))
VPM_STAB = Path('/Users/russ/Documents/Codex/basketball/nba/research/vpm/stabilized_rapm/scripts')
sys.path.insert(0, str(VPM_STAB))

import rapm  # noqa: E402
import tune_forward_component_alphas as tfca  # noqa: E402
from run_stabilized_sum_eval import build_matrix_ctx, solve_component  # noqa: E402

PROD = SCRIPTS.parent / 'processed'
YEARS = [2024, 2025, 2026]
PAIRS = [  # (gross, net) component aliases; same alpha within a pair
    ('DECOMP_RIM_FREQ', 'DECOMP_NET_RIM_FREQ'),
    ('DECOMP_RIM_FG', 'DECOMP_NET_RIM_FG'),
    ('DECOMP_MID_FG', 'DECOMP_NET_MID_FG'),
    ('DECOMP_THREE_FREQ', 'DECOMP_NET_THREE_FREQ'),
    ('DECOMP_THREE_FG', 'DECOMP_NET_THREE_FG'),
]
ALPHA = (2000, 2000)
SC_ALPHA = (3000, 3000)


def row_audit(vdir: Path):
    for y in YEARS:
        for ph in ('', '_PS'):
            fc_v = vdir / f'FIRST_CHANCE{y % 100}{ph}.parquet'
            sc_v = vdir / f'SECOND_CHANCE_CLEAN{y % 100}{ph}.parquet'
            if not fc_v.exists() or not sc_v.exists():
                raise FileNotFoundError(f'missing rebuild outputs for {y}{ph}')
            fc = pd.read_parquet(fc_v)
            sc = pd.read_parquet(sc_v)
            fc_p = pd.read_parquet(PROD / fc_v.name)
            assert len(fc) == len(fc_p), f'{fc_v.name} row count changed'
            for col in ('Net_Diff', 'FC_DECOMP_RIM_FG_Diff', 'FC_DECOMP_EFG_Diff'):
                a = pd.to_numeric(fc[col], errors='coerce').fillna(-9e9)
                b = pd.to_numeric(fc_p[col], errors='coerce').fillna(-9e9)
                assert np.allclose(a, b, atol=1e-12, rtol=0), \
                    f'{fc_v.name} {col} diverges from production'
            gap1 = (fc['FC_DECOMP_NET_EFG_Diff'] - fc['FC_DECOMP_EFG_Diff']
                    - fc['FC_CONT_KAPPA_Diff']).abs().max()
            assert len(fc) == len(sc), f'{y}{ph} FC/SC row mismatch'
            gap2 = (sc['SC_NET_CONT_Diff'] - sc['Net_Diff']
                    + fc['FC_CONT_KAPPA_Diff'].values).abs().max()
            grand = ((fc['FC_DECOMP_NET_EFG_Diff'].values
                      + sc['SC_NET_CONT_Diff'].values)
                     - (fc['FC_DECOMP_EFG_Diff'].values
                        + sc['Net_Diff'].values)).max()
            assert max(gap1, gap2, abs(grand)) < 1e-9, f'{y}{ph} closure failed'
            print(f'{y}{ph or " RS"}: parity + closure exact '
                  f'({len(fc):,} possessions)', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--processed-dir', default=str(
        SCRIPTS.parent / 'processed_netcont_validation'))
    args = ap.parse_args()
    vdir = Path(args.processed_dir)

    print('=== Row audit ===', flush=True)
    row_audit(vdir)

    print('\n=== Matched-alpha solves (pooled 2024-2026, uniform) ===',
          flush=True)
    tfca.PROCESSED_DIR = vdir
    fc_raw = tfca.load_family_raw('FIRST_CHANCE', YEARS)
    sc_raw = tfca.load_family_raw('SECOND_CHANCE_CLEAN', YEARS)

    t_fc, X_fc, pl_fc, p2c_fc = build_matrix_ctx(
        fc_raw, 'DECOMP_RIM_FREQ', 'FIRST_CHANCE', None)
    w = np.ones(len(t_fc))
    t_sc, X_sc, pl_sc, p2c_sc = build_matrix_ctx(
        sc_raw, 'SECOND_CHANCE_CLEAN', 'SECOND_CHANCE_CLEAN', None)
    w_sc = np.ones(len(t_sc))

    results = {}
    for gross, net in PAIRS:
        for comp in (gross, net):
            b = tfca.train_baselines(fc_raw, comp, None)
            y = tfca.prepare_target(fc_raw, comp, 'FIRST_CHANCE', b)
            off, dfn, _ = solve_component(X_fc, y, w, pl_fc, p2c_fc, *ALPHA)
            results[comp] = (off, dfn)
            print(f'solved {comp}', flush=True)

    y_sc = tfca.prepare_target(sc_raw, 'SECOND_CHANCE_CLEAN',
                               'SECOND_CHANCE_CLEAN', None)
    off, dfn, _ = solve_component(X_sc, y_sc, w_sc, pl_sc, p2c_sc, *SC_ALPHA)
    results['SC'] = (off, dfn)
    sc_net_raw = sc_raw.copy()
    for col in ('Net_Diff', 'Off_Diff'):
        sc_net_raw[col] = sc_net_raw['SC_NET_CONT_Diff']
    sc_net_raw['Def_Diff'] = -sc_net_raw['SC_NET_CONT_Diff']
    y_scn = tfca.prepare_target(sc_net_raw, 'SECOND_CHANCE_CLEAN',
                                'SECOND_CHANCE_CLEAN', None)
    off, dfn, _ = solve_component(X_sc, y_scn, w_sc, pl_sc, p2c_sc, *SC_ALPHA)
    results['SC_NET'] = (off, dfn)
    print('solved SC + SC_NET', flush=True)

    # possession counts for eligibility
    ocols = [f'O{i}' for i in range(1, 6)]
    poss_ct = pd.concat([pd.to_numeric(fc_raw[c], errors='coerce')
                         for c in ocols]).value_counts()

    bx = pd.read_csv(Path('/Users/russ/Documents/Codex/basketball/nba/research'
                          '/vpm/boxscore.csv'),
                     usecols=['nba_id', 'year', 'Name'], low_memory=False)
    names = bx.sort_values('year').groupby('nba_id')['Name'].last()

    rows = []
    for p in pl_fc:
        pid = int(p)
        if poss_ct.get(pid, 0) < 4000:
            continue
        r = {'player_id': pid, 'name': names.get(pid, str(pid)),
             'poss': int(poss_ct.get(pid, 0))}
        for gross, net in PAIRS:
            key = gross.replace('DECOMP_', '')
            for side, idx in (('o', 0), ('d', 1)):
                g = results[gross][idx].get(p, 0.0) * 100
                n = results[net][idx].get(p, 0.0) * 100
                r[f'{side}_{key}_gross'] = g
                r[f'{side}_{key}_net'] = n
                r[f'{side}_{key}_delta'] = n - g
        for side, idx in (('o', 0), ('d', 1)):
            g = results['SC'][idx].get(p, 0.0) * 100
            n = results['SC_NET'][idx].get(p, 0.0) * 100
            r[f'{side}_SC_gross'] = g
            r[f'{side}_SC_net'] = n
            r[f'{side}_SC_delta'] = n - g
        rows.append(r)
    board = pd.DataFrame(rows)
    out = vdir / 'netcont_comparison.csv'
    board.to_csv(out, index=False)

    print('\n=== Biggest movers (pts/100, defensive sign: raw solver) ===')
    for col, label in [('d_RIM_FG_delta', 'defensive rim FG (net - gross)'),
                       ('d_SC_delta', 'defensive SC (net - gross)'),
                       ('o_RIM_FREQ_delta', 'offensive rim freq'),
                       ('o_THREE_FREQ_delta', 'offensive three freq'),
                       ('o_RIM_FG_delta', 'offensive rim FG')]:
        top = board.reindex(board[col].abs().sort_values(ascending=False)
                            .head(8).index)
        print(f'\n{label}:')
        print(top[['name', 'poss', col.replace('_delta', '_gross'),
                   col.replace('_delta', '_net'), col]]
              .round(2).to_string(index=False))

    refs = board[board['name'].isin([
        'Luke Kornet', 'Victor Wembanyama', 'Rudy Gobert', 'Nikola Jokić',
        'Chet Holmgren', 'Walker Kessler', 'Giannis Antetokounmpo',
        'Shai Gilgeous-Alexander'])]
    print('\n=== Reference players: defensive rim FG + SC (raw def coef; '
          'lower=better) ===')
    print(refs[['name', 'd_RIM_FG_gross', 'd_RIM_FG_net', 'd_SC_gross',
                'd_SC_net']].round(2).to_string(index=False))
    print(f'\nfull table: {out}', flush=True)


if __name__ == '__main__':
    main()
