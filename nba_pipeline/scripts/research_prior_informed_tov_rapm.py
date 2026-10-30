"""Product B prototype: walk-forward prior-informed time-decay TOV RAPM.

Question: does a mix-adjusted allowance prior built from the rZTOV joint lane
prices improve *future-possession* turnover prediction over plain and
persistence-prior time-decay ridge?

Protocol (all regular season):
- Train: trailing 5 seasons ending y, possession rows from processed
  TOV{yy}.parquet, per-season-centered target (season effects), time-decay
  weights 0.5^(days_before_asof / half_life).
- Solve: one joint ridge over offense/defense player blocks,
  y_100 = mu_t + sum(off_i) + sum(def_j). Positive off = commits more,
  positive def = forces more. Prior-centered ridge: (G + aI) b = r + a p.
- Mix prior (offense only): p_i = s * |kappa_y| * trailing decayed rZTOV
  excess_i. Excess comes from results/rztov_seasons_2014_2026.csv, which
  prices season t with the joint window fit ending t (walk-forward by
  construction); kappa_y is the window fit ending y. As-of 2018 is the first
  fully non-inherited price year, so scored seasons start 2019.
- State prior: the same-recipe solve as-of y-1, both sides (persistence).
- Eval: score season y+1 possessions, MSE of yhat vs Is_Turnover (0/1 units,
  deltas reported e-6), paired game-clustered bootstrap.
- Folds: dev selection on scored seasons 2019-2024; frozen report 2025, 2026.
  Everything through 2026 is development-era data; the untouched confirmation
  season for any promotion is 2026-27.

Research lane only. Nothing here feeds production solves, the published rZTOV
surface, or VPM inputs.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'
PROC = ROOT / 'processed'

SEASONS = list(range(2014, 2027))
DEV_SCORE = list(range(2019, 2025))
FROZEN_SCORE = [2025, 2026]
TRAIL = 5
ALPHAS = [750.0, 1500.0, 3000.0, 6000.0]
HALF_LIVES = [365.0, 700.0, 1500.0, None]          # None = no decay
MIX_SCALES = [0.5, 1.0, 1.5]
STATE_SCALES = [0.5, 1.0, 1.5]
COMBO_GRID = [(sm, ss) for sm in (0.0, 0.5, 1.0) for ss in (0.0, 0.5, 1.0)
              if not (sm == 0.0 and ss == 0.0)]
BOOT_DRAWS = 2000
SEED = 11

# |kappa| per as-of year from the joint window fit ending that year
# (build_rztov_explorer payload; 2018 is the first non-inherited window).
KAPPA = {2018: 0.9012, 2019: 0.8561, 2020: 0.8364, 2021: 0.8241,
         2022: 0.7584, 2023: 0.7604, 2024: 0.7813, 2025: 0.8024,
         2026: 0.8275}


def load_seasons(seasons=None):
    """Per-season possession blocks with a global player column index.

    `seasons` defaults to the harness range; the daily board passes an
    explicit list so an in-progress season can be appended.
    """
    cols = ['game_id', 'O1', 'O2', 'O3', 'O4', 'O5',
            'D1', 'D2', 'D3', 'D4', 'D5', 'Is_Turnover', 'game_date']
    raw = {}
    players = set()
    for y in (SEASONS if seasons is None else seasons):
        d = pd.read_parquet(PROC / f'TOV{y - 2000}.parquet', columns=cols)
        ids = d[['O1', 'O2', 'O3', 'O4', 'O5', 'D1', 'D2', 'D3', 'D4', 'D5']]
        ok = (ids.notna().all(axis=1) & (ids > 0).all(axis=1))
        d = d[ok].reset_index(drop=True)
        raw[y] = d
        players.update(ids[ok.values].to_numpy().astype(np.int64).ravel())
    pidx = {p: i for i, p in enumerate(sorted(players))}
    P = len(pidx)
    blocks = {}
    for y, d in raw.items():
        n = len(d)
        rows = np.repeat(np.arange(n), 10)
        oc = d[['O1', 'O2', 'O3', 'O4', 'O5']].to_numpy().astype(np.int64)
        dc = d[['D1', 'D2', 'D3', 'D4', 'D5']].to_numpy().astype(np.int64)
        colo = np.vectorize(pidx.__getitem__)(oc)
        cold = np.vectorize(pidx.__getitem__)(dc) + P
        cols_ = np.concatenate([colo, cold], axis=1).ravel()
        X = sparse.csr_matrix((np.ones(10 * n), (rows, cols_)),
                              shape=(n, 2 * P))
        days = pd.to_datetime(d['game_date']).to_numpy().astype(
            'datetime64[D]').astype(np.int64)
        blocks[y] = {
            'X': X,
            'y': d['Is_Turnover'].to_numpy(np.float64) * 100.0,
            'days': days,
            'games': d['game_id'].to_numpy(),
            'asof': int(days.max()) + 1,
        }
    return blocks, pidx, P


def fold_matrices(blocks, asof_year, half_life):
    """Stacked X, season-centered y, decay weights for the trailing window."""
    years = [t for t in SEASONS if asof_year - TRAIL < t <= asof_year]
    asof = blocks[asof_year]['asof']
    Xs, ys, ws = [], [], []
    for t in years:
        b = blocks[t]
        yc = b['y'] - b['y'].mean()                  # season effects
        if half_life is None:
            w = np.ones(len(yc))
        else:
            w = 0.5 ** ((asof - b['days']) / half_life)
        Xs.append(b['X'])
        ys.append(yc)
        ws.append(w)
    X = sparse.vstack(Xs, format='csr')
    y = np.concatenate(ys)
    w = np.concatenate(ws)
    return X, y, w


def gram(blocks, asof_year, half_life, active):
    """Dense normal equations restricted to active player columns."""
    X, y, w = fold_matrices(blocks, asof_year, half_life)
    Xa = X[:, active]
    Xw = Xa.multiply(w[:, None]).tocsr()
    G = (Xa.T @ Xw).toarray()
    r = Xa.T @ (w * y)
    return G, np.asarray(r).ravel()


def active_cols(blocks, asof_year, P):
    years = [t for t in SEASONS if asof_year - TRAIL < t <= asof_year]
    seen = np.zeros(2 * P, dtype=bool)
    for t in years:
        seen |= np.asarray(blocks[t]['X'].sum(axis=0)).ravel() > 0
    return np.where(seen)[0]


def solve(G, r, alpha, prior=None):
    A = G + alpha * np.eye(len(G))
    rhs = r if prior is None else r + alpha * prior
    return np.linalg.solve(A, rhs)


def score(blocks, target_year, b_full, mu):
    """Possession MSE on the target season, y in 0/1 units."""
    bt = blocks[target_year]
    yhat = (bt['X'] @ (b_full / 100.0)) + mu / 100.0
    err = (bt['y'] / 100.0) - yhat
    return err, bt['games']


def mse(err):
    return float(np.mean(err * err))


def build_mix_prior(sea, asof_year, half_life, kappa, pidx, P):
    """p_i = |kappa| * decayed trailing possession-weighted rZTOV excess."""
    h = half_life if half_life is not None else 1e12
    s = sea[sea['year'] <= asof_year].copy()
    s['w'] = s['off_poss'] * 0.5 ** ((asof_year - s['year']) * 365.0 / h)
    g = s.groupby('nba_id').apply(
        lambda x: np.average(x['excess'], weights=x['w']), include_groups=False)
    p = np.zeros(2 * P)
    hit = 0
    for pid, e in g.items():
        i = pidx.get(int(pid))
        if i is not None:
            p[i] = kappa * e
            hit += 1
    return p, hit


FIXED_BASE = (3000.0, 1500.0)   # dev-selected (alpha, half_life), 2026-08-15


def run_dev_mix_extension(args, blocks, pidx, P, sea):
    """Dev-only mix-scale grid extension at the fixed base config.

    The preregistered grid selected mix scale 0.5 at its low edge, so this
    re-scores the dev folds on a wider scale grid. It never touches the
    frozen 2025/2026 folds; the headline frozen numbers stand as reported.
    --dump-maps additionally writes the as-of-2025 C2/C3 offense maps
    (a training-window solve, not a frozen-fold score).
    """
    A2, H2 = FIXED_BASE
    scales = [float(s) for s in args.dev_mix_scales.split(',')]
    fold_asof = {sy: sy - 1 for sy in DEV_SCORE}
    mu = {sy: blocks[fold_asof[sy]]['y'].mean() for sy in DEV_SCORE}
    grams, acts, mixp = {}, {}, {}
    for sy in DEV_SCORE:
        ac = active_cols(blocks, fold_asof[sy], P)
        acts[sy] = ac
        grams[sy] = gram(blocks, fold_asof[sy], H2, ac)
        mixp[sy], _ = build_mix_prior(sea, fold_asof[sy], H2,
                                      KAPPA[fold_asof[sy]], pidx, P)
    print(f'\ndev-only C3 mix-scale extension (alpha={A2:.0f}, h={H2:.0f}):',
          flush=True)
    for s in [0.0] + scales:
        tot = 0.0
        for sy in DEV_SCORE:
            ac = acts[sy]
            G, r = grams[sy]
            b = np.zeros(2 * P)
            prior = None if s == 0.0 else (s * mixp[sy])[ac]
            b[ac] = solve(G, r, A2, prior=prior)
            err, _ = score(blocks, sy, b, mu[sy])
            tot += mse(err)
        tag = 'C2 reference' if s == 0.0 else f'scale {s:g}'
        print(f'  {tag:>13}  dev mean MSE {tot / len(DEV_SCORE):.6f}')
    if args.dump_maps:
        yr = 2025
        ac = active_cols(blocks, yr, P)
        G, r = gram(blocks, yr, H2, ac)
        b2 = np.zeros(2 * P)
        b2[ac] = solve(G, r, A2)
        p, _ = build_mix_prior(sea, yr, H2, KAPPA[yr], pidx, P)
        b3 = np.zeros(2 * P)
        b3[ac] = solve(G, r, A2, prior=(0.5 * p)[ac])
        inv = {i: pid for pid, i in pidx.items()}
        off = [i for i in ac if i < P]
        rows = pd.DataFrame({
            'nba_id': [inv[i] for i in off],
            'off_c2': b2[off],
            'off_c3': b3[off],
            'delta': b3[off] - b2[off],
            'prior': (0.5 * p)[off],
        })
        names = pd.read_csv(RES / 'rztov_research_22_26.csv')
        rows = rows.merge(names[['nba_id', 'name']].drop_duplicates('nba_id'),
                          on='nba_id', how='left')
        out = RES / 'tov_prior_rapm_maps_2026.csv'
        rows.sort_values('delta').to_csv(out, index=False)
        print(f'wrote {out} ({len(rows)} offense rows)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boot-draws', type=int, default=BOOT_DRAWS)
    ap.add_argument('--dev-mix-scales', type=str, default=None,
                    help='comma floats: dev-only mix-scale extension at the '
                         'fixed base config (skips sweeps, chain, frozen)')
    ap.add_argument('--dump-maps', action='store_true',
                    help='with --dev-mix-scales: also write as-of-2025 '
                         'C2/C3 offense maps for player-level comparison')
    args = ap.parse_args()

    print('loading possession blocks 2014-2026 (RS)...', flush=True)
    blocks, pidx, P = load_seasons()
    n_all = sum(len(b['y']) for b in blocks.values())
    print(f'  {n_all:,} possessions, {P:,} players')

    sea = pd.read_csv(RES / 'rztov_seasons_2014_2026.csv')
    need = {'nba_id', 'year', 'off_poss', 'excess', 'act'}
    assert need <= set(sea.columns), sorted(need - set(sea.columns))

    if args.dev_mix_scales is not None:
        run_dev_mix_extension(args, blocks, pidx, P, sea)
        return

    score_years = DEV_SCORE + FROZEN_SCORE
    fold_asof = {sy: sy - 1 for sy in score_years}

    # level term shared by every candidate: last train season's mean rate
    mu = {sy: blocks[fold_asof[sy]]['y'].mean() for sy in score_years}

    act_cache = {sy: active_cols(blocks, fold_asof[sy], P)
                 for sy in score_years}

    # ---------------- C2 sweep: (alpha, half-life) on dev folds ----------
    print('\nC2 ridge-td5 sweep (alpha x half-life), dev folds '
          f'{DEV_SCORE[0]}-{DEV_SCORE[-1]}:', flush=True)
    c2_tot = {}
    gram_cache = {}
    for h in HALF_LIVES:
        for sy in DEV_SCORE:
            ac = act_cache[sy]
            G, r = gram(blocks, fold_asof[sy], h, ac)
            gram_cache[(sy, h)] = (G, r)
            for a in ALPHAS:
                b = np.zeros(2 * P)
                b[ac] = solve(G, r, a)
                err, _ = score(blocks, sy, b, mu[sy])
                c2_tot[(a, h)] = c2_tot.get((a, h), 0.0) + mse(err)
    for (a, h), v in sorted(c2_tot.items(), key=lambda kv: kv[1]):
        print(f'  alpha {a:>6.0f}  h {str(h):>6}  dev mean MSE '
              f'{v / len(DEV_SCORE):.6f}')
    (A2, H2) = min(c2_tot, key=c2_tot.get)
    print(f'  -> C2 selected alpha={A2:.0f}, half_life={H2}')

    # ---------------- C1 sweep: single-season ridge ----------------------
    c1_tot = {}
    for sy in DEV_SCORE:
        yr = fold_asof[sy]
        seen = np.asarray(blocks[yr]['X'].sum(axis=0)).ravel() > 0
        ac1 = np.where(seen)[0]
        X1 = blocks[yr]['X'][:, ac1]
        y1 = blocks[yr]['y'] - blocks[yr]['y'].mean()
        G1 = (X1.T @ X1).toarray()
        r1 = np.asarray(X1.T @ y1).ravel()
        for a in ALPHAS:
            b = np.zeros(2 * P)
            b[ac1] = solve(G1, r1, a)
            err, _ = score(blocks, sy, b, mu[sy])
            c1_tot[a] = c1_tot.get(a, 0.0) + mse(err)
    A1 = min(c1_tot, key=c1_tot.get)
    print(f'C1 ridge-1y selected alpha={A1:.0f} '
          f'(dev mean MSE {c1_tot[A1] / len(DEV_SCORE):.6f})')

    # ------------- state-prior chain: C2 recipe as-of every year ---------
    print('\nbuilding C2-recipe chain (state priors)...', flush=True)
    chain = {}
    for yr in range(SEASONS[0], FROZEN_SCORE[-1]):
        ac = np.where(
            np.logical_or.reduce([
                np.asarray(blocks[t]['X'].sum(axis=0)).ravel() > 0
                for t in SEASONS if yr - TRAIL < t <= yr]))[0]
        key = (yr + 1, H2)
        if key in gram_cache and (yr + 1) in score_years:
            G, r = gram_cache[key]
        else:
            G, r = gram(blocks, yr, H2, ac)
            if (yr + 1) in score_years:
                gram_cache[key] = (G, r)
        b = np.zeros(2 * P)
        b[ac] = solve(G, r, A2)
        chain[yr] = b

    # ---------------- mix priors per fold --------------------------------
    mixp = {}
    for sy in score_years:
        yr = fold_asof[sy]
        p, hit = build_mix_prior(sea, yr, H2, KAPPA[yr], pidx, P)
        mixp[sy] = p
        if sy in (DEV_SCORE[0], FROZEN_SCORE[0]):
            print(f'  mix prior as-of {yr}: {hit} players covered')

    # ---------------- C3/C4/C5 scale selection on dev --------------------
    def dev_total(prior_fn):
        tot = 0.0
        for sy in DEV_SCORE:
            ac = act_cache[sy]
            G, r = gram_cache[(sy, H2)]
            b = np.zeros(2 * P)
            b[ac] = solve(G, r, A2, prior=prior_fn(sy)[ac])
            err, _ = score(blocks, sy, b, mu[sy])
            tot += mse(err)
        return tot

    c3_tot = {s: dev_total(lambda sy, s=s: s * mixp[sy]) for s in MIX_SCALES}
    S3 = min(c3_tot, key=c3_tot.get)
    c4_tot = {s: dev_total(lambda sy, s=s: s * chain[fold_asof[sy] - 1])
              for s in STATE_SCALES}
    S4 = min(c4_tot, key=c4_tot.get)

    def combo_prior(sy, sm, ss):
        p = ss * chain[fold_asof[sy] - 1].copy()
        p[:P] = p[:P] + sm * mixp[sy][:P]
        return p

    c5_tot = {(sm, ss): dev_total(lambda sy, sm=sm, ss=ss:
                                  combo_prior(sy, sm, ss))
              for (sm, ss) in COMBO_GRID}
    (SM5, SS5) = min(c5_tot, key=c5_tot.get)
    print(f'\nC3 mix scale selected {S3} '
          f'(dev {c3_tot[S3] / len(DEV_SCORE):.6f})')
    print(f'C4 state scale selected {S4} '
          f'(dev {c4_tot[S4] / len(DEV_SCORE):.6f})')
    print(f'C5 combo selected mix={SM5} state={SS5} '
          f'(dev {c5_tot[(SM5, SS5)] / len(DEV_SCORE):.6f})')
    print(f'C2 dev reference {c2_tot[(A2, H2)] / len(DEV_SCORE):.6f}')

    # ---------------- frozen evaluation ----------------------------------
    def c0_map(sy):
        yr = fold_asof[sy]
        h = H2 if H2 is not None else 1e12
        s = sea[sea['year'] <= yr].copy()
        s['w'] = s['off_poss'] * 0.5 ** ((yr - s['year']) * 365.0 / h)
        g = s.groupby('nba_id').apply(
            lambda x: np.average(x['act'], weights=x['w']),
            include_groups=False)
        wsum = s.groupby('nba_id')['w'].sum()
        m = float(np.average(g.reindex(wsum.index), weights=wsum))
        b = np.zeros(2 * P)
        for pid, v in g.items():
            i = pidx.get(int(pid))
            if i is not None:
                b[i] = v - m
        return b

    def cand_map(name, sy):
        ac = act_cache[sy]
        if name == 'C0-box':
            return c0_map(sy)
        if name == 'C1-1y':
            yr = fold_asof[sy]
            seen = np.asarray(blocks[yr]['X'].sum(axis=0)).ravel() > 0
            ac1 = np.where(seen)[0]
            X1 = blocks[yr]['X'][:, ac1]
            y1 = blocks[yr]['y'] - blocks[yr]['y'].mean()
            b = np.zeros(2 * P)
            b[ac1] = solve((X1.T @ X1).toarray(),
                           np.asarray(X1.T @ y1).ravel(), A1)
            return b
        G, r = gram_cache.get((sy, H2)) or gram(blocks, fold_asof[sy], H2, ac)
        gram_cache[(sy, H2)] = (G, r)
        prior = None
        if name == 'C3-mix':
            prior = (S3 * mixp[sy])[ac]
        elif name == 'C4-state':
            prior = (S4 * chain[fold_asof[sy] - 1])[ac]
        elif name == 'C5-combo':
            prior = combo_prior(sy, SM5, SS5)[ac]
        b = np.zeros(2 * P)
        b[ac] = solve(G, r, A2, prior=prior)
        return b

    CANDS = ['C0-box', 'C1-1y', 'C2-td5', 'C3-mix', 'C4-state', 'C5-combo']
    rng = np.random.default_rng(SEED)
    rows = []
    frozen_err = {}
    print('\nfrozen folds:', FROZEN_SCORE, flush=True)
    for sy in FROZEN_SCORE:
        errs = {}
        for name in CANDS:
            b = cand_map(name, sy)
            err, games = score(blocks, sy, b, mu[sy])
            errs[name] = err
            rows.append({'fold': sy, 'cand': name, 'mse': mse(err),
                         'n': len(err)})
            print(f'  {sy} {name:<9} MSE {mse(err):.6f}')
        frozen_err[sy] = (errs, games)

    PAIRS = [('C3-mix', 'C2-td5'), ('C4-state', 'C2-td5'),
             ('C5-combo', 'C2-td5'), ('C5-combo', 'C4-state'),
             ('C2-td5', 'C1-1y'), ('C2-td5', 'C0-box')]
    print('\npaired deltas, e-6 (negative = first candidate better), '
          'game-clustered bootstrap 5-95%:')
    for sy in FROZEN_SCORE + ['pooled']:
        if sy == 'pooled':
            parts = [frozen_err[s] for s in FROZEN_SCORE]
            games = np.concatenate(
                [np.char.add(str(s) + '_', g[1].astype(str))
                 for s, g in zip(FROZEN_SCORE, parts)])
            errs = {n: np.concatenate([p[0][n] for p in parts])
                    for n in CANDS}
        else:
            errs, games = frozen_err[sy]
        ug, inv = np.unique(games, return_inverse=True)
        counts = np.bincount(inv)
        per_game = {n: np.bincount(inv, weights=errs[n] * errs[n])
                    for n in CANDS}
        print(f'  fold {sy}:')
        for a, bnm in PAIRS:
            d_pt = (np.sum(per_game[a]) - np.sum(per_game[bnm])) / len(games)
            draws = rng.integers(0, len(ug), (args.boot_draws, len(ug)))
            da = per_game[a][draws].sum(axis=1)
            db = per_game[bnm][draws].sum(axis=1)
            nn = counts[draws].sum(axis=1)
            dd = (da - db) / nn
            lo, hi = np.percentile(dd, [5, 95])
            rows.append({'fold': sy, 'cand': f'{a} vs {bnm}',
                         'mse': d_pt, 'n': len(games),
                         'lo': lo, 'hi': hi})
            print(f'    {a:<9} vs {bnm:<8} {d_pt * 1e6:+8.2f}  '
                  f'[{lo * 1e6:+.2f},{hi * 1e6:+.2f}]  '
                  f'P(better) {np.mean(dd < 0):.2f}')

    out = RES / 'tov_prior_rapm_folds.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f'\nwrote {out}')
    print('config:',
          {'C1_alpha': A1, 'C2_alpha': A2, 'C2_half_life': H2,
           'C3_scale': S3, 'C4_scale': S4, 'C5_mix': SM5, 'C5_state': SS5})


if __name__ == '__main__':
    sys.exit(main())
