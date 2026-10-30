"""rZTOV break-even review: fit diagnostics, two-lane second stage, within-player
panel, forward gate, and 2026 board preview.

Research companion to build_rztov_explorer.py / docs/RZTOV.md (August 2026 review).
Nothing here changes the production build; it evaluates how the three involvement
break-evens (self-created, assisted, potential-assist) are estimated.

Subcommands
-----------
  diag                  reproduce all 9 trailing-window fits, bootstrap CIs,
                        regressor collinearity, min-poss/weighting sensitivity
  stage2 [2226|1418|26|all]
                        constrained two-lane fit: SCORING_TOV and BADPASS_TOV
                        count RAPMs as separate targets; totals = lane sums
  panel                 within-player estimator: adjacent-season deltas of
                        single-season TOV RAPM on volume deltas (2019-2026)
  panel-lanes           same panel design per lane, summed to totals
  forward               forward gate: fit candidates on data <= y, score season
                        y's centered excess against season y+1's independent
                        single-season TOV RAPM (folds 2024->2025, 2025->2026)
  preview SELF AST PASS 2026 qualified-board impact of candidate break-evens
                        (fractions, e.g. preview 0.0996 0.0906 0.1156)

Required solve artifacts under nba_pipeline/results/ (build with rapm.py,
alphas 1500/1500; windows use --season-effects):
  tov / scoring_tov / badpass_tov  {lo}_{hi}_rs_se_a1500_1500_results.csv
      windows: 14_18 .. 22_26 (diag/stage2), 20_24 and 21_25 (forward lane2)
  tov / scoring_tov / badpass_tov  {yy}_rs_a1500_1500_results.csv
      singles: 19 .. 26 (panel, forward)
Plus the explorer seasons table results/rztov_seasons_2014_2026.csv.

Headline findings (August 2026 run)
-----------------------------------
- Production reproduction is exact; 2022-26 break-evens 15.80/5.05/9.26 with
  bootstrap sd ~0.6 and draw correlation self~pass ~ -0.71 (collinearity seesaw;
  weighted design condition number 31.6, v_self~v_potast r=0.70).
- Two-lane totals barely move (16.75/4.40/9.32 on 22-26) but the lane split
  shows ~half of the self allowance is claimed *bad-pass* suppression.
- Within-player panel (n=1704 pairs): 10.16/8.79/11.31 - self is over-priced
  and passing under-priced relative to the cross-section, supporting the
  operator's intuition in the causal reading.
- Forward gate: current and lane2 window fits tie (-0.595/-0.591 and
  -0.541/-0.550); within panel loses the 2025->2026 fold (-0.467). The
  cross-sectional fit wins the cross-sectional proxy task; the within fit
  answers the causal question. Raw committed TOV/100 has ~zero forward
  correlation (-0.07/+0.01) - the allowance does all the predictive work.
"""
import sys

import numpy as np
import pandas as pd
from pathlib import Path

RES = Path(__file__).resolve().parent.parent / 'results'
SEASONS = RES / 'rztov_seasons_2014_2026.csv'
VOLS = ['v_self', 'v_ast', 'v_potast']
FIT_MIN_POSS = 2000
PANEL_MIN_POSS = 1500
PANEL_YEARS = range(2019, 2027)

rows = pd.read_csv(SEASONS)


def wls_ni(y, X, w):
    """No-intercept possession-weighted least squares; returns beta, weighted R2."""
    sw = np.sqrt(w)
    b = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
    fit = X @ b
    r2 = 1 - ((y - fit) ** 2 * w).sum() / (((y - np.average(y, weights=w)) ** 2 * w).sum())
    return b, r2


def window_frame(lo, hi):
    """Aggregate the explorer season rows to one row per player over [lo, hi]."""
    w = rows[rows['year'].between(lo, hi)].copy()
    w['n_sc'] = w['sc_100'] * w['off_poss'] / 100
    w['n_bp'] = w['bp_100'] * w['off_poss'] / 100
    g = w.groupby('nba_id').apply(lambda x: pd.Series({
        'poss': x['off_poss'].sum(),
        'sc_100': x['n_sc'].sum() / x['off_poss'].sum() * 100,
        'bp_100': x['n_bp'].sum() / x['off_poss'].sum() * 100,
        **{c: np.average(x[c], weights=x['off_poss']) for c in VOLS},
    }), include_groups=False).reset_index()
    g['act'] = g['sc_100'] + g['bp_100']
    return g


def read_off(path, col='y'):
    t = pd.read_csv(path)
    return t[['player_id', 'off']].rename(columns={'player_id': 'nba_id', 'off': col})


def solve_path(stub, lo, hi=None):
    if hi is None or lo == hi:
        return RES / f'{stub}_{lo - 2000}_rs_a1500_1500_results.csv'
    return RES / f'{stub}_{lo - 2000}_{hi - 2000}_rs_se_a1500_1500_results.csv'


def load_singles(stub, col, years):
    return pd.concat([read_off(solve_path(stub, yy), col).assign(year=yy)
                      for yy in years], ignore_index=True)


def fit_single(f):
    X = f[['act'] + VOLS].to_numpy(float)
    b, r2 = wls_ni(f['y'].to_numpy(float), X, f['poss'].to_numpy(float))
    return -b[1:] / b[0], b[0], r2


# ---------------------------------------------------------------- diag
def cmd_diag():
    rng = np.random.default_rng(7)
    print('=== single-target window fits (reproduction + drift) ===')
    print(f'{"window":<9}{"self":>7}{"ast":>7}{"pass":>7}{"kappa":>8}{"R2w":>7}{"n":>5}'
          f'{"ast+pass":>10}')
    store = {}
    for end in range(2018, 2027):
        lo = end - 4
        g = window_frame(lo, end).merge(read_off(solve_path('tov', lo, end)), on='nba_id')
        f = g[g['poss'] >= FIT_MIN_POSS].dropna(subset=['y'])
        store[end] = f
        be, k, r2 = fit_single(f)
        print(f'{lo}-{end % 100:02d}{be[0] * 100:7.2f}{be[1] * 100:7.2f}{be[2] * 100:7.2f}'
              f'{k:8.3f}{r2:7.3f}{len(f):5d}{(be[1] + be[2]) * 100:10.2f}')

    for end in (2018, 2026):
        f = store[end]
        X = f[['act'] + VOLS].to_numpy(float)
        y = f['y'].to_numpy(float)
        w = f['poss'].to_numpy(float)
        draws = []
        for _ in range(4000):
            i = rng.integers(0, len(f), len(f))
            b, _ = wls_ni(y[i], X[i], w[i])
            draws.append(-b[1:] / b[0])
        B = np.array(draws)
        lo_, hi_ = np.percentile(B, [5, 95], axis=0)
        cc = np.corrcoef(B.T)
        print(f'\n--- bootstrap, window ending {end} (n={len(f)}, 4000 draws, 90% CI) ---')
        for j, k in enumerate(('self', 'ast', 'pass')):
            print(f'  be_{k:<5} {B[:, j].mean() * 100:6.2f}  '
                  f'[{lo_[j] * 100:6.2f},{hi_[j] * 100:6.2f}]  sd {B[:, j].std() * 100:5.2f}')
        print(f'  draw correlations: self~ast {cc[0, 1]:+.2f}  self~pass {cc[0, 2]:+.2f}  '
              f'ast~pass {cc[1, 2]:+.2f}')

    f = store[2026]
    Xw = f[['act'] + VOLS].to_numpy(float) * np.sqrt(f['poss'].to_numpy(float))[:, None]
    lab = ['act', 'sc', 'bp', 'self', 'ast', 'potast']
    cm = np.corrcoef(f[['act', 'sc_100', 'bp_100'] + VOLS].T.to_numpy(float))
    print('\n=== 22-26 fit sample: regressor correlations ===')
    print('        ' + ''.join(f'{c:>8}' for c in lab))
    for i, c in enumerate(lab):
        print(f'{c:<8}' + ''.join(f'{cm[i, j]:8.2f}' for j in range(len(lab))))
    print(f'design condition number (weighted, no intercept): {np.linalg.cond(Xw):.1f}')

    print('\n=== 22-26 sensitivity of break-evens ===')
    g26 = store[2026]
    for mp in (1000, 2000, 4000):
        be, _, _ = fit_single(g26[g26['poss'] >= mp])
        print(f'  min poss {mp:>5}: self {be[0] * 100:6.2f}  ast {be[1] * 100:6.2f}  '
              f'pass {be[2] * 100:6.2f}  (n={(g26["poss"] >= mp).sum()})')
    b_eq, _ = wls_ni(g26['y'].to_numpy(float), g26[['act'] + VOLS].to_numpy(float),
                     np.ones(len(g26)))
    print(f'  equal-weight   : self {-b_eq[1] / b_eq[0] * 100:6.2f}  '
          f'ast {-b_eq[2] / b_eq[0] * 100:6.2f}  pass {-b_eq[3] / b_eq[0] * 100:6.2f}')

    print('\n=== league volume context per window (poss-weighted mean) ===')
    for end in (2018, 2022, 2026):
        g = window_frame(end - 4, end)
        W = g['poss'].to_numpy(float)
        print(f'  {end - 4}-{end % 100:02d}: '
              + '  '.join(f'{c} {np.average(g[c], weights=W):5.2f}'
                          for c in VOLS + ['sc_100', 'bp_100']))


# ---------------------------------------------------------------- stage2
def two_lane(f, rng, verbose=True, boot_n=0):
    """Constrained two-lane fit:
       y_sc = k_sc*(sc - be1.v),  y_bp = k_bp*(bp - be2.v);  be_total = be1 + be2."""
    w = f['poss'].to_numpy(float)
    V = f[VOLS].to_numpy(float)
    sc, bp = f['sc_100'].to_numpy(float), f['bp_100'].to_numpy(float)
    ysc, ybp = f['y_sc'].to_numpy(float), f['y_bp'].to_numpy(float)

    b1, r2s = wls_ni(ysc, np.column_stack([sc, V]), w)
    b2, r2b = wls_ni(ybp, np.column_stack([bp, V]), w)
    be1, be2 = -b1[1:] / b1[0], -b2[1:] / b2[0]
    tot = be1 + be2
    if verbose:
        print(f'  scoring lane : k {b1[0]:+.3f} R2w {r2s:.3f}  be '
              + ' '.join(f'{x * 100:6.2f}' for x in be1))
        print(f'  bad-pass lane: k {b2[0]:+.3f} R2w {r2b:.3f}  be '
              + ' '.join(f'{x * 100:6.2f}' for x in be2))
        print(f'  TOTAL be     : self {tot[0] * 100:6.2f}  ast {tot[1] * 100:6.2f}  '
              f'pass {tot[2] * 100:6.2f}')
    if boot_n:
        draws = []
        for _ in range(boot_n):
            i = rng.integers(0, len(f), len(f))
            d1, _ = wls_ni(ysc[i], np.column_stack([sc[i], V[i]]), w[i])
            d2, _ = wls_ni(ybp[i], np.column_stack([bp[i], V[i]]), w[i])
            draws.append((-d1[1:] / d1[0]) + (-d2[1:] / d2[0]))
        D = np.array(draws)
        lo_, hi_ = np.percentile(D, [5, 95], axis=0)
        cc = np.corrcoef(D.T)
        for j, k in enumerate(('self', 'ast', 'pass')):
            print(f'    be_{k:<5} total {D[:, j].mean() * 100:6.2f}  '
                  f'[{lo_[j] * 100:6.2f},{hi_[j] * 100:6.2f}]  sd {D[:, j].std() * 100:5.2f}')
        print(f'    draw correlations: self~pass {cc[0, 2]:+.2f}  self~ast {cc[0, 1]:+.2f}')
    return tot, be1, be2


def run_stage2_window(lo, hi, rng, min_poss=FIT_MIN_POSS):
    print(f'\n================ window {lo}-{hi} ================')
    g = window_frame(lo, hi)
    for stub, col in (('tov', 'y_tov'), ('scoring_tov', 'y_sc'), ('badpass_tov', 'y_bp')):
        g = g.merge(read_off(solve_path(stub, lo, hi), col), on='nba_id', how='inner')
    f = g[g['poss'] >= min_poss].dropna(subset=['y_tov', 'y_sc', 'y_bp']).copy()
    print(f'n = {len(f)} (min poss {min_poss})')

    gap = (f['y_sc'] + f['y_bp'] - f['y_tov'])
    print(f'additivity of lane solves: max |sc+bp-tov| = {gap.abs().max():.4f}, '
          f'corr(sc+bp, tov) = {np.corrcoef(f.y_sc + f.y_bp, f.y_tov)[0, 1]:.6f}')

    be, k, r2 = fit_single(f.rename(columns={'y_tov': 'y'}))
    print(f'single-target (current): self {be[0] * 100:6.2f}  ast {be[1] * 100:6.2f}  '
          f'pass {be[2] * 100:6.2f}   kappa {k:+.3f} R2w {r2:.3f}')
    print('two-lane second stage:')
    two_lane(f, rng, boot_n=4000)

    # diagnostic: unrestricted 5-col per lane -> cross-commit contamination
    w = f['poss'].to_numpy(float)
    X5 = f[['sc_100', 'bp_100'] + VOLS].to_numpy(float)
    b1, _ = wls_ni(f['y_sc'].to_numpy(float), X5, w)
    b2, _ = wls_ni(f['y_bp'].to_numpy(float), X5, w)
    print(f'  cross-commit coefs (want ~0): bp in sc-lane {b1[1]:+.3f}, '
          f'sc in bp-lane {b2[0]:+.3f}')


def cmd_stage2(which):
    rng = np.random.default_rng(11)
    if which in ('2226', 'all'):
        run_stage2_window(2022, 2026, rng)
    if which in ('1418', 'all'):
        run_stage2_window(2014, 2018, rng)
    if which in ('26', 'all'):
        run_stage2_window(2026, 2026, rng, min_poss=1000)


# ---------------------------------------------------------------- panel
def panel_pairs(hi_year, cols):
    """Adjacent-season within-player pairs on single-season solves, pair-year
    weighted-demeaned so league drift between adjacent solves drops out."""
    years = range(PANEL_YEARS.start, hi_year + 1)
    d = rows.copy()
    for stub, col in cols:
        d = d.merge(load_singles(stub, col, years), on=['nba_id', 'year'], how='inner')
    d = d[d['off_poss'] >= PANEL_MIN_POSS]
    base = ['sc_100', 'bp_100'] + VOLS + [c for _, c in cols]
    lag = d[['nba_id', 'year', 'off_poss'] + base].copy()
    lag['year'] += 1
    p = d.merge(lag, on=['nba_id', 'year'], suffixes=('', '_p'))
    for c in base:
        p[f'd_{c}'] = p[c] - p[f'{c}_p']
    p['d_act'] = p['d_sc_100'] + p['d_bp_100']
    p['w'] = np.minimum(p['off_poss'], p['off_poss_p'])
    for c in ['d_act'] + [f'd_{c}' for c in base]:
        mu = p.groupby('year').apply(
            lambda g, c=c: np.average(g[c], weights=g['w']), include_groups=False)
        p[c] = p[c] - p['year'].map(mu)
    return p


def cluster_boot(p, fit_fn, rng, n=4000):
    ids = p['nba_id'].to_numpy()
    uid = np.unique(ids)
    idx_by = {u: np.flatnonzero(ids == u) for u in uid}
    draws = []
    for _ in range(n):
        take = np.concatenate([idx_by[u] for u in rng.choice(uid, len(uid))])
        draws.append(fit_fn(take))
    return np.array(draws)


def cmd_panel():
    rng = np.random.default_rng(3)
    d = rows.merge(load_singles('tov', 'y', PANEL_YEARS), on=['nba_id', 'year'],
                   how='inner')
    d = d[d['off_poss'] >= PANEL_MIN_POSS].copy()
    b, r2 = wls_ni(d['y'].to_numpy(float), d[['act'] + VOLS].to_numpy(float),
                   d['off_poss'].to_numpy(float))
    print(f'between (pooled 2019-26 singles, n={len(d)}): '
          f'self {-b[1] / b[0] * 100:6.2f}  ast {-b[2] / b[0] * 100:6.2f}  '
          f'pass {-b[3] / b[0] * 100:6.2f}  kappa {b[0]:+.3f}  R2w {r2:.3f}')

    p = panel_pairs(2026, [('tov', 'y_tov')])
    Xd = p[['d_act'] + [f'd_{c}' for c in VOLS]].to_numpy(float)
    yd = p['d_y_tov'].to_numpy(float)
    wd = p['w'].to_numpy(float)
    bd, r2d = wls_ni(yd, Xd, wd)
    print(f'within (adjacent-season deltas, n={len(p)}): '
          f'self {-bd[1] / bd[0] * 100:6.2f}  ast {-bd[2] / bd[0] * 100:6.2f}  '
          f'pass {-bd[3] / bd[0] * 100:6.2f}  kappa {bd[0]:+.3f}  R2w {r2d:.3f}')

    def one(take):
        bb, _ = wls_ni(yd[take], Xd[take], wd[take])
        return -bb[1:] / bb[0]

    D = cluster_boot(p, one, rng)
    lo_, hi_ = np.percentile(D, [5, 95], axis=0)
    for j, k in enumerate(('self', 'ast', 'pass')):
        print(f'  within be_{k:<5} {D[:, j].mean() * 100:6.2f}  '
              f'[{lo_[j] * 100:6.2f},{hi_[j] * 100:6.2f}]  sd {D[:, j].std() * 100:5.2f}')

    hp = p[np.minimum(p['off_poss'], p['off_poss_p']) >= 2500]
    bh, _ = wls_ni(hp['d_y_tov'].to_numpy(float),
                   hp[['d_act'] + [f'd_{c}' for c in VOLS]].to_numpy(float),
                   hp['w'].to_numpy(float))
    print(f'  robust: both seasons >=2500 poss (n={len(hp)}): '
          f'self {-bh[1] / bh[0] * 100:6.2f}  ast {-bh[2] / bh[0] * 100:6.2f}  '
          f'pass {-bh[3] / bh[0] * 100:6.2f}')
    q = p[(p['d_v_self'].abs() <= p['d_v_self'].abs().quantile(0.95))
          & (p['d_v_potast'].abs() <= p['d_v_potast'].abs().quantile(0.95))]
    bt, _ = wls_ni(q['d_y_tov'].to_numpy(float),
                   q[['d_act'] + [f'd_{c}' for c in VOLS]].to_numpy(float),
                   q['w'].to_numpy(float))
    print(f'  robust: trim top-5% |role change|   (n={len(q)}): '
          f'self {-bt[1] / bt[0] * 100:6.2f}  ast {-bt[2] / bt[0] * 100:6.2f}  '
          f'pass {-bt[3] / bt[0] * 100:6.2f}')


def cmd_panel_lanes():
    rng = np.random.default_rng(5)
    p = panel_pairs(2026, [('scoring_tov', 'y_sc'), ('badpass_tov', 'y_bp')])
    print(f'lane panel pairs: n={len(p)}')
    w = p['w'].to_numpy(float)
    dV = p[[f'd_{c}' for c in VOLS]].to_numpy(float)
    Xsc = np.column_stack([p['d_sc_100'].to_numpy(float), dV])
    Xbp = np.column_stack([p['d_bp_100'].to_numpy(float), dV])
    ysc, ybp = p['d_y_sc'].to_numpy(float), p['d_y_bp'].to_numpy(float)

    b1, r1 = wls_ni(ysc, Xsc, w)
    b2, r2_ = wls_ni(ybp, Xbp, w)
    be1, be2 = -b1[1:] / b1[0], -b2[1:] / b2[0]
    print(f'within scoring lane : k {b1[0]:+.3f} R2w {r1:.3f}  be '
          + ' '.join(f'{x * 100:6.2f}' for x in be1))
    print(f'within bad-pass lane: k {b2[0]:+.3f} R2w {r2_:.3f}  be '
          + ' '.join(f'{x * 100:6.2f}' for x in be2))
    tot = be1 + be2
    print(f'within TOTAL        : self {tot[0] * 100:6.2f}  ast {tot[1] * 100:6.2f}  '
          f'pass {tot[2] * 100:6.2f}')

    def one(take):
        a1, _ = wls_ni(ysc[take], Xsc[take], w[take])
        a2, _ = wls_ni(ybp[take], Xbp[take], w[take])
        return np.concatenate([-a1[1:] / a1[0], -a2[1:] / a2[0]])

    D = cluster_boot(p, one, rng)
    lo_, hi_ = np.percentile(D, [5, 95], axis=0)
    for j, k in enumerate(('sc:self', 'sc:ast', 'sc:pass',
                           'bp:self', 'bp:ast', 'bp:pass')):
        print(f'  {k:<8} {D[:, j].mean() * 100:6.2f}  '
              f'[{lo_[j] * 100:6.2f},{hi_[j] * 100:6.2f}]')
    Dt = D[:, :3] + D[:, 3:]
    lo_, hi_ = np.percentile(Dt, [5, 95], axis=0)
    for j, k in enumerate(('self', 'ast', 'pass')):
        print(f'  total be_{k:<5} {Dt[:, j].mean() * 100:6.2f}  '
              f'[{lo_[j] * 100:6.2f},{hi_[j] * 100:6.2f}]  sd {Dt[:, j].std() * 100:5.2f}')


# ---------------------------------------------------------------- forward
def fit_current(y):
    lo = y - 4
    f = window_frame(lo, y).merge(read_off(solve_path('tov', lo, y)),
                                  on='nba_id').dropna(subset=['y'])
    be, _, _ = fit_single(f[f['poss'] >= FIT_MIN_POSS])
    return be


def fit_lane2_win(y):
    lo = y - 4
    f = window_frame(lo, y)
    for stub, col in (('scoring_tov', 'y_sc'), ('badpass_tov', 'y_bp')):
        f = f.merge(read_off(solve_path(stub, lo, y), col), on='nba_id', how='inner')
    f = f[f['poss'] >= FIT_MIN_POSS].dropna(subset=['y_sc', 'y_bp'])
    tot, _, _ = two_lane(f, None, verbose=False)
    return tot


def fit_within_tot(y):
    p = panel_pairs(y, [('tov', 'y_tov')])
    b = wls_ni(p['d_y_tov'].to_numpy(float),
               p[['d_act'] + [f'd_{c}' for c in VOLS]].to_numpy(float),
               p['w'].to_numpy(float))[0]
    return -b[1:] / b[0]


def fit_within_lane(y):
    p = panel_pairs(y, [('scoring_tov', 'y_sc'), ('badpass_tov', 'y_bp')])
    w = p['w'].to_numpy(float)
    dV = p[[f'd_{c}' for c in VOLS]].to_numpy(float)
    b1 = wls_ni(p['d_y_sc'].to_numpy(float),
                np.column_stack([p['d_sc_100'].to_numpy(float), dV]), w)[0]
    b2 = wls_ni(p['d_y_bp'].to_numpy(float),
                np.column_stack([p['d_bp_100'].to_numpy(float), dV]), w)[0]
    return (-b1[1:] / b1[0]) + (-b2[1:] / b2[0])


def wcorr(a, b, w):
    am = a - np.average(a, weights=w)
    bm = b - np.average(b, weights=w)
    return float((am * bm * w).sum() / np.sqrt((am ** 2 * w).sum() * (bm ** 2 * w).sum()))


def cmd_forward():
    print(f'{"fold":<12}{"candidate":<13}{"be_self":>8}{"be_ast":>8}{"be_pass":>8}'
          f'{"fwd_r":>8}{"n":>5}')
    for y in (2024, 2025):
        sy = rows[(rows['year'] == y) & (rows['off_poss'] >= 1500)].copy()
        nxt = read_off(solve_path('tov', y + 1), 'y_next')
        nxt_poss = rows[(rows['year'] == y + 1) & (rows['off_poss'] >= 1000)][
            ['nba_id', 'off_poss']].rename(columns={'off_poss': 'poss_next'})
        sy = sy.merge(nxt, on='nba_id').merge(nxt_poss, on='nba_id')
        sy['w'] = np.minimum(sy['off_poss'], sy['poss_next'])

        cands = {
            'current': fit_current(y),
            'lane2_win': fit_lane2_win(y),
            'within_tot': fit_within_tot(y),
            'within_lane': fit_within_lane(y),
        }
        cands['blend50'] = (cands['current'] + cands['within_tot']) / 2
        for name, be in cands.items():
            allow = sum(be[j] * sy[c] for j, c in enumerate(VOLS))
            excess = sy['act'] - allow
            excess -= np.average(excess, weights=sy['off_poss'])
            r = wcorr(excess.to_numpy(float), sy['y_next'].to_numpy(float),
                      sy['w'].to_numpy(float))
            print(f'{y}->{y + 1:<6}{name:<13}{be[0] * 100:8.2f}{be[1] * 100:8.2f}'
                  f'{be[2] * 100:8.2f}{r:8.3f}{len(sy):5d}')
        r_act = wcorr(sy['act'].to_numpy(float), sy['y_next'].to_numpy(float),
                      sy['w'].to_numpy(float))
        print(f'{y}->{y + 1:<6}{"act (raw)":<13}{"-":>8}{"-":>8}{"-":>8}'
              f'{r_act:8.3f}{len(sy):5d}')


# ---------------------------------------------------------------- preview
def cmd_preview(alt):
    cur = np.array([0.1580, 0.0505, 0.0926])  # production 2022-26 window fit
    s = rows[(rows['year'] == 2026) & (rows['qualified'] == 1)].copy()

    def rz(be):
        allow = sum(be[j] * s[c] for j, c in enumerate(VOLS))
        exc = s['act'] - allow
        exc = exc - np.average(exc, weights=s['off_poss'])
        return exc / s['plays'].clip(lower=1) * 100, -exc

    s['rz_cur'], s['saved_cur'] = rz(cur)
    s['rz_alt'], s['saved_alt'] = rz(alt)
    s['d_saved'] = s['saved_alt'] - s['saved_cur']
    s['rank_cur'], s['rank_alt'] = s['rz_cur'].rank(), s['rz_alt'].rank()

    print(f'candidate be: self {alt[0] * 100:.2f} ast {alt[1] * 100:.2f} '
          f'pass {alt[2] * 100:.2f}  (current {cur[0] * 100:.2f}/{cur[1] * 100:.2f}/'
          f'{cur[2] * 100:.2f})')
    print(f'2026 qualified n={len(s)}; corr(rz_cur, rz_alt) = '
          f'{np.corrcoef(s.rz_cur, s.rz_alt)[0, 1]:.3f}; '
          f'mean |rank shift| = {np.abs(s.rank_cur - s.rank_alt).mean():.1f} of {len(s)}')

    cols = ['name', 'off_poss', 'v_self', 'v_potast', 'saved_cur', 'saved_alt', 'd_saved']
    pd.set_option('display.width', 140)
    fmt = dict(index=False, float_format=lambda x: f'{x:.2f}')
    print('\nbiggest gainers (turnovers saved/100, current -> candidate):')
    print(s.nlargest(10, 'd_saved')[cols].to_string(**fmt))
    print('\nbiggest losers:')
    print(s.nsmallest(10, 'd_saved')[cols].to_string(**fmt))
    print('\ntop 12 by candidate rZTOV (saved/100):')
    print(s.nlargest(12, 'saved_alt')[cols].to_string(**fmt))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'diag'
    if cmd == 'diag':
        cmd_diag()
    elif cmd == 'stage2':
        cmd_stage2(sys.argv[2] if len(sys.argv) > 2 else 'all')
    elif cmd == 'panel':
        cmd_panel()
    elif cmd == 'panel-lanes':
        cmd_panel_lanes()
    elif cmd == 'forward':
        cmd_forward()
    elif cmd == 'preview':
        cmd_preview(np.array([float(x) for x in sys.argv[2:5]]))
    else:
        raise SystemExit(f'unknown subcommand {cmd!r}; see module docstring')
