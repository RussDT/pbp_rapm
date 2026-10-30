"""Forward test: do ShotQuality closest-defender features improve a dTS prior?

Design
------
Only two SQ seasons exist (2024-25 well covered, 2025-26 half covered), so there is
exactly one honest forward fold:

    features from season 2025 (2024-25)  ->  target dTS in season 2026 (2025-26)

The incumbent prior is the player's own prior-season dTS. The question is whether
SQ defender features add anything beyond that. Generalization is estimated by
K-fold CV *across players* inside the fixed forward fold; the temporal direction is
never crossed, so this cannot leak future information into the past.

Arms are kept separate on purpose:
  geom     pre-release geometry and role only (no outcome ever touched)
  outcome  prior-season allowed value / suppression (legitimate for a prior, but
           it is a lagged outcome, so it is reported apart from geometry)

Sign convention: the solver's ``def`` column is negative-is-better. This flips it
so higher = better defense throughout.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / 'results'
MIN_POSS = 1000
MIN_SHOTS = 150
FOLDS = 5
SEED = 0

GEOM = ['shots_defended', 'rim_share', 'three_share', 'mean_dist', 'pct_tight',
        'pct_wide', 'mean_contest', 'mean_bbox_diff', 'pct_blocking', 'pct_behind',
        'pct_inside', 'mean_cos', 'mean_spacing', 'mean_n_closer', 'pct_alone',
        'mean_def_in_paint', 'pct_vs_ra', 'mean_d2_dist', 'rim_dist', 'three_dist']
OUTCOME = ['ev_allowed', 'pts_allowed', 'ev_suppression', 'shot_taking_allowed',
           'rim_ev_allowed', 'rim_suppression', 'mid_suppression', 'three_suppression']

# Motion features, recovered by moving to the parquet store. The per-game CSVs
# carried these on only 951 of 2,740 files, so they were previously measured at
# ~2% populated and dropped without ever being tested.
EFFORT = ['mean_closing_v', 'mean_closing_a', 'pct_closing', 'mean_d1_speed',
          'mean_d1_v_basket', 'mean_d2_closing_v', 'mean_sh_speed',
          'rim_closing_v', 'three_closing_v']

# 20 geometry features on ~290 players is over-parameterized. These are the ones
# whose univariate sign is physically coherent (tighter/closer contests and a
# rim-facing assignment predict better defense), chosen before seeing the CV.
GEOM_C = ['rim_share', 'three_share', 'mean_dist', 'pct_vs_ra', 'mean_n_closer']
OUT_C = ['rim_suppression', 'pts_allowed']


def load():
    a = pd.read_csv(RES / 'ts_25_all_results.csv')
    b = pd.read_csv(RES / 'ts_26_all_results.csv')
    a = a[a['def_poss'] >= MIN_POSS][['player_id', 'player_name', 'def', 'def_poss']]
    a.columns = ['player_id', 'player_name', 'dts_prev', 'poss_prev']
    b = b[b['def_poss'] >= MIN_POSS][['player_id', 'def', 'def_poss']]
    b.columns = ['player_id', 'dts_next', 'poss_next']
    d = a.merge(b, on='player_id')
    d['dts_prev'] *= -1.0   # higher = better defense
    d['dts_next'] *= -1.0

    f = pd.read_csv(RES / 'sq_defender_season_features.csv')
    f = f[(f['season'] == 2025) & (f['shots_defended'] >= MIN_SHOTS)]
    d = d.merge(f.rename(columns={'nba_id': 'player_id'}), on='player_id', how='left')
    d['_sq'] = d['shots_defended'].notna()
    print(f'{len(d)} players with dTS in both seasons (>= {MIN_POSS} def poss each)')
    print(f'{d["_sq"].sum()} of them have SQ defender features (>= {MIN_SHOTS} shots defended)')
    return d[d['_sq']].reset_index(drop=True)


def ridge_cv(X, y, w, alphas=(0.1, 1, 3, 10, 30, 100, 300, 1000)):
    """Weighted ridge, alpha chosen by inner CV on the training rows only."""
    rng = np.random.RandomState(SEED)
    inner = rng.permutation(len(y)) % 4
    best, ba = np.inf, alphas[0]
    for a in alphas:
        err = []
        for k in range(4):
            tr, te = inner != k, inner == k
            if te.sum() == 0:
                continue
            c = _fit(X[tr], y[tr], w[tr], a)
            err.append(np.average((y[te] - _pred(X[te], c)) ** 2, weights=w[te]))
        if np.mean(err) < best:
            best, ba = np.mean(err), a
    return _fit(X, y, w, ba), ba


def _fit(X, y, w, a):
    X1 = np.c_[np.ones(len(X)), X]
    P = np.eye(X1.shape[1]) * a
    P[0, 0] = 0.0                                  # never penalize the intercept
    return np.linalg.solve(X1.T @ (w[:, None] * X1) + P, X1.T @ (w * y))


def _pred(X, c):
    return np.c_[np.ones(len(X)), X] @ c


def evaluate(d, cols, label, folds=FOLDS):
    y = d['dts_next'].values
    w = d['poss_next'].values.astype(float)
    if cols:
        X = d[cols].values.astype(float)
        mu = np.nanmean(X, axis=0)
        X = np.where(np.isnan(X), mu, X)
        X = (X - X.mean(0)) / np.where(X.std(0) == 0, 1, X.std(0))
    else:
        X = np.zeros((len(d), 0))
    rng = np.random.RandomState(SEED)
    fold = rng.permutation(len(d)) % folds
    pred = np.zeros(len(d))
    for k in range(folds):
        tr, te = fold != k, fold == k
        c, _ = ridge_cv(X[tr], y[tr], w[tr])
        pred[te] = _pred(X[te], c)
    rmse = np.sqrt(np.average((y - pred) ** 2, weights=w))
    base = np.sqrt(np.average((y - np.average(y, weights=w)) ** 2, weights=w))
    print(f'  {label:34s} RMSE {rmse:.4f}   R2 {1 - (rmse / base) ** 2:+.4f}   ({len(cols)} feat)')
    return pred, rmse


def paired_boot(d, p_a, p_b, name_a, name_b, n=2000):
    """Player-clustered paired bootstrap on the RMSE difference (a - b)."""
    y, w = d['dts_next'].values, d['poss_next'].values.astype(float)
    rng = np.random.RandomState(SEED)
    diffs = []
    for _ in range(n):
        i = rng.randint(0, len(d), len(d))
        ra = np.sqrt(np.average((y[i] - p_a[i]) ** 2, weights=w[i]))
        rb = np.sqrt(np.average((y[i] - p_b[i]) ** 2, weights=w[i]))
        diffs.append(ra - rb)
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    verdict = 'IMPROVES' if hi < 0 else ('WORSE' if lo > 0 else 'neutral')
    print(f'  {name_a} vs {name_b}: dRMSE {diffs.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]  {verdict}')


if __name__ == '__main__':
    d = load()
    print(f'\nforward fold: SQ 2024-25 features -> dTS 2025-26, n={len(d)}\n')
    print('out-of-sample (5-fold across players, weighted by 2025-26 def possessions):')
    p_mean, _ = evaluate(d, [], 'league mean')
    p_base, _ = evaluate(d, ['dts_prev'], 'prior dTS (incumbent)')
    p_geom, _ = evaluate(d, ['dts_prev'] + GEOM, 'prior dTS + SQ geometry')
    p_out, _ = evaluate(d, ['dts_prev'] + OUTCOME, 'prior dTS + SQ allowed/suppression')
    p_all, _ = evaluate(d, ['dts_prev'] + GEOM + OUTCOME, 'prior dTS + all SQ')
    p_sq, _ = evaluate(d, GEOM + OUTCOME, 'SQ only (no prior dTS)')
    print('  -- effort / motion (newly recovered) --')
    p_eff, _ = evaluate(d, ['dts_prev'] + EFFORT, 'prior dTS + effort')
    p_ge, _ = evaluate(d, ['dts_prev'] + GEOM + EFFORT, 'prior dTS + geometry + effort')
    p_all3, _ = evaluate(d, ['dts_prev'] + GEOM + EFFORT + OUTCOME, 'prior dTS + all three arms')
    print('  -- compact specs --')
    p_gc, _ = evaluate(d, ['dts_prev'] + GEOM_C, 'prior dTS + compact geometry')
    p_oc, _ = evaluate(d, ['dts_prev'] + OUT_C, 'prior dTS + compact outcome')
    p_cc, _ = evaluate(d, ['dts_prev'] + GEOM_C + OUT_C, 'prior dTS + compact both')
    p_r1, _ = evaluate(d, ['dts_prev', 'rim_suppression'], 'prior dTS + rim_suppression')
    p_r0, _ = evaluate(d, ['rim_suppression'], 'rim_suppression alone')

    print('\npaired bootstrap vs incumbent (negative = SQ helps):')
    for p, n in ((p_geom, 'geometry '), (p_out, 'outcome  '), (p_all, 'all SQ   '),
                 (p_eff, 'effort   '), (p_ge, 'geom+eff '), (p_all3, 'all three'),
                 (p_gc, 'geom cmpt'), (p_oc, 'out cmpt '), (p_cc, 'both cmpt'),
                 (p_r1, 'rim_supp '), (p_r0, 'rim alone')):
        paired_boot(d, p, p_base, n, 'prior dTS')
    print()
    paired_boot(d, p_base, p_mean, 'prior dTS', 'league mean')

    print('\nunivariate correlation with next-season dTS (weighted):')
    y, w = d['dts_next'].values, d['poss_next'].values.astype(float)
    rows = []
    for c in ['dts_prev'] + GEOM + EFFORT + OUTCOME:
        v = d[c].values.astype(float)
        m = ~np.isnan(v)
        if m.sum() < 50:
            continue
        vv = v[m] - np.average(v[m], weights=w[m])
        yy = y[m] - np.average(y[m], weights=w[m])
        r = np.average(vv * yy, weights=w[m]) / np.sqrt(
            np.average(vv ** 2, weights=w[m]) * np.average(yy ** 2, weights=w[m]))
        rows.append((c, r))
    for c, r in sorted(rows, key=lambda x: -abs(x[1])):
        print(f'  {c:24s} {r:+.3f}')
