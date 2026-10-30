"""Product C prototype: EPM predictive-ledger zTOV prior.

Builds a fully forward-looking version of the rZTOV excess from date-gated
EPM predictive skills and tests it as a ridge prior in the Product B
harness (see research_prior_informed_tov_rapm.py and docs/TOV_PRIOR_RAPM.md).

Predictive ledger per player, as-of year y (to score season y+1):
- act_hat        = p_tov_100 (predicted committed TOV per 100)
- v_hat          = walk-forward volume map a + b*(p_fg2a_100 + p_fg3a_100)
                   + c*p_fta_100 onto the tracking TSA basis. The observed
                   self/assisted volumes count fouled non-FGA attempts
                   (shot_fixed_* files: TSA ~ FGA + 0.36*FTA, +8.5% on
                   average), so an FGA-only volume shorts foul-drawers.
- self/ast split = trailing decayed volume-weighted assisted share,
                   lightly shrunk to league (YoY r=0.93); league for cold starts
- pot_hat        = walk-forward season-level linear map AST/100 -> potential
                   assists/100 fit on the tracking passing export
                   (R2 0.951), applied to p_ast_100
- prices/centres = recovered exactly from the year-y rows of
                   results/rztov_seasons_2014_2026.csv (window ending y)
- excess_hat     = act_hat - allowance_hat, recentred over the EPM-covered
                   set weighted by p_mp_48; prior p_i = |kappa_y| * excess_hat

EPM snapshot rule: latest cache/epm snapshot requested strictly before the
scored season's first game (manifest date contract: predictions requested
for date D exclude games on D). Snapshots may include playoffs of season y;
that is legitimate pre-scored-season information the RS-only tracking excess
does not have.

Candidates (base solve fixed at the Product B frozen recipe a=3000 h=1500):
- C2-td5   plain TD ridge (reference)
- C3-mix   frozen Product B recipe, scale 0.5 (reference)
- C6-pz    pure predictive-ledger prior, scale on dev
- C7-ext   C3 where tracking-covered + predictive where EPM-only
- C8-blend lambda*mix + (1-lambda)*pz on both-covered, single-source kept

Fold accounting: the 2025/2026 frozen folds were opened once for the C3
headline (2026-08-15). This run selects on dev folds 2019-2024 and reports
frozen folds descriptively; the untouched confirmation season is 2026-27.

Research lane only. Date-gated p_* skills follow the sanctioned EPM12
pattern; eWINS is forbidden; nothing here feeds production solves, the
published rZTOV surface, or VPM inputs.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from research_prior_informed_tov_rapm import (
    RES, ROOT, DEV_SCORE, FROZEN_SCORE, KAPPA, FIXED_BASE, BOOT_DRAWS, SEED,
    load_seasons, active_cols, gram, solve, score, mse, build_mix_prior)

EPM_DIR = ROOT / 'cache' / 'epm'
PASSES = ROOT.parent / 'geniussports' / 'passes_more_rs.csv'
EPM_COLS = ['player_id', 'p_ast_100', 'p_tov_100',
            'p_fg2a_100', 'p_fg3a_100', 'p_fta_100', 'p_mp_48']

PZ_SCALES = [0.25, 0.5, 0.75, 1.0, 1.25]
EXT_SCALES = [0.25, 0.5, 0.75, 1.0, 1.25]
BLEND_LAMBDAS = [0.25, 0.5, 0.75]
BLEND_SCALES = [0.5, 0.75, 1.0, 1.25]
MIX_SCALE = 0.5                      # frozen Product B C3 scale


def epm_files():
    out = []
    for f in EPM_DIR.glob('epm_all_*_go0.csv'):
        m = re.match(r'epm_all_(\d{4}-\d{2}-\d{2})_go0\.csv', f.name)
        if m:
            out.append((m.group(1), f))
    return sorted(out)


def read_epm(path):
    d = pd.read_csv(path, usecols=EPM_COLS)
    d = d.groupby('player_id', as_index=False).mean()
    return d[(d['p_mp_48'] > 0) & d[EPM_COLS[1:]].notna().all(axis=1)]


def epm_snapshot(first_scored_day):
    """Latest snapshot requested strictly before the first scored game."""
    best = None
    for day_s, f in epm_files():
        day = np.datetime64(day_s, 'D').astype(np.int64)
        if day < first_scored_day:
            best = f
    assert best is not None, 'no EPM snapshot before scored season'
    return best.name, read_epm(best)


def season_end_snapshot(year):
    """Latest snapshot requested Apr-Sep of the season's end year, if any."""
    c = [f for d, f in epm_files() if f'{year}-04-01' <= d <= f'{year}-09-30']
    return read_epm(c[-1]) if c else None


def vol_map(sea, asof_year):
    """Walk-forward map: EPM predicted FGA/FTA per 100 -> tracking TSA-basis
    self+assisted attempt volume (fouled non-FGA attempts included).
    Fit on season-end snapshots joined by nba_id, seasons <= as-of year,
    possession-weighted."""
    parts = []
    for y in range(2014, asof_year + 1):
        e = season_end_snapshot(y)
        if e is None:
            continue
        s = sea.loc[sea['year'] == y, ['nba_id', 'v_self', 'v_ast', 'off_poss']]
        parts.append(s.merge(e, left_on='nba_id', right_on='player_id'))
    d = pd.concat(parts)
    A = np.column_stack([np.ones(len(d)), d['p_fg2a_100'] + d['p_fg3a_100'],
                         d['p_fta_100']])
    w = np.sqrt(d['off_poss'].to_numpy(float))
    coef, *_ = np.linalg.lstsq(A * w[:, None],
                               (d['v_self'] + d['v_ast']).to_numpy() * w,
                               rcond=None)
    return coef, len(d)


def load_passes():
    pm = pd.read_csv(PASSES)
    pm['end_year'] = pm['Season'].str[:4].astype(int) + 1
    pm['ast100'] = pm['Passes Per 100 Possessions'] * pm['% Assist'] / 100.0
    pm['pot100'] = pm['Potential Assists Per 100 Possessions']
    pm = pm[(pm['Passes'] >= 500) & pm['ast100'].notna() & pm['pot100'].notna()]
    return pm[['end_year', 'ast100', 'pot100']]


def ast_map(pm, max_year):
    """Walk-forward season-level AST/100 -> potential AST/100 linear map."""
    d = pm[pm['end_year'] <= max_year]
    A = np.column_stack([np.ones(len(d)), d['ast100']])
    (a, b), *_ = np.linalg.lstsq(A, d['pot100'].to_numpy(), rcond=None)
    return a, b


def recover_prices(sea, year):
    """Exact lane prices + centres from the year's priced season rows."""
    d = sea[sea['year'] == year]
    A = np.column_stack([d['v_self'], d['v_ast'], np.ones(len(d))])
    x, *_ = np.linalg.lstsq(A, d['sc_allow'].to_numpy(), rcond=None)
    B = np.column_stack([d['v_potast'], np.ones(len(d))])
    z, *_ = np.linalg.lstsq(B, d['bp_allow'].to_numpy(), rcond=None)
    resid = max(np.abs(A @ x - d['sc_allow']).max(),
                np.abs(B @ z - d['bp_allow']).max())
    assert resid < 5e-3, f'price recovery not exact ({resid:.2e})'
    return {'be_self': x[0], 'be_ast': x[1], 'c_sc': x[2],
            'be_pot': z[0], 'c_bp': z[1]}


def trailing_share(sea, asof_year, half_life):
    """Decayed volume-weighted assisted share per player, lightly shrunk."""
    s = sea[sea['year'] <= asof_year].copy()
    s['w'] = s['off_poss'] * 0.5 ** ((asof_year - s['year']) * 365.0 / half_life)
    s['num'] = s['w'] * s['v_ast']
    s['den'] = s['w'] * (s['v_self'] + s['v_ast'])
    g = s.groupby('nba_id')[['num', 'den']].sum()
    league = g['num'].sum() / g['den'].sum()
    k = 0.1 * g['den'].median()
    share = ((g['num'] + k * league) / (g['den'] + k)).to_dict()
    return share, league


def build_pz_prior(sea, pm, blocks, sy, pidx, P):
    """|kappa| * recentred EPM predictive-ledger excess, offense side."""
    yr = sy - 1
    fname, e = epm_snapshot(int(blocks[sy]['days'].min()))
    pr = recover_prices(sea, yr)
    a, b = ast_map(pm, yr)
    (va, vb, vc), n_vol = vol_map(sea, yr)
    share, league = trailing_share(sea, yr, FIXED_BASE[1])

    pid = e['player_id'].astype(int).to_numpy()
    keep = np.array([p in pidx for p in pid])
    e, pid = e[keep], pid[keep]
    v_hat = np.clip(va + vb * (e['p_fg2a_100'] + e['p_fg3a_100'])
                    + vc * e['p_fta_100'], 0, None).to_numpy()
    sh = np.array([share.get(p, league) for p in pid])
    pot_hat = np.clip(a + b * e['p_ast_100'].to_numpy(), 0, None)
    allow = (pr['be_self'] * (1 - sh) * v_hat + pr['be_ast'] * sh * v_hat
             + pr['c_sc'] + pr['be_pot'] * pot_hat + pr['c_bp'])
    excess = e['p_tov_100'].to_numpy() - allow
    w = e['p_mp_48'].to_numpy()
    excess = excess - np.average(excess, weights=w)

    p = np.zeros(2 * P)
    for pp, ex in zip(pid, excess):
        p[pidx[pp]] = KAPPA[yr] * ex
    diag = (f'vol map n={n_vol} tsa={va:+.2f}+{vb:.3f}fga+{vc:.3f}fta  '
            f'ast map pot={a:+.2f}+{b:.3f}ast')
    return p, fname, len(pid), diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boot-draws', type=int, default=BOOT_DRAWS)
    args = ap.parse_args()
    A2, H2 = FIXED_BASE

    print('loading possession blocks 2014-2026 (RS)...', flush=True)
    blocks, pidx, P = load_seasons()
    sea = pd.read_csv(RES / 'rztov_seasons_2014_2026.csv')
    pm = load_passes()

    score_years = DEV_SCORE + FROZEN_SCORE
    fold_asof = {sy: sy - 1 for sy in score_years}
    mu = {sy: blocks[fold_asof[sy]]['y'].mean() for sy in score_years}

    print('building priors + grams per fold...', flush=True)
    grams, acts, mixp, pzp = {}, {}, {}, {}
    for sy in score_years:
        ac = active_cols(blocks, fold_asof[sy], P)
        acts[sy] = ac
        grams[sy] = gram(blocks, fold_asof[sy], H2, ac)
        mixp[sy], _ = build_mix_prior(sea, fold_asof[sy], H2,
                                      KAPPA[fold_asof[sy]], pidx, P)
        pzp[sy], fname, n_epm, diag = build_pz_prior(sea, pm, blocks, sy,
                                                     pidx, P)
        both = (mixp[sy] != 0) & (pzp[sy] != 0)
        pz_only = (pzp[sy] != 0) & (mixp[sy] == 0)
        corr = np.corrcoef(mixp[sy][both], pzp[sy][both])[0, 1]
        print(f'  as-of {fold_asof[sy]}: {fname}  epm {n_epm}  '
              f'both {both.sum()}  pz-only {pz_only.sum()}  '
              f'corr(obs,pred) {corr:+.3f}\n      {diag}', flush=True)

    def run(sy, prior_vec):
        ac = acts[sy]
        G, r = grams[sy]
        b = np.zeros(2 * P)
        prior = None if prior_vec is None else prior_vec[ac]
        b[ac] = solve(G, r, A2, prior=prior)
        err, games = score(blocks, sy, b, mu[sy])
        return err, games

    def dev_total(prior_fn):
        return sum(mse(run(sy, prior_fn(sy))[0]) for sy in DEV_SCORE)

    def blend_vec(sy, lam):
        m, z = mixp[sy], pzp[sy]
        out = np.where(m != 0, m, z)
        both = (m != 0) & (z != 0)
        out[both] = lam * m[both] + (1 - lam) * z[both]
        return out

    def ext_vec(sy, s6):
        out = MIX_SCALE * mixp[sy].copy()
        pz_only = (pzp[sy] != 0) & (mixp[sy] == 0)
        out[pz_only] = s6 * pzp[sy][pz_only]
        return out

    # ---------------- dev selection --------------------------------------
    c2_dev = dev_total(lambda sy: None)
    c3_dev = dev_total(lambda sy: MIX_SCALE * mixp[sy])
    c6_tot = {s: dev_total(lambda sy, s=s: s * pzp[sy]) for s in PZ_SCALES}
    S6 = min(c6_tot, key=c6_tot.get)
    c7_tot = {s: dev_total(lambda sy, s=s: ext_vec(sy, s)) for s in EXT_SCALES}
    S7 = min(c7_tot, key=c7_tot.get)
    c8_tot = {(lam, s): dev_total(lambda sy, lam=lam, s=s:
                                  s * blend_vec(sy, lam))
              for lam in BLEND_LAMBDAS for s in BLEND_SCALES}
    (L8, S8) = min(c8_tot, key=c8_tot.get)

    n = len(DEV_SCORE)
    print(f'\ndev means (2019-2024):')
    print(f'  C2-td5    {c2_dev / n:.6f}')
    print(f'  C3-mix    {c3_dev / n:.6f} (scale {MIX_SCALE})')
    for s in PZ_SCALES:
        tag = ' <- selected' if s == S6 else ''
        print(f'  C6-pz     {c6_tot[s] / n:.6f} (scale {s}){tag}')
    for s in EXT_SCALES:
        tag = ' <- selected' if s == S7 else ''
        print(f'  C7-ext    {c7_tot[s] / n:.6f} (ext scale {s}){tag}')
    for (lam, s), v in sorted(c8_tot.items()):
        tag = ' <- selected' if (lam, s) == (L8, S8) else ''
        print(f'  C8-blend  {v / n:.6f} (lambda {lam}, scale {s}){tag}')

    # ---------------- frozen report (opened-once, descriptive) -----------
    CANDS = {'C2-td5': lambda sy: None,
             'C3-mix': lambda sy: MIX_SCALE * mixp[sy],
             'C6-pz': lambda sy: S6 * pzp[sy],
             'C7-ext': lambda sy: ext_vec(sy, S7),
             'C8-blend': lambda sy: S8 * blend_vec(sy, L8)}
    rng = np.random.default_rng(SEED)
    rows, frozen_err = [], {}
    print('\nfrozen folds (opened once on 2026-08-15; descriptive report):',
          FROZEN_SCORE, flush=True)
    for sy in FROZEN_SCORE:
        errs = {}
        for name, fn in CANDS.items():
            err, games = run(sy, fn(sy))
            errs[name] = err
            rows.append({'fold': sy, 'cand': name, 'mse': mse(err),
                         'n': len(err)})
            print(f'  {sy} {name:<9} MSE {mse(err):.6f}')
        frozen_err[sy] = (errs, games)

    PAIRS = [('C6-pz', 'C2-td5'), ('C6-pz', 'C3-mix'), ('C7-ext', 'C3-mix'),
             ('C7-ext', 'C2-td5'), ('C8-blend', 'C3-mix'),
             ('C8-blend', 'C2-td5'), ('C8-blend', 'C6-pz')]
    print('\npaired deltas, e-6 (negative = first candidate better), '
          'game-clustered bootstrap 5-95%:')
    for sy in FROZEN_SCORE + ['pooled']:
        if sy == 'pooled':
            parts = [frozen_err[s] for s in FROZEN_SCORE]
            games = np.concatenate(
                [np.char.add(str(s) + '_', g[1].astype(str))
                 for s, g in zip(FROZEN_SCORE, parts)])
            errs = {nm: np.concatenate([p[0][nm] for p in parts])
                    for nm in CANDS}
        else:
            errs, games = frozen_err[sy]
        ug, inv = np.unique(games, return_inverse=True)
        counts = np.bincount(inv)
        per_game = {nm: np.bincount(inv, weights=errs[nm] * errs[nm])
                    for nm in CANDS}
        print(f'  fold {sy}:')
        for anm, bnm in PAIRS:
            d_pt = (np.sum(per_game[anm]) - np.sum(per_game[bnm])) / len(games)
            draws = rng.integers(0, len(ug), (args.boot_draws, len(ug)))
            da = per_game[anm][draws].sum(axis=1)
            db = per_game[bnm][draws].sum(axis=1)
            nn = counts[draws].sum(axis=1)
            dd = (da - db) / nn
            lo, hi = np.percentile(dd, [5, 95])
            rows.append({'fold': sy, 'cand': f'{anm} vs {bnm}',
                         'mse': d_pt, 'n': len(games),
                         'lo': lo, 'hi': hi})
            print(f'    {anm:<9} vs {bnm:<8} {d_pt * 1e6:+8.2f}  '
                  f'[{lo * 1e6:+.2f},{hi * 1e6:+.2f}]  '
                  f'P(better) {np.mean(dd < 0):.2f}')

    out = RES / 'tov_predictive_ztov_folds.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f'\nwrote {out}')
    print('config:', {'alpha': A2, 'half_life': H2, 'C6_scale': S6,
                      'C7_ext_scale': S7, 'C8_lambda': L8, 'C8_scale': S8})


if __name__ == '__main__':
    sys.exit(main())
