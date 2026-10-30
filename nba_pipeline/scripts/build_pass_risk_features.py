"""Pass-risk features for testing a two-level pass lane against bpTOV / oTOV RAPM.

The rZTOV model prices one pass exposure: potential assists. But a turnover can
happen on a pass that was never going to produce a shot -- the post entry, the
outlet, the swing that resets -- and those passes carry no denominator at all.
Tracking says potential assists are ~16% of pass volume and carry ~80% of pass
turnovers, so the current exposure is not wrong about where the risk is, it just
charges nothing for a lane that should carry a small nonzero price.

Feature engineering here is deliberately narrow. Three rules:

  1. Exposures are DISJOINT and never contain the regression numerator. A pass
     that turns over produces no shot, so every pass turnover sits inside the
     outermost shell; `v_other` subtracts them, otherwise the fit is partly
     regressing y on y.
  2. Context enters as `exposure x centered context`, never as a bare column, so
     it shifts the PRICE of a lane rather than adding a free level term. That is
     what keeps `load x rate x price = impact` checkable in the display.
  3. Anything computed from the shot that followed a pass (qSQ, qSP, % assist)
     is outcome selection -- a turned-over pass has no value for it -- and stays
     out of the predictor set. Benchmark only.

Targets are the 22-26 window, all point-valued and mutually consistent:
  bp_rapm   oTOV_bp   bad-pass component of the offensive turnover factor
  bp_alt    alt_badpass_tov_value, an independent standalone solve
  otov      oTOV      the headline offensive turnover factor
  sc_rapm   oTOV_sc   scoring-turnover complement, carried as a control

Writes `results/pass_risk_features_{lo}_{hi}.csv` (a gitignored artifact) and
prints the nested model comparison.
"""
import argparse

import numpy as np
import pandas as pd

from build_rztov_explorer import (RES, TRACK, PANEL, norm_name, tov_counts,
                                  FIRST_YEAR, LAST_YEAR)

WF = ('/Users/russ/Documents/Codex/basketball/nba/research/rapm/nba_pipeline/'
      'master_results/weighted_factors_22_26_all_rb_se_a2000_4000.csv')
BP_ALT = f'{RES}/alt_badpass_tov_value_22_26_all_rb_se_a2000_4000_results.csv'

PASS = 'pass_rs.csv'
PASS_MORE = 'passes_more_rs.csv'

# Per-100 exposures. Disjoint by construction; none contains a turnover.
EXPOSURES = ['v_self', 'v_ast', 'v_pot', 'v_other']
# Season-mean context, centered before interaction.
CONTEXT = {'dist': 'Avg. Completed Pass Distance (ft)',
           'pspace': 'Avg. Closest Passer Defender Distance (ft)',
           'rspace': 'Avg. Closest Receiver Defender Distance (ft)'}
# Overlapping mix descriptors. Candidates, not exposures -- they sit inside the
# lanes above and are only ever tested as additions.
MIX = ['v_postentry', 'v_kickout', 'v_secondary']


def load_track(fname, cols):
    g = pd.read_csv(f'{TRACK}/{fname}')
    g.columns = [c.strip() for c in g.columns]
    g = g[g['Season'].astype(str).str.match(r'\d{4}-\d{2}')].copy()
    g['year'] = g['Season'].str[:4].astype(int) + 1
    g['norm'] = g['Passer'].map(norm_name)
    return g[['norm', 'year'] + list(cols)].rename(columns=cols).drop_duplicates(
        ['norm', 'year'])


def shots(fname, col):
    g = pd.read_csv(f'{TRACK}/{fname}')
    g.columns = [c.strip() for c in g.columns]
    g = g[g['Season'].astype(str).str.match(r'\d{4}-\d{2}')].copy()
    g['year'] = g['Season'].str[:4].astype(int) + 1
    g['norm'] = g['Shooter'].map(norm_name)
    return (g[['norm', 'year', 'Shots Attempted Per 100 Possessions']]
            .rename(columns={'Shots Attempted Per 100 Possessions': col})
            .drop_duplicates(['norm', 'year']))


def build_seasons():
    """One row per player-season with disjoint exposures and centered context."""
    p = load_track(PASS, {'Passes Per 100 Possessions': 'v_pass',
                          'Potential Assists Per 100 Possessions': 'v_pot',
                          'Passes Leading to Shot Per 100 Possessions': 'v_lead',
                          'Secondary Assists Per 100 Possessions': 'v_secondary',
                          '% Turned Over': 'pct_to'})
    m = load_track(PASS_MORE, {'Post Entry Per 100 Possessions': 'v_postentry',
                               'Kickout Per 100 Possessions': 'v_kickout',
                               **{v: k for k, v in CONTEXT.items()}})
    trk = (p.merge(m, on=['norm', 'year'], how='left')
           .merge(shots('shot_fixed_selfcreated_rs.csv', 'v_self'),
                  on=['norm', 'year'], how='outer')
           .merge(shots('shot_fixed_assisted_rs.csv', 'v_ast'),
                  on=['norm', 'year'], how='outer'))
    trk['matched'] = 1

    panel = pd.read_csv(PANEL, usecols=['nba_id', 'year', 'Name', 'playoffs', 'OffPoss'],
                        low_memory=False)
    panel = panel[(panel['playoffs'] == 0) & panel['OffPoss'].gt(0)].copy()
    panel['nba_id'] = panel['nba_id'].astype('int64')
    panel['norm'] = panel['Name'].map(norm_name)
    panel.loc[panel['norm'] == 'enes kanter', 'norm'] = 'enes freedom'
    panel.loc[(panel['norm'] == 'kenyon martin') & (panel['year'] >= 2020), 'norm'] = 'kj martin'
    panel.loc[panel['norm'] == 'juan hernangomez', 'norm'] = 'juancho hernangomez'

    cnt = pd.concat([tov_counts(y) for y in range(FIRST_YEAR, LAST_YEAR + 1)],
                    ignore_index=True)
    d = panel.merge(cnt, on=['nba_id', 'year'], how='left')
    for c in ('n_tov', 'n_bp'):
        d[c] = d[c].fillna(0)
    d = d.merge(trk, on=['norm', 'year'], how='left')
    d = d[d['matched'].notna()].copy()

    # Actuals per 100, from PBP. The bad-pass count is the regression numerator.
    d['act_bp'] = d['n_bp'] / d['OffPoss'] * 100
    d['act_sc'] = (d['n_tov'] - d['n_bp']) / d['OffPoss'] * 100
    d['act_all'] = d['n_tov'] / d['OffPoss'] * 100
    # Tracking-side pass turnovers, used ONLY to make v_other disjoint.
    d['trk_pass_tov'] = d['v_pass'] * d['pct_to'] / 100

    for c in ['v_pass', 'v_pot', 'v_lead', 'v_self', 'v_ast'] + MIX:
        d[c] = d[c].fillna(0)
    # Rule 1: connective lane excludes creation passes AND the turnovers themselves.
    d['v_other'] = (d['v_pass'] - d['v_pot'] - d['trk_pass_tov']).clip(lower=0)
    return d


def window(d, lo, hi, min_poss):
    """Possession-weighted aggregate over the target window."""
    g = d[(d['year'] >= lo) & (d['year'] <= hi) & d[list(CONTEXT)].notna().all(axis=1)]
    cols = EXPOSURES + MIX + list(CONTEXT) + ['v_pass', 'act_bp', 'act_sc', 'act_all',
                                              'trk_pass_tov']
    w = g.groupby('nba_id').apply(
        lambda x: pd.Series({'poss': x['OffPoss'].sum(),
                             **{c: np.average(x[c], weights=x['OffPoss']) for c in cols}}),
        include_groups=False).reset_index()
    return w[w['poss'] >= min_poss].copy()


def add_interactions(w):
    """Rule 2: context shifts a lane's price, so it enters multiplied by exposure."""
    for k in CONTEXT:
        w[f'c_{k}'] = w[k] - np.average(w[k], weights=w['poss'])
        w[f'pot_{k}'] = w['v_pot'] * w[f'c_{k}']
        w[f'oth_{k}'] = w['v_other'] * w[f'c_{k}']
    return w


def targets(w):
    wf = pd.read_csv(WF, usecols=['player_id', 'player_name', 'oTOV', 'oTOV_bp', 'oTOV_sc'])
    wf = wf.rename(columns={'player_id': 'nba_id', 'player_name': 'name',
                            'oTOV': 'otov', 'oTOV_bp': 'bp_rapm', 'oTOV_sc': 'sc_rapm'})
    alt = pd.read_csv(BP_ALT, usecols=['player_id', 'off']).rename(
        columns={'player_id': 'nba_id', 'off': 'bp_alt'})
    return w.merge(wf, on='nba_id', how='inner').merge(alt, on='nba_id', how='left')


def fit(w, target, act, cols):
    """rapm ~ kappa*(act - sum be_j v_j), no intercept. be_j = -b_j / kappa."""
    X = w[[act] + cols].values
    y = w[target].values
    sw = w['poss'].values
    b = np.linalg.solve(X.T @ (sw[:, None] * X), X.T @ (sw * y))
    resid = y - X @ b
    r2 = 1 - (sw * resid ** 2).sum() / (sw * (y - np.average(y, weights=sw)) ** 2).sum()
    kappa = b[0]
    return {'kappa': kappa, 'r2': r2,
            'be': {c: -bj / kappa for c, bj in zip(cols, b[1:])}, 'resid': resid}


def tilt(w, resid, by='v_pot'):
    """Residual slope across exposure quintiles -- the diagnostic that motivated this."""
    q = pd.qcut(w[by], 5, labels=False, duplicates='drop')
    m = pd.Series(resid).groupby(q).apply(
        lambda s: np.average(s, weights=w['poss'].values[s.index]))
    return float(m.iloc[-1] - m.iloc[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lo', type=int, default=2022)
    ap.add_argument('--hi', type=int, default=2026)
    ap.add_argument('--min-poss', type=int, default=2000)
    a = ap.parse_args()

    d = build_seasons()
    w = targets(add_interactions(window(d, a.lo, a.hi, a.min_poss)))
    out = f'{RES}/pass_risk_features_{a.lo % 100}_{a.hi % 100}.csv'
    w.to_csv(out, index=False)

    print(f'{len(w)} players, {a.lo}-{a.hi}, >= {a.min_poss} off poss\n')
    print('cross-source check (tracking pass turnovers vs PBP bad-pass count, per 100)')
    r = np.corrcoef(w['trk_pass_tov'], w['act_bp'])[0, 1]
    print(f'  tracking {np.average(w.trk_pass_tov, weights=w.poss):.3f}   '
          f'PBP {np.average(w.act_bp, weights=w.poss):.3f}   r = {r:+.3f}\n')

    print('league means per 100:  ' + '   '.join(
        f'{c} {np.average(w[c], weights=w.poss):6.2f}' for c in EXPOSURES + ['v_pass']))
    print(f'  potential assists are {100 * np.average(w.v_pot, weights=w.poss) / np.average(w.v_pass, weights=w.poss):.1f}%'
          f' of pass volume\n')

    print('=' * 78)
    print('TEST 1  bad-pass turnover RAPM, pass lane only')
    print('=' * 78)
    for lbl, cols in [
            ('A  potential assists only  (current)', ['v_pot']),
            ('B  + connective lane', ['v_pot', 'v_other']),
            ('C  + distance price modifier', ['v_pot', 'v_other', 'pot_dist']),
            ('D  + defender space modifiers', ['v_pot', 'v_other', 'pot_dist',
                                               'pot_pspace', 'pot_rspace']),
            ('E  + post entry / kickout mix', ['v_pot', 'v_other', 'pot_dist',
                                               'v_postentry', 'v_kickout'])]:
        for tgt in ('bp_rapm', 'bp_alt'):
            f = fit(w.dropna(subset=[tgt]), tgt, 'act_bp', cols)
            ww = w.dropna(subset=[tgt])
            print(f'{lbl if tgt == "bp_rapm" else "":<38} {tgt:<8} '
                  f'R2 {f["r2"]:6.3f}  k {f["kappa"]:+6.3f}  tilt {tilt(ww, f["resid"]):+6.3f}  '
                  + '  '.join(f'{k} {100 * v:+6.2f}' for k, v in f['be'].items()))
        print()

    print('=' * 78)
    print('TEST 2  offensive turnover RAPM, full model')
    print('=' * 78)
    for lbl, cols in [
            ('A  three exposures  (current)', ['v_self', 'v_ast', 'v_pot']),
            ('B  + connective lane', EXPOSURES),
            ('C  + distance price modifier', EXPOSURES + ['pot_dist']),
            ('D  + both lanes shifted by distance', EXPOSURES + ['pot_dist', 'oth_dist'])]:
        f = fit(w, 'otov', 'act_all', cols)
        print(f'{lbl:<38} R2 {f["r2"]:6.3f}  k {f["kappa"]:+6.3f}  '
              f'tilt {tilt(w, f["resid"]):+6.3f}')
        print(f'{"":<38} ' + '  '.join(f'{k} {100 * v:+6.2f}' for k, v in f['be'].items()))
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
