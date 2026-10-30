"""rZTOV: a mix-adjusted turnover stat, per season, plus the explorer payload.

Involvement buys a turnover allowance, split into the two lanes turnovers
actually happen in. Self-created and potential-assisted attempts buy a
scoring-turnover allowance; potential assists buy a bad-pass allowance; and

    rZTOV = (actual turnovers - allowance) / plays, per 100 offensive possessions

is what a player commits above or below what that load buys. Negative is good.

The three break-even rates are one shared price vector solved jointly against
three count-basis RAPM solves at once -- the scoring-turnover lane, the
bad-pass lane, and total turnovers -- so each lane's prices are anchored to
what that lane's own RAPM tolerates while the total equation ties the vector
to total turnover impact (August 2026 change; the previous single unrestricted
fit on total TOV RAPM let the self-created price absorb bad-pass credit it
could not decompose). Kappa still has a known physical value of -1.00 on each
equation (one extra turnover committed is one turnover not prevented) and the
fit can be checked against it.

The prices also move across the tracking era: self-created rises, assisted and
passing fall, only partly tracked by the league turnover rate. One frozen set
would be several points wrong on the self-created term a decade back, so each
season is scored on break-evens fit to the trailing five-season window ending
that year. Every year is then true to that year and the current season stays
live.

Two adjustments, not one:
  1. break-evens refit per trailing window, so the rates move with the league;
  2. each lane of each season re-centered on its own possession-weighted mean,
     so league turnovers saved sums to zero inside both lanes every year and
     seasons share a scale.

Writes the explorer JSON, `results/rztov_seasons_{first}_{last}.csv` (the stat),
and `results/rztov_research_{window}.csv` (the aggregate research surface).
"""
import argparse
import json
import os
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from build_historical_scoring_security import fit_turnover_point_prices

RAPM = '/Users/russ/Documents/Codex/basketball/nba/research/rapm'
TRACK = f'{RAPM}/geniussports'
PANEL = '/Users/russ/Documents/Codex/basketball/nba/research/vpm/boxscore.csv'
RES = f'{RAPM}/nba_pipeline/results'
PUTBACK_CACHE = f'{RES}/rztov_putbacks.csv'

# Count-basis offensive TOV RAPM solves, season effects on, three per trailing
# window: the total plus its two physical lanes. Build any missing one with
#   rapm.py {TOV,SCORING_TOV,BADPASS_TOV} LO HI RS --season-effects \
#       --off-alpha 1500 --def-alpha 1500
def tov_target(lo, hi):
    return f'{RES}/tov_{lo - 2000}_{hi - 2000}_rs_se_a1500_1500_results.csv'


def sc_target(lo, hi):
    return f'{RES}/scoring_tov_{lo - 2000}_{hi - 2000}_rs_se_a1500_1500_results.csv'


def bp_target(lo, hi):
    return f'{RES}/badpass_tov_{lo - 2000}_{hi - 2000}_rs_se_a1500_1500_results.csv'


# Point-valued net-continuation lane solves, used only to price an excess turnover in
# points. Each lane of each season is priced by a through-origin fit of the window's
# offensive lane RAPM on rolled-up turnovers saved in that lane -- the identical fit the
# historical scoring-security board uses, so the two surfaces carry the same points.
COMPONENT_DIR = f'{RES}/tov_price_targets'  # low-alpha price targets; see build_tov_price_targets.py
PRICE_DRAWS, PRICE_SEED = 200, 37

FIRST_YEAR, LAST_YEAR = 2014, 2026
WINDOW = 5
FIRST_END = FIRST_YEAR + WINDOW - 1   # first season with a full trailing window
BOARD_FROM, BOARD_TO = LAST_YEAR - WINDOW + 1, LAST_YEAR

# Published point-valued offensive turnover factor over the board window, carried as an
# outside anchor on the board.
OTOV_BOARD = (f'{RAPM}/nba_pipeline/master_results/weighted_factors_'
              f'{BOARD_FROM - 2000}_{BOARD_TO - 2000}_all_rb_se_a2000_4000.csv')

FIT_MIN_POSS = 2000         # players entering a window's break-even fit
SEASON_MIN_POSS = 500       # season rows published at all
QUAL_MIN_POSS = 1000        # rZTOV reported as a rate
QUAL_MIN_INVOLVE = 10       # ... and only above a real shot/passing load
YOY_MIN_POSS = 1500         # both sides of an adjacent-season pair

SUFFIX = {'jr', 'sr', 'ii', 'iii', 'iv', 'v'}
AREAS = ('self', 'ast', 'potast')


def norm_name(s):
    s = unicodedata.normalize('NFD', str(s))
    s = ''.join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[.'’]", '', s.lower())
    s = re.sub(r'-', ' ', s)
    return ' '.join(t for t in s.split() if t not in SUFFIX)


def wls(y, X, w, intercept=True):
    """Weighted least squares. Returns (beta, weighted R2, correlation)."""
    y = np.asarray(y, float)
    w = np.asarray(w, float)
    cols = [np.ones(len(y))] if intercept else []
    Xd = np.column_stack(cols + [np.asarray(X[c], float) for c in X.columns])
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(Xd * sw[:, None], y * sw, rcond=None)
    fit = Xd @ beta
    resid = y - fit
    r2 = 1 - (resid ** 2 * w).sum() / (((y - np.average(y, weights=w)) ** 2 * w).sum())
    return beta, r2, float(np.corrcoef(fit, y)[0, 1])


def raw_path(year):
    return f'{RAPM}/nba_pipeline/raw_data/NBA{year % 100:02d}.parquet'


def tov_counts(year):
    """Player-attributed turnovers from raw PBP, split bad-pass vs scoring."""
    d = pd.read_parquet(raw_path(year),
                        columns=['event_type', 'player1_id', 'home_description',
                                 'visitor_description', 'neutral_description'])
    et = pd.to_numeric(d['event_type'], errors='coerce')
    p1 = pd.to_numeric(d['player1_id'], errors='coerce').fillna(0).astype('int64')
    desc = (d['home_description'].fillna('') + ' ' + d['visitor_description'].fillna('')
            + ' ' + d['neutral_description'].fillna(''))
    ok = (p1 > 0) & (p1 < 1_000_000_000)
    is_tov = ok & (et == 5) & ~desc.str.contains('No Turnover', case=False)
    is_bp = is_tov & desc.str.contains('Bad Pass', case=False)
    out = pd.DataFrame({'nba_id': p1, 'n_tov': is_tov.astype(int), 'n_bp': is_bp.astype(int)})
    out = out[ok].groupby('nba_id', as_index=False).sum()
    out['n_sc'] = out['n_tov'] - out['n_bp']
    out['year'] = year
    return out


def putback_counts(year):
    """FGAs taken directly off the shooter's own offensive rebound.

    A rebound and a field-goal attempt by the same player on consecutive events is
    necessarily an offensive rebound followed by a putback, so the pattern needs no
    team-side inference. Modern raw PBP is ordered clock-first, not event-number-first.
    """
    d = pd.read_parquet(raw_path(year),
                        columns=['game_id', 'event_num', 'event_type', 'period',
                                 'minute_remaining_quarter', 'seconds_remaining_quarter',
                                 'player1_id'])
    d['game_id'] = d['game_id'].astype(str).str.zfill(10).str[-10:]
    d['clock'] = (pd.to_numeric(d['minute_remaining_quarter'], errors='coerce').fillna(0) * 60
                  + pd.to_numeric(d['seconds_remaining_quarter'], errors='coerce').fillna(0))
    d['event_num'] = pd.to_numeric(d['event_num'], errors='coerce').fillna(0)
    d['event_type'] = pd.to_numeric(d['event_type'], errors='coerce')
    d['player1_id'] = pd.to_numeric(d['player1_id'], errors='coerce').fillna(0).astype('int64')
    d = d.sort_values(['game_id', 'period', 'clock', 'event_num'],
                      ascending=[True, True, False, True])

    same = (d['game_id'] == d['game_id'].shift()) & (d['period'] == d['period'].shift())
    is_pb = (same & d['event_type'].isin([1, 2])
             & (d['event_type'].shift() == 4)
             & (d['player1_id'] == d['player1_id'].shift())
             & (d['player1_id'] > 0))
    out = (d.loc[is_pb, ['player1_id']].assign(n_pb=1)
           .groupby('player1_id', as_index=False).sum()
           .rename(columns={'player1_id': 'nba_id'}))
    out['year'] = year
    return out


def load_putbacks(rebuild=False):
    if not rebuild and os.path.exists(PUTBACK_CACHE):
        pb = pd.read_csv(PUTBACK_CACHE)
        if set(range(FIRST_YEAR, LAST_YEAR + 1)) <= set(pb['year'].unique()):
            return pb
    pb = pd.concat([putback_counts(y) for y in range(FIRST_YEAR, LAST_YEAR + 1)],
                   ignore_index=True)
    pb.to_csv(PUTBACK_CACHE, index=False)
    return pb


def load_track(fname, actor, rename):
    g = pd.read_csv(f'{TRACK}/{fname}')
    g = g[g['Season'].str.match(r'\d{4}-\d{2}')].copy()
    g['year'] = g['Season'].str[:4].astype(int) + 1
    g['norm'] = g[actor].map(norm_name)
    return g[['norm', 'year'] + list(rename)].rename(columns=dict(rename))


def build_panel(rebuild_putbacks=False):
    cnt = pd.concat([tov_counts(y) for y in range(FIRST_YEAR, LAST_YEAR + 1)],
                    ignore_index=True)

    panel = pd.read_csv(PANEL, usecols=['nba_id', 'year', 'Name', 'playoffs', 'OffPoss'],
                        low_memory=False)
    panel = panel[(panel['playoffs'] == 0) & panel['OffPoss'].gt(0)].copy()
    panel['nba_id'] = panel['nba_id'].astype('int64')
    panel['norm'] = panel['Name'].map(norm_name)
    # Tracking-file name aliases: retroactive rename, father/son collision, accent variant.
    panel.loc[panel['norm'] == 'enes kanter', 'norm'] = 'enes freedom'
    panel.loc[(panel['norm'] == 'kenyon martin') & (panel['year'] >= 2020), 'norm'] = 'kj martin'
    panel.loc[panel['norm'] == 'juan hernangomez', 'norm'] = 'juancho hernangomez'

    # shot_fixed_* are the TSA-basis files (fouled attempts counted); shot_* are FGA-only.
    vol = (load_track('shot_fixed_selfcreated_rs.csv', 'Shooter',
                      {'Shots Attempted Per 100 Possessions': 'v_self'})
           .merge(load_track('shot_fixed_assisted_rs.csv', 'Shooter',
                             {'Shots Attempted Per 100 Possessions': 'v_ast'}),
                  on=['norm', 'year'], how='outer')
           .merge(load_track('pass_rs.csv', 'Passer',
                             {'Potential Assists Per 100 Possessions': 'v_potast'}),
                  on=['norm', 'year'], how='outer')
           .drop_duplicates(['norm', 'year']))
    vol['matched'] = 1

    df = panel.merge(cnt, on=['nba_id', 'year'], how='left')
    for c in ('n_tov', 'n_bp', 'n_sc'):
        df[c] = df[c].fillna(0)
    df = df.merge(vol, on=['norm', 'year'], how='left')
    df = df[df['matched'].notna() & (df['OffPoss'] >= SEASON_MIN_POSS)].copy()
    for c in ('v_self', 'v_ast', 'v_potast'):
        df[c] = df[c].fillna(0)

    df = df.merge(load_putbacks(rebuild_putbacks), on=['nba_id', 'year'], how='left')
    df['n_pb'] = df['n_pb'].fillna(0)
    # A putback is not a self-created look, so it is charged to the assisted bucket.
    # Putbacks are counted FGA-basis off raw PBP while v_self is TSA-basis, so for a
    # handful of low-minute bigs the putback count exceeds measured self-created
    # volume; clamp the transfer at what is actually there rather than going negative.
    v_pb = np.minimum(df['n_pb'] / df['OffPoss'] * 100, df['v_self'].clip(lower=0))
    df['v_self'] = df['v_self'] - v_pb
    df['v_ast'] = df['v_ast'] + v_pb

    df['sc_100'] = df['n_sc'] / df['OffPoss'] * 100
    df['bp_100'] = df['n_bp'] / df['OffPoss'] * 100
    df['act'] = df['sc_100'] + df['bp_100']
    df['involve'] = df['v_self'] + df['v_ast'] + df['v_potast']
    return df


def window_frame(df, lo, hi):
    """Possession-weighted player aggregate over one window."""
    w = df[df['year'].between(lo, hi)]
    g = w.groupby('nba_id').apply(lambda x: pd.Series({
        'poss': x['OffPoss'].sum(),
        'act': (x['n_sc'].sum() + x['n_bp'].sum()) / x['OffPoss'].sum() * 100,
        'sc_100': x['n_sc'].sum() / x['OffPoss'].sum() * 100,
        'bp_100': x['n_bp'].sum() / x['OffPoss'].sum() * 100,
        **{c: np.average(x[c], weights=x['OffPoss']) for c in ('v_self', 'v_ast', 'v_potast')},
    }), include_groups=False).reset_index()
    return g


def read_solve(path, col):
    t = pd.read_csv(path)
    return t[['player_id', 'off']].rename(columns={'player_id': 'nba_id', 'off': col})


def fit_window(df, lo, hi):
    """Joint two-lane break-evens from the window's three count-basis TOV solves.

    One shared price vector (self, ast, potast) is solved simultaneously against
    the scoring-turnover, bad-pass, and total turnover RAPM targets:

        sc_rapm  ~ k_sc  * (sc_100 - be_self*self - be_ast*ast)
        bp_rapm  ~ k_bp  * (bp_100 - be_pot*potast)
        tov_rapm ~ k_tot * (act    - all three)

    stacked with inverse-residual-variance equation weights and solved by
    alternating least squares: given prices, each lane's kappa and residual
    variance close in one line; given kappas, the three row blocks stack into
    one weighted lstsq for the prices. The scoring lane pins the two attempt
    prices to its own lane's RAPM, the passing lane pins the potential-assist
    price to its own, and the total equation keeps the vector honest against
    total turnover impact -- self-created volume relieves team-level bad-pass
    exposure, and only the total solve can see that. The previous single
    unrestricted fit on total TOV RAPM alone let strongly correlated volumes
    trade break-even weight (self-created and potential assists correlate about
    +0.7 among fit players), which bootstrap draws showed as offsetting
    self/pass wobble. Break-evens are ratios, so uniform ridge shrinkage of a
    target cancels out of them. All fits are through the origin, so league
    allowance lands on league actual within each equation.
    """
    g = window_frame(df, lo, hi)
    g = g[g['poss'] >= FIT_MIN_POSS]
    for path, col in ((sc_target(lo, hi), 'y_sc'), (bp_target(lo, hi), 'y_bp'),
                      (tov_target(lo, hi), 'y_tov')):
        g = g.merge(read_solve(path, col), on='nba_id')
    g = g.dropna(subset=['y_sc', 'y_bp', 'y_tov'])

    wt = g['poss'].to_numpy(float)
    sc, bp, act = (g[c].to_numpy(float) for c in ('sc_100', 'bp_100', 'act'))
    vs, va, vp = (g[c].to_numpy(float) for c in ('v_self', 'v_ast', 'v_potast'))
    y1, y2, y3 = (g[c].to_numpy(float) for c in ('y_sc', 'y_bp', 'y_tov'))
    z = np.zeros_like(sc)
    sw = np.sqrt(wt)

    def through_origin(y, X):
        return np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]

    # Lane-own anchors: what each lane's RAPM tolerates per unit of volume with
    # no total equation. These initialize the joint solve and are kept for the
    # method copy.
    bl1 = through_origin(y1, np.column_stack([sc, vs, va]))
    bl2 = through_origin(y2, np.column_stack([bp, vp]))
    lane = {'self': float(-bl1[1] / bl1[0]), 'ast': float(-bl1[2] / bl1[0]),
            'potast': float(-bl2[1] / bl2[0])}

    bs, ba, bpp = lane['self'], lane['ast'], lane['potast']
    for _ in range(300):
        al_sc, al_bp = bs * vs + ba * va, bpp * vp
        r1, r2r, r3 = sc - al_sc, bp - al_bp, act - al_sc - al_bp
        k1 = (wt * y1 * r1).sum() / (wt * r1 * r1).sum()
        k2 = (wt * y2 * r2r).sum() / (wt * r2r * r2r).sum()
        k3 = (wt * y3 * r3).sum() / (wt * r3 * r3).sum()
        s1 = (wt * (y1 - k1 * r1) ** 2).sum() / wt.sum()
        s2 = (wt * (y2 - k2 * r2r) ** 2).sum() / wt.sum()
        s3 = (wt * (y3 - k3 * r3) ** 2).sum() / wt.sum()
        X = np.vstack([np.column_stack([-k1 * vs, -k1 * va, z]),
                       np.column_stack([z, z, -k2 * vp]),
                       np.column_stack([-k3 * vs, -k3 * va, -k3 * vp])])
        yst = np.concatenate([y1 - k1 * sc, y2 - k2 * bp, y3 - k3 * act])
        sww = np.sqrt(np.concatenate([wt / s1, wt / s2, wt / s3]))
        prev = (bs, ba, bpp)
        bs, ba, bpp = np.linalg.lstsq(X * sww[:, None], yst * sww, rcond=None)[0]
        if max(abs(bs - prev[0]), abs(ba - prev[1]), abs(bpp - prev[2])) < 1e-10:
            break

    def excess_fit(y, e):
        k = float((wt * y * e).sum() / (wt * e * e).sum())
        fit = k * e
        r2 = 1 - ((y - fit) ** 2 * wt).sum() / (((y - np.average(y, weights=wt)) ** 2
                                                 * wt).sum())
        return k, float(r2), float(np.corrcoef(fit, y)[0, 1])

    al_sc, al_bp = bs * vs + ba * va, bpp * vp
    k_sc, r2_sc, _ = excess_fit(y1, sc - al_sc)
    k_bp, r2_bp, _ = excess_fit(y2, bp - al_bp)
    kappa, r2, r = excess_fit(y3, act - al_sc - al_bp)

    # Display-only allocation of the observed scoring-turnover bucket across the two
    # attempt types; raw PBP never names which one caused it.
    mix, _, _ = wls(g['sc_100'], g[['v_self', 'v_ast']], wt)

    return {'window': [lo, hi],
            'self': float(bs), 'ast': float(ba), 'potast': float(bpp),
            'kappa': kappa, 'k_sc': k_sc, 'k_bp': k_bp,
            'lane_self': lane['self'], 'lane_ast': lane['ast'],
            'lane_potast': lane['potast'],
            'r2': r2, 'r2_sc': r2_sc, 'r2_bp': r2_bp, 'r': r,
            'alloc_base': float(mix[0]),
            'alloc_self': float(mix[1]), 'alloc_ast': float(mix[2]),
            'n': int(len(g)), 'poss': float(wt.sum())}


def score_seasons(df, fits):
    """Apply each season's window break-evens, then center each lane of the season."""
    earliest, latest = min(fits), max(fits)
    rows = df.copy()
    src = rows['year'].clip(lower=earliest, upper=latest)
    rows['be_source'] = src
    for k in AREAS:
        rows[f'be_{k}'] = src.map(lambda y, k=k: fits[y][k])
        rows[f'allow_{k}'] = rows[f'be_{k}'] * rows[f'v_{k}']

    # Attempt volume buys scoring-turnover allowance; potential assists buy bad-pass
    # allowance. The window fit calibrates each lane across five seasons, so an
    # individual season inside it can sit slightly off. Re-center each lane on its own
    # season possession-weighted mean: league turnovers saved then sums to zero inside
    # both lanes every year (and therefore in total), and seasons share a scale.
    rows['sc_allow'] = rows['allow_self'] + rows['allow_ast']
    rows['bp_allow'] = rows['allow_potast']
    centres = {}
    for lane, com, al in (('sc', 'sc_100', 'sc_allow'), ('bp', 'bp_100', 'bp_allow')):
        c = rows.groupby('year').apply(
            lambda g, com=com, al=al: np.average(g[com] - g[al], weights=g['OffPoss']),
            include_groups=False)
        centres[lane] = c
        rows[f'centre_{lane}'] = rows['year'].map(c)
        rows[al] += rows[f'centre_{lane}']
        rows[f'{lane}_excess'] = rows[com] - rows[al]
    rows['centre'] = rows['centre_sc'] + rows['centre_bp']
    rows['allowance'] = rows['sc_allow'] + rows['bp_allow']
    rows['excess'] = rows['sc_excess'] + rows['bp_excess']

    # A play is an attempt, a potential assist, or a turnover. Turnovers belong in the
    # denominator they occurred on -- the same shape boxscore TOV% uses, TOV / (FGA +
    # 0.44 FTA + TOV) -- otherwise a turnover silently removes its own chance from the
    # rate. Allowance and excess are counts and are unaffected by the choice.
    rows['plays'] = rows['involve'] + rows['act']
    rows['rztov'] = rows['excess'] / rows['plays'].clip(lower=1) * 100
    rows['tov_saved_100'] = -rows['excess']
    rows['tov_saved'] = -rows['excess'] * rows['OffPoss'] / 100
    # rZTOV divides by plays, so a player with almost no shot or passing load produces a
    # huge rate off a trivial raw excess. Keep those rows but flag them.
    rows['qualified'] = ((rows['OffPoss'] >= QUAL_MIN_POSS)
                         & (rows['involve'] >= QUAL_MIN_INVOLVE)).astype(int)
    return rows, centres


def roll_up(rows, lo, hi):
    """Aggregate the scored season rows, preserving each season's own break-evens."""
    w = rows[rows['year'].between(lo, hi)]
    cols = ['v_self', 'v_ast', 'v_potast', 'sc_allow', 'bp_allow', 'sc_100', 'bp_100']
    g = w.groupby('nba_id').apply(lambda x: pd.Series({
        'poss': x['OffPoss'].sum(),
        'qualified': int(x['qualified'].max()),
        **{c: np.average(x[c], weights=x['OffPoss']) for c in cols},
    }), include_groups=False).reset_index()
    g['act'] = g['sc_100'] + g['bp_100']
    g['involve'] = g['v_self'] + g['v_ast'] + g['v_potast']
    g['sc_excess'] = g['sc_100'] - g['sc_allow']
    g['bp_excess'] = g['bp_100'] - g['bp_allow']
    g['allowance'] = g['sc_allow'] + g['bp_allow']
    g['excess'] = g['sc_excess'] + g['bp_excess']
    g['plays'] = g['involve'] + g['act']
    g['rztov'] = g['excess'] / g['plays'].clip(lower=1) * 100
    g['tov_saved_100'] = -g['excess']
    g['tov_saved'] = -g['excess'] * g['poss'] / 100
    return g


def inject_payload(html_path: Path, payload: str) -> None:
    """Replace the page's one `const DATA = ...;` line; the layout is left untouched."""
    lines = html_path.read_text(encoding='utf-8').split('\n')
    hits = [i for i, line in enumerate(lines) if line.startswith('const DATA = ')]
    if len(hits) != 1:
        raise ValueError(f'{html_path} must contain exactly one `const DATA = ` line, found {len(hits)}')
    lines[hits[0]] = f'const DATA = {payload};'
    html_path.write_text('\n'.join(lines), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', required=True, help='directory for the explorer JSON')
    ap.add_argument('--rebuild-putbacks', action='store_true')
    ap.add_argument('--html', type=Path, default=None,
                    help='explorer page whose single `const DATA = ...;` line is replaced '
                         'with the new payload (e.g. docs/rztov-explorer.html)')
    args = ap.parse_args()

    df = build_panel(args.rebuild_putbacks)

    fits = {}
    for end in range(FIRST_END, LAST_YEAR + 1):
        try:
            fits[end] = fit_window(df, end - WINDOW + 1, end)
        except FileNotFoundError:
            print(f'  window ending {end}: solve missing, skipped')
    if not fits:
        raise SystemExit('no trailing-window TOV solves available')

    print(f'{"season":<8}{"window":<11}{"self":>7}{"ast":>7}{"pass":>7}'
          f'{"k_tot":>7}{"k_sc":>7}{"k_bp":>7}{"R2w":>6}{"R2sc":>6}{"R2bp":>6}{"n":>6}')
    for y in range(FIRST_YEAR, LAST_YEAR + 1):
        src = min(max(y, min(fits)), max(fits))
        f = fits[src]
        tag = f'{f["window"][0]}-{str(f["window"][1])[2:]}' + ('*' if src != y else '')
        print(f'{y:<8}{tag:<11}{f["self"] * 100:7.2f}{f["ast"] * 100:7.2f}'
              f'{f["potast"] * 100:7.2f}{f["kappa"]:7.3f}{f["k_sc"]:7.3f}'
              f'{f["k_bp"]:7.3f}{f["r2"]:6.3f}{f["r2_sc"]:6.3f}{f["r2_bp"]:6.3f}'
              f'{f["n"]:6d}')
    print('  * predates the first full window; inherits the earliest window fit')

    rows, centres = score_seasons(df, fits)
    agg = roll_up(rows, BOARD_FROM, BOARD_TO)

    # ---- validation against the board window's own count-basis solve
    tgt = pd.read_csv(tov_target(BOARD_FROM, BOARD_TO))
    agg = agg.merge(tgt[['player_id', 'off']].rename(
        columns={'player_id': 'nba_id', 'off': 'tov_rapm'}), on='nba_id', how='left')
    fit_pool = agg[(agg['poss'] >= FIT_MIN_POSS) & agg['tov_rapm'].notna()].copy()
    W = fit_pool['poss'].to_numpy(float)
    T = fit_pool['tov_rapm'].to_numpy(float)

    corr = {
        'raw': float(np.corrcoef(fit_pool['act'], T)[0, 1]),
        'per_attempt': float(np.corrcoef(
            fit_pool['act'] / (fit_pool['v_self'] + fit_pool['v_ast']).clip(lower=1), T)[0, 1]),
        'excess': float(np.corrcoef(fit_pool['excess'], T)[0, 1]),
    }

    # ---- price an excess turnover in points, per lane and per window
    lane_prices = fit_turnover_point_prices(
        # Fit on the published (4-dp) ledger so the scoring board, which reads that
        # CSV, reproduces these prices exactly.
        rows[['nba_id', 'year', 'OffPoss', 'sc_excess', 'bp_excess']].round(4),
        Path(COMPONENT_DIR), first_end=min(fits), last_end=max(fits),
        bootstrap_draws=PRICE_DRAWS, seed=PRICE_SEED).set_index('window_hi')
    board_price = lane_prices.loc[max(fits)]
    price = {'sc': float(board_price['scoring_point_price']),
             'bp': float(board_price['badpass_point_price']),
             'source': 'alt_{scoring,badpass}_tov_net_value_{window}_all_rb_se_a20_40'}

    print(f'board {BOARD_FROM}-{BOARD_TO}: n={len(fit_pool)} '
          f'raw r {corr["raw"]:+.3f} per-attempt {corr["per_attempt"]:+.3f} '
          f'excess {corr["excess"]:+.3f} | '
          f'latest lane prices scoring {price["sc"]:.3f} / bad-pass {price["bp"]:.3f} pts')

    # ---- year over year on the published stat
    lag = rows[['nba_id', 'year', 'rztov', 'OffPoss']].copy()
    lag['year'] += 1
    pair = rows.merge(lag, on=['nba_id', 'year'], suffixes=('', '_p'))
    pair = pair[(pair['OffPoss'] >= YOY_MIN_POSS) & (pair['OffPoss_p'] >= YOY_MIN_POSS)]
    yoy = float(pair['rztov'].corr(pair['rztov_p']))

    # ---- how much of the fit each piece carries, against the same target
    def fit_cols(cols):
        _, rr2, rr = wls(T, fit_pool[cols], W)
        return [round(rr2, 3), round(rr, 3)]
    board_fit = fits[max(fits)]
    _, r2_unres, _ = wls(T, fit_pool[['sc_100', 'bp_100', 'v_self', 'v_ast', 'v_potast']], W)
    ladder = [
        ['actual TOV/100 alone', *fit_cols(['act'])],
        ['volumes alone', *fit_cols(['v_self', 'v_ast', 'v_potast'])],
        ['rZTOV x plays', *fit_cols(['excess'])],
        ['unrestricted (5 free terms)', round(r2_unres, 3), None],
    ]
    print(f'validation: yoy {yoy:+.3f} (n={len(pair)}) | '
          f'ladder {[l[:2] for l in ladder]}')

    # ---- volume dependence of the raw scoring-TOV rate (why one flat rate is wrong)
    curve = []
    big = rows[rows['OffPoss'] >= 1000]
    mix_base, mix_ast = board_fit['alloc_base'], board_fit['alloc_ast']
    for lo, hi, lab in [(3, 6, '3-6'), (6, 10, '6-10'), (10, 15, '10-15'),
                        (15, 20, '15-20'), (20, 99, '20+')]:
        m = big[(big['v_self'] >= lo) & (big['v_self'] < hi)]
        wm = m['OffPoss']
        rate = (((m['sc_100'] - mix_base - mix_ast * m['v_ast']) * wm).sum()
                / (m['v_self'] * wm).sum())
        curve.append([lab, round(100 * rate, 1), int(len(m))])

    # ---- within-player marginal rates (adjacent-season deltas)
    keep = ['nba_id', 'year', 'sc_100', 'bp_100', 'v_self', 'v_ast', 'v_potast', 'OffPoss']
    lag2 = big[keep].copy()
    lag2['year'] += 1
    b2 = big[keep].merge(lag2, on=['nba_id', 'year'], suffixes=('', '_p'))
    b2 = b2[(b2['OffPoss'] >= YOY_MIN_POSS) & (b2['OffPoss_p'] >= YOY_MIN_POSS)]
    for c in ('sc_100', 'bp_100', 'v_self', 'v_ast', 'v_potast'):
        b2[f'd_{c}'] = b2[c] - b2[f'{c}_p']
    wmin = np.minimum(b2['OffPoss'], b2['OffPoss_p'])
    bw, _, _ = wls(b2['d_sc_100'], b2[['d_v_self', 'd_v_ast']], wmin)
    bwp, _, _ = wls(b2['d_bp_100'], b2[['d_v_potast']], wmin)
    within = {'self': round(100 * bw[1], 1), 'ast': round(100 * bw[2], 1),
              'potast': round(100 * bwp[1], 1), 'n': int(len(b2))}

    # ---- league context, computed (not typed)
    cur = df[df['year'] == LAST_YEAR]
    league = {
        'season': f'{LAST_YEAR - 1}-{str(LAST_YEAR)[2:]}',
        'sc_100': round(cur['n_sc'].sum() / cur['OffPoss'].sum() * 100, 3),
        'bp_100': round(cur['n_bp'].sum() / cur['OffPoss'].sum() * 100, 3),
        'v_self': round(np.average(cur['v_self'], weights=cur['OffPoss']), 2),
        'v_ast': round(np.average(cur['v_ast'], weights=cur['OffPoss']), 2),
        'v_potast': round(np.average(cur['v_potast'], weights=cur['OffPoss']), 2),
    }

    # ---- payload
    names = df.sort_values('year').groupby('nba_id')['Name'].last()
    be = {}
    for y in range(FIRST_YEAR, LAST_YEAR + 1):
        f = fits[min(max(y, min(fits)), max(fits))]
        be[str(y)] = {
            'self': round(f['self'], 5), 'ast': round(f['ast'], 5),
            'potast': round(f['potast'], 5), 'kappa': round(f['kappa'], 4),
            'k_sc': round(f['k_sc'], 4), 'k_bp': round(f['k_bp'], 4),
            'window': f['window'], 'n': f['n'],
            'inherited': int(f['window'][1] != y),
            'centre_sc': round(float(centres['sc'].get(y, 0.0)), 5),
            'centre_bp': round(float(centres['bp'].get(y, 0.0)), 5),
        }
        lp = lane_prices.loc[f['window'][1]]
        be[str(y)].update({
            'price_sc': round(float(lp['scoring_point_price']), 4),
            'price_sc_ci': [round(float(lp['scoring_point_price_p05']), 4),
                            round(float(lp['scoring_point_price_p95']), 4)],
            'price_bp': round(float(lp['badpass_point_price']), 4),
            'price_bp_ci': [round(float(lp['badpass_point_price_p05']), 4),
                            round(float(lp['badpass_point_price_p95']), 4)],
            'price_r2': [round(float(lp['scoring_r2']), 3), round(float(lp['badpass_r2']), 3)],
        })

    seasons = [[int(t.nba_id), int(t.year), int(t.OffPoss),
                round(float(t.v_self), 3), round(float(t.v_ast), 3),
                round(float(t.v_potast), 3),
                round(float(t.sc_100), 4), round(float(t.bp_100), 4), int(t.qualified)]
               for t in rows.itertuples()]
    agg_rapm = {str(int(t.nba_id)): round(float(t.tov_rapm), 3)
                for t in agg.itertuples() if pd.notna(t.tov_rapm)}

    # Published point-valued offensive turnover factor over the same board window, as an
    # outside anchor on the board. Different solve (RS+PS, rubberband, its own alphas)
    # than the count-basis target these break-evens are fit against, so it is a familiar
    # reference rather than an independent check: the two correlate about 0.99.
    agg_otov = {}
    try:
        wf = pd.read_csv(OTOV_BOARD)
        agg_otov = {str(int(t.player_id)): round(float(t.oTOV), 3)
                    for t in wf.itertuples() if pd.notna(t.oTOV)}
        print(f'oTOV anchor: {len(agg_otov)} players from {os.path.basename(OTOV_BOARD)}')
    except (FileNotFoundError, AttributeError) as exc:
        print(f'oTOV anchor unavailable ({exc}); board column omitted')

    data = {
        'built': pd.Timestamp.today().strftime('%Y-%m-%d'),
        'season_range': [FIRST_YEAR, LAST_YEAR],
        'board_window': f'{BOARD_FROM - 1}-{str(BOARD_FROM)[2:]} through '
                        f'{BOARD_TO - 1}-{str(BOARD_TO)[2:]}',
        'board_years': [BOARD_FROM, BOARD_TO],
        'window_len': WINDOW,
        'min_poss': {'qual': QUAL_MIN_POSS, 'involve': QUAL_MIN_INVOLVE,
                     'fit': FIT_MIN_POSS, 'board': FIT_MIN_POSS},
        'be': be,
        'price': {k: (round(v, 4) if isinstance(v, float) else v) for k, v in price.items()},
        'fit': {'r2': round(board_fit['r2'], 3), 'r': round(board_fit['r'], 3),
                'n': board_fit['n'], 'r2_unrestricted': round(r2_unres, 3),
                'kappa': round(board_fit['kappa'], 4),
                'k_sc': round(board_fit['k_sc'], 4), 'k_bp': round(board_fit['k_bp'], 4),
                'r2_sc': round(board_fit['r2_sc'], 3), 'r2_bp': round(board_fit['r2_bp'], 3),
                'lane_self': round(board_fit['lane_self'], 5),
                'lane_ast': round(board_fit['lane_ast'], 5),
                'lane_potast': round(board_fit['lane_potast'], 5)},
        'validation': {**{k: round(v, 3) for k, v in corr.items()},
                       'yoy': round(yoy, 3), 'yoy_n': int(len(pair))},
        'ladder': ladder,
        'curve': curve,
        'within': within,
        'league': league,
        'target': os.path.basename(tov_target(BOARD_FROM, BOARD_TO)),
        'names': {str(int(k)): v for k, v in names.items()},
        'seasons': seasons,
        'agg_rapm': agg_rapm,
        'agg_otov': agg_otov,
        'otov_source': os.path.basename(OTOV_BOARD),
    }
    out_json = os.path.join(args.out_dir, 'rztov_explorer_data.json')
    with open(out_json, 'w') as f:
        json.dump(data, f, separators=(',', ':'))
    if args.html is not None:
        inject_payload(args.html, json.dumps(data, separators=(',', ':')))

    # ---- the stat itself, one row per player-season
    stat = rows[['nba_id', 'Name', 'year', 'OffPoss', 'v_self', 'v_ast', 'v_potast',
                 'involve', 'plays', 'sc_100', 'bp_100', 'act',
                 'sc_allow', 'bp_allow', 'sc_excess', 'bp_excess',
                 'allowance', 'excess', 'rztov',
                 'tov_saved_100', 'tov_saved', 'qualified']].rename(
        columns={'Name': 'name', 'OffPoss': 'off_poss'})
    # Points saved per 100 at the season's lane prices (positive good), the exact
    # turnover term of the historical scoring-security board.
    src = stat['year'].clip(lower=min(fits), upper=max(fits))
    stat['sc_point_price'] = src.map(lane_prices['scoring_point_price'])
    stat['bp_point_price'] = src.map(lane_prices['badpass_point_price'])
    stat['tov_points_saved_100'] = -(stat['sc_excess'] * stat['sc_point_price']
                                     + stat['bp_excess'] * stat['bp_point_price'])
    stat_csv = f'{RES}/rztov_seasons_{FIRST_YEAR}_{LAST_YEAR}.csv'
    stat.round(4).sort_values(['year', 'rztov']).to_csv(stat_csv, index=False)

    out = agg[['nba_id', 'poss', 'v_self', 'v_ast', 'v_potast', 'sc_100', 'bp_100',
               'involve', 'plays', 'act', 'sc_allow', 'bp_allow', 'sc_excess', 'bp_excess',
               'allowance', 'excess', 'rztov', 'tov_saved_100',
               'tov_saved', 'qualified', 'tov_rapm']].copy()
    out['name'] = out['nba_id'].map(names)
    agg_csv = f'{RES}/rztov_research_{BOARD_FROM % 100}_{BOARD_TO % 100}.csv'
    out.round(4).sort_values('rztov').to_csv(agg_csv, index=False)

    print(f'wrote {out_json} ({os.path.getsize(out_json) / 1024:.0f} KB), '
          f'{stat_csv} ({len(stat):,} player-seasons), '
          f'{agg_csv} ({len(out)} players)')


if __name__ == '__main__':
    main()
