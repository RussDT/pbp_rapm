"""Product C daily extension: within-season refresh test + as-of board.

The EPM predictive-ledger prior (research_predictive_ztov_prior.py) is built
from a dated snapshot, so the prior-informed time-decay TOV RAPM can refresh
on any day instead of once per season. This script answers "does refreshing
inside the season help future-possession prediction?" and builds the as-of
board that a daily job would publish.

Within-season rolling test (`evaluate`):
- Seasons S = 2019..2026, as-of year y = S-1. Structural parameters (lane
  prices/centres, |kappa_y|, FGA/FTA->TSA volume map, AST->potential-assist
  map, assisted-share table) are frozen at year y exactly as in the
  season-level harness; only the EPM snapshot and the possession window move.
- Refresh dates D_k = 1st of Dec, Jan, Feb, Mar, Apr. The forecast made at
  D_k uses possessions with day < D_k (trailing 5 seasons plus season S so
  far, TD weights 0.5^((D_k-day)/1500), full-season centering for complete
  seasons, running mean for the in-progress season) and scores possessions
  in [D_k, D_{k+1}); the last window runs to the RS end. Every possession
  from Dec 1 to the RS end is scored exactly once by the most recent
  forecast, which is what a daily-updating board does at monthly granularity.
- Candidates (alpha 3000, half-life 1500, pz scale 1.0, mix scale 0.5):
    A0 annual-plain    season-start solve, no prior (static all season)
    A1 annual-pz       season-start solve + season-start snapshot prior
                       (the season-level harness's C6 map, static all season)
    W0 refresh-plain   solve through D_k, no prior
    W1 refresh-stale   solve through D_k + season-start snapshot prior
    W2 refresh-fresh   solve through D_k + latest snapshot <= D_k  <- daily
    W3 refresh-mix     solve through D_k + tracking mix prior (frozen at y)
  --diagnose adds W1/W2 at scale 1.5, alpha 1500 and alpha 6000, and
    W2-blend50 = solve through D_k + 0.5*(season-start + fresh) prior.
- Every candidate is scored with the same running season level, so paired
  deltas isolate the player-coefficient maps.
- Pre-declared reads: W2 vs A1 (daily vs annual, primary), W2 vs W1 (prior
  refresh alone), W1 vs A1 (solve refresh alone), W2 vs W0 (fresh prior
  given a fresh solve), W2 vs W3 (fresh predictive vs frozen tracking),
  W0 vs A0 (solve refresh, plain). Game-clustered bootstrap per season and
  pooled. No selection happens here: the recipe is the dev-selected one from
  research_predictive_ztov_prior.py. 2025/2026 possessions were already
  opened at season level; everything through 2026 is development-era
  evidence and the untouched confirmation is 2026-27.

Snapshot rules: the season-start prior uses the harness rule (latest snapshot
requested strictly before the first game); refresh k uses the latest snapshot
requested on or before D_k (a snapshot requested for D excludes games on D,
and the solve uses days < D, so the two are aligned). Coverage: in-season
snapshots list only players with a completed game this season and the
preseason snapshot lacks rookies, so the fresh prior is fresh-where-covered
and season-start otherwise (carry_forward); without it injured returners'
priors dropped to 0 mid-season.

Result (2026-08-16 run, 8 seasons, 1.42M possessions): refreshing the solve
is worth -89.7e-6 pooled vs the static annual map (P 1.00 every season and
window); the pz prior beats plain refresh (-15.1e-6) and edges the frozen
tracking prior (-5.7e-6); the full fresh-snapshot swap ties the season-start
prior with carry-forward (-0.04e-6 [-3.7,+3.5]).

`evaluate --diagnose` (2026-08-18) explains the tie: the fresh snapshot does
carry information the refreshed stale-prior solve lacks (regressing W1's
forward residuals on the prior movement gives slope +0.26 [+0.20,+0.32];
the annual map A1 is +0.57 as a positive control), but the full swap
over-corrects (W2 slope -0.07 [-0.13,-0.01]). Because the ridge is linear
in the prior, u = e_stale - e_fresh is exactly the adjustment the swap
induces and gamma = sum(e_stale*u)/sum(u^2) is its MSE-optimal multiple:
gamma = 0.50 [0.36,0.64] pooled, 0.555 [0.39,0.72] on 2019-2024, 0.34
[0.07,0.63] on 2025-2026, so using the fresh movement fully and ignoring it
tie by construction. Mechanism: EPM's in-season update and the live
possession rows overlap (its movement correlates +0.39 with what the plain
live solve had already moved a player, rising to +0.54 in the top movement
quintile, which is where the full swap hurts); roughly half of the movement
is new to the lineup solve, half is double counted. The pre-specified half
blend W2-blend50 = 0.5*season-start + 0.5*fresh(carry-forward) beats the
stale prior -3.17e-6 [-4.98,-1.39] pooled (dev -3.84 [-6.01,-1.83], frozen
-1.17 [-4.80,+2.43]; 8/8 seasons, 5/5 windows) and beats the full swap
-3.13e-6 [-4.94,-1.31]. The stale-prior weighting is locally optimal within
season (scale 1.5, alpha 1500 and alpha 6000 all worse for both stale and
fresh), so the tie was never an alpha artefact. Diagnose outputs:
results/tov_predictive_ztov_daily_{eval,windows}_diag.csv,
tov_predictive_ztov_daily_diag.csv (slopes, gammas, exposure quintiles,
ledger drift) and tov_predictive_ztov_daily_diag_players.csv (per-player
per-window ledger, coefficients, live and forward exposure). See
docs/TOV_PRIOR_RAPM.md, "Within-season refresh and the daily board".

Board (`board`): as-of date D, latest cached snapshot requested <= D (pass
--fetch to pull D first; needs EPM_API_KEY), structural parameters from the
last complete season that has lane prices and kappa, TD solve on possessions
with day < D, writes results/tov_predictive_ztov_board_<D>.csv plus a
manifest and a *_latest copy. In-season prior rule: `--fresh-weight` (default
0.5) times the latest snapshot with carry-forward plus the complement times
the season-start snapshot; `off_pz_stale` / `off_pz_fresh` carry the W1 / W2
solves beside the default `off_pz`. With no live season rows the prior is the
latest snapshot alone (the season-level C6 structure). Positive offense =
commits more turnovers per 100 possessions than lineup context predicts;
positive defense = forces more.

Research lane only: nothing here feeds production solves, the published rZTOV
surface, or VPM inputs.
"""
import argparse
import json
import subprocess
import sys
from datetime import date

import numpy as np
import pandas as pd
from scipy import sparse

from research_prior_informed_tov_rapm import (
    RES, ROOT, PROC, SEASONS, TRAIL, KAPPA, FIXED_BASE, BOOT_DRAWS, SEED,
    load_seasons, active_cols, gram, solve, mse, build_mix_prior)
from research_predictive_ztov_prior import (
    EPM_DIR, MIX_SCALE, epm_files, read_epm, epm_snapshot, vol_map,
    load_passes, ast_map, recover_prices, trailing_share)

PZ_SCALE = 1.0                      # dev-selected C6 scale, 2026-08-16
EVAL_SEASONS = list(range(2019, 2027))
REFRESH_MONTHS = [12, 1, 2, 3, 4]   # 1st-of-month refresh dates
LEVEL_MIN_ROWS = 5000               # running-mean level needs this many rows
CANDS = ['A0-annual-plain', 'A1-annual-pz', 'W0-refresh-plain',
         'W1-refresh-stale', 'W2-refresh-fresh', 'W3-refresh-mix']
PAIRS = [('W2-refresh-fresh', 'A1-annual-pz'),
         ('W2-refresh-fresh', 'W1-refresh-stale'),
         ('W1-refresh-stale', 'A1-annual-pz'),
         ('W2-refresh-fresh', 'W0-refresh-plain'),
         ('W2-refresh-fresh', 'W3-refresh-mix'),
         ('W0-refresh-plain', 'A0-annual-plain')]

# --diagnose: extra weightings of the same stale/fresh priors on the same
# windows (is the null a weighting artefact?) plus a half-and-half blend (is
# the fresh content real but over-counted at full weight?).
DIAG_CANDS = ['W1-stale-s1.5', 'W2-fresh-s1.5',
              'W1-stale-a1500', 'W2-fresh-a1500',
              'W1-stale-a6000', 'W2-fresh-a6000',
              'W2-blend50']
DIAG_PAIRS = [('W2-fresh-s1.5', 'W1-stale-s1.5'),
              ('W2-fresh-a1500', 'W1-stale-a1500'),
              ('W2-fresh-a6000', 'W1-stale-a6000'),
              ('W2-blend50', 'W1-refresh-stale'),
              ('W2-blend50', 'W2-refresh-fresh'),
              ('W1-stale-s1.5', 'W1-refresh-stale'),
              ('W1-stale-a1500', 'W1-refresh-stale'),
              ('W1-stale-a6000', 'W1-refresh-stale'),
              ('W2-fresh-s1.5', 'W2-refresh-fresh'),
              ('W2-fresh-a1500', 'W2-refresh-fresh'),
              ('W2-fresh-a6000', 'W2-refresh-fresh')]
SLOPE_CANDS = ['A1-annual-pz', 'W0-refresh-plain', 'W1-refresh-stale',
               'W2-refresh-fresh', 'W2-blend50', 'W1-stale-a6000',
               'W2-fresh-a6000']


def day_int(s):
    return int(np.datetime64(s, 'D').astype(np.int64))


def day_str(d):
    return str(np.datetime64(int(d), 'D'))


def snapshot_at(day):
    """Latest cached snapshot requested on or before `day` (int days)."""
    best = None
    for ds, f in epm_files():
        if day_int(ds) <= day:
            best = (ds, f)
    assert best is not None, 'no EPM snapshot on/before requested day'
    return best


class Structural:
    """Year-y structural parameters shared by every snapshot of that year."""

    def __init__(self, sea, pm, yr, half_life):
        self.yr = yr
        self.prices = recover_prices(sea, yr)
        self.ast = ast_map(pm, yr)
        self.vol, self.n_vol = vol_map(sea, yr)
        self.share, self.league = trailing_share(sea, yr, half_life)
        self.kappa = KAPPA[yr]

    def describe(self):
        (va, vb, vc), (a, b) = self.vol, self.ast
        return (f'year {self.yr} kappa {self.kappa:.4f} '
                f'vol tsa={va:+.2f}+{vb:.3f}fga+{vc:.3f}fta (n={self.n_vol}) '
                f'ast pot={a:+.2f}+{b:.3f}ast league share {self.league:.3f}')


def pz_ledger(e, st, pidx):
    """Per-player predictive ledger from one snapshot; prior = kappa*excess."""
    e = e[e['player_id'].astype(int).isin(list(pidx))].copy()
    pid = e['player_id'].astype(int).to_numpy()
    va, vb, vc = st.vol
    a, b = st.ast
    pr = st.prices
    v_hat = np.clip(va + vb * (e['p_fg2a_100'] + e['p_fg3a_100'])
                    + vc * e['p_fta_100'], 0, None).to_numpy()
    sh = np.array([st.share.get(p, st.league) for p in pid])
    pot_hat = np.clip(a + b * e['p_ast_100'].to_numpy(), 0, None)
    allow = (pr['be_self'] * (1 - sh) * v_hat + pr['be_ast'] * sh * v_hat
             + pr['c_sc'] + pr['be_pot'] * pot_hat + pr['c_bp'])
    excess = e['p_tov_100'].to_numpy() - allow
    w = e['p_mp_48'].to_numpy()
    excess = excess - np.average(excess, weights=w)
    return pd.DataFrame({
        'nba_id': pid,
        'p_tov_100': e['p_tov_100'].to_numpy(),
        'p_mp_48': w,
        'v_hat': v_hat,
        'ast_share': sh,
        'pot_hat': pot_hat,
        'allowance_hat': allow,
        'excess_hat': excess,
        'prior': st.kappa * excess,
    })


def prior_vec(ledger, pidx, P):
    """Offense-side prior vector plus its coverage mask."""
    p = np.zeros(2 * P)
    cov = np.zeros(2 * P, dtype=bool)
    idx = [pidx[int(i)] for i in ledger['nba_id']]
    p[idx] = ledger['prior'].to_numpy()
    cov[idx] = True
    return p, cov


def carry_forward(p_fresh, cov_fresh, p_start):
    """Daily semantics: latest snapshot value per player, season-start value
    for players the in-season snapshot does not carry yet (in-season
    snapshots list only players with a completed game this season, so an
    injured returner keeps his preseason prior instead of dropping to 0)."""
    return np.where(cov_fresh, p_fresh, p_start)


def season_level(y_partial, fallback):
    return float(y_partial.mean()) if len(y_partial) >= LEVEL_MIN_ROWS \
        else float(fallback)


def load_inputs():
    sea = pd.read_csv(RES / 'rztov_seasons_2014_2026.csv')
    pm = load_passes()
    return sea, pm


# ----------------------------------------------------------------------------
# evaluate
# ----------------------------------------------------------------------------
def window_stats(X, y, days, wref, edges, cols):
    """Per window j: normal-equation pieces for rows edges[j] <= day <
    edges[j+1], restricted to `cols`, with reference weights `wref`."""
    out = []
    for j in range(len(edges) - 1):
        m = (days >= edges[j]) & (days < edges[j + 1])
        idx = np.where(m)[0]
        Xj = X[idx][:, cols]
        wj = wref[idx]
        yj = y[idx]
        Xw = Xj.multiply(wj[:, None]).tocsr()
        out.append({
            'G': (Xj.T @ Xw).toarray(),
            'r': np.asarray(Xj.T @ (wj * yj)).ravel(),
            's': np.asarray(Xj.T @ wj).ravel(),
            'n': int(len(idx)),
            'ysum': float(yj.sum()),
            'seen': np.asarray(Xj.sum(axis=0)).ravel() > 0,
            'cnt': np.asarray(Xj.sum(axis=0)).ravel(),
            'idx': idx,
        })
    return out


def refresh_edges(S, blocks):
    b = blocks[S]
    first, last = int(b['days'].min()), int(b['days'].max())
    ds = []
    for mth in REFRESH_MONTHS:
        yy = S - 1 if mth == 12 else S
        ds.append(day_int(f'{yy}-{mth:02d}-01'))
    return [first] + ds + [last + 1], first, last


def evaluate(args):
    A, H = FIXED_BASE
    print('loading possession blocks 2014-2026 (RS)...', flush=True)
    blocks, pidx, P = load_seasons()
    sea, pm = load_inputs()
    seasons = [int(s) for s in args.seasons.split(',')] if args.seasons \
        else EVAL_SEASONS
    diag = bool(getattr(args, 'diagnose', False))
    cands = CANDS + DIAG_CANDS if diag else CANDS
    pairs = PAIRS + DIAG_PAIRS if diag else PAIRS

    errs = {}        # (S, k) -> {cand: err}
    games = {}       # (S, k) -> games
    expo = {}        # (S, k) -> {'signed': x, 'abs': x}  (diagnose)
    player_rows = []  # per-player per-window diagnostics (diagnose)
    meta_rows = []
    for S in seasons:
        y = S - 1
        st = Structural(sea, pm, y, H)
        edges, first, last = refresh_edges(S, blocks)
        bS = blocks[S]
        f0, e0 = epm_snapshot(first)
        led_start = pz_ledger(e0, st, pidx)
        p_start, cov_start = prior_vec(led_start, pidx, P)
        p_mix, _ = build_mix_prior(sea, y, H, KAPPA[y], pidx, P)
        p_mix = MIX_SCALE * p_mix

        mask_y = np.zeros(2 * P, dtype=bool)
        mask_y[active_cols(blocks, y, P)] = True
        seen_S = np.asarray(bS['X'].sum(axis=0)).ravel() > 0
        ac_sup = np.where(mask_y | seen_S)[0]
        print(f'\nseason {S}: as-of {y} | {st.describe()}', flush=True)
        print(f'  season-start snapshot {f0}  first game {day_str(first)}  '
              f'last {day_str(last)}  columns {len(ac_sup)}', flush=True)
        G_h, r_h = gram(blocks, y, H, ac_sup)
        asof_y = blocks[y]['asof']
        mu_y = float(blocks[y]['y'].mean())
        wref = 0.5 ** ((asof_y - bS['days']) / H)
        W = window_stats(bS['X'], bS['y'], bS['days'], wref, edges, ac_sup)
        sub_y = mask_y[ac_sup]

        def annual_map(prior):
            b = np.zeros(2 * P)
            pr = None if prior is None else prior[ac_sup][sub_y]
            b[ac_sup[sub_y]] = solve(G_h[np.ix_(sub_y, sub_y)], r_h[sub_y],
                                     A, prior=pr)
            return b

        bA0 = annual_map(None)
        bA1 = annual_map(PZ_SCALE * p_start)

        Gacc, racc = G_h.copy(), r_h.copy()
        sacc = np.zeros(len(ac_sup))
        cnt_acc = np.zeros(len(ac_sup))
        nacc, ysacc = 0, 0.0
        seen_acc = sub_y.copy()
        for k in range(1, len(edges) - 1):
            wj = W[k - 1]
            Gacc += wj['G']
            racc += wj['r']
            sacc += wj['s']
            cnt_acc += wj['cnt']
            nacc += wj['n']
            ysacc += wj['ysum']
            seen_acc |= wj['seen']
            m_k = ysacc / nacc if nacc >= LEVEL_MIN_ROWS else mu_y
            c_k = 0.5 ** ((edges[k] - asof_y) / H)
            G = c_k * Gacc[np.ix_(seen_acc, seen_acc)]
            r = c_k * (racc - m_k * sacc)[seen_acc]
            cols_k = ac_sup[seen_acc]
            ds, f = snapshot_at(edges[k])
            led_new = pz_ledger(read_epm(f), st, pidx)
            p_new, cov_new = prior_vec(led_new, pidx, P)
            p_fresh = carry_forward(p_new, cov_new, p_start)

            def refresh_map(prior, alpha=A):
                b = np.zeros(2 * P)
                pr = None if prior is None else prior[cols_k]
                b[cols_k] = solve(G, r, alpha, prior=pr)
                return b

            maps = {'A0-annual-plain': bA0,
                    'A1-annual-pz': bA1,
                    'W0-refresh-plain': refresh_map(None),
                    'W1-refresh-stale': refresh_map(PZ_SCALE * p_start),
                    'W2-refresh-fresh': refresh_map(PZ_SCALE * p_fresh),
                    'W3-refresh-mix': refresh_map(p_mix)}
            if diag:
                p_blend = 0.5 * (p_start + p_fresh)
                maps.update({
                    'W1-stale-s1.5': refresh_map(1.5 * p_start),
                    'W2-fresh-s1.5': refresh_map(1.5 * p_fresh),
                    'W1-stale-a1500': refresh_map(PZ_SCALE * p_start, 1500.0),
                    'W2-fresh-a1500': refresh_map(PZ_SCALE * p_fresh, 1500.0),
                    'W1-stale-a6000': refresh_map(PZ_SCALE * p_start, 6000.0),
                    'W2-fresh-a6000': refresh_map(PZ_SCALE * p_fresh, 6000.0),
                    'W2-blend50': refresh_map(PZ_SCALE * p_blend)})
            idx = W[k]['idx']
            Xk = bS['X'][idx]
            yk = bS['y'][idx] / 100.0
            errs[(S, k)] = {}
            for name, b in maps.items():
                yhat = Xk @ (b / 100.0) + m_k / 100.0
                errs[(S, k)][name] = yk - yhat
            games[(S, k)] = bS['games'][idx]
            if diag:
                # prior movement (offense side, per 100) and each scored
                # possession's exposure to it through its five offensive
                # players; per-player ledger/coefficient snapshot
                delta = PZ_SCALE * (p_fresh - p_start)
                Xo = Xk[:, :P]
                expo[(S, k)] = {
                    'signed': np.asarray(Xo @ (delta[:P] / 100.0)).ravel(),
                    'abs': np.asarray(Xo @ (np.abs(delta[:P]) / 100.0)).ravel()}
                cnt_full = np.zeros(2 * P)
                cnt_full[ac_sup] = cnt_acc
                both = led_start.merge(led_new, on='nba_id',
                                       suffixes=('_start', '_fresh'))
                ii = np.array([pidx[int(i)] for i in both['nba_id']],
                              dtype=int)
                both['season'] = S
                both['k'] = k
                both['refresh'] = day_str(edges[k])
                both['delta_prior'] = delta[ii]
                both['live_poss'] = cnt_full[ii]
                both['b_W0'] = maps['W0-refresh-plain'][ii]
                both['b_W1'] = maps['W1-refresh-stale'][ii]
                both['b_W2'] = maps['W2-refresh-fresh'][ii]
                both['b_A1'] = maps['A1-annual-pz'][ii]
                # forward W1 residual mass carried by this player's offensive
                # possessions in the scored window (sum and count)
                e1 = errs[(S, k)]['W1-refresh-stale']
                both['w1_resid_sum'] = np.asarray(Xo.T @ e1).ravel()[ii]
                both['fwd_poss'] = np.asarray(Xo.sum(axis=0)).ravel()[ii]
                player_rows.append(both)
            n_start = int(cov_start.sum())
            n_new = int(cov_new.sum())
            n_carry = int((cov_start & ~cov_new).sum())
            n_added = int((cov_new & ~cov_start).sum())
            print(f'  refresh {day_str(edges[k])}: snapshot {ds}  train rows '
                  f'S {nacc:,}  level {m_k:.3f}  cols {len(cols_k)}  '
                  f'prior start {n_start} fresh {n_new} '
                  f'(carried {n_carry}, new {n_added})  scored '
                  f'{len(idx):,} rows to {day_str(edges[k + 1] - 1)}',
                  flush=True)
            row = {'season': S, 'k': k, 'refresh': day_str(edges[k]),
                   'snapshot': ds, 'train_rows_S': nacc, 'level': m_k,
                   'cols': len(cols_k), 'scored_rows': len(idx),
                   'prior_start': n_start, 'prior_fresh': n_new,
                   'prior_carried': n_carry, 'prior_new': n_added}
            for name in cands:
                row[f'mse_{name}'] = mse(errs[(S, k)][name])
            meta_rows.append(row)

    # ---------------- reporting --------------------------------------------
    rng = np.random.default_rng(SEED)
    out_rows = []

    def report(label, keys):
        g = np.concatenate([np.char.add(f'{S}_', games[(S, k)].astype(str))
                            for (S, k) in keys])
        e = {n: np.concatenate([errs[key][n] for key in keys]) for n in cands}
        ug, inv = np.unique(g, return_inverse=True)
        counts = np.bincount(inv)
        per_game = {n: np.bincount(inv, weights=e[n] * e[n]) for n in cands}
        print(f'  {label} ({len(g):,} possessions, {len(ug):,} games):')
        for n in cands:
            out_rows.append({'scope': label, 'cand': n, 'mse': mse(e[n]),
                             'n': len(g)})
        for a, bnm in pairs:
            d_pt = (per_game[a].sum() - per_game[bnm].sum()) / len(g)
            draws = rng.integers(0, len(ug), (args.boot_draws, len(ug)))
            dd = ((per_game[a][draws].sum(axis=1)
                   - per_game[bnm][draws].sum(axis=1))
                  / counts[draws].sum(axis=1))
            lo, hi = np.percentile(dd, [5, 95])
            pb = float(np.mean(dd < 0))
            out_rows.append({'scope': label, 'cand': f'{a} vs {bnm}',
                             'mse': d_pt, 'n': len(g), 'lo': lo, 'hi': hi,
                             'p_better': pb})
            print(f'    {a:<17} vs {bnm:<17} {d_pt * 1e6:+8.2f}  '
                  f'[{lo * 1e6:+.2f},{hi * 1e6:+.2f}]  P(better) {pb:.2f}')

    print('\nMSE by season (all scored windows Dec 1 -> RS end):')
    for S in seasons:
        keys = [key for key in errs if key[0] == S]
        e = {n: np.concatenate([errs[key][n] for key in keys]) for n in cands}
        print(f'  {S}: ' + '  '.join(f'{n.split("-")[0]} {mse(e[n]):.6f}'
                                     for n in CANDS))
        if diag:
            print('        ' + '  '.join(f'{n} {mse(e[n]):.6f}'
                                        for n in DIAG_CANDS))

    print('\npaired deltas, e-6 (negative = first candidate better), '
          'game-clustered bootstrap 5-95%:')
    for S in seasons:
        report(f'season {S}', [key for key in errs if key[0] == S])
    dev = [key for key in errs if key[0] <= 2024]
    frz = [key for key in errs if key[0] >= 2025]
    if dev:
        report('pooled 2019-2024', dev)
    if frz:
        report('pooled 2025-2026', frz)
    report('pooled all', list(errs))
    print('\nby refresh window (pooled across seasons; descriptive):')
    for k in range(1, len(REFRESH_MONTHS) + 1):
        keys = [key for key in errs if key[1] == k]
        if keys:
            report(f'window {k} ({REFRESH_MONTHS[k - 1]:02d}-01)', keys)

    suffix = '_diag' if diag else ''
    out = RES / f'tov_predictive_ztov_daily_eval{suffix}.csv'
    pd.DataFrame(out_rows).to_csv(out, index=False)
    pd.DataFrame(meta_rows).to_csv(
        RES / f'tov_predictive_ztov_daily_windows{suffix}.csv', index=False)
    print(f'\nwrote {out}')
    if diag:
        diagnose_report(errs, games, expo, player_rows, seasons, rng,
                        args.boot_draws)
    print('config:', {'alpha': A, 'half_life': H, 'pz_scale': PZ_SCALE,
                      'mix_scale': MIX_SCALE, 'refresh_months': REFRESH_MONTHS,
                      'level_min_rows': LEVEL_MIN_ROWS, 'diagnose': diag})


def _wcorr(x, y, w):
    w = np.asarray(w, float)
    if w.sum() <= 0 or len(x) < 3:
        return np.nan
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    vx = np.average((x - mx) ** 2, weights=w)
    vy = np.average((y - my) ** 2, weights=w)
    return cov / np.sqrt(vx * vy) if vx > 0 and vy > 0 else np.nan


def _wsd(x, w):
    w = np.asarray(w, float)
    m = np.average(x, weights=w)
    return float(np.sqrt(np.average((x - m) ** 2, weights=w)))


def diagnose_report(errs, games, expo, player_rows, seasons, rng, draws_n):
    """Why doesn't the fresh snapshot help? Three reads.

    1. Does the prior's movement (fresh - start) predict each candidate's
       forward residuals? Row regression e = beta * (sum over the five
       offensive players of delta_i / 100); beta ~ 0 means the candidate
       already priced what the fresh snapshot learned, beta > 0 means it
       under-uses it. A1 (annual, no live rows) is the positive control.
    2. W2 vs W1 by possession exposure to prior movement (quintiles of the
       summed |delta|): does the fresh prior help where it actually differs?
    3. Per-player ledger: how much p_tov / allowance / excess move
       preseason -> refresh, and how EPM's movement lines up with what the
       live solve had already moved (b_W0 - p_start, b_W1 - p_start).
    """
    print('\n=== diagnose: does the fresh snapshot carry information the '
          'refreshed solve lacks? ===')
    keys_all = list(errs)
    scopes = [('pooled 2019-2024', [k for k in keys_all if k[0] <= 2024]),
              ('pooled 2025-2026', [k for k in keys_all if k[0] >= 2025]),
              ('pooled all', keys_all)]
    for S in seasons:
        scopes.append((f'season {S}', [k for k in keys_all if k[0] == S]))
    diag_rows = []

    print('\n(1) slope of forward residual on signed prior-movement exposure '
          '(dimensionless; 0 = already priced, >0 = under-used, <0 = '
          'over-corrected), game-clustered 5-95%:')
    for label, keys in scopes:
        if not keys:
            continue
        g = np.concatenate([np.char.add(f'{S}_', games[(S, k)].astype(str))
                            for (S, k) in keys])
        x = np.concatenate([expo[key]['signed'] for key in keys])
        ug, inv = np.unique(g, return_inverse=True)
        sxx = np.bincount(inv, weights=x * x)
        share_nonzero = float(np.mean(x != 0))
        print(f'  {label} ({len(g):,} rows, {share_nonzero:.0%} with '
              f'nonzero exposure, sd exposure {x.std() * 100:.3f}/100):')
        boot = label.startswith('pooled')
        for n in SLOPE_CANDS:
            e = np.concatenate([errs[key][n] for key in keys])
            sxy = np.bincount(inv, weights=x * e)
            beta = sxy.sum() / sxx.sum()
            row = {'read': 'slope', 'scope': label, 'cand': n, 'value': beta,
                   'n': len(g)}
            if boot:
                d = rng.integers(0, len(ug), (draws_n, len(ug)))
                bb = sxy[d].sum(axis=1) / sxx[d].sum(axis=1)
                lo, hi = np.percentile(bb, [5, 95])
                row.update({'lo': lo, 'hi': hi})
                print(f'    {n:<17} beta {beta:+.4f}  [{lo:+.4f},{hi:+.4f}]')
            else:
                print(f'    {n:<17} beta {beta:+.4f}')
            diag_rows.append(row)

    print('\n(1b) optimal weight on the fresh-prior adjustment: the ridge is '
          'linear in the prior, so u = e_stale - e_fresh is exactly the '
          'per-possession adjustment the snapshot swap induces; gamma = '
          'sum(e_stale*u)/sum(u^2) is the MSE-optimal multiple of that '
          'adjustment (1 = use fully, 0 = pure noise, 0.5 = using and '
          'ignoring tie). best-case gain = -gamma^2 * mean(u^2):')
    swap_pairs = [('W1-refresh-stale', 'W2-refresh-fresh'),
                  ('W1-stale-a1500', 'W2-fresh-a1500'),
                  ('W1-stale-a6000', 'W2-fresh-a6000'),
                  ('W1-stale-s1.5', 'W2-fresh-s1.5')]
    for label, keys in scopes:
        if not keys:
            continue
        g = np.concatenate([np.char.add(f'{S}_', games[(S, k)].astype(str))
                            for (S, k) in keys])
        ug, inv = np.unique(g, return_inverse=True)
        cnt = np.bincount(inv)
        boot = label.startswith('pooled')
        print(f'  {label}:')
        for stale, fresh in swap_pairs:
            es = np.concatenate([errs[key][stale] for key in keys])
            ef = np.concatenate([errs[key][fresh] for key in keys])
            u = es - ef
            suu = np.bincount(inv, weights=u * u)
            seu = np.bincount(inv, weights=es * u)
            gamma = seu.sum() / suu.sum()
            m_u2 = suu.sum() / len(g)
            row = {'read': 'gamma', 'scope': label,
                   'cand': f'{fresh} - {stale}', 'value': gamma,
                   'mean_u2': m_u2, 'best_gain': -gamma ** 2 * m_u2,
                   'n': len(g)}
            if boot:
                d = rng.integers(0, len(ug), (draws_n, len(ug)))
                gg = seu[d].sum(axis=1) / suu[d].sum(axis=1)
                lo, hi = np.percentile(gg, [5, 95])
                row.update({'lo': lo, 'hi': hi})
                print(f'    {fresh:<15} - {stale:<15} gamma {gamma:+.3f}  '
                      f'[{lo:+.3f},{hi:+.3f}]  rms adj {np.sqrt(m_u2) * 100:.3f}'
                      f'/100  best-case gain {-gamma ** 2 * m_u2 * 1e6:+.2f}e-6')
            else:
                print(f'    {fresh:<15} - {stale:<15} gamma {gamma:+.3f}  '
                      f'rms adj {np.sqrt(m_u2) * 100:.3f}/100  best-case gain '
                      f'{-gamma ** 2 * m_u2 * 1e6:+.2f}e-6')
            diag_rows.append(row)

    print('\n(2) W2 vs W1 by possession exposure to prior movement '
          '(quintiles of summed |delta| over the five offensive players; '
          'pooled all; e-6, game-clustered 5-95%):')
    g = np.concatenate([np.char.add(f'{S}_', games[(S, k)].astype(str))
                        for (S, k) in keys_all])
    xa = np.concatenate([expo[key]['abs'] for key in keys_all])
    e1 = np.concatenate([errs[key]['W1-refresh-stale'] for key in keys_all])
    e2 = np.concatenate([errs[key]['W2-refresh-fresh'] for key in keys_all])
    eb = np.concatenate([errs[key]['W2-blend50'] for key in keys_all])
    qs = np.quantile(xa, [0.2, 0.4, 0.6, 0.8])
    bins = np.digitize(xa, qs)
    for q in range(5):
        m = bins == q
        gq = g[m]
        ug, inv = np.unique(gq, return_inverse=True)
        cnt = np.bincount(inv)
        for a_nm, ea in (('W2-refresh-fresh', e2), ('W2-blend50', eb)):
            pa = np.bincount(inv, weights=ea[m] ** 2)
            pb_ = np.bincount(inv, weights=e1[m] ** 2)
            d_pt = (pa.sum() - pb_.sum()) / m.sum()
            d = rng.integers(0, len(ug), (draws_n, len(ug)))
            dd = (pa[d].sum(axis=1) - pb_[d].sum(axis=1)) / cnt[d].sum(axis=1)
            lo, hi = np.percentile(dd, [5, 95])
            lab = f'Q{q + 1} |expo| {xa[m].mean() * 100:.2f}/100'
            print(f'  {lab:<22} {a_nm:<17} vs W1 {d_pt * 1e6:+7.2f}  '
                  f'[{lo * 1e6:+.2f},{hi * 1e6:+.2f}]  n {m.sum():,}')
            diag_rows.append({'read': 'expo_quintile', 'scope': lab,
                              'cand': f'{a_nm} vs W1-refresh-stale',
                              'value': d_pt, 'lo': lo, 'hi': hi,
                              'n': int(m.sum())})

    print('\n(3) per-player ledger movement, season-start snapshot -> refresh '
          '(players in both snapshots, weighted by live offensive '
          'possessions through the refresh date):')
    pl = pd.concat(player_rows, ignore_index=True)
    pl.to_csv(RES / 'tov_predictive_ztov_daily_diag_players.csv', index=False)
    for (S, k), d in pl.groupby(['season', 'k']):
        w = d['live_poss'].to_numpy()
        if w.sum() <= 0:
            continue
        dp = (d['p_tov_100_fresh'] - d['p_tov_100_start']).to_numpy()
        da = (d['allowance_hat_fresh'] - d['allowance_hat_start']).to_numpy()
        de = (d['excess_hat_fresh'] - d['excess_hat_start']).to_numpy()
        dpr = d['delta_prior'].to_numpy()
        w0 = (d['b_W0'] - d['prior_start']).to_numpy()
        w1 = (d['b_W1'] - d['prior_start']).to_numpy()
        row = {
            'season': S, 'k': k, 'refresh': d['refresh'].iloc[0],
            'players': len(d),
            'sd_p_tov_start': _wsd(d['p_tov_100_start'].to_numpy(), w),
            'sd_p_tov_fresh': _wsd(d['p_tov_100_fresh'].to_numpy(), w),
            'corr_p_tov': _wcorr(d['p_tov_100_start'].to_numpy(),
                                 d['p_tov_100_fresh'].to_numpy(), w),
            'sd_d_p_tov': _wsd(dp, w),
            'sd_d_allow': _wsd(da, w),
            'sd_d_excess': _wsd(de, w),
            'corr_d_p_tov_d_allow': _wcorr(dp, da, w),
            'sd_prior_start': _wsd(d['prior_start'].to_numpy(), w),
            'sd_d_prior': _wsd(dpr, w),
            'corr_prior_start_fresh': _wcorr(d['prior_start'].to_numpy(),
                                             d['prior_fresh'].to_numpy(), w),
            'sd_W0_minus_start': _wsd(w0, w),
            'sd_W1_minus_start': _wsd(w1, w),
            'corr_dprior_W0move': _wcorr(dpr, w0, w),
            'corr_dprior_W1move': _wcorr(dpr, w1, w),
        }
        # does the prior movement line up with W1's forward residual rate?
        fw = d['fwd_poss'].to_numpy()
        ok = fw > 0
        if ok.sum() > 10:
            rate = d['w1_resid_sum'].to_numpy()[ok] / fw[ok] * 100
            row['corr_dprior_W1fwdresid'] = _wcorr(dpr[ok], rate, fw[ok])
        diag_rows.append({'read': 'ledger', 'scope': f'{S}/{k}', **row})
        if k in (1, 5):
            print(f'  {S} k{k} {row["refresh"]}: n {len(d)}  '
                  f'p_tov sd {row["sd_p_tov_start"]:.3f}->'
                  f'{row["sd_p_tov_fresh"]:.3f} corr {row["corr_p_tov"]:.3f}  '
                  f'sd d(p_tov) {row["sd_d_p_tov"]:.3f} d(allow) '
                  f'{row["sd_d_allow"]:.3f} d(excess) {row["sd_d_excess"]:.3f} '
                  f'corr(dp,da) {row["corr_d_p_tov_d_allow"]:+.3f}')
            print(f'      prior sd {row["sd_prior_start"]:.3f} moved sd '
                  f'{row["sd_d_prior"]:.3f} corr(start,fresh) '
                  f'{row["corr_prior_start_fresh"]:.3f}  solve moved from '
                  f'prior sd W0 {row["sd_W0_minus_start"]:.3f} W1 '
                  f'{row["sd_W1_minus_start"]:.3f}  corr(dprior, W0 move) '
                  f'{row["corr_dprior_W0move"]:+.3f} (W1 move) '
                  f'{row["corr_dprior_W1move"]:+.3f}  corr(dprior, W1 fwd '
                  f'resid rate) {row.get("corr_dprior_W1fwdresid", np.nan):+.3f}')
    out = RES / 'tov_predictive_ztov_daily_diag.csv'
    pd.DataFrame(diag_rows).to_csv(out, index=False)
    print(f'wrote {out}')


# ----------------------------------------------------------------------------
# board
# ----------------------------------------------------------------------------
def fetch_snapshot(D):
    csv = EPM_DIR / f'epm_all_{D}_go0.csv'
    if csv.exists():
        return True
    cmd = [sys.executable, str(ROOT / 'scripts' / 'fetch_epm_data.py'),
           '--dates', D]
    try:
        subprocess.run(cmd, check=True)
        return True
    except (subprocess.CalledProcessError, OSError) as exc:
        print(f'  fetch failed ({exc}); set EPM_API_KEY (repo .env) and run: '
              + ' '.join(cmd) + '\n  falling back to the latest cached '
              'snapshot', flush=True)
        return False


def build_board(args):
    A, H = FIXED_BASE
    scale = args.scale
    D = args.as_of or date.today().isoformat()
    Dint = day_int(D)
    if args.fetch:
        fetch_snapshot(D)
    ds, f = snapshot_at(Dint)
    e = read_epm(f)
    man = json.loads((EPM_DIR / f'{f.stem}.manifest.json').read_text()) \
        if (EPM_DIR / f'{f.stem}.manifest.json').exists() else {}
    print(f'as-of {D}: snapshot {f.name} (provider latest game '
          f'{man.get("provider_latest_game_date")}, {len(e)} players)',
          flush=True)

    seasons = [y for y in range(SEASONS[0], int(D[:4]) + 2)
               if (PROC / f'TOV{y - 2000}.parquet').exists()]
    print(f'loading possession blocks {seasons[0]}-{seasons[-1]} (RS)...',
          flush=True)
    blocks, pidx, P = load_seasons(seasons)
    sea, pm = load_inputs()
    complete = [y for y in seasons if blocks[y]['asof'] <= Dint]
    assert complete, 'no complete season before as-of date'
    have = [y for y in complete if y in KAPPA and (sea['year'] == y).any()]
    yr = max(have)
    if yr < max(complete):
        print(f'  WARNING structural parameters lag: last complete season '
              f'{max(complete)} lacks kappa/lane prices; using {yr}. Rebuild '
              f'the rZTOV explorer payload and extend KAPPA.', flush=True)
    st = Structural(sea, pm, yr, H)
    print(f'  structural {st.describe()}', flush=True)

    train_years = [t for t in seasons if yr - TRAIL < t <= yr]
    live = [t for t in seasons if t > yr and int(blocks[t]['days'].min()) < Dint]
    Xs, ys, ws, levels = [], [], [], {}
    for t in train_years + live:
        b = blocks[t]
        m = b['days'] < Dint
        yt = b['y'][m]
        lvl = float(b['y'].mean()) if t <= yr else \
            season_level(yt, blocks[yr]['y'].mean())
        levels[t] = lvl
        Xs.append(b['X'][np.where(m)[0]])
        ys.append(yt - lvl)
        ws.append(0.5 ** ((Dint - b['days'][m]) / H))
    X = sparse.vstack(Xs, format='csr')
    y = np.concatenate(ys)
    w = np.concatenate(ws)
    active = np.where(np.asarray(X.sum(axis=0)).ravel() > 0)[0]
    Xa = X[:, active]
    Xw = Xa.multiply(w[:, None]).tocsr()
    G = (Xa.T @ Xw).toarray()
    r = np.asarray(Xa.T @ (w * y)).ravel()
    expo = np.asarray(Xa.T @ w).ravel()          # decayed possessions
    raw_poss = np.asarray(Xa.sum(axis=0)).ravel()
    print(f'  training seasons {train_years} + live {live}: {len(y):,} rows, '
          f'{len(active)} columns', flush=True)

    ledger = pz_ledger(e, st, pidx)
    p_new, cov_new = prior_vec(ledger, pidx, P)
    ledger['snapshot'] = ds
    fallback = None
    p_stale, p_fresh = p_new, p_new
    prior_rule = 'latest snapshot (no live season rows)'
    fw = float(args.fresh_weight)
    if live:
        # in-season: the season-start snapshot is the stale prior; the latest
        # snapshot is fresh where it covers a player (in-season snapshots list
        # only players with a completed game) and season-start otherwise.
        # Default fresh weight 0.5: the live rows and EPM's in-season update
        # overlap, so the full swap over-corrects (evaluate --diagnose:
        # optimal weight gamma 0.50 [0.36,0.64] pooled, 0.555 dev; the
        # half blend beats the stale prior in 8/8 seasons, -3.2e-6 pooled).
        f_start, e_start = epm_snapshot(int(blocks[live[0]]['days'].min()))
        if f_start != f.name:
            led0 = pz_ledger(e_start, st, pidx)
            p0, cov0 = prior_vec(led0, pidx, P)
            fallback = f_start
            n_carry = int((cov0 & ~cov_new).sum())
            print(f'  carry-forward from season-start snapshot {f_start}: '
                  f'{n_carry} players not in {ds}', flush=True)
            led0['snapshot'] = f_start
            ledger = ledger.merge(
                led0[['nba_id', 'prior']].rename(columns={'prior': 'prior_start'}),
                on='nba_id', how='outer')
            ledger['snapshot'] = ledger['snapshot'].fillna(f_start)
            for c in ('p_tov_100', 'p_mp_48', 'v_hat', 'ast_share', 'pot_hat',
                      'allowance_hat', 'excess_hat', 'prior'):
                if c in led0.columns:
                    ledger[c] = ledger[c].fillna(
                        ledger['nba_id'].map(led0.set_index('nba_id')[c]))
            p_stale = p0
            p_fresh = carry_forward(p_new, cov_new, p0)
            p_new = (1.0 - fw) * p_stale + fw * p_fresh
            prior_rule = (f'{1 - fw:.2f} * season-start {f_start} + {fw:.2f} '
                          f'* latest {f.name} (carry-forward)')
        else:
            prior_rule = f'season-start snapshot {f.name} (no in-season snapshot yet)'
    print(f'  prior rule: {prior_rule}', flush=True)
    if 'prior_start' not in ledger.columns:
        ledger['prior_start'] = ledger['prior']
    p_pz = scale * p_new
    p_mix, n_mix = build_mix_prior(sea, yr, H, KAPPA[yr], pidx, P)
    p_mix = MIX_SCALE * p_mix
    b_plain = np.zeros(2 * P)
    b_pz = np.zeros(2 * P)
    b_mix = np.zeros(2 * P)
    b_stale = np.zeros(2 * P)
    b_fresh = np.zeros(2 * P)
    b_plain[active] = solve(G, r, A)
    b_pz[active] = solve(G, r, A, prior=p_pz[active])
    b_mix[active] = solve(G, r, A, prior=p_mix[active])
    if live and fallback is not None:
        b_stale[active] = solve(G, r, A, prior=scale * p_stale[active])
        b_fresh[active] = solve(G, r, A, prior=scale * p_fresh[active])
    else:
        b_stale, b_fresh = b_pz, b_pz

    inv = {i: pid for pid, i in pidx.items()}
    off = active[active < P]
    pos = {c: i for i, c in enumerate(active)}
    live_poss = np.zeros(2 * P)
    for t in live:
        b = blocks[t]
        m = b['days'] < Dint
        live_poss += np.asarray(b['X'][np.where(m)[0]].sum(axis=0)).ravel()
    rows = pd.DataFrame({
        'nba_id': [inv[i] for i in off],
        'off_pz': b_pz[off],
        'off_plain': b_plain[off],
        'off_mix': b_mix[off],
        'off_pz_stale': b_stale[off],
        'off_pz_fresh': b_fresh[off],
        'prior_pz': p_pz[off],
        'prior_pz_stale': scale * p_stale[off],
        'prior_pz_fresh': scale * p_fresh[off],
        'prior_mix': p_mix[off],
        'def_pz': [b_pz[i + P] for i in off],
        'def_plain': [b_plain[i + P] for i in off],
        'def_mix': [b_mix[i + P] for i in off],
        'off_poss_decayed': [expo[pos[i]] for i in off],
        'off_poss_window': [raw_poss[pos[i]] for i in off],
        'off_poss_live': [live_poss[i] for i in off],
        'def_poss_window': [raw_poss[pos[i + P]] if (i + P) in pos else 0.0
                            for i in off],
    })
    rows = rows.merge(ledger, on='nba_id', how='left')
    # names: rZTOV research surface -> daily weighted-factors exports
    # (fringe players below the research volume floor) -> EPM player_name
    names = pd.read_csv(RES / 'rztov_research_22_26.csv',
                        usecols=['nba_id', 'name']).drop_duplicates('nba_id')
    wf_names = []
    for wf in sorted((ROOT / 'master_results').glob('weighted_factors_*.csv')):
        try:
            w = pd.read_csv(wf, usecols=['player_id', 'player_name'])
        except (ValueError, OSError):
            continue
        wf_names.append(w.rename(columns={'player_id': 'nba_id',
                                          'player_name': 'wf_name'}))
    epm_names = pd.read_csv(f, usecols=['player_id', 'player_name']) \
        .rename(columns={'player_id': 'nba_id', 'player_name': 'epm_name'}) \
        .drop_duplicates('nba_id')
    rows = rows.merge(names, on='nba_id', how='left') \
        .merge(epm_names, on='nba_id', how='left')
    if wf_names:
        wf_all = pd.concat(wf_names).dropna().drop_duplicates('nba_id')
        rows = rows.merge(wf_all, on='nba_id', how='left')
        rows['name'] = rows['name'].fillna(rows['wf_name'])
        rows = rows.drop(columns=['wf_name'])
    rows['name'] = rows['name'].fillna(rows['epm_name'])
    rows = rows.drop(columns=['epm_name'])
    rows['pz_covered'] = rows['snapshot'].notna()
    rows['mix_covered'] = rows['prior_mix'] != 0
    rows['as_of'] = D
    rows['structural_year'] = yr
    lead = ['name', 'nba_id', 'off_pz', 'off_plain', 'off_mix', 'prior_pz',
            'prior_mix', 'def_pz', 'def_plain', 'def_mix']
    rows = rows[lead + [c for c in rows.columns if c not in lead]]
    rows = rows.sort_values('off_pz').reset_index(drop=True)

    out = RES / f'tov_predictive_ztov_board_{D}.csv'
    rows.to_csv(out, index=False)
    latest = RES / 'tov_predictive_ztov_board_latest.csv'
    rows.to_csv(latest, index=False)
    manifest = {
        'as_of': D,
        'snapshot_requested_date': ds,
        'snapshot_file': f.name,
        'fallback_snapshot': fallback,
        'provider_latest_game_date': man.get('provider_latest_game_date'),
        'structural_year': yr,
        'kappa': st.kappa,
        'lane_prices': {k: float(v) for k, v in st.prices.items()},
        'vol_map_tsa': [float(v) for v in st.vol],
        'ast_map_pot': [float(v) for v in st.ast],
        'league_ast_share': float(st.league),
        'alpha': A, 'half_life': H, 'pz_scale': scale,
        'mix_scale': MIX_SCALE,
        'prior_rule': prior_rule,
        'fresh_weight': fw,
        'train_years': train_years, 'live_seasons': live,
        'levels': {str(k): float(v) for k, v in levels.items()},
        'n_train_rows': int(len(y)), 'n_columns': int(len(active)),
        'n_offense_rows': int(len(rows)),
        'n_pz_covered': int(rows['pz_covered'].sum()),
        'n_mix_covered': int(rows['mix_covered'].sum()),
        'n_epm_rows': int(len(e)),
        'sign': 'positive off = commits more TOV per 100 than lineup context '
                'predicts; positive def = forces more',
        'lane': 'research only; not a production or published surface',
    }
    (RES / f'tov_predictive_ztov_board_{D}.manifest.json').write_text(
        json.dumps(manifest, indent=2) + '\n')
    (RES / 'tov_predictive_ztov_board_latest.manifest.json').write_text(
        json.dumps(manifest, indent=2) + '\n')
    print(f'wrote {out} ({len(rows)} offense rows; pz covered '
          f'{manifest["n_pz_covered"]}, mix covered {manifest["n_mix_covered"]})')

    show = rows[rows['off_poss_window'] >= args.min_poss]
    cols = ['name', 'off_pz', 'off_plain', 'prior_pz', 'off_mix',
            'off_poss_window']
    with pd.option_context('display.width', 140, 'display.max_columns', 20):
        print(f'\nlowest off_pz (fewest TOV vs context), poss >= '
              f'{args.min_poss}:')
        print(show.head(15)[cols].to_string(index=False,
                                             float_format=lambda v: f'{v:.3f}'))
        print(f'\nhighest off_pz:')
        print(show.tail(15)[cols].to_string(index=False,
                                             float_format=lambda v: f'{v:.3f}'))
    both = rows['pz_covered'] & rows['mix_covered']
    print(f'\ncorr(off_pz, off_plain) {rows["off_pz"].corr(rows["off_plain"]):.3f}'
          f'  corr(off_pz, off_mix) {rows["off_pz"].corr(rows["off_mix"]):.3f}'
          f'  corr(prior_pz, prior_mix | both) '
          f'{rows.loc[both, "prior_pz"].corr(rows.loc[both, "prior_mix"]):.3f}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    ev = sub.add_parser('evaluate', help='within-season rolling forward test')
    ev.add_argument('--boot-draws', type=int, default=BOOT_DRAWS)
    ev.add_argument('--seasons', type=str, default=None,
                    help='comma list of scored seasons (default 2019-2026)')
    ev.add_argument('--diagnose', action='store_true',
                    help='add stale/fresh weighting variants, prior-movement '
                         'residual slopes, exposure quintiles and per-player '
                         'ledger drift; writes *_diag outputs, leaves the '
                         'main eval files untouched')
    bd = sub.add_parser('board', help='as-of predictive-ledger TOV board')
    bd.add_argument('--as-of', type=str, default=None, help='YYYY-MM-DD')
    bd.add_argument('--fetch', action='store_true',
                    help='fetch the as-of snapshot first (needs EPM_API_KEY)')
    bd.add_argument('--scale', type=float, default=PZ_SCALE)
    bd.add_argument('--fresh-weight', type=float, default=0.5,
                    help='in-season weight on the latest snapshot vs the '
                         'season-start snapshot (0 = stale/W1, 1 = full '
                         'fresh/W2; default 0.5 = the dev-supported blend). '
                         'Ignored when no live season rows exist.')
    bd.add_argument('--min-poss', type=float, default=2000.0,
                    help='window offensive possessions for the printed lists')
    args = ap.parse_args()
    if args.cmd == 'evaluate':
        evaluate(args)
    else:
        build_board(args)


if __name__ == '__main__':
    sys.exit(main())
