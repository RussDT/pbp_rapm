"""zTOV v3: fixed TSA-basis Genius shot files (fouled attempts included), three buckets,
refit box + joint models, recompute volume-curve and within-player findings, regenerate outputs."""
import json
import re
import unicodedata
import numpy as np
import pandas as pd

RAPM = '/Users/russ/Documents/Codex/basketball/nba/research/rapm'
GEN = f'{RAPM}/geniussports'
PANEL = '/Users/russ/Documents/Codex/basketball/nba/research/vpm/boxscore.csv'
RES = f'{RAPM}/nba_pipeline/results'
SCRATCH = ('/private/tmp/claude-501/-Users-russ-Documents-Codex-basketball-nba-research-rapm/'
           '355347cc-62b1-448e-9dda-ce36f24e9a96/scratchpad')

SUFFIX = {'jr', 'sr', 'ii', 'iii', 'iv', 'v'}

def norm_name(s):
    s = unicodedata.normalize('NFD', str(s))
    s = ''.join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[.'’]", '', s.lower())
    s = re.sub(r'-', ' ', s)
    return ' '.join(t for t in s.split() if t not in SUFFIX)

def wls(y, X, w, intercept=True):
    cols = [np.ones(len(y))] if intercept else []
    Xd = np.column_stack(cols + [np.asarray(X[c], float) for c in X.columns])
    sw = np.sqrt(np.asarray(w, float))
    beta, *_ = np.linalg.lstsq(Xd * sw[:, None], np.asarray(y, float) * sw, rcond=None)
    resid = np.asarray(y, float) - Xd @ beta
    r2 = 1 - (resid**2 * w).sum() / (((y - np.average(y, weights=w))**2 * w).sum())
    r = np.corrcoef(Xd @ beta, y)[0, 1]
    return beta, r2, r

def counts(year):
    d = pd.read_parquet(f'{RAPM}/nba_pipeline/raw_data/NBA{year:02d}.parquet',
                        columns=['event_type', 'player1_id', 'home_description',
                                 'visitor_description', 'neutral_description'])
    et = pd.to_numeric(d['event_type'], errors='coerce')
    p1 = pd.to_numeric(d['player1_id'], errors='coerce').fillna(0).astype('int64')
    desc = (d['home_description'].fillna('') + ' ' + d['visitor_description'].fillna('')
            + ' ' + d['neutral_description'].fillna(''))
    ok = (p1 > 0) & (p1 < 1_000_000_000)
    t = ok & (et == 5) & ~desc.str.contains('No Turnover', case=False)
    bp = t & desc.str.contains('Bad Pass', case=False)
    out = pd.DataFrame({'nba_id': p1, 'n_tov': t.astype(int), 'n_bp': bp.astype(int)})
    out = out[ok].groupby('nba_id').sum().reset_index()
    out['n_sc'] = out['n_tov'] - out['n_bp']
    out['year'] = 2000 + year
    return out

cnt = pd.concat([counts(y) for y in range(14, 27)], ignore_index=True)
panel = pd.read_csv(PANEL, usecols=['nba_id', 'year', 'Name', 'playoffs', 'OffPoss'], low_memory=False)
panel = panel[(panel['playoffs'] == 0) & panel['OffPoss'].gt(0)].copy()
panel['nba_id'] = panel['nba_id'].astype('int64')
panel['norm'] = panel['Name'].map(norm_name)
panel.loc[panel['norm'] == 'enes kanter', 'norm'] = 'enes freedom'
panel.loc[(panel['norm'] == 'kenyon martin') & (panel['year'] >= 2020), 'norm'] = 'kj martin'
panel.loc[panel['norm'] == 'juan hernangomez', 'norm'] = 'juancho hernangomez'

def load_gen(fname, actor, cols):
    g = pd.read_csv(f'{GEN}/{fname}')
    g = g[g['Season'].str.match(r'\d{4}-\d{2}')].copy()
    g['year'] = g['Season'].str[:4].astype(int) + 1
    g['norm'] = g[actor].map(norm_name)
    return g[['norm', 'year'] + list(cols)].rename(columns=dict(cols))

gen = load_gen('shot_fixed_selfcreated_rs.csv', 'Shooter',
               {'Shots Attempted Per 100 Possessions': 'v_self'}) \
    .merge(load_gen('shot_fixed_assisted_rs.csv', 'Shooter',
                    {'Shots Attempted Per 100 Possessions': 'v_ast'}),
           on=['norm', 'year'], how='outer') \
    .merge(load_gen('pass_rs.csv', 'Passer',
                    {'Potential Assists Per 100 Possessions': 'v_potast'}),
           on=['norm', 'year'], how='outer').drop_duplicates(['norm', 'year'])
gen['matched'] = 1

df = panel.merge(cnt, on=['nba_id', 'year'], how='left')
for c in ['n_tov', 'n_bp', 'n_sc']:
    df[c] = df[c].fillna(0)
df = df.merge(gen, on=['norm', 'year'], how='left')
df = df[df['matched'].notna() & (df['OffPoss'] > 300)].copy()
for c in ['v_self', 'v_ast', 'v_potast']:
    df[c] = df[c].fillna(0)
df['sc_100'] = df['n_sc'] / df['OffPoss'] * 100
df['bp_100'] = df['n_bp'] / df['OffPoss'] * 100
big = df['OffPoss'] > 1000

# ---- 22-26 aggregates + RAPM
w22 = df[df['year'].between(2022, 2026)]
agg = w22.groupby('nba_id').apply(
    lambda g: pd.Series({
        'poss': g['OffPoss'].sum(),
        'sc_100': g['n_sc'].sum() / g['OffPoss'].sum() * 100,
        'bp_100': g['n_bp'].sum() / g['OffPoss'].sum() * 100,
        **{c: np.average(g[c], weights=g['OffPoss']) for c in ['v_self', 'v_ast', 'v_potast']},
    }), include_groups=False).reset_index()
sc_r = pd.read_csv(f'{RES}/alt_scoring_tov_value_22_26_all_rb_se_a2000_4000_results.csv')
bp_r = pd.read_csv(f'{RES}/alt_badpass_tov_value_22_26_all_rb_se_a2000_4000_results.csv')
agg = agg.merge(sc_r[['player_id', 'off']].rename(columns={'player_id': 'nba_id', 'off': 'sc_rapm'}), on='nba_id') \
         .merge(bp_r[['player_id', 'off']].rename(columns={'player_id': 'nba_id', 'off': 'bp_rapm'}), on='nba_id')
agg = agg[agg['poss'] >= 2000].copy()
W = agg['poss']

# ---- box-space expected rates on fixed volumes
bs, r2s, _ = wls(agg['sc_100'], agg[['v_self', 'v_ast']], W)
bpb, r2pb, _ = wls(agg['bp_100'], agg[['v_potast']], W)
print(f'box scoring: base {bs[0]:.3f} self {100*bs[1]:.2f}% ast {100*bs[2]:.2f}% (R2 {r2s:.3f})')
print(f'box badpass: base {bpb[0]:.3f} potast {100*bpb[1]:.2f}% (R2 {r2pb:.3f})')

# ---- per-target impact solves (for the impact column)
bi_sc, r2i_sc, _ = wls(agg['sc_rapm'], agg[['sc_100', 'v_self', 'v_ast']], W)
bi_bp, r2i_bp, _ = wls(agg['bp_rapm'], agg[['bp_100', 'v_potast']], W)
be_sc = bi_sc[2:] / -bi_sc[1]
be_bp = bi_bp[2] / -bi_bp[1]
print(f'impact per-target: sc tov {bi_sc[1]:.3f} be self {100*be_sc[0]:.1f}% ast {100*be_sc[1]:.1f}% '
      f'(R2 {r2i_sc:.3f}) | bp tov {bi_bp[1]:.3f} be potast {100*be_bp:.1f}% (R2 {r2i_bp:.3f})')

# ---- joint solve
tot = agg['sc_rapm'] + agg['bp_rapm']
act = agg['sc_100'] + agg['bp_100']
bj, r2j, rj = wls(tot, pd.DataFrame({'act': act, 's': agg['v_self'], 'a': agg['v_ast'],
                                     'p': agg['v_potast']}), W)
aj, kj = bj[0], bj[1]
be = bj[2:] / -kj
print(f'joint: a={aj:+.4f} kappa={kj:.4f} R2w={r2j:.3f} r={rj:+.3f} | '
      f'breakevens self {100*be[0]:.2f}% ast {100*be[1]:.2f}% potast {100*be[2]:.2f}%')

def row(y, cols):
    _, r2, r = wls(y, agg[cols], W)
    return round(r2, 3), round(r, 3)
fit_sc = [['actual scoring TOV/100 alone', *row(agg['sc_rapm'], ['sc_100'])],
          ['volumes alone (self/ast)', *row(agg['sc_rapm'], ['v_self', 'v_ast'])],
          ['rate + volumes (main)', *row(agg['sc_rapm'], ['sc_100', 'v_self', 'v_ast'])],
          ['+ potential assists', *row(agg['sc_rapm'], ['sc_100', 'v_self', 'v_ast', 'v_potast'])]]
fit_bp = [['actual badpass TOV/100 alone', *row(agg['bp_rapm'], ['bp_100'])],
          ['rate + potential assists (main)', *row(agg['bp_rapm'], ['bp_100', 'v_potast'])],
          ['+ shot volumes too', *row(agg['bp_rapm'], ['bp_100', 'v_potast', 'v_self', 'v_ast'])]]
_, r2c, rc = wls(tot, agg[['sc_100', 'bp_100', 'v_self', 'v_ast', 'v_potast']], W)
print(f'combined unrestricted R2w={r2c:.3f}')

# ---- validation on the new definition
def enrich_frame(fr):
    fr = fr.copy()
    fr['exp'] = (bs[0] + bpb[0] + bs[1]*fr['v_self'] + bs[2]*fr['v_ast'] + bpb[1]*fr['v_potast'])
    fr['act'] = fr['sc_100'] + fr['bp_100']
    fr['involve'] = fr[['v_self', 'v_ast', 'v_potast']].sum(axis=1)
    fr['ztov'] = (fr['act'] - fr['exp']) / fr['involve'].clip(lower=1) * 100
    return fr
agg = enrich_frame(agg)
r_raw = np.corrcoef(agg['act'], tot)[0, 1]
r_naive = np.corrcoef(agg['act'] / (agg['v_self'] + agg['v_ast']).clip(lower=1), tot)[0, 1]
r_ztov = np.corrcoef(agg['act'] - agg['exp'], tot)[0, 1]
rows = enrich_frame(df[big])
lag = rows[['nba_id', 'year', 'ztov', 'OffPoss']].copy(); lag['year'] += 1
both = rows.merge(lag, on=['nba_id', 'year'], suffixes=('', '_p'))
both = both[(both['OffPoss'] > 1500) & (both['OffPoss_p'] > 1500)]
yoy = both['ztov'].corr(both['ztov_p'])
print(f'validation: raw {r_raw:+.3f} naive {r_naive:+.3f} ztov {r_ztov:+.3f} yoy {yoy:+.3f}')

# ---- volume curve + within-player on fixed volumes
curve = []
for lo, hi, lab in [(3, 6, '3–6'), (6, 10, '6–10'), (10, 15, '10–15'),
                    (15, 20, '15–20'), (20, 99, '20+')]:
    m = rows[(rows['v_self'] >= lo) & (rows['v_self'] < hi)]
    wm = m['OffPoss']
    attrib = ((m['sc_100'] - bs[0] - bs[2]*m['v_ast']) * wm).sum() / (m['v_self'] * wm).sum()
    curve.append([lab, round(100*attrib, 1)])
for c in ['sc_100', 'v_self', 'v_ast', 'v_potast', 'bp_100']:
    both[f'd_{c}'] = both[c] - both[f'{c}_p'] if f'{c}_p' in both else np.nan
lag2 = rows[['nba_id', 'year', 'sc_100', 'bp_100', 'v_self', 'v_ast', 'v_potast', 'OffPoss']].copy()
lag2['year'] += 1
b2 = rows.merge(lag2, on=['nba_id', 'year'], suffixes=('', '_p'))
b2 = b2[(b2['OffPoss'] > 1500) & (b2['OffPoss_p'] > 1500)]
for c in ['sc_100', 'bp_100', 'v_self', 'v_ast', 'v_potast']:
    b2[f'd_{c}'] = b2[c] - b2[f'{c}_p']
wmin = np.minimum(b2['OffPoss'], b2['OffPoss_p'])
bw, _, _ = wls(b2['d_sc_100'], b2[['d_v_self', 'd_v_ast']], wmin)
bwp, _, _ = wls(b2['d_bp_100'], b2[['d_v_potast']], wmin)
print(f'curve: {curve}')
print(f'within-player: self {100*bw[1]:.1f}% ast {100*bw[2]:.1f}% potast {100*bwp[1]:.1f}% (n={len(b2)})')

# ---- outputs
names = (df.sort_values('year').groupby('nba_id')['Name'].last())
seasons = [[int(r.nba_id), int(r.year), int(r.OffPoss),
            round(float(r.v_self), 2), round(float(r.v_ast), 2), round(float(r.v_potast), 2),
            round(float(r.sc_100), 3), round(float(r.bp_100), 3)]
           for r in df.itertuples()]
board = [[int(r.nba_id), int(r.poss),
          round(float(r.v_self), 2), round(float(r.v_ast), 2), round(float(r.v_potast), 2),
          round(float(r.sc_100), 3), round(float(r.bp_100), 3),
          round(float(r.sc_rapm), 3), round(float(r.bp_rapm), 3)]
         for r in agg.itertuples()]

data = {
    'built': '2026-08-09 (fixed TSA-basis shot files)',
    'window': '2021-22 through 2025-26 (aggregate board); seasons 2013-14 through 2025-26',
    'rates': {'sc': {'base': round(bs[0], 3), 'self': round(bs[1], 4), 'ast': round(bs[2], 4)},
              'bp': {'base': round(bpb[0], 3), 'potast': round(bpb[1], 4)}},
    'impact': {'sc': {'const': round(bi_sc[0], 4), 'tov': round(bi_sc[1], 4),
                      'r2': round(r2i_sc, 3), 'be_self': round(100*be_sc[0], 2),
                      'be_ast': round(100*be_sc[1], 2)},
               'bp': {'const': round(bi_bp[0], 4), 'tov': round(bi_bp[1], 4),
                      'r2': round(r2i_bp, 3), 'be_potast': round(100*be_bp, 2)}},
    'joint': {'a': round(aj, 4), 'k': round(kj, 4), 'r2': round(r2j, 3),
              'be': {'self': round(be[0], 4), 'ast': round(be[1], 4), 'potast': round(be[2], 4)},
              'unrestricted_r2': round(r2c, 3)},
    'validation': {'corr_raw': round(r_raw, 3), 'corr_naive': round(r_naive, 3),
                   'corr_ztov': round(r_ztov, 3), 'yoy': round(yoy, 3), 'n': int(len(agg))},
    'fit_table': {'sc': fit_sc, 'bp': fit_bp},
    'volume_curve': curve,
    'within_player': {'self': round(100*bw[1], 1), 'ast': round(100*bw[2], 1),
                      'potast': round(100*bwp[1], 1), 'n': int(len(b2))},
    'league': {'sc_per_teamgame': 6.90, 'bp_per_teamgame': 6.89, 'season': '2025-26'},
    'names': {str(int(k)): v for k, v in names.items()},
    'board': board,
    'seasons': seasons,
}
with open(f'{SCRATCH}/ztov_explorer_data.json', 'w') as f:
    json.dump(data, f, separators=(',', ':'))
out = agg[['nba_id', 'poss', 'v_self', 'v_ast', 'v_potast', 'sc_100', 'bp_100',
           'act', 'exp', 'ztov', 'involve', 'sc_rapm', 'bp_rapm']].copy()
out['name'] = out['nba_id'].map(names)
out.round(4).sort_values('ztov').to_csv(f'{RES}/ztov_research_22_26.csv', index=False)
import os
print(f"wrote JSON ({os.path.getsize(f'{SCRATCH}/ztov_explorer_data.json')/1024:.0f} KB), "
      f"CSV ({len(out)} players)")
sga = agg[agg['nba_id'] == 1628983]
if len(sga):
    r = sga.iloc[0]
    print(f"SGA 22-26: self {r.v_self:.1f} ast {r.v_ast:.1f} potast {r.v_potast:.1f} "
          f"involve {r.involve:.1f} ztov {r.ztov:+.2f}")
EOF = None
