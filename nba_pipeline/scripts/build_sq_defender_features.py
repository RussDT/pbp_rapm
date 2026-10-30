"""Build defender-season prior features from the ShotQuality parquet store.

Reads `nba/research/shotquality/store/shots_{season}.parquet`, written by
`scripts/sync_shotquality.py`. The previous version of this script scanned 2,740
ragged per-game CSVs and unioned their schemas on every run; the store does that
once at write time.

Switching to the store also recovers a whole feature family. The per-game CSVs
carried velocity/acceleration columns in only 951 of 2,740 files, so a scan
measured "closing speed" at ~2% populated and it was dropped. In the store those
fields are present on ~75% of shots inside covered games.

Three arms, always reported separately:

  geom     pre-release position: contest distance, angle, behind/inside/blocking,
           bounding-box height difference, defenders closer to the basket, diet
  effort   pre-release motion: closing velocity toward the shooter, acceleration
           toward the shooter, defender speed. Never tested before.
  outcome  lagged realized value: points allowed, provider EV allowed, and
           suppression = EV allowed - points allowed, split rim/mid/three

Only `outcome` touches the result of the shot. Geometry and effort are known at
release, so a consumer that wants a leak-free feature set can take those two.

Usage:
    python build_sq_defender_features.py all --min-shots 100
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SQ = Path('/Users/russ/Documents/Codex/basketball/nba/research/shotquality')
STORE = SQ / 'store'
RES = Path(__file__).resolve().parents[1] / 'results'
FS = 'feature_store.'
KEEP_TYPES = {'2pt', '3pt'}

# provider column (minus the feature_store. prefix) -> output name
GEOM = {
    'Distance to the Closest Defender (Feet)': 'd1_dist',
    'Distance to the Second-Closest Defender (Feet)': 'd2_dist',
    'Cosine of the Angle with the Closest Defender': 'd1_cos',
    'Cosine of the Angle with the Second-Closest Defender': 'd2_cos',
    'Closest Defender Behind': 'd1_behind',
    'Closest Defender Inside': 'd1_inside',
    'Closest Defender Blocking': 'd1_blocking',
    'Second-Closest Defender Behind': 'd2_behind',
    'Second-Closest Defender Inside': 'd2_inside',
    'Number of Defenders Closer to the Basket': 'n_closer',
    'Harmonic Mean Defender Distance': 'harm_dist',
    'Average Defender Distance (Feet)': 'avg_dist',
    'Height of the Closest Defender (Inches)': 'd1_height',
    'Weight of the Closest Defender (Pounds)': 'd1_weight',
    'Difference in Bounding Box Height with the Closest Defender (Pixels)': 'd1_bbox_diff',
    'Height of the Closest Defender\'s Bounding Box (Pixels)': 'd1_bbox',
    'Shot Distance (ft.)': 'shot_dist',
    'Shooter in the Paint': 'in_paint',
    'Shooter in Restricted Area': 'in_ra',
    'Defenders in Paint': 'def_in_paint',
    'Defenders in the Restricted Area': 'def_in_ra',
    'Number of Teammates in the Paint': 'tm_in_paint',
}
EFFORT = {
    'Closest Defender Velocity to the Shooter (ft/s)': 'd1_v_shooter',
    'Closest Defender Acceleration to the Shooter (ft/s^2)': 'd1_a_shooter',
    'Closest Defender Velocity to the Basket (ft/s)': 'd1_v_basket',
    'Closest Defender Velocity (ft/s)': 'd1_speed',
    'Closest Defender Acceleration (ft/s^2)': 'd1_accel',
    'Second-Closest Defender Velocity to the Shooter (ft/s)': 'd2_v_shooter',
    'Shooter Velocity (ft/s)': 'sh_speed',
    'Shooter Velocity to the Basket (ft/s)': 'sh_v_basket',
}
FLAT = {
    'contest_score': 'contest',
    'spacing': 'spacing',
    'ev.initial_ev.value': 'ev',
    'shot_taking.initial_shot_taking.value': 'shot_taking',
}


def bridge():
    """ShotQuality internal player_id -> NBA player id, with a name fallback."""
    m = pd.read_csv(SQ / 'nba_id_to_player_id_mapping.csv').dropna(subset=['player_id', 'nba_id'])
    out = dict(zip(m['player_id'].astype('int64'), m['nba_id'].astype('int64')))
    direct = len(out)
    try:
        auto = pd.read_csv('/Users/russ/Documents/Codex/basketball/data/csvs/autocomplete_map.csv')
        ncol = next(c for c in auto.columns if 'name' in c.lower())
        icol = next(c for c in auto.columns if 'id' in c.lower())
        by_name = {}
        for n, i in zip(auto[ncol], auto[icol]):
            by_name.setdefault(str(n).strip().lower(), i)
        pl = pd.read_csv(SQ / 'players.csv', low_memory=False)
        for pid, nm in zip(pl['player_id'], pl['player_name']):
            pid = int(pid)
            if pid in out or not isinstance(nm, str):
                continue
            hit = by_name.get(nm.strip().lower())
            if hit is not None and not pd.isna(hit):
                out[pid] = int(hit)
        print(f'  bridge: {direct} direct + {len(out) - direct} name-matched = {len(out)}')
    except Exception as e:
        print(f'  bridge: {direct} direct (name fallback unavailable: {e})')
    return out


def extract():
    parts = sorted(STORE.glob('shots_*.parquet'))
    if not parts:
        sys.exit(f'no parquet store at {STORE}; run sync_shotquality.py first')
    b = bridge()
    RES.mkdir(exist_ok=True)
    for p in parts:
        d = pd.read_parquet(p)
        season = int(d['season'].iloc[0])
        d = d[d['action_type'].isin(KEEP_TYPES)].copy()
        o = pd.DataFrame({
            'game_id': d['game_id'].values, 'season': season,
            'shooter_sq': d['player_id'].values, 'team_id': d['team_id'].values,
            'defender_sq': d['defender_id'].values,
            'is_three': (d['action_type'] == '3pt').values,
            'made': pd.to_numeric(d['made'], errors='coerce').values,
        })
        missing = []
        for src, dst in FLAT.items():
            o[dst] = pd.to_numeric(d[src], errors='coerce').values if src in d else np.nan
        for group in (GEOM, EFFORT):
            for src, dst in group.items():
                col = FS + src
                if col in d.columns:
                    o[dst] = pd.to_numeric(d[col], errors='coerce').values
                else:
                    o[dst] = np.nan
                    missing.append(dst)
        o['shooter_id'] = o['shooter_sq'].map(b)
        o['defender_id'] = o['defender_sq'].map(b)
        o['pts'] = o['made'].fillna(0) * np.where(o['is_three'], 3, 2)
        out = RES / f'sq_defender_shots_{season % 100}.parquet'
        o.to_parquet(out, index=False)
        cov = o['defender_id'].notna().mean()
        eff = o['d1_v_shooter'].notna().mean()
        print(f'  season {season}: {len(o):,} FGA -> {out.name}  '
              f'defender {100 * cov:.1f}%  closing-velocity {100 * eff:.1f}%')
        if missing:
            print(f'    absent from store: {sorted(set(missing))}')


# The provider's motion fields carry a ~1% tail of corrupt values: d1_speed peaks
# at 53,629 ft/s and rim closing velocity at -2,268 ft/s. An NBA player tops out
# near 25 ft/s. Percentiles are sane through p99 and explode by p99.9, so this is
# a tracking glitch, not a real extreme. Set out-of-envelope values to missing
# rather than clipping them, which would pile garbage onto the boundary and still
# drag the mean. Without this the effort arm's RMSE doubles.
PHYSICAL = {
    'd1_speed': (0, 30), 'sh_speed': (0, 30),
    'd1_v_shooter': (-30, 30), 'd2_v_shooter': (-30, 30),
    'd1_v_basket': (-30, 30), 'sh_v_basket': (-30, 30),
    'd1_a_shooter': (-40, 40), 'd1_accel': (-40, 40),
}


def sanitize(o: pd.DataFrame) -> pd.DataFrame:
    dropped = {}
    for c, (lo, hi) in PHYSICAL.items():
        if c not in o:
            continue
        v = o[c]
        bad = v.notna() & ((v < lo) | (v > hi))
        if bad.any():
            dropped[c] = int(bad.sum())
            o.loc[bad, c] = np.nan
    if dropped:
        tot = sum(dropped.values())
        print(f'    dropped {tot:,} implausible motion values: '
              + ', '.join(f'{k}={v}' for k, v in sorted(dropped.items())))
    return o


def wmean(g, col):
    v = g[col]
    return float(v.mean()) if v.notna().any() else np.nan


def game_dates() -> pd.Series:
    """SQ game_id -> Eastern game date, from the sync store."""
    g = pd.read_parquet(SQ / 'store' / 'games.parquet',
                        columns=['game_id', 'game_datetime_utc'])
    dt = pd.to_datetime(g['game_datetime_utc'], errors='coerce', utc=True)
    return pd.Series(dt.dt.tz_convert('US/Eastern').dt.normalize().values,
                     index=g['game_id'].astype('int64'))


def features(min_shots, half=None, out_name=None):
    parts = sorted(RES.glob('sq_defender_shots_*.parquet'))
    if not parts:
        sys.exit('no extracted shots; run `extract` first')
    d = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    d = d[d['defender_id'].notna()].copy()
    d['defender_id'] = d['defender_id'].astype('int64')
    d = sanitize(d)

    if half:
        # Split-half gate (docs/SQ_DEFENDER_DTS_PRIOR.md step 1). The cut date is
        # the median *possession* date from the matching TS solve, not the median
        # SQ shot date, so features and target are split on the same boundary.
        dates = d['game_id'].map(game_dates())
        if dates.isna().any():
            sys.exit(f'{int(dates.isna().sum())} shots have no game date')
        cut = pd.Timestamp(half[1])          # game_dates() returns tz-naive Eastern
        keep = dates <= cut if half[0] == 'h1' else dates > cut
        print(f'  {half[0]}: keeping {int(keep.sum()):,} of {len(d):,} shots '
              f'(cut {cut:%Y-%m-%d})')
        d = d[keep].copy()
    d['is_rim'] = (d['shot_dist'] <= 4) & ~d['is_three']
    d['is_mid'] = ~d['is_rim'] & ~d['is_three']
    d['ev_resid'] = d['pts'] - d['ev']

    out = []
    for (pid, season), g in d.groupby(['defender_id', 'season']):
        if len(g) < min_shots:
            continue
        r = {'nba_id': int(pid), 'season': int(season), 'shots_defended': len(g)}
        # role / diet
        r['rim_share'] = float(g['is_rim'].mean())
        r['mid_share'] = float(g['is_mid'].mean())
        r['three_share'] = float(g['is_three'].mean())
        # --- geometry ---
        r['mean_dist'] = wmean(g, 'd1_dist')
        r['pct_tight'] = float((g['d1_dist'] <= 4).mean())
        r['pct_wide'] = float((g['d1_dist'] >= 8).mean())
        r['mean_contest'] = wmean(g, 'contest')
        r['mean_bbox_diff'] = wmean(g, 'd1_bbox_diff')
        r['pct_blocking'] = wmean(g, 'd1_blocking')
        r['pct_behind'] = wmean(g, 'd1_behind')
        r['pct_inside'] = wmean(g, 'd1_inside')
        r['mean_cos'] = wmean(g, 'd1_cos')
        r['mean_spacing'] = wmean(g, 'spacing')
        r['mean_n_closer'] = wmean(g, 'n_closer')
        r['pct_alone'] = float((g['n_closer'] <= 1).mean())
        r['mean_def_in_paint'] = wmean(g, 'def_in_paint')
        r['pct_vs_ra'] = wmean(g, 'in_ra')
        r['mean_d2_dist'] = wmean(g, 'd2_dist')
        # --- effort / motion (new) ---
        r['eff_n'] = int(g['d1_v_shooter'].notna().sum())
        r['mean_closing_v'] = wmean(g, 'd1_v_shooter')
        r['mean_closing_a'] = wmean(g, 'd1_a_shooter')
        r['pct_closing'] = float((g['d1_v_shooter'] > 0).mean())
        r['mean_d1_speed'] = wmean(g, 'd1_speed')
        r['mean_d1_v_basket'] = wmean(g, 'd1_v_basket')
        r['mean_d2_closing_v'] = wmean(g, 'd2_v_shooter')
        r['mean_sh_speed'] = wmean(g, 'sh_speed')
        # --- outcome ---
        r['ev_allowed'] = wmean(g, 'ev')
        r['pts_allowed'] = float(g['pts'].mean())
        r['ev_suppression'] = -wmean(g, 'ev_resid')
        r['shot_taking_allowed'] = wmean(g, 'shot_taking')
        for nm, m in (('rim', g['is_rim']), ('mid', g['is_mid']), ('three', g['is_three'])):
            s = g[m]
            r[f'{nm}_n'] = int(len(s))
            r[f'{nm}_dist'] = wmean(s, 'd1_dist') if len(s) else np.nan
            r[f'{nm}_ev_allowed'] = wmean(s, 'ev') if len(s) else np.nan
            r[f'{nm}_suppression'] = -wmean(s, 'ev_resid') if len(s) else np.nan
            r[f'{nm}_closing_v'] = wmean(s, 'd1_v_shooter') if len(s) else np.nan
        out.append(r)

    f = pd.DataFrame(out)
    p = RES / (out_name or 'sq_defender_season_features.csv')
    f.sort_values(['season', 'shots_defended'], ascending=[True, False]).to_csv(p, index=False)
    print(f'{len(f)} defender-seasons (>= {min_shots} shots) -> {p}')
    print(f.groupby('season')['shots_defended'].agg(['count', 'median', 'max']).to_string())
    print(f'closing-velocity populated on {100 * f["mean_closing_v"].notna().mean():.1f}% '
          f'of defender-seasons')
    return f


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['extract', 'features', 'all'])
    ap.add_argument('--min-shots', type=int, default=100)
    ap.add_argument('--half', choices=['h1', 'h2'],
                    help='restrict to one calendar half (split-half gate)')
    ap.add_argument('--cut-date', help='split boundary, e.g. 2025-01-25; '
                                       'use the date solve_ts_split_half.py reports')
    ap.add_argument('--out', help='output filename under results/')
    a = ap.parse_args()
    if a.half and not a.cut_date:
        ap.error('--half requires --cut-date')
    if a.stage in ('extract', 'all'):
        extract()
    if a.stage in ('features', 'all'):
        features(a.min_shots,
                 half=(a.half, a.cut_date) if a.half else None,
                 out_name=a.out)
