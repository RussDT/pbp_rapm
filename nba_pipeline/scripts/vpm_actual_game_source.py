"""Observed player-game counts and strict actual-scoring census for VPM actuals.

No filtering, prior fitting or attribution occurs here. Native terminal-row
exposures and provider exposure remain separate; only complete matched games
are admitted. Call through the sibling VPM harness.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SLOTS = [f'{side}{i}' for side in ['O', 'D'] for i in range(1, 6)]
COUNT_SOURCES = {
    'rim_fgm': ['at_rim_fgm'], 'rim_fga': ['at_rim_fga'],
    'mid_fgm': ['short_midrange_fgm', 'long_midrange_fgm'],
    'mid_fga': ['short_midrange_fga', 'long_midrange_fga'],
    'three_fgm': ['arc3_fgm', 'corner3_fgm'],
    'three_fga': ['arc3_fga', 'corner3_fga'],
    'ftm': ['ftm'], 'fta': ['fta'],
    'ast': ['ast_at_rim', 'ast_short_midrange', 'ast_long_midrange', 'ast_arc3', 'ast_corner3'],
    'tov': ['tov'], 'oreb': ['oreb'], 'dreb': ['dreb'],
    'stl': ['stl'], 'blk': ['blk'], 'shooting_pf': ['foul_shooting'],
}
OFF_FEATURES = ['rim_fgm', 'rim_fga', 'mid_fgm', 'mid_fga',
                'three_fgm', 'three_fga', 'ftm', 'fta', 'ast', 'tov', 'oreb']
DEF_FEATURES = ['dreb', 'stl', 'blk', 'shooting_pf']
SOURCE_COLUMNS = ['game_id', 'player_id', 'season', 'playoffs', 'team_id',
                  'opponent_id', 'is_home', 'game_date', 'off_poss', 'def_poss',
                  'seconds_pbp', 'lineups_clean'] + sorted({
                      'core.' + c for values in COUNT_SOURCES.values() for c in values})


def exact_ids(values):
    a = np.asarray(pd.to_numeric(values, errors='raise'), dtype=float)
    if not np.isfinite(a).all() or (a <= 0).any() or (a != np.floor(a)).any():
        raise ValueError('Positive exact integer identity required')
    return a.astype(np.int64)


def admit(atoms, pricing, observations, year, phase='RS', cutoff=None):
    """Return complete observed/core/native games and an explicit rejection log."""
    if not 1997 <= year <= 2026 or phase not in ['RS', 'PS']:
        raise ValueError('Only completed 1997–2026 seasons admitted; 2027 protected')
    d = atoms.copy().reset_index(drop=True)
    d['game_id'] = exact_ids(d.game_id)
    d['game_date'] = pd.to_datetime(d.game_date)
    if not d.source_season_end_year.eq(year).all() or not d.season_phase.eq(phase).all():
        raise ValueError('Wrong native season/phase')
    if cutoff is not None:
        d = d.loc[d.game_date.lt(pd.Timestamp(cutoff))].reset_index(drop=True)
    o = observations.loc[observations.season.eq(year) & observations.playoffs.eq(phase == 'PS')].copy()
    o['game_id'] = exact_ids(o.game_id); o['player_id'] = exact_ids(o.player_id)
    o['game_date'] = pd.to_datetime(o.game_date)
    if o.duplicated(['game_id', 'player_id']).any():
        raise ValueError('Repeated observed player-game')
    o = o.loc[o.game_id.isin(d.game_id.unique())].reset_index(drop=True)
    for name, fields in COUNT_SOURCES.items():
        o[name] = o[['core.' + x for x in fields]].sum(axis=1, min_count=len(fields))
    o['observed_points'] = 2 * (o.rim_fgm + o.mid_fgm) + 3 * o.three_fgm + o.ftm
    counts = o[list(COUNT_SOURCES)].to_numpy(float)
    o['valid_counts'] = np.isfinite(counts).all(axis=1) & (counts >= 0).all(axis=1)
    o['valid_counts'] &= (counts == np.floor(counts)).all(axis=1)
    for zone in ['rim', 'mid', 'three', 'ft']:
        m, a = ('ftm', 'fta') if zone == 'ft' else (zone + '_fgm', zone + '_fga')
        o['valid_counts'] &= o[m].le(o[a])
    # Native identities must be source-resolved for every slot, not repaired.
    keys = pd.MultiIndex.from_frame(o[['game_id', 'player_id']])
    slot = np.empty((len(d), 10), dtype=np.int64)
    for j, name in enumerate(SLOTS):
        ids = exact_ids(d[name])
        slot[:, j] = keys.get_indexer(pd.MultiIndex.from_arrays([d.game_id.to_numpy(), ids]))
    known = (slot >= 0).all(axis=1)
    safe = np.maximum(slot, 0)
    if o.empty:
        raise ValueError('No matching observed game source')
    teams = o.team_id.to_numpy()[safe]
    home = o.is_home.to_numpy()[safe]
    valid = known & (teams[:, :5] == teams[:, :1]).all(axis=1)
    valid &= (teams[:, 5:] == teams[:, 5:6]).all(axis=1) & (teams[:, 0] != teams[:, 5])
    valid &= (home[:, :5] == home[:, :1]).all(axis=1) & (home[:, 5:] == home[:, 5:6]).all(axis=1)
    valid &= home[:, 0] != home[:, 5]
    valid &= o.valid_counts.to_numpy()[safe].all(axis=1)
    valid &= (o.game_date.to_numpy()[safe] == d.game_date.to_numpy()[:, None]).all(axis=1)
    invalid_games = set(d.loc[~valid, 'game_id'])
    log = []
    source_points = o.groupby('game_id').observed_points.sum(min_count=1)
    native_points = d.groupby('game_id').actual_points.sum()
    source_valid = o.groupby('game_id').valid_counts.all()
    for game in sorted(native_points.index):
        reasons = []
        if game in invalid_games: reasons.append('incomplete_identity_team_date_or_core_counts')
        if game not in source_points or source_points.get(game) != native_points[game]:
            reasons.append('whole_game_actual_scoring_mismatch')
        if not bool(source_valid.get(game, False)): reasons.append('invalid_observed_counts')
        log.append(dict(game_id=int(game), accepted=not reasons, reasons=','.join(reasons),
                        native_points=float(native_points[game]),
                        observed_points=float(source_points.get(game, np.nan))))
    audit = pd.DataFrame(log)
    accepted = set(audit.loc[audit.accepted, 'game_id'])
    use = d.game_id.isin(accepted).to_numpy()
    d = d.loc[use].reset_index(drop=True); slot = slot[use]
    # Keep precisely the native playing universe. Zero-opportunity actors with
    # actual terminal exposure are retained. No roster-only eligibility rows.
    used = np.unique(slot)
    q = o.iloc[used].copy().reset_index(drop=True)
    remap = np.full(len(o), -1, dtype=np.int64); remap[used] = np.arange(len(used))
    slot = remap[slot]
    q['native_off_poss'] = np.bincount(slot[:, :5].ravel(), minlength=len(q))
    q['native_def_poss'] = np.bincount(slot[:, 5:].ravel(), minlength=len(q))
    q = q.rename(columns={'off_poss':'source_off_poss', 'def_poss':'source_def_poss'})
    q['phase'] = phase
    q = q[['game_id','player_id','season','phase','game_date','team_id','opponent_id',
           'is_home','seconds_pbp','native_off_poss','native_def_poss',
           'source_off_poss','source_def_poss','observed_points', *COUNT_SOURCES]]
    d['off_home'] = q.is_home.to_numpy()[slot[:, 0]].astype(float) * 2 - 1
    p = pricing.loc[pd.to_numeric(pricing.game_id).isin(accepted)].copy().reset_index(drop=True)
    if set(pd.to_numeric(p.game_id)) != set(d.game_id):
        raise ValueError('Admitted pricing differs from native census')
    if not np.isclose(q.native_off_poss.sum(), 5 * len(d)) or not np.isclose(q.native_def_poss.sum(),5 * len(d)):
        raise ValueError('Native side exposure does not close')
    return d, p, q, slot, audit
