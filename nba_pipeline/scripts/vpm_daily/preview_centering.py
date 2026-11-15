"""One fixed active-player reporting baseline per dated VPM publication.

The model board, priors and likelihood are unchanged. Every final leaf gets
its own side-specific TD-possession offset; intercept adjustments preserve
all five-on-five predictions. Table filtering never recalculates the baseline.
"""
import hashlib
import json

import numpy as np

SCHEMA = 'active365_td550_reporting_centering_v1'
LEAVES = ['ft', 'rim_frequency', 'rim_conversion', 'mid_ev', 'mid_sq_residual',
          'three_frequency', 'three_ev', 'three_sq_residual', 'tov', 'second_chance']
FACTORS = LEAVES + ['mid_reference_conversion', 'three_conversion', 'ts', 'total']


def center_active_preview(frame, parent_intercept):
    """Center the complete, already selected active365 publication once."""
    d = frame.copy()
    id_col = 'nba_id' if 'nba_id' in d else 'player_id'
    if d.empty or not d[id_col].is_unique:
        raise ValueError('Complete unique active-player publication required')
    dates = d.estimate_date.astype(str).unique()
    if len(dates) != 1 or not np.isfinite(parent_intercept):
        raise ValueError('One dated board and finite source intercept required')
    cut = np.datetime64(dates[0], 'D')
    last = d.last_archive_positive_observation_date.to_numpy(dtype='datetime64[D]')
    if np.isnat(last).any() or (last < cut - np.timedelta64(365, 'D')).any() or (last >= cut).any():
        raise ValueError('Every reference player needs strictly earlier positive exposure within365days')
    offsets, totals, means = {}, {}, {}
    for side, headline in [('off', 'ovpm'), ('def', 'dvpm')]:
        weights = d['td_' + side + '_possessions'].to_numpy(float)
        fields = [headline] + ['final_' + side + '_' + f for f in FACTORS]
        values = d[fields].to_numpy(float)
        if (not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0
                or not np.isfinite(values).all()):
            raise ValueError('Finite ratings and nonnegative TD-possession mass required')
        leaf_sum = sum(d['final_' + side + '_' + f] for f in LEAVES)
        np.testing.assert_allclose(leaf_sum, d[headline], atol=1e-9, rtol=0)
        m = {f: float(np.average(d['final_' + side + '_' + f], weights=weights)) for f in LEAVES}
        m.update(mid_reference_conversion=m['mid_ev'] + m['mid_sq_residual'],
                 three_conversion=m['three_ev'] + m['three_sq_residual'],
                 ts=sum(m[f] for f in LEAVES[:-2]), total=sum(m[f] for f in LEAVES))
        means[side] = m
        totals[side] = float(np.average(d[headline], weights=weights))
        offsets[headline] = totals[side]
        for factor in FACTORS:
            offsets['final_' + side + '_' + factor] = m[factor]
    offsets['vpm'] = totals['off'] + totals['def']
    for factor in FACTORS:
        offsets['final_total_' + factor] = means['off'][factor] + means['def'][factor]
    for field, offset in offsets.items():
        d[field] = frame[field] - offset
    # Side-wise reporting shifts imply these exact PPP-intercept changes.
    adjustment = 5 * (totals['off'] - totals['def']) / 100
    for side, headline in [('off', 'ovpm'), ('def', 'dvpm')]:
        w = d['td_' + side + '_possessions'].to_numpy(float)
        if abs(np.average(d[headline], weights=w)) > 1e-12:
            raise ValueError('Active-player headline centering failed')
        for factor in LEAVES:
            if abs(np.average(d['final_' + side + '_' + factor], weights=w)) > 1e-12:
                raise ValueError('Active-player component centering failed')
        np.testing.assert_allclose(sum(d['final_' + side + '_' + f] for f in LEAVES), d[headline], atol=1e-9, rtol=0)
    np.testing.assert_allclose(d.ovpm + d.dvpm, d.vpm, atol=1e-9, rtol=0)
    # A constant total shift must preserve the original ranking, including ties.
    if not frame.vpm.rank(method='first').equals(d.vpm.rank(method='first')):
        raise ValueError('Reporting shift changed player ranking')
    ids = sorted(int(x) for x in d[id_col])
    meta = dict(schema=SCHEMA, estimate_date=dates[0], reference_players=len(d),
        reference='Positive playing exposure within365days before estimate date; fixed before table filters',
        reference_player_ids_sha256=hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest(),
        weights='Side-specific full-history NBA possession exposure with550-day half-life',
        off_td_possessions=float(d.td_off_possessions.sum()), def_td_possessions=float(d.td_def_possessions.sum()),
        off_offset=totals['off'], def_offset=totals['def'], total_offset=offsets['vpm'],
        field_offsets_points_per_100=offsets, source_parent_intercept_ppp=float(parent_intercept),
        parent_intercept_adjustment_ppp=adjustment,
        display_parent_intercept_ppp=float(parent_intercept + adjustment),
        leaf_intercept_adjustments_ppp={f: 5 * (means['off'][f] - means['def'][f]) / 100 for f in LEAVES},
        scope='Final headline/decomposition reporting only; own/combined/background priors and raw model arrays unchanged')
    return d, meta
