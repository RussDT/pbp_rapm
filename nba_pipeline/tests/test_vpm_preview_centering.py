from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/vpm_daily'))
from preview_centering import center_active_preview, LEAVES, FACTORS


def board():
    rng = np.random.default_rng(42)
    d = pd.DataFrame(dict(nba_id=range(1, 13), estimate_date='2026-11-15',
        last_archive_positive_observation_date='2026-11-14',
        td_off_possessions=np.arange(1., 13.)**2, td_def_possessions=np.arange(12., 0., -1.)**2,
        own_prior_off=rng.normal(size=12), own_prior_def=rng.normal(size=12)))
    for side in ['off', 'def']:
        for f in LEAVES:
            d[f'final_{side}_{f}'] = rng.normal(size=12) + .3
        for f, parts in dict(mid_reference_conversion=['mid_ev', 'mid_sq_residual'],
            three_conversion=['three_ev', 'three_sq_residual'], ts=LEAVES[:-2], total=LEAVES).items():
            d[f'final_{side}_{f}'] = sum(d[f'final_{side}_{p}'] for p in parts)
    for f in FACTORS:
        d['final_total_'+f] = d['final_off_'+f] + d['final_def_'+f]
    d['ovpm'] = d.final_off_total; d['dvpm'] = d.final_def_total; d['vpm'] = d.ovpm+d.dvpm
    return d


def test_td_weighted_active_zero_components_and_five_on_five_prediction_preservation():
    d = board(); centered, m = center_active_preview(d, 1.1)
    for side, h in [('off', 'ovpm'), ('def', 'dvpm')]:
        w = d['td_'+side+'_possessions']
        assert abs(np.average(centered[h], weights=w)) < 1e-12
        for f in LEAVES:
            assert abs(np.average(centered['final_'+side+'_'+f], weights=w)) < 1e-12
    offense, defense = [0, 1, 2, 3, 4], [5, 6, 7, 8, 9]
    prediction = lambda x, i: i + (x.ovpm.iloc[offense].sum()-x.dvpm.iloc[defense].sum())/100
    assert prediction(centered, m['display_parent_intercept_ppp']) == pytest.approx(prediction(d, 1.1), abs=1e-12)
    for f in LEAVES:
        off, de = 'final_off_'+f, 'final_def_'+f
        old = (d[off].iloc[offense].sum()-d[de].iloc[defense].sum())/100
        new = m['leaf_intercept_adjustments_ppp'][f] + (centered[off].iloc[offense].sum()-centered[de].iloc[defense].sum())/100
        assert old == pytest.approx(new, abs=1e-12)
    pd.testing.assert_frame_equal(d[['own_prior_off', 'own_prior_def']], centered[['own_prior_off', 'own_prior_def']])
    assert d.vpm.sort_values().index.tolist() == centered.vpm.sort_values().index.tolist()
    assert m['reference_players'] == 12
    assert abs(centered.dvpm.mean()) > .001  # This is not an unweighted mean.


def test_missing_or_invalid_reference_exposure_is_rejected():
    d = board(); d.loc[0, 'td_def_possessions'] = -1
    with pytest.raises(ValueError, match='TD-possession'): center_active_preview(d, 1.)
    d = board(); d.loc[0, 'last_archive_positive_observation_date'] = '2025-01-01'
    with pytest.raises(ValueError, match='within365days'): center_active_preview(d, 1.)
