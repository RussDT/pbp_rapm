"""Date/population gates around canonical historical VPM target algebra."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from historical_targets import load_before_cut, target_matrix, validate_population, sha
from test_vpm_net_lane_targets import sample as canonical_sample


def sample():
    d, g = canonical_sample()
    d["event_num"] = np.arange(len(d))
    return d, g


def test_historical_target_fences_future_pricing_even_for_existing_target_games():
    d, g = sample()
    g.loc[0, 'game_date'] = pd.Timestamp('2021-01-01')
    with pytest.raises(ValueError, match='date fence'):
        validate_population(d, g, '2020-02-01')


def test_exact_game_population_forbids_extra_past_pricing_games():
    d, g = sample()
    extra = g.copy(); extra.game_id = 2
    with pytest.raises(ValueError, match='populations differ'):
        validate_population(d, pd.concat([g, extra]), '2020-02-01')


def test_price_and_atoms_require_the_same_date():
    d, g = sample(); g.game_date += pd.Timedelta(days=1)
    with pytest.raises(ValueError, match='dates differ'):
        validate_population(d, g, '2020-02-01')


def test_historical_standard_midrange_reference_closes_with_weighted_turnover():
    d, g = sample()
    g.loc[0, ['mid_attempts', 'mid_fg_points', 'mid_misses', 'mid_points']] = [2, 2, 1, 0]
    d['sq_active'] = False
    for z in ['mid', 'three']:
        d[z + '_gross_residual'] = 0.
    y, p = target_matrix(d, g, '2020-02-01', weights=np.arange(1, 8), standard=True)
    np.testing.assert_allclose(y.sum(axis=1), d.actual_points, atol=1e-12, rtol=0)
    assert y[0, 0] == pytest.approx(1.)
    assert y[3, 3] == pytest.approx(2.)
    assert p['sq_active_rows'] == 0 and p['sq_inactive_rows'] == 7
    assert d.mid_residual.eq(0).all()


def test_loader_filters_future_and_quarantine_from_both_populations(tmp_path):
    d, g = sample()
    for c in ['O1', 'O2', 'O3', 'O4', 'O5', 'D1', 'D2', 'D3', 'D4', 'D5']:
        d[c] = int(c[1:]) + (10 if c.startswith('D') else 0)
    d['event_num'] = np.arange(len(d))
    all_d = [d]; all_g = [g]
    for game, date in [(2, '2020-01-02'), (3, '2020-03-01')]:
        dd = d.copy(); dd.game_id = game; dd.game_date = pd.Timestamp(date)
        gg = g.copy(); gg.game_id = game; gg.game_date = pd.Timestamp(date)
        all_d.append(dd); all_g.append(gg)
    d = pd.concat(all_d, ignore_index=True); g = pd.concat(all_g, ignore_index=True)
    dp = tmp_path / 'atoms_2020_RS.parquet'; gp = tmp_path / 'games_2020_RS.parquet'
    d.to_parquet(dp, index=False); g.to_parquet(gp, index=False)
    (tmp_path / 'historical_2020_RS.json').write_text(json.dumps(dict(
        artifacts={dp.name: sha(dp), gp.name: sha(gp)}, excluded_game_ids=[2])))
    dd, gg = load_before_cut(tmp_path, '2020-02-01', [2020], phases=['RS'], sq=False)
    assert set(dd.game_id) == set(gg.game_id) == {1}
    assert not dd.sq_active.any()


def test_duplicate_possession_identity_is_rejected():
    d, g = sample(); d['event_num'] = 1
    with pytest.raises(ValueError, match='Duplicate canonical'):
        validate_population(d, g, '2020-02-01')


def test_raw_ft_accounting_does_not_confuse_surname_missi_with_a_miss():
    from historical_targets import compare_scores
    raw = pd.DataFrame(dict(game_id=[1, 1, 1], event_type=[3, 3, 3],
        home_description=['Missi Free Throw 1 of 2 (1 PTS)',
                          'MISS Missi Free Throw 2 of 2', 'Smith Free Throw Technical (1 PTS)'],
        visitor_description=[None, None, None]))
    result = compare_scores(raw, pd.DataFrame(dict(game_id=[1], actual_points=[2.])))
    assert result.accepted.all() and result.raw_event_points.iloc[0] == 2


def test_annual_identity_includes_playoff_only_and_zero_minute_appearances():
    from historical_targets_identities import season_identity
    rs = pd.DataFrame(dict(PLAYER_ID=[1, 2], PLAYER_NAME=['Tiny player', 'Roster only'],
                           GP=[1, 0], MIN=[0., 0.]))
    ps = pd.DataFrame(dict(PLAYER_ID=[3], PLAYER_NAME=['Playoff only'], GP=[7], MIN=[16.9]))
    packet = season_identity(rs, ps)
    assert packet['admitted_ids'] == [1, 3]
    assert packet['phase_support']['3'] == ['PS']
    assert packet['names']['1'] == 'Tiny player'


def test_annual_identity_is_season_specific_and_does_not_promote_names():
    from historical_targets_identities import season_identity
    empty = pd.DataFrame(columns=['PLAYER_ID', 'PLAYER_NAME', 'GP', 'MIN'])
    first = pd.DataFrame(dict(PLAYER_ID=[1], PLAYER_NAME=['Former player'], GP=[2], MIN=[1.]))
    second = pd.DataFrame(dict(PLAYER_ID=[2], PLAYER_NAME=['Current player'], GP=[2], MIN=[1.]))
    assert season_identity(first, empty)['admitted_ids'] == [1]
    assert season_identity(second, empty)['admitted_ids'] == [2]
