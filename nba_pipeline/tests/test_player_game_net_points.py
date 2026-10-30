import numpy as np
import pandas as pd
import pytest
from nba_pipeline.scripts import build_player_game_net_points as b


def test_observed_make_changes_only_shooter_when_target_changes_with_it():
    prior = np.array([2., 1., -1., 0., 0.])
    exposure = np.array([70., 65., 60., 50., 40.])
    before, _ = b.project_credit(prior, np.zeros(5), exposure, 4.)
    innovation = np.array([3., 0., 0., 0., 0.])
    after, _ = b.project_credit(prior, innovation, exposure, 7.)
    np.testing.assert_allclose(after-before, innovation, atol=1e-12)


@pytest.mark.parametrize('lane', ['TS', 'SC'])
def test_opponent_three_conversion_is_removed_in_its_original_lane(lane):
    # Same chance, changed make outcome. Defense must be unchanged on either
    # first or second chance, while the actual scoreboard moves three points.
    expected = 1.08
    defensive_targets = []
    for actual in [0., 3.]:
        original = {'TS': 0., 'SC': 0.}
        original[lane] = actual
        residual = actual-expected
        defense = b.adjusted_defense(pd.Series(original), {lane: residual})
        defensive_targets.append(defense)
        assert defense.sum()-residual == pytest.approx(-actual)
    np.testing.assert_allclose(defensive_targets[0], defensive_targets[1])


def test_projection_keeps_target_and_no_prior_control_explicit():
    prior = np.array([1., -2., 3.])
    actual = np.array([-4., 2., 1.])
    exposure = np.array([10., 30., 60.])
    credit, shared = b.project_credit(prior, actual, exposure, -8.)
    control, _ = b.project_credit(np.zeros(3), actual, exposure, -8.)
    assert credit.sum() == pytest.approx(-8.)
    np.testing.assert_allclose(credit, prior+actual+shared)
    assert (credit-control).sum() == pytest.approx(0.)
    np.testing.assert_allclose(credit-control,prior-prior.sum()*exposure/exposure.sum())


def test_prior_rejects_target_season_and_wrong_substrate():
    x=pd.DataFrame({'player_id':[1],'Latest_Year':[2026],'DECOMP_PRICING':['net_continuation_kappa']})
    with pytest.raises(ValueError,match='before'):
        b.prior_factors(x,2026)
    x['Latest_Year']=2025
    x['DECOMP_PRICING']='gross'
    with pytest.raises(ValueError,match='Net DECOMP'):
        b.prior_factors(x,2026)


def test_zero_exposure_and_nonfinite_inputs_fail():
    with pytest.raises(ValueError,match='exposure'):
        b.project_credit([1],[0],[0],2)
    with pytest.raises(ValueError,match='Nonfinite'):
        b.project_credit([np.nan],[0],[1],2)


def test_rebound_correction_repeated_cumulative_counts():
    raw=pd.DataFrame({'game_id':['g']*3,'event_num':[1,2,3],'player1_id':[7]*3,'event_type':[4]*3,'home_description':['A REBOUND (Off:2 Def:0)','A REBOUND (Off:2 Def:0)','A REBOUND (Off:2 Def:1)'],'visitor_description':[None]*3})
    out=b.rebound_events(raw)
    assert out.orb.tolist()==[1,1,0]
    assert out.drb.tolist()==[0,0,1]
    raw.loc[2,'home_description']='A REBOUND (Off:5 Def:1)'
    with pytest.raises(ValueError,match='inconsistent'):
        b.rebound_events(raw)


def test_score_bridge_with_unequal_possessions_and_asymmetric_shooting():
    actual_margin=11.
    baseline=1.12*(102-99)
    opponent_variance=-6.3
    player_total=actual_margin-baseline+opponent_variance
    assert player_total+baseline-opponent_variance==pytest.approx(actual_margin)
    # Omitting the possession bridge would incorrectly call this nonclosure.
    assert player_total-opponent_variance!=pytest.approx(actual_margin)


def test_raw_conflicts_fail_before_any_keep_first_selection():
    raw=pd.DataFrame({'game_id':['1','1'],'event_num':[2,2],'event_type':[1,2]})
    with pytest.raises(ValueError,match='Conflicting raw event'):
        b.normalize_raw(raw)
