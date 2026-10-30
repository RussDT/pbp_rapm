import importlib.util
from pathlib import Path
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location('rim_skills', Path(__file__).parents[1]/'scripts/build_rim_defense_matchup_skills.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_rim_saved_sign_and_algebra():
    assert m.rim_values(100,55,.65) == pytest.approx(20)
    assert m.rim_values(100,75,.65) == pytest.approx(-20)
    assert m.rim_values(0,0,.65) == 0


def test_matchup_weights_and_missingness():
    pairs = pd.DataFrame({'year':[2025]*4,'nba_id_off':[1,2,3,2], 'nba_id_def':[8,8,8,9], 'PARTIAL_POSS':[10,30,10,10]})
    strengths = pd.DataFrame({'year':[2025]*2,'nba_id_off':[1,2], 'pts100':[40,20], 'ast100':[10,2]})
    out = m.assignment_summary(pairs,strengths).set_index('player_id')
    assert out.loc[8,'opponent_scoring_load100'] == 25
    assert out.loc[8,'opponent_assist_load100'] == 4
    assert out.loc[8,'strength_coverage'] == .8
    assert (out.scoring_burden_above_league100*out.covered_poss).sum() == pytest.approx(0)


def test_matchup_outcomes_do_not_change_difficulty():
    pairs = pd.DataFrame({'year':[2025], 'nba_id_off':[1], 'nba_id_def':[8], 'PARTIAL_POSS':[10], 'PLAYER_PTS':[0]})
    strengths = pd.DataFrame({'year':[2025], 'nba_id_off':[1], 'pts100':[40], 'ast100':[10]})
    a = m.assignment_summary(pairs,strengths)
    pairs.PLAYER_PTS = 1000
    pd.testing.assert_frame_equal(a,m.assignment_summary(pairs,strengths))


def test_cumulative_recovery_gaps_and_season_resets():
    dates = pd.to_datetime(['2020-01-01','2020-01-02','2020-01-03','2020-01-04','2021-01-01'])
    a = pd.DataFrame(dict(season=[2020]*4+[2021], season_type=['rs']*5,
        player_id=[1]*5, game_date=dates, minutes=[20]*5))
    t = a.iloc[[0,2,3,4]].drop(columns='minutes').copy()
    t['rim_dfga']=[2,9,12,4];t['rim_dfgm']=[1,5,6,2]
    t['provider_normal_pct']=.65
    out=m.recover_cumulative_counts(t,a)
    assert out.rim_dfga.iloc[0]==2
    assert pd.isna(out.rim_dfga.iloc[1])  # cannot assign two games to one
    assert out.rim_dfga.iloc[2]==3
    assert out.rim_dfga.iloc[3]==4  # new season starts from zero
    assert out.provider_normal_pct.isna().all()


def test_cumulative_invalid_increment_and_future_invariance():
    a=pd.DataFrame(dict(season=[2020]*3,season_type=['rs']*3,player_id=[1]*3,
        game_date=pd.date_range('2020-01-01',periods=3),minutes=[20]*3))
    t=a.drop(columns='minutes').assign(rim_dfga=[5,4,8],rim_dfgm=[2,2,3],provider_normal_pct=.65)
    full=m.recover_cumulative_counts(t,a)
    assert pd.isna(full.rim_dfga.iloc[1])
    assert full.rim_dfga.iloc[2]==4
    prefix=m.recover_cumulative_counts(t.iloc[:2],a.iloc[:2])
    pd.testing.assert_frame_equal(full.iloc[:2].reset_index(drop=True),prefix.reset_index(drop=True))


def test_cumulative_trade_freeze_is_not_measured_zero():
    a=pd.DataFrame(dict(season=[2020]*3,season_type=['rs']*3,player_id=[1]*3,
        game_date=pd.date_range('2020-01-01',periods=3),minutes=[20]*3,team_id=[10,20,20]))
    t=a.drop(columns=['minutes','team_id']).assign(rim_dfga=[5,5,5],rim_dfgm=[2,2,2])
    out=m.recover_cumulative_counts(t,a)
    assert out.rim_dfga.iloc[0]==5
    assert out.rim_dfga.iloc[1:].isna().all()
    assert out.recovery_status.iloc[1:].eq('post_trade_unverified_cumulative').all()
