import pytest
from nba_pipeline.scripts.build_player_game_attribution_audit import split_dtov,turnover_lane

@pytest.mark.parametrize('turnover,actor',[(False,None),(True,None),(True,123)])
def test_known_unknown_and_baseline_close(turnover,actor):
    d=split_dtov(turnover,1.3,-.18,actor)
    assert d['team_points']==pytest.approx(d['identified_actor_points']+d['unidentified_event_points']+d['baseline_points'])
    if actor is None:assert d['identified_actor_points']==0
    if not turnover:assert d['team_points']==-.18

def test_second_chance_steal_is_not_a_first_chance_tov():
    assert turnover_lane(True)=='dTOV'
    assert turnover_lane(False)=='dSC'
