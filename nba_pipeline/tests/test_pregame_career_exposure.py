"""Unknown entrant weights require actual exposure and common whole-game gates."""
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from audit_pregame_career_exposure import required_game_exclusions


def test_only_positive_deployment_exposure_requires_a_legal_unknown_weight():
    queries=pd.DataFrame(dict(game_id=["a","a","b","c"],player_id=[1,2,3,4],
        background_model_available=[True,True,False,True]))
    exposures=pd.DataFrame(dict(game_id=["a","a","b","c"],player_id=[1,2,3,4],
        exposure=[0.,2.,3.,4.]))
    admitted=pd.DataFrame(dict(game_id=["a","b","c"],admitted_target_game=[True,True,False]))
    result=required_game_exclusions(queries,exposures,admitted)
    assert result.game_id.tolist()==["a","c"]
    assert result.unknown_player_ids.tolist()==["2","4"]
    assert result.additional_common_exclusion.tolist()==[True,False]


def test_multiple_unknown_players_produce_one_whole_game_exclusion():
    queries=pd.DataFrame(dict(game_id=["a","a"],player_id=[2,1],background_model_available=[True,True]))
    exposures=pd.DataFrame(dict(game_id=["a","a","a"],player_id=[1,1,2],exposure=[1.,1.,3.]))
    admitted=pd.DataFrame(dict(game_id=["a"],admitted_target_game=[True]))
    result=required_game_exclusions(queries,exposures,admitted)
    assert len(result)==1
    assert result.unknown_player_ids.iloc[0]=="1,2"
