"""Meaningful negative controls for the diagnostic's non-admission bounds."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_pregame_exposure_definition import counter_ownership_audit, ejection_ownership_bounds, fetch


def test_forbids_unregistered_and_2027_network_queries(tmp_path):
    with pytest.raises(ValueError):
        fetch("0022600001",tmp_path)
    with pytest.raises(ValueError):
        fetch("0022400002",tmp_path)


def test_absent_later_entrant_is_possible_unlogged_replacement():
    game={"game_id":"0022400763","ejections":[{"person_id":1,"team":"DEN","event_index":20}],
          "openers":[{"valid":True,"lineup_valid":True,"event_index":21,"players":[1,2,3,4,5,11,12,13,14,15],
                      "offense":"POR","action_number":517}]}
    players=pd.DataFrame([{"person_id":p,"team":t,"off_poss_source":10,"def_poss_source":11}
                          for p,t in [(1,"DEN"),(2,"DEN"),(99,"DEN"),(11,"POR")]])
    out=ejection_ownership_bounds(game,players).set_index("person_id")
    assert (out.loc[1,"def_lower"],out.loc[1,"def_upper"]) == (10,11)
    assert (out.loc[99,"def_lower"],out.loc[99,"def_upper"]) == (11,12)
    assert out.loc[2,"career_game_lower"] == out.loc[2,"career_game_upper"] == 10.5
    assert out.loc[11,"career_game_lower"] == out.loc[11,"career_game_upper"] == 10.5
    assert not out.recovery_admitted.any()


def test_ejection_after_opener_cannot_change_pregame_source_possession_membership():
    game={"game_id":"0022400763","ejections":[{"person_id":1,"team":"DEN","event_index":20}],
          "openers":[{"valid":True,"lineup_valid":True,"event_index":19,"players":[1],
                      "offense":"POR","action_number":510}]}
    players=pd.DataFrame([{"person_id":99,"team":"DEN","off_poss_source":10,"def_poss_source":11}])
    out=ejection_ownership_bounds(game,players)
    assert out.loc[0,"career_game_lower"] == out.loc[0,"career_game_upper"] == 10.5
    assert out.loc[0,"affected_opener_actions"] == ""


def test_flags_explained_only_if_no_bad_counter_owner():
    def row(action,number,actor):
        return {"action_type":action,"action_number":number,"person_id":actor,"players_on":list(range(1,11)),
                "off_players_on":list(range(1,6)),"def_players_on":list(range(6,11)),"flags":{}}
    raw={"plays":[row("foul",611,1),row("ejection",515,1),row("3pt",517,6)]}
    replay={"openers":[{"valid":True,"lineup_valid":True,"action_number":611}],"invalid_non_sub_rows":0}
    good=counter_ownership_audit(raw,replay,1,0)
    assert good["count_ownership_unaffected_by_explained_flags"]
    replay["openers"][0]["action_number"]=517
    bad=counter_ownership_audit(raw,replay,1,0)
    assert bad["affected_counter_owner_actions"] == [517]
    assert not bad["count_ownership_unaffected_by_explained_flags"]
    assert not counter_ownership_audit(raw,replay,2,0)["source_flags_explained_by_ejection_rule"]
    assert not good["recovery_admitted"]
