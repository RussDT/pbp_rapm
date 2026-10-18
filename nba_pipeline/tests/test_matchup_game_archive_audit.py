import copy
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location("matchup_audit", Path(__file__).parents[1] / "scripts/audit_matchup_game_archive.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture():
    stats = dict(matchupMinutes="0:10", matchupMinutesSort=10.4, partialPossessions=2.5,
        playerPoints=3, matchupFieldGoalsMade=1, matchupFieldGoalsAttempted=2,
        matchupThreePointersMade=1, matchupThreePointersAttempted=1)
    raw = dict(boxScoreMatchups=dict(gameId="0021700002",
        homeTeam=dict(teamId=1, players=[dict(personId=11, matchups=[dict(personId=22, statistics=stats)])]),
        awayTeam=dict(teamId=2, players=[dict(personId=22, matchups=[])])))
    game = dict(game_id="0021700002", game_date="2017-10-17", season=2018, season_type="rs")
    normalized = pd.DataFrame([dict(game_id=game["game_id"], game_date=game["game_date"],
        season=2018, season_type="rs", offense_player_id=11, defense_player_id=22,
        offense_team_id=1, defense_team_id=2, **stats)])
    return raw, normalized, game


def test_independent_raw_pair_fidelity_and_fractional_exposure():
    raw, normalized, game = fixture()
    result = m.audit_pair_fidelity(raw, normalized, game)
    assert result["rows"] == 1 and not result["source_count_defects"]
    assert result["time_max_difference"] == pytest.approx(.4)


@pytest.mark.parametrize("bad", ["reverse", "date", "team", "counts", "duplicate", "clock", "protected"])
def test_audit_detects_mutations(bad):
    raw, normalized, game = fixture()
    if bad == "reverse":
        normalized["offense_player_id"], normalized["defense_player_id"] = [22], [11]
    elif bad == "date":
        normalized["game_date"] = "2017-10-18"
    elif bad == "team":
        normalized["defense_team_id"] = 1
    elif bad == "counts":
        normalized["playerPoints"] = 99
    elif bad == "duplicate":
        normalized = pd.concat([normalized, normalized], ignore_index=True)
    elif bad == "clock":
        normalized["matchupMinutes"] = "1:10"
    else:
        game["season"] = 2027
        normalized["season"] = 2027
    with pytest.raises((ValueError, AssertionError)):
        m.audit_pair_fidelity(raw, normalized, game)


def test_older_absent_fta_is_absent_not_a_complete_zero_stat():
    _, normalized, _ = fixture()
    box = pd.DataFrame(dict(player_id=[11, 22], pts=[3, 0], fga=[2, 0], fgm=[1, 0], fta=[2, 0], ftm=[1, 0]))
    records = {r["field"]: r for r in m.field_coverage(normalized, box)}
    assert records["matchupFreeThrowsAttempted"]["status"] == "absent"
    assert records["matchupFreeThrowsAttempted"]["tracked"] is None
    assert records["playerPoints"]["offense_absolute_gap"] == 0
    assert records["playerPoints"]["reversed_absolute_gap"] == 6


def test_source_shot_defect_is_retained_and_flagged():
    raw, normalized, game = fixture()
    stats = raw["boxScoreMatchups"]["homeTeam"]["players"][0]["matchups"][0]["statistics"]
    stats["matchupFieldGoalsMade"] = 5
    normalized["matchupFieldGoalsMade"] = 5
    result = m.audit_pair_fidelity(raw, normalized, game)
    assert "matchupFieldGoalsMade>matchupFieldGoalsAttempted" in result["source_count_defects"]


def test_defender_missing_offense_roster_is_retained_and_flag_checked():
    raw, normalized, game = fixture()
    raw["boxScoreMatchups"]["homeTeam"]["players"][0]["matchups"][0]["personId"] = 99
    normalized["defense_player_id"] = 99
    normalized["defender_in_opposing_native_roster"] = False
    result = m.audit_pair_fidelity(raw, normalized, game)
    assert result["unrostered_defender_rows"] == 1
    normalized["defender_in_opposing_native_roster"] = True
    with pytest.raises(AssertionError):
        m.audit_pair_fidelity(raw, normalized, game)
