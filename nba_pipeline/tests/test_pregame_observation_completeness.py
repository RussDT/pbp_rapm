"""Admission masks must remove uncertain measurements without losing queries."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_pregame_observation_completeness import (
    FG, SCORING, apply_masks, audit_phase, build, recount_raw,
)


def source():
    rows = []
    for pid in [1, 2]:
        row = dict(game_id="0021300001", player_id=pid, season=2014,
            game_date=pd.Timestamp("2013-10-29"), season_type="RS", player_name=str(pid),
            off_poss=20., def_poss=21., minutes=12., pf_committed=3.,
            ast=0., stl=0., blk=0., tov=0., oreb=0., dreb=0.)
        row.update({field: 0. for field in SCORING})
        if pid == 1:
            row.update(fga=1., fgm=1., rim_fga=1., rim_fgm=1., pts=2.)
        rows.append(row)
    return pd.DataFrame(rows)


def raw(events=None, score="2 - 0"):
    events = events or [(1, 1, "Player Shot (2 PTS)")]
    rows = []
    for i, (kind, pid, description) in enumerate(events):
        rows.append(dict(game_id="0021300001", event_num=i+1, event_type=kind, period=4,
            minute_game=47.0, score="0 - 0", player1_id=pid,
            home_description=description, visitor_description=None, neutral_description=None))
    rows.append(dict(game_id="0021300001", event_num=100, event_type=13, period=4,
        minute_game=48., score=score, player1_id=0,
        home_description=None, visitor_description=None, neutral_description=None))
    return pd.DataFrame(rows)


def test_unresolved_terminal_masks_shooting_but_retains_queries_pf_and_exposure():
    original = source()
    games, masks = audit_phase(original, raw(score="0 - 0"), 2014, "RS")
    result = apply_masks(original, masks)
    assert games.scoreboard_status.iloc[0] == "terminal_unresolved"
    assert result[SCORING].isna().all().all()
    pd.testing.assert_frame_equal(result[["off_poss", "def_poss", "minutes", "pf_committed"]],
                                  original[["off_poss", "def_poss", "minutes", "pf_committed"]])
    assert result[["game_id", "player_id"]].equals(original[["game_id", "player_id"]])


def test_local_missing_ft_does_not_discard_other_player_or_fg_measurements():
    original = source()
    games, masks = audit_phase(original, raw([(1, 1, "Player Shot (2 PTS)"),
        (3, 2, "Nance Jr. Free Throw 1 of 1 (1 PTS)")], "3 - 0"), 2014, "RS")
    result = apply_masks(original, masks)
    assert games.scoreboard_status.iloc[0] == "raw_scoring_reconciled"
    assert result.loc[0, SCORING].notna().all()
    assert result.loc[1, ["fta", "ftm", "pts"]].isna().all()
    assert result.loc[1, FG].notna().all()
    assert result.loc[1, "tov"] == 0.


def test_word_miss_does_not_turn_missi_into_a_missed_free_throw():
    _, games, _, _, _ = recount_raw(raw([(3, 1, "Missi Free Throw 1 of 1 (1 PTS)")], "1 - 0"))
    assert games.raw_ftm.iloc[0] == 1
    assert games.raw_uncertain_ft.iloc[0] == 0


def test_free_throw_without_miss_or_points_is_unknown_not_a_made_zero():
    original = source()
    _, masks = audit_phase(original, raw([(1, 1, "Player Shot (2 PTS)"),
        (3, 2, "Free Throw 2 of 2")]), 2014, "RS")
    result = apply_masks(original, masks)
    assert result[["fta", "ftm", "pts"]].isna().all().all()
    assert result[FG].notna().all().all()
    assert "raw_free_throw_outcome_unknown" in set(masks.reason)


def test_raw_scoring_disagreement_masks_all_shooting_without_inventing_counts():
    original = source()
    games, masks = audit_phase(original, raw(score="3 - 0"), 2014, "RS")
    assert games.scoreboard_status.iloc[0] == "raw_scoring_disagrees"
    result = apply_masks(original, masks)
    assert result[SCORING].isna().all().all()
    assert original.pts.tolist() == [2., 0.]


def test_non_scoring_actor_disagreement_masks_only_affected_player_family():
    original = source()
    _, masks = audit_phase(original, raw([(1, 1, "Player Shot (2 PTS)"),
        (5, 2, "Player Bad Pass Turnover (P1.T1)")]), 2014, "RS")
    result = apply_masks(original, masks)
    assert result.loc[1, "tov"] is np.nan or pd.isna(result.loc[1, "tov"])
    assert result.loc[0, "tov"] == 0.
    assert result[SCORING].notna().all().all()


def test_rebound_total_mismatch_masks_both_split_counts_only_for_actor():
    original = source()
    _, masks = audit_phase(original, raw([(1, 1, "Player Shot (2 PTS)"),
        (4, 2, "Player REBOUND (Off:1 Def:0)")]), 2014, "RS")
    result = apply_masks(original, masks)
    assert result.loc[1, ["oreb", "dreb"]].isna().all()
    assert result.loc[0, ["oreb", "dreb"]].eq(0.).all()


def test_unavailable_tags_are_not_compared_as_observed_zero():
    _, masks = audit_phase(source(), raw(), 2014, "RS")
    assert "raw_ast_tags_unavailable" in set(masks.reason)
    assert "raw_ast_game_tag_count_mismatch" not in set(masks.reason)


def test_block_foul_is_not_a_blocked_shot_and_team_turnover_not_personal():
    _, games, _, _, available = recount_raw(raw([(1, 1, "Player Shot (2 PTS)"),
        (6, 2, "Player Shooting Block Foul (P1.T1)"),
        (5, 1610612738, "Team Turnover")]))
    assert games.raw_blk.iloc[0] == 0
    assert not available["blk"]
    assert games.raw_tov.iloc[0] == 0


def test_unknown_scoring_actor_keeps_all_queries_but_masks_scoring():
    original = source()
    _, masks = audit_phase(original, raw([(1, 1, "Player Shot (2 PTS)"),
        (3, 0, "Unknown Free Throw (1 PTS)")], "3 - 0"), 2014, "RS")
    result = apply_masks(original, masks)
    assert len(result) == 2
    assert result[SCORING].isna().all().all()


def test_explicit_raw_actor_absent_from_census_is_not_assigned_to_someone_else():
    original = source()
    _, masks = audit_phase(original, raw([(1, 1, "Player Shot (2 PTS)"),
        (5, 999, "Other Player Turnover")]), 2014, "RS")
    result = apply_masks(original, masks)
    assert result.player_id.tolist() == [1, 2]
    assert result[["tov", "oreb", "dreb"]].isna().all().all()
    assert result[SCORING].isna().all().all()
    assert result.pf_committed.eq(3.).all()


def test_field_goal_disagreement_masks_zones_and_usage_inputs_not_ft():
    original = source()
    _, masks = audit_phase(original, raw([(1, 1, "Player Shot (2 PTS)"),
        (2, 2, "MISS Player 3PT Jump Shot")]), 2014, "RS")
    result = apply_masks(original, masks)
    assert result.loc[1, FG].isna().all()
    assert result.loc[1, ["fta", "ftm"]].eq(0.).all()
    assert pd.isna(result.loc[1, "fga"] + .44*result.loc[1, "fta"] + result.loc[1, "tov"])


def test_masks_cannot_modify_pf_or_missing_query_and_output_is_immutable(tmp_path):
    mask = pd.DataFrame(dict(game_id=["0021300001"], player_id=[1], field=["pf_committed"]))
    with pytest.raises(ValueError, match="protected"):
        apply_masks(source(), mask)
    mask["field"] = "pts"
    mask["player_id"] = 999
    with pytest.raises(ValueError, match="nonexistent"):
        apply_masks(source(), mask)
    with pytest.raises(FileExistsError, match="immutable"):
        build(tmp_path / "absent_source", tmp_path)
