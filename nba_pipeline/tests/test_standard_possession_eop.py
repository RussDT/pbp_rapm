import pandas as pd

from nba_pipeline.scripts.process_rapm_blocks.common import prepare_standard_possession_df


def _row(**overrides):
    base = {
        "game_id": "0020800010",
        "event_type": "",
        "event_action_type": "",
        "home_description": "",
        "visitor_description": "",
        "neutral_description": "",
        "Prev_visitor_desc": "",
        "Prev_visitor_desc2": "",
        "Next_visitor_desc": "",
        "Prev_home_desc": "",
        "Prev_home_desc2": "",
        "Next_home_desc": "",
        "Prev_Event": "",
        "Next_event": "",
        "prev_seconds": 1,
        "player1_team_abbreviation": "",
        "player1_team_city": "",
        "player1_team_nickname": "",
    }
    for prefix, start in [("h", 100), ("a", 200)]:
        for i in range(1, 6):
            base[f"{prefix}{i}"] = start + i
    base.update(overrides)
    return base


def _prepare(rows):
    df = pd.DataFrame(rows)
    out, _ = prepare_standard_possession_df(df, "TEST")
    return out


def test_neutral_shot_clock_turnover_gets_game_local_offense():
    out = _prepare([
        _row(
            event_type="MISS",
            home_description="MISS Durant 3PT Jump Shot",
            player1_team_abbreviation="OKC",
            player1_team_city="Oklahoma City",
            player1_team_nickname="Thunder",
            Next_event="Turnover",
        ),
        _row(
            event_type="Turnover",
            neutral_description="THUNDER Turnover: Shot Clock (T#18)",
            player1_team_abbreviation="MIL",
            player1_team_city="Milwaukee",
            player1_team_nickname="Bucks",
        ),
    ])
    assert out.loc[1, "End_of_Possession"]
    assert out.loc[1, "poss_offense"] == "Home"
    assert out.loc[1, "O1"] == 101
    assert out.loc[1, "D1"] == 201


def test_neutral_turnover_matches_nickname_from_game_local_abbreviation():
    out = _prepare([
        _row(
            event_type="MAKE",
            home_description="Pierce 1' Layup (2 PTS)",
            player1_team_abbreviation="BOS",
            player1_team_city="",
            player1_team_nickname="",
        ),
        _row(
            event_type="MISS",
            visitor_description="MISS James 3PT Jump Shot",
            player1_team_abbreviation="CLE",
            player1_team_city="",
            player1_team_nickname="",
            Next_event="Turnover",
        ),
        _row(
            event_type="Turnover",
            neutral_description="Cavaliers Turnover: Shot Clock (T#9)",
            player1_team_abbreviation="",
            player1_team_city="",
            player1_team_nickname="",
        ),
    ])
    assert out.loc[2, "poss_offense"] == "Away"
    assert out.loc[2, "O1"] == 201
    assert out.loc[2, "D1"] == 101


def test_end_of_period_uses_possession_offense_from_prior_shot():
    out = _prepare([
        _row(
            event_type="MISS",
            home_description="MISS Durant 18' Jump Shot",
            Next_event="EndOfPeriod",
        ),
        _row(
            event_type="EndOfPeriod",
            home_description="End of 1st Period (8:40 PM EST)",
            prev_seconds=4,
        ),
    ])
    assert out.loc[1, "End_of_Possession"]
    assert out.loc[1, "TeamOnOffense"] == ""
    assert out.loc[1, "poss_offense"] == "Home"
    assert out.loc[1, "O1"] == 101


def test_blank_end_of_period_lineup_is_filled_from_prior_row():
    blank_lineup = {f"{prefix}{i}": 0 for prefix in ["h", "a"] for i in range(1, 6)}
    out = _prepare([
        _row(
            event_type="MISS",
            home_description="MISS Durant 18' Jump Shot",
            Next_event="EndOfPeriod",
        ),
        _row(
            event_type="EndOfPeriod",
            home_description="End of 1st Period (8:40 PM EST)",
            prev_seconds=4,
            **blank_lineup,
        ),
    ])
    assert out.loc[1, "End_of_Possession"]
    assert out.loc[1, "poss_offense"] == "Home"
    assert out.loc[1, "O1"] == 101
    assert out.loc[1, "D1"] == 201


def test_blank_end_of_period_offense_flips_from_previous_terminal_possession():
    blank_text = {
        "player1_team_abbreviation": "",
        "player1_team_city": "",
        "player1_team_nickname": "",
    }
    out = _prepare([
        _row(
            event_type="Turnover",
            home_description="Durant Bad Pass Turnover (P1.T1)",
        ),
        _row(
            event_type="EndOfPeriod",
            home_description="End of 1st Period (8:40 PM EST)",
            prev_seconds=4,
            **blank_text,
        ),
    ])
    assert out.loc[0, "poss_offense"] == "Home"
    assert out.loc[1, "poss_offense"] == "Away"
    assert out.loc[1, "O1"] == 201


def test_blank_neutral_turnover_offense_flips_from_previous_terminal_possession():
    out = _prepare([
        _row(
            event_type="MAKE",
            visitor_description="Mason 1' Layup (2 PTS)",
        ),
        _row(
            event_type="Turnover",
            neutral_description="Thunder Turnover: Shot Clock (T#8)",
            player1_team_abbreviation="",
            player1_team_city="",
            player1_team_nickname="",
        ),
    ])
    assert out.loc[0, "poss_offense"] == "Away"
    assert out.loc[1, "poss_offense"] == "Home"
    assert out.loc[1, "O1"] == 101


def test_ft_off_check_inserted_eop_keeps_free_throw_offense():
    rebound_lineup = {f"h{i}": 300 + i for i in range(1, 6)}
    out = _prepare([
        _row(
            event_type="Foul",
            visitor_description="Mason S.FOUL (P2.T2)",
            Next_event="Substitution",
        ),
        _row(
            event_type="Substitution",
            visitor_description="SUB: Weaver FOR Mason",
            Next_event="FreeThrow",
        ),
        _row(
            event_type="FreeThrow",
            home_description="MISS Durant Free Throw 2 of 2",
            Prev_home_desc="",
            Next_event="Rebound",
            Next_home_desc="Durant REBOUND (Off:1 Def:0)",
        ),
        _row(
            event_type="Rebound",
            home_description="Durant REBOUND (Off:1 Def:0)",
            Prev_home_desc="MISS Durant Free Throw 2 of 2",
            **rebound_lineup,
        ),
        _row(event_type="MISS", visitor_description="MISS Bucks 3PT Jump Shot"),
    ])
    assert out.loc[2, "End_of_Possession"]
    assert out.loc[2, "poss_offense"] == "Home"
    assert out.loc[2, "O1"] == 101


def test_and_one_make_does_not_end_before_free_throw():
    out = _prepare([
        _row(
            event_type="MAKE",
            home_description="Durant Driving Layup (2 PTS)",
            Next_visitor_desc="Mason S.FOUL (P1.T1)",
            Next_event="Foul",
        ),
        _row(
            event_type="Foul",
            visitor_description="Mason S.FOUL (P1.T1)",
            Next_event="FreeThrow",
        ),
        _row(
            event_type="FreeThrow",
            home_description="Durant Free Throw 1 of 1 (3 PTS)",
        ),
    ])
    assert not out.loc[0, "End_of_Possession"]
    assert out.loc[0, "poss_offense"] == "Home"
    assert out.loc[2, "End_of_Possession"]
    assert out.loc[2, "poss_offense"] == "Home"


def test_offensive_rebound_after_missed_final_ft_does_not_end_on_miss():
    out = _prepare([
        _row(
            event_type="FreeThrow",
            home_description="MISS Durant Free Throw 2 of 2",
            Next_event="Rebound",
            Next_home_desc="Durant REBOUND (Off:1 Def:0)",
        ),
        _row(
            event_type="Rebound",
            home_description="Durant REBOUND (Off:1 Def:0)",
            Prev_home_desc="MISS Durant Free Throw 2 of 2",
            Next_event="MAKE",
        ),
        _row(
            event_type="MAKE",
            home_description="Durant Layup (2 PTS)",
        ),
    ])
    assert not out.loc[0, "End_of_Possession"]
    assert out.loc[0, "poss_offense"] == "Home"
    assert out.loc[2, "End_of_Possession"]
