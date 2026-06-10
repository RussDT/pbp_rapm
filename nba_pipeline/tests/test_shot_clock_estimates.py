import pandas as pd
import pytest

from nba_pipeline.scripts.build_shot_clock_estimates import estimate_shot_clock


def _row(**overrides):
    base = {
        "game_id": "0042500001",
        "event_num": 1,
        "event_type": "",
        "period": 1,
        "time_quarter": "12:00",
        "minute_remaining_quarter": 12,
        "seconds_remaining_quarter": 0,
        "home_description": "",
        "visitor_description": "",
        "neutral_description": "",
        "player1_id": 0,
        "player1_name": "",
    }
    base.update(overrides)
    return base


def _estimate(rows):
    return estimate_shot_clock(pd.DataFrame(rows)).reset_index(drop=True)


def _shot(out, event_num):
    return out[out["event_num"].eq(event_num)].iloc[0]


@pytest.mark.parametrize(
    "tip_side,shot_desc,expected_side",
    [
        ("Home Guard", {"home_description": "MISS Home Guard 25' 3PT Jump Shot"}, "Home"),
        ("Away Guard", {"visitor_description": "MISS Away Guard 25' 3PT Jump Shot"}, "Away"),
    ],
)
def test_opening_jump_tip_resolves_to_either_team(tip_side, shot_desc, expected_side):
    out = _estimate(
        [
            _row(
                event_num=2,
                event_type=12,
                neutral_description="Start of 1st Period",
            ),
            _row(
                event_num=4,
                event_type=10,
                home_description=f"Jump Ball Home Center vs. Away Center: Tip to {tip_side}",
                player1_name="Home Center",
            ),
            _row(
                event_num=6,
                event_type=2,
                time_quarter="11:45",
                minute_remaining_quarter=11,
                seconds_remaining_quarter=45,
                player1_name=tip_side,
                **shot_desc,
            ),
        ]
    )

    shot = _shot(out, 6)
    assert shot["shot_side"] == expected_side
    assert shot["shot_clock_est"] == 9
    assert shot["confidence"] == "high"


def test_offensive_rebound_resets_next_shot_to_14():
    out = _estimate(
        [
            _row(
                event_num=10,
                event_type=2,
                time_quarter="11:40",
                minute_remaining_quarter=11,
                seconds_remaining_quarter=40,
                home_description="MISS Home Guard 25' 3PT Jump Shot",
                player1_name="Home Guard",
            ),
            _row(
                event_num=11,
                event_type=4,
                time_quarter="11:38",
                minute_remaining_quarter=11,
                seconds_remaining_quarter=38,
                home_description="Home Big REBOUND (Off:1 Def:0)",
                player1_name="Home Big",
            ),
            _row(
                event_num=12,
                event_type=2,
                time_quarter="11:31",
                minute_remaining_quarter=11,
                seconds_remaining_quarter=31,
                home_description="MISS Home Wing 3PT Jump Shot",
                player1_name="Home Wing",
            ),
        ]
    )

    shot = _shot(out, 12)
    assert shot["reset_len"] == 14
    assert shot["reset_reason"] == "off_rebound"
    assert shot["shot_clock_est"] == 7


def test_made_field_goal_and_turnover_reset_to_24():
    out = _estimate(
        [
            _row(
                event_num=10,
                event_type=1,
                time_quarter="11:40",
                minute_remaining_quarter=11,
                seconds_remaining_quarter=40,
                home_description="Home Guard 26' 3PT Jump Shot (3 PTS)",
                player1_name="Home Guard",
            ),
            _row(
                event_num=12,
                event_type=2,
                time_quarter="11:25",
                minute_remaining_quarter=11,
                seconds_remaining_quarter=25,
                visitor_description="MISS Away Guard 12' Pullup Jump Shot",
                player1_name="Away Guard",
            ),
            _row(
                event_num=14,
                event_type=5,
                time_quarter="11:00",
                minute_remaining_quarter=11,
                seconds_remaining_quarter=0,
                visitor_description="Away Guard Bad Pass Turnover (P1.T1)",
                player1_name="Away Guard",
            ),
            _row(
                event_num=16,
                event_type=2,
                time_quarter="10:50",
                minute_remaining_quarter=10,
                seconds_remaining_quarter=50,
                home_description="MISS Home Wing 25' 3PT Jump Shot",
                player1_name="Home Wing",
            ),
        ]
    )

    assert _shot(out, 12)["reset_reason"] == "made_fg"
    assert _shot(out, 12)["shot_clock_est"] == 9
    assert _shot(out, 16)["reset_reason"] == "turnover"
    assert _shot(out, 16)["shot_clock_est"] == 14


def test_final_missed_ft_offensive_rebound_resets_to_14():
    out = _estimate(
        [
            _row(
                event_num=20,
                event_type=3,
                time_quarter="11:40",
                minute_remaining_quarter=11,
                seconds_remaining_quarter=40,
                home_description="MISS Home Guard Free Throw 2 of 2",
                player1_name="Home Guard",
            ),
            _row(
                event_num=21,
                event_type=4,
                time_quarter="11:38",
                minute_remaining_quarter=11,
                seconds_remaining_quarter=38,
                home_description="Home Big REBOUND (Off:1 Def:0)",
                player1_name="Home Big",
            ),
            _row(
                event_num=22,
                event_type=1,
                time_quarter="11:30",
                minute_remaining_quarter=11,
                seconds_remaining_quarter=30,
                home_description="Home Big 1' Putback Layup (2 PTS)",
                player1_name="Home Big",
            ),
        ]
    )

    shot = _shot(out, 22)
    assert shot["reset_reason"] == "off_rebound"
    assert shot["reset_len"] == 14
    assert shot["shot_clock_est"] == 6


def test_final_missed_ft_defensive_rebound_resets_to_24():
    out = _estimate(
        [
            _row(
                event_num=20,
                event_type=3,
                time_quarter="11:40",
                minute_remaining_quarter=11,
                seconds_remaining_quarter=40,
                home_description="MISS Home Guard Free Throw 2 of 2",
                player1_name="Home Guard",
            ),
            _row(
                event_num=21,
                event_type=4,
                time_quarter="11:38",
                minute_remaining_quarter=11,
                seconds_remaining_quarter=38,
                visitor_description="Away Big REBOUND (Off:0 Def:1)",
                player1_name="Away Big",
            ),
            _row(
                event_num=22,
                event_type=2,
                time_quarter="11:20",
                minute_remaining_quarter=11,
                seconds_remaining_quarter=20,
                visitor_description="MISS Away Guard 25' 3PT Jump Shot",
                player1_name="Away Guard",
            ),
        ]
    )

    shot = _shot(out, 22)
    assert shot["reset_reason"] == "def_rebound"
    assert shot["reset_len"] == 24
    assert shot["shot_clock_est"] == 6


def test_technical_free_throw_preserves_current_shot_clock():
    out = _estimate(
        [
            _row(
                event_num=10,
                event_type=1,
                time_quarter="11:40",
                minute_remaining_quarter=11,
                seconds_remaining_quarter=40,
                home_description="Home Guard 26' 3PT Jump Shot (3 PTS)",
                player1_name="Home Guard",
            ),
            _row(
                event_num=12,
                event_type=3,
                time_quarter="11:30",
                minute_remaining_quarter=11,
                seconds_remaining_quarter=30,
                home_description="Home Guard Free Throw Technical (1 PTS)",
                player1_name="Home Guard",
            ),
            _row(
                event_num=14,
                event_type=2,
                time_quarter="11:20",
                minute_remaining_quarter=11,
                seconds_remaining_quarter=20,
                visitor_description="MISS Away Guard 3PT Jump Shot",
                player1_name="Away Guard",
            ),
        ]
    )

    shot = _shot(out, 14)
    assert shot["reset_reason"] == "made_fg"
    assert shot["shot_clock_est"] == 4


def test_corrected_events_use_clock_order_before_event_number_order():
    out = _estimate(
        [
            _row(
                event_num=143,
                event_type=1,
                time_quarter="01:57",
                minute_remaining_quarter=1,
                seconds_remaining_quarter=57,
                home_description="Home Guard 2' Running Layup (2 PTS)",
                player1_name="Home Guard",
            ),
            _row(
                event_num=155,
                event_type=1,
                time_quarter="01:19",
                minute_remaining_quarter=1,
                seconds_remaining_quarter=19,
                visitor_description="Away Guard 27' 3PT Pullup Jump Shot (3 PTS)",
                player1_name="Away Guard",
            ),
            _row(
                event_num=276,
                event_type=2,
                time_quarter="01:26",
                minute_remaining_quarter=1,
                seconds_remaining_quarter=26,
                visitor_description="MISS Away Big 3' Driving Layup",
                player1_name="Away Big",
            ),
            _row(
                event_num=278,
                event_type=4,
                time_quarter="01:26",
                minute_remaining_quarter=1,
                seconds_remaining_quarter=26,
                visitor_description="Away Rebound",
                player1_name="Away Big",
            ),
        ]
    )

    shot = _shot(out, 155)
    assert shot["reset_reason"] == "off_rebound"
    assert shot["shot_clock_est"] == 7
    assert shot["confidence"] == "high"
