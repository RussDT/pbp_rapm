import sys
from pathlib import Path

import pandas as pd


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from process_rapm_blocks.process_stop import build_stop_flags
from build_stop_display import build_stop_display


def _row(event_num, poss_group, event_type, home="", away="", offense="Home", eop=False):
    return {
        "game_id": "0022500001",
        "event_num": event_num,
        "period": 1,
        "poss_group": poss_group,
        "event_type": event_type,
        "home_description": home,
        "visitor_description": away,
        "TeamOnOffense": offense,
        "End_of_Possession": eop,
        "player1_id": 1,
    }


def test_stop_is_union_of_turnovers_and_defense_recovered_blocks():
    frame = pd.DataFrame(
        [
            _row(1, 0, "Turnover", home="Turnover", eop=True),
            _row(2, 1, "MISS", home="MISS Layup", away="BLOCK", offense="Home"),
            _row(3, 1, "Rebound", away="PLAYER REBOUND", offense="Home", eop=True),
            _row(4, 2, "MISS", away="MISS Layup", home="BLOCK", offense="Away"),
            _row(5, 2, "Rebound", away="PLAYER REBOUND", offense="Away", eop=True),
            _row(6, 3, "MAKE", home="2 PTS", eop=True),
        ]
    )

    result = build_stop_flags(frame)
    terminal = result[result["End_of_Possession"]].set_index("poss_group")

    assert terminal.loc[0, "Is_Stop"] == 1
    assert terminal.loc[0, "Is_Turnover_Stop"] == 1
    assert terminal.loc[1, "Is_Stop"] == 1
    assert terminal.loc[1, "Is_Recovered_Block_Stop"] == 1
    assert terminal.loc[2, "Is_Stop"] == 0
    assert terminal.loc[3, "Is_Stop"] == 0


def test_offensive_foul_without_separate_turnover_is_a_stop():
    frame = pd.DataFrame(
        [
            _row(
                1,
                0,
                "Foul",
                home="Offensive Foul",
                offense="Home",
                eop=True,
            )
        ]
    )

    result = build_stop_flags(frame)
    assert result.loc[0, "Is_Turnover_Stop"] == 1
    assert result.loc[0, "Is_Stop"] == 1


def test_stop_display_flips_defense_once_and_closes_to_net():
    raw = pd.DataFrame(
        [
            {
                "player_id": 1,
                "player_name": "Fallback",
                "off": 1.25,
                "def": -0.75,
                "net_rapm": 2.0,
                "possessions": 1000,
                "off_poss": 500,
                "def_poss": 500,
            }
        ]
    )
    identity = pd.DataFrame([{"nba_id": 1, "name": "Full Name"}])

    display = build_stop_display(raw, identity)

    assert display.loc[0, "name"] == "Full Name"
    assert display.loc[0, "oSTOP"] == 1.25
    assert display.loc[0, "dSTOP"] == 0.75
    assert display.loc[0, "STOP"] == 2.0
