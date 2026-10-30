import pandas as pd

from wnba_test.audit_wnba_source_inputs import (
    is_complete_last_row,
    parse_game_id,
)
from wnba_test.benchmark_wnba_metrics import latest_snapshot
from wnba_test.fetch_wnba_cdn import normalize_actions
from wnba_test.import_gabriel_modern_pbp import (
    game_metadata,
    json_safe_records,
)
from wnba_test.process_rapm_wnba import prepare_local_wnba_game_dates
from wnba_test.validate_wnba_normalized_lineups import lineup_quality_masks
from wnba_test.wnba_names import load_wnba_name_map


def test_wnba_game_id_metadata() -> None:
    assert parse_game_id("1022200001") == (2022, "RS")
    assert game_metadata("1042500304") == (2025, "PS")


def test_final_event_detection_rejects_schedule_placeholders() -> None:
    assert not is_complete_last_row("classic", None)
    assert not is_complete_last_row(
        "classic",
        {
            "EVENTMSGTYPE": "13",
            "PERIOD": "1",
            "NEUTRALDESCRIPTION": "End of 1st Period",
        },
    )
    assert is_complete_last_row(
        "classic",
        {
            "EVENTMSGTYPE": "13",
            "PERIOD": "4",
            "NEUTRALDESCRIPTION": "End of 4th Period",
        },
    )
    assert is_complete_last_row(
        "modern",
        {"actionType": "game", "subType": "end", "description": "Game End"},
    )


def test_json_safe_records_turns_nan_into_none() -> None:
    records = json_safe_records(pd.DataFrame([{"actionNumber": 1, "x": float("nan")}]))
    assert records == [{"actionNumber": 1, "x": None}]


def test_local_wnba_game_dates_use_single_calendar_season(tmp_path) -> None:
    path = tmp_path / "wnba_game_dates.csv"
    pd.DataFrame(
        [
            {
                "gameId": "1022200001",
                "date": "2022-05-06T00:00:00Z",
                "seasonType": "Regular Season",
            }
        ]
    ).to_csv(path, index=False)

    prepared = prepare_local_wnba_game_dates(path)

    assert prepared.to_dict("records") == [
        {"GAME_ID": 1022200001, "game_date": "2022-05-06", "season": 2022}
    ]


def test_normalize_actions_uses_boxscore_starters_and_grouped_substitutions() -> None:
    away_players = [
        {"personId": player_id, "starter": "1", "name": f"A{player_id}", "nameI": f"A.{player_id}"}
        for player_id in range(1, 6)
    ]
    away_players.append(
        {"personId": 6, "starter": "0", "name": "A6", "nameI": "A.6"}
    )
    home_players = [
        {
            "personId": player_id,
            "starter": "1",
            "name": f"H{player_id}",
            "nameI": f"H.{player_id}",
        }
        for player_id in range(11, 16)
    ]
    boxscore = {
        "game": {
            "gameId": "1022200001",
            "awayTeam": {
                "teamId": 100,
                "teamTricode": "AWY",
                "players": away_players,
            },
            "homeTeam": {
                "teamId": 200,
                "teamTricode": "HME",
                "players": home_players,
            },
        }
    }
    actions = [
        {
            "actionNumber": 1,
            "period": 1,
            "clock": "PT10M00.00S",
            "actionType": "period",
            "subType": "start",
            "description": "Period Start",
        },
        {
            "actionNumber": 2,
            "period": 1,
            "clock": "PT05M00.00S",
            "actionType": "2pt",
            "subType": "Jump Shot",
            "teamId": 100,
            "personId": 5,
            "shotResult": "Made",
            "description": "A5 2PT Jump Shot",
        },
        {
            "actionNumber": 3,
            "period": 1,
            "clock": "PT05M00.00S",
            "actionType": "substitution",
            "subType": "out",
            "teamId": 100,
            "personId": 5,
            "description": "A5 substitution out",
        },
        {
            "actionNumber": 4,
            "period": 1,
            "clock": "PT05M00.00S",
            "actionType": "substitution",
            "subType": "in",
            "teamId": 100,
            "personId": 6,
            "description": "A6 substitution in",
        },
    ]

    normalized = normalize_actions("1022200001", actions, boxscore)

    assert len(normalized) == 4
    assert normalized.loc[0, "time_quarter"] == "10:00"
    assert normalized.loc[0, "score_margin"] == "TIE"
    assert set(normalized.loc[0, [f"away_player{i}" for i in range(1, 6)]]) == {
        1,
        2,
        3,
        4,
        5,
    }
    assert set(normalized.loc[1, [f"away_player{i}" for i in range(1, 6)]]) == {
        1,
        2,
        3,
        4,
        5,
    }
    assert set(normalized.loc[2, [f"away_player{i}" for i in range(1, 6)]]) == {
        1,
        2,
        3,
        4,
        6,
    }


def test_normalize_actions_preserves_chained_substitution_terminal_player() -> None:
    away_players = [
        {"personId": player_id, "starter": "1", "name": f"A{player_id}", "nameI": f"A.{player_id}"}
        for player_id in range(1, 6)
    ]
    away_players.extend(
        [
            {"personId": 6, "starter": "0", "name": "A6", "nameI": "A.6"},
            {"personId": 7, "starter": "0", "name": "A7", "nameI": "A.7"},
        ]
    )
    home_players = [
        {
            "personId": player_id,
            "starter": "1",
            "name": f"H{player_id}",
            "nameI": f"H.{player_id}",
        }
        for player_id in range(11, 16)
    ]
    boxscore = {
        "game": {
            "gameId": "1022200001",
            "awayTeam": {
                "teamId": 100,
                "teamTricode": "AWY",
                "players": away_players,
            },
            "homeTeam": {
                "teamId": 200,
                "teamTricode": "HME",
                "players": home_players,
            },
        }
    }
    actions = [
        {
            "actionNumber": 1,
            "period": 1,
            "clock": "PT05M00.00S",
            "actionType": "substitution",
            "subType": "out",
            "teamId": 100,
            "personId": 5,
        },
        {
            "actionNumber": 2,
            "period": 1,
            "clock": "PT05M00.00S",
            "actionType": "substitution",
            "subType": "in",
            "teamId": 100,
            "personId": 6,
        },
        {
            "actionNumber": 3,
            "period": 1,
            "clock": "PT05M00.00S",
            "actionType": "substitution",
            "subType": "out",
            "teamId": 100,
            "personId": 6,
        },
        {
            "actionNumber": 4,
            "period": 1,
            "clock": "PT05M00.00S",
            "actionType": "substitution",
            "subType": "in",
            "teamId": 100,
            "personId": 7,
        },
        {
            "actionNumber": 5,
            "period": 1,
            "clock": "PT05M00.00S",
            "actionType": "freethrow",
            "subType": "1 of 1",
            "teamId": 100,
            "personId": 7,
            "description": "A7 Free Throw 1 of 1",
        },
    ]

    normalized = normalize_actions("1022200001", actions, boxscore)
    terminal_lineup = set(
        normalized.loc[4, [f"away_player{i}" for i in range(1, 6)]]
    )

    assert terminal_lineup == {1, 2, 3, 4, 7}
    assert not lineup_quality_masks(normalized)["invalid"].any()


def test_normalize_actions_uses_pre_sub_lineup_for_corrected_same_clock_actor() -> None:
    away_players = [
        {"personId": player_id, "starter": "1", "name": f"A{player_id}", "nameI": f"A.{player_id}"}
        for player_id in range(1, 6)
    ]
    away_players.append(
        {"personId": 6, "starter": "0", "name": "A6", "nameI": "A.6"}
    )
    home_players = [
        {
            "personId": player_id,
            "starter": "1",
            "name": f"H{player_id}",
            "nameI": f"H.{player_id}",
        }
        for player_id in range(11, 16)
    ]
    boxscore = {
        "game": {
            "gameId": "1022200001",
            "awayTeam": {
                "teamId": 100,
                "teamTricode": "AWY",
                "players": away_players,
            },
            "homeTeam": {
                "teamId": 200,
                "teamTricode": "HME",
                "players": home_players,
            },
        }
    }
    actions = [
        {
            "actionNumber": 1,
            "period": 1,
            "clock": "PT05M00.00S",
            "actionType": "substitution",
            "subType": "out",
            "teamId": 100,
            "personId": 5,
        },
        {
            "actionNumber": 2,
            "period": 1,
            "clock": "PT05M00.00S",
            "actionType": "substitution",
            "subType": "in",
            "teamId": 100,
            "personId": 6,
        },
        {
            "actionNumber": 3,
            "period": 1,
            "clock": "PT05M00.00S",
            "actionType": "turnover",
            "subType": "bad pass",
            "teamId": 100,
            "personId": 5,
            "description": "A5 bad pass turnover",
        },
        {
            "actionNumber": 4,
            "period": 1,
            "clock": "PT04M59.00S",
            "actionType": "2pt",
            "subType": "Jump Shot",
            "teamId": 100,
            "personId": 6,
            "shotResult": "Made",
            "description": "A6 2PT Jump Shot",
        },
    ]

    normalized = normalize_actions("1022200001", actions, boxscore)

    assert set(
        normalized.loc[2, [f"away_player{i}" for i in range(1, 6)]]
    ) == {1, 2, 3, 4, 5}
    assert set(
        normalized.loc[3, [f"away_player{i}" for i in range(1, 6)]]
    ) == {1, 2, 3, 4, 6}
    assert not lineup_quality_masks(normalized)["invalid"].any()


def test_lineup_quality_ignores_admin_actor_but_flags_live_actor() -> None:
    rows = []
    for actiontype, subtype, personid in [
        ("Foul", "technical", 999),
        ("Made Shot", "Jump Shot", 999),
    ]:
        row = {
            "game_id": "1022200001",
            "teamid": 100,
            "personid": personid,
            "location": "v",
            "actiontype": actiontype,
            "subtype": subtype,
        }
        for idx, player_id in enumerate(range(1, 6), start=1):
            row[f"away_player{idx}"] = player_id
        for idx, player_id in enumerate(range(11, 16), start=1):
            row[f"home_player{idx}"] = player_id
        rows.append(row)
    masks = lineup_quality_masks(pd.DataFrame(rows))

    assert masks["admin_actor_off_lineup"].tolist() == [True, False]
    assert masks["live_actor_off_lineup"].tolist() == [False, True]


def test_wnba_name_map_falls_back_to_normalized_events(tmp_path) -> None:
    normalized = tmp_path / "normalized"
    normalized.mkdir()
    pd.DataFrame(
        [
            {"personid": 203826, "playername": "Alyssa Thomas"},
            {"personid": 0, "playername": "TEAM"},
        ]
    ).to_csv(normalized / "WNBA25.csv", index=False)

    names = load_wnba_name_map(map_paths=[], normalized_dirs=[normalized])

    assert names == {"203826": "Alyssa Thomas"}


def test_rating_snapshot_is_strictly_before_cutoff() -> None:
    ratings = pd.DataFrame(
        [
            {"date": "2026-05-07", "nba_id": "1", "o": 1.0},
            {"date": "2026-05-08", "nba_id": "1", "o": 9.0},
            {"date": "2026-05-06", "nba_id": "2", "o": 2.0},
        ]
    )
    ratings["date"] = pd.to_datetime(ratings["date"])

    snapshot = latest_snapshot(ratings, pd.Timestamp("2026-05-08"))

    values = dict(zip(snapshot["nba_id"], snapshot["o"]))
    assert values == {"1": 1.0, "2": 2.0}
