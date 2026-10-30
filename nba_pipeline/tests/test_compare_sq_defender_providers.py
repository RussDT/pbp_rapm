import numpy as np
import pandas as pd

from nba_pipeline.scripts.compare_sq_defender_providers import (
    attach_event_lineups,
    attach_lineup_depm_features,
    attach_nba_assignment_priors,
    load_event_lineups,
    load_nba_assignments,
    parse_defender_ids,
    platt_calibrate,
)


def test_parse_defender_ids_preserves_every_unique_assignment():
    assert parse_defender_ids("203999| 1641705 |203999") == (203999, 1641705)
    assert parse_defender_ids("") == ()
    assert parse_defender_ids(np.nan) == ()


def test_load_nba_assignments_collapses_player_file_duplicates(tmp_path):
    directory = tmp_path / "defender" / "2025"
    directory.mkdir(parents=True)
    rows = pd.DataFrame(
        {
            "GAME_ID": ["0022400001", "0022400002"],
            "GAME_EVENT_ID": [10, 20],
            "PLAYER_ID": [7, 8],
            "SHOT_MADE_FLAG": [1, 0],
            "GAME_DATE": ["20241022", "20241023"],
            "DEF_ID": ["11|12", "13"],
        }
    )
    rows.to_csv(directory / "11.csv", index=False)
    rows.iloc[[0]].to_csv(directory / "12.csv", index=False)
    resolved, audit = load_nba_assignments(tmp_path, seasons=[2025])

    assert len(resolved) == 2
    assert resolved.loc[
        resolved["GAME_EVENT_ID"].eq(10), "nba_defender_ids"
    ].iloc[0] == (11, 12)
    assert audit["exact_duplicate_rows_removed"] == 1
    assert audit["multi_assignment_events"] == 1


def test_assignment_prior_excludes_all_current_game_outcomes():
    shots = pd.DataFrame(
        {
            "game_id_normalized": ["1", "1", "2", "2"],
            "GAME_EVENT_ID": [10, 11, 20, 21],
            "game_date": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-01-03", "2024-01-03"]
            ),
            "segment": ["three"] * 4,
            "made": [1, 1, 0, 0],
        }
    )
    assignments = pd.DataFrame(
        {
            "game_id_normalized": ["1", "1", "2", "2"],
            "GAME_EVENT_ID": [10, 11, 20, 21],
            "nba_defender_ids": [(9,), (9,), (9,), (9,)],
            "nba_assignment_count": [1.0] * 4,
            "nba_single_assignment": [1.0] * 4,
            "nba_assignment_available": [1.0] * 4,
        }
    )
    out, audit = attach_nba_assignment_priors(
        shots,
        assignments,
        shrink_attempts=2.0,
        baseline_by_segment={"three": 0.5},
    )

    np.testing.assert_allclose(
        out.loc[:1, "nba_assignment_prior_make_probability_allowed"],
        [0.5, 0.5],
    )
    np.testing.assert_allclose(
        out.loc[2:, "nba_assignment_prior_make_probability_allowed"],
        [0.75, 0.75],
    )
    assert audit["matched_events"] == 4


def test_platt_calibration_returns_probabilities_and_parameters():
    calibrated, parameters = platt_calibrate(
        np.array([0.2, 0.3, 0.7, 0.8]),
        np.array([0, 0, 1, 1]),
        np.array([0.25, 0.75]),
    )

    assert np.all((calibrated > 0.0) & (calibrated < 1.0))
    assert calibrated[0] < calibrated[1]
    assert set(parameters) == {"intercept", "slope"}


def test_event_lineup_join_orients_defense_away_from_shooter():
    shots = pd.DataFrame(
        {
            "game_id_normalized": ["1", "2"],
            "GAME_EVENT_ID": [10, 20],
            "personId": [1, 15],
        }
    )
    lineups = pd.DataFrame(
        {
            "game_id_normalized": ["1", "2"],
            "GAME_EVENT_ID": [10, 20],
            "home_player1": [1, 11],
            "home_player2": [2, 12],
            "home_player3": [3, 13],
            "home_player4": [4, 14],
            "home_player5": [5, 15],
            "away_player1": [11, 1],
            "away_player2": [12, 2],
            "away_player3": [13, 3],
            "away_player4": [14, 4],
            "away_player5": [15, 5],
        }
    )
    out, audit = attach_event_lineups(shots, lineups)

    assert out.loc[0, "def_lineup_player1"] == 11
    assert out.loc[1, "def_lineup_player1"] == 1
    assert out["def_lineup_available"].tolist() == [1.0, 1.0]
    assert audit["valid_oriented_lineups"] == 2


def test_event_lineup_join_rejects_duplicate_or_missing_shooter_sides():
    shots = pd.DataFrame(
        {
            "game_id_normalized": ["1", "2"],
            "GAME_EVENT_ID": [10, 20],
            "personId": [1, 99],
        }
    )
    lineups = pd.DataFrame(
        {
            "game_id_normalized": ["1", "2"],
            "GAME_EVENT_ID": [10, 20],
            "home_player1": [1, 1],
            "home_player2": [2, 2],
            "home_player3": [3, 3],
            "home_player4": [4, 4],
            "home_player5": [5, 5],
            "away_player1": [1, 11],
            "away_player2": [12, 12],
            "away_player3": [13, 13],
            "away_player4": [14, 14],
            "away_player5": [15, 15],
        }
    )
    out, audit = attach_event_lineups(shots, lineups)

    assert out["def_lineup_available"].tolist() == [0.0, 0.0]
    assert out["def_lineup_player1"].isna().all()
    assert audit["invalid_or_ambiguous_orientation_rows"] == 2


def test_lineup_depm_uses_latest_nonfuture_snapshot_and_zero_missing():
    shots = pd.DataFrame(
        {
            "game_date": pd.to_datetime(["2024-01-10", "2024-01-10"]),
            "def_lineup_available": [1.0, 0.0],
            "def_lineup_player1": pd.Series([11, pd.NA], dtype="Int64"),
            "def_lineup_player2": pd.Series([12, pd.NA], dtype="Int64"),
            "def_lineup_player3": pd.Series([13, pd.NA], dtype="Int64"),
            "def_lineup_player4": pd.Series([14, pd.NA], dtype="Int64"),
            "def_lineup_player5": pd.Series([15, pd.NA], dtype="Int64"),
        }
    )
    snapshots = pd.DataFrame(
        {
            "player_id": pd.Series(
                [11, 12, 13, 14, 11], dtype="Int64"
            ),
            "depm": [1.0, -1.0, 0.5, 2.0, 99.0],
            "snapshot_date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-11",
                ]
            ),
        }
    ).sort_values(["snapshot_date", "player_id"])
    out, audit = attach_lineup_depm_features(shots, snapshots)

    assert out.loc[0, "def_lineup_depm_coverage"] == 4
    assert out.loc[0, "def_lineup_depm_sum"] == 2.5
    assert out.loc[0, "def_lineup_depm_min"] == -1.0
    assert out.loc[0, "def_lineup_depm_max"] == 2.0
    assert out.loc[0, "def_lineup_epm_snapshot_age_max"] == 9
    assert pd.isna(out.loc[1, "def_lineup_depm_sum"])
    assert audit["complete_five_player_epm_rows"] == 0


def test_load_event_lineups_excludes_conflicting_event_keys(tmp_path):
    rows = pd.DataFrame(
        {
            "game_id": ["0022400001", "0022400001"],
            "event_num": [10, 10],
            "home_player1": [1, 1],
            "home_player2": [2, 2],
            "home_player3": [3, 3],
            "home_player4": [4, 4],
            "home_player5": [5, 99],
            "away_player1": [11, 11],
            "away_player2": [12, 12],
            "away_player3": [13, 13],
            "away_player4": [14, 14],
            "away_player5": [15, 15],
        }
    )
    rows.to_parquet(tmp_path / "NBA25.parquet", index=False)
    resolved, audit = load_event_lineups(tmp_path, seasons=[2025])

    assert resolved.empty
    assert audit["conflicting_event_keys_removed"] == 1
