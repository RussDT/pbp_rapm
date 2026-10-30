from __future__ import annotations

import numpy as np
import pandas as pd

from nba_pipeline.scripts.build_championship_help import (
    CONTRACT_REPLACEMENT_SALARY_SHARE,
    add_title_lift_over_contract,
    attach_playoff_adjusted_ratings,
    attach_salary_share,
    attach_neutral_playoff_baseline,
    attribute_team_probabilities,
    build_salary_peer_focal_seasons,
    build_player_title_context,
    build_game_context,
    build_career_championship_help,
    collapse_processed_team_exposures,
    contract_epm_from_salary_share,
    cross_validated_team_predictions,
    extract_champion_from_games,
    fit_team_title_model,
    load_epm_snapshot_history,
    select_epm_snapshot,
    select_team_model_candidate,
    select_final_team_segments,
    shapley_teammate_attribution,
    summarize_retrospective_team_season,
    summarize_team_pair_exposures,
    summarize_team_lineup_exposures,
)
from nba_pipeline.scripts.build_darko_teammate_help import PlayerRatingHistory


def test_playoff_adjusted_team_rating_sums_adjusted_rotation_contributions() -> None:
    seasons = pd.DataFrame(
        [
            {
                "season": 2025,
                "team_id": 1,
                "player_id": 10,
                "own_dpm": 2.0,
                "playoff_workload_share": 0.60,
            },
            {
                "season": 2025,
                "team_id": 1,
                "player_id": 11,
                "own_dpm": 1.0,
                "playoff_workload_share": 0.40,
            },
        ]
    )
    teams = pd.DataFrame([{"season": 2025, "team_id": 1, "team_dpm": 1.60}])
    ratings = pd.DataFrame(
        [
            {
                "season": 2025,
                "player_id": 10,
                "playoff_adjusted_dpm": 3.0,
                "playoff_translation_dpm": 1.0,
                "playoff_prior_possessions": 1000,
            },
            {
                "season": 2025,
                "player_id": 11,
                "playoff_adjusted_dpm": 0.5,
                "playoff_translation_dpm": -0.5,
                "playoff_prior_possessions": 800,
            },
        ]
    )

    adjusted_seasons, adjusted_teams = attach_playoff_adjusted_ratings(
        seasons, teams, ratings
    )

    assert np.isclose(adjusted_teams.loc[0, "playoff_adjusted_team_dpm"], 2.0)
    assert np.allclose(
        adjusted_seasons["playoff_adjusted_team_contribution_dpm"], [1.8, 0.2]
    )


def test_epm_history_uses_latest_monthly_snapshot_before_postseason(tmp_path) -> None:
    pd.DataFrame(
        [
            {
                "requested_date": "2025-04-01",
                "date": "2025-04-01",
                "game_dt": "2025-03-31",
                "game_optimized": 0,
                "player_id": 1,
                "player_name": "One",
                "oepm": 2.0,
                "depm": 1.0,
                "epm": 3.0,
            }
        ]
    ).to_csv(tmp_path / "epm_all_2025-04-01_go0.csv", index=False)
    pd.DataFrame(
        [
            {
                "requested_date": "2025-05-01",
                "date": "2025-05-01",
                "game_dt": "2025-04-30",
                "game_optimized": 0,
                "player_id": 1,
                "player_name": "One",
                "oepm": 9.0,
                "depm": 1.0,
                "epm": 10.0,
            }
        ]
    ).to_csv(tmp_path / "epm_all_2025-05-01_go0.csv", index=False)

    history, report = load_epm_snapshot_history(tmp_path)
    snapshot = select_epm_snapshot(history, "2025-04-20")

    assert report["snapshots"] == 2
    assert snapshot.loc[0, "nba_id"] == 1
    assert snapshot.loc[0, "dpm"] == 3.0
    assert snapshot.loc[0, "o_dpm"] == 2.0
    assert snapshot.loc[0, "d_dpm"] == 1.0
    assert snapshot.loc[0, "snapshot_requested_date"] == pd.Timestamp("2025-04-01")


def test_epm_snapshot_selection_never_backfills_from_the_future(tmp_path) -> None:
    pd.DataFrame(
        [
            {
                "requested_date": "2025-05-01",
                "date": "2025-05-01",
                "game_dt": "2025-04-30",
                "game_optimized": 0,
                "player_id": 1,
                "player_name": "One",
                "oepm": 2.0,
                "depm": 1.0,
                "epm": 3.0,
            }
        ]
    ).to_csv(tmp_path / "epm_all_2025-05-01_go0.csv", index=False)

    history, _ = load_epm_snapshot_history(tmp_path)

    with np.testing.assert_raises(ValueError):
        select_epm_snapshot(history, "2025-04-20")


def test_epm_snapshot_keeps_explicit_bubble_season_label(tmp_path) -> None:
    pd.DataFrame(
        [
            {
                "requested_date": "2020-08-01",
                "date": "2020-08-01",
                "game_dt": "2020-03-10",
                "game_optimized": 0,
                "player_id": 1,
                "player_name": "One",
                "oepm": 2.0,
                "depm": 1.0,
                "epm": 3.0,
            }
        ]
    ).to_csv(tmp_path / "epm_all_2020-08-01_go0.csv", index=False)

    history, _ = load_epm_snapshot_history(tmp_path)
    snapshot = select_epm_snapshot(history, "2020-08-17", season=2020)

    assert snapshot.loc[0, "season"] == 2020


def test_neutral_playoff_rotation_is_the_same_season_field_average() -> None:
    teams = pd.DataFrame(
        [
            {"season": 2025, "team_id": 1, "team_dpm": 2.0, "team_o_dpm": 1.5, "team_d_dpm": 0.5},
            {"season": 2025, "team_id": 2, "team_dpm": 6.0, "team_o_dpm": 4.0, "team_d_dpm": 2.0},
        ]
    )

    result = attach_neutral_playoff_baseline(teams)

    assert result["neutral_team_dpm"].tolist() == [4.0, 4.0]
    assert result["neutral_team_o_dpm"].tolist() == [2.75, 2.75]
    assert result["neutral_team_d_dpm"].tolist() == [1.25, 1.25]


def test_common_title_field_does_not_remove_the_focal_players_actual_team() -> None:
    training = pd.DataFrame(
        [
            {"season": 2024, "team_id": 1, "team_dpm": 2.0, "won_championship": 0},
            {"season": 2024, "team_id": 2, "team_dpm": 4.0, "won_championship": 0},
            {"season": 2024, "team_id": 3, "team_dpm": 6.0, "won_championship": 1},
            {"season": 2024, "team_id": 4, "team_dpm": 8.0, "won_championship": 0},
        ]
    )
    model = fit_team_title_model(training, spec="overall", c_value=1.0)
    states = pd.DataFrame(
        [
            {"season": 2024, "team_id": 99, "team_dpm": 5.0},
            {"season": 2024, "team_id": 99, "team_dpm": 7.0},
        ]
    )

    shared = model.common_field_target_probabilities(states, training)
    remove_weak_team = model.target_probabilities(states, training, team_id=1)
    remove_strong_team = model.target_probabilities(states, training, team_id=4)

    assert shared[1] > shared[0]
    assert not np.allclose(remove_weak_team, remove_strong_team)
    assert np.allclose(
        shared,
        model.common_field_target_probabilities(states, training.sample(frac=1, random_state=7)),
    )


def test_contract_epm_hurdle_is_monotone_and_anchors_replacement_salary() -> None:
    shares = np.array([0.01, CONTRACT_REPLACEMENT_SALARY_SHARE, 0.20, 0.35])
    hurdles = contract_epm_from_salary_share(shares)

    assert np.all(np.diff(hurdles) > 0)
    assert abs(hurdles[1] - (-2.0)) < 1e-12
    assert 2.0 < hurdles[-1] < 2.5


def test_salary_share_attachment_never_guesses_another_teams_cost() -> None:
    players = pd.DataFrame(
        [
            {"season": 2025, "player_id": 1, "team_abbr": "AAA"},
            {"season": 2025, "player_id": 2, "team_abbr": "BBB"},
        ]
    )
    salary = pd.DataFrame(
        [
            {
                "season": 2025,
                "player_id": 1,
                "salary_team_abbr": "AAA",
                "salary": 10.0,
                "salary_cap": 100.0,
                "salary_share": 0.10,
            },
            {
                "season": 2025,
                "player_id": 2,
                "salary_team_abbr": "CCC",
                "salary": 40.0,
                "salary_cap": 100.0,
                "salary_share": 0.40,
            },
        ]
    )

    result, report = attach_salary_share(players, salary)

    assert result.loc[result["player_id"].eq(1), "salary_share"].item() == 0.10
    assert pd.isna(result.loc[result["player_id"].eq(2), "salary_share"].item())
    assert report["covered_player_seasons"] == 1


def test_salary_peer_focal_seasons_include_nonplayoff_players() -> None:
    playoff_seasons = pd.DataFrame(
        [
            {
                "season": 2025,
                "player_id": 1,
                "player_name": "Playoff Player",
                "team_id": 100,
                "team_abbr": "AAA",
                "own_dpm": 3.0,
                "title_probability": 0.10,
            }
        ]
    )
    rating_snapshots = pd.DataFrame(
        [
            {"season": 2025, "nba_id": 1, "player_name": "Playoff Player", "dpm": 3.0},
            {"season": 2025, "nba_id": 2, "player_name": "Nonplayoff Player", "dpm": 0.5},
        ]
    )
    salary_history = pd.DataFrame(
        [
            {
                "season": 2025,
                "player_id": 1,
                "salary_team_abbr": "AAA",
                "salary": 10.0,
                "salary_cap": 100.0,
                "salary_share": 0.10,
            },
            {
                "season": 2025,
                "player_id": 2,
                "salary_team_abbr": "BBB",
                "salary": 15.0,
                "salary_cap": 100.0,
                "salary_share": 0.15,
            },
            {
                "season": 2025,
                "player_id": 2,
                "salary_team_abbr": "CCC",
                "salary": 5.0,
                "salary_cap": 100.0,
                "salary_share": 0.05,
            },
        ]
    )

    result, report = build_salary_peer_focal_seasons(
        playoff_seasons,
        rating_snapshots,
        salary_history,
    )

    assert result[["player_id", "is_playoff_participant"]].to_dict("records") == [
        {"player_id": 1, "is_playoff_participant": True},
        {"player_id": 2, "is_playoff_participant": False},
    ]
    nonplayoff = result.loc[result["player_id"].eq(2)].iloc[0]
    assert nonplayoff["own_dpm"] == 0.5
    assert nonplayoff["salary"] == 20.0
    assert nonplayoff["salary_share"] == 0.20
    assert nonplayoff["team_abbr"] == "BBB"
    assert pd.isna(nonplayoff["title_probability"])
    assert report["salary_peer_focal_player_seasons"] == 2
    assert report["nonplayoff_salary_peer_player_seasons"] == 1


def test_career_salary_peer_totals_include_nonplayoff_seasons() -> None:
    context_values = {
        "won_championship": 0.0,
        "title_probability": 0.10,
        "neutral_teammate_title_probability": 0.08,
        "self_context_title_probability": 0.02,
        "neutral_team_title_probability": 0.05,
        "neutral_field_neutral_title_probability": 0.04,
        "neutral_field_hard_carry_title_probability": 0.09,
        "era_neutral_title_probability": 0.04,
        "era_neutral_hard_carry_title_probability": 0.09,
        "normalized_self_title_probability_added": 0.03,
        "self_replacement_title_probability_added": 0.02,
        "on_court_teammate_title_probability_added": 0.01,
        "bench_title_probability_added": 0.01,
        "teammate_help_championships": 0.02,
        "help_dpm": 1.0,
    }
    seasons = pd.DataFrame(
        [
            {
                "season": 2024,
                "player_id": 1,
                "player_name": "One",
                "is_playoff_participant": True,
                "title_lift_over_contract": 0.10,
                "neutral_contract_actual_title_probability": 0.16,
                "neutral_contract_hurdle_title_probability": 0.06,
                **context_values,
            },
            {
                "season": 2025,
                "player_id": 1,
                "player_name": "One",
                "is_playoff_participant": False,
                "title_lift_over_contract": 0.05,
                "neutral_contract_actual_title_probability": 0.11,
                "neutral_contract_hurdle_title_probability": 0.06,
                **{key: np.nan for key in context_values},
            },
            {
                "season": 2025,
                "player_id": 2,
                "player_name": "Two",
                "is_playoff_participant": False,
                "title_lift_over_contract": 0.02,
                "neutral_contract_actual_title_probability": 0.08,
                "neutral_contract_hurdle_title_probability": 0.06,
                **{key: np.nan for key in context_values},
            },
        ]
    )
    attributed = pd.DataFrame(
        [
            {
                "player_id": 1,
                "teammate_id": 9,
                "player_name": "One",
                "teammate_name": "Nine",
                "season": 2024,
                "shared_exposures": 10.0,
                "rated_exposures": 10.0,
                "weighted_dpm": 0.0,
                "won_championship": 0.0,
                "championships_added": 0.02,
            }
        ]
    )

    career, _, _ = build_career_championship_help(seasons, attributed)

    one = career.loc[career["player_id"].eq(1)].iloc[0]
    two = career.loc[career["player_id"].eq(2)].iloc[0]
    assert one["seasons_modeled"] == 2
    assert one["playoff_seasons_modeled"] == 1
    assert np.isclose(one["title_lift_over_contract_championships"], 0.15)
    assert two["seasons_modeled"] == 1
    assert two["playoff_seasons_modeled"] == 0
    assert two["title_lift_over_contract_championships"] == 0.02
    assert pd.isna(two["expected_championships"])


def test_title_lift_over_contract_holds_playoff_placements_and_workload_fixed() -> None:
    training = pd.DataFrame(
        [
            {"season": 2022, "team_id": 1, "team_dpm": -1.0, "won_championship": 0},
            {"season": 2022, "team_id": 2, "team_dpm": 1.0, "won_championship": 1},
            {"season": 2023, "team_id": 3, "team_dpm": -0.5, "won_championship": 0},
            {"season": 2023, "team_id": 4, "team_dpm": 1.5, "won_championship": 1},
        ]
    )
    model = fit_team_title_model(training, spec="overall", c_value=1.0)
    hurdle = float(contract_epm_from_salary_share(0.20))
    seasons = pd.DataFrame(
        [
            {
                "own_dpm": hurdle,
                "salary_share": 0.20,
                "total_exposures": 60.0,
                "team_total_exposures": 100.0,
            },
            {
                "own_dpm": hurdle + 2.0,
                "salary_share": 0.20,
                "total_exposures": 60.0,
                "team_total_exposures": 100.0,
            },
            {
                "own_dpm": hurdle - 2.0,
                "salary_share": 0.20,
                "total_exposures": 60.0,
                "team_total_exposures": 100.0,
            },
        ]
    )

    result = add_title_lift_over_contract(seasons, model, training)

    assert abs(result.loc[0, "title_lift_over_contract"]) < 1e-12
    assert result.loc[1, "title_lift_over_contract"] > 0
    assert result.loc[2, "title_lift_over_contract"] < 0
    assert result["playoff_workload_share"].tolist() == [0.6, 0.6, 0.6]
    assert np.allclose(
        result["neutral_contract_actual_title_probability"],
        result["era_neutral_hard_carry_title_probability"],
    )
    assert np.allclose(
        result["playoff_field_zero_title_probability"],
        result["era_neutral_title_probability"],
    )
    raw_title_lift = (
        result["era_neutral_hard_carry_title_probability"]
        - result["era_neutral_title_probability"]
    )
    assert result.loc[1, "title_lift_over_contract"] < raw_title_lift.loc[1]


def test_final_team_segment_uses_last_regular_season_game() -> None:
    player_team = pd.DataFrame(
        [
            {
                "season": 2025,
                "player_id": 7,
                "team_id": 100,
                "last_game_date": "2025-02-01",
                "total_exposures": 4_000,
            },
            {
                "season": 2025,
                "player_id": 7,
                "team_id": 200,
                "last_game_date": "2025-04-10",
                "total_exposures": 2_000,
            },
            {
                "season": 2025,
                "player_id": 8,
                "team_id": 100,
                "last_game_date": "2025-04-11",
                "total_exposures": 5_000,
            },
        ]
    )
    pair_team = pd.DataFrame(
        [
            {"season": 2025, "player_id": 7, "team_id": 100, "teammate_id": 8},
            {"season": 2025, "player_id": 7, "team_id": 200, "teammate_id": 9},
            {"season": 2025, "player_id": 8, "team_id": 100, "teammate_id": 7},
        ]
    )

    selected_player, selected_pair = select_final_team_segments(player_team, pair_team)

    assert selected_player[["player_id", "team_id"]].to_dict("records") == [
        {"player_id": 7, "team_id": 200},
        {"player_id": 8, "team_id": 100},
    ]
    assert selected_pair[["player_id", "team_id", "teammate_id"]].to_dict("records") == [
        {"player_id": 7, "team_id": 200, "teammate_id": 9},
        {"player_id": 8, "team_id": 100, "teammate_id": 7},
    ]


def test_champion_is_the_winner_of_the_last_playoff_game() -> None:
    games = pd.DataFrame(
        [
            {
                "game_id": "0042400405",
                "game_date": "2025-06-19",
                "home_team_id": 1,
                "home_team_abbr": "AAA",
                "away_team_id": 2,
                "away_team_abbr": "BBB",
                "home_score": 101,
                "away_score": 99,
            },
            {
                "game_id": "0042400406",
                "game_date": "2025-06-22",
                "home_team_id": 2,
                "home_team_abbr": "BBB",
                "away_team_id": 1,
                "away_team_abbr": "AAA",
                "home_score": 94,
                "away_score": 103,
            },
        ]
    )

    champion = extract_champion_from_games(games, season=2025)

    assert champion == {
        "season": 2025,
        "champion_team_id": 1,
        "champion_team_abbr": "AAA",
        "clinching_game_id": "0042400406",
        "clinching_game_date": "2025-06-22",
    }


def test_team_lineup_summary_keeps_own_and_teammate_dpm_on_one_sample() -> None:
    exposures = pd.DataFrame(
        [
            {
                "game_date": "2025-04-01",
                "team_id": 100,
                "team_abbr": "AAA",
                "P1": 1,
                "P2": 2,
                "P3": 3,
                "P4": 4,
                "P5": 5,
                "exposures": 2.0,
            }
        ]
    )
    histories = {
        player_id: PlayerRatingHistory(
            dates=np.array([np.datetime64("2025-03-01")]),
            seasons=np.array([2025], dtype=np.int16),
            values=np.array([[float(player_id), player_id * 0.6, player_id * 0.4]]),
        )
        for player_id in range(1, 6)
    }

    player, pair = summarize_team_lineup_exposures(exposures, histories, season=2025)

    anchor = player.loc[player["player_id"] == 1].iloc[0]
    assert anchor["team_id"] == 100
    assert anchor["total_exposures"] == 2
    assert anchor["rated_exposures"] == 2
    assert anchor["own_dpm"] == 1
    assert anchor["help_dpm"] == 14
    anchor_pairs = pair.loc[pair["player_id"] == 1]
    assert anchor_pairs["contribution_dpm"].sum() == 14


def test_teammate_attributions_close_to_the_total_title_probability_change() -> None:
    teammates = pd.DataFrame(
        [
            {"teammate_id": 10, "contribution_dpm": 1.0},
            {"teammate_id": 20, "contribution_dpm": 2.0},
        ]
    )

    attributed = shapley_teammate_attribution(
        teammates,
        probability_for_help=lambda help_dpm: 0.10 + 0.05 * help_dpm,
        permutations=64,
        seed=7,
    )

    values = attributed.set_index("teammate_id")["championships_added"]
    assert abs(values.sum() - 0.15) < 1e-12
    assert values[20] > values[10] > 0


def test_processed_possession_becomes_two_team_identified_exposures() -> None:
    possessions = pd.DataFrame(
        [
            {
                "game_id": "0000000001",
                "game_date": "2025-01-01",
                **{f"O{slot}": slot for slot in range(1, 6)},
                **{f"D{slot}": slot + 5 for slot in range(1, 6)},
            }
        ]
    )
    player_game_teams = pd.DataFrame(
        [
            {
                "game_id": "0000000001",
                "player_id": player_id,
                "team_id": 100 if player_id <= 5 else 200,
                "team_abbr": "AAA" if player_id <= 5 else "BBB",
            }
            for player_id in range(1, 11)
        ]
    )

    exposures = collapse_processed_team_exposures(possessions, player_game_teams)

    assert exposures[["team_id", "team_abbr", "exposures"]].to_dict("records") == [
        {"team_id": 100, "team_abbr": "AAA", "exposures": 1.0},
        {"team_id": 200, "team_abbr": "BBB", "exposures": 1.0},
    ]
    assert exposures.loc[0, ["P1", "P2", "P3", "P4", "P5"]].tolist() == [1, 2, 3, 4, 5]
    assert exposures.loc[1, ["P1", "P2", "P3", "P4", "P5"]].tolist() == [6, 7, 8, 9, 10]


def test_raw_game_context_recovers_sides_rosters_and_final_score() -> None:
    raw = pd.DataFrame(
        [
            {
                "game_id": "0000000001",
                "event_num": 1,
                "home_description": "Home basket",
                "visitor_description": None,
                "player1_team_id": 100,
                "player1_team_abbreviation": "AAA",
                "home_score": 2,
                "away_score": 0,
                **{f"home_player{slot}": slot for slot in range(1, 6)},
                **{f"away_player{slot}": slot + 5 for slot in range(1, 6)},
            },
            {
                "game_id": "0000000001",
                "event_num": 2,
                "home_description": None,
                "visitor_description": "Away basket",
                "player1_team_id": 200,
                "player1_team_abbreviation": "BBB",
                "home_score": 2,
                "away_score": 3,
                **{f"home_player{slot}": slot for slot in range(1, 6)},
                **{f"away_player{slot}": slot + 5 for slot in range(1, 6)},
            },
            {
                "game_id": "0000000001",
                "event_num": 3,
                "home_description": None,
                "visitor_description": None,
                "player1_team_id": None,
                "player1_team_abbreviation": None,
                "home_score": 0,
                "away_score": 0,
                **{f"home_player{slot}": slot for slot in range(1, 6)},
                **{f"away_player{slot}": slot + 5 for slot in range(1, 6)},
            },
        ]
    )
    dates = pd.DataFrame([{"game_id": "0000000001", "game_date": "2025-01-01"}])

    games, assignments = build_game_context(raw, dates)

    assert games.iloc[0][
        ["home_team_id", "home_team_abbr", "away_team_id", "away_team_abbr", "home_score", "away_score"]
    ].to_dict() == {
        "home_team_id": 100,
        "home_team_abbr": "AAA",
        "away_team_id": 200,
        "away_team_abbr": "BBB",
        "home_score": 2.0,
        "away_score": 3.0,
    }
    assert len(assignments) == 10
    assert assignments.loc[assignments["player_id"] == 1, "team_id"].item() == 100
    assert assignments.loc[assignments["player_id"] == 10, "team_id"].item() == 200


def test_raw_game_context_uses_majority_side_for_one_corrupted_lineup_slot() -> None:
    rows = []
    for event_num in range(1, 4):
        row = {
            "game_id": "0000000002",
            "event_num": event_num,
            "home_description": "Home action" if event_num < 3 else None,
            "visitor_description": "Away action" if event_num == 3 else None,
            "player1_team_id": 100 if event_num < 3 else 200,
            "player1_team_abbreviation": "AAA" if event_num < 3 else "BBB",
            "home_score": 2,
            "away_score": 0,
            **{f"home_player{slot}": slot for slot in range(1, 6)},
            **{f"away_player{slot}": slot + 5 for slot in range(1, 6)},
        }
        rows.append(row)
    rows[-1]["away_player5"] = 1  # One bad raw slot; player 1 is otherwise home.
    dates = pd.DataFrame([{"game_id": "0000000002", "game_date": "2025-01-02"}])

    _, assignments = build_game_context(pd.DataFrame(rows), dates)

    player_one = assignments.loc[assignments["player_id"] == 1]
    assert len(player_one) == 1
    assert player_one["team_id"].item() == 100


def test_raw_game_context_uses_actor_roster_side_when_description_side_is_misleading() -> None:
    rows = []
    actor_rows = [
        (6, 200, "BBB", "Home text naming an away blocker", None),
        (6, 200, "BBB", "Another misleading home description", None),
        (1, 100, "AAA", "Home action", None),
        (6, 200, "BBB", None, "Away action"),
    ]
    for event_num, (player_id, team_id, abbr, home_desc, away_desc) in enumerate(actor_rows, 1):
        rows.append(
            {
                "game_id": "0000000003",
                "event_num": event_num,
                "home_description": home_desc,
                "visitor_description": away_desc,
                "player1_id": player_id,
                "player1_team_id": team_id,
                "player1_team_abbreviation": abbr,
                "home_score": 2,
                "away_score": 0,
                **{f"home_player{slot}": slot for slot in range(1, 6)},
                **{f"away_player{slot}": slot + 5 for slot in range(1, 6)},
            }
        )
    dates = pd.DataFrame([{"game_id": "0000000003", "game_date": "2025-01-03"}])

    games, _ = build_game_context(pd.DataFrame(rows), dates)

    assert games.loc[0, "home_team_id"] == 100
    assert games.loc[0, "away_team_id"] == 200


def test_retrospective_team_summary_uses_same_season_ratings_and_all_team_minutes() -> None:
    exposures = pd.DataFrame(
        [
            {
                "game_date": "2025-01-01",
                "team_id": 100,
                "team_abbr": "AAA",
                "P1": 1,
                "P2": 2,
                "P3": 3,
                "P4": 4,
                "P5": 5,
                "exposures": 2.0,
            },
            {
                "game_date": "2025-01-02",
                "team_id": 100,
                "team_abbr": "AAA",
                "P1": 2,
                "P2": 6,
                "P3": 7,
                "P4": 8,
                "P5": 9,
                "exposures": 3.0,
            },
        ]
    )
    ratings = pd.DataFrame(
        [
            {
                "nba_id": player_id,
                "player_name": f"P{player_id}",
                "season": 2025,
                "date": "2025-06-20",
                "dpm": float(player_id),
                "o_dpm": player_id * 0.6,
                "d_dpm": player_id * 0.4,
            }
            for player_id in range(1, 10)
        ]
    )

    teams, roster = summarize_retrospective_team_season(exposures, ratings, season=2025)

    team = teams.iloc[0]
    assert team["total_exposures"] == 5
    assert team["rating_coverage_pct"] == 100
    assert abs(team["team_dpm"] - 25.2) < 1e-12
    assert abs(roster["contribution_dpm"].sum() - team["team_dpm"]) < 1e-12
    assert roster.set_index("player_id").loc[1, "contribution_dpm"] == 0.4
    # Players 6-9 never shared the floor with player 1, but their team minutes
    # still contribute to the team's championship environment.
    assert roster.loc[roster["player_id"].isin([6, 7, 8, 9]), "contribution_dpm"].sum() > 0


def test_shared_floor_and_off_floor_contributions_close_to_full_team_strength() -> None:
    exposures = pd.DataFrame(
        [
            {
                "game_date": "2025-01-01",
                "team_id": 100,
                "team_abbr": "AAA",
                "P1": 1,
                "P2": 2,
                "P3": 3,
                "P4": 4,
                "P5": 5,
                "exposures": 2.0,
            },
            {
                "game_date": "2025-01-02",
                "team_id": 100,
                "team_abbr": "AAA",
                "P1": 2,
                "P2": 6,
                "P3": 7,
                "P4": 8,
                "P5": 9,
                "exposures": 3.0,
            },
        ]
    )
    ratings = pd.DataFrame(
        [
            {
                "nba_id": player_id,
                "player_name": f"P{player_id}",
                "season": 2025,
                "date": "2025-06-20",
                "dpm": float(player_id),
                "o_dpm": player_id * 0.6,
                "d_dpm": player_id * 0.4,
            }
            for player_id in range(1, 10)
        ]
    )
    teams, roster = summarize_retrospective_team_season(exposures, ratings, season=2025)

    pairs = summarize_team_pair_exposures(exposures, ratings, season=2025)
    players, _ = build_player_title_context(
        teams.assign(champion_team_id=100, won_championship=1),
        roster,
        pairs,
        min_exposures=1,
    )

    focal = players.loc[players["player_id"].eq(1)].iloc[0]
    assert abs(focal["own_team_contribution_dpm"] - 0.4) < 1e-12
    assert abs(focal["on_court_help_dpm"] - 5.6) < 1e-12
    assert abs(focal["bench_help_dpm"] - 19.2) < 1e-12
    assert abs(
        focal["own_team_contribution_dpm"]
        + focal["on_court_help_dpm"]
        + focal["bench_help_dpm"]
        - focal["team_dpm"]
    ) < 1e-12


def test_player_title_context_excludes_focal_contribution_from_teammate_help() -> None:
    teams = pd.DataFrame(
        [
            {
                "season": 2025,
                "team_id": 100,
                "team_abbr": "AAA",
                "team_dpm": 3.0,
                "team_o_dpm": 2.0,
                "team_d_dpm": 1.0,
                "total_exposures": 100.0,
                "rating_coverage_pct": 100.0,
                "champion_team_id": 100,
                "won_championship": 1,
            }
        ]
    )
    roster = pd.DataFrame(
        [
            {
                "season": 2025,
                "team_id": 100,
                "team_abbr": "AAA",
                "player_id": 1,
                "player_name": "Focal",
                "first_game_date": "2024-10-20",
                "last_game_date": "2025-04-10",
                "player_exposures": 80.0,
                "rated_exposures": 80.0,
                "player_dpm": 2.0,
                "player_o_dpm": 1.5,
                "player_d_dpm": 0.5,
                "contribution_dpm": 1.6,
                "contribution_o_dpm": 1.2,
                "contribution_d_dpm": 0.4,
            },
            {
                "season": 2025,
                "team_id": 100,
                "team_abbr": "AAA",
                "player_id": 2,
                "player_name": "Starter",
                "first_game_date": "2024-10-20",
                "last_game_date": "2025-04-10",
                "player_exposures": 70.0,
                "rated_exposures": 70.0,
                "player_dpm": 1.0,
                "player_o_dpm": 0.5,
                "player_d_dpm": 0.5,
                "contribution_dpm": 0.7,
                "contribution_o_dpm": 0.35,
                "contribution_d_dpm": 0.35,
            },
            {
                "season": 2025,
                "team_id": 100,
                "team_abbr": "AAA",
                "player_id": 3,
                "player_name": "Bench",
                "first_game_date": "2024-10-20",
                "last_game_date": "2025-04-10",
                "player_exposures": 35.0,
                "rated_exposures": 35.0,
                "player_dpm": 2.0,
                "player_o_dpm": 1.0,
                "player_d_dpm": 1.0,
                "contribution_dpm": 0.7,
                "contribution_o_dpm": 0.45,
                "contribution_d_dpm": 0.25,
            },
        ]
    )

    pairs = pd.DataFrame(
        [
            {
                "season": 2025,
                "team_id": 100,
                "player_id": 1,
                "teammate_id": 2,
                "shared_exposures": 60.0,
                "on_court_contribution_dpm": 0.6,
                "on_court_contribution_o_dpm": 0.3,
                "on_court_contribution_d_dpm": 0.3,
            },
            {
                "season": 2025,
                "team_id": 100,
                "player_id": 1,
                "teammate_id": 3,
                "shared_exposures": 20.0,
                "on_court_contribution_dpm": 0.4,
                "on_court_contribution_o_dpm": 0.2,
                "on_court_contribution_d_dpm": 0.2,
            },
        ]
    )
    players, teammates = build_player_title_context(
        teams, roster, pairs, min_exposures=50
    )

    focal = players.loc[players["player_id"].eq(1)].iloc[0]
    assert abs(focal["help_dpm"] - 1.4) < 1e-12
    focal_teammates = teammates.loc[teammates["player_id"].eq(1)]
    assert set(focal_teammates["teammate_id"]) == {2, 3}
    assert abs(focal_teammates["contribution_dpm"].sum() - focal["help_dpm"]) < 1e-12
    assert abs(focal["on_court_help_dpm"] - 1.0) < 1e-12
    assert abs(focal["bench_help_dpm"] - 0.4) < 1e-12


def test_team_title_model_probabilities_sum_to_one_within_every_season() -> None:
    rows = []
    for season in range(2000, 2008):
        for team in range(4):
            rows.append(
                {
                    "season": season,
                    "team_id": team,
                    "team_dpm": float(6 - 2 * team),
                    "team_o_dpm": float(4 - team),
                    "team_d_dpm": float(2 - team),
                    "won_championship": int(team == 0),
                }
            )
    sample = pd.DataFrame(rows)
    model = fit_team_title_model(sample, spec="overall", c_value=1.0)

    probabilities = model.predict_probabilities(sample)
    probability_sums = pd.DataFrame(
        {"season": sample["season"], "probability": probabilities}
    ).groupby("season")["probability"].sum()

    assert np.allclose(probability_sums.to_numpy(), 1.0, atol=1e-12)
    first_season = probabilities[sample["season"].eq(2000)]
    assert np.all(np.diff(first_season) < 0)


def test_leave_one_season_out_team_probabilities_close_for_every_held_out_year() -> None:
    rows = []
    for season in range(2000, 2008):
        for team in range(4):
            rows.append(
                {
                    "season": season,
                    "team_id": team,
                    "team_dpm": float(6 - 2 * team + (season % 2) * 0.1),
                    "team_o_dpm": float(4 - team),
                    "team_d_dpm": float(2 - team),
                    "won_championship": int(team == 0),
                }
            )

    predictions, models = cross_validated_team_predictions(
        pd.DataFrame(rows), spec="overall", c_value=1.0
    )

    assert set(models) == set(range(2000, 2008))
    probability_sums = predictions.groupby("season")["predicted_probability"].sum()
    assert np.allclose(probability_sums.to_numpy(), 1.0, atol=1e-12)


def test_one_standard_error_rule_prefers_the_stronger_regularized_model() -> None:
    diagnostics = pd.DataFrame(
        [
            {
                "spec": "sides",
                "c_value": 5.0,
                "champion_log_loss": 1.668,
                "champion_log_loss_se": 0.10,
                "season_brier": 0.73,
                "top_one_accuracy": 0.25,
                "max_annual_probability_sum_gap": 0.0,
            },
            {
                "spec": "sides",
                "c_value": 0.2,
                "champion_log_loss": 1.74,
                "champion_log_loss_se": 0.09,
                "season_brier": 0.75,
                "top_one_accuracy": 0.25,
                "max_annual_probability_sum_gap": 0.0,
            },
            {
                "spec": "sides",
                "c_value": 0.05,
                "champion_log_loss": 1.90,
                "champion_log_loss_se": 0.08,
                "season_brier": 0.80,
                "top_one_accuracy": 0.25,
                "max_annual_probability_sum_gap": 0.0,
            },
        ]
    )

    selected = select_team_model_candidate(diagnostics)

    assert selected["c_value"] == 0.2
    assert selected["champion_log_loss"] == 1.74


def test_team_probability_attribution_holds_focal_fixed_and_reconciles_teammates() -> None:
    team_rows = []
    for season in range(2000, 2008):
        for team in range(3):
            team_rows.append(
                {
                    "season": season,
                    "team_id": team,
                    "team_abbr": f"T{team}",
                    "team_dpm": float(3 - team),
                    "team_o_dpm": float(2 - team * 0.5),
                    "team_d_dpm": float(1 - team * 0.5),
                    "total_exposures": 60.0,
                    "rating_coverage_pct": 100.0,
                    "champion_team_id": 0,
                    "won_championship": int(team == 0),
                }
            )
    teams = attach_neutral_playoff_baseline(pd.DataFrame(team_rows))
    roster = pd.DataFrame(
        [
            {
                "season": 2007,
                "team_id": 0,
                "team_abbr": "T0",
                "player_id": player_id,
                "player_name": f"P{player_id}",
                "first_game_date": "2006-11-01",
                "last_game_date": "2007-04-15",
                "player_exposures": exposure,
                "rated_exposures": exposure,
                "player_dpm": dpm,
                "player_o_dpm": dpm * 2 / 3,
                "player_d_dpm": dpm / 3,
                "contribution_dpm": contribution,
                "contribution_o_dpm": contribution * 2 / 3,
                "contribution_d_dpm": contribution / 3,
                "weighted_dpm": exposure * dpm,
                "weighted_o_dpm": exposure * dpm * 2 / 3,
                "weighted_d_dpm": exposure * dpm / 3,
            }
            for player_id, exposure, dpm, contribution in (
                (1, 100.0, 1.0, 1.0),
                (2, 100.0, 1.2, 1.2),
                (3, 100.0, 0.8, 0.8),
            )
        ]
    )
    pair_rows = []
    for focal_id in (1, 2, 3):
        for teammate_id in (1, 2, 3):
            if focal_id == teammate_id:
                continue
            teammate = roster.loc[roster["player_id"].eq(teammate_id)].iloc[0]
            pair_rows.append(
                {
                    "season": 2007,
                    "team_id": 0,
                    "player_id": focal_id,
                    "teammate_id": teammate_id,
                    "shared_exposures": 50.0,
                    "on_court_contribution_dpm": teammate["contribution_dpm"] / 2,
                    "on_court_contribution_o_dpm": teammate["contribution_o_dpm"] / 2,
                    "on_court_contribution_d_dpm": teammate["contribution_d_dpm"] / 2,
                }
            )
    players, teammates = build_player_title_context(
        teams, roster, pd.DataFrame(pair_rows), min_exposures=50
    )
    _, models = cross_validated_team_predictions(teams, spec="overall", c_value=1.0)

    seasons, attributed, report = attribute_team_probabilities(
        players, teammates, teams, models, permutations=32
    )

    assert seasons["title_probability"].nunique() == 1
    assert np.allclose(
        seasons["neutral_team_title_probability"]
        + seasons["normalized_self_title_probability_added"]
        + seasons["on_court_teammate_title_probability_added"]
        + seasons["bench_title_probability_added"],
        seasons["title_probability"],
        atol=1e-12,
    )
    normalized_self = seasons.set_index("player_id")[
        "normalized_self_title_probability_added"
    ]
    assert normalized_self[2] > normalized_self[1] > normalized_self[3] > 0
    assert (
        seasons["self_replacement_title_probability_added"] > 0
    ).all()
    assert (
        seasons["self_context_title_probability"]
        == seasons["neutral_teammate_title_probability"]
    ).all()
    assert (
        seasons["self_context_title_probability"]
        > seasons["neutral_team_title_probability"]
    ).all()
    assert seasons["neutral_field_neutral_title_probability"].nunique() == 1
    assert (
        seasons["neutral_field_hard_carry_title_probability"]
        > seasons["neutral_field_neutral_title_probability"]
    ).all()
    assert np.allclose(seasons["era_neutral_title_probability"], 1 / 16)
    assert (
        seasons["era_neutral_hard_carry_title_probability"]
        > seasons["era_neutral_title_probability"]
    ).all()
    for player_id, group in attributed.groupby("player_id"):
        target = seasons.loc[
            seasons["player_id"].eq(player_id), "teammate_help_championships"
        ].item()
        assert abs(group["championships_added"].sum() - target) < 1e-12
    assert report["max_actual_annual_probability_sum_gap"] < 1e-12
