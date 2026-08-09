from __future__ import annotations

import numpy as np
import pandas as pd

from nba_pipeline.scripts.build_championship_help import (
    attribute_team_probabilities,
    build_player_title_context,
    build_game_context,
    collapse_processed_team_exposures,
    cross_validated_team_predictions,
    extract_champion_from_games,
    fit_team_title_model,
    select_team_model_candidate,
    select_final_team_segments,
    shapley_teammate_attribution,
    summarize_retrospective_team_season,
    summarize_team_pair_exposures,
    summarize_team_lineup_exposures,
)
from nba_pipeline.scripts.build_darko_teammate_help import PlayerRatingHistory


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
                    "total_exposures": 100.0,
                    "rating_coverage_pct": 100.0,
                    "champion_team_id": 0,
                    "won_championship": int(team == 0),
                }
            )
    teams = pd.DataFrame(team_rows)
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
        seasons["self_context_title_probability"]
        + seasons["on_court_teammate_title_probability_added"]
        + seasons["bench_title_probability_added"],
        seasons["title_probability"],
        atol=1e-12,
    )
    for player_id, group in attributed.groupby("player_id"):
        target = seasons.loc[
            seasons["player_id"].eq(player_id), "teammate_help_championships"
        ].item()
        assert abs(group["championships_added"].sum() - target) < 1e-12
    assert report["max_actual_annual_probability_sum_gap"] < 1e-12
