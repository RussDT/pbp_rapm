from __future__ import annotations

import numpy as np
import pandas as pd

from nba_pipeline.scripts.playoff_title_simulation import (
    PlayoffGameModel,
    add_salary_peer_title_lift,
    best_of_seven_win_probability,
    bracket_title_probabilities,
    build_bracket_nodes,
    build_playoff_seed_fields,
    build_playoff_series,
    fit_playoff_game_model,
    regular_season_seed_probabilities,
    seeded_target_title_probability_curve,
    select_salary_peer_slots,
    target_title_probability_curve,
)


def test_regular_season_strength_changes_playoff_odds_and_expected_seed() -> None:
    teams = pd.DataFrame(
        [
            {
                "season": 2025,
                "team_id": team_id,
                "team_abbr": f"T{team_id}",
                "conference": "East" if team_id <= 10 else "West",
                "team_dpm": 0.0,
            }
            for team_id in range(1, 21)
        ]
    )
    games = pd.DataFrame(
        [
            {
                "season": 2025,
                "home_team_id": home,
                "away_team_id": away,
            }
            for home in range(1, 21)
            for away in range(1, 21)
            if home != away
        ]
    )

    result = regular_season_seed_probabilities(
        games,
        teams,
        PlayoffGameModel(home_advantage=0.0, epm_coefficient=0.30),
        rating_shift_grid=np.array([-4.0, 0.0, 4.0]),
        simulations=4096,
        seed=17,
    )
    target = result[result["team_id"].eq(1)].set_index("rating_shift")

    assert target.loc[4.0, "playoff_probability"] > target.loc[-4.0, "playoff_probability"]
    assert target.loc[4.0, "conditional_mean_seed"] < target.loc[-4.0, "conditional_mean_seed"]


def test_seeded_title_curve_credits_home_court_to_the_higher_seed() -> None:
    field = pd.DataFrame(
        [
            {
                "team_id": team_id,
                "conference": "East" if team_id <= 8 else "West",
                "seed": team_id if team_id <= 8 else team_id - 8,
            }
            for team_id in range(1, 17)
        ]
    )
    ratings = pd.Series(0.0, index=field["team_id"])
    model = PlayoffGameModel(home_advantage=0.40, epm_coefficient=0.12)

    one_seed = seeded_target_title_probability_curve(
        field, ratings, target_team_id=1, target_ratings=0.0, model=model
    )
    eight_seed = seeded_target_title_probability_curve(
        field, ratings, target_team_id=8, target_ratings=0.0, model=model
    )

    assert one_seed > eight_seed


def test_playoff_seed_field_uses_regular_season_records_within_conference() -> None:
    teams = pd.DataFrame(
        [
            {
                "season": 2025,
                "team_id": team_id,
                "team_abbr": (
                    ["ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DET", "IND"]
                    + ["DAL", "DEN", "GSW", "HOU", "LAC", "LAL", "MEM", "MIN"]
                )[team_id - 1],
                "team_dpm": float(17 - team_id),
            }
            for team_id in range(1, 17)
        ]
    )
    game_rows = []
    for team_id in range(1, 17):
        wins = 60 - team_id
        for game_index in range(82):
            game_rows.append(
                {
                    "season": 2025,
                    "home_team_id": team_id,
                    "away_team_id": 99,
                    "home_score": 100.0 if game_index < wins else 90.0,
                    "away_score": 90.0 if game_index < wins else 100.0,
                }
            )

    result = build_playoff_seed_fields(pd.DataFrame(game_rows), teams)

    east = result[result["conference"].eq("East")].sort_values("seed")
    west = result[result["conference"].eq("West")].sort_values("seed")
    assert east["team_id"].tolist() == list(range(1, 9))
    assert west["team_id"].tolist() == list(range(9, 17))


def _four_round_series(season: int = 2025) -> pd.DataFrame:
    records: list[dict[str, int | str]] = []
    prior_winners: list[int] = []
    for index in range(8):
        team_a = 2 * index + 1
        team_b = 2 * index + 2
        records.append(
            {
                "season": season,
                "round": 1,
                "series_id": f"r1_{index}",
                "team_a_id": team_a,
                "team_b_id": team_b,
                "winner_team_id": team_a,
            }
        )
        prior_winners.append(team_a)
    for round_number, series_count in ((2, 4), (3, 2), (4, 1)):
        next_winners: list[int] = []
        for index in range(series_count):
            team_a = prior_winners[2 * index]
            team_b = prior_winners[2 * index + 1]
            records.append(
                {
                    "season": season,
                    "round": round_number,
                    "series_id": f"r{round_number}_{index}",
                    "team_a_id": team_a,
                    "team_b_id": team_b,
                    "winner_team_id": team_a,
                }
            )
            next_winners.append(team_a)
        prior_winners = next_winners
    return pd.DataFrame(records)


def test_best_of_seven_probability_is_symmetric_and_monotone() -> None:
    model = PlayoffGameModel(home_advantage=0.20, epm_coefficient=0.12)
    differences = np.array([-4.0, 0.0, 4.0])

    probability = best_of_seven_win_probability(differences, model)
    reverse = best_of_seven_win_probability(-differences, model)

    assert probability[0] < probability[1] < probability[2]
    assert abs(probability[1] - 0.5) < 1e-12
    assert np.allclose(probability + reverse, 1.0)


def test_salary_peer_slot_uses_closest_cap_share_and_the_peers_role() -> None:
    roster = pd.DataFrame(
        [
            {
                "season": 2024,
                "team_id": 1,
                "player_id": 10,
                "salary_share": 0.24,
                "own_dpm": 1.0,
                "own_team_contribution_dpm": 0.40,
                "playoff_workload_share": 0.40,
            },
            {
                "season": 2024,
                "team_id": 1,
                "player_id": 11,
                "salary_share": 0.26,
                "own_dpm": 2.0,
                "own_team_contribution_dpm": 1.40,
                "playoff_workload_share": 0.70,
            },
            {
                "season": 2024,
                "team_id": 2,
                "player_id": 20,
                "salary_share": 0.18,
                "own_dpm": -1.0,
                "own_team_contribution_dpm": -0.60,
                "playoff_workload_share": 0.60,
            },
            {
                "season": 2025,
                "team_id": 3,
                "player_id": 30,
                "salary_share": 0.25,
                "own_dpm": 3.0,
                "own_team_contribution_dpm": 2.40,
                "playoff_workload_share": 0.80,
            },
        ]
    )

    result = select_salary_peer_slots(
        focal_salary_share=0.25,
        focal_season=2025,
        roster=roster,
    )

    assert result[["season", "team_id", "peer_player_id"]].to_dict("records") == [
        {"season": 2024, "team_id": 1, "peer_player_id": 11},
        {"season": 2024, "team_id": 2, "peer_player_id": 20},
    ]
    assert result.loc[result["team_id"].eq(1), "peer_workload_share"].item() == 0.70
    assert result.loc[result["team_id"].eq(1), "peer_contribution_dpm"].item() == 1.40


def test_playoff_games_reconstruct_series_winner_and_round() -> None:
    games = pd.DataFrame(
        [
            {
                "season": 2025,
                "game_id": f"004240010{game}",
                "home_team_id": 1 if game % 2 else 2,
                "away_team_id": 2 if game % 2 else 1,
                "home_score": 110 if game % 2 else 90,
                "away_score": 90 if game % 2 else 110,
            }
            for game in range(1, 5)
        ]
    )

    result = build_playoff_series(games)

    assert result[["season", "round", "series_id", "winner_team_id", "games"]].to_dict(
        "records"
    ) == [
        {
            "season": 2025,
            "round": 1,
            "series_id": "4240010",
            "winner_team_id": 1,
            "games": 4,
        }
    ]
    assert {result.loc[0, "team_a_id"], result.loc[0, "team_b_id"]} == {1, 2}


def test_four_round_bracket_gives_equal_teams_one_in_sixteen_odds() -> None:
    series = _four_round_series()
    nodes = build_bracket_nodes(series)
    ratings = pd.Series(0.0, index=np.arange(1, 17))
    model = PlayoffGameModel(home_advantage=0.20, epm_coefficient=0.12)

    equal = bracket_title_probabilities(nodes, ratings, model)
    stronger_ratings = ratings.copy()
    stronger_ratings.loc[1] = 5.0
    stronger = bracket_title_probabilities(nodes, stronger_ratings, model)

    assert np.allclose(equal.sort_index().to_numpy(), np.repeat(1 / 16, 16))
    assert abs(stronger.sum() - 1.0) < 1e-12
    assert stronger.loc[1] > 1 / 16
    target_curve = target_title_probability_curve(
        nodes,
        ratings,
        target_team_id=1,
        target_ratings=np.array([0.0, 5.0]),
        model=model,
    )
    assert np.allclose(target_curve, [equal.loc[1], stronger.loc[1]])


def test_playoff_game_model_learns_that_stronger_teams_win_more() -> None:
    rows = pd.DataFrame(
        {
            "rating_difference": [-6, -5, -4, -3, 3, 4, 5, 6] * 8,
            "home_win": [0, 0, 0, 0, 1, 1, 1, 1] * 8,
        }
    )

    model = fit_playoff_game_model(rows, c_value=1.0)

    assert model.epm_coefficient > 0
    assert best_of_seven_win_probability(4.0, model) > 0.5


def test_salary_peer_title_lift_swaps_into_every_historical_playoff_team() -> None:
    nodes = build_bracket_nodes(_four_round_series())
    teams = pd.DataFrame(
        [
            {"season": 2025, "team_id": team_id, "team_dpm": 0.0}
            for team_id in range(1, 17)
        ]
    )
    peers = pd.DataFrame(
        [
            {
                "season": 2025,
                "team_id": team_id,
                "player_id": 100 + team_id,
                "salary_share": 0.20,
                "own_dpm": 0.0,
                "own_team_contribution_dpm": 0.0,
                "playoff_workload_share": 0.75,
            }
            for team_id in range(1, 17)
        ]
    )
    focal = pd.DataFrame(
        [
            {
                "season": 2026,
                "player_id": 999,
                "own_dpm": 4.0,
                "salary_share": 0.20,
            }
        ]
    )
    model = PlayoffGameModel(home_advantage=0.20, epm_coefficient=0.12)

    result = add_salary_peer_title_lift(focal, peers, teams, nodes, model)

    assert result.loc[0, "salary_peer_placement_count"] == 16
    assert abs(result.loc[0, "salary_peer_baseline_title_probability"] - 1 / 16) < 1e-12
    assert result.loc[0, "salary_peer_actual_title_probability"] > 1 / 16
    assert result.loc[0, "title_lift_over_contract"] > 0


def test_era_local_salary_peer_lift_uses_only_the_four_neighboring_seasons() -> None:
    series_2025 = _four_round_series()
    series_2019 = series_2025.assign(season=2019)
    nodes = build_bracket_nodes(pd.concat([series_2019, series_2025], ignore_index=True))
    teams = pd.DataFrame(
        [
            {"season": season, "team_id": team_id, "team_dpm": 0.0}
            for season in (2019, 2025)
            for team_id in range(1, 17)
        ]
    )
    peers = pd.DataFrame(
        [
            {
                "season": season,
                "team_id": team_id,
                "player_id": season * 100 + team_id,
                "salary_share": 0.20,
                "own_dpm": 0.0,
                "own_team_contribution_dpm": 0.0,
                "playoff_workload_share": 0.75,
            }
            for season in (2019, 2025)
            for team_id in range(1, 17)
        ]
    )
    focal = pd.DataFrame(
        [{"season": 2026, "player_id": 999, "own_dpm": 4.0, "salary_share": 0.20}]
    )

    result = add_salary_peer_title_lift(
        focal,
        peers,
        teams,
        nodes,
        PlayoffGameModel(home_advantage=0.20, epm_coefficient=0.12),
    )

    assert result.loc[0, "salary_peer_placement_count"] == 32
    assert result.loc[0, "era_local_salary_peer_placement_count"] == 16
    assert np.isclose(
        result.loc[0, "title_lift_over_contract_v2"],
        result.loc[0, "title_lift_over_contract"],
    )


def test_era_local_salary_peer_lift_debits_expensive_players_and_credits_cheap_ones() -> None:
    nodes = build_bracket_nodes(_four_round_series())
    teams = pd.DataFrame(
        [
            {"season": 2025, "team_id": team_id, "team_dpm": 0.0}
            for team_id in range(1, 17)
        ]
    )
    peers = pd.DataFrame(
        [
            {
                "season": 2025,
                "team_id": team_id,
                "player_id": 100 + team_id,
                "salary_share": 0.20,
                "own_dpm": 0.0,
                "own_team_contribution_dpm": 0.0,
                "playoff_workload_share": 0.75,
            }
            for team_id in range(1, 17)
        ]
    )
    focal = pd.DataFrame(
        [
            {"season": 2026, "player_id": 1, "own_dpm": 0.0, "salary_share": 0.27},
            {"season": 2026, "player_id": 2, "own_dpm": 0.0, "salary_share": 0.13},
        ]
    )

    result = add_salary_peer_title_lift(
        focal,
        peers,
        teams,
        nodes,
        PlayoffGameModel(home_advantage=0.20, epm_coefficient=0.12),
    ).set_index("player_id")

    assert np.isclose(result.loc[1, "era_local_budget_mean_playoff_debit"], 0.546)
    assert np.isclose(result.loc[2, "era_local_budget_mean_playoff_debit"], -0.546)
    assert result.loc[1, "title_lift_over_contract_v2"] < 0
    assert result.loc[2, "title_lift_over_contract_v2"] > 0
    assert np.isclose(result.loc[1, "title_lift_over_contract"], 0.0)
    assert np.isclose(result.loc[2, "title_lift_over_contract"], 0.0)


def test_salary_peer_title_lift_adds_regular_season_seeding_credit() -> None:
    nodes = build_bracket_nodes(_four_round_series())
    teams = pd.DataFrame(
        [
            {"season": 2025, "team_id": team_id, "team_dpm": 0.0}
            for team_id in range(1, 17)
        ]
    )
    peers = pd.DataFrame(
        [
            {
                "season": 2025,
                "team_id": team_id,
                "player_id": 100 + team_id,
                "salary_share": 0.20,
                "own_dpm": 0.0,
                "own_team_contribution_dpm": 0.0,
                "playoff_workload_share": 0.75,
                "regular_season_workload_share": 0.75,
            }
            for team_id in range(1, 17)
        ]
    )
    focal = pd.DataFrame(
        [{"season": 2026, "player_id": 999, "own_dpm": 4.0, "salary_share": 0.20}]
    )
    seed_rows = []
    for team_id in range(1, 17):
        for shift, seed_number, seed_probability in [
            (-4.0, None, 0.0),
            (0.0, 8, 0.5),
            (4.0, 1, 1.0),
        ]:
            row = {
                "season": 2025,
                "team_id": team_id,
                "rating_shift": shift,
                "playoff_probability": seed_probability,
                "conditional_mean_seed": float(seed_number) if seed_number else np.nan,
            }
            for seed in range(1, 9):
                row[f"seed_{seed}_probability"] = (
                    seed_probability if seed == seed_number else 0.0
                )
            seed_rows.append(row)
    seed_probabilities = pd.DataFrame(seed_rows)
    seed_field = pd.DataFrame(
        [
            {
                "season": 2025,
                "team_id": team_id,
                "conference": "East" if team_id <= 8 else "West",
                "seed": team_id if team_id <= 8 else team_id - 8,
            }
            for team_id in range(1, 17)
        ]
    )
    model = PlayoffGameModel(home_advantage=0.40, epm_coefficient=0.12)

    result = add_salary_peer_title_lift(
        focal,
        peers,
        teams,
        nodes,
        model,
        regular_seed_probabilities=seed_probabilities,
        playoff_seed_fields=seed_field,
    )

    assert result.loc[0, "salary_peer_actual_playoff_probability"] > result.loc[
        0, "salary_peer_baseline_playoff_probability"
    ]
    assert result.loc[0, "salary_peer_actual_mean_seed"] < result.loc[
        0, "salary_peer_baseline_mean_seed"
    ]
    assert result.loc[0, "title_lift_over_contract"] > result.loc[
        0, "salary_peer_fixed_bracket_title_lift"
    ]
    assert np.isclose(
        result.loc[0, "title_lift_over_contract"],
        result.loc[0, "salary_peer_fixed_bracket_title_lift"]
        + result.loc[0, "salary_peer_seeding_title_lift"],
    )


def test_playoff_adjusted_salary_peer_lift_changes_series_but_not_regular_season_seed() -> None:
    nodes = build_bracket_nodes(_four_round_series())
    teams = pd.DataFrame(
        [
            {
                "season": 2025,
                "team_id": team_id,
                "team_dpm": 0.0,
                "playoff_adjusted_team_dpm": 0.0,
            }
            for team_id in range(1, 17)
        ]
    )
    peers = pd.DataFrame(
        [
            {
                "season": 2025,
                "team_id": team_id,
                "player_id": 100 + team_id,
                "salary_share": 0.20,
                "own_dpm": 0.0,
                "playoff_adjusted_dpm": 0.0,
                "own_team_contribution_dpm": 0.0,
                "playoff_adjusted_team_contribution_dpm": 0.0,
                "playoff_workload_share": 0.75,
                "regular_season_workload_share": 0.75,
            }
            for team_id in range(1, 17)
        ]
    )
    focal = pd.DataFrame(
        [
            {
                "season": 2026,
                "player_id": 999,
                "own_dpm": 0.0,
                "playoff_adjusted_dpm": 2.0,
                "salary_share": 0.20,
            }
        ]
    )

    result = add_salary_peer_title_lift(
        focal,
        peers,
        teams,
        nodes,
        PlayoffGameModel(home_advantage=0.20, epm_coefficient=0.12),
    )

    assert np.isclose(result.loc[0, "title_lift_over_contract_v2"], 0.0)
    assert result.loc[0, "playoff_adjusted_title_lift_over_contract_v2"] > 0.0
    assert np.isclose(
        result.loc[0, "playoff_adjusted_title_lift_over_contract_v2"],
        result.loc[0, "playoff_adjusted_fixed_bracket_title_lift"]
        + result.loc[0, "playoff_adjusted_seeding_title_lift"],
    )
