from __future__ import annotations

import pandas as pd

from nba_pipeline.scripts.playoff_epm_prior import (
    build_dynamic_playoff_epm_ratings,
    build_walk_forward_playoff_epm_ratings,
    evaluate_walk_forward_playoff_ratings,
)


def _dynamic_prior_fixture(
    second_season_focal_one_points: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, float | int | str]] = []
    for season, focal_one_points, focal_two_points in (
        (2020, 2.0, 0.0),
        (2021, second_season_focal_one_points, 0.0),
    ):
        for repeat in range(30):
            for focal, points in ((1, focal_one_points), (2, focal_two_points)):
                rows.append(
                    {
                        "season": season,
                        "game_id": f"{season}-{repeat}",
                        "O1": focal,
                        "O2": 3,
                        "O3": 4,
                        "O4": 5,
                        "O5": 6,
                        "D1": 7,
                        "D2": 8,
                        "D3": 9,
                        "D4": 10,
                        "D5": 11,
                        "points": points,
                    }
                )
    possessions = pd.DataFrame(rows)
    priors = pd.DataFrame(
        [
            {
                "season": season,
                "player_id": player_id,
                "o_dpm": 0.0,
                "d_dpm": 0.0,
                "dpm": 0.0,
            }
            for season in (2020, 2021, 2022)
            for player_id in range(1, 12)
        ]
    )
    return possessions, priors


def test_dynamic_playoff_prior_carries_only_completed_postseasons_forward() -> None:
    ordinary_possessions, priors = _dynamic_prior_fixture(2.0)
    extreme_possessions, _ = _dynamic_prior_fixture(20.0)

    ordinary, _ = build_dynamic_playoff_epm_ratings(
        ordinary_possessions,
        priors,
        alpha=10.0,
        persistence=1.0,
    )
    extreme, _ = build_dynamic_playoff_epm_ratings(
        extreme_possessions,
        priors,
        alpha=10.0,
        persistence=1.0,
    )
    ordinary = ordinary.set_index(["season", "player_id"])
    extreme = extreme.set_index(["season", "player_id"])

    assert ordinary.loc[(2020, 1), "playoff_translation_dpm"] == 0.0
    assert ordinary.loc[(2021, 1), "playoff_translation_dpm"] > 0.0
    assert ordinary.loc[(2021, 2), "playoff_translation_dpm"] < 0.0
    assert ordinary.loc[(2021, 1), "playoff_translation_dpm"] == extreme.loc[
        (2021, 1), "playoff_translation_dpm"
    ]
    assert ordinary.loc[(2022, 1), "playoff_translation_dpm"] < extreme.loc[
        (2022, 1), "playoff_translation_dpm"
    ]


def test_dynamic_playoff_prior_mean_reverts_during_missed_postseasons() -> None:
    possessions, priors = _dynamic_prior_fixture(2.0)
    possessions = possessions.loc[possessions["season"].eq(2020)].copy()

    result, _ = build_dynamic_playoff_epm_ratings(
        possessions,
        priors,
        alpha=10.0,
        persistence=0.5,
    )
    result = result.set_index(["season", "player_id"])
    first_prior = result.loc[(2021, 1), "playoff_translation_dpm"]
    second_prior = result.loc[(2022, 1), "playoff_translation_dpm"]

    assert first_prior > 0.0
    assert second_prior == first_prior * 0.5


def test_dynamic_playoff_prior_adjusted_sides_close_to_adjusted_net() -> None:
    possessions, priors = _dynamic_prior_fixture(2.0)
    priors["dpm"] = 123.0

    result, _ = build_dynamic_playoff_epm_ratings(
        possessions,
        priors,
        alpha=10.0,
        persistence=1.0,
    )

    assert (
        result["playoff_adjusted_dpm"]
        == result["playoff_adjusted_o_dpm"] + result["playoff_adjusted_d_dpm"]
    ).all()


def test_walk_forward_playoff_prior_uses_only_earlier_postseasons() -> None:
    rows: list[dict[str, float | int | str]] = []
    for season, focal_one_points, focal_two_points in (
        (2020, 2.0, 0.0),
        (2021, 0.0, 2.0),
    ):
        for repeat in range(30):
            for focal, points in ((1, focal_one_points), (2, focal_two_points)):
                rows.append(
                    {
                        "season": season,
                        "game_id": f"{season}-{repeat}",
                        "O1": focal,
                        "O2": 3,
                        "O3": 4,
                        "O4": 5,
                        "O5": 6,
                        "D1": 7,
                        "D2": 8,
                        "D3": 9,
                        "D4": 10,
                        "D5": 11,
                        "points": points,
                    }
                )
    possessions = pd.DataFrame(rows)
    priors = pd.DataFrame(
        [
            {
                "season": season,
                "player_id": player_id,
                "o_dpm": 0.0,
                "d_dpm": 0.0,
                "dpm": 0.0,
            }
            for season in (2020, 2021)
            for player_id in range(1, 13)
        ]
    )

    result, report = build_walk_forward_playoff_epm_ratings(
        possessions,
        priors,
        alpha_offense=10.0,
        alpha_defense=10.0,
        history_window=None,
    )
    indexed = result.set_index(["season", "player_id"])

    assert indexed.loc[(2020, 1), "playoff_translation_dpm"] == 0.0
    assert indexed.loc[(2021, 1), "playoff_translation_o_dpm"] > 0.0
    assert indexed.loc[(2021, 2), "playoff_translation_o_dpm"] < 0.0
    assert indexed.loc[(2021, 1), "playoff_translation_o_dpm"] > indexed.loc[
        (2021, 2), "playoff_translation_o_dpm"
    ]
    assert indexed.loc[(2021, 12), "playoff_translation_dpm"] == 0.0
    assert report["timing"] == "earlier postseasons only"


def test_playoff_prior_gate_is_zero_when_adjusted_ratings_equal_epm() -> None:
    possessions = pd.DataFrame(
        [
            {
                "season": 2021,
                "game_id": "g1",
                "O1": 1,
                "O2": 2,
                "O3": 3,
                "O4": 4,
                "O5": 5,
                "D1": 6,
                "D2": 7,
                "D3": 8,
                "D4": 9,
                "D5": 10,
                "points": points,
            }
            for points in (0.0, 2.0, 3.0)
        ]
    )
    priors = pd.DataFrame(
        [
            {
                "season": 2021,
                "player_id": player_id,
                "o_dpm": 0.1 * player_id,
                "d_dpm": -0.05 * player_id,
                "dpm": 0.05 * player_id,
            }
            for player_id in range(1, 11)
        ]
    )
    adjusted = priors.assign(
        playoff_adjusted_o_dpm=priors["o_dpm"],
        playoff_adjusted_d_dpm=priors["d_dpm"],
        playoff_adjusted_dpm=priors["dpm"],
    )

    report = evaluate_walk_forward_playoff_ratings(
        possessions,
        priors,
        adjusted,
        start_season=2021,
        end_season=2021,
        bootstrap_samples=100,
        seed=7,
    )

    assert report["mse_delta_vs_preplayoff_epm"] == 0.0
