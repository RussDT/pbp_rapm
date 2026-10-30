from __future__ import annotations

import numpy as np
import pandas as pd

from nba_pipeline.scripts.build_sq_three_point_difficulty import (
    _select_stat_match,
    aggregate_player_difficulty,
)
from nba_pipeline.scripts.build_sq_three_point_difficulty_html import (
    _records as qsq_records,
)
from nba_pipeline.scripts.build_russell_v2_counterfactual import (
    aggregate_counterfactual_difficulty,
    neutralize_shooter_features,
)
from nba_pipeline.scripts.build_sq_rich_three_point_counterfactual import (
    aggregate_rich_counterfactual_difficulty,
    neutralize_rich_shooter_features,
)


def test_player_difficulty_is_centered_and_closes_effective_identity() -> None:
    shots = pd.DataFrame(
        {
            "game_id": [100, 100, 200, 200],
            "player_id": [1, 1, 2, 2],
            "shooter_name": ["Hard Diet", "Hard Diet", "Easy Diet", "Easy Diet"],
            "team_id": [10, 10, 20, 20],
            "made_3": [1, 0, 1, 1],
            "expected_make_probability": [0.30, 0.30, 0.60, np.nan],
        }
    )

    result, audit = aggregate_player_difficulty(shots, season_label="test")
    result = result.set_index("sq_player_id")

    assert np.isclose(result.loc[1, "difficulty_pp"], 10.0)
    assert np.isclose(result.loc[2, "difficulty_pp"], -10.0)
    np.testing.assert_allclose(
        result["raw_3p_pct"] + result["difficulty"],
        result["effective_3p_pct"],
        atol=1e-12,
    )
    assert abs(float(audit["attempt_weighted_difficulty"])) < 1e-12
    assert audit["model_covered_three_pa"] == 3


def test_stat_match_requires_unique_or_clearly_better_candidate() -> None:
    candidates = pd.DataFrame(
        {
            "player1_id": [11, 22],
            "three_pa": [100, 80],
            "three_pm": [35, 30],
        }
    )
    exact = _select_stat_match(
        sq_three_pa=100, sq_three_pm=35, candidates=candidates
    )
    assert exact is not None
    assert int(exact["player1_id"]) == 11

    nearest = _select_stat_match(
        sq_three_pa=102, sq_three_pm=35, candidates=candidates
    )
    assert nearest is not None
    assert int(nearest["player1_id"]) == 11

    ambiguous = candidates.assign(three_pa=[100, 101], three_pm=[35, 35])
    assert (
        _select_stat_match(
            sq_three_pa=100, sq_three_pm=35, candidates=ambiguous
        )
        is not None
    )
    tied = candidates.assign(three_pa=[100, 100], three_pm=[35, 35])
    assert (
        _select_stat_match(sq_three_pa=100, sq_three_pm=35, candidates=tied)
        is None
    )


def test_qsq_records_use_attempt_weighted_season_baseline(tmp_path) -> None:
    source = pd.DataFrame(
        {
            "Shooter": ["Hard", "Easy"],
            "Season": ["test", "test"],
            "Shots Attempted": [100, 300],
            "Shots Attempted Per 100 Possessions": [10.0, 12.0],
            "eFG%": [52.5, 60.0],
            "qSQ": [45.0, 60.0],
            "qSI": [7.5, 0.0],
            "qSP": [50.0, 58.0],
            "SSI Player ID": ["hard-id", "easy-id"],
        }
    )
    path = tmp_path / "3p_shot_sq.csv"
    source.to_csv(path, index=False)

    result = pd.DataFrame(qsq_records(path)).set_index("player")
    assert np.isclose(result.loc["Hard", "baseline"], 0.375)
    assert np.isclose(result.loc["Hard", "difficulty"], 0.075)
    assert np.isclose(result.loc["Easy", "difficulty"], -0.025)
    np.testing.assert_allclose(
        result["raw"] + result["difficulty"], result["effective"]
    )


def test_russell_v2_counterfactual_neutralizes_only_shooter_features() -> None:
    shots = pd.DataFrame(
        {
            "shot_distance_ft": [24.0, 30.0],
            "desc_step_back": [0, 1],
            "epm_fg3pct": [0.31, 0.44],
            "epm_midpct": [0.38, 0.51],
            "epm_ftpct": [0.70, 0.91],
            "epm_fg3a": [3.0, 12.0],
            "sh_zone_rate": [0.32, 0.42],
            "sh_zone_n": [40.0, 800.0],
            "sh_jumper_rate": [0.35, 0.48],
        }
    )
    neutral = neutralize_shooter_features(shots)
    np.testing.assert_allclose(neutral["shot_distance_ft"], [24.0, 30.0])
    np.testing.assert_array_equal(neutral["desc_step_back"], [0, 1])
    for column in [
        "epm_fg3pct",
        "epm_midpct",
        "epm_ftpct",
        "epm_fg3a",
        "sh_zone_rate",
        "sh_zone_n",
        "sh_jumper_rate",
    ]:
        assert neutral[column].nunique() == 1
    np.testing.assert_allclose(shots["epm_fg3pct"], [0.31, 0.44])


def test_russell_v2_counterfactual_difficulty_centers_and_attaches(tmp_path) -> None:
    scored = pd.DataFrame(
        {
            "season_end_year": [2022] * 4,
            "game_id": ["0022100001"] * 4,
            "shooter_id": [11, 11, 22, 22],
            "shooter_name": ["Hard", "Hard", "Easy", "Easy"],
            "made": [1, 0, 1, 1],
            "p_actual_skill": [0.35, 0.35, 0.45, 0.45],
            "p_neutral_skill": [0.30, 0.30, 0.50, 0.50],
        }
    )
    player = aggregate_counterfactual_difficulty(scored)
    player = player.set_index("shooter_name")
    assert np.isclose(player.loc["Hard", "v2_difficulty"], 0.10)
    assert np.isclose(player.loc["Easy", "v2_difficulty"], -0.10)
    assert np.isclose(
        np.average(player["v2_difficulty"], weights=player["v2_attempts"]),
        0.0,
    )

    source = pd.DataFrame(
        {
            "Shooter": ["Hard", "Easy"],
            "Season": ["2021-22", "2021-22"],
            "Shots Attempted": [3, 3],
            "Shots Attempted Per 100 Possessions": [10.0, 11.0],
            "eFG%": [50.0, 60.0],
            "qSQ": [48.0, 57.0],
            "qSI": [2.0, 3.0],
            "qSP": [49.0, 58.0],
            "SSI Player ID": ["hard", "easy"],
        }
    )
    source_path = tmp_path / "sq.csv"
    source.to_csv(source_path, index=False)
    counterfactual_path = tmp_path / "v2_counterfactual.csv"
    player.reset_index().to_csv(counterfactual_path, index=False)
    stats = pd.DataFrame(
        {
            "year": [2022, 2022],
            "nba_id": [11, 22],
            "ShortName": ["Hard", "Easy"],
        }
    )
    stats_path = tmp_path / "stats.csv"
    stats.to_csv(stats_path, index=False)
    result = pd.DataFrame(
        qsq_records(source_path, counterfactual_path, stats_path)
    ).set_index("player")
    assert result["v2_difficulty"].notna().all()
    np.testing.assert_allclose(
        result["raw"] + result["v2_difficulty"], result["v2_effective"]
    )


def test_sq_rich_counterfactual_centers_and_reports_geometry_coverage() -> None:
    scored = pd.DataFrame(
        {
            "season_end_year": [2026] * 4,
            "sq_player_id": [11, 11, 22, 22],
            "shooter_name": ["Hard", "Hard", "Easy", "Easy"],
            "made": [1, 0, 1, 1],
            "p_actual_hybrid": [0.35, 0.35, 0.45, 0.45],
            "p_neutral_hybrid": [0.30, 0.30, 0.50, 0.50],
            "p_neutral_rich": [0.20, np.nan, 0.60, 0.60],
            "geometry_available": [1, 0, 1, 1],
        }
    )
    player = aggregate_rich_counterfactual_difficulty(scored).set_index(
        "shooter_name"
    )
    assert np.isclose(player.loc["Hard", "sq_rich_difficulty"], 0.10)
    assert np.isclose(player.loc["Easy", "sq_rich_difficulty"], -0.10)
    assert np.isclose(player.loc["Hard", "sq_rich_geometry_coverage"], 0.50)
    assert np.isclose(player.loc["Easy", "sq_rich_geometry_coverage"], 1.00)
    assert np.isclose(
        player.loc["Hard", "sq_rich_geometry_only_difficulty"],
        0.4666666666666667 - 0.20,
    )
    assert np.isclose(
        np.average(
            player["sq_rich_difficulty"],
            weights=player["sq_rich_attempts"],
        ),
        0.0,
    )


def test_qsq_viewer_attaches_sq_rich_difficulty_and_coverage(tmp_path) -> None:
    source = pd.DataFrame(
        {
            "Shooter": ["Hard", "Easy"],
            "Season": ["2025-26", "2025-26"],
            "Shots Attempted": [100, 300],
            "Shots Attempted Per 100 Possessions": [10.0, 12.0],
            "eFG%": [52.5, 60.0],
            "qSQ": [45.0, 60.0],
            "qSI": [7.5, 0.0],
            "qSP": [50.0, 58.0],
            "SSI Player ID": ["hard-id", "easy-id"],
        }
    )
    qsq_path = tmp_path / "sq.csv"
    source.to_csv(qsq_path, index=False)
    rich = pd.DataFrame(
        {
            "season_end_year": [2026, 2026],
            "shooter_name": ["Hard", "Easy"],
            "sq_rich_attempts": [100, 300],
            "sq_rich_neutral_expected": [0.30, 0.50],
            "sq_rich_neutral_baseline": [0.45, 0.45],
            "sq_rich_difficulty": [0.15, -0.05],
            "sq_rich_geometry_coverage": [0.80, 0.40],
            "sq_rich_geometry_only_difficulty": [0.18, -0.04],
        }
    )
    rich_path = tmp_path / "rich.csv"
    rich.to_csv(rich_path, index=False)

    result = pd.DataFrame(
        qsq_records(qsq_path, sq_rich_counterfactual=rich_path)
    ).set_index("player")
    np.testing.assert_allclose(result["sq_rich_difficulty"], [0.15, -0.05])
    np.testing.assert_allclose(result["sq_rich_geometry_coverage"], [0.80, 0.40])
    np.testing.assert_allclose(
        result["sq_rich_geometry_only_difficulty"], [0.18, -0.04]
    )
    np.testing.assert_allclose(
        result["raw"] + result["sq_rich_difficulty"],
        result["sq_rich_effective"],
    )


def test_sq_rich_neutralization_preserves_defender_geometry() -> None:
    features = pd.DataFrame(
        {
            "shot_distance_ft": [24.0, 29.0],
            "closest_defender_distance": [1.0, 7.0],
            "descriptor_step_back": [0.0, 1.0],
            "shooter_prior_make_probability": [0.31, 0.44],
            "shooter_prior_log_attempts": [2.0, 6.0],
        }
    )
    neutral = neutralize_rich_shooter_features(features)
    np.testing.assert_allclose(neutral["shot_distance_ft"], [24.0, 29.0])
    np.testing.assert_allclose(
        neutral["closest_defender_distance"], [1.0, 7.0]
    )
    np.testing.assert_array_equal(neutral["descriptor_step_back"], [0.0, 1.0])
    assert neutral["shooter_prior_make_probability"].nunique() == 1
    assert neutral["shooter_prior_log_attempts"].nunique() == 1
