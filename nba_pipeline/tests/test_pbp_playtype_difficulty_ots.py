import pandas as pd
import pytest

from nba_pipeline.scripts.analyze_pbp_playtype_difficulty_ots import (
    CREATION_CONTROL_FEATURES,
    FLOATING_SCORING_FEATURES,
    aggregate_player_difficulty,
    build_model_features,
    classify_pbp_shots,
    estimate_category_prices,
    load_creation_controls,
)


def test_classification_ignores_outcome_and_made_only_assist_tag():
    frame = pd.DataFrame(
        {
            "home_description": [
                "Player 26' Pullup 3PT Jump Shot (3 PTS) (Passer 4 AST)",
                "MISS Player 26' Pullup 3PT Jump Shot",
                "Player 2' Cutting Layup Shot (2 PTS) (Passer 2 AST)",
                "MISS Player 2' Cutting Layup Shot",
            ],
            "visitor_description": ["", "", "", ""],
        }
    )
    categories = classify_pbp_shots(
        frame, pd.Series([True, True, False, False])
    )
    assert categories.tolist() == [
        "SELF_CREATED_3",
        "SELF_CREATED_3",
        "CUT_2",
        "CUT_2",
    ]


def test_feature_mix_is_priced_only_with_supplied_prior_season():
    pricing = pd.DataFrame(
        {
            "pbp_playtype_category": ["CUT_2"] * 100 + ["OTHER_2"] * 100,
            "actual_points": [2.0] * 100 + [0.0] * 100,
        }
    )
    prices = estimate_category_prices(
        pricing,
        pricing_season=2023,
        feature_season=2024,
        shrink_fga=0,
    )
    current = pd.DataFrame(
        {
            "player_id": [1, 1, 2, 2],
            "player_name": ["A", "A", "B", "B"],
            "pbp_playtype_category": ["CUT_2", "CUT_2", "OTHER_2", "OTHER_2"],
            # These outcomes must not affect the already supplied category prices.
            "actual_points": [0.0, 0.0, 2.0, 2.0],
        }
    )
    result = aggregate_player_difficulty(current, prices, feature_season=2024)
    a = result.loc[result["player_id"].eq(1)].iloc[0]
    b = result.loc[result["player_id"].eq(2)].iloc[0]
    assert a["pbp_expected_points_per_fga"] == pytest.approx(2.0)
    assert b["pbp_expected_points_per_fga"] == pytest.approx(0.0)
    assert a["pbp_difficulty_pp"] == pytest.approx(-50.0)
    assert b["pbp_difficulty_pp"] == pytest.approx(50.0)


def test_primary_models_include_creation_controls_identically():
    models = build_model_features(include_creation_controls=True)
    for features in models.values():
        for control in CREATION_CONTROL_FEATURES:
            assert features.count(control) == 1

    uncontrolled = build_model_features(include_creation_controls=False)
    for features in uncontrolled.values():
        assert not set(CREATION_CONTROL_FEATURES) & set(features)

    assert not any(
        feature.endswith("_role_added_100")
        for feature in models["box_scoring_baseline"]
    )
    assert FLOATING_SCORING_FEATURES == ["pts100", "TSA100"]
    assert "TS_added_100" not in models["pbp_role_added"]
    assert set(FLOATING_SCORING_FEATURES).issubset(models["pbp_role_added"])
    assert "pbp_role_added_100" in models["pbp_role_added"]
    assert "synergy_role_added_100" not in models["pbp_role_added"]
    assert "TS_added_100" not in models["synergy_role_added"]
    assert set(FLOATING_SCORING_FEATURES).issubset(models["synergy_role_added"])
    assert "synergy_role_added_100" in models["synergy_role_added"]
    assert {
        "pts100",
        "TSA100",
        "pbp_role_added_100",
        "synergy_role_added_100",
    }.issubset(models["both_roles_added"])


def test_creation_controls_join_role_cache_and_season_specific_rim_sources(tmp_path):
    role_path = tmp_path / "roles.parquet"
    passing_path = tmp_path / "passing.csv"
    current_path = tmp_path / "merged_2025.csv"
    pd.DataFrame(
        {
            "year": [2024, 2025],
            "nba_id": [1, 1],
            "PASSING_on-ball-time%": [20.0, 25.0],
            "PASSING_Potential Assists/100": [10.0, 12.0],
        }
    ).to_parquet(role_path, index=False)
    pd.DataFrame(
        {"year": [2024], "nba_id": [1], "AtRimAssists/100": [3.0]}
    ).to_csv(passing_path, index=False)
    pd.DataFrame(
        {
            "year": [2025],
            "nba_id": [1],
            "PASSING_AtRimAssists/100": [4.0],
        }
    ).to_csv(current_path, index=False)

    result = load_creation_controls(
        role_path,
        passing_path,
        current_path,
        first_feature_season=2024,
        last_feature_season=2025,
    )
    assert result["rim_assists_per100"].tolist() == [3.0, 4.0]
    assert result["on_ball_time_pct"].tolist() == [20.0, 25.0]
    assert result["potential_assists_per100"].tolist() == [10.0, 12.0]
