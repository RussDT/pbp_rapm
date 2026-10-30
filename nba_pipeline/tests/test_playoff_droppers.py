from __future__ import annotations

import numpy as np
import pandas as pd

from nba_pipeline.scripts.analyze_playoff_droppers import (
    _season_window_name,
    apply_linear_adjustment,
    attach_lineup_archetypes,
    build_rotation_translation,
    evaluate_prediction,
    fit_linear_adjustment,
    load_current_rs_factor_profiles,
    load_lagged_factor_profiles,
    load_offensive_role_profiles,
    select_ridge_alpha,
)
from nba_pipeline.scripts.build_playoff_rs_factor_profiles import assemble_season_profiles


def test_factor_profile_is_strictly_lagged_and_builds_rebound_only(tmp_path) -> None:
    factor_dir = tmp_path / "factors"
    factor_dir.mkdir()
    path = factor_dir / "weighted_factors_14_18_all_rb_se_a2000_4000.csv"
    pd.DataFrame(
        [
            {
                "player_id": 1,
                "player_name": "Low dTS rebounder",
                "oTS": 0.0,
                "oTOV": 0.0,
                "oREB": 0.0,
                "oTHREE_FREQ": 0.0,
                "dTS": -2.0,
                "dTOV": 0.0,
                "dREB": 2.0,
                "off_poss": 1000,
                "def_poss": 1000,
            },
            {
                "player_id": 2,
                "player_name": "High dTS nonrebounder",
                "oTS": 0.0,
                "oTOV": 0.0,
                "oREB": 0.0,
                "oTHREE_FREQ": 0.0,
                "dTS": 2.0,
                "dTOV": 0.0,
                "dREB": -2.0,
                "off_poss": 1000,
                "def_poss": 1000,
            },
        ]
    ).to_csv(path, index=False)

    profiles, audit = load_lagged_factor_profiles(factor_dir, [2019])

    assert _season_window_name(2019) == "14_18"
    assert audit[0]["max_source_year"] == 2018
    focal = profiles.set_index("player_id").loc[1]
    assert focal["low_dts"] > 0
    assert focal["high_dreb"] > 0
    assert focal["rebound_only"] > 0
    assert profiles.set_index("player_id").loc[2, "rebound_only"] == 0


def test_lineup_archetype_uses_defenders_and_offensive_spacing() -> None:
    profiles = pd.DataFrame(
        [
            {
                "season": 2020,
                "player_id": player_id,
                "dTS_z": -1.0 if player_id >= 6 else 0.0,
                "dTOV_z": 0.0,
                "dREB_z": 1.0 if player_id >= 6 else 0.0,
                "oTS_z": 0.0,
                "oTOV_z": 0.0,
                "oREB_z": 0.0,
                "oTHREE_FREQ_z": 2.0 if player_id <= 5 else 0.0,
                "low_dts": 1.0 if player_id >= 6 else 0.0,
                "high_dreb": 1.0 if player_id >= 6 else 0.0,
                "rebound_only": 1.0 if player_id >= 6 else 0.0,
                "spacing_positive": 2.0 if player_id <= 5 else 0.0,
                "creation_load": 1.0 if player_id <= 5 else 0.0,
                "assisted_dependence": 0.5 if player_id <= 5 else 0.0,
                "transition_dependence": 0.25 if player_id <= 5 else 0.0,
            }
            for player_id in range(1, 11)
        ]
    )
    row = {"season": 2020, "game_id": "g", "points": 1.0}
    row.update({f"O{slot}": slot for slot in range(1, 6)})
    row.update({f"D{slot}": slot + 5 for slot in range(1, 6)})

    result = attach_lineup_archetypes(pd.DataFrame([row]), profiles).iloc[0]

    assert result["off_factor_coverage"] == 5
    assert result["def_factor_coverage"] == 5
    assert result["def_rebound_only"] == 1.0
    assert result["off_spacing_z"] == 2.0
    assert result["rebound_only_x_spacing"] == 2.0
    assert result["off_role_coverage"] == 5
    assert result["off_creation_load"] == 1.0


def test_current_rs_profile_is_same_season_and_positive_good(tmp_path) -> None:
    path = tmp_path / "current.csv"
    pd.DataFrame(
        [
            {
                "season": 2019,
                "player_id": player_id,
                "player_name": f"P{player_id}",
                "oTS": 0.0,
                "oTOV": 0.0,
                "oREB": 0.0,
                "oTHREE_FREQ": float(player_id),
                "dTS": -1.0 if player_id == 1 else 1.0,
                "dTOV": 0.0,
                "dREB": 1.0 if player_id == 1 else -1.0,
                "off_poss": 1000,
                "def_poss": 1000,
                "source_kind": "current_regular_season_only",
                "source_max_season": 2019,
            }
            for player_id in (1, 2)
        ]
    ).to_csv(path, index=False)

    profiles, audit = load_current_rs_factor_profiles(path, [2019])

    assert audit[0]["max_source_year"] == 2019
    assert profiles.set_index("player_id").loc[1, "rebound_only"] > 0
    assert profiles["source_window"].unique().tolist() == ["2019_RS"]


def test_current_rs_assembler_flips_solver_defense_once(tmp_path) -> None:
    paths = {}
    for metric, off, defense in (
        ("TS", 1.0, -2.0),
        ("TOV", 3.0, -4.0),
        ("REB", 5.0, -6.0),
        ("THREE_FREQ", 7.0, -8.0),
    ):
        path = tmp_path / f"{metric}.csv"
        pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "player_name": "P1",
                    "off": off,
                    "def": defense,
                    "off_poss": 100,
                    "def_poss": 100,
                }
            ]
        ).to_csv(path, index=False)
        paths[metric] = path

    result = assemble_season_profiles(2020, paths).iloc[0]

    assert result["oTS"] == 1.0
    assert result["dTS"] == 2.0
    assert result["oTOV"] == 3.0
    assert result["dTOV"] == 4.0
    assert result["oREB"] == 5.0
    assert result["dREB"] == 6.0
    assert result["oTHREE_FREQ"] == 7.0
    assert result["dTHREE_FREQ"] == 8.0


def test_offensive_role_profiles_use_target_regular_season(tmp_path) -> None:
    path = tmp_path / "box.csv"
    rows = []
    for player_id in (1, 2, 3):
        rows.append(
            {
                "nba_id": player_id,
                "year": 2022,
                "Name": f"P{player_id}",
                "Minutes": 100 + player_id,
                "final_creation_TSA100": float(player_id),
                "PASSING_on-ball-time%": float(player_id * 2),
                "PASSING_Potential Assists/100": float(player_id * 3),
                "v2_pct_assisted_2pt": 0.1 * player_id,
                "v2_pct_assisted_3pt": 0.2 * player_id,
                "v2_pct_pts_fast_break": 0.05 * player_id,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)

    profiles, audit = load_offensive_role_profiles(path, [2022])

    assert profiles["season"].unique().tolist() == [2022]
    assert profiles["role_source_year"].unique().tolist() == [2022]
    assert profiles[["creation_load", "assisted_dependence", "transition_dependence"]].notna().all().all()
    assert audit[0]["source_year"] == 2022


def test_one_se_ridge_can_select_exact_null_update() -> None:
    rows = []
    for season in (2014, 2015, 2016):
        for game in range(10):
            rows.append(
                {
                    "season": season,
                    "actual_centered": 0.0,
                    "prediction_base": 0.0,
                    "noise": -1.0 if game % 2 else 1.0,
                }
            )
    frame = pd.DataFrame(rows)

    alpha, _ = select_ridge_alpha(
        frame,
        target_prediction="prediction_base",
        features=["noise"],
    )
    model = fit_linear_adjustment(
        frame,
        target_prediction="prediction_base",
        features=["noise"],
        ridge_alpha=alpha,
    )

    assert alpha == 1_000_000_000.0
    assert model.coefficients[0] == 0.0


def test_linear_adjustment_recovers_signal_and_improves_mse() -> None:
    rows = []
    for season in (2017, 2018):
        for game in range(20):
            feature = -1.0 if game % 2 == 0 else 1.0
            rows.append(
                {
                    "season": season,
                    "game_key": f"{season}_{game}",
                    "actual_centered": 0.2 * feature,
                    "prediction_base": 0.0,
                    "signal": feature,
                }
            )
    frame = pd.DataFrame(rows)

    model = fit_linear_adjustment(
        frame, target_prediction="prediction_base", features=["signal"]
    )
    predicted = apply_linear_adjustment(frame, model, output_column="candidate")
    report = evaluate_prediction(
        predicted,
        candidate="candidate",
        bootstrap_samples=100,
        seed=3,
    )

    assert np.isclose(model.coefficients[0], 0.2)
    assert report["equal_season_mse_delta"] < 0
    assert report["equal_season_ci_high"] < 0


def test_rotation_share_is_one_team_unit_and_common_players_only(tmp_path) -> None:
    profiles = pd.DataFrame(
        [
            {
                "season": 2021,
                "player_id": player_id,
                "dTS_z": -1.0,
                "dREB_z": 1.0,
                "low_dts": 1.0,
                "high_dreb": 1.0,
                "rebound_only": 1.0,
                "source_window": "16_20",
            }
            for player_id in range(1, 6)
        ]
    )
    phase_paths = []
    for label, possessions in (("rs", 80), ("ps", 20)):
        rows = []
        for player_id in range(1, 6):
            rows.append(
                {
                    "season": 2021,
                    "player_id": player_id,
                    "player_name": f"P{player_id}",
                    "off_team_id": 10,
                    "possessions": possessions,
                    "game_id": f"{label}_g",
                }
            )
        path = tmp_path / f"{label}.parquet"
        pd.DataFrame(rows).to_parquet(path, index=False)
        phase_paths.append(path)

    rotation, _ = build_rotation_translation(phase_paths[0], phase_paths[1], profiles)

    assert len(rotation) == 5
    assert np.isclose(rotation["rs_rotation_share"].sum(), 5.0)
    assert np.isclose(rotation["ps_rotation_share"].sum(), 5.0)
    assert np.allclose(rotation["rotation_share_delta"], 0.0)
