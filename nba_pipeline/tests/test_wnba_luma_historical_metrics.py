import json

import numpy as np
import pandas as pd

from wnba_test.build_luma_historical_metrics import (
    COMPONENTS,
    DEFENSE_FEATURES,
    OFFENSE_FEATURES,
    _side_record,
    _target_matrix,
    add_component_targets,
    build_design,
    build_prior_matrix,
    compare_to_event_history,
    map_features_to_luma,
    load_luma_season,
    load_pbpstats_features,
    solve_block,
)
from wnba_test.build_wnba_box_priors import feature_columns


def _observation(offense, defense, points, tally, seconds=60.0):
    return _side_record(
        season=2025,
        season_type="rs",
        game_id="g1",
        date="2025-01-01",
        seconds=seconds,
        offense=offense,
        defense=defense,
        points=points,
        tally=np.asarray(tally, dtype=float),
    )


def test_luma_component_numerators_close_to_points_above_possession_baseline():
    a = [f"A{i}" for i in range(5)]
    b = [f"B{i}" for i in range(5)]
    rows = [
        _observation(a, b, 7, [1, 2, 1, 2, 1, 3, 2, 0, 1, 1, 0, 0, 0, 1, 0, 0]),
        _observation(b, a, 5, [1, 2, 1, 0, 1, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        _observation(a, b, 4, [1, 0, 1, 2, 1, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0]),
    ]
    frame = pd.DataFrame(rows)

    baselines = add_component_targets(frame)
    component_sum = frame[list(COMPONENTS)].sum(axis=1)

    np.testing.assert_allclose(component_sum, frame["TOTAL_CENTERED"], atol=1e-12)
    assert baselines["max_row_algebra_error"] < 1e-12
    np.testing.assert_allclose(
        frame["TOTAL_CENTERED"],
        frame["points"] - baselines["league_ppp"] * frame["possessions"],
    )


def test_joint_ridge_preserves_player_level_component_closure():
    a = [f"A{i}" for i in range(5)]
    b = [f"B{i}" for i in range(5)]
    c = ["A0", "A1", "A2", "B3", "B4"]
    d = ["B0", "B1", "B2", "A3", "A4"]
    tallies = [
        [1, 2, 1, 0, 1, 3, 2, 1, 1, 0, 0, 0, 0, 1, 0, 0],
        [1, 0, 1, 2, 1, 3, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [1, 2, 1, 2, 1, 0, 2, 2, 1, 0, 0, 0, 0, 0, 1, 0],
        [1, 2, 1, 0, 1, 3, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    points = [6, 5, 6, 7]
    frame = pd.DataFrame(
        [
            _observation(a, b, points[0], tallies[0]),
            _observation(b, a, points[1], tallies[1]),
            _observation(c, d, points[2], tallies[2]),
            _observation(d, c, points[3], tallies[3]),
        ]
    )
    add_component_targets(frame)
    design, players, weights, _ = build_design(frame)
    targets = _target_matrix(frame)
    rng = np.random.default_rng(7)
    prior = rng.normal(0, 0.01, size=(2 * len(players), len(COMPONENTS) + 1))
    prior[:, -1] = prior[:, : len(COMPONENTS)].sum(axis=1)

    result = solve_block(
        design,
        players,
        weights,
        targets,
        off_alpha=50.0,
        def_alpha=80.0,
        prior=prior,
    )

    np.testing.assert_allclose(
        result.beta[:, : len(COMPONENTS)].sum(axis=1),
        result.beta[:, -1],
        atol=1e-12,
    )


def test_loader_rejects_score_mismatch_and_zero_second_stints(tmp_path):
    stints = tmp_path / "data" / "stints"
    stints.mkdir(parents=True)
    a = [f"A{i}" for i in range(5)]
    b = [f"B{i}" for i in range(5)]
    valid_tally = [1, 2, 1, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    payload = {
        "game": {
            "date": "2025-01-01",
            "stints": [
                [a, b, 60, 5, 5, valid_tally, valid_tally],
                [a, b, 60, 4, 5, valid_tally, valid_tally],
                [a, b, 0, 5, 5, valid_tally, valid_tally],
            ],
        }
    }
    (stints / "rs_2025.json").write_text(json.dumps(payload))

    loaded = load_luma_season(tmp_path, 2025, ("rs",))

    assert loaded.audit["candidate_sides"] == 6
    assert loaded.audit["kept_sides"] == 3
    assert loaded.audit["exclusion_reasons"]["score_tally_mismatch"] == 1
    assert loaded.audit["exclusion_reasons"]["nonpositive_seconds"] == 2


def test_prior_builder_never_uses_target_or_future_seasons():
    players = [f"P{i:02d}" for i in range(25)]
    target_rows = []
    box_rows = []
    for season in (2022, 2023, 2024):
        for index, player in enumerate(players):
            row = {
                "season": season,
                "player_id": player,
                "off_possessions": 1000.0,
                "def_possessions": 1000.0,
            }
            for component in COMPONENTS:
                row[f"{component}_off"] = float(index) / 10.0
                row[f"{component}_def"] = -float(index) / 20.0
            target_rows.append(row)
            feature_row = {"season": season, "player_id": player}
            all_features = {
                feature
                for mapping in (OFFENSE_FEATURES, DEFENSE_FEATURES)
                for features in mapping.values()
                for feature in features
            }
            for feature_number, feature in enumerate(sorted(all_features)):
                feature_row[feature] = index / 10.0 + feature_number / 100.0
            box_rows.append(feature_row)

    prior, metadata = build_prior_matrix(
        season=2024,
        players=players,
        target_panel=pd.DataFrame(target_rows),
        box_panel=pd.DataFrame(box_rows),
        mapper_alpha_grid=(0.1, 1.0, 10.0, 100.0),
        prior_scale=1.5,
    )

    assert metadata["training_seasons"] == [2022, 2023]
    assert 2024 not in metadata["training_seasons"]
    assert prior.shape == (2 * len(players), len(COMPONENTS) + 1)
    np.testing.assert_allclose(prior[:, -1], prior[:, : len(COMPONENTS)].sum(axis=1))


def test_historical_feature_contract_matches_modern_component_prior():
    prefix_by_component = {
        "RIM_FREQ": "ALT_EFG_RIM_FREQ",
        "RIM_FG": "ALT_EFG_RIM_FG",
        "MID_BASE": "ALT_EFG_MID_FG",
        "THREE_FREQ": "ALT_EFG_THREE_FREQ",
        "THREE_FG": "ALT_EFG_THREE_FG",
        "FT_VALUE": "ALT_FT",
        "TOV_VALUE": "ALT_TOV_VALUE",
        "OREB_OPPORTUNITY_VALUE": "SECOND_CHANCE_CLEAN",
    }
    for component, prefix in prefix_by_component.items():
        assert list(OFFENSE_FEATURES[component]) == feature_columns(prefix, "off")
        assert list(DEFENSE_FEATURES[component]) == feature_columns(prefix, "def")


def test_offensive_prior_distinguishes_assisted_and_self_created_scoring(tmp_path):
    source = pd.DataFrame(
        {
            "EntityId": [1, 2],
            "Name": ["Assisted Finisher", "Self Creator"],
            "OffPoss": [100.0, 100.0],
            "DefPoss": [100.0, 100.0],
            "PtsAssisted2s": [20.0, 0.0],
            "PtsUnassisted2s": [0.0, 20.0],
            "PtsAssisted3s": [18.0, 0.0],
            "PtsUnassisted3s": [0.0, 18.0],
        }
    )
    path = tmp_path / "2025_pbp.csv"
    source.to_csv(path, index=False)

    features = load_pbpstats_features(path, 2025)

    assert features.loc[0, "two_assisted_share"] > features.loc[1, "two_assisted_share"]
    assert features.loc[0, "three_assisted_share"] > features.loc[1, "three_assisted_share"]
    assert features.loc[0, "assisted_two_points100"] > features.loc[1, "assisted_two_points100"]
    assert features.loc[0, "unassisted_two_points100"] < features.loc[1, "unassisted_two_points100"]
    assert {
        "assisted_two_points100",
        "unassisted_two_points100",
        "two_assisted_share",
    }.issubset(OFFENSE_FEATURES["RIM_FG"])
    assert {
        "assisted_three_points100",
        "unassisted_three_points100",
        "three_assisted_share",
    }.issubset(OFFENSE_FEATURES["THREE_FG"])


def test_foul_creation_and_foul_cost_are_in_the_ft_prior_contract():
    assert {"fta100", "fouls_drawn100", "shooting_fouls_drawn100"}.issubset(
        OFFENSE_FEATURES["FT_VALUE"]
    )
    assert {"fouls100", "shooting_fouls100"}.issubset(
        DEFENSE_FEATURES["FT_VALUE"]
    )


def test_pbpstats_name_mapping_is_season_specific_and_audited():
    features = pd.DataFrame(
        {
            "source_player_id": ["100", "200"],
            "source_player_name": ["A'ja Example", "Name Only"],
            "normalized_name": ["ajaexample", "nameonly"],
            "season": [2015, 2015],
        }
    )
    crosswalk = pd.DataFrame(
        {
            "luma_id": ["L1", "L2", "L3"],
            "display_name": ["Different Name", "Name-Only", "Name Only"],
            "wnba_stats_id": ["100", "", ""],
            "first_season": [2010, 2010, 2020],
            "last_season": [2020, 2019, 2025],
        }
    )

    mapped, audit = map_features_to_luma(features, crosswalk, 2015)

    assert mapped["player_id"].tolist() == ["L1", "L2"]
    assert mapped["match_method"].tolist() == ["wnba_stats_id", "normalized_name"]
    assert audit["matched_feature_rows"] == 2


def test_event_comparison_keeps_rebound_opportunity_distinct_from_second_chance(tmp_path):
    luma = pd.DataFrame(
        {
            "season": [2024, 2024],
            "wnba_stats_id": ["1", "2"],
            "possessions": [1200, 1300],
            "LUMA_VPM": [2.0, -1.0],
            "LUMA_ORAPM": [1.0, -0.5],
            "LUMA_DRAPM": [1.0, -0.5],
            "TS_off": [0.7, -0.2],
            "TS_def": [0.6, -0.1],
            "TOV_off": [0.2, -0.1],
            "TOV_def": [0.3, -0.2],
            "REB_off": [0.1, -0.1],
            "REB_def": [0.1, -0.2],
        }
    )
    event = pd.DataFrame(
        {
            "season": [2024, 2024],
            "player_id": [1, 2],
            "possessions": [1400, 1500],
            "net_rapm": [1.5, -0.8],
            "off": [0.8, -0.4],
            "def": [0.7, -0.4],
            "oALT_EFG_VALUE": [0.5, -0.2],
            "oALT_FT": [0.1, 0.0],
            "dALT_EFG_VALUE": [0.4, -0.1],
            "dALT_FT": [0.1, 0.0],
            "oALT_TOV_VALUE": [0.1, -0.1],
            "dALT_TOV_VALUE": [0.2, -0.2],
            "oSC": [0.2, -0.1],
            "dSC": [0.3, -0.2],
        }
    )
    event_path = tmp_path / "event.csv"
    event.to_csv(event_path, index=False)

    comparison = compare_to_event_history(luma, event_path)

    assert "REB_vs_SECOND_CHANCE_off" in set(comparison["metric"])
    assert not comparison["metric"].str.fullmatch("REB_off").any()
    qualified = comparison.loc[
        (comparison["scope"] == "common_1000_possessions")
        & (comparison["metric"] == "TOTAL_net")
    ].iloc[0]
    assert qualified["n_players"] == 2
