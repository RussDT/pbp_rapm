import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from nba_pipeline.scripts.build_adaptive_young_player_penalties import (
    build_current_multiplier_audit,
    evaluate_multiplier_grid,
    select_development_multipliers,
)
from nba_pipeline.scripts.rapm import (
    build_output_suffix,
    joint_generalized_rapm,
    load_player_penalty_multipliers,
)


def test_generalized_ridge_applies_player_specific_penalty_multiplier():
    beta = joint_generalized_rapm(
        csr_matrix([[1.0]]),
        np.array([10.0]),
        {"7_off": 0},
        ["7"],
        alpha_offense=1.0,
        alpha_defense=1.0,
        off_penalty_multipliers={"7": 4.0},
    )
    assert np.isclose(beta[0], 2.0)


def test_multiplier_loader_rejects_weakening_and_loads_both_sides(tmp_path):
    path = tmp_path / "multipliers.csv"
    pd.DataFrame(
        {"player_id": [7], "off_multiplier": [4.0], "def_multiplier": [3.0]}
    ).to_csv(path, index=False)
    offense, defense = load_player_penalty_multipliers(path)
    assert offense == {"7": 4.0}
    assert defense == {"7": 3.0}

    pd.DataFrame(
        {"player_id": [7], "off_multiplier": [0.5], "def_multiplier": [3.0]}
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="at least 1.0"):
        load_player_penalty_multipliers(path)


def test_penalty_label_keeps_adaptive_output_distinct():
    suffix = build_output_suffix(
        "all",
        pure=True,
        penalty_label="young1s_o4_d3",
    )
    assert suffix == "all_pure_young1s_o4_d3"


def test_forward_proxy_selects_distinct_offense_and_defense_multipliers():
    transitions = pd.DataFrame(
        {
            "feature_season": [2020],
            "feature_orapm_age27": [6.0],
            "feature_drapm_age27": [5.0],
            "feature_off_poss": [1000.0],
            "feature_def_poss": [1000.0],
            "target_orapm_age27": [2.0],
            "target_drapm_age27": [2.0],
            "target_off_poss": [1000.0],
            "target_def_poss": [1000.0],
        }
    )
    grid = evaluate_multiplier_grid(
        transitions,
        [1.0, 3.0, 4.0],
        off_alpha=2000.0,
        def_alpha=3000.0,
        development_end_year=2021,
    )
    assert select_development_multipliers(grid) == (4.0, 3.0)


def test_current_audit_only_penalizes_one_season_young_long_extrapolations():
    panel = pd.DataFrame(
        {
            "player_id": ["1", "2", "2", "3"],
            "player_name": ["Young One", "Young Two", "Young Two", "Prime One"],
            "season": [2026, 2025, 2026, 2026],
            "age": [20.3, 20.5, 21.5, 27.2],
            "off_poss": [1000, 1000, 1000, 1000],
            "def_poss": [1000, 1000, 1000, 1000],
        }
    )
    multipliers, audit = build_current_multiplier_audit(
        panel,
        off_multiplier=4.0,
        def_multiplier=3.0,
    )
    assert multipliers.to_dict("records") == [
        {"player_id": "1", "off_multiplier": 4.0, "def_multiplier": 3.0}
    ]
    assert audit.set_index("player_id").loc["2", "off_multiplier"] == 1.0
    assert audit.set_index("player_id").loc["3", "def_multiplier"] == 1.0
