from __future__ import annotations

import numpy as np
import pandas as pd

from wnba_test.cross_validate_wnba_rtss_total_priors import (
    TARGET_ALPHA_SPECS,
    load_target_rows,
    rtss_feature_family,
    rtss_features_for_side,
    select_compact_rtss_features,
    target_path,
)


def test_target_alpha_grid_uses_orthogonal_low_alpha_sweeps() -> None:
    specs = {(off, defense) for _label, off, defense in TARGET_ALPHA_SPECS}
    assert {(off, 1000.0) for off in (25.0, 50.0, 100.0, 250.0, 500.0, 750.0)} <= specs
    assert {(750.0, defense) for defense in (50.0, 100.0, 250.0, 500.0, 1000.0)} <= specs
    assert max(off for off, _defense in specs) <= 750.0
    assert max(defense for _off, defense in specs) <= 1000.0


def test_rtss_side_routing_keeps_defense_separate() -> None:
    columns = [
        "z_rtss_touch_rate100",
        "z_rtss_pass_rate100",
        "z_rtss_def_contest_rate100",
        "z_rtss_drive_help_rate100",
        "z_rtss_pass_deflection_rate100",
    ]
    assert rtss_features_for_side(columns, "off") == columns[:2]
    assert rtss_features_for_side(columns, "def") == columns[2:]


def test_feature_families_are_mechanism_readable() -> None:
    assert rtss_feature_family("z_rtss_pick_pop_share", "off") == "screening_offball"
    assert rtss_feature_family("z_rtss_mid_qsq_epv", "off") == "shot_quality_and_talent"
    assert rtss_feature_family("z_rtss_def_contest_distance", "def") == "contest_geometry"
    assert rtss_feature_family("z_rtss_def_rb_opp_rate100", "def") == "rebounding_position"


def test_defense_target_is_positive_good(tmp_path) -> None:
    path = target_path(tmp_path, "off25_def50", 2023)
    path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "player_id": [1],
            "off": [2.0],
            "def": [-3.0],
            "off_poss": [500],
            "def_poss": [600],
        }
    ).to_csv(path, index=False)
    target = load_target_rows(tmp_path, "off25_def50", 2023, "def")
    assert target.loc[0, "target"] == 3.0
    assert target.loc[0, "target_poss"] == 600


def test_compact_selection_requires_cross_season_signal() -> None:
    rng = np.random.default_rng(7)
    rows_by_year = {}
    for year in (2023, 2024):
        signal = np.linspace(-2.0, 2.0, 60) + rng.normal(0.0, 0.05, 60)
        rows_by_year[year] = pd.DataFrame(
            {
                "target": 2.5 * signal + rng.normal(0.0, 0.1, 60),
                "target_poss": 1000.0,
                "z_rtss_signal": signal,
                "z_rtss_noise": rng.normal(size=60),
            }
        )
    selected, history = select_compact_rtss_features(
        rows_by_year,
        [2023, 2024],
        base_features=[],
        rtss_candidates=["z_rtss_signal", "z_rtss_noise"],
        max_features=2,
    )
    assert selected[0] == "z_rtss_signal"
    accepted = history.loc[history["accepted"].eq(1), "feature"].tolist()
    assert accepted[0] == "z_rtss_signal"
