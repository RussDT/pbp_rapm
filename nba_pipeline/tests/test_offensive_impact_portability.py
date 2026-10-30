from pathlib import Path
import sys

import numpy as np
import pandas as pd


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from analyze_offensive_impact_portability import (  # noqa: E402
    EpmStore,
    classify_role,
    display_oepm_tier,
    full_oepm_tier,
    playoff_game_number,
    playoff_round,
    playoff_series_id,
    weighted_demean,
    weighted_quantile,
)


def test_oepm_tiers_match_declared_boundaries():
    values = pd.Series([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    assert full_oepm_tier(values).tolist() == [
        "lt_-2", "-2_-1", "-1_0", "0_1", "1_2", "2_3", "3_4", "4_plus"
    ]
    display = display_oepm_tier(pd.Series([-0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0]))
    assert display.tolist() == ["avg_0", "avg_0", "avg_0", "1_2", "2_3", "3_4", "4_plus"]


def test_playoff_game_id_decoding():
    game_ids = pd.Series([42400117, 42400224, 42400315, 42400407])
    assert playoff_round(game_ids).tolist() == [
        "First round", "Second round", "Conference finals", "Finals"
    ]
    assert playoff_game_number(game_ids).tolist() == [7, 4, 5, 7]
    assert playoff_series_id(game_ids).tolist() == ["4240011", "4240022", "4240031", "4240040"]


def test_weighted_quantile_and_demean():
    assert np.allclose(weighted_quantile([1, 2, 3], [0.25, 0.5, 0.75]), [1.25, 2.0, 2.75])
    frame = pd.DataFrame({
        "group": ["a", "a", "b", "b"],
        "weight": [1.0, 3.0, 2.0, 2.0],
        "value": [0.0, 4.0, 2.0, 6.0],
    })
    within = weighted_demean(frame, ["value"], "group", "weight")
    for group, rows in frame.groupby("group").groups.items():
        assert abs(np.average(within.loc[rows, "value"], weights=frame.loc[rows, "weight"])) < 1e-12


def test_epm_store_uses_latest_snapshot_on_or_before_date():
    first = pd.DataFrame({"oepm": [1.0], "depm": [0.0]}, index=pd.Index([1], name="player_id"))
    second = pd.DataFrame({"oepm": [2.0], "depm": [0.0]}, index=pd.Index([1], name="player_id"))
    dates = np.array([pd.Timestamp("2025-12-01"), pd.Timestamp("2026-01-01")], dtype="datetime64[ns]")
    store = EpmStore(
        snapshots={pd.Timestamp("2025-12-01"): first, pd.Timestamp("2026-01-01"): second},
        dates=dates,
        source_files=[],
    )
    assert store.latest_date("2025-12-01") == pd.Timestamp("2025-12-01")
    assert store.latest_date("2025-12-31") == pd.Timestamp("2025-12-01")
    assert store.asof("2026-01-01").loc[1, "oepm"] == 2.0


def test_role_classification_is_mutually_exclusive():
    frame = pd.DataFrame({
        "p_usg": [0.17, 0.28, 0.19, 0.22, 0.18],
        "p_ast_100": [3.0, 5.0, 2.0, 6.0, 3.0],
        "p_fga_rim_100": [3.0, 6.0, 10.0, 5.0, 4.0],
        "p_fg3a_100": [8.0, 5.0, 1.0, 4.0, 3.0],
        "p_orb_100": [0.5, 1.0, 4.0, 1.0, 1.0],
    })
    assert classify_role(frame).tolist() == [
        "low_usage_shooter", "creator", "big_finisher", "secondary_creator", "connector_other"
    ]
