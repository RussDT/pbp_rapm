from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_historical_playtype_proxy_backcast.py"
SPEC = importlib.util.spec_from_file_location("historical_playtype_proxy_backcast", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

RAPM_SCRIPT = SCRIPT.with_name("rapm.py")
sys.path.insert(0, str(RAPM_SCRIPT.parent))
RAPM_SPEC = importlib.util.spec_from_file_location("historical_playtype_rapm", RAPM_SCRIPT)
assert RAPM_SPEC is not None and RAPM_SPEC.loader is not None
RAPM = importlib.util.module_from_spec(RAPM_SPEC)
RAPM_SPEC.loader.exec_module(RAPM)


def test_category_prices_are_shrunk_and_phase_centered_by_global_reference() -> None:
    category = pd.Series(["TRANSITION", "TRANSITION", "OTHER_2", "OTHER_2"])
    points = pd.Series([2.0, 2.0, 0.0, 0.0])
    result = MODULE.price_categories(category, points).set_index("mix_category")
    assert result.loc["TRANSITION", "playtype_proxy_points"] > 0
    assert result.loc["OTHER_2", "playtype_proxy_points"] < 0
    assert np.isclose(result["global_points_per_fga"].dropna().iloc[0], 1.0)


def test_duplicate_keys_must_not_conflict() -> None:
    frame = pd.DataFrame({"game_id": [1, 1], "event_num": [2, 2], "value": [3, 4]})
    try:
        MODULE.unique_key_frame(frame, label="test")
    except ValueError as exc:
        assert "conflicting" in str(exc)
    else:
        raise AssertionError("conflicting duplicate keys must fail")


def test_explicit_playtype_backcast_target_beats_inherited_helper_columns() -> None:
    frame = pd.DataFrame(
        {
            "Playtype_Proxy_PTS_Backcast": [0.25, -0.10],
            "Is_Rim_Attempt": [0, 1],
        }
    )
    prepared = RAPM.detect_file_type_and_prepare(
        frame, prefix="PLAYTYPE_PROXY_PTS_BACKCAST"
    )
    assert prepared["Off_Diff"].tolist() == [0.25, -0.10]
    assert prepared["Def_Diff"].tolist() == [-0.25, 0.10]
