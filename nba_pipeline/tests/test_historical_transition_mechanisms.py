from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_historical_transition_mechanisms.py"
SPEC = importlib.util.spec_from_file_location("historical_transition_mechanisms", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def base_rows() -> pd.DataFrame:
    return pd.DataFrame({"game_id": [1, 2], "event_num": [10, 20], "O1": [1, 2]})


def label_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["0000000001", "0000000002"],
            "event_num": [10, 20],
            "transition_label_operating": [1, 0],
            "transition_rim_label_operating": [1, 0],
            "transition_probability": [0.8, 0.1],
            "transition_rim_probability": [0.7, 0.05],
            "transition_calibration_scope": ["retro", "retro"],
            "transition_rim_calibration_scope": ["retro", "retro"],
            "feature_contract_version": ["v1", "v1"],
        }
    )


def test_attach_backcast_preserves_rows_and_labels() -> None:
    result = MODULE.attach_backcast(base_rows(), label_rows(), year=2020, phase="rs")
    assert len(result) == 2
    assert result["Is_Transition_Backcast"].tolist() == [1, 0]
    assert result["Is_Transition_Rim_Backcast"].tolist() == [1, 0]
    assert result["target_provenance"].nunique() == 1


def test_attach_backcast_fails_closed_on_missing_fga() -> None:
    with pytest.raises(ValueError, match="lack transition backcasts"):
        MODULE.attach_backcast(base_rows(), label_rows().iloc[:1], year=2020, phase="rs")

