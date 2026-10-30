import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from wnba_test.update_wnba_metrics import validate_current_board


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "wnba_test" / "update_wnba_metrics.py"


def test_daily_update_plan_exposes_frozen_model_and_validation_gates() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--as-of",
            "2026-07-29",
            "--plan-only",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["season"] == 2026
    assert plan["season_type"] == "Regular Season"
    assert plan["model_policy"] == "reuse_frozen_pre_2026_selection"
    assert plan["steps"] == [
        "stage_current_source",
        "fetch_completed_games",
        "validate_lineups",
        "refresh_box_totals",
        "rebuild_processed_surfaces",
        "rebuild_clean_decomposition",
        "build_current_box_prior_board",
        "validate_current_board",
        "promote_source_and_board",
    ]


def test_current_board_validation_enforces_additive_contract(tmp_path) -> None:
    board_path = tmp_path / "board.csv"
    pd.DataFrame(
        [
            {
                "player_id": "1",
                "player_name": "Player One",
                "oALT_EFG_RIM": 0.3,
                "oALT_EFG_RIM_FREQ": 0.1,
                "oALT_EFG_RIM_FG": 0.2,
                "dALT_EFG_RIM": -0.1,
                "dALT_EFG_RIM_FREQ": -0.2,
                "dALT_EFG_RIM_FG": 0.1,
                "oALT_EFG_MID": 0.0,
                "oALT_EFG_MID_FREQ": 0.0,
                "oALT_EFG_MID_FG": 0.0,
                "dALT_EFG_MID": 0.0,
                "dALT_EFG_MID_FREQ": 0.0,
                "dALT_EFG_MID_FG": 0.0,
                "oALT_EFG_THREE": 0.0,
                "oALT_EFG_THREE_FREQ": 0.0,
                "oALT_EFG_THREE_FG": 0.0,
                "dALT_EFG_THREE": 0.0,
                "dALT_EFG_THREE_FREQ": 0.0,
                "dALT_EFG_THREE_FG": 0.0,
                "oALT_FT": 0.4,
                "oALT_FT_FREQ": 0.3,
                "oALT_FT_SEVERITY": 0.1,
                "dALT_FT": -0.2,
                "dALT_FT_FREQ": -0.1,
                "dALT_FT_SEVERITY": -0.1,
                "oALT_TOV_VALUE": 0.3,
                "oALT_TOV_LOSS_VALUE": 0.3,
                "oALT_BADPASS_TOV_VALUE": 0.1,
                "oALT_SCORING_TOV_VALUE": 0.2,
                "dALT_TOV_VALUE": 0.1,
                "dALT_TOV_LOSS_VALUE": 0.1,
                "dALT_BADPASS_TOV_VALUE": 0.0,
                "dALT_SCORING_TOV_VALUE": 0.1,
                "off": 1.2,
                "def": 0.4,
                "net_rapm": 1.6,
                "prior_off": 0.7,
                "prior_def": 0.2,
                "prior_net_rapm": 0.9,
                "delta_off": 0.5,
                "delta_def": 0.2,
                "delta_net_rapm": 0.7,
                "possessions": 100,
            }
        ]
    ).to_csv(board_path, index=False)

    report = validate_current_board(board_path)

    assert report["players"] == 1
    assert report["max_identity_gap"] < 0.0021

    broken = pd.read_csv(board_path)
    broken.loc[0, "net_rapm"] = 2.0
    broken.to_csv(board_path, index=False)
    try:
        validate_current_board(board_path)
    except RuntimeError as exc:
        assert "additive identity" in str(exc)
    else:
        raise AssertionError("broken current board was accepted")
