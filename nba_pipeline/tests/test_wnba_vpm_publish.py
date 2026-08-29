import pandas as pd
import pytest

from wnba_test.prepare_wnba_vpm_publish import (
    build_supabase_stage_rows,
    reallocate_completion_baseline_to_shot_value,
)


def _row() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "off": [4.0],
            "def": [2.0],
            "net_rapm": [6.0],
            "oALT_EFG_RIM": [0.5],
            "oALT_EFG_MID_FREQ": [0.1],
            "oALT_EFG_MID_FG": [0.4],
            "oALT_EFG_THREE": [0.5],
            "oALT_EFG_VALUE": [1.5],
            "oALT_FT": [0.5],
            "oALT_TOV_VALUE": [1.0],
            "oALT_TOV_LOSS_VALUE": [0.25],
            "oALT_BADPASS_TOV_VALUE": [0.10],
            "oALT_SCORING_TOV_VALUE": [0.15],
            "oALT_TOV_VALUE_split": [0.25],
            "oSC": [1.0],
            "oFC": [3.0],
            "oDISPLAY_SUM": [4.0],
            "oRESID": [0.0],
            "dALT_EFG_RIM": [0.6],
            "dALT_EFG_MID_FREQ": [0.2],
            "dALT_EFG_MID_FG": [0.2],
            "dALT_EFG_THREE": [0.4],
            "dALT_EFG_VALUE": [1.4],
            "dALT_FT": [0.2],
            "dALT_TOV_VALUE": [-0.2],
            "dALT_TOV_LOSS_VALUE": [0.3],
            "dALT_BADPASS_TOV_VALUE": [0.1],
            "dALT_SCORING_TOV_VALUE": [0.2],
            "dALT_TOV_VALUE_split": [0.3],
            "dSC": [0.6],
            "dFC": [1.4],
            "dDISPLAY_SUM": [2.0],
            "dRESID": [0.0],
            "nALT_EFG_VALUE": [2.9],
            "nALT_TOV_VALUE": [0.8],
            "nDISPLAY_SUM": [6.0],
            "RESID": [0.0],
        }
    )


def test_direct_turnover_reallocation_preserves_headline_impact_and_closure():
    before = _row()
    after = reallocate_completion_baseline_to_shot_value(before)

    assert after.loc[0, "oALT_TOV_VALUE"] == pytest.approx(0.25)
    assert after.loc[0, "dALT_TOV_VALUE"] == pytest.approx(0.30)
    assert after.loc[0, "oALT_EFG_MID_FG"] == pytest.approx(1.15)
    assert after.loc[0, "dALT_EFG_MID_FG"] == pytest.approx(-0.30)
    assert after.loc[0, "oALT_EFG_VALUE"] == pytest.approx(2.25)
    assert after.loc[0, "dALT_EFG_VALUE"] == pytest.approx(0.90)
    assert after.loc[0, "oDISPLAY_SUM"] == pytest.approx(4.0)
    assert after.loc[0, "dDISPLAY_SUM"] == pytest.approx(2.0)
    assert after.loc[0, "nDISPLAY_SUM"] == pytest.approx(6.0)
    assert after.loc[0, "RESID"] == pytest.approx(0.0)
    assert after.loc[0, "off"] == before.loc[0, "off"]
    assert after.loc[0, "def"] == before.loc[0, "def"]
    assert after.loc[0, "net_rapm"] == before.loc[0, "net_rapm"]


def test_direct_turnover_reallocation_is_idempotent():
    once = reallocate_completion_baseline_to_shot_value(_row())
    twice = reallocate_completion_baseline_to_shot_value(once)
    pd.testing.assert_frame_equal(once, twice)


def test_current_aja_row_uses_positive_direct_turnover_value_when_available():
    path = "wnba_test/Results/current/wnba_current_metrics.csv"
    try:
        board = pd.read_csv(path)
    except FileNotFoundError:
        pytest.skip("current WNBA board is not available")
    row = board.loc[board["player_id"].astype(str) == "1628932"]
    if row.empty:
        pytest.skip("A'ja Wilson is not present in the current WNBA board")

    corrected = reallocate_completion_baseline_to_shot_value(row)
    assert corrected.iloc[0]["dALT_TOV_VALUE"] == pytest.approx(
        row.iloc[0]["dALT_TOV_LOSS_VALUE"]
    )
    assert corrected.iloc[0]["dALT_TOV_VALUE"] > 0.0
    assert corrected.iloc[0]["def"] == pytest.approx(row.iloc[0]["def"])
    assert corrected.iloc[0]["net_rapm"] == pytest.approx(row.iloc[0]["net_rapm"])


def test_supabase_stage_rows_preserve_csv_payload_strings(tmp_path):
    current = tmp_path / "current.csv"
    history = tmp_path / "history.csv"
    columns = [
        "season",
        "season_type",
        "player_id",
        "player_name",
        "team",
        "net_rapm",
        "possessions",
        "qualified",
    ]
    pd.DataFrame(
        [["2026", "RS", "1628932", "A'ja Wilson", "LVA", "7.716", "4903", "True"]],
        columns=columns,
    ).to_csv(current, index=False)
    pd.DataFrame(
        [["2026", "RS", "1628932", "A'ja Wilson", "", "7.716", "4903", "True"]],
        columns=columns,
    ).to_csv(history, index=False)

    rows = build_supabase_stage_rows(current, history, "batch-1")

    assert [row["dataset_scope"] for row in rows] == ["current", "history"]
    assert rows[0]["player_id"] == "1628932"
    assert rows[0]["net_rapm"] == pytest.approx(7.716)
    assert rows[0]["qualified"] is True
    assert rows[0]["payload"]["net_rapm"] == "7.716"
    assert rows[1]["team"] is None


def test_supabase_stage_rows_reject_duplicate_player_seasons(tmp_path):
    current = tmp_path / "current.csv"
    history = tmp_path / "history.csv"
    row = {
        "season": "2026",
        "season_type": "RS",
        "player_id": "1628932",
        "player_name": "A'ja Wilson",
        "team": "LVA",
        "net_rapm": "7.716",
        "possessions": "4903",
        "qualified": "True",
    }
    pd.DataFrame([row, row]).to_csv(current, index=False)
    pd.DataFrame([row]).to_csv(history, index=False)

    with pytest.raises(RuntimeError, match="duplicate player-season"):
        build_supabase_stage_rows(current, history, "batch-1")
