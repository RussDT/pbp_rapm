"""Season, freeze and promotion rules for the daily Scoring Impact + rZTOV build."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PIPELINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_DIR / "scripts"))

import run_scoring_rztov_daily as daily  # noqa: E402
import scoring_season as seasons  # noqa: E402
from audit_scoring_rztov_reconciliation import board_vs_rztov_estimated  # noqa: E402
from build_historical_scoring_impact_html import load_live_seasons  # noqa: E402
from build_historical_scoring_security import (  # noqa: E402
    apply_role_sources,
    snapshot_skew_receipt,
)


def test_season_tags_and_env(monkeypatch) -> None:
    assert seasons.season_label(2027) == "2026-27"
    assert seasons.season_label(2000) == "1999-00"
    assert seasons.span_tag(1997, 2027) == "1997_2027"
    monkeypatch.delenv(seasons.SEASON_ENV, raising=False)
    assert seasons.current_season() == seasons.FALLBACK_SEASON
    monkeypatch.setenv(seasons.SEASON_ENV, "2028")
    assert seasons.current_season() == 2028


def test_fit_never_extends_past_the_freeze() -> None:
    assert seasons.fit_cap(2027) == seasons.FROZEN_FIT_SEASON
    assert seasons.fit_cap(2030) == seasons.FROZEN_FIT_SEASON
    assert seasons.fit_cap(2020) == 2020


def test_partial_season_floors_scale_with_schedule_progress() -> None:
    early = seasons.season_state(2027, 61)
    assert early["status"] == seasons.STATUS_IN_PROGRESS
    assert early["row_min_poss"] == seasons.PARTIAL_ROW_MIN_POSS
    assert early["qual_min_poss"] == seasons.PARTIAL_QUAL_MIN_POSS

    half = seasons.season_state(2027, 615)
    assert np.isclose(half["progress"], 0.5)
    assert np.isclose(half["row_min_poss"], 250.0)
    assert np.isclose(half["qual_min_poss"], 500.0)

    done = seasons.season_state(2027, seasons.FULL_SEASON_GAMES)
    assert done["status"] == seasons.STATUS_COMPLETE
    assert done["row_min_poss"] == seasons.ROW_MIN_POSS
    assert done["qual_min_poss"] == seasons.QUAL_MIN_POSS


def test_short_seasons_inside_the_fit_stay_complete() -> None:
    lockout = seasons.season_state(2012, 990)
    assert lockout["status"] == seasons.STATUS_COMPLETE
    assert lockout["progress"] == 1.0
    assert lockout["row_min_poss"] == seasons.ROW_MIN_POSS


def _scoring_row(year: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "nba_id": [7],
            "year": [year],
            "rts": [1.0],
            "tsa100": [20.0],
            "playtype_diff": [0.4],
            "scoring_context_source": ["core_merged_exact"],
        }
    )


def _empty_proxy() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "player_id", "feature_season", "pbp_fga", "pbp_difficulty_pp",
            "estimated_role", "pricing_status",
        ]
    )


def _snapshot(year: int) -> pd.DataFrame:
    return pd.DataFrame(
        {"nba_id": [7], "year": [year], "role": [1.5], "zts": [4.0], "rts": [2.5], "tsa100": [22.0]}
    )


def test_in_progress_season_takes_rts_tsa_and_role_from_one_snapshot() -> None:
    output = apply_role_sources(_scoring_row(2027), _empty_proxy(), _snapshot(2027), snapshot_after=2026)
    row = output.iloc[0]
    assert np.isclose(row["rts"], 2.5)
    assert np.isclose(row["tsa100"], 22.0)
    assert np.isclose(row["role_difficulty"], 1.5)
    assert np.isclose(row["zts"], row["rts"] + row["role_difficulty"])
    assert row["scoring_context_source"] == "databallr_zts_observed"
    assert np.isclose(row["local_rts"], 1.0)
    assert np.isclose(row["local_tsa100"], 20.0)

    skew = snapshot_skew_receipt(output, 2026)
    assert skew[0]["season"] == 2027
    assert skew[0]["rows_on_zts_snapshot"] == 1
    assert np.isclose(skew[0]["local_minus_snapshot_abs_rts_max"], 1.5)
    assert np.isclose(skew[0]["local_minus_snapshot_abs_tsa100_max"], 2.0)


def test_snapshot_rule_leaves_frozen_seasons_on_local_context() -> None:
    for kwargs in ({}, {"snapshot_after": 2026}):
        output = apply_role_sources(_scoring_row(2026), _empty_proxy(), _snapshot(2026), **kwargs)
        assert np.isclose(output.loc[0, "rts"], 1.0)
        assert np.isclose(output.loc[0, "tsa100"], 20.0)
        assert output.loc[0, "scoring_context_source"] == "core_merged_exact"


def test_estimated_rows_reconcile_and_catch_a_moved_price() -> None:
    rztov = pd.DataFrame(
        {
            "nba_id": [1, 2, 1],
            "year": [2027, 2027, 2026],
            "source": ["box_estimated", "box_estimated", "observed"],
            "rztov": [0.5, -0.25, 0.1],
            "sc_excess": [0.2, -0.1, 0.0],
            "bp_excess": [0.3, -0.15, 0.1],
            "sc_point_price": [0.9, 0.9, 0.9],
            "bp_point_price": [1.0, 1.0, 1.0],
            "tov_points_saved_100": [-0.48, 0.24, -0.1],
        }
    )
    board = pd.DataFrame(
        {
            "nba_id": [1, 2],
            "year": [2027, 2027],
            "turnover_source": ["box_estimated", "box_estimated"],
            "rztov": [0.5, -0.25],
            "sc_excess": [0.2, -0.1],
            "bp_excess": [0.3, -0.15],
            "scoring_tov_point_price": [0.9, 0.9],
            "badpass_tov_point_price": [1.0, 1.0],
            "turnover_value100": [-0.48, 0.24],
        }
    )
    checks = board_vs_rztov_estimated(board, rztov)
    assert len(checks) == 8
    assert all(item["pass"] for item in checks)

    moved = board.assign(scoring_tov_point_price=[0.9, 0.95])
    failed = {item["check"] for item in board_vs_rztov_estimated(moved, rztov) if not item["pass"]}
    assert failed == {"estimated_sc_price_board_vs_csv"}

    assert board_vs_rztov_estimated(board, rztov[rztov["source"].eq("observed")]) == []


def test_page_reads_live_floors_only_for_in_progress_seasons(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    assert load_live_seasons(manifest, {2027}) == {}
    manifest.write_text(
        json.dumps(
            {
                "coverage": {
                    "post_fit_seasons": {
                        "2027": seasons.season_state(2027, 247),
                        "2028": seasons.season_state(2028, seasons.FULL_SEASON_GAMES),
                    }
                }
            }
        )
    )
    live = load_live_seasons(manifest, {2027, 2028})
    assert set(live) == {"2027"}
    assert live["2027"]["label"] == "2026-27"
    assert live["2027"]["games"] == 247
    assert np.isclose(live["2027"]["qual"], 200.8, atol=0.05)


def _board(values: list[float], years: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {"nba_id": range(1, len(values) + 1), "year": years, "value": values, "name": "x"}
    )


def test_frozen_rows_check_ignores_new_seasons_and_new_columns() -> None:
    frozen = _board([1.0, np.nan], [2025, 2026])
    new = _board([1.0, np.nan, 9.0], [2025, 2026, 2027]).assign(season_status="complete")
    result = daily.compare_frozen_rows(new, frozen, 2026)
    assert result["ok"] and result["rows_daily"] == 2 and result["max_abs_gap"] == 0.0


@pytest.mark.parametrize(
    "new",
    [
        _board([1.0, 0.0], [2025, 2026]),  # a frozen blank became a zero
        _board([1.5, np.nan], [2025, 2026]),  # a frozen value moved
        _board([1.0], [2025]),  # a frozen row disappeared
        _board([1.0, np.nan], [2025, 2026]).drop(columns="value"),  # a frozen column disappeared
        _board([1.0, np.nan], [2025, 2026]).assign(name="y"),  # a frozen label changed
    ],
)
def test_frozen_rows_check_fails_on_any_change(new) -> None:
    assert not daily.compare_frozen_rows(new, _board([1.0, np.nan], [2025, 2026]), 2026)["ok"]


def test_chain_runs_children_with_this_interpreter_and_honours_flags(tmp_path) -> None:
    online = daily.step_commands(2027, 2026, tmp_path, offline=False, solves=False)
    names = [name for name, _ in online]
    assert names == [
        "transition", "box_backcast", "explorer", "playtype_mix",
        "lineup_tov", "scoring_board", "scoring_page", "audit",
    ]
    assert all(command[0] == sys.executable for _, command in online)
    commands = dict(online)
    assert "--refresh-anchor-from-databallr" in commands["scoring_board"]
    assert "--fetch-live" in commands["audit"]
    assert "--rebuild-calibration" not in commands["transition"]
    assert commands["lineup_tov"][commands["lineup_tov"].index("--first-season") + 1] == "2027"

    offline = dict(daily.step_commands(2027, 2026, tmp_path, offline=True, solves=False))
    assert "--refresh-anchor-from-databallr" not in offline["scoring_board"]
    assert "--fetch-live" not in offline["audit"]

    solved = daily.step_commands(2027, 2026, tmp_path, offline=False, solves=True)
    assert [name for name, _ in solved][:4] == [
        "count_solve_tov", "count_solve_scoring_tov", "count_solve_badpass_tov", "price_targets",
    ]
    assert solved[0][1][2:6] == ["TOV", "23", "27", "RS"]


def test_promotion_keeps_the_replaced_build_and_drops_logs(tmp_path) -> None:
    root = tmp_path / "2027"
    work = root / "work"
    (work / "scoring").mkdir(parents=True)
    (work / "logs").mkdir()
    (work / "logs" / "audit.log").write_text("log")

    (work / "scoring" / "board.csv").write_text("first")
    daily.promote(root, work)
    assert (root / "current" / "scoring" / "board.csv").read_text() == "first"
    assert not (root / "current" / "logs").exists()
    assert not (root / "previous").exists()

    (work / "scoring" / "board.csv").write_text("second")
    daily.promote(root, work)
    assert (root / "current" / "scoring" / "board.csv").read_text() == "second"
    assert (root / "previous" / "scoring" / "board.csv").read_text() == "first"
    assert not (root / "current.new").exists()


def test_runner_refuses_a_season_inside_the_frozen_fit() -> None:
    with pytest.raises(SystemExit, match="frozen fit"):
        daily.main(["--season", str(seasons.FROZEN_FIT_SEASON)])
