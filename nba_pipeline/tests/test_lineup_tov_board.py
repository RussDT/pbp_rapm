"""Contract tests for the two-season TD550 lineup turnover board and its
scoring-board join.

Covers the focal own-turnover contrast normal equations against an explicit
five-row design, the board's split identities, the time-decay weights, and the
scoring-board pricing, blank-not-zero join, closure check, and pass-through.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PIPELINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_DIR / "scripts"))

from build_historical_scoring_security import (  # noqa: E402
    LINEUP_TOV_COLUMNS,
    attach_lineup_tov,
    identity_gaps,
    lineup_pass_through,
    lineup_tov_gaps,
    validate_identities,
)
from build_lineup_tov_board import contrast_stats, design, td_weights, validate  # noqa: E402


def possessions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "O1": [1, 1, 2],
            "O2": [2, 3, 3],
            "O3": [3, 4, 4],
            "O4": [4, 5, 5],
            "O5": [5, 6, 6],
            "D1": [7, 7, 8],
            "D2": [8, 8, 9],
            "D3": [9, 9, 10],
            "D4": [10, 10, 11],
            "D5": [11, 11, 12],
            "scoring_att": [1, 0, 1],
            "actor": [2, 0, 6],
        }
    )


def test_contrast_normal_equations_match_explicit_five_row_design() -> None:
    frame = possessions()
    off, _de, ids = design(frame)
    weight = np.array([1.0, 0.5, 0.8])
    gram, rhs, _ss = contrast_stats(frame, off, ids, weight, "scoring")

    # Explicit design: one row per focal offensive player, x centred within the
    # possession (focal indicator minus one fifth of the lineup), target
    # -I(focal committed the attributed turnover).
    x_rows, y_rows, w_rows = [], [], []
    lineups = frame[["O1", "O2", "O3", "O4", "O5"]].to_numpy()
    for i, lineup in enumerate(lineups):
        share = np.isin(ids, lineup) / 5.0
        for player in lineup:
            x_rows.append((ids == player).astype(float) - share)
            y_rows.append(-float(frame["scoring_att"].iat[i] == 1 and frame["actor"].iat[i] == player))
            w_rows.append(weight[i])
    x, y, w = np.array(x_rows), np.array(y_rows), np.array(w_rows)
    np.testing.assert_allclose(gram, x.T @ (x * w[:, None]), atol=1e-12)
    np.testing.assert_allclose(rhs, x.T @ (w * y), atol=1e-12)


def test_time_decay_halves_at_the_half_life() -> None:
    weights = td_weights(np.array([0.0, 550.0, 1100.0]), 1100.0, 550.0)
    np.testing.assert_allclose(weights, [0.25, 0.5, 1.0])


def lineup_source(path: Path) -> Path:
    rows = {"nba_id": [2, 3], "year": [2020, 2026], "history_seasons": [2, 2], "team_tov_coverage": ["full", "full"]}
    frame = pd.DataFrame(rows)
    rng = np.random.default_rng(0)
    for lane in ("scoring", "badpass"):
        for prefix in (lane, f"{lane}_raw"):
            parts = {part: rng.normal(0, 0.5, 2) for part in ("own", "teammate", "unassigned")}
            for part, value in parts.items():
                frame[f"{prefix}_{part}_saved100"] = value
            frame[f"{prefix}_total_saved100"] = sum(parts.values())
    frame["lineup_tov_saved100"] = frame["scoring_total_saved100"] + frame["badpass_total_saved100"]
    frame["tov_point_price"] = [1.25, 1.30]
    for lane in ("scoring", "badpass"):
        frame[f"{lane}_total_value100"] = frame[f"{lane}_total_saved100"] * frame["tov_point_price"]
    frame["lineup_tov_value100"] = frame["scoring_total_value100"] + frame["badpass_total_value100"]
    frame.to_csv(path, index=False)
    return path


def scoring_board() -> pd.DataFrame:
    board = pd.DataFrame(
        {
            "nba_id": [1, 2, 3],
            "year": [2014, 2020, 2026],
            "OffPoss": [1500.0, 2000.0, 2500.0],
            "qualified_scoring_security": [1, 1, 1],
            "rts": [1.0, -2.0, 1.446],
            "role_difficulty": [0.5, 1.0, 2.399],
            "scoring_value100": [0.4, -0.6, 1.77],
            "scoring_tov_value100": [0.1, -0.2, 0.63],
            "badpass_tov_value100": [-0.05, 0.3, 0.58],
        }
    )
    board["zts"] = board["rts"] + board["role_difficulty"]
    board["turnover_value100"] = board["scoring_tov_value100"] + board["badpass_tov_value100"]
    board["scoring_impact_no_badpass100"] = board["scoring_value100"] + board["scoring_tov_value100"]
    board["scoring_security_value100"] = board["scoring_value100"] + board["turnover_value100"]
    return board


def test_lineup_board_validate_flags_a_broken_split(tmp_path: Path) -> None:
    frame = pd.read_csv(lineup_source(tmp_path / "lineup.csv"))
    assert max(validate(frame).values()) <= 1e-12
    frame.loc[0, "badpass_teammate_saved100"] += 0.01
    with pytest.raises(AssertionError):
        validate(frame)


def test_lineup_join_prices_every_piece_and_leaves_missing_blank(tmp_path: Path) -> None:
    path = lineup_source(tmp_path / "lineup.csv")
    board = attach_lineup_tov(scoring_board(), path)
    assert set(board.columns) - set(scoring_board().columns) == set(LINEUP_TOV_COLUMNS)
    source = pd.read_csv(path).set_index(["nba_id", "year"])
    late = board[board["year"].eq(2026)].iloc[0]
    price = source.loc[(3, 2026), "tov_point_price"]
    assert late["lineup_scoring_tov_own_value100"] == pytest.approx(source.loc[(3, 2026), "scoring_own_saved100"] * price)
    assert late["lineup_badpass_tov_team_value100"] == pytest.approx(
        source.loc[(3, 2026), "badpass_unassigned_saved100"] * price
    )
    assert late["lineup_scoring_tov_raw_value100"] == pytest.approx(
        source.loc[(3, 2026), "scoring_raw_total_saved100"] * price
    )
    # No lineup solve: every lineup column stays missing, never zero.
    early = board[board["year"].eq(2014)].iloc[0]
    assert early[list(LINEUP_TOV_COLUMNS)].isna().all()
    # The ledger is untouched and gains no lineup identity.
    assert max(validate_identities(board).values()) <= 1e-12
    assert not any("lineup" in name for name in identity_gaps(board))
    assert max(lineup_tov_gaps(board).values()) <= 1e-12
    broken = board.copy()
    broken.loc[broken["year"].eq(2026), "lineup_scoring_tov_teammate_value100"] += 0.02
    assert lineup_tov_gaps(broken)["lineup_scoring_split_gap"] == pytest.approx(0.02)


def test_lineup_join_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = lineup_source(tmp_path / "lineup.csv")
    frame = pd.read_csv(path)
    pd.concat([frame, frame.iloc[[0]]]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        attach_lineup_tov(scoring_board(), path)


def test_pass_through_slope_is_one_when_lineup_equals_box() -> None:
    rng = np.random.default_rng(1)
    n = 40
    box_sc, box_bp = rng.normal(0, 0.5, n), rng.normal(0, 0.5, n)
    board = pd.DataFrame(
        {
            "year": 2026,
            "OffPoss": rng.uniform(1000, 3000, n),
            "qualified_scoring_security": 1,
            "scoring_tov_value100": box_sc,
            "badpass_tov_value100": box_bp,
            "turnover_value100": box_sc + box_bp,
            "lineup_team_tov_coverage": "full",
        }
    )
    for lane, box in (("scoring", box_sc), ("badpass", box_bp)):
        board[f"lineup_{lane}_tov_own_value100"] = box
        board[f"lineup_{lane}_tov_total_value100"] = 0.5 * box + 0.1
        board[f"lineup_{lane}_tov_raw_value100"] = 2.0 * box
    board["lineup_tov_value100"] = board["lineup_scoring_tov_total_value100"] + board["lineup_badpass_tov_total_value100"]
    out = lineup_pass_through(board).iloc[0]
    assert out["scoring_own_slope"] == pytest.approx(1.0)
    assert out["badpass_total_slope"] == pytest.approx(0.5)
    assert out["scoring_raw_slope"] == pytest.approx(2.0)
    assert out["turnover_total_slope"] == pytest.approx(0.5)
    assert out["scoring_own_corr"] == pytest.approx(1.0)
    assert out["team_tov_coverage"] == "full"
