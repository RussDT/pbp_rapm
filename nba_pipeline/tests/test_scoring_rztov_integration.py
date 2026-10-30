"""Contract tests for the scoring-impact / rZTOV Databallr integration pieces.

Covers the scoring-board identity helpers and the one-column playtype-mix
join, the time-decay playtype-mix reader identities, the rZTOV page payload injection, the
reconciliation audit's page arithmetic, and the Supabase registry column
contracts for the four integration datasets. Lineup turnover join tests live in
``test_lineup_tov_board.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PIPELINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_DIR / "scripts"))

from audit_scoring_rztov_reconciliation import _page_constant, rztov_page  # noqa: E402
from build_historical_scoring_security import (  # noqa: E402
    attach_playtype_mix,
    identity_gaps,
    validate_identities,
)
from build_rztov_explorer import inject_payload  # noqa: E402
from build_td_playtype_mix_board import (  # noqa: E402
    DIFFICULTY_POINTS_PER_PP,
    season_on_court_fga,
    to_reader,
    validate,
)

INTEGRATION_SLUGS = (
    "historical_scoring_security_1997_2026",
    "td_playtype_mix_board_1997_2026",
    "lineup_tov_board_1997_2026",
    "nba_rztov_explorer_seasons_2014_2026",
)


def scoring_board() -> pd.DataFrame:
    board = pd.DataFrame(
        {
            "nba_id": [1, 2, 3],
            "year": [2014, 2020, 2026],
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


def mix_sidecar(path: Path) -> Path:
    share = np.array([0.30, 0.251])
    own = np.array([-1.0, 1.62])
    mate = np.array([0.4, 0.585])
    own_c, mate_c = share * own, (1 - share) * mate
    price = DIFFICULTY_POINTS_PER_PP * 85.0
    frame = pd.DataFrame(
        {
            "nba_id": [2, 3],
            "year": [2020, 2026],
            "mix_rapm": own_c + mate_c,
            "own_mix_contribution": own_c,
            "teammate_mix_contribution": mate_c,
            "td_shot_share": share,
            "history_seasons": [2, 2],
            "league_fga_per100": [85.0, 85.0],
            "own_mix_value100": -own_c * price,
            "teammate_mix_value100": -mate_c * price,
            "mix_value100": -(own_c + mate_c) * price,
        }
    )
    frame.to_csv(path, index=False)
    return path


def test_identity_gaps_are_zero_on_closed_board_and_flag_a_break() -> None:
    board = scoring_board()
    assert max(validate_identities(board).values()) <= 1e-12
    broken = board.copy()
    broken.loc[1, "scoring_security_value100"] += 0.01
    assert identity_gaps(broken)["composite_gap"] == pytest.approx(0.01)
    with pytest.raises(AssertionError):
        validate_identities(broken)


def test_playtype_mix_is_one_full_value_column_and_leaves_missing_blank(tmp_path: Path) -> None:
    path = mix_sidecar(tmp_path / "mix.csv")
    board = attach_playtype_mix(scoring_board(), path)
    added = set(board.columns) - set(scoring_board().columns)
    assert added == {"playtype_mix_rapm"}
    # The column is the full parent value (own + teammate halves), not a split.
    mix = pd.read_csv(path).set_index(["nba_id", "year"])
    late = board[board["year"].eq(2026)].iloc[0]
    assert late["playtype_mix_rapm"] == pytest.approx(mix.loc[(3, 2026), "mix_value100"])
    assert late["playtype_mix_rapm"] == pytest.approx(
        mix.loc[(3, 2026), "own_mix_value100"] + mix.loc[(3, 2026), "teammate_mix_value100"]
    )
    # A player-season without a mix solve stays missing, never zero.
    assert pd.isna(board[board["year"].eq(2014)].iloc[0]["playtype_mix_rapm"])
    # Scoring value and security are untouched by the join, and no mix identity is added.
    assert late["scoring_security_value100"] == pytest.approx(1.77 + 0.63 + 0.58)
    assert max(validate_identities(board).values()) <= 1e-12
    assert not any("mix" in name for name in identity_gaps(board))


def test_mix_sidecar_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = mix_sidecar(tmp_path / "mix.csv")
    frame = pd.read_csv(path)
    pd.concat([frame, frame.iloc[[0]]]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        attach_playtype_mix(scoring_board(), path)


def test_mix_reader_prices_and_closes_exactly() -> None:
    share = np.array([0.2, 0.5])
    own = np.array([1.0, -0.5])
    mate = np.array([0.3, 0.8])
    board = pd.DataFrame(
        {
            "player_id": [10, 11],
            "player_name": ["A", "B"],
            "feature_season": [2025, 2026],
            "td_overall_playtype_difficulty_rapm": share * own + (1 - share) * mate,
            "conditional_own_playtype_difficulty": own,
            "conditional_teammate_playtype_difficulty": mate,
            "own_mix_contribution": share * own,
            "teammate_mix_contribution": (1 - share) * mate,
            "td_shot_share": share,
            "weighted_offensive_fga_exposure": [2000.0, 3000.0],
            "def_playtype_difficulty_rapm": [0.0, 0.1],
            "history_seasons": [8, 8],
        }
    )
    rates = pd.DataFrame({"season": [2025, 2026], "league_fga_per100": [88.0, 86.0]})
    reader = to_reader(board, rates)
    assert max(validate(reader).values()) <= 1e-12
    # Positive difficulty (harder) is negative value (points lost vs an average diet).
    expected = -reader["mix_rapm"] * reader["league_fga_per100"] / 50.0
    np.testing.assert_allclose(reader["mix_value100"], expected)
    broken = reader.copy()
    broken.loc[0, "teammate_mix_contribution"] += 0.1
    with pytest.raises(AssertionError):
        validate(broken)


def test_season_on_court_fga_counts_focal_and_teammates() -> None:
    attempts = pd.DataFrame(
        {
            "Season": [2026, 2026, 2025],
            "Focal_Player": [1, 2, 1],
            "O1": [2, 1, 3],
            "O2": [3, 3, 4],
            "O3": [4, 4, 5],
            "O4": [5, 5, 6],
        }
    )
    out = season_on_court_fga(attempts).set_index(["year", "nba_id"])["season_on_court_fga"]
    assert out[(2026, 1)] == 2
    assert (2026, 6) not in out.index
    assert out[(2025, 6)] == 1
    assert out.sum() == 5 * len(attempts)


def test_inject_payload_replaces_exactly_one_line(tmp_path: Path) -> None:
    page = tmp_path / "page.html"
    page.write_text("<script>\nconst DATA = {\"old\": 1};\nconst OTHER = 2;\n</script>", encoding="utf-8")
    inject_payload(page, '{"new": 2}')
    text = page.read_text(encoding="utf-8")
    assert 'const DATA = {"new": 2};' in text and "old" not in text
    assert "const OTHER = 2;" in text
    assert _page_constant(text, "DATA") == {"new": 2}

    for body in ("<p>no payload</p>", "const DATA = 1;\nconst DATA = 2;"):
        page.write_text(body, encoding="utf-8")
        with pytest.raises(ValueError, match="exactly one"):
            inject_payload(page, "{}")


def test_page_constant_requires_the_named_line() -> None:
    assert _page_constant("  const PACKED = [1, 2];  \n", "PACKED") == [1, 2]
    with pytest.raises(ValueError):
        _page_constant("const DATA = 1;", "PACKED")


def test_rztov_page_arithmetic_matches_hand_computed_ledger(tmp_path: Path) -> None:
    be = {"2026": {"self": 0.10, "ast": 0.05, "potast": 0.08, "centre_sc": 0.2, "centre_bp": -0.1,
                   "price_sc": 0.535, "price_bp": 0.767}}
    # nba_id, year, poss, self, assisted, potential assists, sc TOV, bp TOV, qualified
    seasons = [[7, 2026, 2500, 20.0, 5.0, 10.0, 1.5, 1.0, 1]]
    page = tmp_path / "rztov.html"
    page.write_text(f"<script>\nconst DATA = {json.dumps({'be': be, 'seasons': seasons})};\n</script>", encoding="utf-8")

    allow_sc = 0.10 * 20 + 0.05 * 5 + 0.2
    allow_bp = 0.08 * 10 - 0.1
    excess = 2.5 - allow_sc - allow_bp
    csv = pd.DataFrame(
        {
            "nba_id": [7],
            "year": [2026],
            "rztov": [excess / (20 + 5 + 10 + 2.5) * 100],
            "tov_saved_100": [-excess],
            "tov_points_saved_100": [(allow_sc - 1.5) * 0.535 + (allow_bp - 1.0) * 0.767],
        }
    )
    results = rztov_page(page, csv)
    assert all(row["pass"] for row in results), results
    assert max(row["max_abs_gap"] for row in results) <= 1e-12

    off = csv.assign(rztov=csv["rztov"] + 0.01)
    assert not rztov_page(page, off)[0]["pass"]


def test_registry_columns_match_generated_files() -> None:
    registry = json.loads((PIPELINE_DIR / "config" / "supabase_publications.json").read_text())
    datasets = registry["datasets"] if isinstance(registry, dict) else registry
    by_slug = {entry["slug"]: entry for entry in datasets}
    for slug in INTEGRATION_SLUGS:
        entry = by_slug[slug]
        columns = entry["expected_columns"]
        assert len(columns) == len(set(columns)), slug
        assert {"nba_id", "year"} <= set(entry["row_key_fields"]) | set(columns)
        assert not any("genius" in column.lower() for column in columns), slug
        source = PIPELINE_DIR.parent / entry["source"]
        if not source.exists():
            continue  # generated artifact is gitignored; column contract still checked above
        header = list(pd.read_csv(source, nrows=0).columns)
        assert header == columns, slug
        assert len(pd.read_csv(source, usecols=["nba_id"])) >= entry["minimum_rows"], slug
