from __future__ import annotations

import pytest

from nba_pipeline.scripts.fetch_monthly_epm_title import normalize_title_snapshot


def _row(team_id: int, epm: float = 3.0) -> dict[str, object]:
    return {
        "date": "2005-03-01",
        "game_dt": "2005-03-01",
        "game_optimized": 0,
        "player_id": 2206,
        "player_name": "Rodney White",
        "team_id": team_id,
        "oepm": 2.0,
        "depm": epm - 2.0,
        "epm": epm,
    }


def test_identical_epm_trade_stints_collapse_to_one_player() -> None:
    frame, duplicate_rows = normalize_title_snapshot(
        [_row(1610612743), _row(1610612744)], "2005-03-01"
    )

    assert len(frame) == 1
    assert duplicate_rows == 2
    assert frame.loc[0, "epm"] == 3.0


def test_conflicting_epm_trade_stints_fail_loudly() -> None:
    with pytest.raises(ValueError, match="Conflicting EPM duplicate"):
        normalize_title_snapshot(
            [_row(1610612743), _row(1610612744, epm=4.0)], "2005-03-01"
        )
