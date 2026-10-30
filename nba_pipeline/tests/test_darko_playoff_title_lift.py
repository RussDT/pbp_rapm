from __future__ import annotations

import pandas as pd

from nba_pipeline.scripts.build_darko_playoff_title_lift import (
    build_lagged_darko_snapshot_history,
    load_player_sheets_2026_salary,
)
from nba_pipeline.scripts.build_championship_help import select_epm_snapshot


def test_lagged_darko_snapshot_uses_only_previous_completed_season() -> None:
    darko = pd.DataFrame(
        [
            {
                "nba_id": 1,
                "player_name": "One",
                "season": 2008,
                "date": "2008-06-01",
                "dpm": 4.0,
                "o_dpm": 3.0,
                "d_dpm": 1.0,
            },
            {
                "nba_id": 1,
                "player_name": "One",
                "season": 2009,
                "date": "2009-06-01",
                "dpm": 9.0,
                "o_dpm": 8.0,
                "d_dpm": 1.0,
            },
        ]
    )

    history, report = build_lagged_darko_snapshot_history(
        darko, start_season=2009, end_season=2010
    )
    entering_2009 = select_epm_snapshot(history, "2009-04-15", season=2009)

    assert entering_2009.loc[0, "dpm"] == 4.0
    assert history.loc[
        history["snapshot_requested_date"].eq(pd.Timestamp("2008-07-01")),
        "source_darko_season",
    ].eq(2008).all()
    assert report["lag_seasons"] == 1
    assert report["start_target_season"] == 2009


def test_lagged_darko_snapshot_never_uses_same_season_row() -> None:
    darko = pd.DataFrame(
        [
            {
                "nba_id": 1,
                "player_name": "One",
                "season": 2009,
                "date": "2009-06-01",
                "dpm": 9.0,
                "o_dpm": 8.0,
                "d_dpm": 1.0,
            }
        ]
    )

    history, _ = build_lagged_darko_snapshot_history(
        darko, start_season=2009, end_season=2009
    )

    assert history.empty


def test_player_sheets_salary_prefers_native_cap_share_and_falls_back(tmp_path) -> None:
    path = tmp_path / "modern_selected_deduped.csv"
    pd.DataFrame(
        [
            {
                "nba_id": 1,
                "PLAYER_NAME": "Native",
                "year": 2026,
                "salary": 30.0,
                "SALARY_CAP": 100.0,
                "SALARY_PCT_CAP": 0.31,
            },
            {
                "nba_id": 2,
                "PLAYER_NAME": "Fallback",
                "year": 2026,
                "salary": 20.0,
                "SALARY_CAP": 100.0,
                "SALARY_PCT_CAP": None,
            },
        ]
    ).to_csv(path, index=False)

    salary, report = load_player_sheets_2026_salary(path)

    assert salary.set_index("player_id")["salary_share"].to_dict() == {1: 0.31, 2: 0.20}
    assert report["native_cap_share_rows"] == 1
    assert report["fallback_cap_share_rows"] == 1
