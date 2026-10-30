import pandas as pd

from nba_pipeline.scripts import run_quarterly_td_rapm as runner


def test_quarterly_snapshots_use_game_count_quartiles(monkeypatch):
    games = pd.DataFrame(
        {
            "game_id": [str(index) for index in range(10)],
            "game_date": pd.to_datetime(
                ["2025-10-01", "2025-10-02", "2025-10-02", "2025-10-04", "2025-10-05",
                 "2025-10-06", "2025-10-07", "2025-10-08", "2025-10-09", "2025-10-10"]
            ),
        }
    )
    monkeypatch.setattr(runner, "season_game_table", lambda _year: games)
    snapshots = runner.build_snapshots(2026)
    assert [snapshot.cutoff_date for snapshot in snapshots] == [
        "2025-10-02", "2025-10-05", "2025-10-08", "2025-10-10"
    ]
    assert [snapshot.games_included for snapshot in snapshots] == [3, 5, 8, 10]
    assert all(snapshot.start_year == 2021 for snapshot in snapshots)


def test_default_range_is_every_complete_six_season_window():
    years = runner.parse_end_years("")
    assert years[0] == 2002
    assert years[-1] == 2026
    assert len(years) == 25


def test_expected_artifacts_include_result_rb_and_se_files(tmp_path):
    snapshot = runner.Snapshot(2026, 4, "2026-06-13", 1320, 1320)
    names = [path.name for path in runner.expected_artifacts(snapshot, tmp_path)]
    assert names == [
        "rapm_21_26_all_pure_rb_se_td550_results.csv",
        "rapm_21_26_all_pure_rb_se_td550_rubberband_coefficients.csv",
        "rapm_21_26_all_pure_rb_se_td550_rubberband_effects.csv",
        "rapm_21_26_all_pure_rb_se_td550_season_effects.csv",
    ]
