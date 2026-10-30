from nba_pipeline.scripts.run_vpm_netcont_asof import parse_years, season_snapshots


def test_season_snapshot_dates_use_season_end_year_convention():
    snapshots = season_snapshots(2024)
    assert [snapshot.cutoff_date for snapshot in snapshots] == [
        "2023-10-01",
        "2024-01-01",
        "2024-05-01",
        "2024-08-30",
    ]
    assert [snapshot.load_end_year for snapshot in snapshots] == [2023, 2024, 2024, 2024]


def test_parse_years():
    assert parse_years("2020-2022,2026") == [2020, 2021, 2022, 2026]
