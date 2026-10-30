import pandas as pd

from wnba_test import wnba_stats_source
from wnba_test.wnba_stats_source import refresh_wnba_pbpstats_totals


def test_refresh_bypasses_stale_backend_cache(monkeypatch) -> None:
    expected = pd.DataFrame(
        {
            "PlayerID": ["1"],
            "nba_id": ["1"],
            "FTPerc": [0.8],
            "ThreePerc": [0.4],
            "source": ["pbpstats"],
        }
    )
    calls = []

    def fake_fetch(season_year, is_playoffs):
        calls.append((season_year, is_playoffs))
        return expected

    monkeypatch.setattr(wnba_stats_source, "_from_pbpstats", fake_fetch)
    wnba_stats_source._CACHE[(2026, False, "old")] = pd.DataFrame()

    refreshed = refresh_wnba_pbpstats_totals(2026, False)

    assert calls == [(2026, False)]
    assert refreshed.equals(expected)
    assert not any(key[:2] == (2026, False) for key in wnba_stats_source._CACHE)
