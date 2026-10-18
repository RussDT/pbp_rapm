import importlib.util
from pathlib import Path

import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location("matchup_bpm", Path(__file__).parents[1] / "scripts/fetch_matchup_bpm.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def row(team, bpm="2.0"):
    cells = dict(name_display="Exact Player", team_name_abbr=team, pos="PG", mp="100", games="10", bpm=bpm, obpm="3.0", dbpm="-1.0")
    return "<tr>" + "".join(f'<td data-stat="{key}" data-append-csv="exact01">{value}</td>' for key, value in cells.items()) + "</tr>"


def test_regular_season_only_and_whole_season_trade_total():
    html = '<table id="advanced">' + row("2TM") + row("AAA", "5.0") + row("BBB", "-1.0") + '</table><table id="advanced_post">' + row("AAA", "99.0") + '</table>'
    got = m.parse_advanced(html, 2025)
    assert len(got) == 1 and got.bpm.iloc[0] == 2
    assert got.team.iloc[0] == "2TM" and got.position.iloc[0] == "PG"


def test_ambiguous_trade_rows_fail_instead_of_averaging_bpm():
    with pytest.raises(ValueError, match="aggregate"):
        m.parse_advanced('<table id="advanced">' + row("AAA") + row("BBB") + '</table>', 2025)


def test_exact_crosswalk_rejects_ambiguity_and_leaves_unmatched_missing():
    stats = m.parse_advanced('<table id="advanced">' + row("AAA") + '</table>', 2025)
    index = pd.DataFrame(dict(year=[2025], bref_id=["other01"], nba_id=[12]))
    good, bad = m.exact_id_join(stats, index)
    assert good.empty and len(bad) == 1
    index = pd.DataFrame(dict(year=[2025, 2025], bref_id=["exact01", "exact01"], nba_id=[12, 13]))
    with pytest.raises(ValueError, match="Ambiguous"):
        m.exact_id_join(stats, index)


def test_protected_season_rejected_before_fetch():
    with pytest.raises(ValueError, match="registered"):
        m.fetch(years=[2027])
