"""Source-clean fit admission and annual prefix boundaries."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_pregame_source_clean_skills import measurement_panel, load_panel


def test_masked_counts_never_become_zero_fit_observations():
    frame = pd.DataFrame(dict(player_id=[1]*6, game_id=list("abcdef"),
        game_date=pd.date_range("2015-01-01", periods=6), season=[2015]*6,
        makes=[np.nan, 0., 1., 3., np.inf, 1.], attempts=[2., 2., 2., 2., 2., np.inf]))
    result = measurement_panel(frame, SimpleNamespace(numer="makes", denom="attempts", kind="pct"))
    assert result.game_id.tolist() == ["b", "c"]
    assert result.obs_rate.tolist() == [0., .5]
    assert result.step_k.tolist() == [0, 1]


def test_source_loader_keeps_queries_and_excludes_current_cut_and_future(tmp_path):
    frame = pd.DataFrame(dict(player_id=[1,1,1], game_id=["a","b","c"],
        game_date=pd.to_datetime(["2015-09-30", "2015-10-01", "2016-01-01"]),
        season=[2015,2016,2016], off_poss=[np.nan,40.,40.], def_poss=[np.nan,40.,40.],
        fga=[np.nan,10.,10.], fta=[np.nan,2.,2.], tov=[np.nan,1.,1.]))
    ctx = pd.DataFrame(dict(player_id=[1,1,1],game_id=["a","b","c"],
        is_home=pd.array([True,False,None],dtype="boolean"),team_id=[1,1,1],opp_team_id=[2,2,2]))
    frame.to_parquet(tmp_path / "player_game_lines.parquet")
    ctx.to_parquet(tmp_path / "game_context.parquet")
    baseline, context = load_panel(tmp_path, pd.Timestamp("2015-10-01"))
    frame.loc[1:, ["fga", "fta", "tov"]] = 999999.
    frame.to_parquet(tmp_path / "player_game_lines.parquet")
    altered, _ = load_panel(tmp_path, pd.Timestamp("2015-10-01"))
    pd.testing.assert_frame_equal(baseline, altered)
    assert baseline.game_id.tolist() == ["a"]
    assert context.game_id.tolist() == ["a"]
    assert np.isnan(baseline.usg_events.iloc[0])
    assert baseline.is_home.iloc[0] == 1.
