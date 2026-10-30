"""Regression checks for the dangerous repair/forecast boundaries."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from build_tracking_backfill import BOX, classify, features


def rows():
    d = pd.DataFrame({"PLAYER_ID": [1, 2], "TEAM_ID": [10, 10], "season": [2025]*2,
        "playoffs": [False]*2, "date": [20250101]*2, "MIN": [25., 25.], "AST": [1., 7.],
        "POTENTIAL_AST": [0., 0.], "PASSES_MADE": [20., 0.], "TOUCHES": [30., 70.],
        "overall_def_D_FGA": [5., np.nan], "overall_def_D_FGM": [2., 3.],
        "two_pt_def_FG2A": [3., 4.], "three_pt_def_FG3A": [2., 2.],
        "POTENTIAL_AST_alternate": [np.nan]*2, "overall_def_D_FGA_alternate": [np.nan]*2})
    return d


def test_assist_disagreement_alone_is_not_corruption():
    d = classify(rows())
    assert d.loc[0, "potential_ast_source"] == "observed"
    assert d.loc[0, "potential_ast_observed"] == 0
    assert d.loc[1, "potential_ast_reason"] == "zero_passing_with_activity"
    assert pd.isna(d.loc[1, "potential_ast_observed"])


def test_whole_team_outage_preserves_surviving_observations():
    d = rows()
    d.MIN = 120
    d.PASSES_MADE = [0., 3.]
    d.POTENTIAL_AST = [0., 2.]
    out = classify(d)
    assert out.passing_team_outage.all()
    assert pd.isna(out.loc[0, "potential_ast_observed"])
    assert out.loc[1, "potential_ast_observed"] == 2


def test_defense_recovery_requires_both_parts():
    d = rows()
    assert classify(d).loc[1, "dfga_observed"] == 6
    d.loc[1, "three_pt_def_FG3A"] = np.nan
    assert pd.isna(classify(d).loc[1, "dfga_observed"])


def test_defense_omitted_zero_requires_short_stint_and_live_team():
    d = pd.concat([rows().iloc[[0]]]*6, ignore_index=True)
    d.PLAYER_ID = np.arange(6)
    d.MIN = [47., 47., 47., 47., 47., 5.]
    d.loc[5, ["overall_def_D_FGA", "overall_def_D_FGM", "two_pt_def_FG2A", "three_pt_def_FG3A"]] = np.nan
    assert classify(d).loc[5, "dfga_source"] == "inferred_endpoint_zero"
    d.loc[5, "MIN"] = 15
    assert classify(d).loc[5, "dfga_source"] == "unresolved"
    d.loc[:, ["overall_def_D_FGA", "overall_def_D_FGM", "two_pt_def_FG2A", "three_pt_def_FG3A"]] = np.nan
    assert classify(d).dfga_reason.eq("team_defense_outage").all()


def test_alternate_recovers_only_flagged_rows():
    d = rows()
    d.POTENTIAL_AST_alternate = [10., 12.]
    out = classify(d)
    assert out.loc[0, "potential_ast_observed"] == 0
    assert out.loc[1, "potential_ast_observed"] == 12


def test_current_and_future_targets_cannot_change_current_history():
    d = pd.DataFrame({"PLAYER_ID": [1, 2, 1, 2, 1], "MIN": [20.]*5,
        "day": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02", "2020-01-03"]),
        "potential_ast_observed": [5., 7., 8., 10., 12.]})
    for c in BOX:
        d[c] = 1.
    x, _ = features(d, "potential_ast")
    d.loc[2:, "potential_ast_observed"] = [100., 200., 300.]
    y, _ = features(d, "potential_ast")
    np.testing.assert_allclose(x.loc[:3, "history_36"], y.loc[:3, "history_36"], equal_nan=True)


def test_missing_values_are_not_observations_in_history():
    d = pd.DataFrame({"PLAYER_ID": [1]*3, "MIN": [20.]*3,
        "day": pd.date_range("2020-01-01", periods=3), "potential_ast_observed": [5., np.nan, 7.]})
    for c in BOX:
        d[c] = 1.
    x, _ = features(d, "potential_ast")
    assert x.loc[2, "history_minutes"] == 20
