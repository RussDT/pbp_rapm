"""Checks for paired evaluation and the frozen FT predictive diagnostic."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import summarize_full_skill_history as summary


def test_supervisor_waits_for_finished_parquet_footer(tmp_path):
    from finish_full_skill_history import replay_ready
    (tmp_path / "baseline").mkdir()
    for year in range(1997, 2027):
        (tmp_path / "baseline" / f"{year}.parquet").write_bytes(b"PAR1not_finished")
    assert not replay_ready(tmp_path)
    for year in range(1997, 2027):
        (tmp_path / "baseline" / f"{year}.parquet").write_bytes(b"PAR1finishedPAR1")
    assert replay_ready(tmp_path)


def test_paired_losses_drop_missing_candidate_and_resample_whole_games():
    totals = summary.paired_game_totals([1, 1, 2, 3], [0., 0., 0., 0.],
                                       [1., 3., 2., 4.], [1., 1., 1., 1.], [0., 0., .5, np.nan])
    assert totals.index.tolist() == [1, 2]
    assert totals.weight.tolist() == [4., 2.]
    result = summary.compare_seasons({2020: totals}, draws=100)
    assert result["baseline_mse"] == 1.
    assert result["candidate_mse"] == pytest.approx(.5 / 6)
    assert result["conditional_game_bootstrap_95"][0] > 0


def test_equal_season_evaluation_does_not_let_exposure_dominate():
    a = summary.paired_game_totals([1], [0.], [1000.], [2.], [1.])
    b = summary.paired_game_totals([2], [0.], [1.], [1.], [2.])
    result = summary.compare_seasons({2020: a, 2021: b}, draws=10)
    assert result["baseline_mse"] == result["candidate_mse"] == 2.5
    assert result["mse_gain"] == 0
    assert result["conditional_game_bootstrap_95"] == [0., 0.]


def test_diagnostic_refuses_export_only_targets_before_reading(tmp_path):
    with pytest.raises(ValueError, match="export-only"):
        summary.read_source(tmp_path, 2026, ["season"])


def test_ft_ablation_is_pregame_chronological_and_ignores_2026_targets(tmp_path, monkeypatch):
    monkeypatch.setattr(summary, "BOOTSTRAPS", 10)
    # Default arguments bind at definition, so wrap the real aggregator here.
    compare = summary.compare_seasons
    monkeypatch.setattr(summary, "compare_seasons", lambda seasons: compare(seasons, draws=20))
    rows = []
    for year in range(1997, 2027):
        for player, ft in enumerate(np.linspace(.5, .95, 20), 1):
            rows.append(dict(player_id=player, game_id=year * 100 + player, season=year,
                             day=(year - 1970) * 365, ft_pct=ft, arc3_pct=.35, corner3_pct=.35,
                             **{"core.arc3_fgm": 20 * (.1 + .5 * ft), "core.arc3_fga": 20.,
                                "core.corner3_fgm": 20 * (.15 + .4 * ft), "core.corner3_fga": 20.}))
    panel = pd.DataFrame(rows)
    roots = [tmp_path / "a", tmp_path / "b"]
    results = []
    for index, root in enumerate(roots):
        (root / "baseline").mkdir(parents=True)
        altered = panel.copy()
        if index:
            altered.loc[altered.season.eq(2026), "core.arc3_fgm"] = 0.
            altered.loc[altered.season.eq(2026), "core.corner3_fgm"] = 20.
        altered.to_parquet(root / "observations.parquet", index=False)
        for year, group in panel.groupby("season"):
            group[["player_id", "game_id", "season", "ft_pct", "arc3_pct", "corner3_pct"]].to_parquet(
                root / "baseline" / f"{year}.parquet", index=False)
        results.append(summary.ft_association(root))
    assert results[0] == results[1]
    for target in ["arc3_pct", "corner3_pct"]:
        assert results[0][target]["mse_gain"] > 0
        a = pd.read_parquet(roots[0] / "diagnostics" / f"ft_association_{target}_2026.parquet")
        b = pd.read_parquet(roots[1] / "diagnostics" / f"ft_association_{target}_2026.parquet")
        pd.testing.assert_frame_equal(a, b)
        assert a.with_ft.iloc[0] < a.with_ft.iloc[-1]
    assert (roots[0] / "diagnostics/ft_association_coefficients.json").read_bytes() == (
        roots[1] / "diagnostics/ft_association_coefficients.json").read_bytes()
