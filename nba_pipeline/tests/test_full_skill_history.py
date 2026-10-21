"""Semantic tests for source admission and chronological query-state replay."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from build_full_skill_history import Engine, compile_kernel
from download_databallr_skill_history import season_bounds
from full_skill_history_specs import Skill
from prepare_full_skill_history import selected_report_column
from full_skill_cross_talk import choose_alpha, solve_moments


@pytest.fixture
def panel():
    return pd.DataFrame({"player_id": [1, 1, 1, 1, 2], "day": [1, 2, 2, 8, 1],
                         "season": [2000] * 5, "n": [1., 2., 3., np.nan, 0.], "d": [2., 4., 5., 0., 3.]})


def replay(panel, tmp_path):
    return Engine(panel, Skill("p", "n", "d", "test", "pct", .5), compile_kernel(tmp_path)).run([100., .01, 10., .002, .001, .25], .5)


def test_query_only_and_same_day_use_only_prior_evidence(panel, tmp_path):
    pred, post, _, exposure, count, last = replay(panel, tmp_path)
    assert pred[1] == pred[2]
    assert exposure.tolist() == [0., 2., 2., 11., 0.]
    assert count.tolist() == [0., 1., 1., 3., 0.]
    assert last.tolist() == [-1., 1., 1., 2., -1.]
    assert post[1] == post[2]
    assert 0 <= pred[3] <= 1


def test_target_and_future_perturbations_leave_pregame_predictions_fixed(panel, tmp_path):
    old = replay(panel, tmp_path)[0]
    other = panel.copy()
    other.loc[1:3, "n"] = [0., 0., 0.]
    new = replay(other, tmp_path)[0]
    np.testing.assert_array_equal(old[:3], new[:3])
    assert old[4] == new[4]


def test_missing_measurement_is_not_an_observed_zero(panel, tmp_path):
    missing = panel.copy()
    missing.loc[0, "n"] = np.nan
    zero = panel.copy()
    zero.loc[0, "n"] = 0.
    a = replay(missing, tmp_path)
    b = replay(zero, tmp_path)
    assert a[3][1] == 0
    assert b[3][1] == 2
    assert a[0][1] != b[0][1]


def test_unsupported_source_epoch_is_not_an_observed_zero(panel, tmp_path):
    spec = Skill("unsupported", "n", "d", "test", "pct", .5, first_season=2018)
    engine = Engine(panel, spec, compile_kernel(tmp_path))
    assert not engine.exposure.any()
    values = engine.run([100., .01, 10., .002, .001, .25], .5)
    assert not values[3].any()


def test_compiled_filter_matches_existing_independent_ou(panel, tmp_path):
    from monthly_ou_filter import FastOU
    sub = panel.iloc[[0, 1, 4]].copy().reset_index(drop=True)
    sub["player_idx"] = pd.factorize(sub.player_id)[0]
    sub["game_date"] = pd.to_datetime(sub.day, unit="D")
    sub["days_gap"] = sub.groupby("player_id").day.diff().fillna(0)
    sub["season_cross"] = 0.
    sub["exposure"] = sub.d
    parameters = [100., .01, 10., .002, .001, .25]
    old = FastOU(sub, tmp_path / "existing").run((sub.n / sub.d).to_numpy(), parameters, .5)
    new = replay(sub, tmp_path)
    np.testing.assert_allclose(new[0], old[0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(new[1], old[2], rtol=0, atol=1e-12)
    np.testing.assert_allclose(new[2], old[1], rtol=0, atol=1e-12)


def test_named_vintage_and_ft_flags_prevent_cross_version_fill():
    report = pd.DataFrame({"tracking_vintage": ["v1", "v2", "v2"], "ours_match": [True] * 3,
                           "report_label_wrong": [False] * 3, "drive_fta_v1": [1., 5., 8.],
                           "drive_fta_v2": [2., np.nan, 7.], "ft_tracking_status": ["ok", "ok", "unreliable_nba_2024_25"]})
    values = selected_report_column(report, "drive_fta")
    assert values[0] == 1
    assert pd.isna(values[1]) and pd.isna(values[2])


def test_manifest_season_fence_runs_before_data_download():
    def bounds(year):
        return [{"key": 2, "value": int(year).to_bytes(4, "little", signed=True)}]
    assert season_bounds({"lower_bounds": bounds(1997), "upper_bounds": bounds(2026)}, 2) == (1997, 2026)
    with pytest.raises(ValueError):
        season_bounds({"lower_bounds": bounds(1997), "upper_bounds": bounds(2027)}, 2)
    with pytest.raises(ValueError):
        season_bounds({}, 2)


def test_cross_stat_profile_matches_direct_weighted_joint_ridge():
    rng = np.random.default_rng(4)
    raw = rng.normal(size=(50, 4))
    x = np.column_stack([np.ones(50), raw])
    y = .2 + raw[:, 0] * .6 + rng.normal(0, .1, 50)
    w = rng.uniform(.1, 4, 50)
    alpha = .1
    mean = np.average(raw, axis=0, weights=w)
    scale = np.sqrt(np.average((raw - mean) ** 2, axis=0, weights=w))
    z = np.column_stack([np.ones(50), (raw - mean) / scale])
    normal = z.T @ (w[:, None] * z) / w.sum() + np.diag([0, alpha, alpha, alpha, alpha])
    direct = np.linalg.solve(normal, z.T @ (w * y) / w.sum())
    moments = solve_moments(x.T @ (w[:, None] * x), x.T @ (w * y), alpha)
    np.testing.assert_allclose(x @ moments, z @ direct, rtol=0, atol=1e-12)


def test_cross_stat_choice_uses_only_older_seasons_and_prefers_null_ties():
    history = [{"year": y, "baseline": 1., "0.1": .9, "1.0": 1.1, "10.0": 1.2} for y in [2000, 2001, 2002]]
    assert choose_alpha(history, 2002) is None
    assert choose_alpha(history, 2003) == .1
    history.append({"year": 2003, "baseline": 1., "0.1": 100., "1.0": .001, "10.0": .01})
    assert choose_alpha(history, 2003) == .1
    assert choose_alpha([{**x, "0.1": 1.} for x in history[:3]], 2003) is None


def test_derived_shooting_and_rebound_accounting_closes():
    from export_full_skill_history import derived, audit
    from full_skill_history_specs import SKILLS
    source = pd.DataFrame({s.name: [.4, .6] if s.kind == "pct" else [5., 10.] for s in SKILLS})
    source["season"] = 2026
    result = derived(source)
    assert audit(result) == 0
    for reb in ["oreb", "dreb"]:
        columns = [reb + "_after_" + k + "100" for k in ["2pt", "3pt", "ft", "other"]]
        np.testing.assert_allclose(result[columns].sum(axis=1), source[reb + "100"])
    assert result.three_fgm100.le(result.three_fga100).all()


def test_tracking_vintages_are_separate_models_not_a_stitched_series():
    from full_skill_history_specs import SKILLS, canonical_aliases
    specs = {s.name: s for s in SKILLS}
    assert specs["drives100__v1"].numerator == "report.drives_v1"
    assert specs["drives100__v2"].numerator == "report.drives_v2"
    assert specs["drives100__v2"].first_season == 2025
    frame = pd.DataFrame({"season": [2024, 2025, 2026], "drives100__v1": [4., 4., 4.], "drives100__v2": [np.nan, 6., np.nan]})
    values = canonical_aliases(frame).drives100
    assert values.iloc[:2].tolist() == [4., 6.]
    assert pd.isna(values.iloc[2])


def test_full_cross_stat_replay_never_scores_or_fits_2026_targets(tmp_path, monkeypatch):
    import full_skill_cross_talk as cross
    skills = [Skill("ft_pct", "ftm", "fta", "shooting", "pct", .75),
              Skill("arc3_pct", "threem", "threea", "shooting", "pct", .35)]
    monkeypatch.setattr(cross, "SKILLS", skills)
    outcomes = []
    for year in range(1997, 2027):
        for player, ft in enumerate([.5, .7, .9], 1):
            outcomes.append(dict(player_id=player, game_id=year * 10 + player,
                                 season=year, game_date=pd.Timestamp(year=year - 1, month=11, day=1),
                                 day=(year - 1970) * 365, ftm=20 * ft, fta=20.,
                                 threem=20 * (.1 + .5 * ft), threea=20., ft_pct=ft, arc3_pct=.35))
    source = pd.DataFrame(outcomes)
    roots = [tmp_path / "a", tmp_path / "b"]
    for index, root in enumerate(roots):
        (root / "baseline").mkdir(parents=True)
        (root / "run_lock.json").write_text("{}")
        data = source.copy()
        if index:
            data.loc[data.season.eq(2026), "threem"] = 0.
        data.to_parquet(root / "observations.parquet", index=False)
        for year, frame in source.groupby("season"):
            frame[["player_id", "game_id", "season", "game_date", "ft_pct", "arc3_pct"]].to_parquet(root / "baseline" / f"{year}.parquet", index=False)
        cross.run(root, workers=1)
        scores = pd.read_csv(root / "cross_stat/historical_scores.csv")
        assert scores.year.max() == 2025
    a = pd.read_parquet(roots[0] / "cross_stat/predictions/2026.parquet")
    b = pd.read_parquet(roots[1] / "cross_stat/predictions/2026.parquet")
    pd.testing.assert_frame_equal(a, b)
    assert a.arc3_pct.iloc[0] < a.arc3_pct.iloc[-1]
