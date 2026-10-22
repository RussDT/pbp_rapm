from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import export_full_history_annual_snapshots as m
from full_skill_history_specs import Skill, SKILLS


@pytest.fixture(scope="module")
def kernel(tmp_path_factory):
    out = tmp_path_factory.mktemp("full_snapshot_kernel") / "kernel.so"
    subprocess.run(["c++", "-O2", "-std=c++11", "-shared", "-fPIC",
                    str(SCRIPTS / "full_skill_history_kernel.cpp"), "-o", str(out)], check=True)
    return out


def query(date, player=1, year=2026):
    return pd.DataFrame([dict(query_id=f"{date}:{player}", player_id=player,
                             query_date=date, query_season=year, cohort_vintage=None)])


def recipe(last="2025-06-01", year=2026):
    return dict(year=year, training_end_season=year - 1, status="annual_past_only_fit",
                last_training_observation=last, mean=.75,
                parameters=[1500., .03, 30., .002, .05, .2])


def panel():
    p = pd.DataFrame(dict(player_id=[1, 1, 1], game_id=["0000000001", "0000000002", "0000000003"],
                         game_date=pd.to_datetime(["2025-04-01", "2025-11-01", "2026-01-01"]),
                         season=[2025, 2026, 2026], **{"core.fta": [4., 8., 2.], "core.ftm": [0., 8., 1.]}))
    p["day"] = (p.game_date - pd.Timestamp("1970-01-01")).dt.days
    return p


def ft():
    return Skill("ft_pct", "core.ftm", "core.fta", "shooting", "pct", .75)


def test_strict_prefix_future_perturbation_and_query_only_missing(kernel):
    p = panel()
    q = pd.concat([query("2025-10-01"), query("2025-10-01", 2)], ignore_index=True)
    a, pa = m.replay_snapshot(p, q, [ft()], {"ft_pct": recipe()}, kernel)
    future = p.copy()
    future.loc[future.game_date.ge("2025-10-01"), ["core.ftm", "core.fta"]] = 1000000.
    b, pb = m.replay_snapshot(future, q, [ft()], {"ft_pct": recipe()}, kernel)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(pa, pb)
    assert np.isfinite(a.ft_pct.iloc[0])  # An earlier observed zero is real data.
    assert np.isnan(a.ft_pct.iloc[1])  # Unobserved population prior is not measured.
    assert list(pa.player_observed) == [True, False]
    assert list(pa.prior_exposure) == [4., 0.]
    assert pa.last_observation_date.dropna().lt(pd.Timestamp("2025-10-01")).all()


def test_independent_snapshots_prevent_query_variance_side_effect(kernel):
    p, recipes = panel(), {"ft_pct": recipe()}
    late = query("2026-09-01")
    expected, _ = m.replay_snapshot(p, late, [ft()], recipes, kernel)
    # The production dispatcher must use separate terminal replays, even for
    # two dates sharing one parameter vintage. An early query must not move
    # the offseason covariance injection in the later source history.
    m.replay_snapshot(p, query("2025-10-01"), [ft()], recipes, kernel)
    actual, _ = m.replay_snapshot(p, late, [ft()], recipes, kernel)
    pd.testing.assert_frame_equal(expected, actual)
    with pytest.raises(ValueError, match="separately"):
        m.replay_snapshot(p, pd.concat([query("2025-10-01"), late]), [ft()], recipes, kernel)


def test_actual_parameter_date_beats_calendar_year_guard(kernel):
    # Ending2021 recipe includes October2020 bubble outcomes; Oct1 is illegal.
    with pytest.raises(ValueError, match="training observation"):
        m.replay_snapshot(panel(), query("2020-10-01", year=2021), [ft()],
                          {"ft_pct": recipe("2020-10-11", 2021)}, kernel)
    m.validate_recipe(recipe("2020-10-11", 2021), 2021, pd.Timestamp("2020-12-01"))
    with pytest.raises(ValueError, match="no2027"):
        m.normalize_queries(query("2026-09-01", year=2027))
    with pytest.raises(ValueError, match="historical queries"):
        m.normalize_queries(query("2026-09-02"))


def test_strict_masks_follow_count_dependencies_not_unrelated_report():
    raw = {x for s in SKILLS for x in [s.numerator, s.denominator] if x.startswith("core.")}
    p = pd.DataFrame({name: [1., 1.] for name in raw})
    p["player_id"], p["game_id"] = [1, 2], ["0000000001", "0000000002"]
    p["game_date"] = pd.to_datetime(["2025-01-01", "2025-01-02"])
    p["off_poss"], p["def_poss"] = 20., 20.
    p["report.potential_ast_v2"] = [0., 3.]
    gates = {name for column in raw for name in m.source_dependencies(column)}
    v = pd.DataFrame({name: [1., 1.] for name in gates})
    for key in ["player_id", "game_id", "game_date", "off_poss", "def_poss"]:
        v[key] = p[key]
    v.loc[0, ["rim_fga", "rim_fgm", "fta", "ftm"]] = np.nan
    out, counts = m.strict_observations(p, v)
    assert np.isnan(out.loc[0, "core.at_rim_fga"])
    assert np.isnan(out.loc[0, "core.at_rim_fgm_assisted"])
    assert np.isnan(out.loc[0, "core.ftm"])
    assert out.loc[0, "report.potential_ast_v2"] == 0.
    assert out.loc[0, "core.ast_at_rim"] == 1.
    assert out.loc[0, "off_poss"] == 20.
    assert p.loc[0, "core.ftm"] == 1.  # Original observations never modified.
    assert counts["core.ftm"] == 1


def test_registry_is_broad_explicit_and_source_vintage_aware():
    schema = m.feature_schema()
    assert len(schema["records"]) == 203
    assert len({r["name"] for r in schema["records"]}) == 203
    r = next(r for r in schema["records"] if r["name"] == "lib__potential_ast100")
    assert r["source_heads_by_era"] == {"through_2024": ["potential_ast100__v1"],
                                       "from_2025": ["potential_ast100__v2"]}
    assert r["exposure_column"].endswith("__exposure")
    assert len(m.primitive_dependencies("tov_bad_pass_any100", "from_2025")) == 5
    assert all(r["name"].startswith("lib__") for r in schema["records"])
    assert not any("observed" in r["name"] for r in schema["records"])
    assert not any("__before_composition" in r["name"] for r in schema["records"])


def test_bootstrap_is_not_a_future_fit_or_a_measured_zero():
    r = recipe()
    r.pop("last_training_observation")
    with pytest.raises(ValueError, match="fit-end"):
        m.validate_recipe(r, 2026, pd.Timestamp("2026-09-01"))
    r["status"] = "bootstrap"
    m.validate_recipe(r, 2026, pd.Timestamp("2026-09-01"))
    r["parameters"][0] = 0
    with pytest.raises(ValueError, match="time/noise"):
        m.validate_recipe(r, 2026, pd.Timestamp("2026-09-01"))


def test_derived_feature_history_and_vintage_do_not_fill_missing_branch():
    q = m.normalize_queries(query("2026-01-10"))
    values = pd.DataFrame({s.name: [.5 if s.kind == "pct" else 1.] for s in SKILLS})
    provenance = []
    for s in SKILLS:
        row = q.iloc[0].to_dict()
        row.update(head_name=s.name, prior_exposure=10., last_observation_date=pd.Timestamp("2026-01-01"),
                   source_supported=True, player_observed=True)
        if s.name == "potential_ast100__v2":
            row.update(prior_exposure=0., last_observation_date=pd.NaT,
                       source_supported=False, player_observed=False)
            values[s.name] = np.nan
        provenance.append(row)
    wide = m.make_features(values, pd.DataFrame(provenance), q, m.feature_schema())
    assert np.isnan(wide["lib__potential_ast100"].iloc[0])
    assert wide["lib__potential_ast100__exposure"].iloc[0] == 0.
    assert np.isfinite(wide["lib__assist_points_created100"].iloc[0])
    assert wide["lib__assist_points_created100__last_observation_date"].iloc[0] < q.query_date.iloc[0]
    assert wide["lib__tov_bad_pass_any_share"].iloc[0] == .25
    assert len(wide.columns) == len(m.KEYS) + 3 * 203
