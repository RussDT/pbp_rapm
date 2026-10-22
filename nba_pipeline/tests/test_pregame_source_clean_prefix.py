"""Independent audit mechanics preserve masks and reject changed prefixes."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_pregame_source_clean_prefix import (
    mutate_future, compare_export_prefix, compare_fit_models,
    copy_counterfactual_source, verify_pins, clean_model_pins,
)
import build_monthly_own_skills as C


def test_future_mutation_preserves_masked_nan_and_all_earlier_observations():
    source = pd.DataFrame(dict(game_date=pd.to_datetime(["2015-09-30", "2015-10-01", "2015-10-02"]),
        ftm=[1., np.nan, 2.], three_fgm=[2., 1., np.nan], ast=[3., np.nan, 4.], fta=[2., np.nan, 3.]))
    before = source.copy()
    result = mutate_future(source, "2015-10-01")
    pd.testing.assert_frame_equal(source, before)
    pd.testing.assert_series_equal(result.iloc[0], source.iloc[0])
    pd.testing.assert_frame_equal(result.isna(), source.isna())
    assert result.ftm.iloc[2] == 0.
    assert result.three_fgm.iloc[1] == 0.
    assert result.fta.tolist()[0] == 2.


def example_exports():
    reference = pd.DataFrame(dict(player_id=[1,1,1], game_id=["a","b","c"],
        game_date=pd.to_datetime(["2020-01-01", "2020-01-22", "2020-01-23"]),
        feature=["own_ft_pct"]*3, value=[.7,.8,.9], state_variance=[.1,.2,.3],
        prior_exposure=[1.,2.,3.], last_observation_day=[1,2,3]))
    altered = reference.copy()
    altered.loc[2, "value"] = .4
    return reference, altered


def test_prefix_comparison_includes_current_game_and_nonvalue_state_metadata():
    reference, altered = example_exports()
    result = compare_export_prefix(reference, altered, "2020-01-22")
    assert result["unchanged_rows"] == 2
    assert result["later_state_max_change"] == .5
    altered.loc[1, "prior_exposure"] = 99.
    with pytest.raises(ValueError, match="prior_exposure"):
        compare_export_prefix(reference, altered, "2020-01-22")


def test_prefix_audit_requires_an_effective_counterfactual():
    reference, _ = example_exports()
    with pytest.raises(ValueError, match="later state"):
        compare_export_prefix(reference, reference.copy(), "2020-01-22")


def test_model_comparison_ignores_only_provenance_and_wall_time():
    a = dict(params=[1.,2.], objective=.1, training_rows=100, seconds=1., source_lock_sha256="a",
             context_c=2., centered=False, stage_a=dict(params=[3.], loss=.2))
    b = dict(a, seconds=2., source_lock_sha256="b")
    assert compare_fit_models(a,b)["parameters_identical"]
    b["context_c"] = 5.
    with pytest.raises(ValueError, match="context_c"):
        compare_fit_models(a,b)


def test_counterfactual_source_changes_future_context_and_opportunities_only(tmp_path):
    source = tmp_path / "original"
    source.mkdir()
    frame = pd.DataFrame(dict(player_id=[1,1,1], game_id=["a","b","c"],
        game_date=pd.to_datetime(["2015-09-30", "2015-10-01", "2015-10-02"]),
        ftm=[1.,np.nan,2.], fta=[2.,np.nan,3.], three_fgm=[1.,np.nan,2.], ast=[2.,3.,4.]))
    context = pd.DataFrame(dict(player_id=[1,1,1],game_id=["a","b","c"],
        is_home=pd.array([True,False,None],dtype="boolean")))
    frame.to_parquet(source/"player_game_lines.parquet",index=False)
    context.to_parquet(source/"game_context.parquet",index=False)
    manifest = dict(artifacts={name:C.sha(source/name) for name in ["player_game_lines.parquet","game_context.parquet"]})
    (source/"manifest.json").write_text(json.dumps(manifest))
    original_hashes = {str(path):C.sha(path) for path in source.iterdir()}
    out = copy_counterfactual_source(source,tmp_path/"counterfactual","2015-10-01")
    verify_pins(original_hashes)
    altered = pd.read_parquet(out/"player_game_lines.parquet")
    pd.testing.assert_series_equal(altered.iloc[0],frame.iloc[0])
    pd.testing.assert_frame_equal(altered.isna(),frame.isna())
    assert altered.fta.iloc[2] == 7.
    changed_context = pd.read_parquet(out/"game_context.parquet")
    assert bool(changed_context.is_home.iloc[0])
    assert bool(changed_context.is_home.iloc[1])
    assert pd.isna(changed_context.is_home.iloc[2])
    assert json.loads((out/"manifest.json").read_text())["counterfactual_source_admitted"] is False


def test_source_clean_audit_rejects_transported_or_short_budget_models(tmp_path):
    fit = tmp_path/"fit"
    fit.mkdir()
    source = tmp_path/"source"
    lock = dict(source_directory=str(source), source_clean_refit=False, parameter_transport_not_refit=True)
    (fit/"source_lock.json").write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="Transported"):
        clean_model_pins(fit,source,[])
    lock.update(source_clean_refit=True,parameter_transport_not_refit=False,
        maxiter=10,popsize=15,seed=20260928,tol=1e-7,polish=False)
    (fit/"source_lock.json").write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="60-iteration"):
        clean_model_pins(fit,source,[])
