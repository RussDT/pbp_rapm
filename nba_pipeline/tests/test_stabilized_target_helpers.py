import numpy as np
import pandas as pd

from nba_pipeline.scripts import tune_forward_component_alphas as helpers


def _first_chance_rows():
    return pd.DataFrame(
        {
            "Net_Diff": [1.0, 2.0, 0.0],
            "Is_Turnover": [0, 0, 1],
            "FC_DECOMP_RIM_FREQ_Diff": [0.2, 0.1, 0.0],
            "FC_DECOMP_RIM_FG_Diff": [0.3, -0.1, 0.0],
            "FC_DECOMP_MID_FG_Diff": [0.4, 0.8, 0.0],
            "FC_DECOMP_THREE_FREQ_Diff": [0.05, 0.2, 0.0],
            "FC_DECOMP_THREE_FG_Diff": [0.05, 1.0, 0.0],
        }
    )


def test_family_routing_covers_public_and_vpm_components():
    assert helpers.family_of("DECOMP_RIM_FREQ") == "FIRST_CHANCE"
    assert helpers.family_of("DECOMP_MID_VALUE") == "FIRST_CHANCE"
    assert helpers.family_of("SECOND_CHANCE_CLEAN") == "SECOND_CHANCE_CLEAN"
    assert helpers.family_of("RAPM") == "RAPM"


def test_research_target_uses_training_split_turnover_baseline():
    raw = _first_chance_rows()
    baselines = helpers.train_baselines(raw, "DECOMP_RIM_FREQ", None)
    target = helpers.prepare_target(
        raw, "DECOMP_RIM_FREQ", "FIRST_CHANCE", baselines
    )
    np.testing.assert_allclose(target, [0.2, 0.1, 0.15])


def test_family_loader_combines_rs_and_ps_and_marks_source_season(tmp_path, monkeypatch):
    monkeypatch.setattr(helpers, "PROCESSED_DIR", tmp_path)
    pd.DataFrame({"value": [1]}).to_parquet(tmp_path / "RAPM26.parquet", index=False)
    pd.DataFrame({"value": [2]}).to_parquet(tmp_path / "RAPM26_PS.parquet", index=False)

    out = helpers.load_family_raw("RAPM", [2026])

    assert out["value"].tolist() == [1, 2]
    assert out["season_phase"].tolist() == ["RS", "PS"]
    assert out["source_season_end_year"].tolist() == [2026, 2026]
