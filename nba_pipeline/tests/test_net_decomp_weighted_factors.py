from pathlib import Path

import numpy as np
import pandas as pd

import nba_pipeline.scripts.build_net_decomp_weighted_factors as builder


def test_net_decomp_builder_closes_component_tree(tmp_path: Path):
    results = tmp_path / "components"
    master = tmp_path / "master"
    results.mkdir()
    year_range = "22_26"
    suffix = "all_rb_se_a2000_4000"
    players = pd.DataFrame(
        {
            "player_id": [1, 2],
            "player_name": ["A", "B"],
            "off": [0.8, -0.8],
            "def": [-0.4, 0.4],
            "net_rapm": [1.2, -1.2],
            "possessions": [100, 80],
            "off_poss": [50, 40],
            "def_poss": [50, 40],
        }
    )
    players.to_csv(results / builder.result_filename("RAPM", year_range, suffix), index=False)

    values = {
        "DECOMP_NET_RIM_FREQ": 0.05,
        "DECOMP_NET_RIM_FG": 0.10,
        "DECOMP_NET_MID_FG": 0.15,
        "DECOMP_NET_THREE_FREQ": 0.05,
        "DECOMP_NET_THREE_FG": 0.05,
        "DECOMP_NET_EFG": 0.40,
        "ALT_FT": 0.10,
        "ALT_TOV_NET_VALUE": 0.15,
        "SECOND_CHANCE_NET": 0.15,
        "ALT_BADPASS_TOV_NET_VALUE": 0.06,
        "ALT_SCORING_TOV_NET_VALUE": 0.09,
    }
    for prefix, value in values.items():
        frame = players.copy()
        frame["off"] = [value, -value]
        frame["def"] = [-value / 2, value / 2]
        frame["net_rapm"] = frame["off"] - frame["def"]
        frame.to_csv(results / builder.result_filename(prefix, year_range, suffix), index=False)

    out = builder.build_net_decomp_weighted_factors(
        year_range,
        suffix,
        "net.csv",
        results_dir=results,
        master_results_dir=master,
    )[-1]
    np.testing.assert_allclose(out[["oRESID", "dRESID", "RESID"]], 0.0, atol=1e-12)
    np.testing.assert_allclose(out["oDECOMP_EFG_PARENT_GAP"], 0.0, atol=1e-12)
    np.testing.assert_allclose(out["oDECOMP_TOV_SPLIT_GAP"], 0.0, atol=1e-12)
    assert set(out["DECOMP_PRICING"]) == {"net_continuation_kappa"}
