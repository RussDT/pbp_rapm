from pathlib import Path

import numpy as np
import pandas as pd

import nba_pipeline.scripts.build_decomp_weighted_factors as builder


def _write_inputs(root: Path, year_range: str, suffix: str) -> tuple[Path, Path]:
    results = root / "results"
    results.mkdir()
    base = root / "base.csv"
    pd.DataFrame(
        {"player_id": [1, 2], "player_name": ["A", "B"], "Latest_Year": [2026, 2026]}
    ).to_csv(base, index=False)
    values = pd.DataFrame(
        {
            "player_id": [1, 2],
            "player_name": ["A", "B"],
            "off": [1.0, -1.0],
            "def": [-0.4, 0.4],
            "net_rapm": [1.4, -1.4],
            "possessions": [100, 80],
            "off_poss": [50, 40],
            "def_poss": [50, 40],
        }
    )
    prefixes = ["RAPM", *builder.SHOT_COMPONENTS.values(), *builder.OTHER_COMPONENTS.values()]
    for prefix in prefixes:
        frame = values.copy()
        if prefix != "RAPM":
            frame[["off", "def", "net_rapm"]] *= 0.1
        frame.to_csv(results / builder.result_filename(prefix, year_range, suffix), index=False)
    return base, results


def test_builder_matches_public_schema_and_closes_display(tmp_path, monkeypatch):
    year_range = "22_26"
    suffix = "all_rb_se_a2000_4000"
    base, results = _write_inputs(tmp_path, year_range, suffix)
    monkeypatch.setattr(builder, "RESULTS_DIR", results)
    monkeypatch.setattr(builder, "MASTER_RESULTS_DIR", tmp_path / "master")

    out = builder.build_decomp_weighted_factors(
        year_range,
        suffix,
        base,
        "weighted_factors_decomp_test.csv",
        results_dir=results,
        transfer_multiplier=1.147,
        write_parquet=False,
    )[-1]

    assert len(out.columns) == 97
    assert out.columns[:5].tolist() == [
        "player_id",
        "player_name",
        "Latest_Year",
        "TOV_SC_ALLOCATION_POLICY",
        "TOV_SC_TRANSFER_MULTIPLIER",
    ]
    np.testing.assert_allclose(out[["oRESID", "dRESID", "RESID"]], 0.0)
    np.testing.assert_allclose(
        out["oDECOMP_EFG_PARENT_GAP"],
        out["oDECOMP_EFG_PARENT"] - out["oDECOMP_EFG_CHILDREN"],
    )
    assert set(out["TOV_SC_ALLOCATION_POLICY"]) == {"full_possession_v1"}


def test_tov_second_chance_transfer_preserves_component_total(tmp_path, monkeypatch):
    year_range = "22_26"
    suffix = "all_rb_se_a2000_4000"
    base, results = _write_inputs(tmp_path, year_range, suffix)
    monkeypatch.setattr(builder, "RESULTS_DIR", results)
    monkeypatch.setattr(builder, "MASTER_RESULTS_DIR", tmp_path / "master")

    unallocated = builder.build_decomp_weighted_factors(
        year_range,
        suffix,
        base,
        "unallocated.csv",
        results_dir=results,
        write_parquet=False,
    )[-1]
    allocated = builder.build_decomp_weighted_factors(
        year_range,
        suffix,
        base,
        "allocated.csv",
        results_dir=results,
        transfer_multiplier=1.147,
        write_parquet=False,
    )[-1]

    for side in ["o", "d"]:
        np.testing.assert_allclose(
            allocated[f"{side}DECOMP_TOV_VALUE"] + allocated[f"{side}SC"],
            unallocated[f"{side}DECOMP_TOV_VALUE"] + unallocated[f"{side}SC"],
            atol=1e-3,
        )
