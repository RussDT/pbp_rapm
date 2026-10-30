from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "nba_pipeline" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


builder = load_script("build_estimated_skills_registry")


@pytest.fixture(scope="module")
def registry():
    return builder.load_registry()


def test_checked_in_registry_is_valid(registry):
    assert builder.validate_registry(registry) == []
    assert {s["stat_id"] for s in registry["stats"] if s["evidence"]["kind"] == "pbp_skill"} == set(builder.PBP_SKILLS)
    assert sorted(l["lane"] for l in registry["lanes"]) == list(range(1, 21))
    assert registry["contract"]["holdout"]["2027"] == "untouched"


def test_validation_catches_problems(registry):
    reg = copy.deepcopy(registry)
    reg["stats"].append(dict(reg["stats"][0]))                       # duplicate id
    reg["stats"][1]["status"] = "shipped"                            # bad status
    reg["lanes"] = [l for l in reg["lanes"] if l["lane"] != 7]       # missing lane
    live = next(s for s in reg["stats"] if s["evidence"]["kind"] != "pbp_skill")
    live["export"]["value_column"] = "epm_p_ast"                     # provider column
    problems = builder.validate_registry(reg)
    joined = "\n".join(problems)
    assert "duplicate stat_id" in joined
    assert "not in vocabulary" in joined
    assert "lanes missing" in joined and "7" in joined
    assert "provider columns" in joined


def test_gain_block_handles_both_receipt_styles():
    pct = builder.gain_block({"gain_pct": 0.35, "gain_ci95": [0.2, 0.5], "season_wins": 3, "seasons": [2023, 2024, 2025]})
    assert pct == {"gain_pct": 0.35, "ci_lo": 0.2, "ci_hi": 0.5, "wins": 3, "of": 3}
    # shooting-lane receipts put the interval on the loss delta; it must be rescaled to percent of the control loss
    delta = builder.gain_block({"gain_pct": -0.1, "delta": -0.005, "control_loss": 5.0, "ci95": [-0.01, -0.001]})
    assert delta["gain_pct"] == -0.1
    assert delta["ci_lo"] == pytest.approx(-0.2) and delta["ci_hi"] == pytest.approx(-0.02)
    assert builder.gain_block({"foo": 1}) is None
    assert builder.fmt_gain(None) == "no value"
    assert builder.fmt_gain(pct) == "+0.350% [+0.200, +0.500] 3/3"


def _mini_registry(registry):
    reg = copy.deepcopy(registry)
    keep = {"usg_pct", "rim_pct", "deflections"}
    reg["stats"] = [s for s in reg["stats"] if s["stat_id"] in keep]
    return reg


def test_build_values_never_fills_missing_with_zero(tmp_path, registry):
    reg = _mini_registry(registry)
    results = tmp_path / "results"
    panel_path = results / reg["pbp_panel"]["export"]["path"]
    panel_path.parent.mkdir(parents=True)
    pd.DataFrame({"player_id": [1, 2, 3], "Player": ["A One", "B Two", "C Three"],
                  "p_usg_pct": [0.20, 0.25, 0.30], "p_usg_pct_per100": [20.0, 25.0, 30.0], "usg_pct_last_date": ["2026-01-01"] * 3, "usg_pct_n_obs": [10, 20, 30],
                  "p_rim_pct": [0.6, np.nan, 0.7], "rim_pct_last_date": ["2026-01-01", None, "2026-01-02"], "rim_pct_n_obs": [5, 0, 6]}).to_csv(panel_path, index=False)
    defl = next(s for s in reg["stats"] if s["stat_id"] == "deflections")
    dpath = results / defl["export"]["path"]
    dpath.parent.mkdir(parents=True)
    pd.DataFrame({"player_id": [1, 3], "player_name": ["One", "Three"], "estimated_deflections_per100": [3.5, 2.0],
                  "game_date": ["2026-02-01", "2026-02-03"], "observed_games": [40, 12]}).to_csv(dpath, index=False)
    names = pd.Series({1: "Alpha One"})
    rc = builder.Receipts(results)
    wide, long, coverage = builder.build_values(reg, results, names, rc)
    assert len(wide) == 3
    assert wide.player_name.tolist() == ["Alpha One", "B Two", "C Three"]          # autocomplete first, table name fallback
    assert wide.rim_pct.tolist()[0] == pytest.approx(60.0)                           # pct skills displayed in %
    assert np.isnan(wide.rim_pct.tolist()[1])                                        # unmeasured stays NaN
    assert wide.usg_pct.tolist() == [20.0, 25.0, 30.0]                               # rates read the per-100 column
    assert np.isnan(wide.deflections.tolist()[1]) and wide.deflections.tolist()[2] == 2.0
    assert wide.deflections_observed_games.tolist()[2] == 12
    assert coverage["deflections"]["players"] == 2 and coverage["rim_pct"]["players"] == 2
    value_cols = [s["stat_id"] for s in reg["stats"]]
    assert not (wide[value_cols].fillna(1) == 0).any().any()                      # value columns are never zero-filled
    assert wide.rim_pct_n_obs.tolist() == [5, 0, 6]                                  # an observation count of zero is a count, not a value
    assert set(long.stat_id) == {"usg_pct", "rim_pct", "deflections"} and len(long) == 7
    assert reg["pbp_panel"]["export"]["path"] in rc.used and defl["export"]["path"] in rc.used


def test_missing_receipts_are_reported_not_invented(tmp_path, registry):
    reg = _mini_registry(registry)
    rc = builder.Receipts(tmp_path)
    rows, errors = builder.collect_evidence(reg, rc)
    assert rows == [] and errors == {}
    assert reg["pbp_panel"]["receipts"]["eval_report"] in rc.missing


@pytest.mark.skipif(not (builder.RESULTS / "current_player_table/current_players_provenance.csv").exists(), reason="local results not present")
def test_full_snapshot_on_local_results(tmp_path):
    manifest = builder.build(builder.REGISTRY, builder.RESULTS, tmp_path / "out")
    assert manifest["universe_players"] == 582
    assert manifest["n_live"] == 28 and manifest["adapter_errors"] == {}
    for name in ("estimated_skills_registry.json", "registry_status.csv", "registry_status.md", "current_values.csv",
                 "current_values_long.csv", "registry_evidence.csv", "registry_lanes.csv", "manifest.json", "index.html"):
        assert (tmp_path / "out" / name).exists()
    wide = pd.read_csv(tmp_path / "out" / "current_values.csv")
    assert len(wide) == 582 and wide.player_name.notna().all()
    for k, v in manifest["coverage"].items():
        assert v.get("players", 0) > 500, k
        assert v.get("state_date_max", "") <= "2026-12-31" and not v.get("error")
    # provider names may appear in prose (lane names, the contract); provider values and columns may not
    provider_cols = [c for c in wide.columns if any(t in c.lower() for t in builder.PROVIDER_TOKENS)]
    assert provider_cols == []
    long = pd.read_csv(tmp_path / "out" / "current_values_long.csv")
    assert set(long.stat_id) == {s["stat_id"] for s in builder.load_registry()["stats"]}
    assert long.value.notna().all() and (long.value != 0).all()                       # long form carries measured states only
    snapshot = json.loads((tmp_path / "out" / "estimated_skills_registry.json").read_text())
    assert len(snapshot["evidence"]) == manifest["evidence_rows"]
    windows = {e["window"] for e in snapshot["evidence"]}
    assert all(w.startswith(("dev", "frozen stress")) for w in windows)                # dev + frozen stress only
    assert not any("2026" in w for w in windows)                                       # 2026 is never a scored window
