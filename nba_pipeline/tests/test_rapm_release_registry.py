from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

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


net_actual = load_script("run_net_decomp_rolling")
net_sq = load_script("run_net_sq_decomp_rolling")
mechanisms = load_script("run_mechanism_rapm_registry")
registry_builder = load_script("build_rapm_table_registry")
overnight = load_script("run_rapm_overnight")


def test_current_release_windows_are_one_through_five_seasons() -> None:
    actual = net_actual.current_release_windows(2026)
    sq = net_sq.current_release_windows(2026)
    assert [(window.start_year, window.end_year) for window in actual] == [
        (2026, 2026),
        (2025, 2026),
        (2024, 2026),
        (2023, 2026),
        (2022, 2026),
    ]
    assert [window.compact for window in sq] == ["26", "25_26", "24_26", "23_26", "22_26"]


def test_full_window_grid_has_all_140_unique_one_to_five_year_cuts() -> None:
    for builder in (
        net_actual.all_one_to_five_season_windows,
        net_sq.all_one_to_five_season_windows,
    ):
        windows = builder(2026)
        assert len(windows) == 140
        assert len(set(windows)) == 140
        assert [window.compact for window in windows[:6]] == [
            "97",
            "98",
            "97_98",
            "99",
            "98_99",
            "97_99",
        ]
        assert [window.compact for window in windows[-5:]] == [
            "26",
            "25_26",
            "24_26",
            "23_26",
            "22_26",
        ]
        assert {window.seasons for window in windows} == {1, 2, 3, 4, 5}


def test_actual_net_window_parser_rejects_more_than_five_seasons() -> None:
    assert net_actual.parse_window("22-26").compact == "22_26"
    assert net_actual.parse_window("2026").compact == "26"
    with pytest.raises(ValueError, match="1-5 seasons"):
        net_actual.parse_window("21-26")


def test_sq_profiles_and_output_names_stay_distinct() -> None:
    assert set(net_sq.alpha_map("explanatory").values()) == {(2000, 4000)}
    assert len(set(net_sq.alpha_map("vpm").values())) > 1
    window = net_sq.Window(2022, 2026)
    assert "decomp_net_sq_resid" in net_sq.output_stem(window, "explanatory", False, 550)
    assert net_sq.output_stem(window, "vpm", True, 550).endswith("_td550")


def write_result(path: Path, *, off: float, solver_def: float, net: float) -> None:
    pd.DataFrame(
        {
            "player_id": [1],
            "player_name": ["Example Player"],
            "off": [off],
            "def": [solver_def],
            "net_rapm": [net],
            "possessions": [1000],
            "off_poss": [500],
            "def_poss": [500],
        }
    ).to_csv(path, index=False)


def test_sq_explanatory_assembly_closes_to_parent(tmp_path: Path) -> None:
    write_result(tmp_path / "rapm_fixture_results.csv", off=1.0, solver_def=-0.5, net=1.5)
    for component in net_sq.COMPONENTS:
        write_result(
            tmp_path / f"{component.lower()}_fixture_results.csv",
            off=0.1,
            solver_def=-0.05,
            net=0.15,
        )
    board, manifest = net_sq.assemble_window(
        net_sq.Window(2022, 2026), tmp_path, "explanatory", False, 550
    )
    assert board.loc[0, "estimate_off"] == pytest.approx(1.0)
    assert board.loc[0, "estimate_def"] == pytest.approx(0.5)
    assert board.loc[0, "estimate_net"] == pytest.approx(1.5)
    assert manifest["identity"]["max_abs_parent_gap_net"] < 1e-9


def test_mechanism_plan_records_incomplete_windows(tmp_path: Path) -> None:
    registry = {
        "mechanism_rapms": [
            {
                "mechanism_id": "rim_frequency",
                "metric": "RIM_FREQ",
                "processed_prefix": "RIM_FREQ",
                "season_type": "ALL",
                "alphas": [3000, 3000],
                "release_windows": "current_up_to_5",
            }
        ]
    }
    for suffix in ("", "_PS"):
        (tmp_path / f"RIM_FREQ26{suffix}.parquet").touch()
    jobs, skipped = mechanisms.build_plan(registry, tmp_path, 2026)
    assert [job.window.compact for job in jobs] == ["26"]
    assert {row["window"] for row in skipped} == {"25_26", "24_26", "23_26", "22_26"}


def test_current_td550_mechanism_plan_uses_explicit_coverage(tmp_path: Path) -> None:
    registry = {
        "current_mechanism_view": {
            "default_start_year": 2021,
            "season_type": "ALL",
            "half_life_days": 550,
            "alpha_selection": {
                "method": "weighted_grouped_game_cv",
                "folds": 5,
                "passes": 2,
                "grid": [100, 1000, 10000],
            },
            "mechanisms": [
                {"mechanism_id": "rim_frequency"},
                {
                    "mechanism_id": "transition_frequency",
                    "start_year": 2026,
                    "coverage_note": "2026-only provider coverage",
                },
            ],
        },
        "mechanism_rapms": [
            {
                "mechanism_id": "rim_frequency",
                "metric": "RIM_FREQ",
                "processed_prefix": "RIM_FREQ",
                "alphas": [3000, 3000],
            },
            {
                "mechanism_id": "transition_frequency",
                "metric": "TRANSITION_FREQ",
                "processed_prefix": "TRANSITION_FREQ",
                "alphas": [3000, 3000],
            },
        ],
    }
    for year in range(2021, 2027):
        for suffix in ("", "_PS"):
            (tmp_path / f"RIM_FREQ{year % 100:02d}{suffix}.parquet").touch()
    for suffix in ("", "_PS"):
        (tmp_path / f"TRANSITION_FREQ26{suffix}.parquet").touch()

    jobs, skipped, _ = mechanisms.build_current_td_plan(registry, tmp_path, 2026)
    assert not skipped
    assert [job.window.compact for job in jobs] == ["21_26", "26"]
    assert all(job.timedecay and job.half_life_days == 550 for job in jobs)
    assert all(job.cv_alpha and job.cv_alpha_grid == (100.0, 1000.0, 10000.0) for job in jobs)


def test_current_td550_command_includes_weighted_cv_contract(tmp_path: Path) -> None:
    job = mechanisms.Job(
        mechanism_id="rim_frequency",
        metric="RIM_FREQ",
        processed_prefix="RIM_FREQ",
        season_type="ALL",
        window=mechanisms.Window(2021, 2026),
        off_alpha=3000,
        def_alpha=3000,
        timedecay=True,
        half_life_days=550,
        cv_alpha=True,
        cv_folds=5,
        cv_passes=2,
        cv_alpha_grid=(100, 1000, 10000),
    )
    command, env = mechanisms.build_command(job, tmp_path, tmp_path / "out", 2)
    assert "--half-life" in command and command[command.index("--half-life") + 1] == "550"
    assert "--cv-alpha" in command
    assert command[-1] == "100,1000,10000"
    assert env["NBA_PIPELINE_PROCESSED_DIR"] == str(tmp_path)


def test_future_selected_td550_plan_uses_registry_pairs_without_cv(tmp_path: Path) -> None:
    registry = {
        "current_mechanism_view": {
            "default_start_year": 2021,
            "season_type": "ALL",
            "half_life_days": 550,
            "future_validation": {
                "recommended_tested_pairs": {
                    "stop": [3000, 3000],
                    "transition_rim_backcast": [3000, 10000],
                }
            },
        },
        "mechanism_rapms": [
            {"mechanism_id": "stop", "metric": "STOP", "processed_prefix": "STOP"},
            {
                "mechanism_id": "transition_rim_backcast",
                "metric": "TRANSITION_RIM_BACKCAST",
                "processed_prefix": "TRANSITION_RIM_BACKCAST",
            },
        ],
    }
    for prefix in ("STOP", "TRANSITION_RIM_BACKCAST"):
        for year in range(2021, 2027):
            for suffix in ("", "_PS"):
                (tmp_path / f"{prefix}{year % 100:02d}{suffix}.parquet").touch()
    jobs, skipped, _ = mechanisms.build_future_selected_td_plan(
        registry, tmp_path, 2026
    )
    assert not skipped
    by_id = {job.mechanism_id: job for job in jobs}
    assert (by_id["stop"].off_alpha, by_id["stop"].def_alpha) == (3000, 3000)
    assert (
        by_id["transition_rim_backcast"].off_alpha,
        by_id["transition_rim_backcast"].def_alpha,
    ) == (3000, 10000)
    assert all(job.timedecay and not job.cv_alpha for job in jobs)


def test_parse_selected_mechanism_alphas() -> None:
    assert mechanisms.parse_selected_alphas(
        "Alpha CV selected off_alpha=7000 and def_alpha=15000"
    ) == (7000.0, 15000.0)


def test_filtered_mechanism_restart_preserves_unaffected_manifest_records(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "records": [
                    {"mechanism_id": "rim_frequency", "window": "21_26", "status": "built"},
                    {"mechanism_id": "reb", "window": "21_26", "status": "built"},
                ],
                "skipped": [],
            }
        )
    )
    records, skipped = mechanisms.merge_partial_manifest(
        manifest_path,
        [
            {
                "mechanism_id": "rim_frequency",
                "window": "21_26",
                "status": "rebuilt",
            }
        ],
        [],
    )
    by_id = {row["mechanism_id"]: row for row in records}
    assert by_id["rim_frequency"]["status"] == "rebuilt"
    assert by_id["reb"]["status"] == "built"
    assert skipped == []


def test_checked_in_registry_has_unique_ids_and_required_families() -> None:
    path = ROOT / "nba_pipeline" / "config" / "rapm_table_registry.json"
    registry = json.loads(path.read_text())
    registry_builder.validate_registry(registry)
    series = {item["series_id"] for item in registry["series"]}
    assert {
        "net_decomp",
        "net_decomp_sq_resid",
        "net_vpm_sq_resid_td550",
        "gross_decomp",
        "alt3",
    } <= series
    mechanism_ids = {item["mechanism_id"] for item in registry["mechanism_rapms"]}
    assert {"rim_frequency", "three_fg_pct", "assist_points", "joint_ts_shooter_interaction"} <= mechanism_ids


def test_combined_current_and_historical_release_expects_thirty_unique_cuts() -> None:
    assert registry_builder.expected_csv_count(
        {"release_windows": "current_1y_5y_plus_all_complete_rolling_5y_1997_2026"}
    ) == 30


def test_full_one_to_five_year_registry_contract_expects_140_cuts() -> None:
    assert registry_builder.expected_csv_count(
        {"release_windows": "all_rolling_1y_5y_1997_2026"}
    ) == 140


def test_net_release_series_require_csv_and_parquet() -> None:
    path = ROOT / "nba_pipeline" / "config" / "rapm_table_registry.json"
    registry = json.loads(path.read_text())
    by_id = {item["series_id"]: item for item in registry["series"]}
    for series_id in ("net_decomp", "net_decomp_sq_resid", "net_vpm_sq_resid_td550"):
        assert by_id[series_id]["required_formats"] == ["csv", "parquet"]


def test_overnight_queue_orders_historical_target_build_before_history() -> None:
    args = argparse.Namespace(
        release_end_year=2026,
        workers=2,
        cores_per_rapm=2,
        force=False,
        skip_history=False,
        dry_run=True,
    )
    stages = overnight.build_stages(args)
    ids = [stage.stage_id for stage in stages]
    assert ids.index("historical_sq_targets") < ids.index("net_sq_td550_history")
    history = next(stage for stage in stages if stage.stage_id == "net_sq_td550_history")
    assert history.depends_on == ("historical_sq_targets",)


def test_full_grid_overnight_queue_has_three_distinct_net_stages() -> None:
    args = argparse.Namespace(
        release_end_year=2026,
        workers=2,
        cores_per_rapm=2,
        force=False,
        skip_history=False,
        full_window_grid=True,
        dry_run=True,
    )
    stages = overnight.build_stages(args)
    assert [stage.stage_id for stage in stages] == [
        "historical_sq_targets",
        "net_actual_full_grid",
        "net_sq_full_grid",
        "net_sq_td550_full_grid",
        "registry_snapshot",
    ]
    for stage in stages[1:4]:
        assert "--all-one-to-five-season-windows" in stage.commands[0]
