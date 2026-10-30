"""Offline pytest coverage for the td550 Net SQ rolling season rollover.

Two literals blocked a 2027 (2026-27) td550 run. The producer's input
precheck demanded an RS and a PS half for every season in the window, so an
in-progress season that has played no playoffs failed all five windows. The
publisher hardcoded `range(1997, 2027)` and then asserted exact set equality,
so nothing with a 2027 window could ever validate.

These tests pin the in-season playoff exemption to the window's own end year,
pin the partial-playoff failure that the exemption must not hide, and pin the
publisher's derived bound against the window set the hardcoded one produced.
No test solves a ridge, reads the real target tree, or touches Supabase.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "nba_pipeline" / "scripts"

sys.path.insert(0, str(SCRIPTS_DIR))


def _load_module(name: str, path: Path):
    """Import a script by path without running it as __main__."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # Registered before exec because the producer's @dataclass resolves its
    # field annotations through sys.modules.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_module(
    "run_net_sq_decomp_rolling_under_test",
    SCRIPTS_DIR / "run_net_sq_decomp_rolling.py",
)
publisher = _load_module(
    "publish_td550_rolling_archive_under_test",
    SCRIPTS_DIR / "publish_td550_rolling_archive.py",
)


def _touch_targets(processed_dir: Path, year: int, *, playoffs: bool) -> None:
    """Create the RS half, and optionally the PS half, for one season."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    for family in runner.TARGET_FAMILIES:
        (processed_dir / f"{family}{year % 100:02d}.parquet").touch()
        if playoffs:
            (processed_dir / f"{family}{year % 100:02d}_PS.parquet").touch()


# --- producer: in-season playoff exemption -------------------------------


def test_expected_inputs_default_demands_both_halves():
    window = runner.Window(start_year=2023, end_year=2027)
    paths = runner.expected_input_paths(Path("/targets"), window)
    assert len(paths) == 5 * 2 * len(runner.TARGET_FAMILIES)
    assert Path("/targets/RAPM27_PS.parquet") in paths


def test_exemption_drops_only_the_end_year_playoff_half():
    window = runner.Window(start_year=2023, end_year=2027)
    paths = runner.expected_input_paths(
        Path("/targets"), window, allow_missing_playoffs=True
    )
    assert len(paths) == 5 * 2 * len(runner.TARGET_FAMILIES) - len(runner.TARGET_FAMILIES)
    # The in-progress season's playoffs are exempt...
    assert Path("/targets/RAPM27_PS.parquet") not in paths
    # ...but its regular season, and every earlier season's playoffs, are not.
    assert Path("/targets/RAPM27.parquet") in paths
    assert Path("/targets/RAPM26_PS.parquet") in paths
    assert Path("/targets/FIRST_CHANCE23_PS.parquet") in paths


def test_exemption_on_a_single_season_window():
    window = runner.Window(start_year=2027, end_year=2027)
    paths = runner.expected_input_paths(
        Path("/targets"), window, allow_missing_playoffs=True
    )
    assert sorted(path.name for path in paths) == [
        "FIRST_CHANCE27.parquet",
        "RAPM27.parquet",
        "SECOND_CHANCE_CLEAN27.parquet",
    ]


def test_playoff_input_paths_names_the_three_families():
    paths = runner.playoff_input_paths(Path("/targets"), 2027)
    assert [path.name for path in paths] == [
        "FIRST_CHANCE27_PS.parquet",
        "SECOND_CHANCE_CLEAN27_PS.parquet",
        "RAPM27_PS.parquet",
    ]


def test_run_window_rejects_a_partial_end_year_playoff_half(tmp_path):
    """One of three PS targets present is a broken build, not an in-season state."""
    processed_dir = tmp_path / "processed"
    _touch_targets(processed_dir, 2027, playoffs=False)
    (processed_dir / "RAPM27_PS.parquet").touch()

    with pytest.raises(FileNotFoundError) as excinfo:
        runner.run_window(
            runner.Window(start_year=2027, end_year=2027),
            processed_dir,
            tmp_path / "out",
            tmp_path / "master",
            "vpm",
            True,
            550,
            1,
            1,
            False,
            allow_missing_playoffs=True,
        )
    message = str(excinfo.value)
    assert "partial 2027 playoff half" in message
    assert "1 of 3" in message


def test_run_window_still_rejects_a_missing_regular_season(tmp_path):
    """The exemption covers playoffs only; a missing RS half must still fail."""
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True)

    with pytest.raises(FileNotFoundError) as excinfo:
        runner.run_window(
            runner.Window(start_year=2027, end_year=2027),
            processed_dir,
            tmp_path / "out",
            tmp_path / "master",
            "vpm",
            True,
            550,
            1,
            1,
            False,
            allow_missing_playoffs=True,
        )
    assert "missing 3 target inputs" in str(excinfo.value)


# --- publisher: derived end-year bound -----------------------------------


def _hardcoded_expected_windows():
    """The window set the publisher's original literal produced."""
    return {(start, end) for end in range(1997, 2027)
            for start in range(max(1997, end - 4), end + 1)}


def test_derived_bound_reproduces_the_hardcoded_2026_set():
    assert publisher.expected_windows(2026) == _hardcoded_expected_windows()
    assert len(publisher.expected_windows(2026)) == 140


def test_bumping_the_bound_adds_exactly_the_new_end_years_windows():
    added = publisher.expected_windows(2027) - publisher.expected_windows(2026)
    assert added == {(2023, 2027), (2024, 2027), (2025, 2027), (2026, 2027), (2027, 2027)}


def test_early_years_stay_truncated_at_1997():
    windows = publisher.expected_windows(2027)
    assert (1996, 1997) not in windows
    assert sorted(start for start, end in windows if end == 1999) == [1997, 1998, 1999]


def test_build_frame_refuses_an_empty_archive(tmp_path, monkeypatch):
    """No window CSVs must fail loudly rather than publish an empty frame."""
    monkeypatch.setattr(publisher, "SOURCE", tmp_path)
    with pytest.raises(AssertionError, match="No td550 window CSVs"):
        publisher.build_frame()
