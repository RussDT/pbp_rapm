"""Offline pytest coverage for season-year window derivation.

The two Supabase uploaders used to carry hardcoded 2026 filenames and are
invoked by `daily_rapm_update.sh` with no arguments, so a season rollover
silently pointed them at files that no longer existed. These tests pin the
token helper against the window list the 2026 nightly actually used, check
both uploaders derive their own source filenames from a season year, and
check the scposs script's source resolver and published-schema guard. No test
touches Supabase, the network, or the real master_results/results trees.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "nba_pipeline" / "scripts"

sys.path.insert(0, str(SCRIPTS_DIR))

from season_windows import (  # noqa: E402
    DEFAULT_SEASON_YEAR_ENV,
    FALLBACK_SEASON_YEAR,
    default_season_year,
    two_digit,
    window_token,
)


def _load_module(name: str, path: Path):
    """Import a script by path without running it as __main__."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- token helper -----------------------------------------------------------

def test_two_digit_pads_and_wraps():
    assert two_digit(2027) == "27"
    assert two_digit(2000) == "00"
    assert two_digit(1997) == "97"


def test_one_year_window_is_the_bare_end_token():
    assert window_token(2027, 1) == "27"


def test_multi_year_windows_span_inclusive():
    assert window_token(2027, 2) == "26_27"
    assert window_token(2027, 6) == "22_27"
    assert window_token(2027, 13) == "15_27"


def test_reproduces_the_hardcoded_2026_window_list():
    """The pre-change uploaders listed these tokens literally."""
    expected = ["26", "25_26", "24_26", "23_26", "22_26", "21_26", "14_26"]
    got = [window_token(2026, n) for n in (1, 2, 3, 4, 5, 6, 13)]
    assert got == expected


def test_window_crossing_the_century_boundary():
    assert window_token(2001, 6) == "96_01"


def test_default_season_year_prefers_the_environment(monkeypatch):
    monkeypatch.setenv(DEFAULT_SEASON_YEAR_ENV, "2031")
    assert default_season_year() == 2031


def test_default_season_year_falls_back_when_unset(monkeypatch):
    monkeypatch.delenv(DEFAULT_SEASON_YEAR_ENV, raising=False)
    assert default_season_year() == FALLBACK_SEASON_YEAR


def test_default_season_year_rejects_garbage(monkeypatch):
    monkeypatch.setenv(DEFAULT_SEASON_YEAR_ENV, "not-a-year")
    with pytest.raises(ValueError):
        default_season_year()


# --- six-factor uploader ----------------------------------------------------

def test_six_factor_sources_follow_the_season_year():
    mod = _load_module("upload_six_factor", SCRIPTS_DIR / "upload_six_factor.py")
    sources = mod.build_source_files(2027)

    intervals = [interval for _, interval, _ in sources]
    assert intervals == ["1Y", "2Y", "3Y", "4Y", "5Y", "6Y", "13Y"]

    by_interval = {interval: (name, rb) for name, interval, rb in sources}
    # 1Y-6Y are the season-effects family; only 13Y is rubberband.
    assert by_interval["1Y"] == ("weighted_factors_27_all_se.csv", False)
    assert by_interval["6Y"] == ("weighted_factors_22_27_all_se.csv", False)
    assert by_interval["13Y"] == ("weighted_factors_15_27_all_rb.csv", True)


def test_six_factor_sources_shift_whole_for_a_backfill_year():
    mod = _load_module("upload_six_factor", SCRIPTS_DIR / "upload_six_factor.py")
    names = [name for name, _, _ in mod.build_source_files(2026)]
    assert "weighted_factors_26_all_se.csv" in names
    assert not any("_27_" in n or n.endswith("_27_all_se.csv") for n in names)


def test_six_factor_source_overrides_parse():
    mod = _load_module("upload_six_factor", SCRIPTS_DIR / "upload_six_factor.py")
    overrides = mod.parse_source_overrides(["1Y=/tmp/a.csv", " 6Y = /tmp/b.csv "])
    assert overrides == {"1Y": Path("/tmp/a.csv"), "6Y": Path("/tmp/b.csv")}


def test_six_factor_rejects_a_malformed_override():
    mod = _load_module("upload_six_factor", SCRIPTS_DIR / "upload_six_factor.py")
    with pytest.raises(SystemExit):
        mod.parse_source_overrides(["1Y:/tmp/a.csv"])


# --- time-decay uploader ----------------------------------------------------

def test_timedecay_source_follows_the_season_year():
    mod = _load_module("upload_timedecay_rapm", SCRIPTS_DIR / "upload_timedecay_rapm.py")
    assert mod.default_source_file(2027).name == "weighted_factors_22_27_all_rb_se_td700.csv"
    assert mod.default_source_file(2026).name == "weighted_factors_21_26_all_rb_se_td700.csv"


# --- scposs source resolution and schema guard ------------------------------

def test_scposs_prefers_the_season_effects_layout(tmp_path, monkeypatch):
    mod = _load_module("update_2026_scposs", ROOT / "update_2026_scposs.py")
    monkeypatch.setattr(mod, "RESULTS_DIR", tmp_path)

    se_dir = tmp_path / "rapm_27_all_se"
    se_dir.mkdir()
    (se_dir / "weighted_factors_27_all_se.csv").write_text("player_id\n")

    assert mod.resolve_window_source("27") == (
        "rapm_27_all_se",
        "weighted_factors_27_all_se.csv",
    )


def test_scposs_falls_back_to_the_legacy_layout(tmp_path, monkeypatch):
    mod = _load_module("update_2026_scposs", ROOT / "update_2026_scposs.py")
    monkeypatch.setattr(mod, "RESULTS_DIR", tmp_path)

    legacy = tmp_path / "rapm_27_all"
    legacy.mkdir()
    (legacy / "weighted_factors_27_all.csv").write_text("player_id\n")

    assert mod.resolve_window_source("27") == (
        "rapm_27_all",
        "weighted_factors_27_all.csv",
    )


def test_scposs_names_every_candidate_when_nothing_exists(tmp_path, monkeypatch):
    mod = _load_module("update_2026_scposs", ROOT / "update_2026_scposs.py")
    monkeypatch.setattr(mod, "RESULTS_DIR", tmp_path)

    with pytest.raises(FileNotFoundError) as excinfo:
        mod.resolve_window_source("27")
    message = str(excinfo.value)
    assert "rapm_27_all_se" in message
    assert "rapm_27_all/" in message or "rapm_27_all" in message


def test_scposs_guard_flags_the_published_decomp_schema(tmp_path):
    """A published scposs file carries DECOMP columns this script cannot write."""
    mod = _load_module("update_2026_scposs", ROOT / "update_2026_scposs.py")

    published = tmp_path / "203999.csv"
    header = list(mod.TARGET_COLUMNS) + [
        "source_file",
        "decomp_model",
        "oRAPM_DECOMP_SHOT",
        "oALT_EFG_VALUE",
    ]
    pd.DataFrame(columns=header).to_csv(published, index=False)

    extra = mod.published_schema_conflict(published)
    assert "decomp_model" in extra
    assert "oRAPM_DECOMP_SHOT" in extra
    assert not set(extra) & set(mod.TARGET_COLUMNS)


def test_scposs_guard_is_quiet_on_a_matching_narrow_file(tmp_path):
    mod = _load_module("update_2026_scposs", ROOT / "update_2026_scposs.py")

    narrow = tmp_path / "203999.csv"
    pd.DataFrame(columns=list(mod.TARGET_COLUMNS)).to_csv(narrow, index=False)

    assert mod.published_schema_conflict(narrow) == []
