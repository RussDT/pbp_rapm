import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from export_long_spm_history import ALIGNMENT, OUTPUT_COLUMNS, SLOTS, actual_rows, export_season
from audit_long_spm_history import compare_scores


def frames():
    canonical = pd.DataFrame({"game_id": [21400001, 21400001], "event_num": [4, 9],
        "game_date": ["2013-10-29"] * 2, "Off_Diff": [.75, 2.75]})
    for i, slot in enumerate(SLOTS):
        canonical[slot] = i + 1
    actual = canonical.copy()
    actual["Off_Diff"] = [1., 3.]
    return actual, canonical


def test_actual_integer_points_replace_expected_ft_on_identical_rows():
    actual, canonical = frames()
    out = actual_rows(actual, canonical, 2014, "RS")
    assert list(out) == OUTPUT_COLUMNS
    np.testing.assert_array_equal(out.actual_points, [1., 3.])
    assert out.game_id.eq("0021400001").all()
    assert out.valid_lineup.all()
    np.testing.assert_array_equal(out.source_row, [0, 1])


@pytest.mark.parametrize("column", ALIGNMENT)
def test_alignment_rejects_reordered_or_changed_metadata(column):
    actual, canonical = frames()
    value = "2013-10-30" if column == "game_date" else 999
    actual.loc[0, column] = value
    with pytest.raises(ValueError, match="order mismatch"):
        actual_rows(actual, canonical, 2014, "RS")


def test_noninteger_expected_ft_target_fails():
    _, canonical = frames()
    with pytest.raises(ValueError, match="integer points"):
        actual_rows(canonical, canonical, 2014, "RS")


def test_invalid_lineup_is_flagged_without_dropping_canonical_rows():
    actual, canonical = frames()
    actual.loc[0, "O1"] = canonical.loc[0, "O1"] = 0
    actual.loc[1, "D1"] = canonical.loc[1, "D1"] = canonical.loc[1, "O1"]
    out = actual_rows(actual, canonical, 2014, "PS")
    assert len(out) == 2 and not out.valid_lineup.any()


def test_protected_or_unregistered_seasons_rejected():
    actual, canonical = frames()
    with pytest.raises(ValueError, match="registered"):
        actual_rows(actual, canonical, 2027, "RS")


def test_export_forces_actual_ft_and_restores_provider_hook(tmp_path):
    actual, canonical = frames()
    (tmp_path / "raw_data").mkdir(); (tmp_path / "processed").mkdir()
    (tmp_path / "raw_data/NBA14.parquet").write_bytes(b"raw")
    canonical.to_parquet(tmp_path / "processed/RAPM14.parquet", index=False)
    calls = []
    original = lambda *args: "must not be called"
    processor = SimpleNamespace(fetch_player_stats_supabase=original)
    def process(*args, **kwargs):
        assert processor.fetch_player_stats_supabase(14, False) is None
        calls.append(kwargs)
        return actual, actual
    processor.process_rapm_py = process
    receipt = export_season(tmp_path, tmp_path / "out", 2014, "RS", processor)
    assert calls == [dict(o_luck=0, d_luck=0, ft_value_mode="actual")]
    assert processor.fetch_player_stats_supabase is original
    assert receipt["rows"] == receipt["valid_lineup_rows"] == 2
    assert export_season(tmp_path, tmp_path / "out", 2014, "RS", processor) == receipt
    assert len(calls) == 1
    (tmp_path / "raw_data/NBA14.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="source/artifact changed"):
        export_season(tmp_path, tmp_path / "out", 2014, "RS", processor)


def test_provider_hook_restored_after_processor_failure(tmp_path):
    _, canonical = frames()
    (tmp_path / "raw_data").mkdir(); (tmp_path / "processed").mkdir()
    (tmp_path / "raw_data/NBA14.parquet").write_bytes(b"raw")
    canonical.to_parquet(tmp_path / "processed/RAPM14.parquet", index=False)
    original = lambda *args: None
    def fail(*args, **kwargs):
        raise RuntimeError("processor failed")
    processor = SimpleNamespace(fetch_player_stats_supabase=original, process_rapm_py=fail)
    with pytest.raises(RuntimeError, match="processor failed"):
        export_season(tmp_path, tmp_path / "out", 2014, "RS", processor)
    assert processor.fetch_player_stats_supabase is original


def test_independent_raw_score_pricing_dedupes_and_counts_makes_only():
    raw = pd.DataFrame(dict(game_id=[1]*6, event_num=list(range(6)),
        event_type=[1, 1, 2, 3, 3, 5], home_description=[
            "Shooter 2 PTS", "Shooter 3PT 5 PTS", "MISS Shooter 3PT",
            "Shooter Free Throw 1 of 2", "MISS Shooter Free Throw 2 of 2", "Turnover"],
        visitor_description=[None]*6))
    raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    exported = pd.DataFrame(dict(game_id=["0000000001"]*2, actual_points=[3., 3.]))
    result = compare_scores(raw, exported)
    assert len(result) == 1 and result.accepted.iloc[0]
    assert result.raw_event_points.iloc[0] == 6
    exported.loc[0, "actual_points"] = 2
    result = compare_scores(raw, exported)
    assert not result.accepted.iloc[0] and result.difference.iloc[0] == -1


def test_scoring_gate_quarantines_a_game_missing_from_either_population():
    raw = pd.DataFrame(dict(game_id=[1], event_type=[1],
        home_description=["Player 2 PTS"], visitor_description=[None]))
    exported = pd.DataFrame(dict(game_id=[2], actual_points=[2.]))
    result = compare_scores(raw, exported)
    assert len(result) == 2 and not result.accepted.any()
