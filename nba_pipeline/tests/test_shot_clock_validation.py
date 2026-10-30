import pandas as pd
import pytest

from nba_pipeline.scripts.build_shot_clock_estimates import (
    estimate_shot_clock,
    normalize_clock_seconds,
)
from nba_pipeline.scripts.validate_shot_clock_estimates import validate_estimates


def _row(**overrides):
    row = {
        "game_id": "0021800001",
        "event_num": 1,
        "event_type": "",
        "period": 1,
        "time_quarter": "12:00",
        "minute_remaining_quarter": 12,
        "seconds_remaining_quarter": 0,
        "home_description": "",
        "visitor_description": "",
        "neutral_description": "",
        "player1_id": 0,
        "player1_name": "",
    }
    row.update(overrides)
    return row


def test_pre_2019_offensive_rebound_uses_24_second_rule():
    rows = [
        _row(
            event_num=10,
            event_type=2,
            time_quarter="10:00",
            minute_remaining_quarter=10,
            home_description="MISS Home Guard 20' Jump Shot",
        ),
        _row(
            event_num=11,
            event_type=4,
            time_quarter="09:58",
            minute_remaining_quarter=9,
            seconds_remaining_quarter=58,
            home_description="Home Big REBOUND (Off:1 Def:0)",
        ),
        _row(
            event_num=12,
            event_type=2,
            time_quarter="09:48",
            minute_remaining_quarter=9,
            seconds_remaining_quarter=48,
            home_description="MISS Home Wing 18' Jump Shot",
        ),
    ]
    out = estimate_shot_clock(pd.DataFrame(rows), season_end_year=2018)
    shot = out[out["event_num"].eq(12)].iloc[0]
    assert shot["rule_reset_len"] == 24
    assert shot["shot_clock_est"] == 14


def test_end_of_period_change_caps_clock_at_game_time():
    rows = [
        _row(
            event_num=20,
            event_type=5,
            time_quarter="00:10",
            minute_remaining_quarter=0,
            seconds_remaining_quarter=10,
            visitor_description="Away Guard Bad Pass Turnover",
        ),
        _row(
            event_num=21,
            event_type=2,
            time_quarter="00:05",
            minute_remaining_quarter=0,
            seconds_remaining_quarter=5,
            home_description="MISS Home Guard 25' 3PT Jump Shot",
        ),
    ]
    out = estimate_shot_clock(pd.DataFrame(rows), season_end_year=2026)
    shot = out[out["event_num"].eq(21)].iloc[0]
    assert shot["rule_reset_len"] == 24
    assert shot["reset_len"] == 10
    assert shot["shot_clock_est"] == 5
    assert shot["shot_clock_lower"] == 4
    assert shot["shot_clock_upper"] == 6


def test_gabriel_total_seconds_clock_is_not_added_to_minutes_twice():
    rows = [
        _row(
            event_num=30,
            event_type=5,
            time_quarter="10:00",
            minute_remaining_quarter=10,
            seconds_remaining_quarter=600,
            visitor_description="Away Guard Bad Pass Turnover",
        ),
        _row(
            event_num=31,
            event_type=2,
            time_quarter="09:50",
            minute_remaining_quarter=9,
            seconds_remaining_quarter=590,
            home_description="MISS Home Guard 18' Jump Shot",
        ),
    ]
    frame = pd.DataFrame(rows)
    assert normalize_clock_seconds(frame).tolist() == [600, 590]
    out = estimate_shot_clock(frame, season_end_year=2013)
    shot = out[out["event_num"].eq(31)].iloc[0]
    assert shot["clock_sec"] == 590
    assert shot["shot_clock_est"] == 14


def test_historical_total_seconds_column_is_not_double_counted():
    rows = [
        _row(
            event_num=1,
            event_type=12,
            time_quarter="12:00",
            minute_remaining_quarter=12,
            seconds_remaining_quarter=720,
        ),
        _row(
            event_num=2,
            event_type=2,
            time_quarter="11:45",
            minute_remaining_quarter=11,
            seconds_remaining_quarter=705,
            home_description="MISS Home Guard 18' Jump Shot",
        ),
    ]
    out = estimate_shot_clock(pd.DataFrame(rows), season_end_year=2010)
    shot = out[out["event_num"].eq(2)].iloc[0]
    assert shot["clock_sec"] == 705
    assert shot["shot_clock_est"] == 9


def test_independent_labels_produce_error_and_interval_metrics():
    estimates = pd.DataFrame(
        {
            "game_id": ["g1", "g1"],
            "period": [1, 1],
            "event_num": [1, 2],
            "shot_clock_est": [8, 3],
            "shot_clock_raw": [8, 3],
            "shot_clock_lower": [7, 2],
            "shot_clock_upper": [9, 4],
            "reset_len": [24, 14],
            "confidence": ["high", "low"],
            "reset_reason": ["turnover", "off_rebound"],
            "confidence_reason": ["ok", "side_mismatch"],
        }
    )
    labels = pd.DataFrame(
        {
            "game_id": ["g1", "g1"],
            "period": [1, 1],
            "event_num": [1, 2],
            "shot_clock_actual": [7, 6],
        }
    )
    rows, grouped, summary = validate_estimates(estimates, labels)
    assert summary["has_independent_labels"] is True
    assert summary["labeled_rows"] == 2
    assert summary["mae"] == pytest.approx(2.0)
    assert summary["interval_coverage"] == pytest.approx(0.5)
    assert int(grouped["labeled_rows"].sum()) == 2
    assert rows.loc[rows["event_num"].eq(2), "abs_error"].iloc[0] == 3


def test_no_label_run_does_not_claim_absolute_error():
    estimates = pd.DataFrame(
        {
            "game_id": ["g1"],
            "period": [1],
            "event_num": [1],
            "shot_clock_est": [8],
            "shot_clock_raw": [8],
            "shot_clock_lower": [7],
            "shot_clock_upper": [9],
            "reset_len": [24],
            "confidence": ["high"],
            "reset_reason": ["turnover"],
            "confidence_reason": ["ok"],
        }
    )
    _, grouped, summary = validate_estimates(estimates)
    assert summary["has_independent_labels"] is False
    assert "mae" not in summary
    assert "mae" not in grouped.columns
